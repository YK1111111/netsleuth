"""Nodes and routing for NetSleuth's harnessed diagnostic graph."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.errors import GraphRecursionError
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from pydantic import ValidationError

from agent.community.netsleuth.graph.context import (
    build_layered_memory,
    compression_payload,
    context_aware_recursion_limit,
    estimate_context_tokens,
    format_layered_memory_for_prompt,
    get_context_token_limit,
    short_text,
)
from agent.community.netsleuth.graph.state import (
    DiagnosticStep,
    NetSleuthState,
    VerificationCheck,
)
from agent.community.netsleuth.harness import NetSleuthHarness
from agent.community.netsleuth.phases import (
    NetSleuthDiagnosisPhase,
    NetSleuthSubmissionPhase,
    NetSleuthVerifierPhase,
)
from agent.community.netsleuth.prompts import (
    NETSLEUTH_PLANNER_PROMPT,
    NETSLEUTH_STATE_COMPRESSION_PROMPT,
)
from agent.llm.model_factory import load_model
from agent.protocols import DIAGNOSIS, SUBMISSION
from agent.utils.loggers import MESSAGES_FILENAME, AgentCallbackLogger, MessageLogger
from agent.utils.mcp_client import begin_submission_mcp_phase
from agent.utils.submission_context import submission_prompt_context

PLANNER = "planner"
VERIFIER = "verifier"
CONTEXT_MONITOR = "context_monitor"
CONTEXT_COMPRESSOR = "context_compressor"

_DEFAULT_STEPS: list[DiagnosticStep] = [
    {
        "id": "step-1",
        "objective": "Establish whether an anomaly exists and characterize its impact.",
        "suggested_checks": [
            "Inspect representative end-to-end connectivity and relevant counters."
        ],
    },
    {
        "id": "step-2",
        "objective": "Localize the earliest faulty network resource.",
        "suggested_checks": [
            "Inspect relevant interfaces, routes, protocol sessions, and configuration."
        ],
    },
    {
        "id": "step-3",
        "objective": "Distinguish the root cause from symptoms and alternatives.",
        "suggested_checks": [
            "Collect confirming evidence and check plausible contradictory explanations."
        ],
    },
]

_DEFAULT_CRITERIA = [
    "The report explicitly determines whether an anomaly exists.",
    "Observed impact is supported by actual diagnostic evidence.",
    "The faulty resource is localized rather than inferred only from a symptom.",
    "The proposed root cause is supported and plausible alternatives are addressed.",
]


class NetSleuthGraphNodes:
    """Session-bound node implementations for one NetSleuth run."""

    def __init__(
        self,
        *,
        session_id: str,
        session_dir: str,
        scenario_name: str,
        llm_provider: str,
        model: str,
        reasoning_effort: str | None,
        max_steps: int,
        callbacks: list[Any] | None = None,
        harness: NetSleuthHarness | None = None,
    ) -> None:
        self.session_id = session_id
        self.session_dir = session_dir
        self.scenario_name = scenario_name
        self.llm_provider = llm_provider
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_steps = max_steps
        self.callbacks = list(callbacks or [])
        self.harness = harness

        common = {
            "session_id": session_id,
            "llm_provider": llm_provider,
            "model": model,
            "scenario_name": scenario_name,
            "reasoning_effort": reasoning_effort,
        }
        self.planner_model = load_model(
            llm_provider=llm_provider,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        self.diagnosis_phase = NetSleuthDiagnosisPhase(**common)
        self.verifier_phase = NetSleuthVerifierPhase(**common)
        self._diagnosis_runner: Any | None = None
        self._verifier_runner: Any | None = None

    async def initialize(self) -> None:
        """Load diagnosis-time MCP tools before the graph starts."""

        if self._diagnosis_runner is not None and self._verifier_runner is not None:
            return
        await asyncio.gather(
            self.diagnosis_phase.load_tools(),
            self.verifier_phase.load_tools(),
        )
        self._diagnosis_runner = self.diagnosis_phase.get_agent()
        self._verifier_runner = self.verifier_phase.get_agent()

    async def run_planner(self, state: NetSleuthState) -> dict[str, Any]:
        """Create or revise an evidence-driven diagnostic plan."""

        memory = build_layered_memory(state, node=PLANNER)
        try:
            response = await self.planner_model.ainvoke(
                [
                    SystemMessage(content=NETSLEUTH_PLANNER_PROMPT),
                    HumanMessage(content=_planner_input(memory)),
                ],
                config={"callbacks": self._callbacks(PLANNER)},
            )
            raw = _message_text(response)
            parsed = _extract_json_object(raw)
            plan = _normalize_plan(parsed)
            if plan is None:
                raise ValueError("planner did not return the required JSON plan")
            return {
                **plan,
                "memory_snapshot": memory,
                "context_next_node": DIAGNOSIS,
            }
        except Exception as exc:  # noqa: BLE001 - a safe fallback keeps the run viable
            MessageLogger(phase=PLANNER, session_dir=self.session_dir).log(
                "error",
                {"message": f"Planner fallback: {type(exc).__name__}: {exc}"},
            )
            return {
                **_fallback_plan(state),
                "memory_snapshot": memory,
                "context_next_node": DIAGNOSIS,
            }

    async def run_diagnosis(self, state: NetSleuthState) -> dict[str, Any]:
        """Execute the current plan using diagnosis-only MCP tools."""

        if self._diagnosis_runner is None:
            raise RuntimeError("NetSleuthGraphNodes.initialize() must run first")
        memory = build_layered_memory(state, node=DIAGNOSIS)
        try:
            result = await self._diagnosis_runner.ainvoke(
                {"messages": [HumanMessage(content=_diagnosis_input(memory))]},
                config={
                    "callbacks": self._callbacks(DIAGNOSIS),
                    "recursion_limit": context_aware_recursion_limit(self.max_steps),
                },
                debug=True,
            )
            report = _message_text(result["messages"][-1])
            return {
                "diagnosis_report": [report],
                "is_max_steps_reached": False,
                "memory_snapshot": memory,
            }
        except ValidationError as exc:
            self._log_error(DIAGNOSIS, f"Validation error: {exc}")
            return {
                "diagnosis_report": [f"Diagnosis failed validation: {exc}"],
                "is_max_steps_reached": False,
                "last_error": str(exc),
                "memory_snapshot": memory,
            }
        except GraphRecursionError:
            self._log_error(DIAGNOSIS, "Diagnosis phase reached max recursion limit.")
            return {
                "diagnosis_report": [self._recent_diagnosis_report()],
                "is_max_steps_reached": True,
                "memory_snapshot": memory,
            }

    async def run_verifier(self, state: NetSleuthState) -> dict[str, Any]:
        """Independently verify the diagnosis against observable network state."""

        attempts = state.get("attempts", 0) + 1
        if self._verifier_runner is None:
            raise RuntimeError("NetSleuthGraphNodes.initialize() must run first")
        memory = build_layered_memory(state, node=VERIFIER)
        try:
            result = await self._verifier_runner.ainvoke(
                {"messages": [HumanMessage(content=_verifier_input(memory))]},
                config={
                    "callbacks": self._callbacks(VERIFIER),
                    "recursion_limit": context_aware_recursion_limit(self.max_steps),
                },
                debug=True,
            )
            raw = _message_text(result["messages"][-1])
            verdict = _normalize_verdict(_extract_json_object(raw), raw=raw)
        except (ValidationError, GraphRecursionError) as exc:
            reason = (
                "Verifier reached its tool budget."
                if isinstance(exc, GraphRecursionError)
                else f"Verifier failed validation: {exc}"
            )
            self._log_error(VERIFIER, reason)
            verdict = _failed_verdict(reason, "Collect the missing evidence and retry.")

        passed = verdict["passed"]
        reason = verdict["reason"]
        recommended = verdict["recommended_next_instruction"]
        update = {
            "passed": passed,
            "attempts": attempts,
            "verifier_summary": reason,
            "verification_checks": verdict["checks"],
            "recommended_next_instruction": recommended,
            "last_error": "" if passed else _verifier_error(reason, recommended),
            "memory_snapshot": memory,
        }
        update["context_next_node"] = verifier_route({**state, **update})
        return update

    async def run_context_monitor(self, state: NetSleuthState) -> dict[str, Any]:
        """Measure retry context and request compaction before it becomes unsafe."""

        token_limit = get_context_token_limit()
        token_count = estimate_context_tokens(state, model=self.planner_model)
        next_node = state.get("context_next_node") or VERIFIER
        should_compress = next_node != SUBMISSION and token_count >= token_limit
        MessageLogger(phase=CONTEXT_MONITOR, session_dir=self.session_dir).log(
            "context_monitor",
            {
                "token_count": token_count,
                "token_limit": token_limit,
                "should_compress": should_compress,
                "next_node": next_node,
                "message_count": len(state.get("messages", [])),
            },
        )
        return {
            "context_token_count": token_count,
            "context_token_limit": token_limit,
            "context_should_compress": should_compress,
            "context_next_node": next_node,
            "memory_snapshot": build_layered_memory(
                state,
                node=CONTEXT_MONITOR,
            ),
        }

    async def run_context_compressor(
        self,
        state: NetSleuthState,
    ) -> dict[str, Any]:
        """Replace verbose retry history with an evidence-faithful summary."""

        before_tokens = state.get("context_token_count") or estimate_context_tokens(
            state,
            model=self.planner_model,
        )
        before_messages = list(state.get("messages", []))
        try:
            response = await self.planner_model.ainvoke(
                [
                    SystemMessage(content=NETSLEUTH_STATE_COMPRESSION_PROMPT),
                    HumanMessage(
                        content=json.dumps(
                            compression_payload(state),
                            ensure_ascii=False,
                            default=str,
                        )
                    ),
                ],
                config={"callbacks": self._callbacks(DIAGNOSIS)},
            )
            compressed = _normalize_compression(
                _extract_json_object(_message_text(response))
            )
            if compressed is None:
                raise ValueError("context compressor did not return valid JSON")
        except Exception as exc:  # noqa: BLE001 - deterministic fallback preserves progress
            compressed = _fallback_compression(state, error=exc)

        summary = _format_compressed_context(compressed, state)
        summary_message = AIMessage(content=summary)
        post_state: NetSleuthState = {
            **state,
            "messages": [summary_message],
            "diagnosis_report": [summary],
            "context_summary": summary,
            "verifier_summary": short_text(
                state.get("verifier_summary", ""),
                1_600,
            ),
            "last_error": short_text(state.get("last_error", ""), 1_600),
        }
        after_tokens = estimate_context_tokens(
            post_state,
            model=self.planner_model,
        )
        event = {
            "before_tokens": int(before_tokens),
            "after_tokens": int(after_tokens),
            "removed_messages": len(before_messages),
            "summary": short_text(summary, 1_600),
            "next_node": state.get("context_next_node", PLANNER),
        }
        events = [*state.get("compression_events", []), event]
        MessageLogger(phase=CONTEXT_COMPRESSOR, session_dir=self.session_dir).log(
            "context_compression",
            event,
        )
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                summary_message,
            ],
            "diagnosis_report": [summary],
            "context_summary": summary,
            "context_token_count": after_tokens,
            "context_should_compress": False,
            "compression_events": events,
            "memory_snapshot": build_layered_memory(
                post_state,
                node=CONTEXT_COMPRESSOR,
            ),
            "verifier_summary": post_state["verifier_summary"],
            "last_error": post_state["last_error"],
        }

    async def run_submission(self, state: NetSleuthState) -> dict[str, Any]:
        """Freeze the latest report, advance the MCP gate, and submit once."""

        diagnosis = _latest_report(state)
        begin_submission_mcp_phase(self.session_id, diagnosis)

        phase = NetSleuthSubmissionPhase(
            session_id=self.session_id,
            llm_provider=self.llm_provider,
            model=self.model,
            scenario_name=self.scenario_name,
            reasoning_effort=self.reasoning_effort,
        )
        await phase.load_tools()
        runner = phase.get_agent()
        frozen_context = submission_prompt_context(self.session_id)
        verification_note = (
            "The verifier accepted this diagnosis."
            if state.get("passed")
            else "The verification attempt limit was reached; submit only what the frozen report supports."
        )

        try:
            result = await runner.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content=(
                                "Use this frozen diagnosis report to make the one final "
                                f"canonical submission:\n{diagnosis}\n\n"
                                f"Verification status: {verification_note}\n\n"
                                f"{frozen_context}"
                            )
                        )
                    ]
                },
                config={
                    "callbacks": self._callbacks(SUBMISSION),
                    "recursion_limit": self.max_steps,
                },
                debug=True,
            )
            return {"messages": result["messages"]}
        except GraphRecursionError:
            submitted = (Path(self.session_dir) / "submission.json").exists()
            suffix = (
                "after a successful submission." if submitted else "without submitting."
            )
            self._log_error(
                SUBMISSION,
                f"Submission phase reached max recursion limit {suffix}",
            )
            return {
                "messages": [
                    AIMessage(
                        content="Error: submission phase did not finish within max steps."
                    )
                ]
            }

    def _callbacks(self, phase: str) -> list[Any]:
        callbacks = [
            AgentCallbackLogger(phase=phase, session_dir=self.session_dir),
            *self.callbacks,
        ]
        if self.harness is not None:
            callbacks.append(self.harness.callback(phase))
        return callbacks

    def _log_error(self, phase: str, message: str) -> None:
        MessageLogger(phase=phase, session_dir=self.session_dir).log(
            "error", {"message": message}
        )

    def _recent_diagnosis_report(self) -> str:
        path = Path(self.session_dir) / MESSAGES_FILENAME
        conclusions: list[str] = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("phase") == DIAGNOSIS and event.get("event") == "llm_end":
                    text = event.get("text")
                    if isinstance(text, str) and text.strip():
                        conclusions.append(text.strip())
        evidence = "\n\n".join(conclusions[-5:])
        return (
            "Diagnosis reached its tool budget; the following observed conclusions "
            f"are the available evidence:\n\n{evidence}"
        )


def verifier_route(state: NetSleuthState) -> str:
    """Route a verified report to submission or an evidence gap to replanning."""

    if state.get("passed"):
        return SUBMISSION
    if state.get("attempts", 0) >= state.get("max_attempts", 3):
        return SUBMISSION
    return PLANNER


def context_monitor_route(state: NetSleuthState) -> str:
    """Compress when needed, otherwise resume the upstream-selected node."""

    next_node = state.get("context_next_node") or VERIFIER
    if next_node == SUBMISSION:
        return SUBMISSION
    if state.get("context_should_compress"):
        return CONTEXT_COMPRESSOR
    return next_node


def context_compressor_route(state: NetSleuthState) -> str:
    """Resume the workflow at the node selected before compaction."""

    return state.get("context_next_node") or VERIFIER


def _planner_input(memory: dict[str, Any]) -> str:
    return (
        "Create or revise the diagnostic plan using this bounded layered context.\n"
        + format_layered_memory_for_prompt(memory)
    )


def _diagnosis_input(memory: dict[str, Any]) -> str:
    return (
        "Execute this diagnostic plan. All actions must remain read-only.\n"
        + format_layered_memory_for_prompt(memory)
    )


def _verifier_input(memory: dict[str, Any]) -> str:
    return (
        "Verify this diagnosis against the acceptance criteria and observable "
        "network state. Return only the required JSON verdict.\n"
        + format_layered_memory_for_prompt(memory)
    )


def _latest_report(state: NetSleuthState) -> str:
    reports = state.get("diagnosis_report", [])
    return reports[-1] if reports else ""


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return str(content).strip()
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
            parts.append(str(block.get("text", "")))
    return "\n".join(part for part in parts if part).strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    candidates = [stripped]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1))
    start = stripped.find("{")
    if start >= 0:
        candidates.append(stripped[start:])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                parsed, _ = json.JSONDecoder().raw_decode(candidate)
            except json.JSONDecodeError:
                continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_plan(parsed: dict[str, Any] | None) -> dict[str, Any] | None:
    if not parsed:
        return None
    summary = str(parsed.get("plan_summary") or "").strip()
    raw_steps = parsed.get("diagnostic_steps")
    raw_criteria = parsed.get("acceptance_criteria")
    if (
        not summary
        or not isinstance(raw_steps, list)
        or not isinstance(raw_criteria, list)
    ):
        return None

    steps: list[DiagnosticStep] = []
    for index, item in enumerate(raw_steps, start=1):
        if isinstance(item, str):
            objective = item.strip()
            checks: list[str] = []
            step_id = f"step-{index}"
        elif isinstance(item, dict):
            objective = str(item.get("objective") or "").strip()
            step_id = str(item.get("id") or f"step-{index}").strip()
            checks = _string_list(item.get("suggested_checks"))
        else:
            continue
        if objective:
            steps.append(
                {
                    "id": step_id or f"step-{index}",
                    "objective": objective,
                    "suggested_checks": checks,
                }
            )
    criteria = _string_list(raw_criteria)
    if not steps or not criteria:
        return None
    return {
        "plan_summary": summary,
        "diagnostic_steps": steps,
        "acceptance_criteria": criteria,
    }


def _fallback_plan(state: NetSleuthState) -> dict[str, Any]:
    feedback = state.get("recommended_next_instruction", "").strip()
    steps = [dict(step) for step in _DEFAULT_STEPS]
    if feedback:
        steps.insert(
            0,
            {
                "id": "step-0",
                "objective": "Resolve the verifier's highest-priority evidence gap.",
                "suggested_checks": [feedback],
            },
        )
    return {
        "plan_summary": "Collect read-only evidence, localize the fault, and test alternatives.",
        "diagnostic_steps": steps,
        "acceptance_criteria": list(_DEFAULT_CRITERIA),
    }


def _normalize_verdict(parsed: dict[str, Any] | None, *, raw: str) -> dict[str, Any]:
    if not parsed:
        return _failed_verdict(
            "Verifier did not return valid JSON.",
            "Re-check the evidence and return a structured diagnosis.",
            detail=raw[:800],
        )
    reason = str(parsed.get("reason") or "").strip()
    recommended = str(parsed.get("recommended_next_instruction") or "").strip()
    checks = _normalize_checks(parsed.get("checks"))
    passed = (
        parsed.get("passed") is True
        and bool(checks)
        and all(check["passed"] for check in checks)
    )
    if not reason or not checks:
        return _failed_verdict(
            "Verifier verdict omitted its reason or checks.",
            recommended or "Collect explicit evidence for every acceptance criterion.",
            detail=raw[:800],
        )
    if not passed and not recommended:
        recommended = "Collect evidence for the failed verification checks."
    return {
        "passed": passed,
        "reason": reason,
        "checks": checks,
        "recommended_next_instruction": "" if passed else recommended,
    }


def _normalize_checks(value: Any) -> list[VerificationCheck]:
    if not isinstance(value, list):
        return []
    checks: list[VerificationCheck] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        checks.append(
            {
                "name": str(item.get("name") or f"check-{index}"),
                "passed": item.get("passed") is True,
                "detail": str(item.get("detail") or ""),
            }
        )
    return checks


def _failed_verdict(
    reason: str, recommended: str, *, detail: str = ""
) -> dict[str, Any]:
    return {
        "passed": False,
        "reason": reason,
        "checks": [
            {
                "name": "verifier_output",
                "passed": False,
                "detail": detail or reason,
            }
        ],
        "recommended_next_instruction": recommended,
    }


def _verifier_error(reason: str, recommended: str) -> str:
    return f"{reason}\nRecommended next instruction: {recommended}".strip()


def _normalize_compression(parsed: dict[str, Any] | None) -> dict[str, Any] | None:
    if not parsed:
        return None
    summary = str(parsed.get("summary") or "").strip()
    if not summary:
        return None
    return {
        "summary": summary,
        "confirmed_observations": _string_list(parsed.get("confirmed_observations")),
        "hypotheses": _string_list(parsed.get("hypotheses")),
        "evidence_gaps": _string_list(parsed.get("evidence_gaps")),
        "next_instruction": str(parsed.get("next_instruction") or "").strip(),
        "uncertainty": str(parsed.get("uncertainty") or "").strip(),
    }


def _fallback_compression(
    state: NetSleuthState,
    *,
    error: Exception,
) -> dict[str, Any]:
    verifier_gap = short_text(state.get("verifier_summary", ""), 1_200)
    return {
        "summary": short_text(
            _latest_report(state)
            or state.get("context_summary", "")
            or "No prior diagnosis report is available.",
            6_000,
        ),
        "confirmed_observations": [],
        "hypotheses": [],
        "evidence_gaps": (
            [verifier_gap]
            if verifier_gap
            else ["Re-check the verifier's unresolved acceptance criteria."]
        ),
        "next_instruction": short_text(
            state.get("recommended_next_instruction", ""),
            1_200,
        ),
        "uncertainty": f"Context compressor fallback: {type(error).__name__}",
    }


def _format_compressed_context(
    compressed: dict[str, Any],
    state: NetSleuthState,
) -> str:
    payload = {
        "type": "netsleuth_context_summary",
        "plan_summary": short_text(state.get("plan_summary", ""), 1_200),
        "acceptance_criteria": state.get("acceptance_criteria", [])[:8],
        "attempts": state.get("attempts", 0),
        "compression": compressed,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
