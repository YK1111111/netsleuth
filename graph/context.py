"""Bounded, layered context for NetSleuth's single-agent diagnostic graph."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately

from agent.community.netsleuth.graph.state import NetSleuthState
from agent.community.netsleuth.prompts.stage4 import (
    NETSLEUTH_REACT_CONTEXT_COMPRESSION_PROMPT,
)

DEFAULT_CONTEXT_TOKEN_LIMIT = 24_000
DEFAULT_CONTEXT_KEEP_TOKENS = 6_000
CONTEXT_TOKEN_LIMIT_ENV = "NETSLEUTH_CONTEXT_TOKEN_LIMIT"
CONTEXT_KEEP_TOKENS_ENV = "NETSLEUTH_CONTEXT_KEEP_TOKENS"

RULES_LAYER = {
    "scope": "current NIKA session",
    "rules": [
        "Use only actual read-only diagnostic observations.",
        "Never read ground truth, answer files, or submission artifacts.",
        "Distinguish root causes from symptoms and affected peers.",
        "Preserve contradictory evidence and uncertainty.",
        "Do not repair, reconfigure, restart, or otherwise mutate the network.",
    ],
}

_TEXT_LIMITS = {
    "task": 8_000,
    "plan_summary": 1_200,
    "step_text": 800,
    "criterion": 600,
    "diagnosis_report": 6_000,
    "verifier_summary": 1_600,
    "recommended_instruction": 1_200,
    "last_error": 1_600,
    "context_summary": 4_000,
}


def get_context_token_limit() -> int:
    """Return the configurable absolute trigger for context compression."""

    return _positive_env_int(CONTEXT_TOKEN_LIMIT_ENV, DEFAULT_CONTEXT_TOKEN_LIMIT)


def get_context_keep_tokens() -> int:
    """Return how much recent ReAct context survives a compression."""

    limit = get_context_token_limit()
    configured = _positive_env_int(
        CONTEXT_KEEP_TOKENS_ENV,
        DEFAULT_CONTEXT_KEEP_TOKENS,
    )
    return min(configured, max(1, limit // 2))


def build_context_middleware(model: Any) -> SummarizationMiddleware:
    """Build reusable in-agent compression for diagnosis and verification."""

    limit = get_context_token_limit()
    return SummarizationMiddleware(
        model=model,
        trigger=("tokens", limit),
        keep=("tokens", get_context_keep_tokens()),
        summary_prompt=NETSLEUTH_REACT_CONTEXT_COMPRESSION_PROMPT,
        trim_tokens_to_summarize=limit,
    )


def context_aware_recursion_limit(max_steps: int) -> int:
    """Preserve the prior tool-loop budget after adding a before-model hook."""

    return max_steps + max(1, max_steps // 2)


def build_layered_memory(
    state: NetSleuthState,
    *,
    node: str = "graph",
) -> dict[str, Any]:
    """Build bounded rules, working state, and compressed-history layers."""

    context_summary_text = state.get("context_summary", "")
    latest_report = _latest_report(state)
    if context_summary_text and latest_report == context_summary_text:
        latest_report = ""
    working_memory = {
        "node": node,
        "task": short_text(state.get("task", ""), _TEXT_LIMITS["task"]),
        "attempt": state.get("attempts", 0) + 1,
        "max_attempts": state.get("max_attempts", 3),
        "plan_summary": short_text(
            state.get("plan_summary", ""),
            _TEXT_LIMITS["plan_summary"],
        ),
        "diagnostic_steps": _bounded_steps(state.get("diagnostic_steps", [])),
        "acceptance_criteria": [
            short_text(str(item), _TEXT_LIMITS["criterion"])
            for item in state.get("acceptance_criteria", [])[:8]
        ],
        "latest_diagnosis_report": short_text(
            latest_report,
            _TEXT_LIMITS["diagnosis_report"],
        ),
        "diagnosis_reached_tool_budget": state.get(
            "is_max_steps_reached",
            False,
        ),
    }
    history_summary = {
        "context_summary": short_text(
            context_summary_text,
            _TEXT_LIMITS["context_summary"],
        ),
        "verifier_summary": short_text(
            state.get("verifier_summary", ""),
            _TEXT_LIMITS["verifier_summary"],
        ),
        "recommended_next_instruction": short_text(
            state.get("recommended_next_instruction", ""),
            _TEXT_LIMITS["recommended_instruction"],
        ),
        "last_error": short_text(
            state.get("last_error", ""),
            _TEXT_LIMITS["last_error"],
        ),
        "compression_events": list(state.get("compression_events", []))[-3:],
    }
    return {
        "rules": dict(RULES_LAYER),
        "working_memory": working_memory,
        "history_summary": history_summary,
    }


def format_layered_memory_for_prompt(memory: dict[str, Any]) -> str:
    """Serialize a layered-memory snapshot for a node prompt."""

    return json.dumps(memory, ensure_ascii=False, indent=2, default=str)


def estimate_context_tokens(state: NetSleuthState, *, model: Any | None) -> int:
    """Estimate raw retry context before node-specific truncation is applied."""

    messages: list[BaseMessage] = list(state.get("messages", []))
    raw_payload = {
        "task": state.get("task", ""),
        "plan_summary": state.get("plan_summary", ""),
        "diagnostic_steps": state.get("diagnostic_steps", []),
        "acceptance_criteria": state.get("acceptance_criteria", []),
        "diagnosis_report": state.get("diagnosis_report", []),
        "verifier_summary": state.get("verifier_summary", ""),
        "verification_checks": state.get("verification_checks", []),
        "recommended_next_instruction": state.get(
            "recommended_next_instruction",
            "",
        ),
        "last_error": state.get("last_error", ""),
        "context_summary": state.get("context_summary", ""),
        "compression_events": state.get("compression_events", []),
    }
    measured = [
        *messages,
        HumanMessage(content=json.dumps(raw_payload, ensure_ascii=False, default=str)),
    ]
    if model is not None:
        try:
            return int(model.get_num_tokens_from_messages(measured))
        except Exception:  # noqa: BLE001 - deterministic fallback is intentional
            return max(1, int(count_tokens_approximately(measured)))
    return max(1, int(count_tokens_approximately(measured)))


def compression_payload(state: NetSleuthState) -> dict[str, Any]:
    """Return the bounded payload given to the graph-level compressor."""

    return {
        "memory": build_layered_memory(state, node="context_compressor"),
        "verification_checks": state.get("verification_checks", [])[:12],
        "context_next_node": state.get("context_next_node", ""),
    }


def short_text(value: Any, limit: int) -> str:
    """Bound text while retaining both its setup and latest conclusion."""

    text = str(value or "")
    if len(text) <= limit:
        return text
    marker = "\n...[truncated]...\n"
    if limit <= len(marker):
        return text[:limit]
    available = max(0, limit - len(marker))
    head_length = available * 2 // 3
    tail_length = available - head_length
    tail = text[-tail_length:].lstrip() if tail_length else ""
    return text[:head_length].rstrip() + marker + tail


def _bounded_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bounded: list[dict[str, Any]] = []
    for index, step in enumerate(steps[:8], start=1):
        bounded.append(
            {
                "id": short_text(step.get("id", f"step-{index}"), 80),
                "objective": short_text(
                    step.get("objective", ""),
                    _TEXT_LIMITS["step_text"],
                ),
                "suggested_checks": [
                    short_text(item, _TEXT_LIMITS["step_text"])
                    for item in step.get("suggested_checks", [])[:6]
                ],
            }
        )
    return bounded


def _latest_report(state: NetSleuthState) -> str:
    reports = state.get("diagnosis_report", [])
    return reports[-1] if reports else ""


def _positive_env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default
