"""Stage 5 runtime safety, checkpointing, and tracing for NetSleuth."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from langchain_core.outputs.generation import Generation

from agent.utils.loggers import MessageLogger
from agent.utils.usage import normalize_usage

HARNESS_PHASE = "netsleuth_harness"
CHECKPOINT_FILENAME = "netsleuth_checkpoint.json"
TRACE_FILENAME = "netsleuth_trace.json"
CHECKPOINT_MODE_ENV = "NETSLEUTH_CHECKPOINT_MODE"
TRACE_MODE_ENV = "NETSLEUTH_TRACE_MODE"
RESUME_ENV = "NETSLEUTH_RESUME_CHECKPOINT"
CHECKPOINT_VERSION = 1
END_NODE = "__end__"

_FALSE_VALUES = {"0", "false", "no", "off"}
_STATE_FIELDS = {
    "task",
    "messages",
    "plan_summary",
    "diagnostic_steps",
    "acceptance_criteria",
    "diagnosis_report",
    "verifier_summary",
    "verification_checks",
    "recommended_next_instruction",
    "passed",
    "attempts",
    "max_attempts",
    "is_max_steps_reached",
    "last_error",
    "context_summary",
    "context_token_count",
    "context_token_limit",
    "context_should_compress",
    "context_next_node",
    "compression_events",
}


class NetSleuthHarness:
    """Session-local harness that does not change benchmark scoring semantics."""

    def __init__(self, *, session_id: str, session_dir: str) -> None:
        self.session_id = session_id
        self.session_dir = Path(session_dir)
        self.checkpoint_path = self.session_dir / CHECKPOINT_FILENAME
        self.trace_path = self.session_dir / TRACE_FILENAME
        self.checkpoint_enabled = _env_enabled(CHECKPOINT_MODE_ENV, default=True)
        self.trace_enabled = _env_enabled(TRACE_MODE_ENV, default=True)
        self.resume_enabled = _env_enabled(RESUME_ENV, default=False)
        self._logger = MessageLogger(phase=HARNESS_PHASE, session_dir=session_dir)
        self._started_at = time.perf_counter()
        self._started_at_iso = _utc_now()
        self._status = "created"
        self._resumed = False
        self._latest_node = ""
        self._node_visits: dict[str, int] = {}
        self._llm_calls: dict[str, int] = {}
        self._tool_calls: dict[str, int] = {}
        self._tool_errors: dict[str, int] = {}
        self._input_tokens = 0
        self._output_tokens = 0
        self._lock = threading.Lock()

    def callback(self, phase: str) -> BaseCallbackHandler:
        """Return a metrics-only callback; canonical events remain in messages.jsonl."""

        return _HarnessCallback(self, phase)

    def load_resume_state(
        self,
        *,
        task: str,
        max_attempts: int,
    ) -> dict[str, Any] | None:
        """Load an explicitly requested, compatible node-boundary checkpoint."""

        if not self.resume_enabled or not self.checkpoint_path.is_file():
            return None
        try:
            payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._log("checkpoint_resume_rejected", reason=f"unreadable: {exc}")
            return None
        if payload.get("version") != CHECKPOINT_VERSION:
            self._log("checkpoint_resume_rejected", reason="version mismatch")
            return None
        if payload.get("session_id") != self.session_id:
            self._log("checkpoint_resume_rejected", reason="session mismatch")
            return None
        if payload.get("task_fingerprint") != _task_fingerprint(task):
            self._log("checkpoint_resume_rejected", reason="task mismatch")
            return None
        next_node = str(payload.get("next_node") or "")
        if payload.get("status") == "completed" or next_node == END_NODE:
            self._log("checkpoint_resume_rejected", reason="checkpoint completed")
            return None
        raw_state = payload.get("state")
        if not isinstance(raw_state, dict) or not next_node:
            self._log("checkpoint_resume_rejected", reason="checkpoint incomplete")
            return None
        try:
            state = _deserialize_state(raw_state)
        except (TypeError, ValueError) as exc:
            self._log("checkpoint_resume_rejected", reason=f"invalid state: {exc}")
            return None
        state["task"] = task
        state["max_attempts"] = max_attempts
        state["resume_from_node"] = next_node
        self._resumed = True
        self._log(
            "checkpoint_resumed",
            checkpoint=str(self.checkpoint_path),
            latest_node=payload.get("latest_node", ""),
            next_node=next_node,
        )
        return state

    def start(self, state: dict[str, Any]) -> None:
        self._status = "running"
        self._started_at = time.perf_counter()
        self._started_at_iso = _utc_now()
        self._log(
            "harness_start",
            resumed=self._resumed,
            checkpoint_enabled=self.checkpoint_enabled,
            trace_enabled=self.trace_enabled,
            safety_policy="NIKA two-phase gateway + NetSleuth no-remediation guard",
        )
        if not self._resumed:
            self.save_checkpoint(
                state,
                status="running",
                latest_node="start",
                next_node="planner",
            )

    def node_started(self, node: str, state: dict[str, Any]) -> None:
        del state
        self._log("node_start", node=node)

    def node_completed(
        self,
        node: str,
        state: dict[str, Any],
        *,
        next_node: str,
    ) -> None:
        self._latest_node = node
        self._node_visits[node] = self._node_visits.get(node, 0) + 1
        status = "completed" if next_node == END_NODE else "running"
        self._log(
            "node_end",
            node=node,
            next_node=next_node,
            attempts=state.get("attempts", 0),
            passed=state.get("passed"),
        )
        self.save_checkpoint(
            state,
            status=status,
            latest_node=node,
            next_node=next_node,
        )

    def node_failed(
        self,
        node: str,
        state: dict[str, Any],
        error: BaseException,
    ) -> None:
        interrupted = isinstance(error, (KeyboardInterrupt, asyncio.CancelledError))
        status = "interrupted" if interrupted else "failed"
        self._latest_node = node
        self._log(
            "node_error",
            node=node,
            status=status,
            error_type=type(error).__name__,
            error=str(error),
        )
        self.save_checkpoint(
            state,
            status=status,
            latest_node=node,
            next_node=node,
        )

    def finish(self, *, status: str, final_state: dict[str, Any] | None) -> None:
        self._status = status
        self._log(
            "harness_end",
            status=status,
            latest_node=self._latest_node,
            attempts=(final_state or {}).get("attempts", 0),
            passed=(final_state or {}).get("passed"),
        )
        self._write_trace_summary(final_state or {})

    def save_checkpoint(
        self,
        state: dict[str, Any],
        *,
        status: str,
        latest_node: str,
        next_node: str,
    ) -> None:
        if not self.checkpoint_enabled:
            return
        payload = {
            "version": CHECKPOINT_VERSION,
            "updated_at": _utc_now(),
            "session_id": self.session_id,
            "status": status,
            "latest_node": latest_node,
            "next_node": next_node,
            "task_fingerprint": _task_fingerprint(str(state.get("task") or "")),
            "state": _serialize_state(state),
        }
        _atomic_write_json(self.checkpoint_path, payload)
        self._log(
            "checkpoint_saved",
            checkpoint=str(self.checkpoint_path),
            status=status,
            latest_node=latest_node,
            next_node=next_node,
        )

    def record_llm_end(self, phase: str, response: Any) -> None:
        usage: dict[str, int] = {}
        try:
            generation: Generation = response.generations[0][0]
            message = getattr(generation, "message", None)
            usage = normalize_usage(getattr(message, "usage_metadata", None))
        except (AttributeError, IndexError, TypeError):
            usage = normalize_usage(None)
        with self._lock:
            self._llm_calls[phase] = self._llm_calls.get(phase, 0) + 1
            self._input_tokens += usage["input_tokens"]
            self._output_tokens += usage["output_tokens"]

    def record_tool_start(self, phase: str) -> None:
        with self._lock:
            self._tool_calls[phase] = self._tool_calls.get(phase, 0) + 1

    def record_tool_error(self, phase: str) -> None:
        with self._lock:
            self._tool_errors[phase] = self._tool_errors.get(phase, 0) + 1

    def _write_trace_summary(self, final_state: dict[str, Any]) -> None:
        if not self.trace_enabled:
            return
        payload = {
            "version": 1,
            "session_id": self.session_id,
            "status": self._status,
            "started_at": self._started_at_iso,
            "ended_at": _utc_now(),
            "duration_ms": round((time.perf_counter() - self._started_at) * 1000),
            "resumed": self._resumed,
            "latest_node": self._latest_node,
            "node_visits": dict(sorted(self._node_visits.items())),
            "llm_calls": dict(sorted(self._llm_calls.items())),
            "tool_calls": dict(sorted(self._tool_calls.items())),
            "tool_errors": dict(sorted(self._tool_errors.items())),
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "attempts": final_state.get("attempts", 0),
            "passed": final_state.get("passed"),
            "checkpoint": str(self.checkpoint_path) if self.checkpoint_enabled else "",
            "canonical_events": str(self.session_dir / "messages.jsonl"),
        }
        _atomic_write_json(self.trace_path, payload)

    def _log(self, event: str, **payload: Any) -> None:
        if self.trace_enabled:
            self._logger.log(event, payload)


class _HarnessCallback(BaseCallbackHandler):
    """Collect aggregate metrics without duplicating canonical tool payloads."""

    def __init__(self, harness: NetSleuthHarness, phase: str) -> None:
        super().__init__()
        self._harness = harness
        self._phase = phase

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        del kwargs
        self._harness.record_llm_end(self._phase, response)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        del serialized, input_str, kwargs
        self._harness.record_tool_start(self._phase)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        del kwargs
        if getattr(output, "status", None) == "error":
            self._harness.record_tool_error(self._phase)

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        del error, kwargs
        self._harness.record_tool_error(self._phase)


def _serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key in _STATE_FIELDS:
        if key not in state:
            continue
        value = state[key]
        if key == "messages":
            serialized[key] = [
                message_to_dict(message)
                for message in value
                if isinstance(message, BaseMessage)
            ]
        else:
            serialized[key] = _json_safe(value)
    return serialized


def _deserialize_state(state: dict[str, Any]) -> dict[str, Any]:
    deserialized = dict(state)
    raw_messages = deserialized.get("messages", [])
    if not isinstance(raw_messages, list):
        raise TypeError("messages is not a list")
    deserialized["messages"] = messages_from_dict(raw_messages)
    return deserialized


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _task_fingerprint(task: str) -> str:
    return hashlib.sha256(task.encode("utf-8")).hexdigest()


def _env_enabled(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSE_VALUES


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
