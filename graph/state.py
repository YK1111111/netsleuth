"""Shared state for NetSleuth's context-engineered diagnostic graph."""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict


class DiagnosticStep(TypedDict):
    """One evidence-gathering objective in a diagnostic plan."""

    id: str
    objective: str
    suggested_checks: list[str]


class VerificationCheck(TypedDict):
    """Verifier judgment for one diagnostic requirement."""

    name: str
    passed: bool
    detail: str


class CompressionEvent(TypedDict):
    """One graph-level context compaction event."""

    before_tokens: int
    after_tokens: int
    removed_messages: int
    summary: str
    next_node: str


class LayeredMemory(TypedDict):
    """Bounded rules, working state, and prior diagnostic history."""

    rules: dict[str, Any]
    working_memory: dict[str, Any]
    history_summary: dict[str, Any]


class NetSleuthState(TypedDict, total=False):
    """State shared by plan, diagnosis, verification, and submission nodes."""

    task: str
    messages: Annotated[list[BaseMessage], add_messages]
    plan_summary: str
    diagnostic_steps: list[DiagnosticStep]
    acceptance_criteria: list[str]
    diagnosis_report: list[str]
    verifier_summary: str
    verification_checks: list[VerificationCheck]
    recommended_next_instruction: str
    passed: bool
    attempts: int
    max_attempts: int
    is_max_steps_reached: bool
    last_error: str
    context_summary: str
    context_token_count: int
    context_token_limit: int
    context_should_compress: bool
    context_next_node: str
    compression_events: list[CompressionEvent]
    memory_snapshot: LayeredMemory
    resume_from_node: str
