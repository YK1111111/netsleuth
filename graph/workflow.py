"""LangGraph construction for NetSleuth's harnessed workflow."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent.community.netsleuth.graph.nodes import (
    CONTEXT_COMPRESSOR,
    CONTEXT_MONITOR,
    PLANNER,
    VERIFIER,
    NetSleuthGraphNodes,
    context_compressor_route,
    context_monitor_route,
)
from agent.community.netsleuth.graph.state import NetSleuthState
from agent.community.netsleuth.harness import END_NODE, NetSleuthHarness
from agent.protocols import DIAGNOSIS, SUBMISSION


def build_workflow(
    nodes: NetSleuthGraphNodes,
    harness: NetSleuthHarness | None = None,
):
    """Build the single-agent diagnostic loop with context and harness hooks."""

    graph = StateGraph(NetSleuthState)
    graph.add_node(
        PLANNER,
        _instrument(PLANNER, nodes.run_planner, harness, lambda state: DIAGNOSIS),
    )
    graph.add_node(
        DIAGNOSIS,
        _instrument(DIAGNOSIS, nodes.run_diagnosis, harness, lambda state: VERIFIER),
    )
    graph.add_node(
        VERIFIER,
        _instrument(
            VERIFIER,
            nodes.run_verifier,
            harness,
            lambda state: CONTEXT_MONITOR,
        ),
    )
    graph.add_node(
        CONTEXT_MONITOR,
        _instrument(
            CONTEXT_MONITOR,
            nodes.run_context_monitor,
            harness,
            context_monitor_route,
        ),
    )
    graph.add_node(
        CONTEXT_COMPRESSOR,
        _instrument(
            CONTEXT_COMPRESSOR,
            nodes.run_context_compressor,
            harness,
            context_compressor_route,
        ),
    )
    graph.add_node(
        SUBMISSION,
        _instrument(
            SUBMISSION,
            nodes.run_submission,
            harness,
            lambda state: END_NODE,
        ),
    )

    graph.add_conditional_edges(
        START,
        _async_entry_route,
        {
            PLANNER: PLANNER,
            DIAGNOSIS: DIAGNOSIS,
            VERIFIER: VERIFIER,
            CONTEXT_MONITOR: CONTEXT_MONITOR,
            CONTEXT_COMPRESSOR: CONTEXT_COMPRESSOR,
            SUBMISSION: SUBMISSION,
        },
    )
    graph.add_edge(PLANNER, DIAGNOSIS)
    graph.add_edge(DIAGNOSIS, VERIFIER)
    graph.add_edge(VERIFIER, CONTEXT_MONITOR)
    graph.add_conditional_edges(
        CONTEXT_MONITOR,
        _async_context_monitor_route,
        {
            CONTEXT_COMPRESSOR: CONTEXT_COMPRESSOR,
            PLANNER: PLANNER,
            SUBMISSION: SUBMISSION,
        },
    )
    graph.add_conditional_edges(
        CONTEXT_COMPRESSOR,
        _async_context_compressor_route,
        {
            PLANNER: PLANNER,
            SUBMISSION: SUBMISSION,
        },
    )
    graph.add_edge(SUBMISSION, END)
    return graph.compile()


async def _async_context_monitor_route(state: NetSleuthState) -> str:
    """Keep cyclic async graphs off LangGraph's synchronous route executor."""

    return context_monitor_route(state)


async def _async_context_compressor_route(state: NetSleuthState) -> str:
    """Resume the selected node after async context compaction."""

    return context_compressor_route(state)


async def _async_entry_route(state: NetSleuthState) -> str:
    """Start normally or continue from an explicitly loaded checkpoint."""

    return state.get("resume_from_node") or PLANNER


def _instrument(
    node: str,
    function: Callable[[NetSleuthState], Awaitable[dict[str, Any]]],
    harness: NetSleuthHarness | None,
    next_node: Callable[[NetSleuthState], str],
):
    """Add node-boundary trace and checkpoint hooks without changing node logic."""

    if harness is None:
        return function

    async def instrumented(state: NetSleuthState) -> dict[str, Any]:
        harness.node_started(node, state)
        try:
            update = await function(state)
        except BaseException as exc:
            harness.node_failed(node, state, exc)
            raise
        merged = _merge_state(state, update)
        harness.node_completed(node, merged, next_node=next_node(merged))
        return update

    return instrumented


def _merge_state(
    state: NetSleuthState,
    update: dict[str, Any],
) -> NetSleuthState:
    """Apply the one reducer needed to serialize a faithful node checkpoint."""

    merged: NetSleuthState = {**state, **update}
    if "messages" in update:
        merged["messages"] = add_messages(
            list(state.get("messages", [])),
            update["messages"],
        )
    merged.pop("resume_from_node", None)
    return merged
