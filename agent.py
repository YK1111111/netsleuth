"""NIKA entry point for NetSleuth's harnessed LangGraph workflow."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from langchain_core.messages import HumanMessage

from agent.community.netsleuth.graph.nodes import NetSleuthGraphNodes
from agent.community.netsleuth.graph.workflow import build_workflow
from agent.community.netsleuth.harness import NetSleuthHarness
from nika.utils.logger import system_logger
from nika.utils.session import Session


class NetSleuthAgent:
    """NIKA-compatible single-agent diagnosis with context and runtime guards."""

    def __init__(
        self,
        session_id: str,
        llm_provider: str = "openai",
        model: str = "gpt-5-mini",
        max_steps: int = 20,
        reasoning_effort: str | None = None,
        stream_output: bool = True,
        max_attempts: int = 3,
    ) -> None:
        self.session_id = session_id
        self.llm_provider = llm_provider
        self.model = model
        self.max_steps = max(1, max_steps)
        self.reasoning_effort = reasoning_effort
        self.stream_output = stream_output
        self.max_attempts = max(1, max_attempts)

        self.session = Session()
        self.session.load_running_session(session_id=session_id)
        self.session_dir = self.session.session_dir
        self.langfuse_handler = self._load_langfuse_handler()
        self.harness = NetSleuthHarness(
            session_id=session_id,
            session_dir=self.session_dir,
        )

        callbacks = [self.langfuse_handler] if self.langfuse_handler is not None else []
        self.nodes = NetSleuthGraphNodes(
            session_id=session_id,
            session_dir=self.session_dir,
            scenario_name=self.session.scenario_name,
            llm_provider=llm_provider,
            model=model,
            reasoning_effort=reasoning_effort,
            max_steps=self.max_steps,
            callbacks=callbacks,
            harness=self.harness,
        )
        self.graph = build_workflow(self.nodes, self.harness)

    async def run(self, task_description: str) -> dict[str, Any]:
        """Run the harnessed workflow for one NIKA task."""

        callbacks: list[Any] = []
        if self.langfuse_handler is not None:
            callbacks.append(self.langfuse_handler)
        inputs = self.harness.load_resume_state(
            task=task_description,
            max_attempts=self.max_attempts,
        ) or {
            "task": task_description,
            "messages": [HumanMessage(content=task_description)],
            "attempts": 0,
            "max_attempts": self.max_attempts,
            "context_summary": "",
            "compression_events": [],
        }
        self.harness.start(inputs)
        try:
            await self.nodes.initialize()
            result = await self.graph.ainvoke(
                inputs,
                config={
                    "callbacks": callbacks,
                    "recursion_limit": max(16, self.max_attempts * 5 + 2),
                },
            )
        except BaseException as exc:
            status = (
                "interrupted"
                if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
                else "failed"
            )
            self.harness.finish(status=status, final_state=None)
            raise
        self.harness.finish(status="completed", final_state=result)
        return result

    def _load_langfuse_handler(self) -> Any | None:
        try:
            from nika.run_config.loader import get_run_config

            observability = get_run_config().nika.observability
            enabled = bool(observability.langfuse_enabled)
            if observability.langfuse_host:
                os.environ.setdefault("LANGFUSE_HOST", observability.langfuse_host)
        except Exception:  # noqa: BLE001 - observability must not block diagnosis
            enabled = False
        if not enabled:
            return None

        try:
            from langfuse import get_client
            from langfuse.langchain import CallbackHandler
        except ImportError as exc:
            raise RuntimeError(
                "Observability langfuse is enabled, but langfuse is not installed. "
                "Install the observability extra or disable langfuse."
            ) from exc

        langfuse = get_client()
        handler = CallbackHandler()
        if langfuse.auth_check():
            system_logger.info("Authentication to Langfuse successful.")
        else:
            system_logger.warning("Authentication to Langfuse failed.")
        return handler
