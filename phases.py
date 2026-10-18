"""Independent LangChain ReAct phases used by NetSleuth's LangGraph."""

from langchain.agents import create_agent
from langchain_core.tools.structured import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.community.netsleuth.graph.context import build_context_middleware
from agent.community.netsleuth.prompts import (
    NETSLEUTH_DIAGNOSIS_PROMPT,
    NETSLEUTH_SUBMISSION_PROMPT,
    NETSLEUTH_VERIFIER_PROMPT,
)
from agent.community.netsleuth.tool_policy import enforce_no_remediation_policy
from agent.llm.model_factory import load_model
from agent.protocols import DIAGNOSIS, SUBMISSION
from agent.utils.mcp_client import load_session_mcp_config
from agent.utils.mcp_servers import harden_mcp_tools
from nika.utils.session import Session


class NetSleuthDiagnosisPhase:
    """Load session MCP tools and build the diagnostic executor."""

    def __init__(
        self,
        session_id: str,
        llm_provider: str,
        model: str,
        scenario_name: str,
        reasoning_effort: str | None = None,
    ) -> None:
        config = load_session_mcp_config(session_id, scenario_name)
        self.client = MultiServerMCPClient(connections=config)
        self.tools: list[StructuredTool] | None = None
        self.llm = load_model(
            llm_provider=llm_provider,
            model=model,
            reasoning_effort=reasoning_effort,
        )

    async def load_tools(self) -> None:
        self.tools = await self.client.get_tools()
        harden_mcp_tools(self.tools)
        enforce_no_remediation_policy(self.tools)
        for tool in self.tools:
            tool.handle_tool_error = True
            tool.handle_validation_error = True

    def get_agent(self):
        return create_agent(
            model=self.llm,
            system_prompt=NETSLEUTH_DIAGNOSIS_PROMPT,
            tools=self.tools,
            middleware=[build_context_middleware(self.llm)],
            name=DIAGNOSIS,
        )


class NetSleuthVerifierPhase:
    """Build an independent, read-only verifier over diagnosis-time tools."""

    def __init__(
        self,
        session_id: str,
        llm_provider: str,
        model: str,
        scenario_name: str,
        reasoning_effort: str | None = None,
    ) -> None:
        config = load_session_mcp_config(session_id, scenario_name)
        self.client = MultiServerMCPClient(connections=config)
        self.tools: list[StructuredTool] | None = None
        self.llm = load_model(
            llm_provider=llm_provider,
            model=model,
            reasoning_effort=reasoning_effort,
        )

    async def load_tools(self) -> None:
        self.tools = await self.client.get_tools()
        harden_mcp_tools(self.tools)
        enforce_no_remediation_policy(self.tools)
        for tool in self.tools:
            tool.handle_tool_error = True
            tool.handle_validation_error = True

    def get_agent(self):
        return create_agent(
            model=self.llm,
            system_prompt=NETSLEUTH_VERIFIER_PROMPT,
            tools=self.tools,
            middleware=[build_context_middleware(self.llm)],
            name="verifier",
        )


class NetSleuthSubmissionPhase:
    """Load submission MCP tools and build the canonical submit worker."""

    def __init__(
        self,
        session_id: str,
        llm_provider: str,
        model: str,
        scenario_name: str,
        reasoning_effort: str | None = None,
    ) -> None:
        session = Session()
        session.load_running_session(session_id=session_id)
        config = load_session_mcp_config(
            session_id,
            scenario_name or session.scenario_name,
        )
        self.client = MultiServerMCPClient(connections=config)
        self.tools: list[StructuredTool] | None = None
        self.llm = load_model(
            llm_provider=llm_provider,
            model=model,
            reasoning_effort=reasoning_effort,
        )

    async def load_tools(self) -> None:
        self.tools = await self.client.get_tools()
        harden_mcp_tools(self.tools)
        for tool in self.tools:
            tool.handle_tool_error = True
            tool.handle_validation_error = True

    def get_agent(self):
        return create_agent(
            model=self.llm,
            system_prompt=NETSLEUTH_SUBMISSION_PROMPT,
            tools=self.tools,
            name=SUBMISSION,
        )
