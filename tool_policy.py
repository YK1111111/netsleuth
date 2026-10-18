"""No-remediation guard for NetSleuth diagnosis-time MCP tools."""

from __future__ import annotations

import functools
import re
from typing import Any

from langchain_core.tools import ToolException

_MUTATING_SHELL = re.compile(
    r"(?:^|[\n;&|]\s*)"
    r"(?:sudo(?:\s+-\S+)*\s+)?"
    r"(?:env\s+(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*)?"
    r"(?:rm|mv|cp|touch|mkdir|rmdir|chmod|chown|kill|pkill|killall|"
    r"reboot|shutdown|mount|umount|apt|apt-get|apk|dnf|yum|pip|npm|"
    r"bash\s+-c|sh\s+-c|"
    r"systemctl\s+(?:start|stop|restart|reload|enable|disable)|"
    r"service\s+\S+\s+(?:start|stop|restart|reload)|"
    r"sysctl\s+-w|sed\s+-i|tee|truncate|dd)\b",
    flags=re.IGNORECASE,
)
_NETWORK_MUTATION = re.compile(
    r"\b(?:ip\s+(?:link|addr|address|route|rule|neigh)\s+"
    r"(?:add|del|delete|replace|change|set|flush)|"
    r"tc\s+(?:qdisc|class|filter)\s+(?:add|del|delete|replace|change)|"
    r"iptables\s+.*(?:-[AIDFRXP]|--append|--insert|--delete|--flush)|"
    r"nft\s+(?:add|delete|insert|replace|flush))\b",
    flags=re.IGNORECASE,
)
_OUTPUT_REDIRECTION = re.compile(r">{1,2}(?!\s*/dev/null(?:\s|$))")
_DEVICE_MUTATION = re.compile(
    r"(?:^|[;\n])\s*(?:configure|conf(?:igure)?\s+t|set|delete|no|clear|reset|"
    r"reload|write|copy|commit|admin)\b",
    flags=re.IGNORECASE,
)
_FORBIDDEN_BENCHMARK_PATH = re.compile(
    r"(?:ground[_-]?truth|submission\.json|answer(?:s)?\.json)",
    flags=re.IGNORECASE,
)
_SHELL_TOOLS = {"exec_shell": ("command",), "exec_shell_dual": ("cmd1", "cmd2")}
_DEVICE_TOOLS = {"frr_exec", "iosxr_exec", "srl_exec_cli", "sdn_ovs_exec"}
_READ_ONLY_SYSTEMCTL = {"status", "show", "is-active", "is-enabled", "list-units"}


def enforce_no_remediation_policy(tools: list[Any]) -> None:
    """Wrap command-capable tools so the benchmark agent cannot repair the lab."""

    for tool in tools:
        tool_name = str(getattr(tool, "name", ""))
        if tool_name == "submit":
            raise RuntimeError("submission tool leaked into the diagnosis phase")
        coroutine = getattr(tool, "coroutine", None)
        if coroutine is None:
            continue
        tool.coroutine = _guarded_coroutine(tool_name, coroutine)


def policy_violation(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Return a stable refusal reason for an unsafe diagnostic request."""

    if tool_name == "systemctl_ops":
        operation = str(arguments.get("operation") or "").strip().lower()
        if operation not in _READ_ONLY_SYSTEMCTL:
            return f"systemctl operation {operation!r} is not read-only"

    if tool_name == "cat_file":
        path = str(arguments.get("file_path") or "")
        if _FORBIDDEN_BENCHMARK_PATH.search(path):
            return "benchmark answer artifacts are outside diagnostic scope"

    for field in _SHELL_TOOLS.get(tool_name, ()):
        violation = _command_violation(str(arguments.get(field) or ""))
        if violation:
            return violation

    if tool_name in _DEVICE_TOOLS:
        command = str(arguments.get("command") or "")
        if _FORBIDDEN_BENCHMARK_PATH.search(command):
            return "benchmark answer artifacts are outside diagnostic scope"
        if _DEVICE_MUTATION.search(command):
            return "device configuration or state mutation is forbidden"
    return None


def _command_violation(command: str) -> str | None:
    if _FORBIDDEN_BENCHMARK_PATH.search(command):
        return "benchmark answer artifacts are outside diagnostic scope"
    if _MUTATING_SHELL.search(command):
        return "host mutation or process control is forbidden"
    if _NETWORK_MUTATION.search(command):
        return "network configuration mutation is forbidden"
    if _OUTPUT_REDIRECTION.search(command):
        return "writing command output to the lab is forbidden"
    return None


def _guarded_coroutine(tool_name: str, coroutine: Any):
    @functools.wraps(coroutine)
    async def guarded(*args: Any, **kwargs: Any):
        arguments = dict(kwargs)
        if args and isinstance(args[0], dict):
            arguments = {**args[0], **arguments}
        violation = policy_violation(tool_name, arguments)
        if violation:
            raise ToolException(
                f"NetSleuth no-remediation policy blocked {tool_name!r}: "
                f"{violation}. Continue with read-only evidence collection."
            )
        return await coroutine(*args, **kwargs)

    return guarded
