#!/usr/bin/env python3
"""Install this standalone NetSleuth source tree into a NIKA checkout."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SOURCE_FILES = (
    "__init__.py",
    "agent.py",
    "harness.py",
    "phases.py",
    "tool_policy.py",
)
SOURCE_DIRS = ("graph", "prompts")


def insert_after(text: str, anchor: str, addition: str, *, path: Path) -> str:
    if anchor + addition in text:
        return text
    if anchor not in text:
        raise ValueError(f"cannot find expected anchor in {path}: {anchor!r}")
    return text.replace(anchor, anchor + addition, 1)


def insert_before(text: str, anchor: str, addition: str, *, path: Path) -> str:
    if addition + anchor in text:
        return text
    if anchor not in text:
        raise ValueError(f"cannot find expected anchor in {path}: {anchor!r}")
    return text.replace(anchor, addition + anchor, 1)


def replace_once(text: str, old: str, new: str, *, path: Path) -> str:
    if new in text:
        return text
    if old not in text:
        raise ValueError(f"cannot find expected text in {path}: {old!r}")
    return text.replace(old, new, 1)


def planned_patches(nika_root: Path) -> dict[Path, str]:
    paths = {
        "registry": nika_root / "src/agent/registry.py",
        "providers": nika_root / "src/agent/utils/provider_env.py",
        "cli": nika_root / "src/nika/cli/commands/agent.py",
        "schema": nika_root / "src/nika/run_config/schema.py",
        "agent_config": nika_root / "src/nika/utils/agent_config.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(
            "not a compatible NIKA checkout; missing: " + ", ".join(missing)
        )

    output = {path: path.read_text(encoding="utf-8") for path in paths.values()}

    path = paths["registry"]
    text = output[path]
    text = insert_after(
        text,
        '        "byo.autogen",\n',
        '        "community.netsleuth",\n',
        path=path,
    )
    registry_case = """        case "community.netsleuth":
            from agent.community.netsleuth.agent import NetSleuthAgent

            return NetSleuthAgent(
                session_id=session_id,
                model=model,
                llm_provider=llm_provider,
                max_steps=max_steps,
                reasoning_effort=reasoning_effort,
                stream_output=stream_output,
            )
"""
    text = insert_before(text, "        case _:\n", registry_case, path=path)
    output[path] = text

    path = paths["providers"]
    text = output[path]
    text = insert_after(
        text,
        '    "byo.langgraph": frozenset({"openai", "anthropic", "deepseek", "custom"}),\n',
        '    "community.netsleuth": frozenset({"openai", "anthropic", "deepseek", "custom"}),\n',
        path=path,
    )
    text = insert_after(
        text,
        '        "byo.langgraph",\n',
        '        "community.netsleuth",\n',
        path=path,
    )
    output[path] = text

    path = paths["cli"]
    output[path] = insert_after(
        output[path],
        '    "community.sade",\n',
        '    "community.netsleuth",\n',
        path=path,
    )

    path = paths["schema"]
    output[path] = replace_once(
        output[path],
        '            case "byo.langgraph":\n',
        '            case "byo.langgraph" | "community.netsleuth":\n',
        path=path,
    )

    path = paths["agent_config"]
    output[path] = replace_once(
        output[path],
        '        case "byo.langgraph":\n',
        '        case "byo.langgraph" | "community.netsleuth":\n',
        path=path,
    )
    return output


def package_conflicts(source_root: Path, target: Path) -> list[str]:
    conflicts: list[str] = []
    for name in SOURCE_FILES:
        source_file = source_root / name
        target_file = target / name
        if (
            target_file.exists()
            and target_file.read_bytes() != source_file.read_bytes()
        ):
            conflicts.append(str(target_file))
    for dirname in SOURCE_DIRS:
        for source_file in (source_root / dirname).rglob("*.py"):
            target_file = target / source_file.relative_to(source_root)
            if (
                target_file.exists()
                and target_file.read_bytes() != source_file.read_bytes()
            ):
                conflicts.append(str(target_file))
    return conflicts


def copy_package(source_root: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        shutil.copy2(source_root / name, target / name)
    for dirname in SOURCE_DIRS:
        shutil.copytree(
            source_root / dirname,
            target / dirname,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy NetSleuth into NIKA and register community.netsleuth."
    )
    parser.add_argument("nika_root", type=Path, help="path to the NIKA repository")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate and list changes only"
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite a different NetSleuth version"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = Path(__file__).resolve().parent.parent
    nika_root = args.nika_root.expanduser().resolve()
    target = nika_root / "src/agent/community/netsleuth"

    try:
        patches = planned_patches(nika_root)
        conflicts = package_conflicts(source_root, target)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if conflicts and not args.force:
        print("error: target contains a different NetSleuth version:", file=sys.stderr)
        for conflict in conflicts:
            print(f"  {conflict}", file=sys.stderr)
        print("rerun with --force only after reviewing those files", file=sys.stderr)
        return 2

    changed_config = [
        path
        for path, content in patches.items()
        if path.read_text(encoding="utf-8") != content
    ]
    print(f"NetSleuth package target: {target}")
    print(f"Package files: {len(SOURCE_FILES)} files plus graph/ and prompts/")
    print(f"NIKA registration files to update: {len(changed_config)}")
    for path in changed_config:
        print(f"  {path.relative_to(nika_root)}")
    if args.dry_run:
        print("Dry run complete; no files were changed.")
        return 0

    try:
        copy_package(source_root, target)
        for path, content in patches.items():
            if path.read_text(encoding="utf-8") != content:
                path.write_text(content, encoding="utf-8")
    except OSError as exc:
        print(f"error while writing files: {exc}", file=sys.stderr)
        return 1

    print(
        "Installation complete. Review the NIKA git diff, then run its tests and lint."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
