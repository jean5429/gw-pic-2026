from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable


COMMANDS = {
    "catalog": "gwproject.commands.run_catalog_pipeline",
    "acquire": "gwproject.commands.run_catalog",
    "process": "gwproject.commands.run_processing",
    "validate-event": "gwproject.commands.validate_gw150914",
    "matched-filter-event": "gwproject.commands.run_matched_filter_gw150914",
    "train-gravity-spy": "gwproject.commands.train_gravity_spy",
    "classify-qgrams": "gwproject.commands.classify_catalog_qgrams",
    "research": "gwproject.research_cli",
}


def _dispatch(main_function: Callable[[], int], arguments: list[str], *, program: str) -> int:
    previous = sys.argv
    try:
        sys.argv = [program, *arguments]
        return int(main_function())
    finally:
        sys.argv = previous


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="gwproject",
        description="Pipeline auditavel para catalogos de ondas gravitacionais e robustez de ML.",
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()
    module = importlib.import_module(COMMANDS[parsed.command])
    arguments = list(parsed.arguments)
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    return _dispatch(module.main, arguments, program=f"gwproject {parsed.command}")


if __name__ == "__main__":
    raise SystemExit(main())
