from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


COMMANDS = {
    "prepare-panels": "prepare_gravity_spy_panels.py",
    "audit-domain": "audit_current_domain_shift.py",
    "build-ood": "build_ood_reference.py",
    "classify-ensemble": "classify_catalog_qgrams_ensemble.py",
    "phase-counterfactuals": "phase_counterfactuals.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Programa de pesquisa open-world para dados GW")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()
    script = Path(__file__).resolve().parent / "scripts" / COMMANDS[parsed.command]
    arguments = list(parsed.args)
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    sys.argv = [str(script), *arguments]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
