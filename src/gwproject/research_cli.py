from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable


COMMANDS = {
    "audit-official-pairs": "gwproject.commands.audit_official_pairs",
    "paired-intensity": "gwproject.commands.paired_intensity",
    "inspect-official": "gwproject.commands.inspect_official_inputs",
    "prepare-panels": "gwproject.commands.prepare_gravity_spy_panels",
    "audit-domain": "gwproject.commands.audit_current_domain_shift",
    "catalog-statistics": "gwproject.commands.compute_catalog_qgram_statistics",
    "closure-set": "gwproject.commands.build_closure_set",
    "closure-report": "gwproject.commands.closure_report",
    "build-ood": "gwproject.commands.build_ood_reference",
    "classify-ensemble": "gwproject.commands.classify_catalog_qgrams_ensemble",
    "compare-ensembles": "gwproject.commands.compare_ensembles",
    "consolidate-events": "gwproject.commands.consolidate_event_results",
    "gradcam": "gwproject.commands.gradcam",
    "render-qgram": "gwproject.commands.render_qgram",
    "replot-confusion": "gwproject.commands.replot_confusion_matrices",
    "phase-counterfactuals": "gwproject.commands.phase_counterfactuals",
    "train-seeds": "gwproject.commands.run_gravity_spy_seeds",
    "aggregate-seeds": "gwproject.commands.aggregate_gravity_spy",
    "validate-ood": "gwproject.commands.validate_ood_reference",
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
        prog="gwproject research",
        description="Experimentos de validade externa, ensemble e reconhecimento open-world.",
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()
    module = importlib.import_module(COMMANDS[parsed.command])
    arguments = list(parsed.arguments)
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    return _dispatch(module.main, arguments, program=f"gwproject research {parsed.command}")


if __name__ == "__main__":
    raise SystemExit(main())
