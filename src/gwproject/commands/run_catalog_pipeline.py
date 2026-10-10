from __future__ import annotations

import argparse
import sys


def _dispatch(module_main, extra: list[str]) -> int:
    old = sys.argv
    try:
        sys.argv = [old[0], *extra]
        return int(module_main())
    finally:
        sys.argv = old


def main() -> int:
    parser = argparse.ArgumentParser(prog="gwproject catalog", description="Orquestrador da analise de catalogo")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("index", "refresh-metadata", "ridge", "matched-filter", "diagnose-ridge"):
        command = sub.add_parser(name)
        command.add_argument("extra", nargs=argparse.REMAINDER)
    classify = sub.add_parser("classify-qgrams")
    classify.add_argument("checkpoint")
    classify.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    extra = args.extra[1:] if args.extra and args.extra[0] == "--" else args.extra

    if args.command == "index":
        from gwproject.commands.build_catalog_index import main as run
        return _dispatch(run, extra)
    if args.command == "refresh-metadata":
        from gwproject.commands.refresh_catalog_metadata import main as run
        return _dispatch(run, extra)
    if args.command == "ridge":
        from gwproject.commands.run_catalog_ridge import main as run
        return _dispatch(run, extra)
    if args.command == "matched-filter":
        from gwproject.commands.run_catalog_matched_filter import main as run
        return _dispatch(run, extra)
    if args.command == "diagnose-ridge":
        from gwproject.commands.summarize_ridge_catalog import main as run
        return _dispatch(run, extra)
    from gwproject.commands.classify_catalog_qgrams import main as run
    return _dispatch(run, [args.checkpoint, *extra])


if __name__ == "__main__":
    raise SystemExit(main())
