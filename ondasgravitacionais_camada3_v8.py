from __future__ import annotations

import argparse
import sys


def _dispatch(main, extra):
    old = sys.argv
    try:
        sys.argv = [old[0], *extra]
        return main()
    finally:
        sys.argv = old


def main() -> int:
    parser = argparse.ArgumentParser(description="Camada 3 v8: GW150914, catalogo ou Gravity Spy")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("matched-filter", "catalog-matched-filter"):
        command = sub.add_parser(name)
        command.add_argument("extra", nargs=argparse.REMAINDER)
    gravity = sub.add_parser("gravity-spy")
    gravity.add_argument("metadata")
    gravity.add_argument("images_root")
    gravity.add_argument("extra", nargs=argparse.REMAINDER)
    classify = sub.add_parser("classify-catalog-qgrams")
    classify.add_argument("checkpoint")
    classify.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    extra = args.extra[1:] if args.extra and args.extra[0] == "--" else args.extra

    if args.command == "matched-filter":
        from scripts.run_matched_filter_gw150914 import main as run
        return _dispatch(run, extra)
    if args.command == "catalog-matched-filter":
        from scripts.run_catalog_matched_filter import main as run
        return _dispatch(run, extra)
    if args.command == "gravity-spy":
        from scripts.train_gravity_spy import main as run
        return _dispatch(run, [args.metadata, args.images_root, *extra])
    from scripts.classify_catalog_qgrams import main as run
    return _dispatch(run, [args.checkpoint, *extra])


if __name__ == "__main__":
    raise SystemExit(main())
