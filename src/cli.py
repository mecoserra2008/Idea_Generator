"""CLI entry point for the arXiv trading research screening pipeline."""
from __future__ import annotations

import argparse
import json
import logging
import sys

from . import config, pipeline
from .schemas import ClassificationError, FetchError


def _setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="idea-screener",
        description="arXiv trading research screening pipeline",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")
    parser.add_argument("-q", "--quiet", action="store_true", help="WARNING-level logging only")
    parser.add_argument("--timeout", type=int, default=120, help="Per-critic timeout in seconds")

    sub = parser.add_subparsers(dest="command", required=True)

    for cmd in ("screen", "re-score", "fetch", "classify"):
        p = sub.add_parser(cmd)
        p.add_argument("arxiv_id", type=str, help="arXiv paper ID or URL")
        if cmd in ("screen", "fetch"):
            p.add_argument("--force", action="store_true", help="Re-fetch / overwrite existing files")

    # discover subcommand — uses long-form --query to avoid conflict with global -q/--quiet
    disc = sub.add_parser("discover", help="Autonomous discovery of recent trading papers")
    disc.add_argument("--query", action="append", default=None,
                      help="Custom search query (repeatable; overrides defaults)")
    disc.add_argument("-n", "--max-results", type=int, default=None,
                      help=f"Max results per query (default: {config.DISCOVERY_MAX_RESULTS})")
    disc.add_argument("-d", "--days-back", type=int, default=None,
                      help=f"Look back N days (default: {config.DISCOVERY_DAYS_BACK})")
    disc.add_argument("--category", action="append", default=None,
                      help="arXiv category filter (repeatable; overrides defaults)")
    disc.add_argument("--rate-delay", type=float, default=None,
                      help=f"Seconds between API calls (default: {config.DISCOVERY_RATE_DELAY})")
    disc.add_argument("--dry-run", action="store_true",
                      help="Search and filter only; don't screen papers")

    args = parser.parse_args(argv)
    _setup_logging(verbose=args.verbose, quiet=args.quiet)

    try:
        if args.command == "discover":
            from .discovery import run_discovery
            report = run_discovery(
                queries=args.query,
                categories=args.category,
                max_results=args.max_results or config.DISCOVERY_MAX_RESULTS,
                days_back=args.days_back or config.DISCOVERY_DAYS_BACK,
                rate_delay=args.rate_delay if args.rate_delay is not None else config.DISCOVERY_RATE_DELAY,
                timeout=args.timeout,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                print(json.dumps(report.to_dict(), indent=2))
            return 0

        elif args.command == "screen":
            entry = pipeline.run_screen(args.arxiv_id, force=args.force, timeout=args.timeout)
        elif args.command == "re-score":
            entry = pipeline.run_rescore(args.arxiv_id, timeout=args.timeout)
        elif args.command == "fetch":
            meta = pipeline.run_fetch_only(args.arxiv_id, force=args.force)
            print(json.dumps(meta.to_dict(), indent=2))
            return 0
        elif args.command == "classify":
            entry = pipeline.run_classify_only(args.arxiv_id)
        else:
            parser.print_help()
            return 2

        print(json.dumps(entry.to_dict(), indent=2))
        return 0

    except FetchError as exc:
        logging.error("Fetch failed: %s", exc)
        return 1
    except ClassificationError as exc:
        logging.error("Classification failed: %s", exc)
        return 1
    except FileNotFoundError as exc:
        logging.error("Missing file: %s", exc)
        return 1
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
