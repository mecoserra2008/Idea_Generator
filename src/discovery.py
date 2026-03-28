"""Autonomous arXiv paper discovery for trading research."""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone

import arxiv

from . import config
from .pipeline import run_screen
from .schemas import BucketEntry, DiscoveryReport, FetchError

logger = logging.getLogger(__name__)


def _extract_id(result: arxiv.Result) -> str:
    """Extract bare arXiv ID from a result's entry_id."""
    m = re.search(r"(\d{4}\.\d{4,5})", result.entry_id)
    if m:
        return m.group(1)
    return result.entry_id


def _build_query(user_query: str, categories: list[str]) -> str:
    """Build an arXiv API query string combining the user query with category filters."""
    cat_clause = " OR ".join(f"cat:{c}" for c in categories)
    return f"({user_query}) AND ({cat_clause})"


def load_known_ids() -> set[str]:
    """Read bucket.json and return set of already-screened arXiv IDs."""
    known: set[str] = set()
    if not config.BUCKET_FILE.exists():
        return known
    with open(config.BUCKET_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                known.add(json.loads(line)["arxiv_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return known


def search_arxiv(
    queries: list[str],
    categories: list[str],
    max_results: int,
    days_back: int,
    rate_delay: float,
) -> list[arxiv.Result]:
    """Search arXiv with multiple queries, merge and deduplicate results."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    seen_ids: set[str] = set()
    merged: list[arxiv.Result] = []
    client = arxiv.Client()

    for i, query_text in enumerate(queries):
        full_query = _build_query(query_text, categories)
        logger.info("Query %d/%d: %s", i + 1, len(queries), query_text)

        search = arxiv.Search(
            query=full_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        try:
            for result in client.results(search):
                arxiv_id = _extract_id(result)
                if arxiv_id in seen_ids:
                    continue
                if result.published and result.published.replace(tzinfo=timezone.utc) < cutoff:
                    continue
                seen_ids.add(arxiv_id)
                merged.append(result)
        except Exception as exc:
            logger.error("Search failed for query %r: %s", query_text, exc)

        if i < len(queries) - 1:
            time.sleep(rate_delay)

    logger.info("Search complete: %d unique papers found", len(merged))
    return merged


def filter_results(
    results: list[arxiv.Result],
    known_ids: set[str],
) -> tuple[list[arxiv.Result], int]:
    """Remove already-screened papers. Returns (filtered, num_skipped)."""
    filtered = []
    skipped = 0
    for result in results:
        arxiv_id = _extract_id(result)
        if arxiv_id in known_ids:
            skipped += 1
            continue
        filtered.append(result)
    return filtered, skipped


def batch_screen(
    results: list[arxiv.Result],
    rate_delay: float,
    timeout: int,
) -> tuple[list[BucketEntry], int]:
    """Screen papers sequentially through the full pipeline.

    Returns (entries, fail_count).
    """
    entries: list[BucketEntry] = []
    failures = 0

    for i, result in enumerate(results):
        arxiv_id = _extract_id(result)
        title = result.title[:60] + "..." if len(result.title) > 60 else result.title
        logger.info("[%d/%d] Screening %s: %s", i + 1, len(results), arxiv_id, title)

        try:
            entry = run_screen(arxiv_id, timeout=timeout)
            entries.append(entry)
            logger.info("  -> Bucket %s (composite %.1f)", entry.bucket, entry.composite)
        except (FetchError, Exception) as exc:
            logger.error("  -> FAILED: %s", exc)
            failures += 1

        if i < len(results) - 1:
            time.sleep(rate_delay)

    return entries, failures


def format_summary(report: DiscoveryReport) -> str:
    """Format a human-readable summary report."""
    lines = [
        "",
        "=" * 65,
        f"  DISCOVERY REPORT -- {report.run_at}",
        "=" * 65,
        f"  Queries:       {len(report.queries_used)}",
        f"  Categories:    {', '.join(report.categories)}",
        f"  Lookback:      {report.days_back} days",
        "-" * 65,
        f"  Search hits:   {report.total_search_results}",
        f"  Already seen:  {report.duplicates_skipped} (in bucket.json)",
        f"  To screen:     {report.total_screened + report.failed}",
        f"  Screened OK:   {report.total_screened}",
        f"  Errors:        {report.failed}",
        "-" * 65,
        "  BUCKET BREAKDOWN",
    ]

    for bucket in ("A", "B", "C", "D"):
        labels = {"A": "actionable", "B": "worth reading", "C": "rejected", "D": "needs work"}
        count = report.by_bucket.get(bucket, 0)
        lines.append(f"    {bucket} ({labels[bucket]}):  {count}")

    # Top papers (bucket A and B, sorted by composite)
    top = sorted(
        [e for e in report.entries if e.bucket in ("A", "B")],
        key=lambda e: e.composite,
        reverse=True,
    )[:10]

    if top:
        lines.append("-" * 65)
        lines.append("  TOP PAPERS")
        for e in top:
            title = e.title[:50] + "..." if len(e.title) > 50 else e.title
            lines.append(f'    [{e.arxiv_id}] "{title}" -- {e.bucket} composite {e.composite}')

    lines.append("=" * 65)
    return "\n".join(lines)


def format_dry_run(results: list[arxiv.Result]) -> str:
    """Format a dry-run listing of papers that would be screened."""
    lines = [
        "",
        f"DRY RUN: {len(results)} papers would be screened:",
        "-" * 65,
    ]
    for i, result in enumerate(results, 1):
        arxiv_id = _extract_id(result)
        title = result.title[:55] + "..." if len(result.title) > 55 else result.title
        cats = ", ".join(result.categories[:3])
        date = result.published.strftime("%Y-%m-%d") if result.published else "?"
        lines.append(f"  {i:3d}. [{arxiv_id}] {date} | {title}")
        lines.append(f"       Categories: {cats}")
    lines.append("-" * 65)
    return "\n".join(lines)


def run_discovery(
    queries: list[str] | None = None,
    categories: list[str] | None = None,
    max_results: int = config.DISCOVERY_MAX_RESULTS,
    days_back: int = config.DISCOVERY_DAYS_BACK,
    rate_delay: float = config.DISCOVERY_RATE_DELAY,
    timeout: int = config.CRITIC_TIMEOUT_SECONDS,
    dry_run: bool = False,
) -> DiscoveryReport:
    """Run the full discovery pipeline: search -> deduplicate -> screen -> report."""
    queries = queries or config.DISCOVERY_DEFAULT_QUERIES
    categories = categories or config.DISCOVERY_DEFAULT_CATEGORIES

    logger.info("=== DISCOVERY START ===")
    logger.info("Queries: %d, Categories: %s, Days back: %d, Max results/query: %d",
                len(queries), categories, days_back, max_results)

    # Search
    results = search_arxiv(queries, categories, max_results, days_back, rate_delay)
    total_search = len(results)

    # Deduplicate against bucket.json
    known_ids = load_known_ids()
    results, skipped = filter_results(results, known_ids)

    if dry_run:
        print(format_dry_run(results))
        return DiscoveryReport(
            queries_used=queries,
            categories=categories,
            days_back=days_back,
            total_search_results=total_search,
            duplicates_skipped=skipped,
            total_screened=0,
            failed=0,
            by_bucket={},
            entries=[],
            run_at=datetime.now(timezone.utc).isoformat(),
        )

    # Batch screen
    logger.info("Screening %d papers...", len(results))
    entries, failures = batch_screen(results, rate_delay, timeout)

    # Build report
    by_bucket: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}
    for entry in entries:
        by_bucket[entry.bucket] = by_bucket.get(entry.bucket, 0) + 1

    report = DiscoveryReport(
        queries_used=queries,
        categories=categories,
        days_back=days_back,
        total_search_results=total_search,
        duplicates_skipped=skipped,
        total_screened=len(entries),
        failed=failures,
        by_bucket=by_bucket,
        entries=entries,
        run_at=datetime.now(timezone.utc).isoformat(),
    )

    print(format_summary(report))
    logger.info("=== DISCOVERY COMPLETE ===")
    return report
