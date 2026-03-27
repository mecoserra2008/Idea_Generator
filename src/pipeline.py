from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config
from .classifier import append_bucket, classify
from .critics import CRITIC_REGISTRY
from .critics.base import BaseCritic
from .fetcher import fetch_paper
from .gate import check_early_exit
from .schemas import CriticResult, FetchError, PaperMeta

logger = logging.getLogger(__name__)


def _run_critic(critic: BaseCritic, paper_text: str, meta: PaperMeta, arxiv_id: str) -> CriticResult:
    """Run a single critic and save its report."""
    result = critic.safe_evaluate(paper_text, meta)
    critic.save_report(arxiv_id, result)
    logger.info("Critic %s scored %s: %d/10", critic.name, arxiv_id, result.score)
    return result


def _run_all_critics(arxiv_id: str, paper_text: str, meta: PaperMeta, timeout: int | None = None) -> dict[str, CriticResult]:
    """Run all four critics in parallel and return results."""
    timeout = timeout or config.CRITIC_TIMEOUT_SECONDS
    results: dict[str, CriticResult] = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for name, critic_cls in CRITIC_REGISTRY.items():
            critic = critic_cls()
            future = executor.submit(_run_critic, critic, paper_text, meta, arxiv_id)
            futures[future] = name

        for future in as_completed(futures, timeout=timeout):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                logger.error("Critic %s timed out or failed: %s", name, exc)
                results[name] = CriticResult(
                    critic_name=name,
                    score=0,
                    flags=["timeout"],
                    summary=f"Critic failed: {exc}",
                    sub_scores={},
                )

    return results


def _load_paper(arxiv_id: str) -> tuple[str, PaperMeta]:
    """Load paper text and metadata from disk."""
    text_path = config.PAPERS_DIR / f"{arxiv_id}.txt"
    meta_path = config.PAPERS_DIR / f"{arxiv_id}_meta.json"

    if not text_path.exists() or not meta_path.exists():
        raise FetchError(f"Paper files not found for {arxiv_id}. Run fetch first.")

    paper_text = text_path.read_text(encoding="utf-8")
    with open(meta_path) as f:
        meta = PaperMeta.from_dict(json.load(f))
    return paper_text, meta


def _load_critic_reports(arxiv_id: str) -> dict[str, CriticResult]:
    """Load existing critic reports from disk."""
    results: dict[str, CriticResult] = {}
    for name in CRITIC_REGISTRY:
        path = config.REPORTS_DIR / f"{arxiv_id}_{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Critic report not found: {path}")
        with open(path) as f:
            results[name] = CriticResult.from_dict(json.load(f))
    return results


def run_screen(arxiv_id: str, force: bool = False, timeout: int | None = None):
    """Full pipeline: fetch → critique → gate → classify."""
    logger.info("=== SCREENING %s ===", arxiv_id)

    # Step 1: Fetch
    meta = fetch_paper(arxiv_id, force=force)
    paper_text = (config.PAPERS_DIR / f"{meta.arxiv_id}.txt").read_text(encoding="utf-8")
    arxiv_id = meta.arxiv_id  # normalized

    # Step 2: Parallel critique
    critic_results = _run_all_critics(arxiv_id, paper_text, meta, timeout=timeout)

    # Step 3: Early-exit gate
    early_exit = check_early_exit(critic_results)
    if early_exit:
        logger.warning("Early exit triggered for %s (statistical score < %d)", arxiv_id, config.EARLY_EXIT_THRESHOLD)

    # Step 4: Classify
    entry = classify(arxiv_id, meta, critic_results, early_exit=early_exit)
    append_bucket(entry)

    logger.info("=== RESULT: %s → Bucket %s (composite %.1f) ===", arxiv_id, entry.bucket, entry.composite)
    return entry


def run_rescore(arxiv_id: str, timeout: int | None = None):
    """Re-run critics + classifier on an already-fetched paper."""
    logger.info("=== RE-SCORING %s ===", arxiv_id)
    paper_text, meta = _load_paper(arxiv_id)
    critic_results = _run_all_critics(arxiv_id, paper_text, meta, timeout=timeout)
    early_exit = check_early_exit(critic_results)
    entry = classify(arxiv_id, meta, critic_results, early_exit=early_exit)
    append_bucket(entry)
    logger.info("=== RESULT: %s → Bucket %s (composite %.1f) ===", arxiv_id, entry.bucket, entry.composite)
    return entry


def run_fetch_only(arxiv_id: str, force: bool = False):
    """Fetch paper only (step 1)."""
    meta = fetch_paper(arxiv_id, force=force)
    logger.info("Fetched: %s — %s", meta.arxiv_id, meta.title)
    return meta


def run_classify_only(arxiv_id: str):
    """Classify from existing critic reports (step 4)."""
    logger.info("=== CLASSIFYING %s ===", arxiv_id)
    _, meta = _load_paper(arxiv_id)
    critic_results = _load_critic_reports(arxiv_id)
    early_exit = check_early_exit(critic_results)
    entry = classify(arxiv_id, meta, critic_results, early_exit=early_exit)
    append_bucket(entry)
    logger.info("=== RESULT: %s → Bucket %s (composite %.1f) ===", arxiv_id, entry.bucket, entry.composite)
    return entry
