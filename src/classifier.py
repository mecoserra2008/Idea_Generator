from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from . import config
from .schemas import BucketEntry, ClassificationError, CriticResult, PaperMeta

logger = logging.getLogger(__name__)


def classify(
    arxiv_id: str,
    meta: PaperMeta,
    critic_results: dict[str, CriticResult],
    early_exit: bool = False,
) -> BucketEntry:
    """Compute composite score and assign a bucket."""
    scores = {name: r.score for name, r in critic_results.items()}
    all_flags = []
    for r in critic_results.values():
        all_flags.extend(r.flags)
    all_flags = sorted(set(all_flags))

    # Composite weighted average
    composite = sum(
        config.COMPOSITE_WEIGHTS.get(name, 0) * score
        for name, score in scores.items()
    )
    composite = round(composite, 1)

    # Count critical flags
    critical_count = sum(1 for f in all_flags if f in config.CRITICAL_FLAGS)

    # Determine bucket
    if early_exit:
        bucket = "C"
        verdict = "Statistical floor not met — auto-rejected."
    elif composite >= 7.5 and critical_count == 0:
        bucket = "A"
        verdict = f"Actionable research (composite {composite}/10) with no critical flags."
    elif composite >= 5.0 and critical_count <= 1:
        bucket = "B"
        verdict = f"Worth reading (composite {composite}/10); minor concerns noted."
    elif composite < 3.0 or critical_count >= 3:
        bucket = "C"
        reasons = []
        if composite < 3.0:
            reasons.append(f"low composite ({composite}/10)")
        if critical_count >= 3:
            reasons.append(f"{critical_count} critical flags")
        verdict = f"Rejected: {'; '.join(reasons)}."
    else:
        bucket = "D"
        verdict = f"Needs work (composite {composite}/10); {critical_count} critical flag(s)."

    return BucketEntry(
        arxiv_id=arxiv_id,
        title=meta.title,
        bucket=bucket,
        scores=scores,
        composite=composite,
        flags=all_flags,
        verdict=verdict,
        screened_at=datetime.now(timezone.utc).isoformat(),
    )


def append_bucket(entry: BucketEntry) -> None:
    """Append a BucketEntry as one JSON line to bucket.json."""
    try:
        with open(config.BUCKET_FILE, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
        logger.info("Appended bucket entry for %s: bucket=%s", entry.arxiv_id, entry.bucket)
    except OSError as exc:
        raise ClassificationError(f"Failed to write to {config.BUCKET_FILE}: {exc}") from exc
