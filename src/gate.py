from __future__ import annotations

from . import config
from .schemas import CriticResult


def check_early_exit(critic_results: dict[str, CriticResult]) -> bool:
    """Return True if the paper should be auto-rejected (bucket C).

    Triggers when the statistical critic score is below the threshold.
    """
    stat_result = critic_results.get("statistical")
    if stat_result is None:
        return False
    return stat_result.score < config.EARLY_EXIT_THRESHOLD
