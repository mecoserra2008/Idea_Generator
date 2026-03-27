from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

from .. import config
from ..schemas import CriticResult, PaperMeta

logger = logging.getLogger(__name__)


class BaseCritic(ABC):
    name: str = ""

    @abstractmethod
    def evaluate(self, paper_text: str, meta: PaperMeta) -> CriticResult:
        ...

    def save_report(self, arxiv_id: str, result: CriticResult) -> Path:
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.REPORTS_DIR / f"{arxiv_id}_{self.name}.json"
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        return path

    def safe_evaluate(self, paper_text: str, meta: PaperMeta) -> CriticResult:
        """Wrapper that never raises."""
        try:
            return self.evaluate(paper_text, meta)
        except Exception as exc:
            logger.error("Critic %s failed: %s", self.name, exc)
            return CriticResult(
                critic_name=self.name,
                score=0,
                flags=["critic_error"],
                summary=f"Internal error: {exc}",
                sub_scores={},
            )

    @staticmethod
    def _count_matches(text: str, patterns: list[str]) -> int:
        """Count how many patterns match anywhere in text (case-insensitive)."""
        text_lower = text.lower()
        return sum(1 for p in patterns if re.search(p, text_lower))

    @staticmethod
    def _has_any(text: str, patterns: list[str]) -> bool:
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in patterns)

    @staticmethod
    def _clamp(value: int | float, lo: int = 0, hi: int = 10) -> int:
        return max(lo, min(hi, round(value)))
