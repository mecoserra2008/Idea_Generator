from __future__ import annotations

from ..schemas import CriticResult, PaperMeta
from .base import BaseCritic


class StatisticalCritic(BaseCritic):
    name = "statistical"

    SIGNIFICANCE_PATTERNS = [
        r"p[\s\-]?value", r"confidence\s+interval", r"statistical\s+significance",
        r"t[\s\-]?test", r"t[\s\-]?statistic", r"z[\s\-]?score",
        r"chi[\s\-]?square", r"f[\s\-]?test", r"anova",
        r"significance\s+level", r"alpha\s*=", r"reject\s+.*null",
    ]
    MULTIPLE_TESTING_PATTERNS = [
        r"bonferroni", r"holm", r"benjamini", r"fdr\b",
        r"false\s+discovery\s+rate", r"multiple\s+(comparisons?|testing|hypothesis)",
        r"family[\s\-]?wise\s+error",
    ]
    SAMPLE_SIZE_PATTERNS = [
        r"sample\s+size", r"\bn\s*=\s*\d", r"observations?\s+\d",
        r"data\s+points?", r"trading\s+days?", r"tick\s+data",
        r"years?\s+of\s+data", r"dataset\s+(contains?|comprises?|includes?)",
    ]
    EFFECT_SIZE_PATTERNS = [
        r"effect\s+size", r"cohen['\u2019]?s\s+d", r"sharpe\s+ratio",
        r"information\s+ratio", r"risk[\s\-]?adjusted\s+return",
        r"annualized\s+return", r"excess\s+return",
    ]
    STATIONARITY_PATTERNS = [
        r"stationar", r"regime\s+(change|switch|shift)",
        r"structural\s+break", r"unit\s+root", r"augmented\s+dickey",
        r"cointegrat", r"non[\s\-]?stationary", r"time[\s\-]?varying",
    ]

    def evaluate(self, paper_text: str, meta: PaperMeta) -> CriticResult:
        flags = []
        text = paper_text

        sig_count = self._count_matches(text, self.SIGNIFICANCE_PATTERNS)
        sig_score = self._clamp(min(sig_count * 2, 10))

        mult_count = self._count_matches(text, self.MULTIPLE_TESTING_PATTERNS)
        mult_score = self._clamp(min(mult_count * 3, 10))
        if mult_count == 0 and sig_count > 0:
            flags.append("no_multiple_test_correction")

        sample_count = self._count_matches(text, self.SAMPLE_SIZE_PATTERNS)
        sample_score = self._clamp(min(sample_count * 2, 10))
        if sample_count == 0:
            flags.append("tiny_sample")

        effect_count = self._count_matches(text, self.EFFECT_SIZE_PATTERNS)
        effect_score = self._clamp(min(effect_count * 2, 10))
        if effect_count == 0 and sig_count > 0:
            flags.append("no_confidence_intervals")

        stat_count = self._count_matches(text, self.STATIONARITY_PATTERNS)
        stat_score = self._clamp(min(stat_count * 2, 10))

        # Check for p-hacking indicators
        if self._has_any(text, [r"p\s*[<>=]\s*0\.0[0-4]"]) and not self._has_any(text, [r"bonferroni", r"correction"]):
            flags.append("p_hacking")

        if self._has_any(text, [r"(200[0-9]|201[0-9])\s*[-–]\s*(200[0-9]|201[0-9])"]) and not self._has_any(text, [r"(199|198|197)"]):
            flags.append("cherry_picked_periods")

        sub_scores = {
            "significance_reporting": sig_score,
            "multiple_testing": mult_score,
            "sample_size": sample_score,
            "effect_size": effect_score,
            "stationarity": stat_score,
        }
        overall = self._clamp(round(sum(sub_scores.values()) / len(sub_scores)))

        return CriticResult(
            critic_name=self.name,
            score=overall,
            flags=flags,
            summary=self._build_summary(overall, flags),
            sub_scores=sub_scores,
        )

    def _build_summary(self, score: int, flags: list[str]) -> str:
        if score >= 8:
            return "Strong statistical rigor with proper reporting and corrections."
        elif score >= 5:
            issues = ", ".join(flags) if flags else "some gaps in reporting"
            return f"Moderate statistical quality; concerns: {issues}."
        else:
            issues = ", ".join(flags) if flags else "insufficient statistical reporting"
            return f"Weak statistical foundation: {issues}."
