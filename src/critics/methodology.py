from __future__ import annotations

from ..schemas import CriticResult, PaperMeta
from .base import BaseCritic


class MethodologyCritic(BaseCritic):
    name = "methodology"

    OOS_PATTERNS = [
        r"out[\s\-]?of[\s\-]?sample", r"walk[\s\-]?forward",
        r"rolling\s+window", r"expanding\s+window",
        r"train.*test\s+split", r"holdout\s+(set|period|sample)",
        r"validation\s+(set|period|sample)",
    ]
    DATA_SNOOPING_PATTERNS = [
        r"data\s+snoop", r"look[\s\-]?ahead", r"in[\s\-]?sample\s+only",
        r"optimiz.*parameter.*same\s+data", r"white['']?s\s+reality\s+check",
        r"hansen['']?s\s+spa\s+test", r"multiple\s+testing\s+bias",
    ]
    BENCHMARK_PATTERNS = [
        r"benchmark", r"buy[\s\-]?and[\s\-]?hold", r"s&p\s*500",
        r"market\s+portfolio", r"equal[\s\-]?weight",
        r"risk[\s\-]?free\s+rate", r"compared?\s+to\s+(the\s+)?market",
        r"baseline\s+strateg", r"passive\s+strateg",
    ]
    ROBUSTNESS_PATTERNS = [
        r"robustness", r"sensitivity\s+analysis", r"parameter\s+sensitivity",
        r"ablation", r"stress\s+test", r"bootstrap",
        r"monte\s+carlo\s+simul", r"permutation\s+test",
    ]
    TIME_PERIOD_PATTERNS = [
        r"(bull|bear)\s+market", r"financial\s+crisis", r"regime",
        r"(multiple|different|various)\s+(market\s+)?(regimes?|conditions?|periods?)",
        r"199\d.*20[12]\d", r"long[\s\-]?term\s+backtest",
        r"decades?\s+of\s+data", r"20\+?\s+years",
    ]

    def evaluate(self, paper_text: str, meta: PaperMeta) -> CriticResult:
        flags = []
        text = paper_text

        oos_count = self._count_matches(text, self.OOS_PATTERNS)
        oos_score = self._clamp(min(oos_count * 2, 10))
        if oos_count == 0:
            flags.append("no_oos_test")

        snoop_count = self._count_matches(text, self.DATA_SNOOPING_PATTERNS)
        # High count of snooping mentions without mitigation is bad
        if self._has_any(text, [r"data\s+snoop", r"look[\s\-]?ahead"]):
            if not self._has_any(text, [r"white", r"hansen", r"correct"]):
                flags.append("data_snooping_bias")
        snoop_score = self._clamp(10 - snoop_count) if snoop_count > 0 else 3

        bench_count = self._count_matches(text, self.BENCHMARK_PATTERNS)
        bench_score = self._clamp(min(bench_count * 2, 10))
        if bench_count == 0:
            flags.append("weak_benchmark")

        robust_count = self._count_matches(text, self.ROBUSTNESS_PATTERNS)
        robust_score = self._clamp(min(robust_count * 2, 10))
        if robust_count == 0:
            flags.append("curve_fitting")

        period_count = self._count_matches(text, self.TIME_PERIOD_PATTERNS)
        period_score = self._clamp(min(period_count * 2, 10))
        if period_count == 0:
            flags.append("single_regime")

        sub_scores = {
            "out_of_sample": oos_score,
            "data_snooping": snoop_score,
            "benchmark": bench_score,
            "robustness": robust_score,
            "time_period": period_score,
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
            return "Sound methodology with proper out-of-sample testing and robustness checks."
        elif score >= 5:
            issues = ", ".join(flags) if flags else "some methodological gaps"
            return f"Acceptable methodology with concerns: {issues}."
        else:
            issues = ", ".join(flags) if flags else "major methodological weaknesses"
            return f"Poor methodology: {issues}."
