from __future__ import annotations

from ..schemas import CriticResult, PaperMeta
from .base import BaseCritic


class MLCritic(BaseCritic):
    name = "ml"

    ML_PRESENCE_PATTERNS = [
        r"machine\s+learning", r"deep\s+learning", r"neural\s+network",
        r"random\s+forest", r"gradient\s+boost", r"xgboost", r"lightgbm",
        r"lstm", r"transformer", r"reinforcement\s+learning",
        r"supervised\s+learning", r"classification", r"regression\s+model",
        r"support\s+vector", r"svm\b", r"convolutional",
    ]
    SPLIT_PATTERNS = [
        r"temporal\s+split", r"time[\s\-]?series\s+split",
        r"walk[\s\-]?forward", r"expanding\s+window",
        r"train.*test.*split", r"chronological",
        r"no\s+future\s+information", r"purged\s+cross[\s\-]?validation",
    ]
    LEAKAGE_PATTERNS = [
        r"look[\s\-]?ahead\s+bias", r"information\s+leakage",
        r"data\s+leakage", r"future\s+information",
        r"target\s+leakage", r"leaky\s+feature",
    ]
    MODEL_SELECTION_PATTERNS = [
        r"hyperparameter\s+(tuning|optimization|search)",
        r"grid\s+search", r"random\s+search", r"bayesian\s+optimization",
        r"cross[\s\-]?validation", r"model\s+selection",
        r"information\s+criterion", r"aic\b", r"bic\b",
    ]
    FEATURE_PATTERNS = [
        r"feature\s+(engineering|selection|importance)",
        r"economic\s+(intuition|rationale|meaning)",
        r"domain\s+knowledge", r"financial\s+theory",
        r"factor\s+model", r"fundamental\s+feature",
    ]
    REPRODUCIBILITY_PATTERNS = [
        r"source\s+code", r"github\.com", r"open[\s\-]?source",
        r"reproducib", r"code\s+availab", r"data\s+availab",
        r"implementation\s+detail",
    ]

    def evaluate(self, paper_text: str, meta: PaperMeta) -> CriticResult:
        text = paper_text

        # Check if paper has ML component at all
        ml_count = self._count_matches(text, self.ML_PRESENCE_PATTERNS)
        if ml_count == 0:
            return CriticResult(
                critic_name=self.name,
                score=5,
                flags=[],
                summary="Paper does not use ML methods; neutral score assigned.",
                sub_scores={
                    "train_test_split": 5,
                    "leakage": 5,
                    "model_selection": 5,
                    "feature_engineering": 5,
                    "reproducibility": 5,
                },
            )

        flags = []

        split_count = self._count_matches(text, self.SPLIT_PATTERNS)
        split_score = self._clamp(min(split_count * 2, 10))

        leak_count = self._count_matches(text, self.LEAKAGE_PATTERNS)
        # Mentioning leakage awareness is good; not mentioning when using ML is bad
        if leak_count > 0:
            leak_score = self._clamp(min(leak_count * 3, 10))
        else:
            leak_score = 3
            flags.append("lookahead_bias")

        if split_count == 0:
            flags.append("leaky_features")

        model_count = self._count_matches(text, self.MODEL_SELECTION_PATTERNS)
        model_score = self._clamp(min(model_count * 2, 10))
        if model_count == 0:
            flags.append("overfit_hyperparams")

        feat_count = self._count_matches(text, self.FEATURE_PATTERNS)
        feat_score = self._clamp(min(feat_count * 2, 10))
        if feat_count == 0:
            flags.append("no_feature_rationale")

        repro_count = self._count_matches(text, self.REPRODUCIBILITY_PATTERNS)
        repro_score = self._clamp(min(repro_count * 2, 10))

        # Check for black-box without interpretability
        if self._has_any(text, [r"neural\s+network", r"deep\s+learning", r"lstm", r"transformer"]):
            if not self._has_any(text, [r"interpret", r"explain", r"shap", r"attention\s+weight"]):
                flags.append("black_box_no_interpretability")

        sub_scores = {
            "train_test_split": split_score,
            "leakage": leak_score,
            "model_selection": model_score,
            "feature_engineering": feat_score,
            "reproducibility": repro_score,
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
            return "Strong ML methodology with proper temporal splits and leakage prevention."
        elif score >= 5:
            issues = ", ".join(flags) if flags else "some ML methodology gaps"
            return f"Moderate ML rigor; concerns: {issues}."
        else:
            issues = ", ".join(flags) if flags else "serious ML methodology issues"
            return f"Weak ML implementation: {issues}."
