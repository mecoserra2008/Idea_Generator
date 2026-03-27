import json

import pytest

from src.critics.statistical import StatisticalCritic
from src.critics.methodology import MethodologyCritic
from src.critics.ml import MLCritic
from src.critics.microstructure import MicrostructureCritic
from src.schemas import CriticResult, PaperMeta
from src import config


MINIMAL_TEXT = "This paper proposes a trading strategy."
MINIMAL_META = PaperMeta(
    arxiv_id="0000.00000",
    title="Minimal",
    authors=[],
    abstract="",
    year=2024,
    categories=[],
)


class TestStatisticalCritic:
    def test_returns_valid_result(self, sample_text, sample_meta):
        critic = StatisticalCritic()
        result = critic.evaluate(sample_text, sample_meta)
        assert isinstance(result, CriticResult)
        assert 0 <= result.score <= 10
        assert result.critic_name == "statistical"
        assert isinstance(result.flags, list)
        assert len(result.sub_scores) == 5

    def test_high_quality_paper_scores_well(self, sample_text, sample_meta):
        critic = StatisticalCritic()
        result = critic.evaluate(sample_text, sample_meta)
        assert result.score >= 5

    def test_minimal_paper_scores_low(self):
        critic = StatisticalCritic()
        result = critic.evaluate(MINIMAL_TEXT, MINIMAL_META)
        assert result.score <= 3

    def test_save_report(self, sample_text, sample_meta, tmp_workspace):
        critic = StatisticalCritic()
        result = critic.evaluate(sample_text, sample_meta)
        path = critic.save_report("2401.12345", result)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["critic_name"] == "statistical"
        assert 0 <= data["score"] <= 10


class TestMethodologyCritic:
    def test_returns_valid_result(self, sample_text, sample_meta):
        critic = MethodologyCritic()
        result = critic.evaluate(sample_text, sample_meta)
        assert isinstance(result, CriticResult)
        assert 0 <= result.score <= 10
        assert result.critic_name == "methodology"
        assert len(result.sub_scores) == 5

    def test_high_quality_paper(self, sample_text, sample_meta):
        critic = MethodologyCritic()
        result = critic.evaluate(sample_text, sample_meta)
        assert result.score >= 5

    def test_minimal_paper(self):
        critic = MethodologyCritic()
        result = critic.evaluate(MINIMAL_TEXT, MINIMAL_META)
        assert result.score <= 4


class TestMLCritic:
    def test_returns_valid_result(self, sample_text, sample_meta):
        critic = MLCritic()
        result = critic.evaluate(sample_text, sample_meta)
        assert isinstance(result, CriticResult)
        assert 0 <= result.score <= 10
        assert result.critic_name == "ml"
        assert len(result.sub_scores) == 5

    def test_non_ml_paper_gets_neutral(self):
        text = "We use a simple moving average crossover strategy based on price momentum."
        critic = MLCritic()
        result = critic.evaluate(text, MINIMAL_META)
        assert result.score == 5
        assert result.flags == []

    def test_ml_paper_with_good_practices(self, sample_text, sample_meta):
        critic = MLCritic()
        result = critic.evaluate(sample_text, sample_meta)
        assert result.score >= 5


class TestMicrostructureCritic:
    def test_returns_valid_result(self, sample_text, sample_meta):
        critic = MicrostructureCritic()
        result = critic.evaluate(sample_text, sample_meta)
        assert isinstance(result, CriticResult)
        assert 0 <= result.score <= 10
        assert result.critic_name == "microstructure"
        assert len(result.sub_scores) == 5

    def test_high_quality_paper(self, sample_text, sample_meta):
        critic = MicrostructureCritic()
        result = critic.evaluate(sample_text, sample_meta)
        assert result.score >= 5

    def test_minimal_paper(self):
        critic = MicrostructureCritic()
        result = critic.evaluate(MINIMAL_TEXT, MINIMAL_META)
        assert result.score <= 3


class TestCriticNeverRaises:
    """All critics must handle garbage input gracefully via safe_evaluate."""

    @pytest.mark.parametrize("CriticCls", [
        StatisticalCritic, MethodologyCritic, MLCritic, MicrostructureCritic,
    ])
    def test_safe_evaluate_on_empty(self, CriticCls):
        critic = CriticCls()
        result = critic.safe_evaluate("", MINIMAL_META)
        assert isinstance(result, CriticResult)
        assert 0 <= result.score <= 10
