from src.gate import check_early_exit
from src.schemas import CriticResult


def _make_results(stat_score: int) -> dict[str, CriticResult]:
    return {
        "statistical": CriticResult(critic_name="statistical", score=stat_score, flags=[], summary="", sub_scores={}),
        "methodology": CriticResult(critic_name="methodology", score=7, flags=[], summary="", sub_scores={}),
        "ml": CriticResult(critic_name="ml", score=7, flags=[], summary="", sub_scores={}),
        "microstructure": CriticResult(critic_name="microstructure", score=7, flags=[], summary="", sub_scores={}),
    }


class TestEarlyExit:
    def test_score_below_threshold_triggers(self):
        assert check_early_exit(_make_results(3)) is True

    def test_score_at_threshold_does_not_trigger(self):
        assert check_early_exit(_make_results(4)) is False

    def test_score_above_threshold_does_not_trigger(self):
        assert check_early_exit(_make_results(10)) is False

    def test_score_zero_triggers(self):
        assert check_early_exit(_make_results(0)) is True

    def test_missing_statistical_does_not_trigger(self):
        results = {
            "methodology": CriticResult(critic_name="methodology", score=7, flags=[], summary="", sub_scores={}),
        }
        assert check_early_exit(results) is False
