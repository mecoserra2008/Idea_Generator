import json
from unittest.mock import patch, MagicMock

import pytest

from src import config
from src.pipeline import run_screen, run_rescore, run_classify_only
from src.schemas import PaperMeta, CriticResult


@pytest.fixture
def mock_fetch(sample_meta, sample_text, populated_workspace):
    """Mock fetch_paper to avoid network calls."""
    with patch("src.pipeline.fetch_paper") as mock:
        mock.return_value = sample_meta
        yield mock


class TestRunScreen:
    def test_full_pipeline(self, mock_fetch, sample_meta, sample_text):
        entry = run_screen("2401.12345")
        assert entry.arxiv_id == sample_meta.arxiv_id
        assert entry.bucket in ("A", "B", "C", "D")
        assert config.BUCKET_FILE.exists()

        # Check JSONL output
        line = config.BUCKET_FILE.read_text().strip()
        data = json.loads(line)
        assert data["arxiv_id"] == sample_meta.arxiv_id

        # Check reports were written
        for critic_name in ("statistical", "methodology", "ml", "microstructure"):
            report_path = config.REPORTS_DIR / f"{sample_meta.arxiv_id}_{critic_name}.json"
            assert report_path.exists()

    def test_early_exit_produces_bucket_c(self, populated_workspace, sample_meta):
        """Paper with very weak statistical content triggers early exit."""
        weak_text = "We trade stocks. Profit is good."
        (config.PAPERS_DIR / f"{sample_meta.arxiv_id}.txt").write_text(weak_text)

        with patch("src.pipeline.fetch_paper") as mock:
            mock.return_value = sample_meta
            entry = run_screen(sample_meta.arxiv_id)

        assert entry.bucket == "C"


class TestRunRescore:
    def test_rescore(self, populated_workspace, sample_meta):
        entry = run_rescore(sample_meta.arxiv_id)
        assert entry.arxiv_id == sample_meta.arxiv_id
        assert entry.bucket in ("A", "B", "C", "D")


class TestRunClassifyOnly:
    def test_classify_from_reports(self, populated_workspace, sample_meta, sample_text):
        # First generate reports by running critics
        from src.critics import CRITIC_REGISTRY
        for name, cls in CRITIC_REGISTRY.items():
            critic = cls()
            result = critic.evaluate(sample_text, sample_meta)
            critic.save_report(sample_meta.arxiv_id, result)

        entry = run_classify_only(sample_meta.arxiv_id)
        assert entry.arxiv_id == sample_meta.arxiv_id
        assert entry.bucket in ("A", "B", "C", "D")

    def test_classify_missing_reports_raises(self, populated_workspace, sample_meta):
        with pytest.raises(FileNotFoundError):
            run_classify_only(sample_meta.arxiv_id)
