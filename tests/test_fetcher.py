import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src import config
from src.fetcher import _normalize_id, fetch_paper
from src.schemas import FetchError


class TestNormalizeId:
    def test_bare_id(self):
        assert _normalize_id("2401.12345") == "2401.12345"

    def test_url(self):
        assert _normalize_id("https://arxiv.org/abs/2401.12345") == "2401.12345"

    def test_url_with_version(self):
        assert _normalize_id("https://arxiv.org/abs/2401.12345v2") == "2401.12345"

    def test_invalid(self):
        with pytest.raises(FetchError):
            _normalize_id("not-an-id")


class TestFetchPaper:
    def test_idempotent_skip(self, tmp_workspace, sample_meta):
        """If files exist and force=False, return cached meta."""
        arxiv_id = "2401.12345"
        text_path = config.PAPERS_DIR / f"{arxiv_id}.txt"
        meta_path = config.PAPERS_DIR / f"{arxiv_id}_meta.json"

        text_path.write_text("cached text")
        with open(meta_path, "w") as f:
            json.dump(sample_meta.to_dict(), f)

        result = fetch_paper(arxiv_id, force=False)
        assert result.arxiv_id == arxiv_id
        assert result.title == sample_meta.title

    @patch("src.fetcher.arxiv.Client")
    def test_fetch_no_results(self, mock_client_cls, tmp_workspace):
        mock_client = MagicMock()
        mock_client.results.return_value = iter([])
        mock_client_cls.return_value = mock_client

        with pytest.raises(FetchError, match="No paper found"):
            fetch_paper("2401.99999", force=True)

    @patch("src.fetcher._extract_text_from_pdf")
    @patch("src.fetcher.arxiv.Client")
    def test_fetch_success(self, mock_client_cls, mock_extract, tmp_workspace):
        mock_result = MagicMock()
        mock_result.title = "Test Paper"
        author_mock = MagicMock()
        author_mock.name = "Author A"
        mock_result.authors = [author_mock]
        mock_result.summary = "Test abstract"
        mock_result.published = MagicMock(year=2024)
        mock_result.categories = ["q-fin.TR"]
        mock_result.download_pdf.return_value = "/tmp/fake.pdf"

        mock_client = MagicMock()
        mock_client.results.return_value = iter([mock_result])
        mock_client_cls.return_value = mock_client

        mock_extract.return_value = "Extracted PDF text here"

        meta = fetch_paper("2401.00001", force=True)
        assert meta.arxiv_id == "2401.00001"
        assert meta.title == "Test Paper"

        text_path = config.PAPERS_DIR / "2401.00001.txt"
        assert text_path.exists()
        assert "Extracted PDF text" in text_path.read_text()

    @patch("src.fetcher.arxiv.Client")
    def test_fetch_pdf_fallback_to_abstract(self, mock_client_cls, tmp_workspace):
        """When PDF extraction fails, fall back to abstract."""
        mock_result = MagicMock()
        mock_result.title = "Fallback Paper"
        mock_result.authors = []
        mock_result.summary = "This is the abstract"
        mock_result.published = MagicMock(year=2024)
        mock_result.categories = ["q-fin.TR"]
        mock_result.download_pdf.side_effect = Exception("PDF download failed")

        mock_client = MagicMock()
        mock_client.results.return_value = iter([mock_result])
        mock_client_cls.return_value = mock_client

        meta = fetch_paper("2401.00002", force=True)
        text_path = config.PAPERS_DIR / "2401.00002.txt"
        assert "abstract" in text_path.read_text().lower()
