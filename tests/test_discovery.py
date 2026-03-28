import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src import config
from src.discovery import (
    _extract_id,
    batch_screen,
    filter_results,
    format_summary,
    load_known_ids,
    run_discovery,
    search_arxiv,
)
from src.schemas import BucketEntry, DiscoveryReport, FetchError


def _make_result(arxiv_id: str, title: str = "Test Paper", days_ago: int = 1,
                 categories: list[str] | None = None) -> MagicMock:
    """Create a mock arxiv.Result."""
    result = MagicMock()
    result.entry_id = f"http://arxiv.org/abs/{arxiv_id}v1"
    result.title = title
    result.summary = "Abstract text"
    result.authors = []
    result.published = datetime.now(timezone.utc) - timedelta(days=days_ago)
    result.categories = categories or ["q-fin.TR"]
    return result


def _make_bucket_entry(arxiv_id: str, bucket: str = "B", composite: float = 6.0) -> BucketEntry:
    return BucketEntry(
        arxiv_id=arxiv_id,
        title="Test",
        bucket=bucket,
        scores={"statistical": 6, "methodology": 6, "ml": 6, "microstructure": 6},
        composite=composite,
        flags=[],
        verdict="Test verdict",
        screened_at=datetime.now(timezone.utc).isoformat(),
    )


class TestExtractId:
    def test_standard_url(self):
        result = MagicMock()
        result.entry_id = "http://arxiv.org/abs/2401.12345v1"
        assert _extract_id(result) == "2401.12345"

    def test_five_digit(self):
        result = MagicMock()
        result.entry_id = "http://arxiv.org/abs/2603.14001v2"
        assert _extract_id(result) == "2603.14001"


class TestLoadKnownIds:
    def test_missing_file(self, tmp_workspace):
        assert load_known_ids() == set()

    def test_populated_file(self, tmp_workspace):
        entries = [
            {"arxiv_id": "2401.11111", "bucket": "A"},
            {"arxiv_id": "2401.22222", "bucket": "C"},
        ]
        with open(config.BUCKET_FILE, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        known = load_known_ids()
        assert known == {"2401.11111", "2401.22222"}

    def test_handles_corrupt_lines(self, tmp_workspace):
        with open(config.BUCKET_FILE, "w") as f:
            f.write('{"arxiv_id": "2401.11111"}\n')
            f.write("not json\n")
            f.write('{"no_id_field": true}\n')

        known = load_known_ids()
        assert known == {"2401.11111"}


class TestFilterResults:
    def test_removes_known(self):
        results = [_make_result("2401.11111"), _make_result("2401.22222")]
        known = {"2401.11111"}
        filtered, skipped = filter_results(results, known)
        assert len(filtered) == 1
        assert skipped == 1
        assert _extract_id(filtered[0]) == "2401.22222"

    def test_no_known(self):
        results = [_make_result("2401.11111")]
        filtered, skipped = filter_results(results, set())
        assert len(filtered) == 1
        assert skipped == 0


class TestSearchArxiv:
    @patch("src.discovery.arxiv.Client")
    @patch("src.discovery.time.sleep")
    def test_merges_and_deduplicates(self, mock_sleep, mock_client_cls):
        r1 = _make_result("2603.00001", "Paper A")
        r2 = _make_result("2603.00002", "Paper B")
        r1_dup = _make_result("2603.00001", "Paper A dup")

        mock_client = MagicMock()
        call_count = [0]
        def fake_results(search):
            call_count[0] += 1
            if call_count[0] == 1:
                return iter([r1, r2])
            else:
                return iter([r1_dup])
        mock_client.results.side_effect = fake_results
        mock_client_cls.return_value = mock_client

        results = search_arxiv(
            queries=["query1", "query2"],
            categories=["q-fin.TR"],
            max_results=10,
            days_back=30,
            rate_delay=0,
        )
        assert len(results) == 2
        ids = {_extract_id(r) for r in results}
        assert ids == {"2603.00001", "2603.00002"}

    @patch("src.discovery.arxiv.Client")
    @patch("src.discovery.time.sleep")
    def test_filters_old_papers(self, mock_sleep, mock_client_cls):
        recent = _make_result("2603.00001", days_ago=2)
        old = _make_result("2603.00002", days_ago=60)

        mock_client = MagicMock()
        mock_client.results.return_value = iter([recent, old])
        mock_client_cls.return_value = mock_client

        results = search_arxiv(
            queries=["query"],
            categories=["q-fin.TR"],
            max_results=10,
            days_back=7,
            rate_delay=0,
        )
        assert len(results) == 1
        assert _extract_id(results[0]) == "2603.00001"

    @patch("src.discovery.arxiv.Client")
    @patch("src.discovery.time.sleep")
    def test_handles_search_failure(self, mock_sleep, mock_client_cls):
        mock_client = MagicMock()
        mock_client.results.side_effect = Exception("API error")
        mock_client_cls.return_value = mock_client

        results = search_arxiv(
            queries=["failing query"],
            categories=["q-fin.TR"],
            max_results=10,
            days_back=7,
            rate_delay=0,
        )
        assert results == []


class TestBatchScreen:
    @patch("src.discovery.run_screen")
    @patch("src.discovery.time.sleep")
    def test_screens_all_papers(self, mock_sleep, mock_run_screen):
        mock_run_screen.return_value = _make_bucket_entry("2603.00001")
        results = [_make_result("2603.00001"), _make_result("2603.00002")]

        entries, failures = batch_screen(results, rate_delay=0, timeout=60)
        assert len(entries) == 2
        assert failures == 0
        assert mock_run_screen.call_count == 2

    @patch("src.discovery.run_screen")
    @patch("src.discovery.time.sleep")
    def test_handles_failure(self, mock_sleep, mock_run_screen):
        mock_run_screen.side_effect = [
            _make_bucket_entry("2603.00001"),
            FetchError("Download failed"),
            _make_bucket_entry("2603.00003"),
        ]
        results = [
            _make_result("2603.00001"),
            _make_result("2603.00002"),
            _make_result("2603.00003"),
        ]

        entries, failures = batch_screen(results, rate_delay=0, timeout=60)
        assert len(entries) == 2
        assert failures == 1


class TestFormatSummary:
    def test_contains_key_info(self):
        report = DiscoveryReport(
            queries_used=["test query"],
            categories=["q-fin.TR"],
            days_back=7,
            total_search_results=10,
            duplicates_skipped=2,
            total_screened=7,
            failed=1,
            by_bucket={"A": 1, "B": 3, "C": 2, "D": 1},
            entries=[_make_bucket_entry("2603.00001", "A", 8.0)],
            run_at="2026-03-28T12:00:00Z",
        )
        text = format_summary(report)
        assert "DISCOVERY REPORT" in text
        assert "Search hits:   10" in text
        assert "Already seen:  2" in text
        assert "Screened OK:   7" in text
        assert "A (actionable)" in text
        assert "TOP PAPERS" in text
        assert "2603.00001" in text


class TestRunDiscovery:
    @patch("src.discovery.search_arxiv")
    @patch("src.discovery.run_screen")
    @patch("src.discovery.time.sleep")
    def test_end_to_end(self, mock_sleep, mock_run_screen, mock_search, tmp_workspace):
        mock_search.return_value = [
            _make_result("2603.00001", "Paper A"),
            _make_result("2603.00002", "Paper B"),
        ]
        mock_run_screen.side_effect = [
            _make_bucket_entry("2603.00001", "A", 8.0),
            _make_bucket_entry("2603.00002", "C", 2.5),
        ]

        report = run_discovery(
            queries=["test"],
            categories=["q-fin.TR"],
            max_results=10,
            days_back=7,
            rate_delay=0,
            timeout=60,
        )

        assert report.total_search_results == 2
        assert report.total_screened == 2
        assert report.failed == 0
        assert report.by_bucket["A"] == 1
        assert report.by_bucket["C"] == 1

    @patch("src.discovery.search_arxiv")
    @patch("src.discovery.run_screen")
    def test_dry_run_skips_screening(self, mock_run_screen, mock_search, tmp_workspace):
        mock_search.return_value = [_make_result("2603.00001")]

        report = run_discovery(
            queries=["test"],
            categories=["q-fin.TR"],
            max_results=10,
            days_back=7,
            rate_delay=0,
            timeout=60,
            dry_run=True,
        )

        mock_run_screen.assert_not_called()
        assert report.total_screened == 0
        assert report.entries == []

    @patch("src.discovery.search_arxiv")
    @patch("src.discovery.run_screen")
    @patch("src.discovery.time.sleep")
    def test_skips_already_screened(self, mock_sleep, mock_run_screen, mock_search, tmp_workspace):
        # Write a known ID to bucket.json
        with open(config.BUCKET_FILE, "w") as f:
            f.write(json.dumps({"arxiv_id": "2603.00001"}) + "\n")

        mock_search.return_value = [
            _make_result("2603.00001"),  # already screened
            _make_result("2603.00002"),  # new
        ]
        mock_run_screen.return_value = _make_bucket_entry("2603.00002")

        report = run_discovery(
            queries=["test"],
            categories=["q-fin.TR"],
            max_results=10,
            days_back=7,
            rate_delay=0,
            timeout=60,
        )

        assert report.duplicates_skipped == 1
        assert report.total_screened == 1
        assert mock_run_screen.call_count == 1
