import json

import pytest

from src import config
from src.classifier import append_bucket, classify
from src.schemas import BucketEntry, CriticResult, PaperMeta


META = PaperMeta(
    arxiv_id="2401.12345",
    title="Test Paper",
    authors=["Author"],
    abstract="Abstract",
    year=2024,
    categories=["q-fin.TR"],
)


def _make_results(stat=7, meth=7, ml=7, micro=7, flags=None) -> dict[str, CriticResult]:
    flags = flags or {}
    return {
        "statistical": CriticResult("statistical", stat, flags.get("statistical", []), "", {}),
        "methodology": CriticResult("methodology", meth, flags.get("methodology", []), "", {}),
        "ml": CriticResult("ml", ml, flags.get("ml", []), "", {}),
        "microstructure": CriticResult("microstructure", micro, flags.get("microstructure", []), "", {}),
    }


class TestClassify:
    def test_bucket_a(self):
        # composite = 0.3*8 + 0.3*8 + 0.2*8 + 0.2*8 = 8.0, no critical flags
        entry = classify("2401.12345", META, _make_results(8, 8, 8, 8))
        assert entry.bucket == "A"
        assert entry.composite >= 7.5

    def test_bucket_b(self):
        # composite = 0.3*6 + 0.3*6 + 0.2*6 + 0.2*6 = 6.0
        entry = classify("2401.12345", META, _make_results(6, 6, 6, 6))
        assert entry.bucket == "B"

    def test_bucket_c_low_composite(self):
        entry = classify("2401.12345", META, _make_results(2, 2, 2, 2))
        assert entry.bucket == "C"

    def test_bucket_c_early_exit(self):
        entry = classify("2401.12345", META, _make_results(3, 8, 8, 8), early_exit=True)
        assert entry.bucket == "C"
        assert "auto-rejected" in entry.verdict.lower()

    def test_bucket_c_many_critical_flags(self):
        flags = {
            "statistical": ["p_hacking"],
            "methodology": ["data_snooping_bias"],
            "ml": ["lookahead_bias", "leaky_features"],
        }
        entry = classify("2401.12345", META, _make_results(6, 6, 6, 6, flags))
        assert entry.bucket == "C"

    def test_bucket_d(self):
        # composite around 4-5, with 2 critical flags -> D
        flags = {"ml": ["lookahead_bias", "leaky_features"]}
        entry = classify("2401.12345", META, _make_results(5, 5, 4, 4, flags))
        assert entry.bucket == "D"

    def test_composite_calculation(self):
        # stat=10, meth=10, ml=0, micro=0
        # composite = 0.3*10 + 0.3*10 + 0.2*0 + 0.2*0 = 6.0
        entry = classify("2401.12345", META, _make_results(10, 10, 0, 0))
        assert entry.composite == 6.0

    def test_critical_flag_demotion(self):
        # High scores but with a critical flag → drops from A to B
        flags = {"ml": ["lookahead_bias"]}
        entry = classify("2401.12345", META, _make_results(8, 8, 8, 8, flags))
        assert entry.bucket == "B"  # demoted from A due to critical flag

    def test_entry_has_all_fields(self):
        entry = classify("2401.12345", META, _make_results(7, 7, 7, 7))
        assert entry.arxiv_id == "2401.12345"
        assert entry.title == "Test Paper"
        assert entry.bucket in ("A", "B", "C", "D")
        assert "statistical" in entry.scores
        assert isinstance(entry.composite, float)
        assert isinstance(entry.flags, list)
        assert isinstance(entry.verdict, str)
        assert entry.screened_at  # non-empty


class TestAppendBucket:
    def test_append_creates_file(self, tmp_workspace):
        entry = classify("2401.12345", META, _make_results(7, 7, 7, 7))
        append_bucket(entry)
        assert config.BUCKET_FILE.exists()
        line = config.BUCKET_FILE.read_text().strip()
        data = json.loads(line)
        assert data["arxiv_id"] == "2401.12345"

    def test_append_multiple(self, tmp_workspace):
        entry1 = classify("2401.11111", META, _make_results(8, 8, 8, 8))
        entry2 = classify("2401.22222", META, _make_results(3, 3, 3, 3))
        append_bucket(entry1)
        append_bucket(entry2)
        lines = config.BUCKET_FILE.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["arxiv_id"] == "2401.11111"
        assert json.loads(lines[1])["arxiv_id"] == "2401.22222"
