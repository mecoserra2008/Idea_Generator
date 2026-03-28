from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PaperMeta:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    year: int
    categories: list[str]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PaperMeta:
        return cls(**d)


@dataclass
class CriticResult:
    critic_name: str
    score: int
    flags: list[str]
    summary: str
    sub_scores: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CriticResult:
        return cls(**d)


@dataclass
class BucketEntry:
    arxiv_id: str
    title: str
    bucket: Literal["A", "B", "C", "D"]
    scores: dict[str, int]
    composite: float
    flags: list[str]
    verdict: str
    screened_at: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BucketEntry:
        return cls(**d)


@dataclass
class DiscoveryReport:
    queries_used: list[str]
    categories: list[str]
    days_back: int
    total_search_results: int
    duplicates_skipped: int
    total_screened: int
    failed: int
    by_bucket: dict[str, int]
    entries: list[BucketEntry]
    run_at: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class FetchError(Exception):
    pass


class ClassificationError(Exception):
    pass
