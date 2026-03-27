from __future__ import annotations

from ..schemas import CriticResult, PaperMeta
from .base import BaseCritic


class MicrostructureCritic(BaseCritic):
    name = "microstructure"

    TRANSACTION_COST_PATTERNS = [
        r"transaction\s+cost", r"trading\s+cost", r"commission",
        r"brokerage\s+fee", r"bid[\s\-]?ask\s+spread", r"spread\s+cost",
        r"round[\s\-]?trip\s+cost", r"basis\s+points?\s+(cost|fee)",
    ]
    SLIPPAGE_PATTERNS = [
        r"slippage", r"execution\s+(price|quality|cost)",
        r"market\s+impact", r"price\s+impact",
        r"fill\s+(price|rate|assumption)", r"limit\s+order",
        r"market\s+order", r"vwap", r"twap",
    ]
    CAPACITY_PATTERNS = [
        r"capacity", r"scalab", r"assets?\s+under\s+management",
        r"aum\b", r"strategy\s+size", r"capital\s+constraint",
        r"diminishing\s+return", r"alpha\s+decay",
    ]
    LIQUIDITY_PATTERNS = [
        r"liquidit", r"trading\s+volume", r"average\s+daily\s+volume",
        r"adv\b", r"market\s+depth", r"order\s+book",
        r"illiquid", r"thinly\s+traded", r"small[\s\-]?cap",
    ]
    LATENCY_PATTERNS = [
        r"latency", r"high[\s\-]?frequency", r"hft\b",
        r"microsecond", r"millisecond", r"co[\s\-]?location",
        r"execution\s+speed", r"order[\s\-]?to[\s\-]?fill",
    ]

    def evaluate(self, paper_text: str, meta: PaperMeta) -> CriticResult:
        flags = []
        text = paper_text

        tc_count = self._count_matches(text, self.TRANSACTION_COST_PATTERNS)
        tc_score = self._clamp(min(tc_count * 2, 10))
        if tc_count == 0:
            flags.append("no_transaction_costs")

        slip_count = self._count_matches(text, self.SLIPPAGE_PATTERNS)
        slip_score = self._clamp(min(slip_count * 2, 10))
        if slip_count == 0:
            flags.append("unrealistic_fills")

        cap_count = self._count_matches(text, self.CAPACITY_PATTERNS)
        cap_score = self._clamp(min(cap_count * 2, 10))
        if cap_count == 0:
            flags.append("capacity_unknown")

        liq_count = self._count_matches(text, self.LIQUIDITY_PATTERNS)
        liq_score = self._clamp(min(liq_count * 2, 10))
        if liq_count == 0:
            flags.append("illiquid_instruments")

        lat_count = self._count_matches(text, self.LATENCY_PATTERNS)
        # Only score latency if paper discusses HFT
        is_hft = self._has_any(text, [r"high[\s\-]?frequency", r"hft\b", r"microsecond"])
        if is_hft:
            lat_score = self._clamp(min(lat_count * 2, 10))
        else:
            lat_score = 5  # neutral for non-HFT papers

        if self._has_any(text, [r"illiquid", r"thinly\s+traded", r"penny\s+stock"]):
            if not self._has_any(text, [r"market\s+impact", r"slippage"]):
                flags.append("ignores_market_impact")

        sub_scores = {
            "transaction_costs": tc_score,
            "slippage": slip_score,
            "capacity": cap_score,
            "liquidity": liq_score,
            "latency": lat_score,
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
            return "Thorough consideration of real-world trading constraints and market microstructure."
        elif score >= 5:
            issues = ", ".join(flags) if flags else "some microstructure gaps"
            return f"Partial microstructure awareness; missing: {issues}."
        else:
            issues = ", ".join(flags) if flags else "ignores real-world trading constraints"
            return f"Poor microstructure realism: {issues}."
