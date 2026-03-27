import json
import pytest
from pathlib import Path
from src import config
from src.schemas import PaperMeta, CriticResult


SAMPLE_PAPER_TEXT = """
Title: Deep Reinforcement Learning for Optimal Trade Execution

Abstract:
We propose a deep reinforcement learning framework for optimal trade execution
in equity markets. Our model uses LSTM neural networks to learn execution
strategies that minimize transaction costs and market impact.

1. Introduction
Optimal trade execution is a critical problem in quantitative finance. We develop
a machine learning approach using temporal difference learning and neural
network function approximators. Our method achieves statistically significant
improvements over TWAP and VWAP benchmarks across multiple market regimes.

2. Methodology
We use a walk-forward validation approach with expanding windows to avoid
look-ahead bias. The training period spans 2005-2015 and the out-of-sample
test period covers 2016-2020, including both bull and bear market conditions
and the 2020 financial crisis.

We apply Bonferroni correction for multiple hypothesis testing across 50
strategy variants. The sample size includes N = 500,000 trading days across
100 stocks from the S&P 500.

3. Model Architecture
Our LSTM model uses hyperparameter tuning via Bayesian optimization with
purged cross-validation to prevent information leakage. Features are selected
based on financial theory and domain knowledge, including order book imbalance,
volume-weighted price momentum, and fundamental factors.

Feature importance is assessed using SHAP values for model interpretability.

4. Results
The strategy achieves an annualized Sharpe ratio of 2.1 (p-value < 0.01,
95% confidence interval [1.8, 2.4]). Effect sizes are reported using
Cohen's d = 0.45. We test for stationarity using augmented Dickey-Fuller
tests and find cointegration relationships.

Robustness checks include bootstrap analysis, Monte Carlo simulation,
and sensitivity analysis across transaction cost assumptions (5-20 basis
points). The strategy is robust to parameter perturbations.

5. Trading Costs and Implementation
We model realistic transaction costs including bid-ask spread (average 5bps),
market impact using the Almgren-Chriss model, and commission costs.
Slippage is estimated at 2-5 bps per trade based on historical fill data.

Strategy capacity is estimated at $50M AUM before alpha decay becomes
significant. We restrict trading to liquid large-cap stocks with average
daily volume exceeding $10M.

6. Conclusion
Source code is available at github.com/example/trade-execution.
"""

SAMPLE_META = PaperMeta(
    arxiv_id="2401.12345",
    title="Deep Reinforcement Learning for Optimal Trade Execution",
    authors=["Alice Smith", "Bob Jones"],
    abstract="We propose a deep reinforcement learning framework for optimal trade execution.",
    year=2024,
    categories=["q-fin.TR", "cs.LG"],
)


@pytest.fixture
def sample_text():
    return SAMPLE_PAPER_TEXT


@pytest.fixture
def sample_meta():
    return SAMPLE_META


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace and patch config paths."""
    papers_dir = tmp_path / "papers"
    reports_dir = tmp_path / "reports"
    papers_dir.mkdir()
    reports_dir.mkdir()
    bucket_file = tmp_path / "bucket.json"

    original_papers = config.PAPERS_DIR
    original_reports = config.REPORTS_DIR
    original_bucket = config.BUCKET_FILE

    config.PAPERS_DIR = papers_dir
    config.REPORTS_DIR = reports_dir
    config.BUCKET_FILE = bucket_file

    yield tmp_path

    config.PAPERS_DIR = original_papers
    config.REPORTS_DIR = original_reports
    config.BUCKET_FILE = original_bucket


@pytest.fixture
def populated_workspace(tmp_workspace, sample_text, sample_meta):
    """Workspace with paper files already written."""
    papers_dir = config.PAPERS_DIR
    arxiv_id = sample_meta.arxiv_id

    (papers_dir / f"{arxiv_id}.txt").write_text(sample_text)
    with open(papers_dir / f"{arxiv_id}_meta.json", "w") as f:
        json.dump(sample_meta.to_dict(), f)

    return tmp_workspace
