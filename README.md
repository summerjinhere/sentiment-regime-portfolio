# AI Finance v2 - 4-Regime Framework

## Duke Feedback Incorporated

This research framework implements the Weakness Compensation Framework with enhancements based on Duke finance professor feedback.

## Key Improvements (v2)

1. **4-Regime Framework**: Vol × Market Direction classification
   - Low Vol + Bear (LB): Quiet decline
   - Low Vol + Bull (LU): Quiet rally  
   - High Vol + Bear (HB): Crisis/Panic
   - High Vol + Bull (HU): Recovery/Euphoria

2. **Train/Test Split**: 2005-2014 design, 2015-2025 evaluation

3. **Realistic Transaction Costs**: NASDAQ-calibrated (8bps default)

4. **FF5 + Momentum**: Factor adjustment beyond CAPM

5. **Multiple Testing Adjustment**: Benjamini-Hochberg FDR

6. **8-Configuration Ablation Study**

## Files

| File | Description |
|------|-------------|
| `enhanced_integrated_model_v3.py` | Core 3-layer model with 4-regime support |
| `research_pipeline_v2.py` | Fama-MacBeth, Ablation, Factor validation |
| `enhanced_dashboard_v4.py` | Streamlit dashboard |
| `fetch_real_data.py` | Yahoo Finance data collection |
| `multi_source_news_scraper.py` | FinBERT sentiment data |

## Quick Start

```bash
# 1. Install dependencies
pip install pandas numpy scipy statsmodels yfinance streamlit plotly transformers torch

# 2. Fetch price data (20 years)
python fetch_real_data.py --start 2005-01-01

# 3. Collect sentiment data
python multi_source_news_scraper.py --trading-days 5040

# 4. Run dashboard
streamlit run enhanced_dashboard_v4.py
```

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Weakness Compensation Framework                 │
├─────────────────────────────────────────────────────────────────┤
│ Layer 1: Dynamic Factor Loading (Kalman Filter)                 │
│          β_t = β_{t-1} + w_t (time-varying beta)               │
├─────────────────────────────────────────────────────────────────┤
│ Layer 2: Sentiment Integration (FinBERT)                        │
│          4-Regime Interaction: sent_LB, sent_LU, sent_HB, sent_HU│
├─────────────────────────────────────────────────────────────────┤
│ Layer 3: GARCH-MIDAS Volatility                                 │
│          σ²_t = τ_t × g_t (long + short term)                  │
└─────────────────────────────────────────────────────────────────┘
```

## 4-Regime Framework

```
                    Market Direction
                    Bear (<0)    Bull (≥0)
                  ┌────────────┬────────────┐
Volatility  Low   │   LB       │    LU      │
(z < 0)           │ Low+Bear   │  Low+Bull  │
                  ├────────────┼────────────┤
            High  │   HB       │    HU      │
(z ≥ 0)           │ High+Bear  │ High+Bull  │
                  └────────────┴────────────┘
```

## Research Pipeline Usage

```python
from research_pipeline_v2 import run_research_v2, ModelConfig, ResearchDesignV2

config = ModelConfig(transaction_cost_bps=8)
design = ResearchDesignV2(
    design_end="2014-12-31",
    eval_start="2015-01-01",
    use_4regime=True,
    vol_threshold=0.0
)

results = run_research_v2(
    prices_file="real_backtest_panel.csv",
    sentiment_file="real_sentiment_data.csv",
    benchmark="SPY",
    model_config=config,
    design=design
)

# Access results
print(results['fama_macbeth_eval'])
print(results['ablation'])
print(results['validation_ff5mom'])
```

## Statistical Validation

- **Fama-MacBeth Regressions**: Cross-sectional tests with 4-regime interactions
- **Bootstrap Analysis**: Block bootstrap (1000 samples, 22-day blocks)
- **Subperiod Robustness**: 4-period analysis
- **Transaction Cost Sensitivity**: 0-50 bps range
- **Multiple Testing**: Benjamini-Hochberg FDR (α=0.10)

## Citation

If you use this framework, please cite:
```
AI Finance Weakness Compensation Framework v2
4-Regime Sentiment Integration Model
2025
```
