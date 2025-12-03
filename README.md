# AI Finance Project - Weakness Compensation Framework

## Master's Thesis Implementation

Generated: 2025-12-03 02:36:05

## Files

1. **fetch_real_data.py** - Downloads real market data from Yahoo Finance
2. **real_news_scraper_bs4.py** - Scrapes Finviz news with FinBERT sentiment
3. **enhanced_integrated_model.py** - Weakness Compensation Framework model
4. **enhanced_dashboard_v3.py** - Streamlit visualization dashboard
5. **requirements.txt** - Python dependencies

## Usage

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Fetch Market Data
```bash
python fetch_real_data.py
```
This creates: `real_backtest_panel.csv`

### Step 3: Scrape News & Sentiment
```bash
python real_news_scraper_bs4.py
```
This creates: `real_sentiment_data.csv`

### Step 4: Run Dashboard
```bash
streamlit run enhanced_dashboard_v3.py
```

## Asset Universe

- **Technology (10):** AAPL, MSFT, NVDA, AVGO, ORCL, ADBE, CRM, AMD, QCOM, INTC
- **Healthcare (10):** LLY, UNH, JNJ, MRK, ABBV, TMO, ABT, PFE, AMGN, DHR
- **Financials (10):** JPM, V, MA, BAC, WFC, MS, GS, BLK, C, AXP
- **Consumer (10):** AMZN, TSLA, HD, PG, COST, WMT, KO, PEP, MCD, NKE
- **Energy (10):** XOM, CVX, COP, SLB, EOG, MPC, PSX, VLO, OXY, HAL
- **Utilities (10):** NEE, SO, DUK, CEG, AEP, SRE, PEG, ED, EXC, XEL
- **Bonds (10):** TLT, IEF, SHY, LQD, HYG, BND, AGG, MUB, TIP, BNDX
- **Commodities (10):** GLD, SLV, USO, UNG, DBC, DBA, GSG, PALL, PPLT, CPER
- **Indices (4):** SPY, QQQ, IWM, DIA

Total: 94 assets

## Model Architecture

### Layer 1: Dynamic Factor Loading (Kalman Filter)
- Time-varying beta estimation
- Adapts to regime changes

### Layer 2: Sentiment Integration (FinBERT)
- ProsusAI/finbert model
- Real news from Finviz
- Daily aggregated sentiment

### Layer 3: GARCH-MIDAS Volatility
- Short-term: GARCH(1,1) component
- Long-term: MIDAS (22-day) component
- Volatility forecasting

## Important Notes

- Uses **ONLY REAL DATA** - No synthetic data
- Requires internet connection for Yahoo Finance and Finviz
- FinBERT requires ~500MB download on first run
- Respects rate limits (1.5-3.0 second delays)

## License

For academic use - Master's Thesis in AI Finance
