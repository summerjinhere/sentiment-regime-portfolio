# AI Finance Project v2
## Weakness Compensation Framework - Master's Thesis

### Key Features
- **Real Data Only** - No synthetic/sample data
- **Multi-Source News Scraper** - Finviz → Yahoo → Google RSS → MarketWatch fallback chain
- **Configurable Trading Days** - 1 year, 2 years, 5 years, or custom
- **Complete Statistical Validation** - HAC alpha/beta, bootstrap CI, subperiod tests
- **FinBERT Sentiment** - Real NLP analysis

### Files
```
fetch_real_data.py           # Yahoo Finance data fetcher
multi_source_news_scraper.py # Multi-source news with fallback
enhanced_integrated_model_v2.py  # Complete model with validation
enhanced_dashboard_v3.py     # Streamlit dashboard
requirements.txt             # Dependencies
```

### Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch market data (2 years)
python fetch_real_data.py --trading-days 504

# 3. Collect sentiment data
python multi_source_news_scraper.py --trading-days 504

# 4. Run dashboard
streamlit run enhanced_dashboard_v3.py
```

### Trading Days Options
```bash
# 1 year
python fetch_real_data.py --trading-days 252

# 2 years (default)
python fetch_real_data.py --trading-days 504

# 3 years
python fetch_real_data.py --trading-days 756

# 5 years
python fetch_real_data.py --trading-days 1260
```

### News Sources (Fallback Chain)
1. **Finviz** - Primary source for stock-specific news
2. **Yahoo Finance** - Backup for major tickers
3. **Google News RSS** - General financial news
4. **MarketWatch** - Additional coverage
5. **Seeking Alpha** - Analysis articles
6. **NewsAPI** - Optional (requires free API key)

If one source fails, the next is tried automatically.

### Model Architecture
```
Layer 1: Dynamic Factor Loading
         └─ Kalman Filter (Time-varying Beta)
         
Layer 2: Sentiment Integration  
         └─ FinBERT (ProsusAI/finbert)
         
Layer 3: GARCH-MIDAS Volatility
         └─ Short-term GARCH + Long-term MIDAS
```

### Statistical Validation
- **HAC Standard Errors** - Newey-West correction for autocorrelation
- **Bootstrap Analysis** - Block bootstrap for confidence intervals
- **Subperiod Tests** - Consistency across time periods
- **Transaction Cost Sensitivity** - Breakeven cost analysis

### Asset Universe (94 assets)
- Technology (10): AAPL, MSFT, NVDA, etc.
- Healthcare (10): LLY, UNH, JNJ, etc.
- Financials (10): JPM, V, MA, etc.
- Consumer (10): AMZN, TSLA, HD, etc.
- Energy (10): XOM, CVX, COP, etc.
- Utilities (10): NEE, SO, DUK, etc.
- Bonds (10): TLT, IEF, SHY, etc.
- Commodities (10): GLD, SLV, USO, etc.
- Indices (4): SPY, QQQ, IWM, DIA
