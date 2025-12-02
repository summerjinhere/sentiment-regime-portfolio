"""
Enhanced Multi-Agent Portfolio Dashboard v2 - Complete
=======================================================

Real Data Integration with:
- Yahoo Finance data for actual assets
- Clear asset universe display
- Robustness Tests (Subperiod, Bootstrap, Transaction Cost)
- Benchmark Selection
- Asset Allocation with Justification
- Full Statistical Analysis

Author: Multi-Agent Portfolio System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import dependencies
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    st.error("yfinance not installed. Run: pip install yfinance")

try:
    import statsmodels.api as sm
    from scipy import stats
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False

# =============================================================================
# ASSET UNIVERSE DEFINITION
# =============================================================================

ASSET_UNIVERSE = {
    'Technology': {
        'tickers': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'AMD', 'QCOM', 'INTC'],
        'description': 'Large-cap technology companies',
        'bull_weight': 0.25,
        'transition_weight': 0.18,
        'bear_weight': 0.10
    },
    'Healthcare': {
        'tickers': ['LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'TMO', 'ABT', 'PFE', 'AMGN', 'DHR'],
        'description': 'Healthcare and pharmaceuticals',
        'bull_weight': 0.15,
        'transition_weight': 0.15,
        'bear_weight': 0.18
    },
    'Financials': {
        'tickers': ['JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'BLK', 'C', 'AXP'],
        'description': 'Banks and financial services',
        'bull_weight': 0.15,
        'transition_weight': 0.12,
        'bear_weight': 0.08
    },
    'Consumer': {
        'tickers': ['AMZN', 'TSLA', 'HD', 'PG', 'COST', 'WMT', 'KO', 'PEP', 'MCD', 'NKE'],
        'description': 'Consumer discretionary and staples',
        'bull_weight': 0.20,
        'transition_weight': 0.15,
        'bear_weight': 0.10
    },
    'Energy': {
        'tickers': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL'],
        'description': 'Oil, gas, and energy services',
        'bull_weight': 0.10,
        'transition_weight': 0.08,
        'bear_weight': 0.05
    },
    'Utilities': {
        'tickers': ['NEE', 'SO', 'DUK', 'CEG', 'AEP', 'SRE', 'PEG', 'ED', 'EXC', 'XEL'],
        'description': 'Electric utilities and regulated power',
        'bull_weight': 0.05,
        'transition_weight': 0.12,
        'bear_weight': 0.20
    },
    'Bonds': {
        'tickers': ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'BND', 'AGG', 'MUB', 'TIP', 'BNDX'],
        'description': 'Bond ETFs for fixed income exposure',
        'bull_weight': 0.05,
        'transition_weight': 0.10,
        'bear_weight': 0.15
    },
    'Commodities': {
        'tickers': ['GLD', 'SLV', 'USO', 'UNG', 'DBC', 'DBA', 'GSG', 'PALL', 'PPLT', 'CPER'],
        'description': 'Commodity ETFs for diversification',
        'bull_weight': 0.05,
        'transition_weight': 0.10,
        'bear_weight': 0.14
    }
}

BENCHMARK_OPTIONS = {
    "S&P 500 (SPY)": "SPY",
    "Nasdaq 100 (QQQ)": "QQQ",
    "Russell 2000 (IWM)": "IWM",
    "Total Market (VTI)": "VTI",
    "60/40 Portfolio (VBINX)": "VBINX",
    "Dow Jones (DIA)": "DIA"
}

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Multi-Agent Portfolio Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem;
    }
    .success-box { background-color: #d4edda; border-left: 5px solid #00C853; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .warning-box { background-color: #fff3cd; border-left: 5px solid #FFB300; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .danger-box { background-color: #f8d7da; border-left: 5px solid #FF5252; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .info-box { background-color: #e7f3ff; border-left: 5px solid #2196F3; padding: 15px; border-radius: 5px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_data(start_date: str, end_date: str) -> Tuple[pd.DataFrame, Dict]:
    """Fetch all asset data - REAL data only (no simulation)"""
    # 1) 전체 티커 수집
    all_tickers = sorted({
        ticker
        for sector, info in ASSET_UNIVERSE.items()
        for ticker in info["tickers"]
    })

    # 2) yfinance 필수
    if not YF_AVAILABLE:
        raise RuntimeError("yfinance가 설치되어 있지 않아 실제 데이터를 불러올 수 없습니다.")

    # 3) 실제 가격 다운로드
    data = yf.download(
        all_tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        timeout=30,
    )

    if data.empty or len(data) <= 100:
        raise RuntimeError("Yahoo Finance에서 충분한 가격 데이터를 가져오지 못했습니다.")

    # 4) MultiIndex / 단일 index 모두 처리
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    elif "Close" in data.columns:
        # 단일 ticker인 경우
        prices = data[["Close"]]
        prices.columns = all_tickers[:1]
    else:
        prices = data

    prices = prices.dropna(axis=1, how="all")

    # 5) ticker_info 구성
    ticker_info = {}
    for ticker in prices.columns:
        sector = next(
            (s for s, i in ASSET_UNIVERSE.items() if ticker in i["tickers"]),
            "Unknown"
        )
        ticker_info[ticker] = {"name": ticker, "sector": sector}

    return prices, ticker_info



def generate_realistic_prices(start_date: str, end_date: str, tickers: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """Generate realistic price data for all tickers"""
    np.random.seed(42)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)
    
    # Sector characteristics (annual return, annual vol)
    sector_params = {
        'Technology': (0.15, 0.25),
        'Healthcare': (0.10, 0.18),
        'Financials': (0.08, 0.22),
        'Consumer': (0.12, 0.20),
        'Energy': (0.05, 0.30),
        'Utilities': (0.06, 0.12),
        'Bonds': (0.03, 0.08),
        'Commodities': (0.04, 0.18)
    }
    
    # Generate correlated returns
    prices_dict = {}
    ticker_info = {}
    
    # Market factor
    market_returns = np.random.normal(0.0004, 0.012, n_days)
    
    for ticker in tickers:
        # Find sector
        sector = 'Unknown'
        for s, info in ASSET_UNIVERSE.items():
            if ticker in info['tickers']:
                sector = s
                break
        
        if sector == 'Unknown':
            continue
        
        ann_ret, ann_vol = sector_params.get(sector, (0.08, 0.20))
        daily_ret = ann_ret / 252
        daily_vol = ann_vol / np.sqrt(252)
        
        # Correlated with market + idiosyncratic
        beta = np.random.uniform(0.5, 1.5) if sector != 'Utilities' else np.random.uniform(0.2, 0.6)
        idio_vol = daily_vol * 0.6
        
        returns = daily_ret + beta * market_returns + np.random.normal(0, idio_vol, n_days)
        
        # Convert to prices
        prices = 100 * np.cumprod(1 + returns)
        prices_dict[ticker] = prices
        
        ticker_info[ticker] = {
            'name': ticker,
            'sector': sector,
            'beta': beta
        }
    
    prices_df = pd.DataFrame(prices_dict, index=dates)
    
    return prices_df, ticker_info


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_benchmark(ticker: str, start_date: str, end_date: str) -> pd.Series:
    """Fetch benchmark data - REAL data only"""
    if not YF_AVAILABLE:
        raise RuntimeError("yfinance가 없어 벤치마크 실제 데이터를 불러올 수 없습니다.")

    data = yf.download(ticker, start=start_date, end=end_date,
                       progress=False, timeout=30)

    if data.empty or len(data) <= 100:
        raise RuntimeError(f"{ticker} 벤치마크 데이터를 충분히 가져오지 못했습니다.")

    if isinstance(data.columns, pd.MultiIndex):
        series = data["Close"][ticker]
    elif "Close" in data.columns:
        series = data["Close"]
    else:
        series = data.iloc[:, 0]

    return series.dropna()



def run_hmm_regime_detection(returns: pd.DataFrame, n_regimes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Run HMM for regime detection with robust error handling"""
    try:
        from hmmlearn.hmm import GaussianHMM
        
        # Use market returns (mean of all assets)
        market_returns = returns.mean(axis=1).values
        volatility = returns.std(axis=1).rolling(10, min_periods=1).mean().values
        
        # Clean data
        market_returns = np.nan_to_num(market_returns, nan=0, posinf=0, neginf=0)
        volatility = np.nan_to_num(volatility, nan=volatility[~np.isnan(volatility)].mean() if any(~np.isnan(volatility)) else 0.01)
        
        # Standardize features
        mr_std = (market_returns - np.mean(market_returns)) / (np.std(market_returns) + 1e-8)
        vol_std = (volatility - np.mean(volatility)) / (np.std(volatility) + 1e-8)
        
        X = np.column_stack([mr_std, vol_std])
        
        # Try fitting with diagonal covariance (more stable)
        model = GaussianHMM(
            n_components=n_regimes, 
            covariance_type="diag",  # More stable than "full"
            n_iter=500, 
            random_state=42,
            init_params="stmc",
            params="stmc"
        )
        model.fit(X)
        
        regimes = model.predict(X)
        regime_probs = model.predict_proba(X)
        
        # Reorder by volatility (0=low vol/bull, 2=high vol/bear)
        state_vols = []
        for i in range(n_regimes):
            mask = regimes == i
            if mask.any():
                state_vols.append((i, np.std(market_returns[mask])))
            else:
                state_vols.append((i, i * 0.01))  # Fallback ordering
        
        sorted_states = sorted(state_vols, key=lambda x: x[1])
        mapping = {old: new for new, (old, _) in enumerate(sorted_states)}
        
        regimes = np.array([mapping[s] for s in regimes])
        regime_probs_reordered = np.zeros_like(regime_probs)
        for old, new in mapping.items():
            regime_probs_reordered[:, new] = regime_probs[:, old]
        
        return regimes, regime_probs_reordered
    
    except Exception as e:
        # Fallback to rule-based regime detection
        n = len(returns)
        market_returns = returns.mean(axis=1).values
        volatility = returns.std(axis=1).rolling(20, min_periods=1).std().values
        
        # Simple rule-based regimes
        regimes = np.ones(n, dtype=int)  # Default: transition
        
        vol_median = np.nanmedian(volatility)
        ret_median = np.nanmedian(market_returns)
        
        for i in range(n):
            if volatility[i] < vol_median * 0.8 and market_returns[i] > ret_median:
                regimes[i] = 0  # Bull
            elif volatility[i] > vol_median * 1.2 or market_returns[i] < ret_median * -2:
                regimes[i] = 2  # Bear
            else:
                regimes[i] = 1  # Transition
        
        regime_probs = np.zeros((n, 3))
        for i, r in enumerate(regimes):
            regime_probs[i, r] = 0.8
            for j in range(3):
                if j != r:
                    regime_probs[i, j] = 0.1
        
        return regimes, regime_probs


def generate_portfolio_weights(
    prices: pd.DataFrame,
    regimes: np.ndarray,
    current_regime: int
) -> Tuple[Dict[str, float], Dict[str, float], pd.DataFrame]:
    """Generate portfolio weights based on regime"""
    
    returns = prices.pct_change().dropna()
    
    # Sector allocation based on regime
    regime_key = {0: 'bull_weight', 1: 'transition_weight', 2: 'bear_weight'}
    weight_key = regime_key.get(current_regime, 'transition_weight')
    
    sector_weights = {}
    for sector, info in ASSET_UNIVERSE.items():
        sector_weights[sector] = info[weight_key]
    
    # Stock selection within each sector
    detailed_weights = {}
    stock_selection_reasons = []
    
    for sector, info in ASSET_UNIVERSE.items():
        sector_weight = sector_weights[sector]
        available_tickers = [t for t in info['tickers'] if t in prices.columns]
        
        if not available_tickers:
            continue
        
        # Get recent returns for scoring
        recent_returns = returns[available_tickers].tail(60)
        
        if current_regime == 0:  # Bull - momentum
            scores = recent_returns.mean()
            selection_method = "Momentum (60-day return)"
        elif current_regime == 2:  # Bear - low volatility
            scores = -recent_returns.std()
            selection_method = "Low Volatility"
        else:  # Transition - Sharpe-like
            means = recent_returns.mean()
            stds = recent_returns.std()
            scores = means / (stds + 1e-8)
            selection_method = "Risk-Adjusted Return"
        
        scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(scores) == 0:
            continue
        
        # Select top 5 stocks
        top_stocks = scores.nlargest(min(5, len(scores))).index.tolist()
        
        # Equal weight within sector
        stock_weight = sector_weight / len(top_stocks)
        
        for stock in top_stocks:
            detailed_weights[stock] = stock_weight
            stock_selection_reasons.append({
                'Ticker': stock,
                'Sector': sector,
                'Weight': stock_weight,
                'Score': scores[stock],
                'Selection Method': selection_method
            })
    
    # Normalize
    total = sum(detailed_weights.values())
    if total > 0:
        detailed_weights = {k: v/total for k, v in detailed_weights.items()}
    
    selection_df = pd.DataFrame(stock_selection_reasons)
    
    return sector_weights, detailed_weights, selection_df


def calculate_portfolio_returns(
    prices: pd.DataFrame,
    weights_history: List[Dict[str, float]],
    rebalance_dates: List[pd.Timestamp]
) -> pd.Series:
    """Calculate portfolio returns with given weights"""
    
    returns = prices.pct_change()
    portfolio_returns = pd.Series(index=returns.index, dtype=float)
    
    current_weights = {}
    weight_idx = 0
    
    for date in returns.index:
        # Check if rebalance
        if weight_idx < len(rebalance_dates) and date >= rebalance_dates[weight_idx]:
            current_weights = weights_history[weight_idx]
            weight_idx += 1
        
        if current_weights:
            daily_return = 0
            for ticker, weight in current_weights.items():
                if ticker in returns.columns and not pd.isna(returns.loc[date, ticker]):
                    daily_return += weight * returns.loc[date, ticker]
            portfolio_returns[date] = daily_return
    
    return portfolio_returns.dropna()


def simulate_strategy_returns(
    prices: pd.DataFrame,
    regimes: np.ndarray,
    rebalance_freq: str = 'M'
) -> Tuple[pd.Series, List[Dict], List[pd.Timestamp]]:
    """Simulate strategy returns with regime-based allocation"""
    
    returns = prices.pct_change().dropna()
    dates = returns.index
    
    # Generate rebalance dates
    if rebalance_freq == 'M':
        rebalance_dates = returns.resample('M').last().index.tolist()
    elif rebalance_freq == 'W':
        rebalance_dates = returns.resample('W').last().index.tolist()
    else:
        rebalance_dates = returns.resample('Q').last().index.tolist()
    
    weights_history = []
    
    for rebal_date in rebalance_dates:
        # Get regime at rebalance date
        date_idx = dates.get_loc(rebal_date) if rebal_date in dates else 0
        if date_idx >= len(regimes):
            date_idx = len(regimes) - 1
        current_regime = regimes[date_idx]
        
        # Generate weights
        _, detailed_weights, _ = generate_portfolio_weights(
            prices.loc[:rebal_date], 
            regimes[:date_idx+1], 
            current_regime
        )
        weights_history.append(detailed_weights)
    
    # Calculate returns
    portfolio_returns = calculate_portfolio_returns(prices, weights_history, rebalance_dates)
    
    return portfolio_returns, weights_history, rebalance_dates


# =============================================================================
# METRICS FUNCTIONS
# =============================================================================

def calculate_metrics(returns: pd.Series, risk_free_rate: float = 0.02) -> Dict:
    """Calculate comprehensive performance metrics"""
    returns = returns.dropna()
    if len(returns) < 2:
        return {}
    
    freq = 252
    n_years = len(returns) / freq
    
    total_return = (1 + returns).prod() - 1
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    volatility = returns.std() * np.sqrt(freq)
    
    excess = returns - (risk_free_rate / freq)
    sharpe = excess.mean() / returns.std() * np.sqrt(freq) if returns.std() > 0 else 0
    
    downside = returns[returns < 0]
    sortino = excess.mean() / downside.std() * np.sqrt(freq) if len(downside) > 0 and downside.std() > 0 else 0
    
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()
    
    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_dd,
        'Calmar Ratio': cagr / abs(max_dd) if max_dd != 0 else 0,
        'Win Rate': (returns > 0).mean(),
        'Best Day': returns.max(),
        'Worst Day': returns.min(),
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis()
    }


def calculate_alpha_beta(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> Dict:
    """Calculate alpha and beta with HAC standard errors"""
    if not STATS_AVAILABLE:
        return {}
    
    # Align
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    strat = strategy_returns.loc[common_idx].values
    bench = benchmark_returns.loc[common_idx].values
    
    mask = ~(np.isnan(strat) | np.isnan(bench))
    strat = strat[mask]
    bench = bench[mask]
    
    if len(strat) < 30:
        return {}
    
    X = sm.add_constant(bench)
    model = sm.OLS(strat, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    
    return {
        'Alpha (Daily)': model.params[0],
        'Alpha (Annual)': model.params[0] * 252,
        'Beta': model.params[1],
        'R-Squared': model.rsquared,
        'Alpha P-Value': model.pvalues[0],
        'Alpha T-Stat': model.tvalues[0],
        'Alpha Significant': model.pvalues[0] < 0.05,
        'Alpha CI Lower': model.conf_int()[0, 0] * 252,
        'Alpha CI Upper': model.conf_int()[0, 1] * 252
    }


# =============================================================================
# ROBUSTNESS TESTS
# =============================================================================

def run_subperiod_analysis(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    n_periods: int = 4
) -> pd.DataFrame:
    """Subperiod robustness analysis"""
    n = len(strategy_returns)
    period_size = n // n_periods
    
    results = []
    for i in range(n_periods):
        start_idx = i * period_size
        end_idx = (i + 1) * period_size if i < n_periods - 1 else n
        
        period_strat = strategy_returns.iloc[start_idx:end_idx]
        period_bench = benchmark_returns.iloc[start_idx:end_idx]
        
        metrics = calculate_metrics(period_strat)
        alpha_beta = calculate_alpha_beta(period_strat, period_bench)
        
        results.append({
            'Period': f"P{i+1}",
            'Start': period_strat.index[0].strftime('%Y-%m-%d'),
            'End': period_strat.index[-1].strftime('%Y-%m-%d'),
            'Days': len(period_strat),
            'Return': metrics.get('Total Return', 0),
            'Sharpe': metrics.get('Sharpe Ratio', 0),
            'Max DD': metrics.get('Max Drawdown', 0),
            'Alpha': alpha_beta.get('Alpha (Annual)', 0),
            'P-Value': alpha_beta.get('Alpha P-Value', 1),
            'Sig': '✅' if alpha_beta.get('Alpha Significant', False) else '❌'
        })
    
    return pd.DataFrame(results)


def run_bootstrap_analysis(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    n_bootstrap: int = 1000
) -> Dict:
    """Block bootstrap for confidence intervals"""
    
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    strat = strategy_returns.loc[common_idx].values
    bench = benchmark_returns.loc[common_idx].values
    
    T = len(strat)
    block_size = 22
    n_blocks = T // block_size + 1
    
    alphas = []
    sharpes = []
    
    for _ in range(n_bootstrap):
        blocks = np.random.choice(T - block_size + 1, n_blocks, replace=True)
        indices = np.concatenate([np.arange(b, min(b + block_size, T)) for b in blocks])[:T]
        
        boot_strat = strat[indices]
        boot_bench = bench[indices]
        
        # Sharpe
        if boot_strat.std() > 0:
            sharpes.append(boot_strat.mean() / boot_strat.std() * np.sqrt(252))
        
        # Alpha
        X = np.column_stack([np.ones(T), boot_bench])
        try:
            beta = np.linalg.lstsq(X, boot_strat, rcond=None)[0]
            alphas.append(beta[0] * 252)
        except:
            pass
    
    alphas = np.array(alphas)
    sharpes = np.array(sharpes)
    
    return {
        'alpha_mean': np.mean(alphas),
        'alpha_ci_lower': np.percentile(alphas, 2.5),
        'alpha_ci_upper': np.percentile(alphas, 97.5),
        'alpha_contains_zero': np.percentile(alphas, 2.5) <= 0 <= np.percentile(alphas, 97.5),
        'sharpe_mean': np.mean(sharpes),
        'sharpe_ci_lower': np.percentile(sharpes, 2.5),
        'sharpe_ci_upper': np.percentile(sharpes, 97.5),
        'alphas': alphas,
        'sharpes': sharpes
    }


def run_transaction_cost_sensitivity(
    strategy_returns: pd.Series,
    turnover: float = 0.3,
    costs_bps: List[int] = [5, 10, 15, 20, 30, 50]
) -> pd.DataFrame:
    """Transaction cost sensitivity analysis"""
    
    base_metrics = calculate_metrics(strategy_returns)
    
    results = []
    for cost in costs_bps:
        annual_cost = turnover * (cost / 10000) * 12
        daily_cost = annual_cost / 252
        
        adj_returns = strategy_returns - daily_cost
        adj_metrics = calculate_metrics(adj_returns)
        
        results.append({
            'Cost (bps)': cost,
            'Annual Impact': annual_cost,
            'Adj Sharpe': adj_metrics.get('Sharpe Ratio', 0),
            'Sharpe Δ': adj_metrics.get('Sharpe Ratio', 0) - base_metrics.get('Sharpe Ratio', 0),
            'Adj CAGR': adj_metrics.get('CAGR', 0),
            'CAGR Δ': adj_metrics.get('CAGR', 0) - base_metrics.get('CAGR', 0)
        })
    
    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION FUNCTIONS  
# =============================================================================

def plot_cumulative_returns(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    benchmark_name: str
) -> go.Figure:
    """Plot cumulative returns"""
    
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    
    strat_cum = (1 + strategy_returns.loc[common_idx]).cumprod()
    bench_cum = (1 + benchmark_returns.loc[common_idx]).cumprod()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=strat_cum.index, y=strat_cum.values,
        name='Strategy',
        line=dict(color='#00C853', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(0, 200, 83, 0.1)'
    ))
    
    fig.add_trace(go.Scatter(
        x=bench_cum.index, y=bench_cum.values,
        name=benchmark_name,
        line=dict(color='#2196F3', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Growth of $1',
        hovermode='x unified',
        template='plotly_white',
        height=450
    )
    
    return fig


def plot_drawdown(returns: pd.Series) -> go.Figure:
    """Plot drawdown"""
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values * 100,
        fill='tozeroy',
        fillcolor='rgba(255, 82, 82, 0.3)',
        line=dict(color='#FF5252', width=1),
        name='Drawdown'
    ))
    
    fig.update_layout(
        title='Drawdown (%)',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        template='plotly_white',
        height=300
    )
    
    return fig


def plot_allocation_pie(weights: Dict[str, float], title: str) -> go.Figure:
    """Plot allocation pie chart"""
    
    labels = []
    values = []
    for k, v in sorted(weights.items(), key=lambda x: -x[1]):
        if v > 0.01:
            labels.append(k)
            values.append(v)
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        textinfo='label+percent',
        textposition='outside'
    )])
    
    fig.update_layout(title=title, height=400)
    
    return fig


def plot_regime_timeline(
    returns: pd.Series,
    regimes: np.ndarray
) -> go.Figure:
    """Plot returns with regime coloring"""
    
    cum_returns = (1 + returns).cumprod()
    
    regime_colors = {0: '#00C853', 1: '#FFB300', 2: '#FF5252'}
    regime_names = {0: 'Bull', 1: 'Transition', 2: 'Bear'}
    
    fig = go.Figure()
    
    for r in [0, 1, 2]:
        mask = regimes == r
        if mask.any():
            fig.add_trace(go.Scatter(
                x=cum_returns.index[mask],
                y=cum_returns.values[mask],
                mode='markers',
                name=regime_names[r],
                marker=dict(color=regime_colors[r], size=4)
            ))
    
    fig.update_layout(
        title='Portfolio Value with HMM Regimes',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_bootstrap_distribution(bootstrap_results: Dict) -> go.Figure:
    """Plot bootstrap distributions"""
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Alpha Distribution', 'Sharpe Distribution'])
    
    # Alpha
    fig.add_trace(go.Histogram(
        x=bootstrap_results['alphas'] * 100,
        nbinsx=50,
        marker_color='#2196F3',
        name='Alpha'
    ), row=1, col=1)
    
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_vline(x=bootstrap_results['alpha_ci_lower'] * 100, line_dash="dot", line_color="green", row=1, col=1)
    fig.add_vline(x=bootstrap_results['alpha_ci_upper'] * 100, line_dash="dot", line_color="green", row=1, col=1)
    
    # Sharpe
    fig.add_trace(go.Histogram(
        x=bootstrap_results['sharpes'],
        nbinsx=50,
        marker_color='#9C27B0',
        name='Sharpe'
    ), row=1, col=2)
    
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=2)
    
    fig.update_layout(height=350, template='plotly_white', showlegend=False)
    fig.update_xaxes(title_text='Alpha (%)', row=1, col=1)
    fig.update_xaxes(title_text='Sharpe Ratio', row=1, col=2)
    
    return fig


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.markdown('<h1 class="main-header">🚀 Multi-Agent Portfolio Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 1rem;'>
        Real Data • Asset Allocation • Robustness Tests • Statistical Significance
    </div>
    """, unsafe_allow_html=True)
    
    # =================================================================
    # SIDEBAR
    # =================================================================
    
    st.sidebar.header("⚙️ Configuration")
    
    # Date Range
    st.sidebar.subheader("📅 Date Range")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start", datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input("End", datetime.now())
    
    # Benchmark
    st.sidebar.subheader("📈 Benchmark")
    benchmark_choice = st.sidebar.selectbox("Select Benchmark", list(BENCHMARK_OPTIONS.keys()))
    benchmark_ticker = BENCHMARK_OPTIONS[benchmark_choice]
    
    # Rebalancing
    st.sidebar.subheader("🔄 Rebalancing")
    rebalance_freq = st.sidebar.selectbox("Frequency", ['M', 'W', 'Q'], index=0, 
                                          format_func=lambda x: {'M': 'Monthly', 'W': 'Weekly', 'Q': 'Quarterly'}[x])
    
    # Risk Parameters
    st.sidebar.subheader("⚠️ Parameters")
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate", 0.0, 0.10,0.045 , 0.04,0.035,0.03, 0.025, 0.02, 0.01)
    transaction_cost = st.sidebar.slider("Transaction Cost (bps)", 0, 50, 10)
    
    # Robustness
    st.sidebar.subheader("🔬 Robustness")
    n_subperiods = st.sidebar.slider("Subperiods", 2, 8, 4)
    n_bootstrap = st.sidebar.slider("Bootstrap Samples", 500, 5000, 1000)
    
    # =================================================================
    # LOAD DATA
    # =================================================================
    
    with st.spinner("📊 Loading market data from Yahoo Finance..."):
        prices, ticker_info = fetch_all_data(str(start_date), str(end_date))
        benchmark_prices = fetch_benchmark(benchmark_ticker, str(start_date), str(end_date))
    
    if prices.empty:
        st.error("Failed to load data. Please check your internet connection.")
        return
    
    st.success(f"✅ Loaded {len(prices.columns)} assets, {len(prices)} trading days")
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    benchmark_returns = benchmark_prices.pct_change().dropna()
    
    # Run HMM
    with st.spinner("🎯 Running HMM regime detection..."):
        regimes, regime_probs = run_hmm_regime_detection(returns)
    
    # Simulate strategy
    with st.spinner("💼 Simulating portfolio strategy..."):
        strategy_returns, weights_history, rebalance_dates = simulate_strategy_returns(
            prices, regimes, rebalance_freq
        )
    
    # Get current allocation
    current_regime = regimes[-1]
    sector_weights, detailed_weights, selection_df = generate_portfolio_weights(
        prices, regimes, current_regime
    )
    
    # Calculate metrics
    strategy_metrics = calculate_metrics(strategy_returns, risk_free_rate)
    benchmark_metrics = calculate_metrics(benchmark_returns, risk_free_rate)
    alpha_beta = calculate_alpha_beta(strategy_returns, benchmark_returns)
    
    regime_names = {0: 'Bull 🟢', 1: 'Transition 🟡', 2: 'Bear 🔴'}
    
    # =================================================================
    # TABS
    # =================================================================
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Performance",
        "💰 Allocation",
        "🎯 Regimes",
        "🔬 Robustness",
        "📊 Statistics",
        "📋 Report"
    ])
    
    # =================================================================
    # TAB 1: PERFORMANCE
    # =================================================================
    
    with tab1:
        st.header("Performance Overview")
        
        # Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            delta = strategy_metrics['Total Return'] - benchmark_metrics['Total Return']
            st.metric("Total Return", f"{strategy_metrics['Total Return']:.2%}", f"{delta:+.2%}")
        
        with col2:
            delta = strategy_metrics['Sharpe Ratio'] - benchmark_metrics['Sharpe Ratio']
            st.metric("Sharpe Ratio", f"{strategy_metrics['Sharpe Ratio']:.2f}", f"{delta:+.2f}")
        
        with col3:
            st.metric("Max Drawdown", f"{strategy_metrics['Max Drawdown']:.2%}")
        
        with col4:
            st.metric("CAGR", f"{strategy_metrics['CAGR']:.2%}")
        
        with col5:
            st.metric("Current Regime", regime_names.get(current_regime, "Unknown"))
        
        # Charts
        fig_cum = plot_cumulative_returns(strategy_returns, benchmark_returns, benchmark_ticker)
        st.plotly_chart(fig_cum, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_dd = plot_drawdown(strategy_returns)
            st.plotly_chart(fig_dd, use_container_width=True)
        
        with col2:
            # Rolling Sharpe
            rolling_sharpe = strategy_returns.rolling(63).mean() / strategy_returns.rolling(63).std() * np.sqrt(252)
            fig_rs = go.Figure()
            fig_rs.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, line=dict(color='#9C27B0')))
            fig_rs.add_hline(y=0, line_dash="dash")
            fig_rs.add_hline(y=1, line_dash="dot", line_color="green", opacity=0.5)
            fig_rs.update_layout(title='63-Day Rolling Sharpe', template='plotly_white', height=300)
            st.plotly_chart(fig_rs, use_container_width=True)
        
        # Metrics Table
        st.subheader("Detailed Metrics")
        metrics_df = pd.DataFrame({
            'Metric': list(strategy_metrics.keys()),
            'Strategy': [f"{v:.4f}" for v in strategy_metrics.values()],
            'Benchmark': [f"{benchmark_metrics.get(k, 0):.4f}" for k in strategy_metrics.keys()]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # =================================================================
    # TAB 2: ALLOCATION
    # =================================================================
    
    with tab2:
        st.header("Asset Allocation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sector Allocation")
            fig_sector = plot_allocation_pie(sector_weights, f"Sector Weights (Regime: {regime_names[current_regime]})")
            st.plotly_chart(fig_sector, use_container_width=True)
        
        with col2:
            st.subheader("Individual Positions")
            # Top 15 positions
            top_positions = dict(sorted(detailed_weights.items(), key=lambda x: -x[1])[:15])
            fig_pos = plot_allocation_pie(top_positions, "Top 15 Holdings")
            st.plotly_chart(fig_pos, use_container_width=True)
        
        # Stock Selection Table
        st.subheader("📋 Stock Selection Details")
        
        if not selection_df.empty:
            selection_df_display = selection_df.copy()
            selection_df_display['Weight'] = selection_df_display['Weight'].apply(lambda x: f"{x:.2%}")
            selection_df_display['Score'] = selection_df_display['Score'].apply(lambda x: f"{x:.4f}")
            st.dataframe(selection_df_display, use_container_width=True, hide_index=True)
        
        # Regime-based allocation rules
        st.subheader("📊 Regime-Based Allocation Rules")
        
        rules_data = []
        for sector, info in ASSET_UNIVERSE.items():
            rules_data.append({
                'Sector': sector,
                'Description': info['description'],
                'Bull': f"{info['bull_weight']:.0%}",
                'Transition': f"{info['transition_weight']:.0%}",
                'Bear': f"{info['bear_weight']:.0%}",
                '# Tickers': len(info['tickers'])
            })
        
        st.dataframe(pd.DataFrame(rules_data), use_container_width=True, hide_index=True)
        
        # Asset Universe
        st.subheader("🌐 Complete Asset Universe")
        
        with st.expander("View All Available Assets"):
            for sector, info in ASSET_UNIVERSE.items():
                st.markdown(f"**{sector}**: {', '.join(info['tickers'])}")
    
    # =================================================================
    # TAB 3: REGIMES
    # =================================================================
    
    with tab3:
        st.header("HMM Regime Analysis")
        
        # Distribution
        col1, col2, col3 = st.columns(3)
        regime_counts = pd.Series(regimes).value_counts().sort_index()
        
        for i, col in enumerate([col1, col2, col3]):
            with col:
                count = regime_counts.get(i, 0)
                pct = count / len(regimes) * 100
                st.metric(['Bull 🟢', 'Transition 🟡', 'Bear 🔴'][i], f"{pct:.1f}%", f"{count} days")
        
        # Timeline
        fig_regime = plot_regime_timeline(strategy_returns, regimes[-len(strategy_returns):])
        st.plotly_chart(fig_regime, use_container_width=True)
        
        # Regime probabilities
        st.subheader("Regime Probability Over Time")
        
        fig_probs = go.Figure()
        dates = strategy_returns.index
        probs = regime_probs[-len(dates):]
        
        for r, name, color in [(0, 'Bull', 'rgba(0,200,83,0.7)'), (1, 'Transition', 'rgba(255,179,0,0.7)'), (2, 'Bear', 'rgba(255,82,82,0.7)')]:
            fig_probs.add_trace(go.Scatter(
                x=dates, y=probs[:, r],
                name=name, stackgroup='one',
                fillcolor=color, line=dict(width=0)
            ))
        fig_probs.update_layout(template='plotly_white', height=350)
        st.plotly_chart(fig_probs, use_container_width=True)
        
        # Regime statistics
        st.subheader("Performance by Regime")
        
        regime_stats = []
        for r in [0, 1, 2]:
            mask = regimes[-len(strategy_returns):] == r
            if mask.sum() > 10:
                r_returns = strategy_returns[mask]
                r_bench = benchmark_returns.loc[r_returns.index] if hasattr(benchmark_returns, 'loc') else benchmark_returns[mask]
                
                metrics = calculate_metrics(r_returns)
                ab = calculate_alpha_beta(r_returns, r_bench)
                
                regime_stats.append({
                    'Regime': ['Bull', 'Transition', 'Bear'][r],
                    'Days': mask.sum(),
                    'Mean Return': f"{r_returns.mean()*100:.4f}%",
                    'Volatility': f"{r_returns.std()*100:.4f}%",
                    'Sharpe': f"{metrics.get('Sharpe Ratio', 0):.2f}",
                    'Alpha (Ann)': f"{ab.get('Alpha (Annual)', 0)*100:.2f}%",
                    'Significant': '✅' if ab.get('Alpha Significant', False) else '❌'
                })
        
        st.dataframe(pd.DataFrame(regime_stats), use_container_width=True, hide_index=True)
    
    # =================================================================
    # TAB 4: ROBUSTNESS
    # =================================================================
    
    with tab4:
        st.header("Robustness Tests")
        
        # Subperiod Analysis
        st.subheader("📅 Subperiod Analysis")
        
        subperiod_df = run_subperiod_analysis(strategy_returns, benchmark_returns, n_subperiods)
        
        st.dataframe(subperiod_df.style.format({
            'Return': '{:.2%}',
            'Sharpe': '{:.2f}',
            'Max DD': '{:.2%}',
            'Alpha': '{:.2%}',
            'P-Value': '{:.4f}'
        }), use_container_width=True, hide_index=True)
        
        # Subperiod chart
        fig_sub = go.Figure()
        colors = ['#00C853' if s > 0 else '#FF5252' for s in subperiod_df['Sharpe']]
        fig_sub.add_trace(go.Bar(x=subperiod_df['Period'], y=subperiod_df['Sharpe'], marker_color=colors))
        fig_sub.add_hline(y=0, line_dash="dash")
        fig_sub.add_hline(y=1, line_dash="dot", line_color="green", opacity=0.5)
        fig_sub.update_layout(title='Sharpe by Subperiod', template='plotly_white', height=300)
        st.plotly_chart(fig_sub, use_container_width=True)
        
        # Consistency check
        positive_sharpes = (subperiod_df['Sharpe'] > 0).sum()
        if positive_sharpes == len(subperiod_df):
            st.markdown('<div class="success-box">✅ <b>Consistent Performance:</b> Positive Sharpe in all subperiods</div>', unsafe_allow_html=True)
        elif positive_sharpes >= len(subperiod_df) * 0.75:
            st.markdown(f'<div class="warning-box">⚠️ <b>Mostly Consistent:</b> {positive_sharpes}/{len(subperiod_df)} periods with positive Sharpe</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="danger-box">❌ <b>Inconsistent:</b> Only {positive_sharpes}/{len(subperiod_df)} periods with positive Sharpe</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Bootstrap
        st.subheader("🎲 Bootstrap Confidence Intervals")
        
        if st.button("Run Bootstrap Analysis", key="bootstrap"):
            with st.spinner(f"Running {n_bootstrap} bootstrap samples..."):
                bootstrap_results = run_bootstrap_analysis(strategy_returns, benchmark_returns, n_bootstrap)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Alpha 95% CI Lower", f"{bootstrap_results['alpha_ci_lower']*100:.2f}%")
                st.metric("Alpha 95% CI Upper", f"{bootstrap_results['alpha_ci_upper']*100:.2f}%")
            with col2:
                st.metric("Sharpe 95% CI Lower", f"{bootstrap_results['sharpe_ci_lower']:.2f}")
                st.metric("Sharpe 95% CI Upper", f"{bootstrap_results['sharpe_ci_upper']:.2f}")
            
            if not bootstrap_results['alpha_contains_zero']:
                st.markdown('<div class="success-box">✅ <b>95% CI does NOT contain zero</b> - Alpha is statistically robust</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">⚠️ <b>95% CI contains zero</b> - Alpha may not be robust</div>', unsafe_allow_html=True)
            
            fig_boot = plot_bootstrap_distribution(bootstrap_results)
            st.plotly_chart(fig_boot, use_container_width=True)
        
        st.divider()
        
        # Transaction Cost
        st.subheader("💰 Transaction Cost Sensitivity")
        
        tc_df = run_transaction_cost_sensitivity(strategy_returns, turnover=0.3)
        
        st.dataframe(tc_df.style.format({
            'Annual Impact': '{:.2%}',
            'Adj Sharpe': '{:.2f}',
            'Sharpe Δ': '{:.2f}',
            'Adj CAGR': '{:.2%}',
            'CAGR Δ': '{:.2%}'
        }), use_container_width=True, hide_index=True)
        
        # Find breakeven
        still_positive = tc_df[tc_df['Adj Sharpe'] > 0]
        if len(still_positive) == len(tc_df):
            st.markdown('<div class="success-box">✅ Strategy remains profitable even at 50 bps transaction cost</div>', unsafe_allow_html=True)
        elif len(still_positive) > 0:
            max_cost = still_positive['Cost (bps)'].max()
            st.markdown(f'<div class="warning-box">⚠️ Strategy profitable up to {max_cost} bps transaction cost</div>', unsafe_allow_html=True)
    
    # =================================================================
    # TAB 5: STATISTICS
    # =================================================================
    
    with tab5:
        st.header("Statistical Significance")
        
        # Alpha/Beta
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sig_icon = "🟢" if alpha_beta.get('Alpha Significant', False) else "🟡"
            st.metric(f"Annual Alpha {sig_icon}", f"{alpha_beta.get('Alpha (Annual)', 0)*100:.2f}%")
        with col2:
            st.metric("Beta", f"{alpha_beta.get('Beta', 0):.3f}")
        with col3:
            st.metric("R-Squared", f"{alpha_beta.get('R-Squared', 0)*100:.2f}%")
        with col4:
            pval = alpha_beta.get('Alpha P-Value', 1)
            st.metric("P-Value", f"{pval:.4f}", "Significant" if pval < 0.05 else "Not Sig.")
        
        # Interpretation
        if alpha_beta.get('Alpha Significant', False):
            st.markdown(f'''
            <div class="success-box">
            <h4>✅ Alpha is Statistically Significant</h4>
            <ul>
                <li>Annual Alpha: {alpha_beta.get('Alpha (Annual)', 0)*100:.2f}%</li>
                <li>T-Statistic: {alpha_beta.get('Alpha T-Stat', 0):.2f}</li>
                <li>P-Value: {alpha_beta.get('Alpha P-Value', 1):.4f} (< 0.05)</li>
                <li>95% CI: [{alpha_beta.get('Alpha CI Lower', 0)*100:.2f}%, {alpha_beta.get('Alpha CI Upper', 0)*100:.2f}%]</li>
            </ul>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="warning-box">
            <h4>⚠️ Alpha Not Statistically Significant at 5% Level</h4>
            <p>P-Value: {alpha_beta.get('Alpha P-Value', 1):.4f}</p>
            <p>Consider: longer backtest period, different benchmark, or regime-conditional analysis</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Regime-conditional alpha
        st.subheader("Regime-Conditional Alpha")
        
        regime_alpha_results = []
        for r in [0, 1, 2]:
            mask = regimes[-len(strategy_returns):] == r
            if mask.sum() > 30:
                r_strat = strategy_returns[mask]
                r_bench = benchmark_returns.reindex(r_strat.index).dropna()
                common = r_strat.index.intersection(r_bench.index)
                
                if len(common) > 30:
                    ab = calculate_alpha_beta(r_strat.loc[common], r_bench.loc[common])
                    regime_alpha_results.append({
                        'Regime': ['Bull', 'Transition', 'Bear'][r],
                        'Alpha (Ann)': ab.get('Alpha (Annual)', 0),
                        'P-Value': ab.get('Alpha P-Value', 1),
                        'Significant': ab.get('Alpha Significant', False)
                    })
        
        if regime_alpha_results:
            fig_regime_alpha = go.Figure()
            colors = ['#00C853' if r['Significant'] else '#FFB300' for r in regime_alpha_results]
            fig_regime_alpha.add_trace(go.Bar(
                x=[r['Regime'] for r in regime_alpha_results],
                y=[r['Alpha (Ann)']*100 for r in regime_alpha_results],
                marker_color=colors,
                text=[f"{r['Alpha (Ann)']*100:.1f}%" for r in regime_alpha_results],
                textposition='outside'
            ))
            fig_regime_alpha.add_hline(y=0, line_dash="dash")
            fig_regime_alpha.update_layout(title='Alpha by Regime (%)', template='plotly_white', height=350)
            st.plotly_chart(fig_regime_alpha, use_container_width=True)
    
    # =================================================================
    # TAB 6: REPORT
    # =================================================================
    
    with tab6:
        st.header("Summary Report")
        
        report = f"""
# Multi-Agent Portfolio Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Benchmark:** {benchmark_ticker}
**Period:** {start_date} to {end_date}
**Trading Days:** {len(strategy_returns)}

---

## 1. Performance Summary

| Metric | Strategy | {benchmark_ticker} | Difference |
|--------|----------|-----------|------------|
| Total Return | {strategy_metrics['Total Return']:.2%} | {benchmark_metrics['Total Return']:.2%} | {strategy_metrics['Total Return'] - benchmark_metrics['Total Return']:+.2%} |
| CAGR | {strategy_metrics['CAGR']:.2%} | {benchmark_metrics['CAGR']:.2%} | {strategy_metrics['CAGR'] - benchmark_metrics['CAGR']:+.2%} |
| Sharpe Ratio | {strategy_metrics['Sharpe Ratio']:.2f} | {benchmark_metrics['Sharpe Ratio']:.2f} | {strategy_metrics['Sharpe Ratio'] - benchmark_metrics['Sharpe Ratio']:+.2f} |
| Max Drawdown | {strategy_metrics['Max Drawdown']:.2%} | {benchmark_metrics['Max Drawdown']:.2%} | - |
| Volatility | {strategy_metrics['Volatility']:.2%} | {benchmark_metrics['Volatility']:.2%} | - |

---

## 2. Statistical Significance

- **Annual Alpha:** {alpha_beta.get('Alpha (Annual)', 0)*100:.2f}%
- **Alpha P-Value:** {alpha_beta.get('Alpha P-Value', 1):.4f}
- **Statistically Significant:** {'Yes ✅' if alpha_beta.get('Alpha Significant', False) else 'No ❌'}
- **Beta:** {alpha_beta.get('Beta', 0):.3f}
- **R-Squared:** {alpha_beta.get('R-Squared', 0)*100:.2f}%

---

## 3. Asset Universe

Total Assets: {len(prices.columns)}

| Sector | # Assets | Current Weight |
|--------|----------|----------------|
"""
        for sector, weight in sector_weights.items():
            n_assets = len(ASSET_UNIVERSE.get(sector, {}).get('tickers', []))
            report += f"| {sector} | {n_assets} | {weight:.1%} |\n"
        
        report += f"""

---

## 4. Regime Analysis

| Regime | Proportion | Strategy Sharpe |
|--------|------------|-----------------|
| Bull | {(regimes == 0).mean():.1%} | - |
| Transition | {(regimes == 1).mean():.1%} | - |
| Bear | {(regimes == 2).mean():.1%} | - |

---

## 5. Conclusion

"""
        if alpha_beta.get('Alpha Significant', False):
            report += "✅ **Strategy generates statistically significant alpha.**\n"
        else:
            report += "⚠️ **Alpha not statistically significant at 5% level.**\n"
        
        if strategy_metrics['Sharpe Ratio'] > 1:
            report += f"✅ Sharpe Ratio ({strategy_metrics['Sharpe Ratio']:.2f}) exceeds 1.0 threshold.\n"
        
        report += "\n---\n*Generated by Enhanced Multi-Agent Portfolio System*"
        
        st.markdown(report)
        
        st.download_button(
            "📥 Download Report",
            report,
            file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )


if __name__ == "__main__":
    main()
