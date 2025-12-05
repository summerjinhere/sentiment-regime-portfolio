"""
Enhanced Dashboard v3 - Complete Visualization
==============================================

Streamlit application with:
- Configurable Trading Days
- FinBERT Sentiment Visualization
- Dynamic Beta (Kalman Filter)
- Portfolio Performance
- Complete Statistical Validation
- Robustness Tests

Uses ONLY REAL DATA.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import model
try:
    from enhanced_integrated_model_v2 import (
        EnhancedIntegratedModel, 
        ModelConfig,
        StatisticalValidator,
        PerformanceCalculator
    )
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Model not imported")

try:
    from research_pipeline_v1 import (
        ResearchDesign,
        CrossSectionalTester,
        DecilePortfolioTester,
        AblationTester,
        FactorValidator,
        SubperiodTester
    )
    RESEARCH_AVAILABLE = True
except ImportError:
    RESEARCH_AVAILABLE = False
    print("Warning: Research pipeline not imported")

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Welcome to J-H Strategy World",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #00C853;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #FFB300;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 5px solid #FF5252;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #e7f3ff;
        border-left: 5px solid #2196F3;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .stat-significant {
        color: #00C853;
        font-weight: bold;
    }
    .stat-not-significant {
        color: #FF5252;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(ttl=3600)
def load_price_data(filepath: str, trading_days: int = None):
    """Load price and returns data"""
    try:
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        ret_cols = [c for c in data.columns if c.endswith('_ret')]
        returns = data[ret_cols].copy()
        returns.columns = [c.replace('_ret', '') for c in ret_cols]
        
        price_cols = [c for c in data.columns if c.endswith('_price')]
        prices = data[price_cols].copy()
        prices.columns = [c.replace('_price', '') for c in price_cols]
        
        # Filter by trading days
        if trading_days and len(returns) > trading_days:
            returns = returns.iloc[-trading_days:]
            prices = prices.iloc[-trading_days:]
        
        return prices, returns
    except Exception as e:
        st.error(f"Error loading price data: {e}")
        return None, None


@st.cache_data(ttl=3600)
def load_sentiment_data(filepath: str, trading_days: int = None):
    """Load sentiment data"""
    try:
        sentiment = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        if trading_days and len(sentiment) > trading_days:
            sentiment = sentiment.iloc[-trading_days:]
        
        return sentiment
    except Exception as e:
        return None


@st.cache_resource
def load_model(trading_days: int, _config: ModelConfig):
    """Load and fit the model"""
    if not MODEL_AVAILABLE:
        return None
    
    try:
        model = EnhancedIntegratedModel(_config)
        if model.load_data(trading_days=trading_days):
            model.fit()
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
    return None


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_sentiment_signal(sentiment: pd.DataFrame, ticker: str = 'MARKET_SENTIMENT'):
    """Plot FinBERT sentiment signal"""
    if sentiment is None or ticker not in sentiment.columns:
        return None
    
    data = sentiment[ticker].dropna()
    
    fig = go.Figure()
    
    # Main line
    fig.add_trace(go.Scatter(
        x=data.index, y=data.values,
        mode='lines', name='Sentiment',
        line=dict(color='#2196F3', width=1.5)
    ))
    
    # Positive fill
    fig.add_trace(go.Scatter(
        x=data.index, y=np.where(data.values > 0, data.values, 0),
        fill='tozeroy', fillcolor='rgba(0, 200, 83, 0.2)',
        line=dict(width=0), name='Positive', showlegend=False
    ))
    
    # Negative fill
    fig.add_trace(go.Scatter(
        x=data.index, y=np.where(data.values < 0, data.values, 0),
        fill='tozeroy', fillcolor='rgba(255, 82, 82, 0.2)',
        line=dict(width=0), name='Negative', showlegend=False
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    fig.add_hline(y=0.3, line_dash="dot", line_color="green", opacity=0.3)
    fig.add_hline(y=-0.3, line_dash="dot", line_color="red", opacity=0.3)
    
    fig.update_layout(
        title=f'FinBERT Sentiment Signal: {ticker}',
        xaxis_title='Date', yaxis_title='Sentiment Score',
        template='plotly_white', height=400,
        yaxis=dict(range=[-1, 1])
    )
    
    return fig


def plot_dynamic_beta(model, ticker: str):
    """Plot time-varying beta from Kalman Filter"""
    if model is None:
        return None
    
    beta_series = model.get_dynamic_beta_series(ticker)
    if beta_series.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=beta_series.index, y=beta_series.values,
        mode='lines', name='Dynamic Beta',
        line=dict(color='#9C27B0', width=2)
    ))
    
    # Rolling mean
    rolling = beta_series.rolling(22).mean()
    fig.add_trace(go.Scatter(
        x=rolling.index, y=rolling.values,
        mode='lines', name='22-Day MA',
        line=dict(color='#FF9800', width=1.5, dash='dot')
    ))
    
    fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f'Dynamic Beta (Kalman Filter): {ticker}',
        xaxis_title='Date', yaxis_title='Beta',
        template='plotly_white', height=400
    )
    
    return fig


def plot_cumulative_returns(strategy_returns: pd.Series, benchmark_returns: pd.Series, benchmark_name: str = 'SPY'):
    """Plot cumulative returns comparison"""
    strategy_cum = (1 + strategy_returns).cumprod()
    benchmark_cum = (1 + benchmark_returns).cumprod()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=strategy_cum.index, y=strategy_cum.values,
        name='Strategy', line=dict(color='#00C853', width=2.5),
        fill='tozeroy', fillcolor='rgba(0, 200, 83, 0.1)'
    ))
    
    fig.add_trace(go.Scatter(
        x=benchmark_cum.index, y=benchmark_cum.values,
        name=benchmark_name, line=dict(color='#2196F3', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Cumulative Returns',
        xaxis_title='Date', yaxis_title='Growth of $1',
        template='plotly_white', height=450,
        hovermode='x unified'
    )
    
    return fig


def plot_drawdown(returns: pd.Series):
    """Plot drawdown chart"""
    cum = (1 + returns).cumprod()
    drawdown = cum / cum.cummax() - 1
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values * 100,
        fill='tozeroy', fillcolor='rgba(255, 82, 82, 0.3)',
        line=dict(color='#FF5252', width=1),
        name='Drawdown'
    ))
    
    fig.update_layout(
        title='Drawdown',
        xaxis_title='Date', yaxis_title='Drawdown (%)',
        template='plotly_white', height=300
    )
    
    return fig


def plot_bootstrap_distribution(bootstrap_results: Dict):
    """Plot bootstrap distribution"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Alpha Distribution', 'Sharpe Distribution'))
    
    # Alpha
    alphas = bootstrap_results['bootstrap_alphas'] * 100
    fig.add_trace(go.Histogram(
        x=alphas, nbinsx=50, name='Alpha',
        marker_color='#667eea', opacity=0.7
    ), row=1, col=1)
    
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_vline(x=bootstrap_results['alpha_ci_lower']*100, line_dash="dot", line_color="green", row=1, col=1)
    fig.add_vline(x=bootstrap_results['alpha_ci_upper']*100, line_dash="dot", line_color="green", row=1, col=1)
    
    # Sharpe
    sharpes = bootstrap_results['bootstrap_sharpes']
    fig.add_trace(go.Histogram(
        x=sharpes, nbinsx=50, name='Sharpe',
        marker_color='#764ba2', opacity=0.7
    ), row=1, col=2)
    
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_vline(x=bootstrap_results['sharpe_ci_lower'], line_dash="dot", line_color="green", row=1, col=2)
    fig.add_vline(x=bootstrap_results['sharpe_ci_upper'], line_dash="dot", line_color="green", row=1, col=2)
    
    fig.update_layout(
        template='plotly_white', height=350,
        showlegend=False
    )
    fig.update_xaxes(title_text="Alpha (%)", row=1, col=1)
    fig.update_xaxes(title_text="Sharpe Ratio", row=1, col=2)
    
    return fig


def plot_subperiod_analysis(subperiods: list):
    """Plot subperiod analysis"""
    df = pd.DataFrame(subperiods)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Returns by Period', 'Sharpe by Period', 'Alpha by Period', 'Significance')
    )
    
    colors = ['#00C853' if r > 0 else '#FF5252' for r in df['total_return']]
    
    # Returns
    fig.add_trace(go.Bar(
        x=[f"P{p}" for p in df['period']], y=df['total_return']*100,
        marker_color=colors, name='Return'
    ), row=1, col=1)
    
    # Sharpe
    sharpe_colors = ['#00C853' if s > 0 else '#FF5252' for s in df['sharpe_ratio']]
    fig.add_trace(go.Bar(
        x=[f"P{p}" for p in df['period']], y=df['sharpe_ratio'],
        marker_color=sharpe_colors, name='Sharpe'
    ), row=1, col=2)
    
    # Alpha
    alpha_colors = ['#00C853' if s else '#FF5252' for s in df['significant']]
    fig.add_trace(go.Bar(
        x=[f"P{p}" for p in df['period']], y=df['alpha_annual']*100,
        marker_color=alpha_colors, name='Alpha'
    ), row=2, col=1)
    
    # P-values
    fig.add_trace(go.Bar(
        x=[f"P{p}" for p in df['period']], y=df['alpha_pval'],
        marker_color=['#00C853' if p < 0.05 else '#FF5252' for p in df['alpha_pval']],
        name='P-Value'
    ), row=2, col=2)
    fig.add_hline(y=0.05, line_dash="dash", line_color="red", row=2, col=2)
    
    fig.update_layout(template='plotly_white', height=500, showlegend=False)
    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
    fig.update_yaxes(title_text="Alpha (%)", row=2, col=1)
    fig.update_yaxes(title_text="P-Value", row=2, col=2)
    
    return fig


def plot_transaction_cost_sensitivity(tc_results: Dict):
    """Plot transaction cost sensitivity"""
    df = pd.DataFrame(tc_results['results'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['cost_bps'], y=df['sharpe_ratio'],
        mode='lines+markers', name='Sharpe Ratio',
        line=dict(color='#667eea', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.7)
    
    if tc_results['breakeven_cost_bps']:
        fig.add_vline(
            x=tc_results['breakeven_cost_bps'],
            line_dash="dot", line_color="orange",
            annotation_text=f"Breakeven: {tc_results['breakeven_cost_bps']:.0f}bps"
        )
    
    fig.update_layout(
        title='Transaction Cost Sensitivity',
        xaxis_title='Transaction Cost (bps)',
        yaxis_title='Sharpe Ratio',
        template='plotly_white', height=350
    )
    
    return fig


def plot_volatility(model, ticker: str):
    """Plot GARCH-MIDAS volatility"""
    if model is None:
        return None
    
    vol_series = model.get_volatility_series(ticker)
    if vol_series.empty:
        return None
    
    vol_annual = vol_series * np.sqrt(252) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=vol_annual.index, y=vol_annual.values,
        mode='lines', name='Volatility',
        line=dict(color='#FF5252', width=1.5),
        fill='tozeroy', fillcolor='rgba(255, 82, 82, 0.1)'
    ))
    
    fig.update_layout(
        title=f'GARCH-MIDAS Volatility: {ticker}',
        xaxis_title='Date', yaxis_title='Annualized Volatility (%)',
        template='plotly_white', height=350
    )
    
    return fig


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    returns = None
    st.markdown('<h1 class="main-header">🚀 AI Finance Dashboard v3</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
        Weakness Compensation Framework | Real Data Only | Statistical Validation
    </div>
    """, unsafe_allow_html=True)
    
    # =================================================================
    # SIDEBAR
    # =================================================================
    
    st.sidebar.header("⚙️ Configuration")
    
    # Trading Days Configuration
    st.sidebar.subheader("📅 Trading Days")
    trading_days_preset = st.sidebar.selectbox(
        "Preset",
        ["Custom", "1 Year (252)", "2 Years (504)", "3 Years (756)", "5 Years (1260)", "10 Years (2520)", "20 Years (5040)", "All Data"],
        index=1
    )
    
    if trading_days_preset == "Custom":
        min_date = returns.index[0].date() if returns is not None else datetime(2020, 1, 1).date()
        max_date = returns.index[-1].date() if returns is not None else datetime.now().date()
        
        start_date = st.sidebar.date_input(
            "Start Date",
            value=min_date + timedelta(days=365),
            min_value=min_date,
            max_value=max_date
        )
        
        if returns is not None:
            # Calculate trading days from start date
            mask = returns.index.date >= start_date
            trading_days = mask.sum()
        else:
            trading_days = 504 # Fallback
            
    elif trading_days_preset == "All Data":
        trading_days = None
    else:
        trading_days = int(trading_days_preset.split("(")[1].replace(")", ""))
    
    st.sidebar.markdown(f"**Selected:** {trading_days if trading_days else 'All'} days")
    
    # Data files
    st.sidebar.subheader("📁 Data Files")
    prices_file = st.sidebar.text_input("Price Data", "real_backtest_panel.csv")
    sentiment_file = st.sidebar.text_input("Sentiment Data", "real_sentiment_data.csv")
    
    # Load data
    with st.spinner("Loading data..."):
        prices, returns = load_price_data(prices_file, trading_days)
        sentiment = load_sentiment_data(sentiment_file, trading_days)
        
        if returns is not None and trading_days is not None:
            available_days = len(returns)
            # Relax check slightly for off-by-one or small data gaps
            if available_days < trading_days * 0.99:
                st.sidebar.warning(f"⚠️ Data limited to {available_days} days (requested {trading_days}).")
                # Update trading_days to reflect reality for downstream logic
                trading_days = available_days
    
    if returns is None:
        st.error("❌ Could not load price data. Please run fetch_real_data.py first.")
        st.info("""
        **Quick Start:**
        ```bash
        python fetch_real_data.py
        python multi_source_news_scraper.py
        streamlit run enhanced_dashboard_v3.py
        ```
        """)
        st.stop()
    
    # Ticker selection
    st.sidebar.subheader("📈 Analysis")
    available_tickers = sorted(list(returns.columns))
    selected_ticker = st.sidebar.selectbox("Select Ticker", available_tickers, index=available_tickers.index('AAPL') if 'AAPL' in available_tickers else 0)
    
    # Benchmark
    benchmark_options = ['SPY', 'QQQ', 'IWM', 'DIA', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'AMD', 'JPM']
    benchmark_options = [b for b in benchmark_options if b in returns.columns]
    
    # Fallback if no default benchmarks found
    if not benchmark_options:
        benchmark_options = sorted(list(returns.columns))
        
    benchmark = st.sidebar.selectbox("Benchmark", benchmark_options, index=0)
    
    if benchmark is None:
        st.error("❌ No benchmark selected or available.")
        st.stop()
    
    # Model settings
    st.sidebar.subheader("🔧 Model Settings")
    risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.5) / 100
    transaction_cost = st.sidebar.slider("Transaction Cost (bps)", 0, 50, 10, 5)
    
    # Initialize model
    use_model = st.sidebar.checkbox("Use Full Model", value=True)
    
    model = None
    if use_model and MODEL_AVAILABLE:
        config = ModelConfig(
            risk_free_rate=risk_free_rate,
            transaction_cost_bps=transaction_cost,
            bootstrap_samples=500
        )
        with st.spinner("Loading model..."):
            model = load_model(trading_days, config)
    
    # =================================================================
    # STATUS BAR
    # =================================================================
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📊 Assets", len(returns.columns))
    with col2:
        st.metric("📅 Trading Days", len(returns))
    with col3:
        date_range = f"{returns.index[0].strftime('%Y-%m')}"
        st.metric("📆 Start", date_range)
    with col4:
        sentiment_status = "✅" if sentiment is not None else "❌"
        st.metric("💭 Sentiment", sentiment_status)
    with col5:
        model_status = "✅" if model is not None else "❌"
        st.metric("🔧 Model", model_status)
    
    st.divider()
    
    # =================================================================
    # MAIN TABS
    # =================================================================
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "💰 Performance",
        "💭 Sentiment",
        "📈 Dynamic Beta",
        "📊 Volatility",
        "🔬 Statistics",
        "📋 Report",
        "🔬 Research"
    ])
    
    # Calculate strategy returns
    if model is not None:
        weights = model.get_optimal_weights()
        strategy_returns = model.run_backtest(weights)
    else:
        # Equal weight
        weights = pd.Series(1.0/len(returns.columns), index=returns.columns)
        strategy_returns = (returns * weights).sum(axis=1)
    
    benchmark_returns = returns[benchmark]
    
    # ----------------------------
    # TAB 1: Performance
    # ----------------------------
    with tab1:
        st.header("Portfolio Performance")
        
        # Cumulative returns
        fig_cum = plot_cumulative_returns(strategy_returns, benchmark_returns, benchmark)
        st.plotly_chart(fig_cum, use_container_width=True)
        
        # Metrics
        rf_daily = risk_free_rate / 252
        
        # Strategy metrics
        strat_total = (1 + strategy_returns).prod() - 1
        strat_vol = strategy_returns.std() * np.sqrt(252)
        strat_sharpe = (strategy_returns.mean() - rf_daily) / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        strat_cum = (1 + strategy_returns).cumprod()
        strat_dd = (strat_cum / strat_cum.cummax() - 1).min()
        
        # Benchmark metrics
        bench_total = (1 + benchmark_returns).prod() - 1
        bench_sharpe = (benchmark_returns.mean() - rf_daily) / benchmark_returns.std() * np.sqrt(252) if benchmark_returns.std() > 0 else 0
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Return", f"{strat_total:.2%}", f"{strat_total - bench_total:+.2%}")
        with col2:
            st.metric("Sharpe Ratio", f"{strat_sharpe:.2f}", f"{strat_sharpe - bench_sharpe:+.2f}")
        with col3:
            st.metric("Volatility", f"{strat_vol:.2%}")
        with col4:
            st.metric("Max Drawdown", f"{strat_dd:.2%}")
        with col5:
            n_years = len(strategy_returns) / 252
            cagr = (1 + strat_total) ** (1/n_years) - 1 if n_years > 0 else 0
            st.metric("CAGR", f"{cagr:.2%}")
        
        # Drawdown chart
        st.subheader("Drawdown")
        fig_dd = plot_drawdown(strategy_returns)
        st.plotly_chart(fig_dd, use_container_width=True)
        
        # Top holdings
        st.subheader("Top Holdings")
        top_weights = weights.nlargest(10)
        fig_weights = px.bar(
            x=top_weights.values * 100,
            y=top_weights.index,
            orientation='h',
            labels={'x': 'Weight (%)', 'y': 'Ticker'}
        )
        fig_weights.update_layout(template='plotly_white', height=350)
        st.plotly_chart(fig_weights, use_container_width=True)
    
    # ----------------------------
    # TAB 2: Sentiment
    # ----------------------------
    with tab2:
        st.header("FinBERT Sentiment Analysis")
        
        if sentiment is not None:
            # Market sentiment
            st.subheader("📊 Market Sentiment")
            fig_market = plot_sentiment_signal(sentiment, 'MARKET_SENTIMENT')
            if fig_market:
                st.plotly_chart(fig_market, use_container_width=True)
            
            # Current market sentiment
            if 'MARKET_SENTIMENT' in sentiment.columns:
                current_mkt = sentiment['MARKET_SENTIMENT'].dropna().iloc[-1] if len(sentiment['MARKET_SENTIMENT'].dropna()) > 0 else 0
                sent_emoji = "🟢" if current_mkt > 0.1 else "🔴" if current_mkt < -0.1 else "🟡"
                
                st.markdown(f"""
                <div class="info-box">
                    <h4>Current Market Sentiment: {sent_emoji} {current_mkt:.3f}</h4>
                </div>
                """, unsafe_allow_html=True)
            
            # Ticker sentiment
            st.subheader(f"📈 {selected_ticker} Sentiment")
            if selected_ticker in sentiment.columns:
                fig_ticker = plot_sentiment_signal(sentiment, selected_ticker)
                if fig_ticker:
                    st.plotly_chart(fig_ticker, use_container_width=True)
            else:
                st.warning(f"No sentiment data for {selected_ticker}")
            
            # Sentiment statistics
            st.subheader("📊 Sentiment Statistics")
            col1, col2, col3 = st.columns(3)
            
            mkt_sent = sentiment['MARKET_SENTIMENT'].dropna() if 'MARKET_SENTIMENT' in sentiment.columns else pd.Series()
            
            with col1:
                st.metric("Average", f"{mkt_sent.mean():.3f}" if len(mkt_sent) > 0 else "N/A")
            with col2:
                st.metric("Std Dev", f"{mkt_sent.std():.3f}" if len(mkt_sent) > 0 else "N/A")
            with col3:
                pct_positive = (mkt_sent > 0.1).mean() if len(mkt_sent) > 0 else 0
                st.metric("% Positive Days", f"{pct_positive:.1%}")
        else:
            st.warning("⚠️ Sentiment data not loaded.")
            st.info("""
            **To collect real sentiment data:**
            ```bash
            python multi_source_news_scraper.py --trading-days 504
            ```
            
            This will scrape news from multiple sources:
            - Finviz (primary)
            - Yahoo Finance
            - Google News RSS
            - MarketWatch
            """)
    
    # ----------------------------
    # TAB 3: Dynamic Beta
    # ----------------------------
    with tab3:
        st.header("Dynamic Factor Loading (Kalman Filter)")
        
        if model is not None:
            fig_beta = plot_dynamic_beta(model, selected_ticker)
            if fig_beta:
                st.plotly_chart(fig_beta, use_container_width=True)
            
            # Current betas
            current_betas = model.layer1_factor.get_current_betas()
            current_beta = current_betas.get(selected_ticker, 1.0)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Beta", f"{current_beta:.3f}")
            with col2:
                avg_beta = np.mean(list(current_betas.values()))
                st.metric("Avg Beta (All)", f"{avg_beta:.3f}")
            with col3:
                beta_series = model.get_dynamic_beta_series(selected_ticker)
                if not beta_series.empty:
                    st.metric("Beta Range", f"{beta_series.min():.2f} - {beta_series.max():.2f}")
            with col4:
                if not beta_series.empty:
                    st.metric("Beta Volatility", f"{beta_series.std():.3f}")
            
            # Beta distribution
            st.subheader("Beta Distribution (All Assets)")
            fig_hist = px.histogram(
                x=list(current_betas.values()),
                nbins=30,
                labels={'x': 'Beta', 'y': 'Count'}
            )
            fig_hist.add_vline(x=1, line_dash="dash", line_color="red")
            fig_hist.update_layout(template='plotly_white', height=300)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("⚠️ Model not loaded. Enable 'Use Full Model' in sidebar.")
    
    # ----------------------------
    # TAB 4: Volatility
    # ----------------------------
    with tab4:
        st.header("GARCH-MIDAS Volatility")
        
        if model is not None:
            fig_vol = plot_volatility(model, selected_ticker)
            if fig_vol:
                st.plotly_chart(fig_vol, use_container_width=True)
            
            # Volatility forecast
            if selected_ticker in model.layer3_volatility:
                garch = model.layer3_volatility[selected_ticker]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    forecast_1d = garch.forecast(1) * np.sqrt(252) * 100
                    st.metric("1-Day Forecast", f"{forecast_1d:.2f}%")
                with col2:
                    forecast_5d = garch.forecast(5) * np.sqrt(252) * 100
                    st.metric("5-Day Forecast", f"{forecast_5d:.2f}%")
                with col3:
                    forecast_22d = garch.forecast(22) * np.sqrt(252) * 100
                    st.metric("1-Month Forecast", f"{forecast_22d:.2f}%")
        else:
            st.warning("⚠️ Model not loaded.")
    
    # ----------------------------
    # TAB 5: Statistics (NEW)
    # ----------------------------
    with tab5:
        st.header("🔬 Statistical Validation")
        
        if model is not None:
            # Run validation
            with st.spinner("Running statistical validation..."):
                validation = model.validate(
                    strategy_returns,
                    benchmark_returns,
                    run_bootstrap=True,
                    n_subperiods=4
                )
            
            # Alpha/Beta Results
            st.subheader("📐 Alpha & Beta Analysis")
            ab = validation['alpha_beta']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Alpha (Annual)", f"{ab['alpha_annual']:.2%}")
            with col2:
                st.metric("Alpha t-stat", f"{ab['alpha_tstat']:.3f}")
            with col3:
                st.metric("Alpha p-value", f"{ab['alpha_pval']:.4f}")
            with col4:
                st.metric("Beta", f"{ab['beta']:.3f}")
            
            # Significance box
            if ab['significant']:
                st.markdown(f"""
                <div class="success-box">
                    <h4>✅ STATISTICALLY SIGNIFICANT</h4>
                    <p>Alpha is significantly different from zero at 95% confidence level.</p>
                    <p>95% CI: [{ab['alpha_ci_lower']:.2%}, {ab['alpha_ci_upper']:.2%}]</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="danger-box">
                    <h4>❌ NOT STATISTICALLY SIGNIFICANT</h4>
                    <p>Cannot reject the null hypothesis that alpha = 0.</p>
                    <p>95% CI: [{ab['alpha_ci_lower']:.2%}, {ab['alpha_ci_upper']:.2%}]</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Bootstrap Analysis
            st.subheader("🎲 Bootstrap Analysis")
            boot = validation['bootstrap']
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **Alpha Bootstrap:**
                - Mean: {boot['alpha_mean']:.2%}
                - Std: {boot['alpha_std']:.2%}
                - 95% CI: [{boot['alpha_ci_lower']:.2%}, {boot['alpha_ci_upper']:.2%}]
                - Zero in CI: {'❌ Yes' if boot['alpha_zero_in_ci'] else '✅ No'}
                """)
            with col2:
                st.markdown(f"""
                **Sharpe Bootstrap:**
                - Mean: {boot['sharpe_mean']:.3f}
                - Std: {boot['sharpe_std']:.3f}
                - 95% CI: [{boot['sharpe_ci_lower']:.3f}, {boot['sharpe_ci_upper']:.3f}]
                - Zero in CI: {'❌ Yes' if boot['sharpe_zero_in_ci'] else '✅ No'}
                """)
            
            # Bootstrap distribution
            fig_boot = plot_bootstrap_distribution(boot)
            st.plotly_chart(fig_boot, use_container_width=True)
            
            st.divider()
            
            # Subperiod Analysis
            st.subheader("📊 Subperiod Robustness")
            subperiods = validation['subperiods']
            
            positive_sharpe = sum(1 for p in subperiods if p['sharpe_ratio'] > 0)
            significant_alpha = sum(1 for p in subperiods if p['significant'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Positive Sharpe", f"{positive_sharpe}/4")
            with col2:
                st.metric("Significant Alpha", f"{significant_alpha}/4")
            
            fig_sub = plot_subperiod_analysis(subperiods)
            st.plotly_chart(fig_sub, use_container_width=True)
            
            # Subperiod table
            sub_df = pd.DataFrame(subperiods)
            sub_df['start_date'] = sub_df['start_date'].dt.strftime('%Y-%m-%d')
            sub_df['end_date'] = sub_df['end_date'].dt.strftime('%Y-%m-%d')
            sub_df['total_return'] = sub_df['total_return'].apply(lambda x: f"{x:.2%}")
            sub_df['sharpe_ratio'] = sub_df['sharpe_ratio'].apply(lambda x: f"{x:.3f}")
            sub_df['alpha_annual'] = sub_df['alpha_annual'].apply(lambda x: f"{x:.2%}")
            sub_df['alpha_pval'] = sub_df['alpha_pval'].apply(lambda x: f"{x:.4f}")
            sub_df['significant'] = sub_df['significant'].apply(lambda x: "✅" if x else "❌")
            
            st.dataframe(sub_df[['period', 'start_date', 'end_date', 'total_return', 'sharpe_ratio', 'alpha_annual', 'alpha_pval', 'significant']], use_container_width=True)
            
            st.divider()
            
            # Transaction Cost Sensitivity
            st.subheader("💰 Transaction Cost Sensitivity")
            tc = validation['transaction_costs']
            
            col1, col2 = st.columns(2)
            with col1:
                if tc['breakeven_cost_bps']:
                    st.metric("Breakeven Cost", f"{tc['breakeven_cost_bps']:.0f} bps")
                else:
                    st.metric("Breakeven Cost", "> 50 bps")
            with col2:
                robust_status = "✅ Yes" if tc['robust_at_50bps'] else "❌ No"
                st.metric("Robust at 50bps", robust_status)
            
            fig_tc = plot_transaction_cost_sensitivity(tc)
            st.plotly_chart(fig_tc, use_container_width=True)
            
        else:
            st.warning("⚠️ Model not loaded. Enable 'Use Full Model' in sidebar.")
            st.info("The Statistics tab provides comprehensive statistical validation including HAC standard errors, bootstrap confidence intervals, subperiod analysis, and transaction cost sensitivity.")
    
    # ----------------------------
    # TAB 6: Report
    # ----------------------------
    with tab6:
        st.header("📋 Summary Report")
        
        if model is not None:
            summary = model.summary()
            perf = validation['performance']
            ab = validation['alpha_beta']
            
            report = f"""# AI Finance Project Report
## Weakness Compensation Framework

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Configuration
- **Trading Days:** {len(strategy_returns)}
- **Date Range:** {strategy_returns.index[0].date()} to {strategy_returns.index[-1].date()}
- **Assets:** {summary['n_assets']}
- **Benchmark:** {benchmark}
- **Risk-Free Rate:** {risk_free_rate:.2%}
- **Transaction Cost:** {transaction_cost} bps

---

## Performance Summary

| Metric | Strategy | {benchmark} | Difference |
|--------|----------|-------------|------------|
| Total Return | {perf['total_return']:.2%} | {bench_total:.2%} | {perf['total_return']-bench_total:+.2%} |
| CAGR | {perf['cagr']:.2%} | - | - |
| Sharpe Ratio | {perf['sharpe_ratio']:.3f} | {bench_sharpe:.3f} | {perf['sharpe_ratio']-bench_sharpe:+.3f} |
| Max Drawdown | {perf['max_drawdown']:.2%} | - | - |
| Volatility | {perf['volatility']:.2%} | - | - |

---

## Statistical Significance

| Metric | Value |
|--------|-------|
| Alpha (Annual) | {ab['alpha_annual']:.2%} |
| Alpha t-statistic | {ab['alpha_tstat']:.3f} |
| Alpha p-value | {ab['alpha_pval']:.4f} |
| Alpha 95% CI | [{ab['alpha_ci_lower']:.2%}, {ab['alpha_ci_upper']:.2%}] |
| Beta | {ab['beta']:.3f} |
| R-squared | {ab['r_squared']:.2%} |

**Verdict:** {'✅ STATISTICALLY SIGNIFICANT' if ab['significant'] else '❌ NOT SIGNIFICANT'}

---

## Model Components

### Layer 1: Dynamic Factor Loading (Kalman Filter)
- Time-varying beta estimation
- Average Current Beta: {summary['avg_beta']:.3f}

### Layer 2: Sentiment Integration
- Source: FinBERT (ProsusAI/finbert)
- Status: {'Active' if summary['sentiment_loaded'] else 'Inactive'}

### Layer 3: GARCH-MIDAS Volatility
- Short-term: GARCH(1,1)
- Long-term: MIDAS (22-day)

---

## Robustness Tests

### Subperiod Analysis
- Positive Sharpe: {sum(1 for p in validation['subperiods'] if p['sharpe_ratio'] > 0)}/4
- Significant Alpha: {sum(1 for p in validation['subperiods'] if p['significant'])}/4

### Transaction Cost Sensitivity
- Breakeven Cost: {f"{validation['transaction_costs']['breakeven_cost_bps']:.0f}" if validation['transaction_costs']['breakeven_cost_bps'] is not None else ">50"} bps
- Robust at 50bps: {'Yes' if validation['transaction_costs']['robust_at_50bps'] else 'No'}

### Bootstrap Analysis ({validation['bootstrap']['n_samples']} samples)
- Alpha 95% CI: [{validation['bootstrap']['alpha_ci_lower']:.2%}, {validation['bootstrap']['alpha_ci_upper']:.2%}]
- Sharpe 95% CI: [{validation['bootstrap']['sharpe_ci_lower']:.3f}, {validation['bootstrap']['sharpe_ci_upper']:.3f}]

---

*Report generated by AI Finance Dashboard v3*
"""
            
            st.markdown(report)
            
            # Download button
            st.download_button(
                "📥 Download Report (Markdown)",
                report,
                file_name=f"ai_finance_report_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
        else:
            st.warning("Load the model to generate report.")


    # ----------------------------
    # TAB 7: Research (NEW)
    # ----------------------------
    with tab7:
        st.header("🔬 Research Pipeline V2 (Duke Feedback)")
        st.markdown("Run advanced cross-sectional tests and decile portfolio analysis with 4-regime framework.")

        if model is not None and RESEARCH_AVAILABLE:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Configuration")
                n_deciles = st.slider("Number of Deciles", 2, 10, 10)
                min_assets = st.slider("Min Assets per Date", 10, 50, 10)
                
                st.markdown("**Active Layers:**")
                use_beta = st.checkbox("Dynamic Beta", value=True)
                use_sent = st.checkbox("Sentiment", value=True)
                use_vol = st.checkbox("GARCH-MIDAS Vol", value=True)
                
                st.markdown("**Advanced Options:**")
                use_zscore = st.checkbox("Standardize Features (Z-Score)", value=False, help="Use z-scores for Fama-MacBeth regressions")
                use_vol_regime = st.checkbox("Use Vol-Regime Conditioned Sentiment", value=False, help="Interact sentiment with Low/Normal/High volatility regimes")
                st.markdown("**Strategic Portfolio Benchmarks:**")
                use_heston = st.checkbox("Hest - Heston-style Sentiment LS", value=False, help="Daily sort on lagged sentiment")
                use_tetlock = st.checkbox("Tetl - Tetlock Abnormal Pessimism LS", value=False, help="Sort on abnormal negative sentiment (vs 60d mean)")
                use_gambarelli = st.checkbox("Gmbl - Gambarelli Monthly Sentiment LS", value=False, help="Monthly sort on smoothed sentiment")
                use_risk = st.checkbox("Risk - Beta+Vol Predicted Return LS", value=False, help="Sort on predicted return from Beta and Volatility")
                use_momentum = st.checkbox("Mmt - Momentum 12-2 LS", value=False, help="Sort on 12-month momentum (skipping last month)")
                
                factor_model = st.selectbox("Validation Model", ["CAPM", "FF5", "FF5+MOM"])
                factor_file = st.text_input("Factor File (CSV)", "F-F_Research_Data_Factors_daily.csv")
                
                run_research = st.button("🚀 Run Research Analysis", type="primary")
                
                st.divider()
                st.subheader("Experiments")
                run_ablation = st.button("🧪 Run Ablation Study")
                
                st.subheader("Subperiod Analysis")
                sub_cutoffs = st.text_input("Date Cutoffs (comma sep)", "2023-01-01")
                run_subperiods = st.button("📅 Run Subperiod Test")
            
            with col2:
                if run_research:
                    with st.spinner("Running Fama-MacBeth and Decile Backtests..."):
                        # Setup design
                        design = ResearchDesign(
                            n_deciles=n_deciles,
                            min_assets_per_date=min_assets,
                            use_dynamic_beta=use_beta,
                            use_sentiment=use_sent,
                            use_midas_vol=use_vol,
                            use_zscore_fm=use_zscore,
                            use_vol_regime=use_vol_regime,
                            use_heston_sentiment=use_heston,
                            use_tetlock_sentiment=use_tetlock,
                            use_gambarelli_sentiment=use_gambarelli,
                            use_risk_benchmark=use_risk,
                            use_momentum_benchmark=use_momentum,
                            factor_file=factor_file
                        )
                        
                        # 1. Build Panel
                        tester = CrossSectionalTester(model, design)
                        try:
                            panel = tester.build_panel()
                            st.success(f"✅ Panel built: {len(panel)} rows")
                            
                            # 2. Fama-MacBeth
                            if use_vol_regime:
                                panel = tester.compute_vol_regimes(panel)
                                panel = tester.add_regime_interactions(panel, standardize_sent=use_zscore)
                                fm_cols = ["beta_dyn", "vol_midas", "sent_low", "sent_norm", "sent_high"]
                                
                                if use_zscore:
                                    # Standardize others
                                    panel_fm = tester.standardize_panel(panel, ["beta_dyn", "vol_midas"])
                                else:
                                    panel_fm = panel
                                    
                                fm_res = tester.fama_macbeth(panel_fm, feature_cols=fm_cols)
                            else:
                                if use_zscore:
                                    panel_fm = tester.standardize_panel(panel, ["beta_dyn", "sentiment", "vol_midas"])
                                else:
                                    panel_fm = panel
                                fm_res = tester.fama_macbeth(panel_fm)
                            
                            st.subheader("1. Fama-MacBeth Regressions")
                            if use_zscore:
                                st.caption("Coefficients represent change in return per 1 SD increase in feature.")
                            if use_vol_regime:
                                st.info("Showing interactions: Sentiment x Volatility Regime (Low < -0.5 SD, High > +0.5 SD)")
                            
                            fm_df = pd.DataFrame({
                                "Mean Coef": fm_res['mean_coefs'],
                                "T-Stat": fm_res['tstats']
                            })
                            st.dataframe(fm_df.style.format("{:.4f}").applymap(
                                lambda x: "color: green; font-weight: bold" if abs(x) > 1.96 else "color: red", 
                                subset=["T-Stat"]
                            ))
                            
                            # 3. Decile Backtest
                            # Calculate score
                            feats = []
                            if use_beta: feats.append("beta_dyn")
                            if use_sent: feats.append("sentiment")
                            if use_vol: feats.append("vol_midas")
                            
                            for f in feats:
                                if f in panel.columns:
                                    mu = panel[f].mean()
                                    sigma = panel[f].std()
                                    panel[f+"_z"] = (panel[f] - mu) / sigma if sigma > 0 else 0.0
                            
                            z_cols = [f+"_z" for f in feats if f+"_z" in panel.columns]
                            if z_cols:
                                panel["score"] = panel[z_cols].mean(axis=1)
                            else:
                                panel["score"] = panel["ret_fwd"]
                                
                            dpt = DecilePortfolioTester(model, design)
                            deciles, long_short = dpt.decile_backtest(panel, score_col="score")
                            
                            st.subheader("2. Decile Portfolios")
                            
                            # Plot Long-Short
                            fig_ls = go.Figure()
                            ls_cum = (1 + long_short).cumprod()
                            bench_cum = (1 + model.market_returns.loc[long_short.index]).cumprod()
                            
                            fig_ls.add_trace(go.Scatter(x=ls_cum.index, y=ls_cum.values, name="Main Strategy LS", line=dict(color="green", width=2)))
                            fig_ls.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values, name="Benchmark", line=dict(color="gray", dash="dash")))
                            
                            # Strategic Benchmarks
                            strategic_res = {}
                            if use_heston: strategic_res["Hest"] = dpt.build_heston_sentiment_ls(panel)
                            if use_tetlock: strategic_res["Tetl"] = dpt.build_tetlock_ls(panel)
                            if use_gambarelli: strategic_res["Gmbl"] = dpt.build_gambarelli_ls(panel)
                            if use_risk: strategic_res["Risk"] = dpt.build_risk_ls(panel)
                            if use_momentum: strategic_res["Mmt"] = dpt.build_momentum_ls(panel)
                            
                            # Plot Strategic Benchmarks
                            colors = {"Hest": "orange", "Tetl": "purple", "Gmbl": "brown", "Risk": "red", "Mmt": "blue"}
                            for name, res in strategic_res.items():
                                if "ls" in res and not res["ls"].empty:
                                    s_ls = res["ls"]
                                    s_cum = (1 + s_ls).cumprod()
                                    fig_ls.add_trace(go.Scatter(
                                        x=s_cum.index, 
                                        y=s_cum.values, 
                                        name=f"{name} LS", 
                                        line=dict(color=colors.get(name, "black"), dash="dot", width=2)
                                    ))
                            
                            fig_ls.update_layout(title="Long-Short Strategy Performance", template="plotly_white", height=400)
                            st.plotly_chart(fig_ls, use_container_width=True)
                            
                            # Stats
                            from research_pipeline_v1 import FactorValidator
                            fv = FactorValidator(factor_file)
                            val_res = fv.run_regression(long_short, model_type=factor_model, market_returns=model.market_returns)
                            
                            if "error" in val_res:
                                st.error(f"Validation Error: {val_res['error']}")
                            else:
                                c1, c2, c3 = st.columns(3)
                                c1.metric("Annual Alpha", f"{val_res['alpha_annual']:.2%}")
                                c2.metric("Alpha T-Stat", f"{val_res['alpha_tstat']:.2f}")
                                c3.metric("R-Squared", f"{val_res['r_squared']:.3f}")
                                
                                if strategic_res:
                                    st.markdown("---")
                                    st.markdown("**Strategic Portfolio Benchmarks**")
                                    
                                    # Create a summary table
                                    strat_summary = []
                                    for name, res in strategic_res.items():
                                        if "ls" in res and not res["ls"].empty:
                                            s_ls = res["ls"]
                                            val = fv.run_regression(s_ls, model_type=factor_model, market_returns=model.market_returns)
                                            if "error" not in val:
                                                strat_summary.append({
                                                    "Strategy": name,
                                                    "Alpha": val['alpha_annual'],
                                                    "T-Stat": val['alpha_tstat'],
                                                    "Sharpe": (s_ls.mean() / s_ls.std() * np.sqrt(252)) if s_ls.std() > 0 else 0.0
                                                })
                                    
                                    if strat_summary:
                                        st.dataframe(pd.DataFrame(strat_summary).style.format({
                                            "Alpha": "{:.2%}",
                                            "T-Stat": "{:.2f}",
                                            "Sharpe": "{:.2f}"
                                        }))
                                    params_df = pd.DataFrame({
                                        "Coef": val_res['params'],
                                        "T-Stat": val_res['tvalues']
                                    })
                                    st.dataframe(params_df.style.format("{:.4f}"))
                                    
                            # Regime Conditional Deciles
                            if use_vol_regime:
                                st.subheader("3. Regime-Conditional Portfolios")
                                regime_res = dpt.run_regime_conditional_backtest(panel, score_col="score")
                                
                                r_cols = st.columns(3)
                                for i, (regime, data) in enumerate(regime_res.items()):
                                    with r_cols[i]:
                                        st.markdown(f"**{regime}**")
                                        if "error" in data:
                                            st.error(data["error"])
                                        else:
                                            stats = data["stats"]
                                            st.metric("Alpha", f"{stats['alpha_annual']:.2%}")
                                            st.metric("T-Stat", f"{stats['alpha_tstat']:.2f}")
                                            st.metric("Days", data["n_days"])
                                            
                                            # Mini chart
                                            ls_r = data["long_short"]
                                            st.line_chart((1 + ls_r).cumprod())
                            
                        except Exception as e:
                            st.error(f"Error running research: {e}")
                            st.exception(e)
 
                elif run_ablation:
                     with st.spinner("Running Ablation Study..."):
                        design = ResearchDesign(
                            n_deciles=n_deciles,
                            min_assets_per_date=min_assets,
                            factor_file=factor_file
                        )
                        ablation = AblationTester(model, design)
                        try:
                            res_df = ablation.run_ablation()
                            st.subheader("🧪 Ablation Results")
                            st.dataframe(res_df.style.format({
                                "Alpha": "{:.2%}",
                                "T-Stat": "{:.2f}",
                                "Sharpe": "{:.2f}"
                            }).highlight_max(subset=["Alpha", "Sharpe"], color="lightgreen"))
                        except Exception as e:
                            st.error(f"Error in ablation: {e}")
 
                elif run_subperiods:
                    with st.spinner("Running Subperiod Analysis..."):
                        design = ResearchDesign(
                            n_deciles=n_deciles,
                            min_assets_per_date=min_assets,
                            use_dynamic_beta=use_beta,
                            use_sentiment=use_sent,
                            use_midas_vol=use_vol,
                            factor_file=factor_file
                        )
                        sub_tester = SubperiodTester(model, design)
                        try:
                            cutoffs = [d.strip() for d in sub_cutoffs.split(",")]
                            res_df = sub_tester.run_subperiods(cutoffs)
                            st.subheader("📅 Subperiod Results")
                            st.dataframe(res_df.style.format({
                                "Alpha": "{:.2%}",
                                "T-Stat": "{:.2f}",
                                "Sharpe": "{:.2f}"
                            }))
                        except Exception as e:
                            st.error(f"Error in subperiods: {e}")

        elif not RESEARCH_AVAILABLE:
            st.error("Research pipeline module not found.")
        else:
            st.warning("Please load the model first.")


if __name__ == "__main__":
    main()
