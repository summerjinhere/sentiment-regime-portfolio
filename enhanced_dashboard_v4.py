"""
Enhanced Dashboard v4 - 4-Regime Framework
============================================

Duke Feedback Incorporated:
- 4-Regime Analysis (Vol × Market Direction)
- Train/Test Split Display
- FF5 + Momentum Results
- Ablation Study Visualization
- Regime Sensitivity Charts
- Strategic Benchmark Comparison

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
    from enhanced_integrated_model_v3 import (
        EnhancedIntegratedModel, 
        ModelConfig,
        StatisticalValidator,
        PerformanceCalculator,
        MarketRegime
    )
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

try:
    from research_pipeline_v2 import (
        ResearchDesignV2,
        CrossSectionalTesterV2,
        DecilePortfolioTesterV2,
        AblationTesterV2,
        FactorValidatorV2,
        RegimeSensitivityAnalyzer,
        MultipleTestingAdjustment,
        run_research_v2
    )
    RESEARCH_AVAILABLE = True
except ImportError:
    RESEARCH_AVAILABLE = False


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="AI Finance v2 - 4-Regime Framework",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .regime-lb { background: #74b9ff; color: white; padding: 10px; border-radius: 5px; text-align: center; margin: 5px 0; }
    .regime-lu { background: #55efc4; color: white; padding: 10px; border-radius: 5px; text-align: center; margin: 5px 0; }
    .regime-hb { background: #fd79a8; color: white; padding: 10px; border-radius: 5px; text-align: center; margin: 5px 0; }
    .regime-hu { background: #ffeaa7; color: black; padding: 10px; border-radius: 5px; text-align: center; margin: 5px 0; }
    .success-box { background-color: #d4edda; border-left: 5px solid #00C853; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .warning-box { background-color: #fff3cd; border-left: 5px solid #FFB300; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .danger-box { background-color: #f8d7da; border-left: 5px solid #FF5252; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .info-box { background-color: #e7f3ff; border-left: 5px solid #2196F3; padding: 15px; border-radius: 5px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(ttl=3600)
def load_price_data(filepath: str, trading_days: int = None):
    try:
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        ret_cols = [c for c in data.columns if c.endswith('_ret')]
        returns = data[ret_cols].copy()
        returns.columns = [c.replace('_ret', '') for c in ret_cols]
        
        price_cols = [c for c in data.columns if c.endswith('_price')]
        prices = data[price_cols].copy()
        prices.columns = [c.replace('_price', '') for c in price_cols]
        
        if trading_days and len(returns) > trading_days:
            returns = returns.iloc[-trading_days:]
            prices = prices.iloc[-trading_days:]
        
        return prices, returns
    except Exception as e:
        return None, None


@st.cache_data(ttl=3600)
def load_sentiment_data(filepath: str, trading_days: int = None):
    try:
        sentiment = pd.read_csv(filepath, index_col=0, parse_dates=True)
        if trading_days and len(sentiment) > trading_days:
            sentiment = sentiment.iloc[-trading_days:]
        return sentiment
    except:
        return None


@st.cache_resource
def load_model(trading_days: int, _config: ModelConfig):
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

def plot_cumulative_returns(strategy_returns: pd.Series, benchmark_returns: pd.Series, benchmark_name: str = 'SPY'):
    strategy_cum = (1 + strategy_returns).cumprod()
    benchmark_cum = (1 + benchmark_returns).cumprod()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strategy_cum.index, y=strategy_cum.values, name='Strategy', 
                             line=dict(color='#00C853', width=2.5), fill='tozeroy', fillcolor='rgba(0, 200, 83, 0.1)'))
    fig.add_trace(go.Scatter(x=benchmark_cum.index, y=benchmark_cum.values, name=benchmark_name, 
                             line=dict(color='#2196F3', width=2, dash='dash')))
    fig.update_layout(title='Cumulative Returns', xaxis_title='Date', yaxis_title='Growth of $1',
                      template='plotly_white', height=450, hovermode='x unified')
    return fig


def plot_drawdown(returns: pd.Series):
    cum = (1 + returns).cumprod()
    drawdown = cum / cum.cummax() - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values * 100, fill='tozeroy', 
                             fillcolor='rgba(255, 82, 82, 0.3)', line=dict(color='#FF5252', width=1), name='Drawdown'))
    fig.update_layout(title='Drawdown', xaxis_title='Date', yaxis_title='Drawdown (%)', template='plotly_white', height=300)
    return fig


def plot_4regime_distribution(regimes: pd.DataFrame):
    regime_counts = regimes['regime'].value_counts()
    colors = ['#74b9ff', '#55efc4', '#fd79a8', '#ffeaa7']
    fig = go.Figure(data=[go.Pie(labels=regime_counts.index, values=regime_counts.values, marker_colors=colors, hole=0.4)])
    fig.update_layout(title='4-Regime Distribution', template='plotly_white', height=350)
    return fig


def plot_regime_performance(regime_results: Dict):
    regimes, sharpes, colors = [], [], []
    color_map = {'LowVol_Bear': '#74b9ff', 'LowVol_Bull': '#55efc4', 'HighVol_Bear': '#fd79a8', 'HighVol_Bull': '#ffeaa7'}
    
    for regime, res in regime_results.items():
        if "sharpe" in res:
            regimes.append(regime)
            sharpes.append(res['sharpe'])
            colors.append(color_map.get(regime, '#999999'))
    
    fig = go.Figure(data=[go.Bar(x=regimes, y=sharpes, marker_color=colors)])
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(title='Sharpe Ratio by Regime', xaxis_title='Regime', yaxis_title='Sharpe Ratio',
                      template='plotly_white', height=350)
    return fig


def plot_ablation_results(ablation_df: pd.DataFrame):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Alpha by Configuration', 'Sharpe by Configuration'))
    
    colors = ['#00C853' if a > 0 else '#FF5252' for a in ablation_df['Alpha'].fillna(0)]
    fig.add_trace(go.Bar(x=ablation_df['Configuration'], y=ablation_df['Alpha'] * 100, marker_color=colors, name='Alpha (%)'), row=1, col=1)
    
    sharpe_colors = ['#00C853' if s > 0 else '#FF5252' for s in ablation_df['Sharpe'].fillna(0)]
    fig.add_trace(go.Bar(x=ablation_df['Configuration'], y=ablation_df['Sharpe'], marker_color=sharpe_colors, name='Sharpe'), row=1, col=2)
    
    fig.update_layout(template='plotly_white', height=400, showlegend=False)
    fig.update_xaxes(tickangle=45)
    return fig


def plot_fama_macbeth_coefficients(fm_result: Dict):
    coefs = fm_result['mean_coefs']
    tstats = fm_result['tstats']
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Coefficients', 'T-Statistics'))
    colors = ['#00C853' if abs(t) > 2 else '#999999' for t in tstats.values]
    
    fig.add_trace(go.Bar(x=coefs.index, y=coefs.values, marker_color=colors, name='Coefficient'), row=1, col=1)
    fig.add_trace(go.Bar(x=tstats.index, y=tstats.values, marker_color=colors, name='T-Stat'), row=1, col=2)
    
    fig.add_hline(y=2, line_dash="dot", line_color="green", row=1, col=2)
    fig.add_hline(y=-2, line_dash="dot", line_color="green", row=1, col=2)
    fig.update_layout(template='plotly_white', height=350, showlegend=False)
    fig.update_xaxes(tickangle=45)
    return fig


def plot_regime_sensitivity(sensitivity_df: pd.DataFrame):
    fig = go.Figure()
    for col in ['sent_LB_tstat', 'sent_LU_tstat', 'sent_HB_tstat', 'sent_HU_tstat']:
        if col in sensitivity_df.columns:
            fig.add_trace(go.Scatter(x=sensitivity_df['vol_threshold'], y=sensitivity_df[col],
                                     mode='lines+markers', name=col.replace('_tstat', '').replace('sent_', '')))
    
    fig.add_hline(y=2, line_dash="dash", line_color="green", opacity=0.5)
    fig.add_hline(y=-2, line_dash="dash", line_color="green", opacity=0.5)
    fig.update_layout(title='Regime Sensitivity: T-Statistics by Vol Threshold', xaxis_title='Volatility Threshold (z-score)',
                      yaxis_title='T-Statistic', template='plotly_white', height=400)
    return fig


def plot_bootstrap_distribution(bootstrap_results: Dict):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Alpha Distribution', 'Sharpe Distribution'))
    
    alphas = bootstrap_results['bootstrap_alphas'] * 100
    fig.add_trace(go.Histogram(x=alphas, nbinsx=50, name='Alpha', marker_color='#667eea', opacity=0.7), row=1, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_vline(x=bootstrap_results['alpha_ci_lower']*100, line_dash="dot", line_color="green", row=1, col=1)
    fig.add_vline(x=bootstrap_results['alpha_ci_upper']*100, line_dash="dot", line_color="green", row=1, col=1)
    
    sharpes = bootstrap_results['bootstrap_sharpes']
    fig.add_trace(go.Histogram(x=sharpes, nbinsx=50, name='Sharpe', marker_color='#764ba2', opacity=0.7), row=1, col=2)
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_vline(x=bootstrap_results['sharpe_ci_lower'], line_dash="dot", line_color="green", row=1, col=2)
    fig.add_vline(x=bootstrap_results['sharpe_ci_upper'], line_dash="dot", line_color="green", row=1, col=2)
    
    fig.update_layout(template='plotly_white', height=350, showlegend=False)
    return fig


def plot_transaction_cost_sensitivity(tc_results: Dict):
    df = pd.DataFrame(tc_results['results'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['cost_bps'], y=df['sharpe_ratio'], mode='lines+markers', name='Sharpe Ratio',
                             line=dict(color='#667eea', width=2), marker=dict(size=8)))
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.7)
    
    breakeven = tc_results.get('breakeven_cost_bps')
    if breakeven and breakeven < float('inf'):
        fig.add_vline(x=breakeven, line_dash="dot", line_color="orange", annotation_text=f"Breakeven: {breakeven:.0f}bps")
    
    fig.update_layout(title='Transaction Cost Sensitivity', xaxis_title='Transaction Cost (bps)',
                      yaxis_title='Sharpe Ratio', template='plotly_white', height=350)
    return fig


def plot_sentiment_signal(sentiment: pd.DataFrame, ticker: str = 'MARKET_SENTIMENT'):
    if sentiment is None or ticker not in sentiment.columns:
        return None
    data = sentiment[ticker].dropna()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data.values, mode='lines', name='Sentiment', line=dict(color='#2196F3', width=1.5)))
    fig.add_trace(go.Scatter(x=data.index, y=np.where(data.values > 0, data.values, 0), fill='tozeroy',
                             fillcolor='rgba(0, 200, 83, 0.2)', line=dict(width=0), name='Positive', showlegend=False))
    fig.add_trace(go.Scatter(x=data.index, y=np.where(data.values < 0, data.values, 0), fill='tozeroy',
                             fillcolor='rgba(255, 82, 82, 0.2)', line=dict(width=0), name='Negative', showlegend=False))
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    fig.update_layout(title=f'FinBERT Sentiment Signal: {ticker}', xaxis_title='Date', yaxis_title='Sentiment Score',
                      template='plotly_white', height=400, yaxis=dict(range=[-1, 1]))
    return fig


def plot_dynamic_beta(model, ticker: str):
    if model is None:
        return None
    beta_series = model.get_dynamic_beta_series(ticker)
    if beta_series.empty:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=beta_series.index, y=beta_series.values, mode='lines', name='Dynamic Beta', line=dict(color='#9C27B0', width=2)))
    rolling = beta_series.rolling(22).mean()
    fig.add_trace(go.Scatter(x=rolling.index, y=rolling.values, mode='lines', name='22-Day MA', line=dict(color='#FF9800', width=1.5, dash='dot')))
    fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(title=f'Dynamic Beta (Kalman Filter): {ticker}', xaxis_title='Date', yaxis_title='Beta', template='plotly_white', height=400)
    return fig


def plot_volatility(model, ticker: str):
    if model is None:
        return None
    vol_series = model.get_volatility_series(ticker)
    if vol_series.empty:
        return None
    vol_annual = vol_series * np.sqrt(252) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vol_annual.index, y=vol_annual.values, mode='lines', name='Volatility',
                             line=dict(color='#FF5252', width=1.5), fill='tozeroy', fillcolor='rgba(255, 82, 82, 0.1)'))
    fig.update_layout(title=f'GARCH-MIDAS Volatility: {ticker}', xaxis_title='Date', yaxis_title='Annualized Volatility (%)',
                      template='plotly_white', height=350)
    return fig


# =============================================================================
# MAIN APP
# =============================================================================

def run_dashboard():
    st.markdown('<h1 class="main-header">🎯 AI Finance Dashboard v4</h1>', unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; color: #666; margin-bottom: 2rem;'>4-Regime Framework | Duke Feedback Incorporated | Train/Test Split</div>", unsafe_allow_html=True)
    
    # SIDEBAR
    st.sidebar.header("⚙️ Configuration")
    
    st.sidebar.subheader("📅 Trading Days")
    trading_days_preset = st.sidebar.selectbox("Preset", ["1 Year (252)", "2 Years (504)", "3 Years (756)", "5 Years (1260)", "10 Years (2520)", "20 Years (5040)", "All Data"], index=5)
    trading_days = None if trading_days_preset == "All Data" else int(trading_days_preset.split("(")[1].replace(")", ""))
    st.sidebar.markdown(f"**Selected:** {trading_days if trading_days else 'All'} days")
    
    st.sidebar.subheader("📁 Data Files")
    prices_file = st.sidebar.text_input("Price Data", "real_backtest_panel.csv")
    sentiment_file = st.sidebar.text_input("Sentiment Data", "real_sentiment_data.csv")
    
    with st.spinner("Loading data..."):
        prices, returns = load_price_data(prices_file, trading_days)
        sentiment = load_sentiment_data(sentiment_file, trading_days)
    
    if returns is None:
        st.error("❌ Could not load price data. Please run fetch_real_data.py first.")
        st.stop()
    
    st.sidebar.subheader("📈 Analysis")
    available_tickers = sorted(list(returns.columns))
    selected_ticker = st.sidebar.selectbox("Select Ticker", available_tickers, index=available_tickers.index('AAPL') if 'AAPL' in available_tickers else 0)
    
    benchmark_options = [b for b in ['SPY', 'QQQ', 'IWM', 'DIA'] if b in returns.columns] or available_tickers[:5]
    benchmark = st.sidebar.selectbox("Benchmark", benchmark_options, index=0)
    
    st.sidebar.subheader("🔧 Model Settings")
    risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 3.0, 0.5) / 100
    transaction_cost = st.sidebar.slider("Transaction Cost (bps)", 0, 50, 8, 1)
    
    st.sidebar.subheader("🎯 4-Regime Settings")
    use_4regime = st.sidebar.checkbox("Enable 4-Regime Framework", value=True)
    vol_threshold = st.sidebar.slider("Vol Threshold (z-score)", -1.0, 1.0, 0.0, 0.1)
    market_dir_window = st.sidebar.slider("Market Direction Window", 5, 60, 20, 5)
    
    st.sidebar.subheader("📊 Train/Test Split")
    design_end = st.sidebar.date_input("Design Period End", datetime(2014, 12, 31))
    eval_start = st.sidebar.date_input("Evaluation Period Start", datetime(2015, 1, 1))
    
    use_model = st.sidebar.checkbox("Use Full Model", value=True)
    
    model = None
    if use_model and MODEL_AVAILABLE:
        config = ModelConfig(risk_free_rate=risk_free_rate, transaction_cost_bps=transaction_cost,
                            vol_threshold=vol_threshold, market_direction_window=market_dir_window, bootstrap_samples=500)
        with st.spinner("Loading model..."):
            model = load_model(trading_days, config)
    
    st.sidebar.subheader("🛡️ Risk Management")
    use_vol_managed = st.sidebar.checkbox("Volatility Targeting (BSC 2015)", value=False)
    target_vol = st.sidebar.slider("Target Volatility (%)", 5.0, 30.0, 12.0, 1.0) / 100
    
    use_crash_overlay = st.sidebar.checkbox("Crash Overlay (DM 2016)", value=False)
    crash_quantile = st.sidebar.slider("Crash Quantile (Top %)", 0.5, 0.95, 0.8, 0.05)
    crash_scale = st.sidebar.slider("Crash Exposure Scale", 0.0, 1.0, 0.5, 0.1)
    
    st.sidebar.subheader("🔬 Inference Settings")
    inference_method = st.sidebar.selectbox("Alpha Inference Method", ["OLS", "NW", "BlockBootstrap"])
    num_strategies = st.sidebar.number_input("Num Strategies (for Adj P-val)", 1, 100, 10)
    
    # STATUS BAR
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("📊 Assets", len(returns.columns))
    with col2: st.metric("📅 Trading Days", len(returns))
    with col3: st.metric("📆 Start", returns.index[0].strftime('%Y-%m'))
    with col4: st.metric("💭 Sentiment", "✅" if sentiment is not None else "❌")
    with col5: st.metric("🔧 Model", "✅" if model is not None else "❌")
    
    st.divider()
    
    # TABS
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["💰 Performance", "🎯 4-Regime", "📊 Fama-MacBeth", "🧪 Ablation", "📈 Factors", "🔬 Robustness", "💭 Sentiment", "📋 Report"])
    
    if model is not None:
        weights = model.get_optimal_weights()
        strategy_returns = model.run_backtest(weights)
        
        # Apply Risk Management
        if use_vol_managed:
            strategy_returns = model.performance.apply_volatility_management(
                strategy_returns, target_vol_annual=target_vol
            )
        
        if use_crash_overlay:
            # Use market returns as proxy for crash detection
            market_vol = benchmark_returns.rolling(22).std() * np.sqrt(252)
            strategy_returns = model.performance.apply_crash_overlay(
                strategy_returns, market_vol, crash_quantile=crash_quantile, crash_scale=crash_scale
            )
            
    else:
        weights = pd.Series(1.0/len(returns.columns), index=returns.columns)
        strategy_returns = (returns * weights).sum(axis=1)
    
    benchmark_returns = returns[benchmark]
    
    # TAB 1: Performance
    with tab1:
        st.header("Portfolio Performance")
        fig_cum = plot_cumulative_returns(strategy_returns, benchmark_returns, benchmark)
        st.plotly_chart(fig_cum, use_container_width=True)
        
        rf_daily = risk_free_rate / 252
        strat_total = (1 + strategy_returns).prod() - 1
        strat_vol = strategy_returns.std() * np.sqrt(252)
        strat_sharpe = (strategy_returns.mean() - rf_daily) / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        strat_cum = (1 + strategy_returns).cumprod()
        strat_dd = (strat_cum / strat_cum.cummax() - 1).min()
        bench_total = (1 + benchmark_returns).prod() - 1
        bench_sharpe = (benchmark_returns.mean() - rf_daily) / benchmark_returns.std() * np.sqrt(252) if benchmark_returns.std() > 0 else 0
        
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.metric("Total Return", f"{strat_total:.2%}", f"{strat_total - bench_total:+.2%}")
        with c2: st.metric("Sharpe Ratio", f"{strat_sharpe:.2f}", f"{strat_sharpe - bench_sharpe:+.2f}")
        with c3: st.metric("Volatility", f"{strat_vol:.2%}")
        with c4: st.metric("Max Drawdown", f"{strat_dd:.2%}")
        with c5:
            n_years = len(strategy_returns) / 252
            cagr = (1 + strat_total) ** (1/n_years) - 1 if n_years > 0 else 0
            st.metric("CAGR", f"{cagr:.2%}")
        
        st.subheader("Drawdown")
        fig_dd = plot_drawdown(strategy_returns)
        st.plotly_chart(fig_dd, use_container_width=True)
        
        st.subheader("Top Holdings")
        top_weights = weights.nlargest(10)
        fig_weights = px.bar(x=top_weights.values * 100, y=top_weights.index, orientation='h', labels={'x': 'Weight (%)', 'y': 'Ticker'})
        fig_weights.update_layout(template='plotly_white', height=350)
        st.plotly_chart(fig_weights, use_container_width=True)
    
    # TAB 2: 4-Regime
    with tab2:
        st.header("🎯 4-Regime Framework")
        st.markdown('<div class="info-box"><h4>Regime Classification</h4><p>Based on Volatility × Market Direction</p></div>', unsafe_allow_html=True)
        
        if model is not None and model.regimes is not None:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("Regime Distribution")
                fig_dist = plot_4regime_distribution(model.regimes)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with c2:
                st.subheader("Regime Cards")
                regime_counts = model.regimes['regime'].value_counts()
                total = len(model.regimes)
                
                r1, r2 = st.columns(2)
                with r1:
                    lb_count = regime_counts.get('LowVol_Bear', 0)
                    st.markdown(f'<div class="regime-lb"><h4>🐻 Low Vol + Bear</h4><p>{lb_count} days ({lb_count/total*100:.1f}%)</p></div>', unsafe_allow_html=True)
                    hb_count = regime_counts.get('HighVol_Bear', 0)
                    st.markdown(f'<div class="regime-hb"><h4>📉 High Vol + Bear</h4><p>{hb_count} days ({hb_count/total*100:.1f}%)</p></div>', unsafe_allow_html=True)
                with r2:
                    lu_count = regime_counts.get('LowVol_Bull', 0)
                    st.markdown(f'<div class="regime-lu"><h4>🐂 Low Vol + Bull</h4><p>{lu_count} days ({lu_count/total*100:.1f}%)</p></div>', unsafe_allow_html=True)
                    hu_count = regime_counts.get('HighVol_Bull', 0)
                    st.markdown(f'<div class="regime-hu"><h4>🚀 High Vol + Bull</h4><p>{hu_count} days ({hu_count/total*100:.1f}%)</p></div>', unsafe_allow_html=True)
            
            st.subheader("Regime Timeline")
            regime_map = {'LowVol_Bear': 1, 'LowVol_Bull': 2, 'HighVol_Bear': 3, 'HighVol_Bull': 4}
            regime_numeric = model.regimes['regime'].map(regime_map)
            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Scatter(x=model.regimes.index, y=regime_numeric, mode='markers',
                marker=dict(color=regime_numeric, colorscale=[[0, '#74b9ff'], [0.33, '#55efc4'], [0.66, '#fd79a8'], [1, '#ffeaa7']], size=3)))
            fig_timeline.update_layout(title='Regime Over Time', yaxis=dict(tickmode='array', tickvals=[1, 2, 3, 4], ticktext=['LB', 'LU', 'HB', 'HU']),
                                       template='plotly_white', height=300)
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.warning("⚠️ Model not loaded or regime data unavailable.")
    
    # TAB 3: Fama-MacBeth
    with tab3:
        st.header("📊 Fama-MacBeth Regression")
        if model is not None and RESEARCH_AVAILABLE:
            with st.spinner("Running Fama-MacBeth..."):
                try:
                    design = ResearchDesignV2(design_end=design_end.strftime("%Y-%m-%d"), eval_start=eval_start.strftime("%Y-%m-%d"),
                                             use_4regime=use_4regime, vol_threshold=vol_threshold)
                    tester = CrossSectionalTesterV2(model, design)
                    panel_eval = tester.build_panel(period="eval")
                    fm_result = tester.fama_macbeth(panel_eval)
                    
                    st.subheader("Evaluation Period Results")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Mean Coefficients:**")
                        st.dataframe(pd.DataFrame({'Coefficient': fm_result['mean_coefs'], 'T-Statistic': fm_result['tstats'], 'P-Value': fm_result['pvalues']}).round(4))
                    with c2:
                        fig_fm = plot_fama_macbeth_coefficients(fm_result)
                        st.plotly_chart(fig_fm, use_container_width=True)
                    st.markdown(f"**Number of cross-sections (T):** {fm_result['T']}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("⚠️ Model or Research Pipeline not available.")
    
    # TAB 4: Ablation
    with tab4:
        st.header("🧪 Ablation Study")
        st.markdown('<div class="info-box"><h4>8-Configuration Ablation</h4><p>Testing each component\'s contribution to alpha.</p></div>', unsafe_allow_html=True)
        
        if model is not None and RESEARCH_AVAILABLE:
            if st.button("🚀 Run Ablation Study", type="primary"):
                with st.spinner("Running 8 configurations..."):
                    try:
                        design = ResearchDesignV2(design_end=design_end.strftime("%Y-%m-%d"), eval_start=eval_start.strftime("%Y-%m-%d"), use_4regime=use_4regime)
                        ablation_tester = AblationTesterV2(model, design)
                        ablation_df = ablation_tester.run_ablation(period="eval")
                        
                        st.subheader("Results")
                        st.dataframe(ablation_df.style.format({'Alpha': '{:.2%}', 'T-Stat': '{:.3f}', 'P-Value': '{:.4f}', 'Sharpe': '{:.3f}'}))
                        fig_ablation = plot_ablation_results(ablation_df)
                        st.plotly_chart(fig_ablation, use_container_width=True)
                        st.session_state['ablation_results'] = ablation_df
                    except Exception as e:
                        st.error(f"Error: {e}")
            if 'ablation_results' in st.session_state:
                st.subheader("Previous Results")
                st.dataframe(st.session_state['ablation_results'])
        else:
            st.warning("⚠️ Model or Research Pipeline not available.")
    
    # TAB 5: Factors
    with tab5:
        st.header("📈 Factor Adjustment")
        st.markdown('<div class="info-box"><h4>CAPM vs FF5 + Momentum</h4><p>Report both CAPM and FF5+MOM alphas for robustness (Duke feedback).</p></div>', unsafe_allow_html=True)
        
        if model is not None:
            validator = StatisticalValidator(model.config)
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("CAPM Alpha")
                ab = validator.calculate_alpha_beta(strategy_returns, benchmark_returns)
                st.metric("Alpha (Annual)", f"{ab['alpha_annual']:.2%}")
                st.metric("T-Statistic", f"{ab['alpha_tstat']:.3f}")
                st.metric("P-Value", f"{ab['alpha_pval']:.4f}")
                st.metric("R-Squared", f"{ab['r_squared']:.2%}")
                st.metric("R-Squared", f"{ab['r_squared']:.2%}")
                if ab['significant']:
                    st.success("✅ Significant")
                else:
                    st.warning("⚠️ Not Significant")
            
            with c2:

                st.subheader("FF5 + Momentum Alpha")
                if RESEARCH_AVAILABLE:
                    fv = FactorValidatorV2(model=model)
                    
                    # Use robust inference if available in model validator
                    # But FactorValidatorV2 is separate. 
                    # Let's use the new compute_factor_alpha from StatisticalValidator if possible,
                    # or update FactorValidatorV2. 
                    # For now, we'll use the new method in StatisticalValidator for the main strategy
                    
                    # Construct factor data for the new method
                    if fv.factor_data is None:
                        fv._construct_proxy_factors()
                    
                    if fv.factor_data is not None:
                        robust_res = validator.compute_factor_alpha(
                            strategy_returns, 
                            fv.factor_data, 
                            method=inference_method,
                            num_strategies=num_strategies
                        )
                        
                        st.metric("Alpha (Annual)", f"{robust_res['alpha_annual']:.2%}")
                        st.metric("T-Statistic", f"{robust_res['alpha_tstat']:.3f}")
                        st.metric("P-Value", f"{robust_res['alpha_pval']:.4f}")
                        st.metric("Adj P-Value (HLZ)", f"{robust_res['adj_pval']:.4f}")
                        st.caption(f"95% CI: [{robust_res['alpha_ci'][0]:.2%}, {robust_res['alpha_ci'][1]:.2%}]")
                        
                        if robust_res['alpha_pval'] < 0.05:
                            st.success("✅ Significant (p < 0.05)")
                        else:
                            st.warning("⚠️ Not Significant")
                    else:
                        st.warning("Could not construct factors")
                else:
                    st.info("FF5+MOM requires research_pipeline_v2")
            
            st.divider()
            st.subheader("Dynamic Beta")
            fig_beta = plot_dynamic_beta(model, selected_ticker)
            if fig_beta: st.plotly_chart(fig_beta, use_container_width=True)
            
            st.subheader("GARCH-MIDAS Volatility")
            fig_vol = plot_volatility(model, selected_ticker)
            if fig_vol: st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.warning("⚠️ Model not loaded.")
    
    # TAB 6: Robustness
    with tab6:
        st.header("🔬 Robustness Tests")
        if model is not None:
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Bootstrap Analysis")
                with st.spinner("Running bootstrap..."):
                    validator = StatisticalValidator(model.config)
                    boot = validator.bootstrap_analysis(strategy_returns, benchmark_returns)
                    st.markdown(f"**Alpha Bootstrap ({boot['n_samples']} samples):**\n- Mean: {boot['alpha_mean']:.2%}\n- 95% CI: [{boot['alpha_ci_lower']:.2%}, {boot['alpha_ci_upper']:.2%}]\n- Zero in CI: {'❌ Yes' if boot['alpha_zero_in_ci'] else '✅ No'}")
                    st.markdown(f"**Sharpe Bootstrap:**\n- Mean: {boot['sharpe_mean']:.3f}\n- 95% CI: [{boot['sharpe_ci_lower']:.3f}, {boot['sharpe_ci_upper']:.3f}]\n- Zero in CI: {'❌ Yes' if boot['sharpe_zero_in_ci'] else '✅ No'}")
                    fig_boot = plot_bootstrap_distribution(boot)
                    st.plotly_chart(fig_boot, use_container_width=True)
            
            with c2:
                st.subheader("Transaction Cost Sensitivity")
                tc = validator.transaction_cost_sensitivity(strategy_returns)
                breakeven = tc.get('breakeven_cost_bps', float('inf'))
                st.metric("Breakeven Cost", f"{breakeven:.0f} bps" if breakeven < float('inf') else "> 50 bps")
                st.metric("Robust at 50bps", "✅ Yes" if tc['robust_at_50bps'] else "❌ No")
                fig_tc = plot_transaction_cost_sensitivity(tc)
                st.plotly_chart(fig_tc, use_container_width=True)
            
            st.divider()
            st.subheader("Subperiod Analysis")
            subperiods = validator.subperiod_analysis(strategy_returns, benchmark_returns, n_periods=4)
            sub_df = pd.DataFrame(subperiods)
            sub_df['start_date'] = sub_df['start_date'].dt.strftime('%Y-%m-%d')
            sub_df['end_date'] = sub_df['end_date'].dt.strftime('%Y-%m-%d')
            st.dataframe(sub_df.style.format({'total_return': '{:.2%}', 'sharpe_ratio': '{:.3f}', 'max_drawdown': '{:.2%}', 'alpha_annual': '{:.2%}', 'alpha_pval': '{:.4f}'}))
            
            positive_sharpe = sum(1 for p in subperiods if p['sharpe_ratio'] > 0)
            significant_alpha = sum(1 for p in subperiods if p['significant'])
            s1, s2 = st.columns(2)
            with s1: st.metric("Positive Sharpe Periods", f"{positive_sharpe}/4")
            with s2: st.metric("Significant Alpha Periods", f"{significant_alpha}/4")
            
            if RESEARCH_AVAILABLE and use_4regime:
                st.divider()
                st.subheader("Regime Threshold Sensitivity")
                if st.button("Run Sensitivity Analysis"):
                    with st.spinner("Testing thresholds..."):
                        try:
                            design = ResearchDesignV2(design_end=design_end.strftime("%Y-%m-%d"), eval_start=eval_start.strftime("%Y-%m-%d"))
                            sensitivity = RegimeSensitivityAnalyzer(model, design)
                            sens_df = sensitivity.run_threshold_sensitivity(period="eval")
                            st.dataframe(sens_df.round(3))
                            fig_sens = plot_regime_sensitivity(sens_df)
                            st.plotly_chart(fig_sens, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error: {e}")
        else:
            st.warning("⚠️ Model not loaded.")
    
    # TAB 7: Sentiment
    with tab7:
        st.header("💭 Sentiment Analysis")
        if sentiment is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Market Sentiment vs Return")
                if 'MARKET_SENTIMENT' in sentiment.columns:
                    mkt_sent = sentiment['MARKET_SENTIMENT']
                    fig_sent = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_sent.add_trace(go.Scatter(x=mkt_sent.index, y=mkt_sent.rolling(22).mean(), name="Sentiment (1M MA)", line=dict(color='blue')), secondary_y=False)
                    fig_sent.add_trace(go.Scatter(x=benchmark_returns.index, y=benchmark_returns.rolling(22).mean(), name="Market Ret (1M MA)", line=dict(color='gray', dash='dot')), secondary_y=True)
                    fig_sent.update_layout(title="Market Sentiment vs Returns", template="plotly_white", height=400)
                    st.plotly_chart(fig_sent, use_container_width=True)
                else:
                    st.info("No MARKET_SENTIMENT column found.")
            
            with c2:
                st.subheader("Ticker Sentiment")
                sel_ticker_sent = st.selectbox("Select Ticker for Sentiment", available_tickers, key='sent_ticker')
                if sel_ticker_sent in sentiment.columns:
                    ts_sent = sentiment[sel_ticker_sent]
                    fig_ts = plot_sentiment_signal(sentiment, sel_ticker_sent)
                    st.plotly_chart(fig_ts, use_container_width=True)
                else:
                    st.warning(f"No sentiment data for {sel_ticker_sent}")
            
            st.divider()
            st.subheader("Sentiment Distribution")
            latest_sent = sentiment.iloc[-1]
            fig_hist = px.histogram(latest_sent, nbins=30, title=f"Sentiment Distribution ({sentiment.index[-1].date()})")
            fig_hist.update_layout(showlegend=False, template="plotly_white", height=350)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            st.subheader("Coverage")
            coverage = sentiment.count(axis=1)
            fig_cov = px.area(coverage, title="Number of Assets with Sentiment Signal")
            fig_cov.update_layout(showlegend=False, template="plotly_white", height=300)
            st.plotly_chart(fig_cov, use_container_width=True)
            
        else:
            st.warning("⚠️ Sentiment data not loaded.")

    # TAB 8: Report
    with tab8:
        st.header("📋 Summary Report")
        if model is not None:
            summary = model.summary()
            validator = StatisticalValidator(model.config)
            ab = validator.calculate_alpha_beta(strategy_returns, benchmark_returns)
            perf = model.performance.calculate_metrics(strategy_returns)
            
            report = f"""# AI Finance Project Report v2
## 4-Regime Framework with Duke Feedback

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Configuration
- **Trading Days:** {len(strategy_returns)}
- **Date Range:** {strategy_returns.index[0].date()} to {strategy_returns.index[-1].date()}
- **Assets:** {summary['n_assets']}
- **Benchmark:** {benchmark}
- **Transaction Cost:** {transaction_cost} bps (NASDAQ-calibrated)
- **4-Regime Framework:** {'Enabled' if use_4regime else 'Disabled'}

---

## Performance Summary

| Metric | Strategy | {benchmark} | Difference |
|--------|----------|-------------|------------|
| Total Return | {perf['total_return']:.2%} | {bench_total:.2%} | {perf['total_return']-bench_total:+.2%} |
| CAGR | {perf['cagr']:.2%} | - | - |
| Sharpe Ratio | {perf['sharpe_ratio']:.3f} | {bench_sharpe:.3f} | {perf['sharpe_ratio']-bench_sharpe:+.3f} |
| Max Drawdown | {perf['max_drawdown']:.2%} | - | - |

---

## Statistical Significance (CAPM)

| Metric | Value |
|--------|-------|
| Alpha (Annual) | {ab['alpha_annual']:.2%} |
| Alpha t-statistic | {ab['alpha_tstat']:.3f} |
| Alpha p-value | {ab['alpha_pval']:.4f} |
| Beta | {ab['beta']:.3f} |

**Verdict:** {'✅ SIGNIFICANT' if ab['significant'] else '❌ NOT SIGNIFICANT'}

---

*Report generated by AI Finance Dashboard v4*
"""
            st.markdown(report)
            st.download_button("📥 Download Report (Markdown)", report, file_name=f"ai_finance_report_v2_{datetime.now().strftime('%Y%m%d')}.md", mime="text/markdown")
        else:
            st.warning("Load the model to generate report.")
    
    return None


if __name__ == "__main__":
    run_dashboard()
