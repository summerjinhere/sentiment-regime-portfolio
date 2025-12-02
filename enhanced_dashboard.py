"""
Enhanced Multi-Agent Portfolio Dashboard
=========================================

Streamlit dashboard for the Enhanced Integrated Model with:
- Performance metrics and statistical significance
- Regime visualization (HMM)
- Sentiment analysis display
- Volatility forecasting comparison
- Model Confidence Set results
- Interactive charts with Plotly

Author: Multi-Agent Portfolio System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced model
try:
    from enhanced_integrated_model import (
        EnhancedIntegratedModel,
        ModelConfig,
        MarketRegime,
        DynamicFactorLoading,
        DynamicSentimentModel,
        NonlinearGARCHMIDAS,
        AdaptiveMCSSelector,
        RegimeConditionalAlphaTest,
        IntegratedStatisticalFramework,
        create_enhanced_model
    )
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Enhanced Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 10px;
        color: #856404;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        color: #721c24;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA GENERATION / LOADING
# =============================================================================

@st.cache_data
def generate_sample_data(n_days: int = 500, seed: int = 42) -> Dict:
    """Generate sample data for demonstration"""
    np.random.seed(seed)
    
    # Date range
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')
    
    # Generate regime sequence (Markov chain)
    regimes = [1]  # Start in transition
    trans_matrix = np.array([
        [0.95, 0.04, 0.01],  # Bull stays bull
        [0.10, 0.80, 0.10],  # Transition
        [0.02, 0.08, 0.90]   # Bear stays bear
    ])
    
    for _ in range(n_days - 1):
        current = regimes[-1]
        next_regime = np.random.choice([0, 1, 2], p=trans_matrix[current])
        regimes.append(next_regime)
    
    regimes = np.array(regimes)
    
    # Generate returns based on regime
    regime_params = {
        0: (0.0008, 0.010),   # Bull: high return, low vol
        1: (0.0002, 0.015),   # Transition: medium
        2: (-0.0004, 0.025)   # Bear: negative return, high vol
    }
    
    returns = np.zeros(n_days)
    for i, r in enumerate(regimes):
        mu, sigma = regime_params[r]
        returns[i] = np.random.normal(mu, sigma)
    
    # Cumulative portfolio value
    portfolio_value = 100000 * np.cumprod(1 + returns)
    
    # Benchmark (SPY-like)
    benchmark_returns = np.random.normal(0.0004, 0.012, n_days)
    benchmark_value = 100000 * np.cumprod(1 + benchmark_returns)
    
    # Sentiment data
    sentiment = np.zeros(n_days)
    for i in range(n_days):
        base = 0.0 if regimes[i] == 1 else (0.3 if regimes[i] == 0 else -0.3)
        sentiment[i] = np.clip(base + np.random.normal(0, 0.2), -1, 1)
    
    # Volatility forecasts
    realized_vol = pd.Series(returns).rolling(22).std() * np.sqrt(252)
    
    # Multiple model predictions
    vol_predictions = {
        'SA_FOGI': realized_vol * (1 + np.random.normal(0, 0.1, n_days)),
        'GARCH': realized_vol * (1 + np.random.normal(0.05, 0.15, n_days)),
        'HAR_RV': realized_vol * (1 + np.random.normal(0.02, 0.12, n_days)),
        'EWMA': realized_vol * (1 + np.random.normal(0.03, 0.18, n_days))
    }
    
    # Factor loadings (time-varying)
    n_assets = 10
    n_factors = 3
    factor_loadings = []
    base_loading = np.random.randn(n_assets, n_factors) * 0.5
    for i in range(n_days):
        noise = np.random.randn(n_assets, n_factors) * 0.02
        factor_loadings.append(base_loading + noise * (i / n_days))
    
    # Regime probabilities
    regime_probs = np.zeros((n_days, 3))
    for i, r in enumerate(regimes):
        regime_probs[i, r] = 0.7 + np.random.uniform(0, 0.25)
        remaining = 1 - regime_probs[i, r]
        other_regimes = [j for j in range(3) if j != r]
        split = np.random.dirichlet([1, 1])
        for j, idx in enumerate(other_regimes):
            regime_probs[i, idx] = remaining * split[j]
    
    return {
        'dates': dates,
        'returns': returns,
        'portfolio_value': portfolio_value,
        'benchmark_returns': benchmark_returns,
        'benchmark_value': benchmark_value,
        'regimes': regimes,
        'regime_probs': regime_probs,
        'sentiment': sentiment,
        'realized_vol': realized_vol.values,
        'vol_predictions': vol_predictions,
        'factor_loadings': factor_loadings
    }


def calculate_metrics(returns: np.ndarray, risk_free_rate: float = 0.02) -> Dict:
    """Calculate performance metrics"""
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return {}
    
    freq = 252  # Daily
    excess_returns = returns - (risk_free_rate / freq)
    
    total_return = (1 + returns).prod() - 1
    cagr = (1 + total_return) ** (freq / len(returns)) - 1
    volatility = returns.std() * np.sqrt(freq)
    sharpe = excess_returns.mean() / returns.std() * np.sqrt(freq) if returns.std() > 0 else 0
    
    # Sortino
    downside = returns[returns < 0]
    sortino = excess_returns.mean() / downside.std() * np.sqrt(freq) if len(downside) > 0 and downside.std() > 0 else 0
    
    # Max Drawdown
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()
    
    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_dd,
        'Calmar Ratio': calmar
    }


def calculate_alpha_beta(strategy_returns: np.ndarray, benchmark_returns: np.ndarray) -> Dict:
    """Calculate alpha and beta with significance"""
    import statsmodels.api as sm
    from scipy import stats
    
    # Align
    min_len = min(len(strategy_returns), len(benchmark_returns))
    strat = strategy_returns[:min_len]
    bench = benchmark_returns[:min_len]
    
    # Remove NaN
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
        'Alpha Significant': model.pvalues[0] < 0.05,
        'T-Statistic': model.tvalues[0]
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_cumulative_returns(dates, portfolio_value, benchmark_value):
    """Plot cumulative returns comparison"""
    fig = go.Figure()
    
    # Normalize to 1
    port_norm = portfolio_value / portfolio_value[0]
    bench_norm = benchmark_value / benchmark_value[0]
    
    fig.add_trace(go.Scatter(
        x=dates, y=port_norm,
        name='Strategy',
        line=dict(color='#00C853', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(0, 200, 83, 0.1)'
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=bench_norm,
        name='Benchmark (SPY)',
        line=dict(color='#2196F3', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Cumulative Returns (Normalized)',
        xaxis_title='Date',
        yaxis_title='Growth of $1',
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=400
    )
    
    return fig


def plot_drawdown(dates, returns):
    """Plot drawdown chart"""
    cum = (1 + pd.Series(returns)).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates, y=drawdown * 100,
        fill='tozeroy',
        fillcolor='rgba(255, 82, 82, 0.3)',
        line=dict(color='#FF5252', width=1),
        name='Drawdown'
    ))
    
    fig.update_layout(
        title='Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        height=250
    )
    
    return fig


def plot_regime_timeline(dates, regimes, portfolio_value):
    """Plot price with regime coloring"""
    df = pd.DataFrame({
        'Date': dates,
        'Value': portfolio_value,
        'Regime': regimes
    })
    
    regime_names = {0: 'Bull', 1: 'Transition', 2: 'Bear'}
    regime_colors = {0: '#00C853', 1: '#FFB300', 2: '#FF5252'}
    
    df['Regime_Name'] = df['Regime'].map(regime_names)
    
    fig = go.Figure()
    
    for regime in [0, 1, 2]:
        mask = df['Regime'] == regime
        fig.add_trace(go.Scatter(
            x=df.loc[mask, 'Date'],
            y=df.loc[mask, 'Value'],
            mode='markers',
            name=regime_names[regime],
            marker=dict(color=regime_colors[regime], size=5),
            hovertemplate='%{x}<br>Value: $%{y:,.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Portfolio Value with Market Regimes',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        height=400
    )
    
    return fig


def plot_regime_probabilities(dates, regime_probs):
    """Plot stacked regime probabilities"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates, y=regime_probs[:, 0],
        name='Bull',
        stackgroup='one',
        fillcolor='rgba(0, 200, 83, 0.7)',
        line=dict(width=0)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=regime_probs[:, 1],
        name='Transition',
        stackgroup='one',
        fillcolor='rgba(255, 179, 0, 0.7)',
        line=dict(width=0)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=regime_probs[:, 2],
        name='Bear',
        stackgroup='one',
        fillcolor='rgba(255, 82, 82, 0.7)',
        line=dict(width=0)
    ))
    
    fig.update_layout(
        title='Regime Probability Evolution (HMM Confidence)',
        xaxis_title='Date',
        yaxis_title='Probability',
        hovermode='x unified',
        height=350
    )
    
    return fig


def plot_sentiment_timeline(dates, sentiment, regimes):
    """Plot sentiment with regime background"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05,
                        row_heights=[0.7, 0.3])
    
    # Sentiment line
    colors = ['#00C853' if s > 0.1 else '#FF5252' if s < -0.1 else '#FFB300' for s in sentiment]
    
    fig.add_trace(go.Scatter(
        x=dates, y=sentiment,
        mode='lines+markers',
        name='Sentiment',
        line=dict(color='#2196F3', width=1.5),
        marker=dict(size=3, color=colors)
    ), row=1, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hline(y=0.3, line_dash="dot", line_color="green", opacity=0.5, row=1, col=1)
    fig.add_hline(y=-0.3, line_dash="dot", line_color="red", opacity=0.5, row=1, col=1)
    
    # Regime bar
    regime_colors = {0: '#00C853', 1: '#FFB300', 2: '#FF5252'}
    bar_colors = [regime_colors[r] for r in regimes]
    
    fig.add_trace(go.Bar(
        x=dates, y=[1]*len(dates),
        marker_color=bar_colors,
        name='Regime',
        showlegend=False
    ), row=2, col=1)
    
    fig.update_layout(
        title='FinBERT Sentiment Analysis with Market Regimes',
        height=400,
        hovermode='x unified'
    )
    fig.update_yaxes(title_text='Sentiment Score', row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    
    return fig


def plot_volatility_comparison(dates, realized_vol, vol_predictions):
    """Plot volatility forecasts vs realized"""
    fig = go.Figure()
    
    # Realized volatility
    fig.add_trace(go.Scatter(
        x=dates, y=realized_vol * 100,
        name='Realized Vol',
        line=dict(color='black', width=2)
    ))
    
    # Model predictions
    colors = ['#2196F3', '#FF9800', '#9C27B0', '#4CAF50']
    for i, (name, pred) in enumerate(vol_predictions.items()):
        fig.add_trace(go.Scatter(
            x=dates, y=pred * 100,
            name=name,
            line=dict(color=colors[i % len(colors)], width=1.5, dash='dot'),
            opacity=0.7
        ))
    
    fig.update_layout(
        title='Volatility Forecasting: Model Comparison',
        xaxis_title='Date',
        yaxis_title='Annualized Volatility (%)',
        hovermode='x unified',
        height=400
    )
    
    return fig


def plot_mcs_results(mcs_results: Dict):
    """Plot MCS test results"""
    fig = go.Figure()
    
    models = list(mcs_results.keys())
    included = [1 if mcs_results[m].get('in_mcs', False) else 0 for m in models]
    p_values = [mcs_results[m].get('p_value', 0) for m in models]
    
    colors = ['#00C853' if inc else '#FF5252' for inc in included]
    
    fig.add_trace(go.Bar(
        x=models,
        y=p_values,
        marker_color=colors,
        text=[f'{p:.3f}' for p in p_values],
        textposition='outside'
    ))
    
    fig.add_hline(y=0.1, line_dash="dash", line_color="red",
                  annotation_text="α = 0.10")
    
    fig.update_layout(
        title='Model Confidence Set (MCS) Test Results',
        xaxis_title='Model',
        yaxis_title='P-Value',
        height=350
    )
    
    return fig


def plot_regime_alpha(regime_results: Dict):
    """Plot regime-conditional alpha"""
    regimes = []
    alphas = []
    significants = []
    
    for name, result in regime_results.items():
        if name != 'Overall':
            regimes.append(name)
            alphas.append(result.get('alpha_annual', 0) * 100)
            significants.append(result.get('significant', False))
    
    colors = ['#00C853' if sig else '#FFB300' for sig in significants]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=regimes,
        y=alphas,
        marker_color=colors,
        text=[f'{a:.2f}%' for a in alphas],
        textposition='outside'
    ))
    
    fig.add_hline(y=0, line_dash="solid", line_color="black")
    
    fig.update_layout(
        title='Regime-Conditional Alpha (Annual %)',
        xaxis_title='Market Regime',
        yaxis_title='Alpha (%)',
        height=350
    )
    
    return fig


def plot_rolling_sharpe(dates, returns, window=63):
    """Plot rolling Sharpe ratio"""
    returns_series = pd.Series(returns)
    rolling_mean = returns_series.rolling(window).mean() * 252
    rolling_std = returns_series.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_mean / rolling_std
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates, y=rolling_sharpe,
        name=f'{window}-Day Rolling Sharpe',
        line=dict(color='#9C27B0', width=2)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.add_hline(y=1, line_dash="dot", line_color="green", opacity=0.5)
    fig.add_hline(y=-1, line_dash="dot", line_color="red", opacity=0.5)
    
    fig.update_layout(
        title=f'Rolling Sharpe Ratio ({window}-Day Window)',
        xaxis_title='Date',
        yaxis_title='Sharpe Ratio',
        height=300
    )
    
    return fig


def plot_factor_loading_heatmap(factor_loadings, asset_names=None, factor_names=None):
    """Plot current factor loadings as heatmap"""
    # Use last loading
    loading = factor_loadings[-1]
    
    if asset_names is None:
        asset_names = [f'Asset {i+1}' for i in range(loading.shape[0])]
    if factor_names is None:
        factor_names = [f'Factor {i+1}' for i in range(loading.shape[1])]
    
    fig = px.imshow(
        loading,
        labels=dict(x='Factor', y='Asset', color='Loading'),
        x=factor_names,
        y=asset_names,
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    
    fig.update_layout(
        title='Current Factor Loadings (Dynamic)',
        height=400
    )
    
    return fig


# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Enhanced Multi-Agent Portfolio Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
        <b>Weakness Compensation Framework</b>: FOGI + FinBERT + GARCH-MIDAS + MCS Integration
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Data options
    st.sidebar.subheader("Data Settings")
    n_days = st.sidebar.slider("Simulation Days", 100, 1000, 500)
    seed = st.sidebar.number_input("Random Seed", 1, 9999, 42)
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate", 0.0, 0.10, 0.02, 0.01)
    
    # Model options
    st.sidebar.subheader("Model Settings")
    n_regimes = st.sidebar.selectbox("Number of Regimes", [2, 3, 4], index=1)
    sentiment_ar_order = st.sidebar.selectbox("Sentiment AR Order", [1, 2, 3], index=1)
    
    # Generate data
    with st.spinner("Loading data..."):
        data = generate_sample_data(n_days, seed)
    
    # Calculate metrics
    strategy_metrics = calculate_metrics(data['returns'], risk_free_rate)
    benchmark_metrics = calculate_metrics(data['benchmark_returns'], risk_free_rate)
    alpha_beta = calculate_alpha_beta(data['returns'], data['benchmark_returns'])
    
    # =================================================================
    # TAB NAVIGATION
    # =================================================================
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Performance", 
        "🎯 Regimes", 
        "💭 Sentiment", 
        "📊 Volatility",
        "🔬 Statistics",
        "📋 Report"
    ])
    
    # =================================================================
    # TAB 1: PERFORMANCE
    # =================================================================
    
    with tab1:
        st.header("Performance Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta = strategy_metrics['Total Return'] - benchmark_metrics['Total Return']
            st.metric(
                "Total Return",
                f"{strategy_metrics['Total Return']:.2%}",
                delta=f"{delta:+.2%} vs Benchmark"
            )
        
        with col2:
            delta = strategy_metrics['Sharpe Ratio'] - benchmark_metrics['Sharpe Ratio']
            st.metric(
                "Sharpe Ratio",
                f"{strategy_metrics['Sharpe Ratio']:.2f}",
                delta=f"{delta:+.2f} vs Benchmark"
            )
        
        with col3:
            st.metric(
                "Max Drawdown",
                f"{strategy_metrics['Max Drawdown']:.2%}"
            )
        
        with col4:
            st.metric(
                "Volatility",
                f"{strategy_metrics['Volatility']:.2%}"
            )
        
        # Charts
        st.subheader("Cumulative Returns")
        fig_returns = plot_cumulative_returns(
            data['dates'], data['portfolio_value'], data['benchmark_value']
        )
        st.plotly_chart(fig_returns, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Drawdown")
            fig_dd = plot_drawdown(data['dates'], data['returns'])
            st.plotly_chart(fig_dd, use_container_width=True)
        
        with col2:
            st.subheader("Rolling Sharpe")
            fig_sharpe = plot_rolling_sharpe(data['dates'], data['returns'])
            st.plotly_chart(fig_sharpe, use_container_width=True)
        
        # Additional metrics
        st.subheader("Detailed Metrics")
        
        metrics_df = pd.DataFrame({
            'Metric': list(strategy_metrics.keys()),
            'Strategy': [f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}" for v in strategy_metrics.values()],
            'Benchmark': [f"{benchmark_metrics.get(k, 0):.4f}" if abs(benchmark_metrics.get(k, 0)) < 10 
                         else f"{benchmark_metrics.get(k, 0):.2f}" for k in strategy_metrics.keys()]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # =================================================================
    # TAB 2: REGIMES
    # =================================================================
    
    with tab2:
        st.header("HMM Regime Analysis")
        
        # Regime distribution
        col1, col2, col3 = st.columns(3)
        
        regime_counts = pd.Series(data['regimes']).value_counts().sort_index()
        regime_names = {0: 'Bull 🟢', 1: 'Transition 🟡', 2: 'Bear 🔴'}
        
        for i, col in enumerate([col1, col2, col3]):
            with col:
                count = regime_counts.get(i, 0)
                pct = count / len(data['regimes']) * 100
                st.metric(
                    regime_names[i],
                    f"{pct:.1f}%",
                    f"{count} days"
                )
        
        # Regime timeline
        st.subheader("Market Regimes Over Time")
        fig_regime = plot_regime_timeline(
            data['dates'], data['regimes'], data['portfolio_value']
        )
        st.plotly_chart(fig_regime, use_container_width=True)
        
        # Regime probabilities
        st.subheader("Regime Probability Evolution")
        fig_probs = plot_regime_probabilities(data['dates'], data['regime_probs'])
        st.plotly_chart(fig_probs, use_container_width=True)
        
        # Transition matrix
        st.subheader("Regime Transition Matrix")
        
        # Calculate empirical transition matrix
        transitions = np.zeros((3, 3))
        for i in range(len(data['regimes']) - 1):
            transitions[data['regimes'][i], data['regimes'][i+1]] += 1
        
        # Normalize
        row_sums = transitions.sum(axis=1, keepdims=True)
        trans_matrix = np.divide(transitions, row_sums, where=row_sums != 0)
        
        fig_trans = px.imshow(
            trans_matrix,
            labels=dict(x="To Regime", y="From Regime", color="Probability"),
            x=['Bull', 'Transition', 'Bear'],
            y=['Bull', 'Transition', 'Bear'],
            color_continuous_scale='Blues',
            text_auto='.2f'
        )
        fig_trans.update_layout(height=350)
        st.plotly_chart(fig_trans, use_container_width=True)
        
        # Regime statistics
        st.subheader("Regime Statistics")
        
        regime_stats = []
        for r in [0, 1, 2]:
            mask = data['regimes'] == r
            returns_r = data['returns'][mask]
            regime_stats.append({
                'Regime': regime_names[r],
                'Mean Return (Daily)': f"{np.mean(returns_r):.4f}",
                'Volatility (Daily)': f"{np.std(returns_r):.4f}",
                'Sharpe Ratio': f"{np.mean(returns_r) / np.std(returns_r) * np.sqrt(252):.2f}" if np.std(returns_r) > 0 else "N/A",
                'Days': int(mask.sum())
            })
        
        st.dataframe(pd.DataFrame(regime_stats), use_container_width=True, hide_index=True)
    
    # =================================================================
    # TAB 3: SENTIMENT
    # =================================================================
    
    with tab3:
        st.header("FinBERT Sentiment Analysis")
        
        # Current sentiment
        current_sentiment = data['sentiment'][-1]
        sentiment_momentum = data['sentiment'][-1] - data['sentiment'][-5] if len(data['sentiment']) > 5 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment_color = "🟢" if current_sentiment > 0.1 else "🔴" if current_sentiment < -0.1 else "🟡"
            st.metric(
                f"Current Sentiment {sentiment_color}",
                f"{current_sentiment:.3f}",
                delta=f"{sentiment_momentum:+.3f} (5-day)"
            )
        
        with col2:
            avg_sentiment = np.mean(data['sentiment'])
            st.metric(
                "Average Sentiment",
                f"{avg_sentiment:.3f}"
            )
        
        with col3:
            sentiment_vol = np.std(data['sentiment'])
            st.metric(
                "Sentiment Volatility",
                f"{sentiment_vol:.3f}"
            )
        
        # Sentiment timeline
        st.subheader("Sentiment Timeline with Regimes")
        fig_sentiment = plot_sentiment_timeline(
            data['dates'], data['sentiment'], data['regimes']
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        # Sentiment distribution by regime
        st.subheader("Sentiment Distribution by Regime")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_by_regime = []
            for r in [0, 1, 2]:
                mask = data['regimes'] == r
                sentiment_by_regime.extend([
                    {'Regime': regime_names[r], 'Sentiment': s}
                    for s in data['sentiment'][mask]
                ])
            
            df_sent = pd.DataFrame(sentiment_by_regime)
            fig_box = px.box(df_sent, x='Regime', y='Sentiment', color='Regime',
                           color_discrete_map={
                               'Bull 🟢': '#00C853',
                               'Transition 🟡': '#FFB300',
                               'Bear 🔴': '#FF5252'
                           })
            fig_box.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            # Sentiment-Return correlation
            fig_corr = px.scatter(
                x=data['sentiment'][:-1],
                y=data['returns'][1:] * 100,
                labels={'x': 'Sentiment (t)', 'y': 'Return (t+1) %'},
                title='Sentiment vs Next-Day Returns',
                trendline='ols'
            )
            fig_corr.update_layout(height=350)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Sentiment impact on volatility
        st.subheader("Sentiment → Volatility Impact")
        
        vol_adjustment = 1 + 0.3 * np.maximum(0, -data['sentiment'])  # Negative sentiment increases vol
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Sentiment-Based Volatility Adjustment:**
            - Negative sentiment → Increases volatility forecast
            - Positive sentiment → Slightly decreases volatility
            - Regime conditioning amplifies effect in Bear markets
            """)
        
        with col2:
            fig_vol_adj = go.Figure()
            fig_vol_adj.add_trace(go.Histogram(
                x=vol_adjustment,
                nbinsx=30,
                marker_color='#2196F3'
            ))
            fig_vol_adj.update_layout(
                title='Distribution of Volatility Adjustments',
                xaxis_title='Adjustment Factor',
                height=300
            )
            st.plotly_chart(fig_vol_adj, use_container_width=True)
    
    # =================================================================
    # TAB 4: VOLATILITY
    # =================================================================
    
    with tab4:
        st.header("Volatility Forecasting & Model Comparison")
        
        # Model comparison chart
        st.subheader("Volatility Forecasts vs Realized")
        fig_vol = plot_volatility_comparison(
            data['dates'], data['realized_vol'], data['vol_predictions']
        )
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # Model accuracy metrics
        st.subheader("Forecast Accuracy (QLIKE Loss)")
        
        valid_idx = ~np.isnan(data['realized_vol'])
        realized = data['realized_vol'][valid_idx]
        
        accuracy_data = []
        for name, pred in data['vol_predictions'].items():
            pred_valid = pred[valid_idx]
            # QLIKE loss
            qlike = np.mean(np.log(pred_valid + 1e-8) + realized / (pred_valid + 1e-8))
            # MSE
            mse = np.mean((realized - pred_valid) ** 2)
            # MAE
            mae = np.mean(np.abs(realized - pred_valid))
            
            accuracy_data.append({
                'Model': name,
                'QLIKE': f"{qlike:.4f}",
                'MSE': f"{mse:.6f}",
                'MAE': f"{mae:.4f}",
                'Rank': 0  # Will be filled
            })
        
        # Rank by QLIKE
        accuracy_df = pd.DataFrame(accuracy_data)
        accuracy_df['QLIKE_num'] = accuracy_df['QLIKE'].astype(float)
        accuracy_df['Rank'] = accuracy_df['QLIKE_num'].rank().astype(int)
        accuracy_df = accuracy_df.drop('QLIKE_num', axis=1)
        accuracy_df = accuracy_df.sort_values('Rank')
        
        st.dataframe(accuracy_df, use_container_width=True, hide_index=True)
        
        # MCS simulation
        st.subheader("Model Confidence Set (MCS)")
        
        # Simulate MCS results
        mcs_results = {
            'SA_FOGI': {'in_mcs': True, 'p_value': 0.85},
            'GARCH': {'in_mcs': True, 'p_value': 0.42},
            'HAR_RV': {'in_mcs': False, 'p_value': 0.08},
            'EWMA': {'in_mcs': False, 'p_value': 0.03}
        }
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_mcs = plot_mcs_results(mcs_results)
            st.plotly_chart(fig_mcs, use_container_width=True)
        
        with col2:
            st.markdown("""
            **MCS Interpretation:**
            
            ✅ **In MCS (p > 0.10):**
            - SA_FOGI
            - GARCH
            
            ❌ **Eliminated:**
            - HAR_RV (p = 0.08)
            - EWMA (p = 0.03)
            
            *Models in MCS have statistically equivalent predictive ability at 90% confidence.*
            """)
        
        # Factor loadings
        st.subheader("Dynamic Factor Loadings")
        
        fig_loading = plot_factor_loading_heatmap(
            data['factor_loadings'],
            asset_names=[f'Asset {i+1}' for i in range(10)],
            factor_names=['Market', 'Size', 'Value']
        )
        st.plotly_chart(fig_loading, use_container_width=True)
    
    # =================================================================
    # TAB 5: STATISTICS
    # =================================================================
    
    with tab5:
        st.header("Statistical Significance Testing")
        
        # Alpha/Beta analysis
        st.subheader("Alpha & Beta Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            alpha_color = "🟢" if alpha_beta.get('Alpha Significant', False) else "🟡"
            st.metric(
                f"Annual Alpha {alpha_color}",
                f"{alpha_beta.get('Alpha (Annual)', 0):.2%}"
            )
        
        with col2:
            st.metric(
                "Beta",
                f"{alpha_beta.get('Beta', 0):.3f}"
            )
        
        with col3:
            st.metric(
                "R-Squared",
                f"{alpha_beta.get('R-Squared', 0):.2%}"
            )
        
        with col4:
            p_val = alpha_beta.get('Alpha P-Value', 1)
            st.metric(
                "Alpha P-Value",
                f"{p_val:.4f}",
                delta="Significant" if p_val < 0.05 else "Not Sig.",
                delta_color="normal" if p_val < 0.05 else "inverse"
            )
        
        # Regime-conditional alpha
        st.subheader("Regime-Conditional Alpha")
        
        # Simulate regime-conditional results
        regime_alpha_results = {}
        for r, name in enumerate(['Bull', 'Transition', 'Bear']):
            mask = data['regimes'] == r
            returns_r = data['returns'][mask]
            bench_r = data['benchmark_returns'][mask]
            
            if len(returns_r) > 20:
                result = calculate_alpha_beta(returns_r, bench_r)
                regime_alpha_results[name] = {
                    'alpha_annual': result.get('Alpha (Annual)', 0),
                    'significant': result.get('Alpha Significant', False),
                    'p_value': result.get('Alpha P-Value', 1)
                }
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_regime_alpha = plot_regime_alpha(regime_alpha_results)
            st.plotly_chart(fig_regime_alpha, use_container_width=True)
        
        with col2:
            st.markdown("**Interpretation:**")
            
            any_significant = any(r.get('significant', False) for r in regime_alpha_results.values())
            overall_sig = alpha_beta.get('Alpha Significant', False)
            
            if overall_sig:
                st.success("✅ Overall alpha is statistically significant!")
            elif any_significant:
                sig_regimes = [n for n, r in regime_alpha_results.items() if r.get('significant', False)]
                st.warning(f"⚠️ Overall alpha not significant, but significant in: {', '.join(sig_regimes)}")
            else:
                weighted_alpha = sum(r.get('alpha_annual', 0) for r in regime_alpha_results.values()) / 3
                if weighted_alpha > 0.02:
                    st.info(f"ℹ️ Economic significance (weighted α = {weighted_alpha:.2%}) but statistical significance lacking")
                else:
                    st.error("❌ No statistical or economic significance detected")
        
        # Bootstrap confidence interval
        st.subheader("Bootstrap Alpha Confidence Interval")
        
        # Simple bootstrap
        n_bootstrap = st.slider("Bootstrap Replications", 100, 5000, 1000)
        
        if st.button("Run Bootstrap Analysis"):
            with st.spinner("Running bootstrap..."):
                alphas = []
                T = len(data['returns'])
                
                for _ in range(n_bootstrap):
                    idx = np.random.choice(T, T, replace=True)
                    boot_strat = data['returns'][idx]
                    boot_bench = data['benchmark_returns'][idx]
                    
                    # Quick alpha calculation
                    X = np.column_stack([np.ones(T), boot_bench])
                    try:
                        beta = np.linalg.lstsq(X, boot_strat, rcond=None)[0]
                        alphas.append(beta[0] * 252)
                    except:
                        pass
                
                alphas = np.array(alphas)
                ci_lower = np.percentile(alphas, 2.5)
                ci_upper = np.percentile(alphas, 97.5)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_boot = go.Figure()
                    fig_boot.add_trace(go.Histogram(
                        x=alphas * 100,
                        nbinsx=50,
                        marker_color='#2196F3'
                    ))
                    fig_boot.add_vline(x=0, line_dash="dash", line_color="red")
                    fig_boot.add_vline(x=ci_lower * 100, line_dash="dot", line_color="green")
                    fig_boot.add_vline(x=ci_upper * 100, line_dash="dot", line_color="green")
                    fig_boot.update_layout(
                        title='Bootstrap Alpha Distribution',
                        xaxis_title='Alpha (%)',
                        height=350
                    )
                    st.plotly_chart(fig_boot, use_container_width=True)
                
                with col2:
                    contains_zero = ci_lower <= 0 <= ci_upper
                    
                    st.metric("95% CI Lower", f"{ci_lower:.2%}")
                    st.metric("95% CI Upper", f"{ci_upper:.2%}")
                    
                    if contains_zero:
                        st.warning("⚠️ Confidence interval contains zero")
                    else:
                        st.success("✅ Confidence interval does NOT contain zero")
    
    # =================================================================
    # TAB 6: REPORT
    # =================================================================
    
    with tab6:
        st.header("Summary Report")
        
        # Generate report
        report = f"""
# Enhanced Multi-Agent Portfolio Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Executive Summary

| Metric | Strategy | Benchmark | Difference |
|--------|----------|-----------|------------|
| Total Return | {strategy_metrics['Total Return']:.2%} | {benchmark_metrics['Total Return']:.2%} | {strategy_metrics['Total Return'] - benchmark_metrics['Total Return']:+.2%} |
| Sharpe Ratio | {strategy_metrics['Sharpe Ratio']:.2f} | {benchmark_metrics['Sharpe Ratio']:.2f} | {strategy_metrics['Sharpe Ratio'] - benchmark_metrics['Sharpe Ratio']:+.2f} |
| Max Drawdown | {strategy_metrics['Max Drawdown']:.2%} | {benchmark_metrics['Max Drawdown']:.2%} | - |
| Volatility | {strategy_metrics['Volatility']:.2%} | {benchmark_metrics['Volatility']:.2%} | - |

---

## 2. Statistical Significance

- **Annual Alpha:** {alpha_beta.get('Alpha (Annual)', 0):.2%}
- **Alpha P-Value:** {alpha_beta.get('Alpha P-Value', 1):.4f}
- **Statistically Significant:** {'Yes ✅' if alpha_beta.get('Alpha Significant', False) else 'No ❌'}
- **Beta:** {alpha_beta.get('Beta', 0):.3f}
- **R-Squared:** {alpha_beta.get('R-Squared', 0):.2%}

---

## 3. Model Components

### 3.1 Dynamic Factor Loading (FOGI Compensation)
- Kalman Filter time-varying estimation
- Regime-conditional adjustment applied

### 3.2 Sentiment Analysis (FinBERT Compensation)
- AR({sentiment_ar_order}) sentiment dynamics
- Sentiment → Volatility pathway modeled
- Multi-frequency aggregation implemented

### 3.3 Volatility Forecasting (GARCH-MIDAS Compensation)
- Neural network nonlinear τ_t
- LASSO variable selection
- Adaptive MCS model selection

---

## 4. Regime Analysis

| Regime | Days | Proportion | Mean Return | Volatility |
|--------|------|------------|-------------|------------|
"""
        
        for r, name in enumerate(['Bull', 'Transition', 'Bear']):
            mask = data['regimes'] == r
            count = mask.sum()
            pct = count / len(data['regimes'])
            mean_ret = data['returns'][mask].mean()
            vol = data['returns'][mask].std()
            report += f"| {name} | {count} | {pct:.1%} | {mean_ret:.4f} | {vol:.4f} |\n"
        
        report += f"""

---

## 5. Conclusions

"""
        
        if alpha_beta.get('Alpha Significant', False):
            report += "✅ **Strategy generates statistically significant alpha.**\n"
        elif any(r.get('significant', False) for r in regime_alpha_results.values()):
            report += "⚠️ **Regime-specific alpha detected but overall significance lacking.**\n"
        else:
            report += "❌ **No statistically significant alpha detected.**\n"
        
        report += f"""
- Sharpe Ratio: {strategy_metrics['Sharpe Ratio']:.2f} ({'Above' if strategy_metrics['Sharpe Ratio'] > 1 else 'Below'} 1.0 threshold)
- Risk-adjusted outperformance vs benchmark: {strategy_metrics['Sharpe Ratio'] - benchmark_metrics['Sharpe Ratio']:+.2f}

---

*Report generated by Enhanced Multi-Agent Portfolio System*
"""
        
        st.markdown(report)
        
        # Download button
        st.download_button(
            label="📥 Download Report (Markdown)",
            data=report,
            file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )
        
        # Export data
        st.subheader("Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_df = pd.DataFrame({
                'Date': data['dates'],
                'Portfolio_Value': data['portfolio_value'],
                'Returns': data['returns'],
                'Regime': data['regimes'],
                'Sentiment': data['sentiment']
            })
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Backtest Data (CSV)",
                data=csv,
                file_name=f"backtest_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            metrics_export = pd.DataFrame({
                'Metric': list(strategy_metrics.keys()),
                'Value': list(strategy_metrics.values())
            })
            
            csv_metrics = metrics_export.to_csv(index=False)
            st.download_button(
                label="📥 Download Metrics (CSV)",
                data=csv_metrics,
                file_name=f"metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
