"""
Enhanced Dashboard v3 - Real Data Visualization
================================================

Streamlit application for visualizing:
- FinBERT Sentiment Signals (Real NLP data)
- Dynamic Beta (Kalman Filter output)
- Portfolio Cumulative Returns
- Model Performance Metrics

Uses ONLY REAL DATA from CSV files.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Import model
try:
    from enhanced_integrated_model import EnhancedIntegratedModel, ModelConfig
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="AI Finance Dashboard",
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
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .success-box { background-color: #d4edda; border-left: 5px solid #00C853; padding: 15px; border-radius: 5px; }
    .warning-box { background-color: #fff3cd; border-left: 5px solid #FFB300; padding: 15px; border-radius: 5px; }
    .danger-box { background-color: #f8d7da; border-left: 5px solid #FF5252; padding: 15px; border-radius: 5px; }
    .info-box { background-color: #e7f3ff; border-left: 5px solid #2196F3; padding: 15px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_price_data(filepath: str = 'real_backtest_panel.csv'):
    """Load price and returns data"""
    try:
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # Extract returns
        ret_cols = [c for c in data.columns if c.endswith('_ret')]
        returns = data[ret_cols].copy()
        returns.columns = [c.replace('_ret', '') for c in ret_cols]
        
        # Extract prices
        price_cols = [c for c in data.columns if c.endswith('_price')]
        prices = data[price_cols].copy()
        prices.columns = [c.replace('_price', '') for c in price_cols]
        
        return prices, returns
    except Exception as e:
        st.error(f"Error loading price data: {e}")
        return None, None


@st.cache_data
def load_sentiment_data(filepath: str = 'real_sentiment_data.csv'):
    """Load sentiment data"""
    try:
        sentiment = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return sentiment
    except Exception as e:
        st.warning(f"Sentiment data not found: {e}")
        return None


@st.cache_resource
def load_model():
    """Load and fit the model"""
    if not MODEL_AVAILABLE:
        return None
    
    try:
        config = ModelConfig()
        model = EnhancedIntegratedModel(config)
        if model.load_data():
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
    
    # Color based on sentiment
    colors = ['#FF5252' if v < -0.1 else '#00C853' if v > 0.1 else '#9E9E9E' for v in data.values]
    
    fig = go.Figure()
    
    # Sentiment line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data.values,
        mode='lines',
        name='Sentiment',
        line=dict(color='#2196F3', width=1.5)
    ))
    
    # Fill positive/negative regions
    fig.add_trace(go.Scatter(
        x=data.index,
        y=np.where(data.values > 0, data.values, 0),
        fill='tozeroy',
        fillcolor='rgba(0, 200, 83, 0.2)',
        line=dict(width=0),
        name='Positive'
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=np.where(data.values < 0, data.values, 0),
        fill='tozeroy',
        fillcolor='rgba(255, 82, 82, 0.2)',
        line=dict(width=0),
        name='Negative'
    ))
    
    # Reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    fig.add_hline(y=0.3, line_dash="dot", line_color="green", opacity=0.3)
    fig.add_hline(y=-0.3, line_dash="dot", line_color="red", opacity=0.3)
    
    fig.update_layout(
        title=f'FinBERT Sentiment Signal: {ticker}',
        xaxis_title='Date',
        yaxis_title='Sentiment Score',
        template='plotly_white',
        height=400,
        showlegend=False
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
        x=beta_series.index,
        y=beta_series.values,
        mode='lines',
        name='Dynamic Beta',
        line=dict(color='#9C27B0', width=2)
    ))
    
    # Reference line at beta = 1
    fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Rolling mean
    rolling_beta = beta_series.rolling(22).mean()
    fig.add_trace(go.Scatter(
        x=rolling_beta.index,
        y=rolling_beta.values,
        mode='lines',
        name='22-Day MA',
        line=dict(color='#FF9800', width=1.5, dash='dot')
    ))
    
    fig.update_layout(
        title=f'Dynamic Beta (Kalman Filter): {ticker}',
        xaxis_title='Date',
        yaxis_title='Beta',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_cumulative_returns(returns: pd.DataFrame, weights: pd.Series, benchmark: str = 'SPY'):
    """Plot portfolio cumulative returns"""
    if returns is None:
        return None
    
    # Portfolio returns
    common_tickers = [t for t in weights.index if t in returns.columns]
    w = weights[common_tickers]
    w = w / w.sum()  # Normalize
    
    portfolio_returns = (returns[common_tickers] * w).sum(axis=1)
    portfolio_cum = (1 + portfolio_returns).cumprod()
    
    # Benchmark
    if benchmark in returns.columns:
        bench_cum = (1 + returns[benchmark]).cumprod()
    else:
        bench_cum = None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=portfolio_cum.index,
        y=portfolio_cum.values,
        name='Strategy',
        line=dict(color='#00C853', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(0, 200, 83, 0.1)'
    ))
    
    if bench_cum is not None:
        fig.add_trace(go.Scatter(
            x=bench_cum.index,
            y=bench_cum.values,
            name=benchmark,
            line=dict(color='#2196F3', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title='Portfolio Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Growth of $1',
        template='plotly_white',
        height=450,
        hovermode='x unified'
    )
    
    return fig, portfolio_returns


def plot_volatility(model, ticker: str):
    """Plot GARCH-MIDAS volatility"""
    if model is None:
        return None
    
    vol_series = model.get_volatility_series(ticker)
    
    if vol_series.empty:
        return None
    
    # Annualize
    vol_annual = vol_series * np.sqrt(252) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=vol_annual.index,
        y=vol_annual.values,
        mode='lines',
        name='GARCH-MIDAS Volatility',
        line=dict(color='#FF5252', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(255, 82, 82, 0.1)'
    ))
    
    fig.update_layout(
        title=f'GARCH-MIDAS Volatility: {ticker}',
        xaxis_title='Date',
        yaxis_title='Annualized Volatility (%)',
        template='plotly_white',
        height=350
    )
    
    return fig


def calculate_metrics(returns: pd.Series) -> Dict:
    """Calculate performance metrics"""
    if returns is None or len(returns) < 2:
        return {}
    
    total_ret = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    cagr = (1 + total_ret) ** (1/n_years) - 1 if n_years > 0 else 0
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252 - 0.02) / vol if vol > 0 else 0
    
    cum = (1 + returns).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    
    return {
        'Total Return': total_ret,
        'CAGR': cagr,
        'Volatility': vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd
    }


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.markdown('<h1 class="main-header">🚀 AI Finance Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
        Weakness Compensation Framework | Real Data Only | Master's Thesis
    </div>
    """, unsafe_allow_html=True)
    
    # =================================================================
    # SIDEBAR
    # =================================================================
    
    st.sidebar.header("⚙️ Configuration")
    
    # Data files
    st.sidebar.subheader("📁 Data Files")
    prices_file = st.sidebar.text_input("Price Data", "real_backtest_panel.csv")
    sentiment_file = st.sidebar.text_input("Sentiment Data", "real_sentiment_data.csv")
    
    # Load data
    with st.spinner("Loading data..."):
        prices, returns = load_price_data(prices_file)
        sentiment = load_sentiment_data(sentiment_file)
    
    if returns is None:
        st.error("❌ Could not load price data. Please run fetch_real_data.py first.")
        st.stop()
    
    # Ticker selection
    st.sidebar.subheader("📈 Analysis")
    available_tickers = list(returns.columns)
    selected_ticker = st.sidebar.selectbox("Select Ticker", available_tickers)
    
    # Benchmark
    benchmark = st.sidebar.selectbox("Benchmark", ['SPY', 'QQQ', 'IWM'], index=0)
    
    # Load model
    st.sidebar.subheader("🔧 Model")
    use_model = st.sidebar.checkbox("Use Integrated Model", value=True)
    
    model = None
    if use_model and MODEL_AVAILABLE:
        with st.spinner("Loading model..."):
            model = load_model()
    
    # =================================================================
    # MAIN CONTENT
    # =================================================================
    
    # Status
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Assets", len(returns.columns))
    with col2:
        st.metric("📅 Trading Days", len(returns))
    with col3:
        sentiment_status = "✅" if sentiment is not None else "❌"
        st.metric("💭 Sentiment", sentiment_status)
    with col4:
        model_status = "✅" if model is not None else "❌"
        st.metric("🔧 Model", model_status)
    
    st.divider()
    
    # =================================================================
    # TABS
    # =================================================================
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💭 Sentiment",
        "📈 Dynamic Beta",
        "💰 Portfolio",
        "📊 Volatility",
        "📋 Summary"
    ])
    
    # ----------------------------
    # TAB 1: Sentiment
    # ----------------------------
    with tab1:
        st.header("FinBERT Sentiment Analysis")
        
        if sentiment is not None:
            # Market sentiment
            st.subheader("Market Sentiment")
            fig_market = plot_sentiment_signal(sentiment, 'MARKET_SENTIMENT')
            if fig_market:
                st.plotly_chart(fig_market, use_container_width=True)
            
            # Ticker sentiment
            st.subheader(f"{selected_ticker} Sentiment")
            if selected_ticker in sentiment.columns:
                fig_ticker = plot_sentiment_signal(sentiment, selected_ticker)
                if fig_ticker:
                    st.plotly_chart(fig_ticker, use_container_width=True)
                
                # Current sentiment
                current_sent = sentiment[selected_ticker].dropna().iloc[-1] if len(sentiment[selected_ticker].dropna()) > 0 else 0
                sent_color = "🟢" if current_sent > 0.1 else "🔴" if current_sent < -0.1 else "🟡"
                st.markdown(f"""
                <div class="info-box">
                    <h4>Current Sentiment: {sent_color} {current_sent:.3f}</h4>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"No sentiment data for {selected_ticker}")
        else:
            st.warning("⚠️ Sentiment data not loaded. Run real_news_scraper_bs4.py first.")
    
    # ----------------------------
    # TAB 2: Dynamic Beta
    # ----------------------------
    with tab2:
        st.header("Dynamic Factor Loading (Kalman Filter)")
        
        if model is not None:
            fig_beta = plot_dynamic_beta(model, selected_ticker)
            if fig_beta:
                st.plotly_chart(fig_beta, use_container_width=True)
                
                # Current beta
                current_betas = model.layer1_factor.get_current_betas()
                current_beta = current_betas.get(selected_ticker, 1.0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Beta", f"{current_beta:.3f}")
                with col2:
                    avg_beta = np.mean(list(current_betas.values()))
                    st.metric("Avg Beta (All)", f"{avg_beta:.3f}")
                with col3:
                    beta_series = model.get_dynamic_beta_series(selected_ticker)
                    if not beta_series.empty:
                        beta_vol = beta_series.std()
                        st.metric("Beta Volatility", f"{beta_vol:.3f}")
                
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
            st.warning("⚠️ Model not loaded. Enable 'Use Integrated Model' in sidebar.")
    
    # ----------------------------
    # TAB 3: Portfolio
    # ----------------------------
    with tab3:
        st.header("Portfolio Performance")
        
        # Get weights
        if model is not None:
            weights = model.get_optimal_weights(method='mean_variance')
        else:
            # Equal weight
            weights = pd.Series(1.0/len(returns.columns), index=returns.columns)
        
        # Plot cumulative returns
        fig_cum, portfolio_returns = plot_cumulative_returns(returns, weights, benchmark)
        if fig_cum:
            st.plotly_chart(fig_cum, use_container_width=True)
        
        # Metrics
        if portfolio_returns is not None:
            metrics = calculate_metrics(portfolio_returns)
            bench_metrics = calculate_metrics(returns[benchmark]) if benchmark in returns.columns else {}
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                delta = metrics['Total Return'] - bench_metrics.get('Total Return', 0)
                st.metric("Total Return", f"{metrics['Total Return']:.2%}", f"{delta:+.2%}")
            with col2:
                delta = metrics['Sharpe Ratio'] - bench_metrics.get('Sharpe Ratio', 0)
                st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}", f"{delta:+.2f}")
            with col3:
                st.metric("CAGR", f"{metrics['CAGR']:.2%}")
            with col4:
                st.metric("Volatility", f"{metrics['Volatility']:.2%}")
            with col5:
                st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
        
        # Weights
        st.subheader("Current Portfolio Weights")
        top_weights = weights.nlargest(15)
        fig_weights = px.bar(
            x=top_weights.values,
            y=top_weights.index,
            orientation='h',
            labels={'x': 'Weight', 'y': 'Ticker'}
        )
        fig_weights.update_layout(template='plotly_white', height=400)
        st.plotly_chart(fig_weights, use_container_width=True)
    
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
                forecast_1d = garch.forecast(1) * np.sqrt(252) * 100
                forecast_5d = garch.forecast(5) * np.sqrt(252) * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("1-Day Forecast", f"{forecast_1d:.2f}%")
                with col2:
                    st.metric("5-Day Forecast", f"{forecast_5d:.2f}%")
        else:
            st.warning("⚠️ Model not loaded.")
    
    # ----------------------------
    # TAB 5: Summary
    # ----------------------------
    with tab5:
        st.header("Model Summary")
        
        if model is not None:
            summary = model.summary()
            
            st.markdown(f"""
            ### 📊 Data Summary
            - **Assets:** {summary['n_assets']}
            - **Trading Days:** {summary['n_days']}
            - **Date Range:** {summary['date_range'][0].date()} to {summary['date_range'][1].date()}
            - **Sentiment Data:** {'✅ Loaded' if summary['sentiment_loaded'] else '❌ Not loaded'}
            
            ### 🔧 Model Components
            
            #### Layer 1: Dynamic Factor Loading
            - Method: Kalman Filter
            - Average Beta: {summary['avg_beta']:.3f}
            
            #### Layer 2: Sentiment Integration
            - Source: FinBERT (ProsusAI/finbert)
            - Status: {'Active' if summary['sentiment_loaded'] else 'Inactive'}
            
            #### Layer 3: Volatility
            - Method: GARCH-MIDAS
            - Short-term: GARCH(1,1)
            - Long-term: MIDAS (22-day)
            """)
        else:
            st.info("Load the model to see summary.")
        
        # Export
        st.subheader("📥 Export")
        
        if st.button("Generate Report"):
            report = f"""
# AI Finance Project Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Data
- Assets: {len(returns.columns)}
- Trading Days: {len(returns)}

## Performance
"""
            if portfolio_returns is not None:
                metrics = calculate_metrics(portfolio_returns)
                report += f"""
- Total Return: {metrics['Total Return']:.2%}
- Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}
- Max Drawdown: {metrics['Max Drawdown']:.2%}
"""
            
            st.download_button(
                "Download Report",
                report,
                file_name="ai_finance_report.md",
                mime="text/markdown"
            )


if __name__ == "__main__":
    main()
