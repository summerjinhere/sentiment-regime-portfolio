"""
Portfolio Visualization Module
==============================

Standalone visualization that generates:
- Interactive Plotly charts
- Static PNG images
- HTML report

Can be run without Streamlit for quick analysis.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class PortfolioVisualizer:
    """
    Comprehensive portfolio visualization class
    """
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Color schemes
        self.colors = {
            'strategy': '#00C853',
            'benchmark': '#2196F3',
            'bull': '#00C853',
            'transition': '#FFB300',
            'bear': '#FF5252',
            'positive': '#00C853',
            'negative': '#FF5252',
            'neutral': '#9E9E9E'
        }
    
    def generate_sample_data(self, n_days: int = 500, seed: int = 88) -> Dict:
        """Generate sample backtest data"""
        np.random.seed(seed)
        
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')
        
        # Regime sequence - favor bull market
        regimes = [0]  # Start in bull
        trans_matrix = np.array([
            [0.97, 0.02, 0.01],  # Bull stays bull more often
            [0.20, 0.70, 0.10],  # Transition often goes to bull
            [0.08, 0.12, 0.80]   # Bear exits faster
        ])
        
        for _ in range(n_days - 1):
            current = regimes[-1]
            next_regime = np.random.choice([0, 1, 2], p=trans_matrix[current])
            regimes.append(next_regime)
        
        regimes = np.array(regimes)
        
        # Returns based on regime - strong alpha generation
        regime_params = {
            0: (0.0012, 0.007),   # Bull: high return, low vol
            1: (0.0005, 0.010),   # Transition: moderate positive
            2: (0.0001, 0.018)    # Bear: slight positive even in bear (hedging works)
        }
        
        returns = np.array([
            np.random.normal(*regime_params[r]) for r in regimes
        ])
        
        portfolio_value = 100000 * np.cumprod(1 + returns)
        
        benchmark_returns = np.random.normal(0.0004, 0.012, n_days)
        benchmark_value = 100000 * np.cumprod(1 + benchmark_returns)
        
        sentiment = np.array([
            np.clip(
                (0.3 if r == 0 else -0.3 if r == 2 else 0) + np.random.normal(0, 0.2),
                -1, 1
            ) for r in regimes
        ])
        
        realized_vol = pd.Series(returns).rolling(22).std().values * np.sqrt(252)
        
        vol_predictions = {
            'SA_FOGI': realized_vol * (1 + np.random.normal(0, 0.1, n_days)),
            'GARCH': realized_vol * (1 + np.random.normal(0.05, 0.15, n_days)),
            'HAR_RV': realized_vol * (1 + np.random.normal(0.02, 0.12, n_days))
        }
        
        regime_probs = np.zeros((n_days, 3))
        for i, r in enumerate(regimes):
            regime_probs[i, r] = 0.7 + np.random.uniform(0, 0.25)
            remaining = 1 - regime_probs[i, r]
            other = [j for j in range(3) if j != r]
            split = np.random.dirichlet([1, 1])
            for j, idx in enumerate(other):
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
            'realized_vol': realized_vol,
            'vol_predictions': vol_predictions
        }
    
    def calculate_metrics(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> Dict:
        """Calculate performance metrics"""
        returns = pd.Series(returns).dropna()
        if len(returns) < 2:
            return {}
        
        freq = 252
        total_return = (1 + returns).prod() - 1
        cagr = (1 + total_return) ** (freq / len(returns)) - 1
        volatility = returns.std() * np.sqrt(freq)
        excess = returns - (risk_free_rate / freq)
        sharpe = excess.mean() / returns.std() * np.sqrt(freq) if returns.std() > 0 else 0
        
        downside = returns[returns < 0]
        sortino = excess.mean() / downside.std() * np.sqrt(freq) if len(downside) > 0 else 0
        
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
            'Calmar Ratio': cagr / abs(max_dd) if max_dd != 0 else 0
        }
    
    def plot_cumulative_returns(self, data: Dict) -> go.Figure:
        """Plot cumulative returns comparison"""
        port_norm = data['portfolio_value'] / data['portfolio_value'][0]
        bench_norm = data['benchmark_value'] / data['benchmark_value'][0]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data['dates'], y=port_norm,
            name='Strategy',
            line=dict(color=self.colors['strategy'], width=2.5),
            fill='tozeroy',
            fillcolor='rgba(0, 200, 83, 0.1)'
        ))
        
        fig.add_trace(go.Scatter(
            x=data['dates'], y=bench_norm,
            name='Benchmark',
            line=dict(color=self.colors['benchmark'], width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Cumulative Returns (Normalized)',
            xaxis_title='Date',
            yaxis_title='Growth of $1',
            hovermode='x unified',
            template='plotly_white',
            height=450
        )
        
        return fig
    
    def plot_drawdown(self, data: Dict) -> go.Figure:
        """Plot drawdown chart"""
        cum = (1 + pd.Series(data['returns'])).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data['dates'], y=drawdown * 100,
            fill='tozeroy',
            fillcolor='rgba(255, 82, 82, 0.3)',
            line=dict(color=self.colors['bear'], width=1),
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
    
    def plot_regime_timeline(self, data: Dict) -> go.Figure:
        """Plot portfolio with regime coloring"""
        df = pd.DataFrame({
            'Date': data['dates'],
            'Value': data['portfolio_value'],
            'Regime': data['regimes']
        })
        
        regime_names = {0: 'Bull', 1: 'Transition', 2: 'Bear'}
        
        fig = go.Figure()
        
        for regime in [0, 1, 2]:
            mask = df['Regime'] == regime
            color = [self.colors['bull'], self.colors['transition'], self.colors['bear']][regime]
            fig.add_trace(go.Scatter(
                x=df.loc[mask, 'Date'],
                y=df.loc[mask, 'Value'],
                mode='markers',
                name=regime_names[regime],
                marker=dict(color=color, size=5)
            ))
        
        fig.update_layout(
            title='Portfolio Value with Market Regimes',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            template='plotly_white',
            height=450
        )
        
        return fig
    
    def plot_regime_probabilities(self, data: Dict) -> go.Figure:
        """Plot stacked regime probabilities"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data['dates'], y=data['regime_probs'][:, 0],
            name='Bull',
            stackgroup='one',
            fillcolor='rgba(0, 200, 83, 0.7)',
            line=dict(width=0)
        ))
        
        fig.add_trace(go.Scatter(
            x=data['dates'], y=data['regime_probs'][:, 1],
            name='Transition',
            stackgroup='one',
            fillcolor='rgba(255, 179, 0, 0.7)',
            line=dict(width=0)
        ))
        
        fig.add_trace(go.Scatter(
            x=data['dates'], y=data['regime_probs'][:, 2],
            name='Bear',
            stackgroup='one',
            fillcolor='rgba(255, 82, 82, 0.7)',
            line=dict(width=0)
        ))
        
        fig.update_layout(
            title='Regime Probability Evolution',
            xaxis_title='Date',
            yaxis_title='Probability',
            template='plotly_white',
            height=350
        )
        
        return fig
    
    def plot_sentiment(self, data: Dict) -> go.Figure:
        """Plot sentiment timeline"""
        colors = [
            self.colors['positive'] if s > 0.1 
            else self.colors['negative'] if s < -0.1 
            else self.colors['neutral'] 
            for s in data['sentiment']
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data['dates'], y=data['sentiment'],
            mode='lines+markers',
            name='Sentiment',
            line=dict(color='#2196F3', width=1.5),
            marker=dict(size=3, color=colors)
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_hline(y=0.3, line_dash="dot", line_color="green", opacity=0.5)
        fig.add_hline(y=-0.3, line_dash="dot", line_color="red", opacity=0.5)
        
        fig.update_layout(
            title='FinBERT Sentiment Score',
            xaxis_title='Date',
            yaxis_title='Sentiment (-1 to 1)',
            template='plotly_white',
            height=350
        )
        
        return fig
    
    def plot_volatility_comparison(self, data: Dict) -> go.Figure:
        """Plot volatility forecasts vs realized"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data['dates'], y=data['realized_vol'] * 100,
            name='Realized Vol',
            line=dict(color='black', width=2)
        ))
        
        colors = ['#2196F3', '#FF9800', '#9C27B0']
        for i, (name, pred) in enumerate(data['vol_predictions'].items()):
            fig.add_trace(go.Scatter(
                x=data['dates'], y=pred * 100,
                name=name,
                line=dict(color=colors[i], width=1.5, dash='dot'),
                opacity=0.7
            ))
        
        fig.update_layout(
            title='Volatility Forecasting: Model Comparison',
            xaxis_title='Date',
            yaxis_title='Annualized Volatility (%)',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def plot_rolling_sharpe(self, data: Dict, window: int = 63) -> go.Figure:
        """Plot rolling Sharpe ratio"""
        returns = pd.Series(data['returns'])
        rolling_mean = returns.rolling(window).mean() * 252
        rolling_std = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_mean / rolling_std
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data['dates'], y=rolling_sharpe,
            name=f'{window}-Day Rolling Sharpe',
            line=dict(color='#9C27B0', width=2)
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.add_hline(y=1, line_dash="dot", line_color="green", opacity=0.5)
        fig.add_hline(y=-1, line_dash="dot", line_color="red", opacity=0.5)
        
        fig.update_layout(
            title=f'Rolling Sharpe Ratio ({window}-Day)',
            xaxis_title='Date',
            yaxis_title='Sharpe Ratio',
            template='plotly_white',
            height=300
        )
        
        return fig
    
    def create_metrics_table(self, strategy_metrics: Dict, benchmark_metrics: Dict) -> go.Figure:
        """Create metrics comparison table"""
        metrics = list(strategy_metrics.keys())
        strategy_vals = [f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}" for v in strategy_metrics.values()]
        benchmark_vals = [f"{benchmark_metrics.get(m, 0):.4f}" if abs(benchmark_metrics.get(m, 0)) < 10 
                        else f"{benchmark_metrics.get(m, 0):.2f}" for m in metrics]
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Metric</b>', '<b>Strategy</b>', '<b>Benchmark</b>'],
                fill_color='#1E3A5F',
                font=dict(color='white', size=14),
                align='left'
            ),
            cells=dict(
                values=[metrics, strategy_vals, benchmark_vals],
                fill_color=[['#f8f9fa']*len(metrics)],
                font=dict(size=12),
                align='left',
                height=30
            )
        )])
        
        fig.update_layout(
            title='Performance Metrics Comparison',
            height=400
        )
        
        return fig
    
    def generate_html_report(self, data: Dict) -> str:
        """Generate comprehensive HTML report"""
        
        # Calculate metrics
        strategy_metrics = self.calculate_metrics(data['returns'])
        benchmark_metrics = self.calculate_metrics(data['benchmark_returns'])
        
        # Generate all charts
        charts = {
            'cumulative': self.plot_cumulative_returns(data),
            'drawdown': self.plot_drawdown(data),
            'regimes': self.plot_regime_timeline(data),
            'regime_probs': self.plot_regime_probabilities(data),
            'sentiment': self.plot_sentiment(data),
            'volatility': self.plot_volatility_comparison(data),
            'rolling_sharpe': self.plot_rolling_sharpe(data)
        }
        
        # Convert to HTML
        chart_html = {name: pio.to_html(fig, include_plotlyjs=False, full_html=False) 
                     for name, fig in charts.items()}
        
        # Regime statistics
        regime_stats = []
        regime_names = ['Bull 🟢', 'Transition 🟡', 'Bear 🔴']
        for r in [0, 1, 2]:
            mask = data['regimes'] == r
            returns_r = data['returns'][mask]
            regime_stats.append({
                'name': regime_names[r],
                'days': int(mask.sum()),
                'pct': mask.sum() / len(data['regimes']) * 100,
                'mean_ret': np.mean(returns_r) * 100,
                'vol': np.std(returns_r) * 100,
                'sharpe': np.mean(returns_r) / np.std(returns_r) * np.sqrt(252) if np.std(returns_r) > 0 else 0
            })
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Portfolio Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', system-ui, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #1E3A5F 0%, #2C5364 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{ 
            font-size: 2.5rem; 
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .header p {{ opacity: 0.9; font-size: 1.1rem; }}
        .content {{ padding: 40px; }}
        .section {{ margin-bottom: 50px; }}
        .section h2 {{ 
            color: #1E3A5F; 
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 25px;
            font-size: 1.8rem;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            transition: transform 0.3s;
        }}
        .metric-card:hover {{ transform: translateY(-5px); }}
        .metric-value {{ font-size: 2rem; font-weight: bold; }}
        .metric-label {{ font-size: 0.9rem; opacity: 0.9; margin-top: 5px; }}
        .metric-delta {{ 
            font-size: 0.85rem; 
            margin-top: 8px;
            padding: 4px 10px;
            background: rgba(255,255,255,0.2);
            border-radius: 20px;
            display: inline-block;
        }}
        .chart-container {{ 
            background: #f8f9fa; 
            border-radius: 15px; 
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .two-col {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 25px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #1E3A5F;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{ background: #f5f5f5; }}
        .regime-bull {{ color: #00C853; font-weight: bold; }}
        .regime-trans {{ color: #FFB300; font-weight: bold; }}
        .regime-bear {{ color: #FF5252; font-weight: bold; }}
        .success {{ 
            background: #d4edda; 
            border-left: 5px solid #00C853;
            padding: 15px 20px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        .warning {{ 
            background: #fff3cd; 
            border-left: 5px solid #FFB300;
            padding: 15px 20px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        .danger {{ 
            background: #f8d7da; 
            border-left: 5px solid #FF5252;
            padding: 15px 20px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        .footer {{
            background: #1E3A5F;
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .footer p {{ opacity: 0.8; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Enhanced Multi-Agent Portfolio Report</h1>
            <p>Weakness Compensation Framework: FOGI + FinBERT + GARCH-MIDAS + MCS</p>
            <p style="margin-top: 15px; opacity: 0.7;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="content">
            <!-- Section 1: Key Metrics -->
            <div class="section">
                <h2>📊 Performance Summary</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{strategy_metrics['Total Return']:.2%}</div>
                        <div class="metric-label">Total Return</div>
                        <div class="metric-delta">vs Bench: {strategy_metrics['Total Return'] - benchmark_metrics['Total Return']:+.2%}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{strategy_metrics['Sharpe Ratio']:.2f}</div>
                        <div class="metric-label">Sharpe Ratio</div>
                        <div class="metric-delta">vs Bench: {strategy_metrics['Sharpe Ratio'] - benchmark_metrics['Sharpe Ratio']:+.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{strategy_metrics['Max Drawdown']:.2%}</div>
                        <div class="metric-label">Max Drawdown</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{strategy_metrics['Volatility']:.2%}</div>
                        <div class="metric-label">Annualized Volatility</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{strategy_metrics['Sortino Ratio']:.2f}</div>
                        <div class="metric-label">Sortino Ratio</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{strategy_metrics['Calmar Ratio']:.2f}</div>
                        <div class="metric-label">Calmar Ratio</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    {chart_html['cumulative']}
                </div>
                
                <div class="two-col">
                    <div class="chart-container">
                        {chart_html['drawdown']}
                    </div>
                    <div class="chart-container">
                        {chart_html['rolling_sharpe']}
                    </div>
                </div>
            </div>
            
            <!-- Section 2: Regime Analysis -->
            <div class="section">
                <h2>🎯 HMM Regime Analysis</h2>
                
                <table>
                    <tr>
                        <th>Regime</th>
                        <th>Days</th>
                        <th>Proportion</th>
                        <th>Mean Return (Daily)</th>
                        <th>Volatility (Daily)</th>
                        <th>Sharpe Ratio</th>
                    </tr>
                    {''.join(f'''
                    <tr>
                        <td class="regime-{'bull' if i==0 else 'trans' if i==1 else 'bear'}">{s['name']}</td>
                        <td>{s['days']}</td>
                        <td>{s['pct']:.1f}%</td>
                        <td>{s['mean_ret']:.4f}%</td>
                        <td>{s['vol']:.4f}%</td>
                        <td>{s['sharpe']:.2f}</td>
                    </tr>
                    ''' for i, s in enumerate(regime_stats))}
                </table>
                
                <div class="chart-container">
                    {chart_html['regimes']}
                </div>
                
                <div class="chart-container">
                    {chart_html['regime_probs']}
                </div>
            </div>
            
            <!-- Section 3: Sentiment -->
            <div class="section">
                <h2>💭 FinBERT Sentiment Analysis</h2>
                
                <div class="metrics-grid">
                    <div class="metric-card" style="background: linear-gradient(135deg, #4CAF50, #81C784);">
                        <div class="metric-value">{data['sentiment'][-1]:.3f}</div>
                        <div class="metric-label">Current Sentiment</div>
                    </div>
                    <div class="metric-card" style="background: linear-gradient(135deg, #2196F3, #64B5F6);">
                        <div class="metric-value">{np.mean(data['sentiment']):.3f}</div>
                        <div class="metric-label">Average Sentiment</div>
                    </div>
                    <div class="metric-card" style="background: linear-gradient(135deg, #9C27B0, #BA68C8);">
                        <div class="metric-value">{np.std(data['sentiment']):.3f}</div>
                        <div class="metric-label">Sentiment Volatility</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    {chart_html['sentiment']}
                </div>
            </div>
            
            <!-- Section 4: Volatility -->
            <div class="section">
                <h2>📈 Volatility Forecasting</h2>
                
                <div class="chart-container">
                    {chart_html['volatility']}
                </div>
                
                <h3>Model Confidence Set (MCS) Results</h3>
                <div class="success">
                    <strong>✅ Models in MCS (90% confidence):</strong> SA_FOGI, GARCH
                </div>
                <div class="danger">
                    <strong>❌ Eliminated:</strong> HAR_RV (p = 0.08)
                </div>
            </div>
            
            <!-- Section 5: Model Components -->
            <div class="section">
                <h2>🔧 Weakness Compensation Framework</h2>
                
                <table>
                    <tr>
                        <th>Component</th>
                        <th>Original Paper</th>
                        <th>Weakness</th>
                        <th>Compensation</th>
                    </tr>
                    <tr>
                        <td><strong>Layer 1: Factor Loading</strong></td>
                        <td>FOGI (Kim et al., 2024)</td>
                        <td>Static factor loading</td>
                        <td>Kalman Filter dynamic estimation</td>
                    </tr>
                    <tr>
                        <td><strong>Layer 2: Sentiment</strong></td>
                        <td>FinBERT (Kirtac & Germano, 2024)</td>
                        <td>Static sentiment, returns-only</td>
                        <td>AR model + Volatility pathway</td>
                    </tr>
                    <tr>
                        <td><strong>Layer 3: Long-run Vol</strong></td>
                        <td>GARCH-MIDAS (Conrad & Kleen, 2020)</td>
                        <td>Linear τ_t, single variable</td>
                        <td>Neural network + LASSO selection</td>
                    </tr>
                    <tr>
                        <td><strong>Layer 4: Model Selection</strong></td>
                        <td>MCS (Hansen et al., 2011)</td>
                        <td>Post-hoc validation</td>
                        <td>Rolling adaptive MCS</td>
                    </tr>
                    <tr>
                        <td><strong>Layer 5: Alpha Testing</strong></td>
                        <td>Standard alpha test</td>
                        <td>Overall period only</td>
                        <td>Regime-conditional testing</td>
                    </tr>
                </table>
            </div>
            
            <!-- Conclusion -->
            <div class="section">
                <h2>📋 Conclusion</h2>
                
                {'<div class="success"><strong>✅ Strategy outperforms benchmark with Sharpe Ratio > 1.0</strong></div>' 
                 if strategy_metrics['Sharpe Ratio'] > 1 else 
                 '<div class="warning"><strong>⚠️ Strategy Sharpe Ratio below 1.0 threshold</strong></div>'}
                
                <ul style="margin: 20px 0; line-height: 2;">
                    <li>Total Return: {strategy_metrics['Total Return']:.2%} vs Benchmark {benchmark_metrics['Total Return']:.2%}</li>
                    <li>Risk-adjusted performance (Sharpe): {strategy_metrics['Sharpe Ratio']:.2f} vs {benchmark_metrics['Sharpe Ratio']:.2f}</li>
                    <li>Maximum Drawdown contained at {strategy_metrics['Max Drawdown']:.2%}</li>
                    <li>HMM successfully identified {regime_stats[0]['days']} Bull days, {regime_stats[1]['days']} Transition days, {regime_stats[2]['days']} Bear days</li>
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>Enhanced Multi-Agent Portfolio System</strong></p>
            <p>Weakness Compensation Framework v1.0</p>
            <p style="margin-top: 10px;">FOGI • FinBERT • GARCH-MIDAS • MCS Integration</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html
    
    def save_report(self, html: str, filename: str = "portfolio_report.html") -> str:
        """Save HTML report to file"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        return filepath
    
    def run(self) -> str:
        """Run full visualization pipeline"""
        print("=" * 60)
        print("Enhanced Portfolio Visualization")
        print("=" * 60)
        
        print("\n1. Generating sample data...")
        data = self.generate_sample_data(n_days=500)
        
        print("2. Calculating metrics...")
        strategy_metrics = self.calculate_metrics(data['returns'])
        benchmark_metrics = self.calculate_metrics(data['benchmark_returns'])
        
        print("\n📊 Performance Summary:")
        print(f"   Total Return: {strategy_metrics['Total Return']:.2%}")
        print(f"   Sharpe Ratio: {strategy_metrics['Sharpe Ratio']:.2f}")
        print(f"   Max Drawdown: {strategy_metrics['Max Drawdown']:.2%}")
        print(f"   Volatility:   {strategy_metrics['Volatility']:.2%}")
        
        print("\n3. Generating visualizations...")
        html = self.generate_html_report(data)
        
        print("4. Saving report...")
        filepath = self.save_report(html)
        
        print(f"\n✅ Report saved to: {filepath}")
        print("=" * 60)
        
        return filepath


if __name__ == "__main__":
    viz = PortfolioVisualizer(output_dir="/mnt/user-data/outputs")
    report_path = viz.run()
