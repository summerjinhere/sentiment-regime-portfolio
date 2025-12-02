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
    
    def load_data(self) -> Dict:
        """Load REAL backtest results from CSV"""
        try:
            df = pd.read_csv("real_backtest_results.csv", parse_dates=["date"])
            df = df.sort_values("date")
            print(f"Loaded real backtest results: {len(df)} rows")
        except FileNotFoundError:
             raise FileNotFoundError("CRITICAL: 'real_backtest_results.csv' not found. You must run 'fetch_real_data.py' first.")

        dates = df["date"]
        strategy_ret = df["strategy_ret"].values
        
        # Load all benchmarks
        benchmark_cols = [c for c in df.columns if c.startswith('benchmark_')]
        benchmarks = {}
        for col in benchmark_cols:
            name = col.replace('benchmark_', '')
            benchmarks[name] = df[col].values

        # Cumulative Value (Strategy)
        portfolio_value = 100000 * (1 + pd.Series(strategy_ret, index=dates)).cumprod().values
        
        # Cumulative Value (Benchmarks)
        benchmark_values = {}
        for name, ret in benchmarks.items():
            benchmark_values[name] = 100000 * (1 + pd.Series(ret, index=dates)).cumprod().values

        # Volatility (Realized)
        realized_vol = pd.Series(strategy_ret, index=dates).rolling(21).std().fillna(0).values * np.sqrt(252)
        
        # Placeholders
        regimes = np.zeros(len(dates), dtype=int) 
        regime_probs = np.tile([1.0, 0.0, 0.0], (len(dates), 1))
        sentiment = np.zeros(len(dates))
        vol_predictions = {'Realized': realized_vol}

        return {
            "dates": dates,
            "returns": strategy_ret,
            "portfolio_value": portfolio_value,
            "benchmarks": benchmarks, # Dictionary of returns
            "benchmark_values": benchmark_values, # Dictionary of cumulative values
            "regimes": regimes,
            "regime_probs": regime_probs,
            "sentiment": sentiment,
            "realized_vol": realized_vol,
            "vol_predictions": vol_predictions,
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
        """Plot cumulative returns with multiple benchmarks"""
        fig = go.Figure()
        
        # Strategy
        fig.add_trace(go.Scatter(
            x=data['dates'],
            y=data['portfolio_value'],
            name='Strategy',
            line=dict(color=self.colors['strategy'], width=3)
        ))
        
        # Benchmarks
        if 'benchmark_values' in data:
            for name, values in data['benchmark_values'].items():
                # Highlight SPY, dim others
                is_primary = name == 'SPY'
                width = 2 if is_primary else 1
                opacity = 1.0 if is_primary else 0.6
                visible = True if is_primary else 'legendonly'
                
                fig.add_trace(go.Scatter(
                    x=data['dates'],
                    y=values,
                    name=f'Benchmark ({name})',
                    line=dict(width=width),
                    opacity=opacity,
                    visible=visible
                ))
        
        fig.update_layout(
            title='Cumulative Portfolio Performance (vs Benchmarks)',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            template='plotly_white',
            hovermode='x unified',
            height=500
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
        # Placeholder since we don't have real regimes yet
        return go.Figure()
        
    def plot_regime_probs(self, data: Dict) -> go.Figure:
         return go.Figure()

    def plot_sentiment_analysis(self, data: Dict) -> go.Figure:
         return go.Figure()

    def plot_volatility_comparison(self, data: Dict) -> go.Figure:
         return go.Figure()

    def plot_rolling_sharpe(self, data: Dict) -> go.Figure:
         return go.Figure()

    def generate_html_report(self, data: Dict, charts: Dict, metrics: Dict, benchmark_metrics_list: List[Dict]) -> str:
        """Generate comprehensive HTML report"""
        
        # Primary benchmark (SPY) for delta comparison
        primary_bench = next((b for b in benchmark_metrics_list if b['Name'] == 'SPY'), benchmark_metrics_list[0] if benchmark_metrics_list else {})
        
        # Benchmark Table Rows
        bench_rows = ""
        for bm in benchmark_metrics_list:
            bench_rows += f"""
            <tr>
                <td>{bm['Name']}</td>
                <td>{bm.get('Total Return', 0):.2%}</td>
                <td>{bm.get('Sharpe Ratio', 0):.2f}</td>
                <td>{bm.get('Max Drawdown', 0):.2%}</td>
                <td>{bm.get('Volatility', 0):.2%}</td>
            </tr>
            """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced Portfolio Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f5f7fa; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }}
                .header {{ text-align: center; margin-bottom: 40px; border-bottom: 2px solid #eee; padding-bottom: 20px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px; margin-bottom: 40px; }}
                .metric-card {{ background: #fff; padding: 20px; border-radius: 10px; border: 1px solid #eee; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; margin-top: 5px; }}
                .metric-delta {{ font-size: 12px; color: #27ae60; margin-top: 5px; }}
                .chart-container {{ margin-bottom: 40px; border: 1px solid #eee; border-radius: 10px; padding: 10px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
                th {{ background-color: #f8f9fa; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Enhanced Portfolio Report</h1>
                    <p>Real Data Analysis (Expanded Universe)</p>
                </div>
                
                <h2>Strategy Performance</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{metrics['Total Return']:.2%}</div>
                        <div class="metric-label">Total Return</div>
                        <div class="metric-delta">vs SPY: {metrics['Total Return'] - primary_bench.get('Total Return', 0):+.2%}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics['Sharpe Ratio']:.2f}</div>
                        <div class="metric-label">Sharpe Ratio</div>
                        <div class="metric-delta">vs SPY: {metrics['Sharpe Ratio'] - primary_bench.get('Sharpe Ratio', 0):+.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics['Max Drawdown']:.2%}</div>
                        <div class="metric-label">Max Drawdown</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics['Volatility']:.2%}</div>
                        <div class="metric-label">Volatility</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    {charts['cumulative']}
                </div>
                
                <h2>Benchmark Comparison</h2>
                <table>
                    <tr>
                        <th>Benchmark</th>
                        <th>Total Return</th>
                        <th>Sharpe Ratio</th>
                        <th>Max Drawdown</th>
                        <th>Volatility</th>
                    </tr>
                    {bench_rows}
                </table>
                
                <div class="chart-container">
                    {charts['drawdown']}
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
        
        print("\n1. Loading REAL backtest data...")
        data = self.load_data()
        
        print("2. Calculating metrics...")
        strategy_metrics = self.calculate_metrics(data['returns'])
        
        benchmark_metrics_list = []
        if 'benchmarks' in data:
            for name, ret in data['benchmarks'].items():
                metrics = self.calculate_metrics(ret)
                metrics['Name'] = name
                benchmark_metrics_list.append(metrics)
        
        print("\nStrategy Performance:")
        for k, v in strategy_metrics.items():
            print(f"  {k}: {v:.4f}")
            
        print("\nBenchmark Performance:")
        for bm in benchmark_metrics_list:
            print(f"  {bm['Name']}: Sharpe={bm.get('Sharpe Ratio', 0):.2f}, Return={bm.get('Total Return', 0):.2%}")
        
        print("\n3. Generating visualizations...")
        figs = {}
        figs['cumulative'] = self.plot_cumulative_returns(data).to_html(full_html=False, include_plotlyjs=False)
        figs['drawdown'] = self.plot_drawdown(data).to_html(full_html=False, include_plotlyjs=False)
        # Placeholders for others
        figs['regimes'] = ""
        figs['regime_probs'] = ""
        figs['sentiment'] = ""
        figs['volatility'] = ""
        figs['rolling_sharpe'] = ""
        
        print("4. Saving report...")
        html = self.generate_html_report(data, figs, strategy_metrics, benchmark_metrics_list)
        filepath = self.save_report(html)
        
        print(f"\n✅ Report saved to: {filepath}")
        print("=" * 60)
        
        return filepath


if __name__ == "__main__":
    viz = PortfolioVisualizer(output_dir="reports")
    report_path = viz.run()
