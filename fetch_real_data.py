import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

def fetch_real_data():
    print("Downloading real historical data from yfinance...")
    
    # 1. Define Tickers (Expanded Universe)
    assets_map = {
        'Technology': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'AMD', 'QCOM', 'INTC'],
        'Healthcare': ['LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'TMO', 'ABT', 'PFE', 'AMGN', 'DHR'],
        'Financials': ['JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'BLK', 'C', 'AXP'],
        'Consumer': ['AMZN', 'TSLA', 'HD', 'PG', 'COST', 'WMT', 'KO', 'PEP', 'MCD', 'NKE'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL'],
        'Utilities': ['NEE', 'SO', 'DUK', 'CEG', 'AEP', 'SRE', 'PEG', 'ED', 'EXC', 'XEL'],
        'Bonds': ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'BND', 'AGG', 'MUB', 'TIP', 'BNDX'],
        'Commodities': ['GLD', 'SLV', 'USO', 'UNG', 'DBC', 'DBA', 'GSG', 'PALL', 'PPLT', 'CPER']
    }
    
    all_assets = []
    for sector_tickers in assets_map.values():
        all_assets.extend(sector_tickers)
        
    benchmarks = ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "XLK", "XLF", "XLE", "XLV"]
    
    # Factors & Macro
    factors = ['SPY', 'QQQ', 'IWM'] 
    macro = ['^VIX', '^TNX'] 
    
    all_tickers = list(set(all_assets + benchmarks + factors + macro))
    
    # 2. Download Data (Last 2 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    print(f"Fetching {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
    
    if data.empty:
        raise RuntimeError("Failed to download data from yfinance.")
        
    # Handle MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data
        
    # Drop columns with all NaNs
    prices = prices.dropna(axis=1, how='all')
    
    # Forward fill then drop remaining NaNs
    prices = prices.ffill().dropna()
    
    if len(prices) < 252:
        raise RuntimeError(f"Insufficient data history. Got {len(prices)} days.")
        
    print(f"Downloaded {len(prices)} days of data.")
    
    # 3. Calculate Returns
    returns = prices.pct_change().dropna()
    
    # 4. Prepare real_backtest_panel.csv
    df_panel = pd.DataFrame()
    df_panel['date'] = returns.index
    
    # Assets
    for asset in all_assets:
        if asset in returns.columns:
            df_panel[f'ret_asset_{asset}'] = returns[asset].values
            
    # Factors
    for i, factor in enumerate(factors):
        if factor in returns.columns:
            df_panel[f'factor_{i}'] = returns[factor].values
            
    # Macro & Sentiment
    if '^VIX' in prices.columns:
        df_panel['VIX'] = prices['^VIX'].values[1:] 
    else:
        df_panel['VIX'] = 20.0 
        
    if '^TNX' in prices.columns:
        df_panel['TERM_SPREAD'] = prices['^TNX'].values[1:]
    else:
        df_panel['TERM_SPREAD'] = 1.5
        
    df_panel['CREDIT_SPREAD'] = df_panel['VIX'] * 0.1 
    
    vix = df_panel['VIX']
    df_panel['sentiment'] = ((vix.mean() - vix) / vix.std()).clip(-2, 2) / 2 
    
    df_panel.to_csv('real_backtest_panel.csv', index=False)
    print("Saved real_backtest_panel.csv")
    
    # 5. Prepare real_backtest_results.csv
    # Strategy: Equal Weight of Assets
    asset_cols = [c for c in df_panel.columns if c.startswith('ret_asset_')]
    strategy_ret = df_panel[asset_cols].mean(axis=1)
    
    df_results = pd.DataFrame({
        'date': df_panel['date'],
        'strategy_ret': strategy_ret
    })
    
    # Add all benchmarks
    for bench in benchmarks:
        if bench in returns.columns:
            df_results[f'benchmark_{bench}'] = returns[bench].values
        else:
            print(f"Warning: Benchmark {bench} missing from data.")
            
    # Ensure primary benchmark exists (SPY)
    if 'benchmark_SPY' not in df_results.columns:
         df_results['benchmark_SPY'] = strategy_ret # Fallback only if SPY download failed completely
    
    # Alias benchmark_ret to SPY for backward compatibility
    df_results['benchmark_ret'] = df_results['benchmark_SPY']
    
    df_results.to_csv('real_backtest_results.csv', index=False)
    print("Saved real_backtest_results.csv")

if __name__ == "__main__":
    fetch_real_data()
