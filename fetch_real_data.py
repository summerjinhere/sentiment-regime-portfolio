"""
Real Market Data Fetcher
========================
Downloads historical price data from Yahoo Finance.
NO SYNTHETIC DATA - Real market data only.
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
import warnings
import sys
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

ASSET_UNIVERSE = {
    'Technology': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'AMD', 'QCOM', 'INTC'],
    'Healthcare': ['LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'TMO', 'ABT', 'PFE', 'AMGN', 'DHR'],
    'Financials': ['JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'BLK', 'C', 'AXP'],
    'Consumer': ['AMZN', 'TSLA', 'HD', 'PG', 'COST', 'WMT', 'KO', 'PEP', 'MCD', 'NKE'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL'],
    'Utilities': ['NEE', 'SO', 'DUK', 'CEG', 'AEP', 'SRE', 'PEG', 'ED', 'EXC', 'XEL'],
    'Bonds': ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'BND', 'AGG', 'MUB', 'TIP', 'BNDX'],
    'Commodities': ['GLD', 'SLV', 'USO', 'UNG', 'DBC', 'DBA', 'GSG', 'PALL', 'PPLT', 'CPER'],
    'Indices': ['SPY', 'QQQ', 'IWM', 'DIA']
}

def get_all_tickers():
    all_tickers = []
    for tickers in ASSET_UNIVERSE.values():
        all_tickers.extend(tickers)
    return list(set(all_tickers))

def main():
    parser = argparse.ArgumentParser(description='Fetch real market data')
    parser.add_argument('--trading-days', type=int, default=5040, help='Number of trading days (~20 years = 5040)')
    parser.add_argument('--output', type=str, default='real_backtest_panel.csv', help='Output file')
    args = parser.parse_args()
    
    print("=" * 70)
    print("REAL DATA FETCHER - Yahoo Finance")
    print("=" * 70)
    
    try:
        import yfinance as yf
    except ImportError:
        print("Error: yfinance not installed. Run: pip install yfinance")
        return
    
    # Calculate date range
    end_date = datetime.now()
    # Add buffer for weekends/holidays
    start_date = end_date - timedelta(days=int(args.trading_days * 1.5))
    
    all_tickers = get_all_tickers()
    print(f"Tickers: {len(all_tickers)}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Target trading days: {args.trading_days}")
    
    # Download
    print("\nDownloading...")
    data = yf.download(
        tickers=all_tickers,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        progress=False,
        auto_adjust=True
    )
    
    # Extract Close prices
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data
    
    # Clean
    prices = prices.dropna(axis=1, how='all')
    prices = prices.ffill().bfill()
    
    # Limit to trading days
    if len(prices) > args.trading_days:
        prices = prices.iloc[-args.trading_days:]
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Create combined DataFrame
    prices_df = prices.copy()
    prices_df.columns = [f"{col}_price" for col in prices_df.columns]
    
    returns_df = returns.copy()
    returns_df.columns = [f"{col}_ret" for col in returns_df.columns]
    
    common_idx = prices_df.index.intersection(returns_df.index)
    combined = pd.concat([prices_df.loc[common_idx], returns_df.loc[common_idx]], axis=1)
    combined.index.name = 'Date'
    
    # Save
    combined.to_csv(args.output)
    
    print(f"\n✅ Saved to {args.output}")
    print(f"   Trading days: {len(combined)}")
    print(f"   Tickers: {len(prices.columns)}")
    print(f"   Date range: {combined.index[0].date()} to {combined.index[-1].date()}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY BY SECTOR")
    print("=" * 70)
    for sector, tickers in ASSET_UNIVERSE.items():
        available = [t for t in tickers if t in returns.columns]
        if available:
            sect_ret = returns[available].mean(axis=1)
            ann_ret = sect_ret.mean() * 252
            ann_vol = sect_ret.std() * np.sqrt(252)
            print(f"{sector:12} | {len(available):2} tickers | Ret: {ann_ret:6.1%} | Vol: {ann_vol:6.1%}")

if __name__ == "__main__":
    main()
