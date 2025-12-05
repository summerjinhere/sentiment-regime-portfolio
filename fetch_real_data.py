"""
Fetch Real Data - Price and Returns Data Collection
====================================================

This script fetches real market data from Yahoo Finance.
Required for running the AI Finance Dashboard v4.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    print("⚠️ yfinance not installed. Run: pip install yfinance")

# Default universe: Large-cap tech + diversified sectors
DEFAULT_TICKERS = [
    # Tech Giants
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM',
    # Financials
    'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'BLK', 'SCHW', 'AXP', 'V',
    # Healthcare
    'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'TMO', 'LLY', 'BMY', 'AMGN', 'GILD',
    # Consumer
    'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'COST', 'LOW', 'TJX', 'BKNG',
    # Industrial
    'CAT', 'HON', 'UPS', 'BA', 'GE', 'MMM', 'RTX', 'LMT', 'DE', 'UNP',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL',
    # Communication
    'DIS', 'NFLX', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR', 'ATVI', 'EA', 'TTWO',
    # Benchmarks
    'SPY', 'QQQ', 'IWM', 'DIA'
]


def fetch_data(
    tickers: list = None,
    start_date: str = "2005-01-01",
    end_date: str = None,
    output_file: str = "real_backtest_panel.csv"
):
    """
    Fetch historical price data from Yahoo Finance.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
        output_file: Output CSV file path
    """
    if not YF_AVAILABLE:
        print("❌ yfinance not available. Install with: pip install yfinance")
        return None
    
    tickers = tickers or DEFAULT_TICKERS
    end_date = end_date or datetime.now().strftime("%Y-%m-%d")
    
    print("=" * 70)
    print("FETCHING REAL MARKET DATA")
    print("=" * 70)
    print(f"Tickers: {len(tickers)}")
    print(f"Period: {start_date} to {end_date}")
    print()
    
    try:
        print("Downloading from Yahoo Finance (Sequential)...")
        
        all_prices = []
        valid_tickers = []
        
        for ticker in tickers:
            try:
                # Download single ticker
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if df.empty:
                    continue
                
                # Extract Close/Adj Close
                if 'Adj Close' in df.columns:
                    price = df['Adj Close']
                elif 'Close' in df.columns:
                    price = df['Close']
                else:
                    # Fallback
                    price = df.iloc[:, 0] if not df.empty else None
                
                if price is not None:
                    # Rename series to ticker
                    if isinstance(price, pd.DataFrame):
                        price = price.iloc[:, 0]
                    price.name = ticker
                    all_prices.append(price)
                    valid_tickers.append(ticker)
                    print(f"  ✅ {ticker}")
                else:
                    print(f"  ⚠️ {ticker} (No price data)")
                    
            except Exception as e:
                print(f"  ❌ {ticker}: {e}")
                continue
        
        print(f"\nSuccessfully fetched {len(valid_tickers)}/{len(tickers)} tickers.")
        
        if not all_prices:
            print("❌ No data downloaded")
            return None
        
        # Combine into DataFrame
        prices = pd.concat(all_prices, axis=1)
        prices.columns = valid_tickers
        
        # Calculate returns
        returns = prices.pct_change()
        
        # Drop rows with all NaN
        prices = prices.dropna(how='all')
        returns = returns.dropna(how='all')
        
        # Create panel format
        panel = pd.DataFrame(index=returns.index)
        
        for ticker in prices.columns:
            panel[f'{ticker}_price'] = prices[ticker]
            panel[f'{ticker}_ret'] = returns[ticker]
        
        # Drop rows with too many NaN
        panel = panel.dropna(thresh=len(panel.columns) * 0.5)
        
        # Fill remaining NaN with 0 for returns
        ret_cols = [c for c in panel.columns if c.endswith('_ret')]
        panel[ret_cols] = panel[ret_cols].fillna(0)
        
        # Forward fill prices
        price_cols = [c for c in panel.columns if c.endswith('_price')]
        panel[price_cols] = panel[price_cols].ffill()
        
        # Save
        panel.to_csv(output_file)
        
        print()
        print("=" * 70)
        print("DATA COLLECTION COMPLETE")
        print("=" * 70)
        print(f"✅ Saved to: {output_file}")
        print(f"   Trading days: {len(panel)}")
        print(f"   Assets: {len(prices.columns)}")
        print(f"   Date range: {panel.index[0].date()} to {panel.index[-1].date()}")
        print(f"   File size: {panel.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        return panel

    except Exception as e:
        print(f"\n❌ Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch real market data")
    parser.add_argument("--start", default="2005-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="real_backtest_panel.csv", help="Output file")
    parser.add_argument("--tickers", nargs="+", default=None, help="List of tickers")
    
    args = parser.parse_args()
    
    fetch_data(
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        output_file=args.output
    )
