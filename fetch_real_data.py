"""
Real Market Data Fetcher
========================

Downloads historical price data from Yahoo Finance for the complete asset universe.
Saves adjusted close prices and returns to CSV files.

NO SYNTHETIC DATA - Real market data only.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# ASSET UNIVERSE DEFINITION
# =============================================================================

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
    """Get flat list of all tickers"""
    all_tickers = []
    for sector, tickers in ASSET_UNIVERSE.items():
        all_tickers.extend(tickers)
    return list(set(all_tickers))


def fetch_price_data(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Fetch adjusted close prices from Yahoo Finance
    
    Args:
        start_date: Start date (default: 2 years ago)
        end_date: End date (default: today)
    
    Returns:
        DataFrame with adjusted close prices
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    all_tickers = get_all_tickers()
    
    print(f"=" * 70)
    print(f"REAL DATA FETCHER - Yahoo Finance")
    print(f"=" * 70)
    print(f"Period: {start_date} to {end_date}")
    print(f"Total Tickers: {len(all_tickers)}")
    print(f"=" * 70)
    
    # Download all at once (more efficient)
    print("\nDownloading price data...")
    data = yf.download(
        tickers=all_tickers,
        start=start_date,
        end=end_date,
        progress=True,
        auto_adjust=True  # Use adjusted prices
    )
    
    # Extract Close prices
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data
    
    # Clean data
    prices = prices.dropna(axis=1, how='all')  # Remove tickers with no data
    prices = prices.ffill().bfill()  # Forward/backward fill missing values
    
    print(f"\n✅ Successfully fetched {len(prices.columns)} tickers")
    print(f"   Trading days: {len(prices)}")
    
    return prices


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns"""
    returns = prices.pct_change().dropna()
    return returns


def create_sector_mapping(prices: pd.DataFrame) -> pd.DataFrame:
    """Create ticker to sector mapping"""
    mapping = []
    for sector, tickers in ASSET_UNIVERSE.items():
        for ticker in tickers:
            if ticker in prices.columns:
                mapping.append({'Ticker': ticker, 'Sector': sector})
    return pd.DataFrame(mapping)


def save_data(prices: pd.DataFrame, output_file: str = 'real_backtest_panel.csv'):
    """
    Save prices and returns to CSV
    
    Format:
    - Date column
    - Price columns for each ticker
    - Return columns for each ticker (suffixed with _ret)
    """
    returns = calculate_returns(prices)
    
    # Create combined DataFrame
    prices_df = prices.copy()
    prices_df.columns = [f"{col}_price" for col in prices_df.columns]
    
    returns_df = returns.copy()
    returns_df.columns = [f"{col}_ret" for col in returns_df.columns]
    
    # Align indices
    common_idx = prices_df.index.intersection(returns_df.index)
    
    combined = pd.concat([
        prices_df.loc[common_idx],
        returns_df.loc[common_idx]
    ], axis=1)
    
    combined.index.name = 'Date'
    combined.to_csv(output_file)
    
    print(f"\n📁 Saved to {output_file}")
    print(f"   Rows: {len(combined)}")
    print(f"   Price columns: {len(prices.columns)}")
    print(f"   Return columns: {len(returns.columns)}")
    
    # Also save sector mapping
    sector_map = create_sector_mapping(prices)
    sector_map.to_csv('sector_mapping.csv', index=False)
    print(f"   Sector mapping saved to sector_mapping.csv")
    
    return combined


def main():
    """Main execution"""
    print("\n" + "=" * 70)
    print("AI FINANCE PROJECT - REAL DATA FETCHER")
    print("Weakness Compensation Framework")
    print("=" * 70)
    
    # Fetch data (last 2 years)
    prices = fetch_price_data()
    
    # Save to CSV
    data = save_data(prices)
    
    # Print summary statistics
    returns = calculate_returns(prices)
    
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    for sector, tickers in ASSET_UNIVERSE.items():
        available = [t for t in tickers if t in returns.columns]
        if available:
            sector_ret = returns[available].mean(axis=1)
            ann_ret = sector_ret.mean() * 252
            ann_vol = sector_ret.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            print(f"{sector:12} | Tickers: {len(available):2} | "
                  f"Ann.Ret: {ann_ret:6.1%} | Vol: {ann_vol:6.1%} | Sharpe: {sharpe:5.2f}")
    
    print("\n✅ Data fetching complete!")
    print("=" * 70)
    
    return data


if __name__ == "__main__":
    main()
