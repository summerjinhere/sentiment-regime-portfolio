import pandas as pd
import fetch_real_data

# Get expected tickers
expected = set(fetch_real_data.get_all_tickers())

# Get actual tickers from CSV
try:
    df = pd.read_csv('real_backtest_panel.csv', index_col=0)
    # Columns are like 'AAPL_price', 'AAPL_ret'. We need to extract the ticker.
    # The code in fetch_real_data.py does: prices_df.columns = [f"{col}_price" for col in prices_df.columns]
    # So we can look for columns ending in '_price'
    actual = set([c.replace('_price', '') for c in df.columns if c.endswith('_price')])
    
    missing = expected - actual
    
    print(f"Expected: {len(expected)}")
    print(f"Actual: {len(actual)}")
    print(f"Missing: {len(missing)}")
    print("-" * 30)
    print("MISSING TICKERS:")
    for t in sorted(list(missing)):
        print(f"- {t}")
        
except Exception as e:
    print(f"Error: {e}")
