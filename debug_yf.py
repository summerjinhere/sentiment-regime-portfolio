import yfinance as yf
import pandas as pd

tickers = ['AAPL', 'MSFT']
print("Downloading...")
data = yf.download(tickers, start="2023-01-01", end="2023-01-10", progress=False)
print("Shape:", data.shape)
print("Columns:", data.columns)
print("Head:")
print(data.head())

if isinstance(data.columns, pd.MultiIndex):
    print("MultiIndex levels:", data.columns.nlevels)
    print("Level 0:", data.columns.get_level_values(0))
else:
    print("Not a MultiIndex")
