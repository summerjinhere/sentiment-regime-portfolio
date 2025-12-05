import yfinance as yf
import pandas as pd

tickers = ['AAPL', 'SPY', 'MSFT', 'JNJ']
print(f"Downloading {tickers}...")
data = yf.download(tickers, period="1mo", progress=True, auto_adjust=True)

print("\nData shape:", data.shape)
print("Data columns:", data.columns)
print("\nHead:")
print(data.head())

if isinstance(data.columns, pd.MultiIndex):
    print("\nMultiIndex detected.")
    try:
        prices = data['Close']
        print("Prices shape:", prices.shape)
        print("Prices columns:", prices.columns)
        print("Prices head:")
        print(prices.head())
    except Exception as e:
        print(f"Error extracting Close: {e}")
else:
    print("\nSingle index detected.")
