import yfinance as yf
print("yfinance version:", yf.__version__)
try:
    data = yf.download("AAPL", start="2024-01-01", end="2024-01-10", progress=True)
    print("Data shape:", data.shape)
    print(data.head())
except Exception as e:
    print("Error:", e)
