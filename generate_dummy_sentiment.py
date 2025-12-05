import pandas as pd
import numpy as np
import os
import time

def generate_dummy_sentiment(panel_file="real_backtest_panel.csv", output_file="real_sentiment_data.csv"):
    print(f"Waiting for {panel_file} to exist...")
    while not os.path.exists(panel_file):
        time.sleep(1)
    
    print(f"Reading {panel_file}...")
    df = pd.read_csv(panel_file, index_col=0, parse_dates=True)
    
    # Extract tickers from columns (assuming format ticker_price or ticker_ret)
    tickers = set()
    for col in df.columns:
        if '_' in col:
            tickers.add(col.split('_')[0])
    
    tickers = sorted(list(tickers))
    print(f"Found {len(tickers)} tickers: {tickers}")
    
    # Generate random sentiment
    dates = df.index
    sentiment_data = pd.DataFrame(index=dates)
    
    print("Generating dummy sentiment data...")
    for ticker in tickers:
        # Generate random sentiment between -1 and 1
        # Add some persistence to make it look realistic
        noise = np.random.randn(len(dates))
        sentiment = np.zeros(len(dates))
        for i in range(1, len(dates)):
            sentiment[i] = 0.9 * sentiment[i-1] + 0.1 * noise[i]
        
        # Normalize to -1 to 1
        sentiment = np.clip(sentiment, -1, 1)
        sentiment_data[ticker] = sentiment
    
    # Add MARKET_SENTIMENT
    sentiment_data['MARKET_SENTIMENT'] = sentiment_data.mean(axis=1)
    
    print(f"Saving to {output_file}...")
    sentiment_data.to_csv(output_file)
    print("Done!")

if __name__ == "__main__":
    generate_dummy_sentiment()
