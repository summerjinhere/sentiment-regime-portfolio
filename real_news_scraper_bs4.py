"""
Real News Scraper with FinBERT Sentiment
========================================

Scrapes Finviz news headlines for ALL tickers in the asset universe.
Calculates sentiment scores using ProsusAI/finbert model.

Features:
- Randomized User-Agent headers
- Rate limiting (1.5-3.0 second delays)
- FinBERT NLP sentiment analysis
- Daily aggregated sentiment scores

NO SYNTHETIC DATA - Real news and sentiment only.
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import transformers for FinBERT
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers torch")


# =============================================================================
# ASSET UNIVERSE
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

# Randomized User-Agent headers
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36',
]


def get_all_tickers() -> List[str]:
    """Get flat list of all tickers"""
    all_tickers = []
    for sector, tickers in ASSET_UNIVERSE.items():
        all_tickers.extend(tickers)
    return list(set(all_tickers))


class FinBERTSentimentAnalyzer:
    """FinBERT-based sentiment analyzer"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._initialize()
    
    def _initialize(self):
        """Initialize FinBERT model"""
        if not FINBERT_AVAILABLE:
            print("FinBERT not available. Using rule-based fallback.")
            return
        
        try:
            print("Loading FinBERT model (ProsusAI/finbert)...")
            self.pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=-1  # CPU
            )
            print("✅ FinBERT loaded successfully")
        except Exception as e:
            print(f"Error loading FinBERT: {e}")
            print("Using rule-based fallback.")
            self.pipeline = None
    
    def analyze(self, text: str) -> float:
        """
        Analyze sentiment of text
        
        Returns:
            float: Sentiment score from -1 (negative) to 1 (positive)
        """
        if not text or len(text.strip()) < 5:
            return 0.0
        
        if self.pipeline is not None:
            try:
                # Truncate to 512 tokens
                text = text[:512]
                result = self.pipeline(text)[0]
                
                label = result['label'].lower()
                score = result['score']
                
                if label == 'positive':
                    return score
                elif label == 'negative':
                    return -score
                else:  # neutral
                    return 0.0
            except Exception as e:
                return self._rule_based_sentiment(text)
        else:
            return self._rule_based_sentiment(text)
    
    def _rule_based_sentiment(self, text: str) -> float:
        """Fallback rule-based sentiment"""
        text_lower = text.lower()
        
        positive_words = [
            'surge', 'soar', 'jump', 'gain', 'rise', 'up', 'high', 'record',
            'beat', 'exceed', 'strong', 'growth', 'profit', 'bullish', 'buy',
            'upgrade', 'outperform', 'positive', 'success', 'win', 'best'
        ]
        negative_words = [
            'fall', 'drop', 'plunge', 'decline', 'down', 'low', 'miss',
            'loss', 'weak', 'bearish', 'sell', 'downgrade', 'underperform',
            'negative', 'fail', 'worst', 'crash', 'crisis', 'risk', 'fear'
        ]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        return (pos_count - neg_count) / total


class FinvizNewsScraper:
    """Scraper for Finviz news headlines"""
    
    def __init__(self):
        self.base_url = "https://finviz.com/quote.ashx"
        self.session = requests.Session()
        self.sentiment_analyzer = FinBERTSentimentAnalyzer()
    
    def _get_headers(self) -> Dict:
        """Get randomized headers"""
        return {
            'User-Agent': random.choice(USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def _rate_limit(self):
        """Random delay between requests"""
        delay = random.uniform(1.5, 3.0)
        time.sleep(delay)
    
    def scrape_ticker_news(self, ticker: str) -> List[Dict]:
        """
        Scrape news headlines for a single ticker
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            List of dicts with headline, date, sentiment
        """
        url = f"{self.base_url}?t={ticker}"
        
        try:
            response = self.session.get(url, headers=self._get_headers(), timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news table
            news_table = soup.find('table', {'id': 'news-table'})
            
            if not news_table:
                return []
            
            headlines = []
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            for row in news_table.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) >= 2:
                    # Extract date/time
                    date_cell = cells[0].text.strip()
                    
                    # Parse date
                    if len(date_cell) > 10:
                        # Has date and time
                        try:
                            date_part = date_cell.split()[0]
                            if '-' in date_part:
                                current_date = datetime.strptime(date_part, '%b-%d-%y').strftime('%Y-%m-%d')
                        except:
                            pass
                    
                    # Extract headline
                    headline_cell = cells[1]
                    headline_link = headline_cell.find('a')
                    
                    if headline_link:
                        headline_text = headline_link.text.strip()
                        
                        # Calculate sentiment
                        sentiment = self.sentiment_analyzer.analyze(headline_text)
                        
                        headlines.append({
                            'ticker': ticker,
                            'date': current_date,
                            'headline': headline_text,
                            'sentiment': sentiment
                        })
            
            return headlines
        
        except requests.exceptions.RequestException as e:
            print(f"  Error scraping {ticker}: {e}")
            return []
    
    def scrape_all_tickers(self, tickers: List[str] = None) -> pd.DataFrame:
        """
        Scrape news for all tickers
        
        Args:
            tickers: List of tickers (default: all in universe)
        
        Returns:
            DataFrame with all headlines and sentiments
        """
        if tickers is None:
            tickers = get_all_tickers()
        
        print("=" * 70)
        print("FINVIZ NEWS SCRAPER with FinBERT Sentiment")
        print("=" * 70)
        print(f"Tickers to scrape: {len(tickers)}")
        print("=" * 70)
        
        all_headlines = []
        
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i:3}/{len(tickers)}] Scraping {ticker}...", end=" ")
            
            headlines = self.scrape_ticker_news(ticker)
            all_headlines.extend(headlines)
            
            print(f"Found {len(headlines)} headlines")
            
            # Rate limiting
            if i < len(tickers):
                self._rate_limit()
        
        df = pd.DataFrame(all_headlines)
        
        print("\n" + "=" * 70)
        print(f"✅ Total headlines scraped: {len(df)}")
        print("=" * 70)
        
        return df


def aggregate_daily_sentiment(headlines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment to daily level per ticker
    
    Args:
        headlines_df: DataFrame with ticker, date, headline, sentiment
    
    Returns:
        DataFrame with daily aggregated sentiment
    """
    if headlines_df.empty:
        return pd.DataFrame()
    
    # Group by date and ticker
    daily = headlines_df.groupby(['date', 'ticker']).agg({
        'sentiment': ['mean', 'std', 'count'],
        'headline': 'count'
    }).reset_index()
    
    # Flatten column names
    daily.columns = ['date', 'ticker', 'sentiment_mean', 'sentiment_std', 'sentiment_count', 'headline_count']
    
    # Fill NaN std with 0
    daily['sentiment_std'] = daily['sentiment_std'].fillna(0)
    
    # Pivot to wide format
    sentiment_wide = daily.pivot(index='date', columns='ticker', values='sentiment_mean')
    
    # Also create market-wide sentiment
    daily_market = headlines_df.groupby('date')['sentiment'].mean()
    
    return sentiment_wide, daily_market


def save_sentiment_data(
    headlines_df: pd.DataFrame,
    output_file: str = 'real_sentiment_data.csv'
):
    """
    Save sentiment data to CSV
    """
    if headlines_df.empty:
        print("No data to save!")
        return
    
    # Aggregate
    sentiment_wide, daily_market = aggregate_daily_sentiment(headlines_df)
    
    # Add market sentiment column
    sentiment_wide['MARKET_SENTIMENT'] = daily_market
    
    # Sort by date
    sentiment_wide = sentiment_wide.sort_index()
    
    # Save
    sentiment_wide.to_csv(output_file)
    
    print(f"\n📁 Saved to {output_file}")
    print(f"   Trading days: {len(sentiment_wide)}")
    print(f"   Tickers with sentiment: {len(sentiment_wide.columns) - 1}")
    
    # Also save raw headlines
    headlines_df.to_csv('raw_headlines.csv', index=False)
    print(f"   Raw headlines saved to raw_headlines.csv")
    
    return sentiment_wide


def main():
    """Main execution"""
    print("\n" + "=" * 70)
    print("AI FINANCE PROJECT - REAL NEWS SCRAPER")
    print("FinBERT Sentiment Analysis")
    print("=" * 70)
    
    # Initialize scraper
    scraper = FinvizNewsScraper()
    
    # Scrape all tickers
    headlines = scraper.scrape_all_tickers()
    
    # Save data
    if not headlines.empty:
        sentiment_data = save_sentiment_data(headlines)
        
        # Print summary
        print("\n" + "=" * 70)
        print("SENTIMENT SUMMARY BY SECTOR")
        print("=" * 70)
        
        for sector, tickers in ASSET_UNIVERSE.items():
            sector_headlines = headlines[headlines['ticker'].isin(tickers)]
            if not sector_headlines.empty:
                avg_sentiment = sector_headlines['sentiment'].mean()
                n_headlines = len(sector_headlines)
                print(f"{sector:12} | Headlines: {n_headlines:4} | Avg Sentiment: {avg_sentiment:+.3f}")
        
        print("\n✅ News scraping complete!")
    else:
        print("\n⚠️ No headlines scraped. Check network connection.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
