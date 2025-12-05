"""
Smart Sentiment Scraper with Caching
=====================================

Wraps multi_source_news_scraper with:
1. CSV caching to avoid timeout
2. Fallback to cached data on failure
3. Incremental updates
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
import json
import threading
from typing import List, Optional

# Default tickers
DEFAULT_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM',
    'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'BLK', 'V', 'MA', 'AXP',
    'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'TMO', 'LLY', 'BMY', 'AMGN', 'GILD',
    'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'COST', 'LOW', 'PG', 'KO',
    'CAT', 'HON', 'UPS', 'BA', 'GE', 'MMM', 'RTX', 'LMT', 'DE', 'UNP',
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL',
    'DIS', 'NFLX', 'CMCSA', 'T', 'VZ', 'TMUS',
    'SPY', 'QQQ', 'IWM', 'DIA'
]


class SmartSentimentScraper:
    """
    Smart scraper with timeout protection and CSV caching.
    
    Usage:
        scraper = SmartSentimentScraper()
        sentiment_df = scraper.get_sentiment(force_update=False)
    """
    
    def __init__(
        self,
        cache_file: str = "real_sentiment_data.csv",
        metadata_file: str = "sentiment_metadata.json",
        cache_dir: str = ".",
        max_age_hours: int = 24,
        timeout_seconds: int = 120,
        trading_days: int = 60
    ):
        self.cache_file = Path(cache_dir) / cache_file
        self.metadata_file = Path(cache_dir) / metadata_file
        self.max_age_hours = max_age_hours
        self.timeout_seconds = timeout_seconds
        self.trading_days = trading_days
        
        self.sentiment_data: Optional[pd.DataFrame] = None
        self.last_update: Optional[datetime] = None
        self.status: str = "not_loaded"
    
    def _load_metadata(self) -> dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_metadata(self):
        """Save cache metadata"""
        metadata = {
            'last_update': datetime.now().isoformat(),
            'n_days': len(self.sentiment_data) if self.sentiment_data is not None else 0,
            'n_tickers': len(self.sentiment_data.columns) if self.sentiment_data is not None else 0,
            'status': self.status
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _is_cache_fresh(self) -> bool:
        """Check if cache is within max_age_hours"""
        metadata = self._load_metadata()
        if 'last_update' not in metadata:
            return False
        
        try:
            last_update = datetime.fromisoformat(metadata['last_update'])
            age_hours = (datetime.now() - last_update).total_seconds() / 3600
            return age_hours < self.max_age_hours
        except:
            return False
    
    def load_cache(self) -> Optional[pd.DataFrame]:
        """Load sentiment from CSV cache"""
        if not self.cache_file.exists():
            print(f"⚠️ No cache file: {self.cache_file}")
            return None
        
        try:
            df = pd.read_csv(self.cache_file, index_col=0, parse_dates=True)
            self.sentiment_data = df
            self.status = "loaded_from_cache"
            
            print(f"✅ Loaded sentiment cache:")
            print(f"   File: {self.cache_file}")
            print(f"   Days: {len(df)}")
            print(f"   Tickers: {len(df.columns)}")
            print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
            
            if self._is_cache_fresh():
                print(f"   Status: ✅ Fresh (< {self.max_age_hours}h)")
            else:
                print(f"   Status: ⚠️ Stale (> {self.max_age_hours}h)")
            
            return df
        
        except Exception as e:
            print(f"❌ Error loading cache: {e}")
            return None
    
    def save_cache(self, df: pd.DataFrame):
        """Save sentiment to CSV cache"""
        try:
            df.to_csv(self.cache_file)
            self.sentiment_data = df
            self._save_metadata()
            print(f"✅ Saved sentiment cache: {self.cache_file}")
        except Exception as e:
            print(f"❌ Error saving cache: {e}")
    
    def _scrape_with_timeout(self, tickers: List[str]) -> Optional[pd.DataFrame]:
        """Scrape sentiment with timeout protection"""
        result = {'data': None, 'error': None}
        
        def scrape_thread():
            try:
                # Try to import scraper
                from multi_source_news_scraper import (
                    MultiSourceNewsScraper, 
                    ScraperConfig,
                    FinBERTAnalyzer
                )
                
                config = ScraperConfig(trading_days=self.trading_days)
                scraper = MultiSourceNewsScraper(config)
                
                # Test sources
                working = scraper.test_all_sources()
                if not working:
                    result['error'] = "No news sources available"
                    return
                
                # Scrape
                headlines_df = scraper.scrape_all_tickers(tickers)
                
                if headlines_df.empty:
                    result['error'] = "No headlines scraped"
                    return
                
                # Aggregate by date and ticker
                sentiment_pivot = headlines_df.pivot_table(
                    index='date',
                    columns='ticker',
                    values='sentiment',
                    aggfunc='mean'
                )
                
                # Add market sentiment
                sentiment_pivot['MARKET_SENTIMENT'] = sentiment_pivot.mean(axis=1)
                
                # Ensure datetime index
                sentiment_pivot.index = pd.to_datetime(sentiment_pivot.index)
                sentiment_pivot = sentiment_pivot.sort_index()
                
                result['data'] = sentiment_pivot
                
            except ImportError as e:
                result['error'] = f"Import error: {e}"
            except Exception as e:
                result['error'] = str(e)
        
        # Run in thread with timeout
        thread = threading.Thread(target=scrape_thread)
        thread.start()
        thread.join(timeout=self.timeout_seconds)
        
        if thread.is_alive():
            print(f"⏱️ Timeout after {self.timeout_seconds}s")
            self.status = "timeout"
            return None
        
        if result['error']:
            print(f"❌ Scraping error: {result['error']}")
            self.status = "error"
            return None
        
        return result['data']
    
    def get_sentiment(
        self,
        tickers: List[str] = None,
        force_update: bool = False
    ) -> pd.DataFrame:
        """
        Get sentiment data with smart caching.
        
        Strategy:
        1. Load cache first
        2. If cache fresh and not forcing, use cache
        3. Try to update with timeout
        4. If timeout/error, fall back to cache
        5. If no cache, generate neutral placeholder
        """
        tickers = tickers or DEFAULT_TICKERS
        
        # Step 1: Load cache
        cached_df = self.load_cache()
        
        # Step 2: Check if we need to update
        if cached_df is not None and self._is_cache_fresh() and not force_update:
            print("📦 Using fresh cached sentiment")
            self.status = "using_fresh_cache"
            return cached_df
        
        # Step 3: Try to update
        print(f"\n🔄 Updating sentiment (timeout: {self.timeout_seconds}s)...")
        new_df = self._scrape_with_timeout(tickers)
        
        if new_df is not None and len(new_df) > 0:
            # Success! Merge with cache if exists
            if cached_df is not None:
                # Merge: keep old data, add/overwrite with new
                all_dates = cached_df.index.union(new_df.index)
                all_tickers = list(set(cached_df.columns) | set(new_df.columns))
                
                merged = pd.DataFrame(index=all_dates, columns=all_tickers)
                
                for col in cached_df.columns:
                    if col in merged.columns:
                        merged.loc[cached_df.index, col] = cached_df[col]
                
                for col in new_df.columns:
                    if col in merged.columns:
                        merged.loc[new_df.index, col] = new_df[col]
                
                merged = merged.ffill().bfill().fillna(0)
                merged = merged.sort_index()
                
                self.save_cache(merged)
                self.status = "updated_and_merged"
                return merged
            else:
                self.save_cache(new_df)
                self.status = "fresh_scrape"
                return new_df
        
        # Step 4: Fall back to cache
        if cached_df is not None:
            print("📦 Falling back to cached sentiment")
            self.status = "fallback_to_cache"
            return cached_df
        
        # Step 5: Generate placeholder (last resort)
        print("⚠️ Generating neutral placeholder sentiment")
        placeholder = self._generate_placeholder(tickers)
        self.save_cache(placeholder)
        self.status = "placeholder"
        return placeholder
    
    def _generate_placeholder(self, tickers: List[str]) -> pd.DataFrame:
        """Generate neutral (0) sentiment as placeholder"""
        dates = pd.date_range(
            end=datetime.now(),
            periods=252,
            freq='B'
        )
        
        df = pd.DataFrame(0.0, index=dates, columns=tickers)
        df['MARKET_SENTIMENT'] = 0.0
        
        return df


def fetch_sentiment_data(
    output_file: str = "real_sentiment_data.csv",
    tickers: List[str] = None,
    force_update: bool = False,
    timeout: int = 120
) -> pd.DataFrame:
    """
    Convenience function to fetch sentiment data.
    
    Usage:
        df = fetch_sentiment_data()
        df = fetch_sentiment_data(force_update=True)
    """
    scraper = SmartSentimentScraper(
        cache_file=output_file,
        timeout_seconds=timeout
    )
    
    return scraper.get_sentiment(tickers=tickers, force_update=force_update)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart Sentiment Scraper")
    parser.add_argument("--output", default="real_sentiment_data.csv", help="Output file")
    parser.add_argument("--force", action="store_true", help="Force update")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout seconds")
    parser.add_argument("--days", type=int, default=60, help="Trading days to fetch")
    
    args = parser.parse_args()
    
    scraper = SmartSentimentScraper(
        cache_file=args.output,
        timeout_seconds=args.timeout,
        trading_days=args.days
    )
    
    df = scraper.get_sentiment(force_update=args.force)
    
    print(f"\n📊 Final Result:")
    print(f"   Status: {scraper.status}")
    print(f"   Shape: {df.shape}")
    if 'MARKET_SENTIMENT' in df.columns:
        print(f"   Market Sentiment (last 5 days):")
        print(df['MARKET_SENTIMENT'].tail())
