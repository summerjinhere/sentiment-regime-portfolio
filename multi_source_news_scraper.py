import os
"""
Multi-Source News Scraper with FinBERT Sentiment
=================================================

Real news data collection with multiple fallback sources:
1. Finviz (Primary)
2. Yahoo Finance News API
3. Google News RSS
4. NewsAPI (requires free API key)
5. Alpha Vantage News (requires free API key)

All sources provide REAL news data - NO synthetic/sample data.

Features:
- Multiple source fallback chain
- Configurable date range (trading days)
- FinBERT sentiment analysis
- Rate limiting per source
- Automatic source switching on failure
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import random
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
import warnings
import json
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class ScraperConfig:
    """Scraper configuration"""
    def __init__(
        self,
        trading_days: int = 504,  # ~2 years
        start_date: str = None,
        end_date: str = None,
        newsapi_key: str = os.getenv("NEWSAPI_KEY", ""),
        alphavantage_key: str = None,
        rate_limit_min: float = 1.5,
        rate_limit_max: float = 3.0,
        max_retries: int = 3
    ):
        self.trading_days = trading_days
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.newsapi_key = newsapi_key
        self.alphavantage_key = alphavantage_key
        self.rate_limit_min = rate_limit_min
        self.rate_limit_max = rate_limit_max
        self.max_retries = max_retries


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

TICKER_TO_COMPANY = {
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'NVIDIA', 'AVGO': 'Broadcom',
    'ORCL': 'Oracle', 'ADBE': 'Adobe', 'CRM': 'Salesforce', 'AMD': 'AMD',
    'QCOM': 'Qualcomm', 'INTC': 'Intel', 'LLY': 'Eli Lilly', 'UNH': 'UnitedHealth',
    'JNJ': 'Johnson Johnson', 'MRK': 'Merck', 'ABBV': 'AbbVie', 'TMO': 'Thermo Fisher',
    'ABT': 'Abbott', 'PFE': 'Pfizer', 'AMGN': 'Amgen', 'DHR': 'Danaher',
    'JPM': 'JPMorgan', 'V': 'Visa', 'MA': 'Mastercard', 'BAC': 'Bank of America',
    'WFC': 'Wells Fargo', 'MS': 'Morgan Stanley', 'GS': 'Goldman Sachs',
    'BLK': 'BlackRock', 'C': 'Citigroup', 'AXP': 'American Express',
    'AMZN': 'Amazon', 'TSLA': 'Tesla', 'HD': 'Home Depot', 'PG': 'Procter Gamble',
    'COST': 'Costco', 'WMT': 'Walmart', 'KO': 'Coca-Cola', 'PEP': 'PepsiCo',
    'MCD': 'McDonalds', 'NKE': 'Nike', 'XOM': 'Exxon', 'CVX': 'Chevron',
    'COP': 'ConocoPhillips', 'SLB': 'Schlumberger', 'EOG': 'EOG Resources',
    'SPY': 'S&P 500', 'QQQ': 'Nasdaq', 'IWM': 'Russell 2000', 'DIA': 'Dow Jones',
    'GLD': 'Gold', 'SLV': 'Silver', 'USO': 'Oil', 'TLT': 'Treasury Bonds'
}

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]


def get_all_tickers() -> List[str]:
    """Get flat list of all tickers"""
    all_tickers = []
    for tickers in ASSET_UNIVERSE.values():
        all_tickers.extend(tickers)
    return list(set(all_tickers))


# =============================================================================
# FINBERT SENTIMENT ANALYZER
# =============================================================================

class FinBERTAnalyzer:
    """FinBERT-based sentiment analyzer"""
    
    def __init__(self):
        self.pipeline = None
        self.available = False
        self._initialize()
    
    def _initialize(self):
        try:
            from transformers import pipeline
            print("🔄 Loading FinBERT model (ProsusAI/finbert)...")
            self.pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=-1,
                truncation=True,
                max_length=512
            )
            self.available = True
            print("✅ FinBERT loaded successfully")
        except Exception as e:
            print(f"⚠️ FinBERT unavailable: {e}")
            print("   Using rule-based sentiment fallback")
            self.available = False
    
    def analyze(self, text: str) -> float:
        """
        Analyze sentiment of text
        Returns: float from -1 (negative) to 1 (positive)
        """
        if not text or len(text.strip()) < 10:
            return 0.0
        
        if self.available and self.pipeline:
            try:
                result = self.pipeline(text[:512])[0]
                label = result['label'].lower()
                score = result['score']
                
                if label == 'positive':
                    return score
                elif label == 'negative':
                    return -score
                else:
                    return 0.0
            except Exception as e:
                return self._rule_based_sentiment(text)
        else:
            return self._rule_based_sentiment(text)

    def analyze_components(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment returning full probability distribution
        Returns: {'net': float, 'pos': float, 'neg': float, 'neu': float}
        """
        if not text or len(text.strip()) < 10:
            return {'net': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}
        
        if self.available and self.pipeline:
            try:
                # Get all scores
                result = self.pipeline(text[:512], return_all_scores=True)[0]
                # result is list of dicts: [{'label': 'positive', 'score': 0.9}, ...]
                
                scores = {item['label'].lower(): item['score'] for item in result}
                pos = scores.get('positive', 0.0)
                neg = scores.get('negative', 0.0)
                neu = scores.get('neutral', 0.0)
                
                return {
                    'net': pos - neg,
                    'pos': pos,
                    'neg': neg,
                    'neu': neu
                }
            except Exception as e:
                # Fallback
                net = self._rule_based_sentiment(text)
                if net > 0:
                    return {'net': net, 'pos': net, 'neg': 0.0, 'neu': 1.0 - net}
                else:
                    return {'net': net, 'pos': 0.0, 'neg': abs(net), 'neu': 1.0 - abs(net)}
        else:
            net = self._rule_based_sentiment(text)
            if net > 0:
                return {'net': net, 'pos': net, 'neg': 0.0, 'neu': 1.0 - net}
            else:
                return {'net': net, 'pos': 0.0, 'neg': abs(net), 'neu': 1.0 - abs(net)}
    
    def _rule_based_sentiment(self, text: str) -> float:
        """Fallback rule-based sentiment"""
        text_lower = text.lower()
        
        positive_words = [
            'surge', 'soar', 'jump', 'gain', 'rise', 'rally', 'high', 'record',
            'beat', 'exceed', 'strong', 'growth', 'profit', 'bullish', 'buy',
            'upgrade', 'outperform', 'positive', 'success', 'win', 'best',
            'boom', 'breakthrough', 'optimistic', 'momentum', 'recovery' ,'captivating'
            ,'LOONSHOT','earning surprise', 'skyrocketed','leads market'
        ]
        negative_words = [
            'fall', 'drop', 'plunge', 'decline', 'down', 'low', 'miss',
            'loss', 'weak', 'bearish', 'sell', 'downgrade', 'underperform',
            'negative', 'fail', 'worst', 'crash', 'crisis', 'risk', 'fear',
            'concern', 'warning', 'cut', 'slash', 'layoff', 'recession','in comma'
            ,'bear', 'suicide', 'short selling','earning shock'
        ]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        return (pos_count - neg_count) / total


# =============================================================================
# ABSTRACT NEWS SOURCE
# =============================================================================

class NewsSource(ABC):
    """Abstract base class for news sources"""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.session = requests.Session()
        self.name = "BaseSource"
        self.working = True
    
    def _get_headers(self) -> Dict:
        return {
            'User-Agent': random.choice(USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
    
    def _rate_limit(self):
        delay = random.uniform(self.config.rate_limit_min, self.config.rate_limit_max)
        time.sleep(delay)
    
    @abstractmethod
    def fetch_news(self, ticker: str) -> List[Dict]:
        """Fetch news for a ticker. Returns list of {'date', 'headline', 'source'}"""
        pass
    
    def test_connection(self) -> bool:
        """Test if source is accessible"""
        return self.working


# =============================================================================
# SOURCE 1: FINVIZ
# =============================================================================

class FinvizSource(NewsSource):
    """Finviz news scraper"""
    
    def __init__(self, config: ScraperConfig):
        super().__init__(config)
        self.name = "Finviz"
        self.base_url = "https://finviz.com/quote.ashx"
    
    def fetch_news(self, ticker: str) -> List[Dict]:
        url = f"{self.base_url}?t={ticker}"
        
        try:
            response = self.session.get(url, headers=self._get_headers(), timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            news_table = soup.find('table', {'id': 'news-table'})
            
            if not news_table:
                return []
            
            headlines = []
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            for row in news_table.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) >= 2:
                    date_cell = cells[0].text.strip()
                    
                    # Parse date
                    if len(date_cell) > 8:
                        try:
                            parts = date_cell.split()
                            if len(parts) >= 1 and '-' in parts[0]:
                                current_date = datetime.strptime(parts[0], '%b-%d-%y').strftime('%Y-%m-%d')
                        except:
                            pass
                    
                    headline_link = cells[1].find('a')
                    if headline_link:
                        text = headline_link.text.strip()
                        if len(text) > 10:
                            headlines.append({
                                'date': current_date,
                                'headline': text,
                                'source': 'Finviz'
                            })
            
            return headlines
            
        except requests.exceptions.RequestException as e:
            self.working = False
            return []
    
    def test_connection(self) -> bool:
        try:
            response = self.session.get(
                f"{self.base_url}?t=AAPL",
                headers=self._get_headers(),
                timeout=10
            )
            self.working = response.status_code == 200
            return self.working
        except:
            self.working = False
            return False


# =============================================================================
# SOURCE 2: YAHOO FINANCE NEWS
# =============================================================================

class YahooFinanceSource(NewsSource):
    """Yahoo Finance news scraper"""
    
    def __init__(self, config: ScraperConfig):
        super().__init__(config)
        self.name = "Yahoo Finance"
        self.base_url = "https://finance.yahoo.com/quote"
    
    def fetch_news(self, ticker: str) -> List[Dict]:
        url = f"{self.base_url}/{ticker}/news"
        
        try:
            response = self.session.get(url, headers=self._get_headers(), timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            headlines = []
            
            # Find news items
            news_items = soup.find_all('h3', class_=re.compile(r'.*title.*', re.I))
            if not news_items:
                news_items = soup.find_all('a', href=re.compile(r'/news/'))
            
            for item in news_items[:20]:
                text = item.get_text().strip()
                if len(text) > 15 and len(text) < 500:
                    headlines.append({
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'headline': text,
                        'source': 'Yahoo Finance'
                    })
            
            # Alternative: Try JSON data
            if not headlines:
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.string and 'news' in script.string.lower():
                        try:
                            # Extract JSON
                            match = re.search(r'"title"\s*:\s*"([^"]+)"', script.string)
                            if match:
                                headlines.append({
                                    'date': datetime.now().strftime('%Y-%m-%d'),
                                    'headline': match.group(1),
                                    'source': 'Yahoo Finance'
                                })
                        except:
                            pass
            
            return headlines
            
        except Exception as e:
            self.working = False
            return []
    
    def test_connection(self) -> bool:
        try:
            response = self.session.get(
                f"{self.base_url}/AAPL",
                headers=self._get_headers(),
                timeout=10
            )
            self.working = response.status_code == 200
            return self.working
        except:
            self.working = False
            return False


# =============================================================================
# SOURCE 3: GOOGLE NEWS RSS
# =============================================================================

class GoogleNewsRSSSource(NewsSource):
    """Google News RSS feed scraper"""
    
    def __init__(self, config: ScraperConfig):
        super().__init__(config)
        self.name = "Google News RSS"
        self.base_url = "https://news.google.com/rss/search"
    
    def fetch_news(self, ticker: str) -> List[Dict]:
        # Get company name for better search
        company = TICKER_TO_COMPANY.get(ticker, ticker)
        query = f"{company} stock"
        
        url = f"{self.base_url}?q={query}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            response = self.session.get(url, headers=self._get_headers(), timeout=15)
            response.raise_for_status()
            
            # Parse RSS XML
            root = ET.fromstring(response.content)
            
            headlines = []
            
            for item in root.findall('.//item')[:20]:
                title = item.find('title')
                pub_date = item.find('pubDate')
                
                if title is not None and title.text:
                    text = title.text.strip()
                    
                    # Parse date
                    date_str = datetime.now().strftime('%Y-%m-%d')
                    if pub_date is not None and pub_date.text:
                        try:
                            dt = datetime.strptime(pub_date.text[:25], '%a, %d %b %Y %H:%M:%S')
                            date_str = dt.strftime('%Y-%m-%d')
                        except:
                            pass
                    
                    if len(text) > 10:
                        headlines.append({
                            'date': date_str,
                            'headline': text,
                            'source': 'Google News'
                        })
            
            return headlines
            
        except Exception as e:
            self.working = False
            return []
    
    def test_connection(self) -> bool:
        try:
            url = f"{self.base_url}?q=stock+market&hl=en-US"
            response = self.session.get(url, headers=self._get_headers(), timeout=10)
            self.working = response.status_code == 200
            return self.working
        except:
            self.working = False
            return False


# =============================================================================
# SOURCE 4: MARKETWATCH
# =============================================================================

class MarketWatchSource(NewsSource):
    """MarketWatch news scraper"""
    
    def __init__(self, config: ScraperConfig):
        super().__init__(config)
        self.name = "MarketWatch"
        self.base_url = "https://www.marketwatch.com/investing/stock"
    
    def fetch_news(self, ticker: str) -> List[Dict]:
        url = f"{self.base_url}/{ticker.lower()}"
        
        try:
            response = self.session.get(url, headers=self._get_headers(), timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            headlines = []
            
            # Find news headlines
            news_items = soup.find_all('a', class_=re.compile(r'.*link.*', re.I))
            
            for item in news_items:
                text = item.get_text().strip()
                if len(text) > 20 and len(text) < 300:
                    # Filter out non-news items
                    if not any(x in text.lower() for x in ['sign in', 'subscribe', 'menu']):
                        headlines.append({
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'headline': text,
                            'source': 'MarketWatch'
                        })
            
            return headlines[:20]
            
        except Exception as e:
            self.working = False
            return []
    
    def test_connection(self) -> bool:
        try:
            response = self.session.get(
                f"{self.base_url}/aapl",
                headers=self._get_headers(),
                timeout=10
            )
            self.working = response.status_code == 200
            return self.working
        except:
            self.working = False
            return False


# =============================================================================
# SOURCE 5: NEWSAPI (Requires free API key)
# =============================================================================

class NewsAPISource(NewsSource):
    """NewsAPI.org news fetcher (requires API key)"""
    
    def __init__(self, config: ScraperConfig):
        super().__init__(config)
        self.name = "NewsAPI"
        self.api_key = config.newsapi_key
        self.base_url = "https://newsapi.org/v2/everything"
        self.working = bool(self.api_key)
    
    def fetch_news(self, ticker: str) -> List[Dict]:
        if not self.api_key:
            return []
        
        company = TICKER_TO_COMPANY.get(ticker, ticker)
        
        params = {
            'q': f'"{company}" stock OR "{ticker}" stock',
            'apiKey': self.api_key,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 20
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=15)
            data = response.json()
            
            if data.get('status') != 'ok':
                self.working = False
                return []
            
            headlines = []
            for article in data.get('articles', []):
                title = article.get('title', '')
                pub_date = article.get('publishedAt', '')
                
                if title and len(title) > 10:
                    date_str = datetime.now().strftime('%Y-%m-%d')
                    if pub_date:
                        try:
                            dt = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                            date_str = dt.strftime('%Y-%m-%d')
                        except:
                            pass
                    
                    headlines.append({
                        'date': date_str,
                        'headline': title,
                        'source': 'NewsAPI'
                    })
            
            return headlines
            
        except Exception as e:
            self.working = False
            return []
    
    def test_connection(self) -> bool:
        if not self.api_key:
            return False
        try:
            params = {'q': 'stock market', 'apiKey': self.api_key, 'pageSize': 1}
            response = self.session.get(self.base_url, params=params, timeout=10)
            data = response.json()
            self.working = data.get('status') == 'ok'
            return self.working
        except:
            self.working = False
            return False


# =============================================================================
# SOURCE 6: SEEKING ALPHA
# =============================================================================

class SeekingAlphaSource(NewsSource):
    """Seeking Alpha news scraper"""
    
    def __init__(self, config: ScraperConfig):
        super().__init__(config)
        self.name = "Seeking Alpha"
        self.base_url = "https://seekingalpha.com/symbol"
    
    def fetch_news(self, ticker: str) -> List[Dict]:
        url = f"{self.base_url}/{ticker}/news"
        
        try:
            headers = self._get_headers()
            headers['Accept'] = 'text/html'
            
            response = self.session.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            headlines = []
            
            # Find article titles
            articles = soup.find_all('a', {'data-test-id': re.compile(r'.*title.*', re.I)})
            if not articles:
                articles = soup.find_all('a', href=re.compile(r'/news/'))
            
            for article in articles[:20]:
                text = article.get_text().strip()
                if len(text) > 15 and len(text) < 400:
                    headlines.append({
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'headline': text,
                        'source': 'Seeking Alpha'
                    })
            
            return headlines
            
        except Exception as e:
            self.working = False
            return []
    
    def test_connection(self) -> bool:
        try:
            response = self.session.get(
                f"{self.base_url}/AAPL",
                headers=self._get_headers(),
                timeout=10
            )
            self.working = response.status_code == 200
            return self.working
        except:
            self.working = False
            return False


# =============================================================================
# MULTI-SOURCE NEWS SCRAPER
# =============================================================================

class MultiSourceNewsScraper:
    """
    Multi-source news scraper with automatic fallback
    
    Source priority:
    1. Finviz (most reliable for stock news)
    2. Yahoo Finance
    3. Google News RSS
    4. MarketWatch
    5. Seeking Alpha
    6. NewsAPI (if API key provided)
    """
    
    def __init__(self, config: ScraperConfig = None):
        self.config = config or ScraperConfig()
        self.sentiment_analyzer = FinBERTAnalyzer()
        
        # Initialize sources in priority order
        self.sources: List[NewsSource] = [
            FinvizSource(self.config),
            YahooFinanceSource(self.config),
            GoogleNewsRSSSource(self.config),
            MarketWatchSource(self.config),
            SeekingAlphaSource(self.config),
        ]
        
        # Add NewsAPI if key provided
        if self.config.newsapi_key:
            self.sources.append(NewsAPISource(self.config))
        
        self.active_sources: List[NewsSource] = []
        self.headlines_by_ticker: Dict[str, List[Dict]] = {}
    
    def test_all_sources(self) -> List[str]:
        """Test all sources and return list of working ones"""
        print("\n" + "=" * 70)
        print("TESTING NEWS SOURCES")
        print("=" * 70)
        
        self.active_sources = []
        
        for source in self.sources:
            print(f"Testing {source.name}...", end=" ")
            if source.test_connection():
                print("✅ Working")
                self.active_sources.append(source)
            else:
                print("❌ Failed")
        
        if not self.active_sources:
            print("\n⚠️ WARNING: No news sources available!")
            print("   Check network connection or provide API keys.")
        else:
            print(f"\n✅ {len(self.active_sources)} source(s) available")
        
        return [s.name for s in self.active_sources]
    
    def scrape_ticker(self, ticker: str) -> List[Dict]:
        """Scrape news for single ticker using fallback chain with deduplication"""
        all_headlines = []
        seen_headlines = set()
        
        for source in self.active_sources:
            if not source.working:
                continue
            
            try:
                headlines = source.fetch_news(ticker)
                
                if headlines:
                    # Process and deduplicate
                    for h in headlines:
                        # Clean headline for deduplication
                        clean_text = re.sub(r'[^\w\s]', '', h['headline'].lower()).strip()
                        
                        # Skip if seen (exact match after cleaning)
                        if clean_text in seen_headlines:
                            continue
                        
                        seen_headlines.add(clean_text)
                        
                        h['ticker'] = ticker
                        
                        # Get detailed sentiment
                        sent_comps = self.sentiment_analyzer.analyze_components(h['headline'])
                        h['sentiment'] = sent_comps['net'] # Keep backward compatibility
                        h['sentiment_pos'] = sent_comps['pos']
                        h['sentiment_neg'] = sent_comps['neg']
                        h['sentiment_neu'] = sent_comps['neu']
                        
                        all_headlines.extend([h])
                    
                    # If we got enough headlines, stop
                    if len(all_headlines) >= 10:
                        break
                
            except Exception as e:
                source.working = False
                continue
            
            # Rate limit between sources
            time.sleep(0.5)
        
        return all_headlines
    
    def scrape_all_tickers(self, tickers: List[str] = None) -> pd.DataFrame:
        """Scrape news for all tickers"""
        if tickers is None:
            tickers = get_all_tickers()
        
        # Test sources first
        if not self.active_sources:
            self.test_all_sources()
        
        if not self.active_sources:
            raise RuntimeError("No news sources available! Check network connection.")
        
        print("\n" + "=" * 70)
        print(f"SCRAPING NEWS FOR {len(tickers)} TICKERS")
        print(f"Active sources: {', '.join(s.name for s in self.active_sources)}")
        print("=" * 70)
        
        all_headlines = []
        failed_tickers = []
        
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i:3}/{len(tickers)}] {ticker:6}...", end=" ")
            
            headlines = self.scrape_ticker(ticker)
            
            if headlines:
                all_headlines.extend(headlines)
                avg_sent = np.mean([h['sentiment'] for h in headlines])
                print(f"{len(headlines):2} headlines | Avg sentiment: {avg_sent:+.3f}")
            else:
                failed_tickers.append(ticker)
                print("No headlines found")
            
            # Rate limit
            if i < len(tickers):
                time.sleep(random.uniform(
                    self.config.rate_limit_min,
                    self.config.rate_limit_max
                ))
        
        df = pd.DataFrame(all_headlines)
        
        print("\n" + "=" * 70)
        print("SCRAPING SUMMARY")
        print("=" * 70)
        print(f"Total headlines: {len(df)}")
        print(f"Successful tickers: {len(tickers) - len(failed_tickers)}/{len(tickers)}")
        
        if failed_tickers and len(failed_tickers) <= 10:
            print(f"Failed tickers: {', '.join(failed_tickers)}")
        
        if not df.empty:
            print(f"\nSentiment distribution:")
            print(f"  Positive (>0.1):  {(df['sentiment'] > 0.1).sum():4} ({(df['sentiment'] > 0.1).mean():.1%})")
            print(f"  Neutral:          {((df['sentiment'] >= -0.1) & (df['sentiment'] <= 0.1)).sum():4}")
            print(f"  Negative (<-0.1): {(df['sentiment'] < -0.1).sum():4} ({(df['sentiment'] < -0.1).mean():.1%})")
        
        return df


# =============================================================================
# SENTIMENT DATA AGGREGATOR
# =============================================================================

class SentimentAggregator:
    """Aggregate headlines to daily sentiment"""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
    
    def aggregate_daily(self, headlines_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sentiment to daily level per ticker.
        Computes advanced metrics: Abnormal Pessimism and Relative Sentiment.
        """
        if headlines_df.empty:
            return pd.DataFrame()
        
        # Convert date
        headlines_df['date'] = pd.to_datetime(headlines_df['date'])
        
        # Ensure component columns exist (for backward compatibility with old CSVs)
        if 'sentiment_neg' not in headlines_df.columns:
            headlines_df['sentiment_neg'] = 0.0
            headlines_df['sentiment_pos'] = 0.0
        
        # Aggregate by date and ticker
        daily = headlines_df.groupby(['date', 'ticker']).agg({
            'sentiment': 'mean',
            'sentiment_pos': 'mean',
            'sentiment_neg': 'mean',
            'headline': 'count'
        }).reset_index()
        
        daily.rename(columns={
            'sentiment': 'net_mean',
            'sentiment_pos': 'pos_mean',
            'sentiment_neg': 'neg_mean',
            'headline': 'count'
        }, inplace=True)
        
        # ---------------------------------------------------------
        # 1. Abnormal Pessimism (Tetlock 2007)
        # ---------------------------------------------------------
        # Abnormal Neg = Neg_t - RollingAvg(Neg, 20)
        # We need to do this per ticker
        daily = daily.sort_values(['ticker', 'date'])
        
        daily['neg_ma20'] = daily.groupby('ticker')['neg_mean'].transform(
            lambda x: x.rolling(window=20, min_periods=5).mean()
        )
        daily['abnormal_pessimism'] = daily['neg_mean'] - daily['neg_ma20']
        
        # ---------------------------------------------------------
        # 2. Relative Sentiment (Cross-Sectional)
        # ---------------------------------------------------------
        # Rel Sent = Net_t - MarketNet_t
        market_daily = daily.groupby('date')['net_mean'].mean().reset_index()
        market_daily.rename(columns={'net_mean': 'market_net'}, inplace=True)
        
        daily = pd.merge(daily, market_daily, on='date', how='left')
        daily['relative_sentiment'] = daily['net_mean'] - daily['market_net']
        
        # ---------------------------------------------------------
        # Select the "Best" Signal for the main output
        # ---------------------------------------------------------
        # Literature suggests Abnormal Pessimism (inverted) or Relative Sentiment is better than raw Net.
        # We will use Relative Sentiment as the primary "sentiment" for the dashboard compatibility,
        # but we could also use -1 * Abnormal Pessimism.
        # Let's use Relative Sentiment as it captures firm-specific news well.
        
        # Pivot to wide format (dates x tickers) using Relative Sentiment
        sentiment_wide = daily.pivot(
            index='date',
            columns='ticker',
            values='relative_sentiment'
        )
        
        # Add market-wide sentiment (Raw Net Mean)
        sentiment_wide['MARKET_SENTIMENT'] = market_daily.set_index('date')['market_net']
        
        # Sort and fill
        sentiment_wide = sentiment_wide.sort_index()
        sentiment_wide = sentiment_wide.ffill(limit=5)  # Forward fill up to 5 days
        
        return sentiment_wide
    
    def filter_by_trading_days(
        self,
        sentiment_df: pd.DataFrame,
        trading_days: int = None,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """Filter sentiment to specified trading day range"""
        if sentiment_df.empty:
            return sentiment_df
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            sentiment_df = sentiment_df[sentiment_df.index <= end_dt]
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
            sentiment_df = sentiment_df[sentiment_df.index >= start_dt]
        
        if trading_days and len(sentiment_df) > trading_days:
            sentiment_df = sentiment_df.iloc[-trading_days:]
        
        return sentiment_df


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main execution with configurable parameters"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Source News Scraper')
    parser.add_argument('--trading-days', type=int, default=504,
                       help='Number of trading days (default: 504 = ~2 years)')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--newsapi-key', type=str, default=os.getenv("NEWSAPI_KEY", ""),
                       help='NewsAPI.org API key (optional)')
    parser.add_argument('--output', type=str, default='real_sentiment_data.csv',
                       help='Output file path')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test sources, do not scrape')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("AI FINANCE PROJECT - MULTI-SOURCE NEWS SCRAPER")
    print("Real Data Only - No Synthetic Data")
    print("=" * 70)
    
    # Configuration
    config = ScraperConfig(
        trading_days=args.trading_days,
        start_date=args.start_date,
        end_date=args.end_date,
        newsapi_key=args.newsapi_key
    )
    
    print(f"\nConfiguration:")
    print(f"  Trading Days: {config.trading_days}")
    print(f"  End Date: {config.end_date}")
    if config.newsapi_key:
        print(f"  NewsAPI: Enabled")
    
    # Initialize scraper
    scraper = MultiSourceNewsScraper(config)
    
    # Test sources
    working_sources = scraper.test_all_sources()
    
    if args.test_only:
        print("\n✅ Test complete")
        return
    
    if not working_sources:
        print("\n❌ No working sources! Cannot proceed.")
        print("\nTroubleshooting:")
        print("  1. Check internet connection")
        print("  2. Try again later (rate limiting)")
        print("  3. Provide NewsAPI key: --newsapi-key YOUR_KEY")
        print("     Get free key at: https://newsapi.org/register")
        return
    
    # Scrape
    try:
        headlines_df = scraper.scrape_all_tickers()
        
        if headlines_df.empty:
            print("\n⚠️ No headlines collected!")
            return
        
        # Aggregate to daily
        aggregator = SentimentAggregator(config)
        sentiment_daily = aggregator.aggregate_daily(headlines_df)
        
        # Filter by trading days
        sentiment_filtered = aggregator.filter_by_trading_days(
            sentiment_daily,
            trading_days=config.trading_days,
            start_date=config.start_date,
            end_date=config.end_date
        )
        
        # Save
        sentiment_filtered.index.name = 'Date'
        sentiment_filtered.to_csv(args.output)
        
        print(f"\n📁 Saved to {args.output}")
        print(f"   Date range: {sentiment_filtered.index[0].date()} to {sentiment_filtered.index[-1].date()}")
        print(f"   Trading days: {len(sentiment_filtered)}")
        print(f"   Tickers: {len(sentiment_filtered.columns) - 1}")  # -1 for MARKET_SENTIMENT
        
        # Also save raw headlines
        headlines_df.to_csv('raw_headlines.csv', index=False)
        print(f"   Raw headlines saved to raw_headlines.csv")
        
        print("\n✅ Sentiment data collection complete!")
        
    except Exception as e:
        print(f"\n❌ Error during scraping: {e}")
        raise


if __name__ == "__main__":
    main()
