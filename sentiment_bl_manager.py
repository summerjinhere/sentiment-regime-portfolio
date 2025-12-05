"""
Smart Sentiment Manager with Black-Litterman Integration
=========================================================

Features:
1. CSV caching - saves sentiment to avoid repeated API calls
2. Timeout fallback - uses cached data when scraping fails
3. Incremental updates - only fetches new data
4. Black-Litterman integration - converts sentiment to portfolio views

References:
- Ferretti et al. (2022) "BERT's sentiment score for portfolio optimization"
- Lopez de Prado (2016) "Building Diversified Portfolios that Outperform Out of Sample"
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import os
import json
warnings.filterwarnings('ignore')

try:
    from scipy.optimize import minimize
    from scipy import linalg
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# SENTIMENT CACHE MANAGER
# =============================================================================

class SentimentCacheManager:
    """
    Manages sentiment data with CSV caching and fallback.
    
    Strategy:
    1. Try to load from cache first
    2. Check if cache is fresh (< max_age_hours)
    3. If stale, try to update with new data
    4. If update fails (timeout), use cached data with warning
    """
    
    def __init__(
        self,
        cache_file: str = "sentiment_cache.csv",
        cache_dir: str = ".",
        max_age_hours: int = 24,
        timeout_seconds: int = 30
    ):
        self.cache_file = Path(cache_dir) / cache_file
        self.metadata_file = Path(cache_dir) / "sentiment_metadata.json"
        self.max_age_hours = max_age_hours
        self.timeout_seconds = timeout_seconds
        self.sentiment_data: Optional[pd.DataFrame] = None
        self.last_update: Optional[datetime] = None
        self.update_status: str = "unknown"
        
    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_metadata(self, metadata: Dict):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
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
        """Load sentiment data from cache"""
        if not self.cache_file.exists():
            print(f"⚠️ No cache file found: {self.cache_file}")
            return None
        
        try:
            df = pd.read_csv(self.cache_file, index_col=0, parse_dates=True)
            self.sentiment_data = df
            
            metadata = self._load_metadata()
            if 'last_update' in metadata:
                self.last_update = datetime.fromisoformat(metadata['last_update'])
            
            print(f"✅ Loaded sentiment cache: {len(df)} days, {len(df.columns)} tickers")
            print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
            
            if self._is_cache_fresh():
                print(f"   Cache is fresh (< {self.max_age_hours}h old)")
                self.update_status = "fresh"
            else:
                print(f"   ⚠️ Cache is stale (> {self.max_age_hours}h old)")
                self.update_status = "stale"
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading cache: {e}")
            return None
    
    def save_cache(self, df: pd.DataFrame):
        """Save sentiment data to cache"""
        try:
            df.to_csv(self.cache_file)
            
            metadata = {
                'last_update': datetime.now().isoformat(),
                'n_days': len(df),
                'n_tickers': len(df.columns),
                'date_start': str(df.index[0].date()) if len(df) > 0 else None,
                'date_end': str(df.index[-1].date()) if len(df) > 0 else None,
            }
            self._save_metadata(metadata)
            
            self.sentiment_data = df
            self.last_update = datetime.now()
            self.update_status = "updated"
            
            print(f"✅ Saved sentiment cache: {self.cache_file}")
            
        except Exception as e:
            print(f"❌ Error saving cache: {e}")
    
    def update_with_timeout(
        self,
        scraper_func,
        tickers: List[str],
        force_update: bool = False
    ) -> pd.DataFrame:
        """
        Update sentiment data with timeout protection.
        
        Args:
            scraper_func: Function that returns DataFrame with sentiment
            tickers: List of tickers to fetch
            force_update: Force update even if cache is fresh
        
        Returns:
            DataFrame with sentiment data (cached or new)
        """
        import signal
        import threading
        
        # Load existing cache first
        cached_df = self.load_cache()
        
        # If cache is fresh and not forcing update, use cache
        if cached_df is not None and self._is_cache_fresh() and not force_update:
            print("📦 Using fresh cached sentiment data")
            return cached_df
        
        # Try to update with timeout
        print(f"\n🔄 Attempting sentiment update (timeout: {self.timeout_seconds}s)...")
        
        result_container = {'data': None, 'error': None}
        
        def fetch_thread():
            try:
                result_container['data'] = scraper_func(tickers)
            except Exception as e:
                result_container['error'] = str(e)
        
        thread = threading.Thread(target=fetch_thread)
        thread.start()
        thread.join(timeout=self.timeout_seconds)
        
        if thread.is_alive():
            # Timeout occurred
            print(f"⏱️ Timeout after {self.timeout_seconds}s - using cached data")
            self.update_status = "timeout_fallback"
            
            if cached_df is not None:
                return cached_df
            else:
                print("⚠️ No cache available, generating placeholder data")
                return self._generate_placeholder(tickers)
        
        if result_container['error']:
            print(f"❌ Scraping error: {result_container['error']}")
            self.update_status = "error_fallback"
            
            if cached_df is not None:
                print("📦 Falling back to cached data")
                return cached_df
            else:
                return self._generate_placeholder(tickers)
        
        if result_container['data'] is not None and len(result_container['data']) > 0:
            new_df = result_container['data']
            
            # Merge with existing cache if available
            if cached_df is not None:
                merged_df = self._merge_sentiment(cached_df, new_df)
            else:
                merged_df = new_df
            
            self.save_cache(merged_df)
            print(f"✅ Sentiment updated successfully")
            return merged_df
        
        # Scraping returned empty
        print("⚠️ Scraping returned no data")
        if cached_df is not None:
            return cached_df
        return self._generate_placeholder(tickers)
    
    def _merge_sentiment(
        self,
        old_df: pd.DataFrame,
        new_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge old and new sentiment data"""
        # Combine indices
        all_dates = old_df.index.union(new_df.index)
        all_tickers = list(set(old_df.columns) | set(new_df.columns))
        
        merged = pd.DataFrame(index=all_dates, columns=all_tickers)
        
        # Fill with old data first
        for col in old_df.columns:
            if col in merged.columns:
                merged.loc[old_df.index, col] = old_df[col]
        
        # Overwrite with new data
        for col in new_df.columns:
            if col in merged.columns:
                merged.loc[new_df.index, col] = new_df[col]
        
        # Forward fill then backward fill
        merged = merged.ffill().bfill().fillna(0)
        
        return merged.sort_index()
    
    def _generate_placeholder(self, tickers: List[str]) -> pd.DataFrame:
        """Generate neutral placeholder sentiment (last resort)"""
        print("⚠️ Generating neutral placeholder sentiment")
        
        dates = pd.date_range(
            end=datetime.now(),
            periods=252,
            freq='B'
        )
        
        # Neutral sentiment (0) instead of random
        data = pd.DataFrame(
            0.0,
            index=dates,
            columns=tickers
        )
        
        data['MARKET_SENTIMENT'] = 0.0
        self.update_status = "placeholder"
        
        return data
    
    def get_sentiment(
        self,
        date: datetime = None,
        ticker: str = None
    ) -> float:
        """Get sentiment value for specific date/ticker"""
        if self.sentiment_data is None:
            return 0.0
        
        if date is None:
            date = self.sentiment_data.index[-1]
        
        if ticker is None:
            ticker = 'MARKET_SENTIMENT'
        
        try:
            # Find nearest date
            idx = self.sentiment_data.index.get_indexer([date], method='nearest')[0]
            nearest_date = self.sentiment_data.index[idx]
            
            if ticker in self.sentiment_data.columns:
                return float(self.sentiment_data.loc[nearest_date, ticker])
            elif 'MARKET_SENTIMENT' in self.sentiment_data.columns:
                return float(self.sentiment_data.loc[nearest_date, 'MARKET_SENTIMENT'])
            else:
                return 0.0
        except:
            return 0.0


# =============================================================================
# BLACK-LITTERMAN WITH SENTIMENT VIEWS
# =============================================================================

class BlackLittermanSentiment:
    """
    Black-Litterman model with sentiment-based views.
    
    Converts FinBERT sentiment scores into investor views
    for optimal portfolio allocation.
    
    Reference:
    - Ferretti et al. (2022) "BERT's sentiment score for portfolio optimization"
    - Black & Litterman (1992) "Global Portfolio Optimization"
    """
    
    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        sentiment_scale: float = 0.10,
        view_confidence: float = 0.5
    ):
        """
        Args:
            risk_aversion: Risk aversion coefficient (λ)
            tau: Uncertainty in equilibrium returns
            sentiment_scale: How much sentiment affects views (default 10%)
            view_confidence: Confidence in sentiment views (0-1)
        """
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.sentiment_scale = sentiment_scale
        self.view_confidence = view_confidence
        
        self.equilibrium_returns = None
        self.posterior_returns = None
        self.optimal_weights = None
    
    def compute_equilibrium_returns(
        self,
        cov_matrix: pd.DataFrame,
        market_weights: pd.Series,
        risk_free_rate: float = 0.02
    ) -> pd.Series:
        """
        Compute equilibrium returns via reverse optimization.
        
        Π = λ * Σ * w_mkt
        """
        Sigma = cov_matrix.values
        w_mkt = market_weights.values.reshape(-1, 1)
        
        # Annualize covariance
        Sigma_annual = Sigma * 252
        
        # Equilibrium excess returns
        Pi = self.risk_aversion * Sigma_annual @ w_mkt
        
        self.equilibrium_returns = pd.Series(
            Pi.flatten() + risk_free_rate,
            index=cov_matrix.columns
        )
        
        return self.equilibrium_returns
    
    def sentiment_to_views(
        self,
        sentiment_scores: pd.Series,
        equilibrium_returns: pd.Series = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert sentiment scores to Black-Litterman views.
        
        View format:
        - Q: Expected returns based on views
        - P: Pick matrix (which assets the view applies to)
        - Omega: Uncertainty in views
        
        Returns:
            (P, Q, Omega) matrices
        """
        if equilibrium_returns is None:
            equilibrium_returns = self.equilibrium_returns
        
        if equilibrium_returns is None:
            raise ValueError("Must compute equilibrium returns first")
        
        # Align sentiment with available assets
        common = sentiment_scores.index.intersection(equilibrium_returns.index)
        sent = sentiment_scores.loc[common]
        eq_ret = equilibrium_returns.loc[common]
        
        n_assets = len(common)
        
        # Create views for assets with non-zero sentiment
        views = []
        for i, (ticker, sentiment) in enumerate(sent.items()):
            if abs(sentiment) > 0.05:  # Only create view if sentiment is meaningful
                # View: expected return = equilibrium + sentiment adjustment
                view_return = eq_ret[ticker] * (1 + self.sentiment_scale * sentiment)
                views.append({
                    'asset_idx': i,
                    'ticker': ticker,
                    'sentiment': sentiment,
                    'view_return': view_return
                })
        
        if not views:
            # No meaningful views, return identity (no adjustment)
            P = np.eye(n_assets)
            Q = eq_ret.values.reshape(-1, 1)
            Omega = np.eye(n_assets) * 1e6  # Very uncertain
            return P, Q, Omega
        
        # Build P matrix (each row is one view)
        n_views = len(views)
        P = np.zeros((n_views, n_assets))
        Q = np.zeros((n_views, 1))
        
        for i, view in enumerate(views):
            P[i, view['asset_idx']] = 1
            Q[i, 0] = view['view_return']
        
        # Omega: uncertainty in views (inversely proportional to confidence)
        # Higher sentiment magnitude → lower uncertainty
        omega_diag = []
        for view in views:
            # Uncertainty decreases with sentiment strength and confidence
            base_uncertainty = 0.01 / (self.view_confidence + 1e-6)
            sentiment_factor = 1 / (abs(view['sentiment']) + 0.1)
            omega_diag.append(base_uncertainty * sentiment_factor)
        
        Omega = np.diag(omega_diag)
        
        return P, Q, Omega
    
    def compute_posterior_returns(
        self,
        cov_matrix: pd.DataFrame,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: np.ndarray
    ) -> pd.Series:
        """
        Compute Black-Litterman posterior returns.
        
        E[R] = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) * [(τΣ)^(-1)Π + P'Ω^(-1)Q]
        """
        if self.equilibrium_returns is None:
            raise ValueError("Must compute equilibrium returns first")
        
        Sigma = cov_matrix.values * 252  # Annualize
        Pi = self.equilibrium_returns.values.reshape(-1, 1)
        
        tau_Sigma = self.tau * Sigma
        
        try:
            tau_Sigma_inv = linalg.inv(tau_Sigma)
            Omega_inv = linalg.inv(Omega)
            
            # Posterior precision
            M_inv = tau_Sigma_inv + P.T @ Omega_inv @ P
            M = linalg.inv(M_inv)
            
            # Posterior mean
            posterior = M @ (tau_Sigma_inv @ Pi + P.T @ Omega_inv @ Q)
            
            self.posterior_returns = pd.Series(
                posterior.flatten(),
                index=cov_matrix.columns
            )
            
        except linalg.LinAlgError:
            # If matrix inversion fails, use equilibrium
            print("⚠️ Matrix inversion failed, using equilibrium returns")
            self.posterior_returns = self.equilibrium_returns.copy()
        
        return self.posterior_returns
    
    def optimize_weights(
        self,
        cov_matrix: pd.DataFrame,
        returns: pd.Series = None,
        min_weight: float = 0.0,
        max_weight: float = 0.20
    ) -> pd.Series:
        """
        Compute optimal portfolio weights from posterior returns.
        
        w* = (λΣ)^(-1) * E[R]
        """
        if returns is None:
            returns = self.posterior_returns
        
        if returns is None:
            raise ValueError("Must compute posterior returns first")
        
        Sigma = cov_matrix.values * 252
        mu = returns.values
        n = len(mu)
        
        if not SCIPY_AVAILABLE:
            # Simple inverse-variance weights
            var = np.diag(Sigma)
            weights = (1 / var) / (1 / var).sum()
            self.optimal_weights = pd.Series(weights, index=cov_matrix.columns)
            return self.optimal_weights
        
        # Mean-variance optimization with posterior returns
        def neg_sharpe(w):
            port_ret = w @ mu
            port_vol = np.sqrt(w @ Sigma @ w)
            return -port_ret / (port_vol + 1e-6)
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        bounds = [(min_weight, max_weight) for _ in range(n)]
        
        result = minimize(
            neg_sharpe,
            x0=np.ones(n) / n,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
        
        self.optimal_weights = pd.Series(weights, index=cov_matrix.columns)
        
        return self.optimal_weights
    
    def fit(
        self,
        returns: pd.DataFrame,
        sentiment_scores: pd.Series,
        market_weights: pd.Series = None,
        risk_free_rate: float = 0.02
    ) -> pd.Series:
        """
        Full Black-Litterman pipeline with sentiment views.
        
        Args:
            returns: Historical returns DataFrame
            sentiment_scores: Sentiment scores per asset
            market_weights: Market cap weights (default: equal weight)
            risk_free_rate: Risk-free rate
        
        Returns:
            Optimal portfolio weights
        """
        cov_matrix = returns.cov()
        
        # Default to equal weights if market weights not provided
        if market_weights is None:
            market_weights = pd.Series(
                1 / len(cov_matrix),
                index=cov_matrix.columns
            )
        
        # Step 1: Equilibrium returns
        self.compute_equilibrium_returns(cov_matrix, market_weights, risk_free_rate)
        print(f"📊 Equilibrium returns: {self.equilibrium_returns.mean():.2%} avg")
        
        # Step 2: Convert sentiment to views
        P, Q, Omega = self.sentiment_to_views(sentiment_scores)
        print(f"📝 Created {len(Q)} sentiment-based views")
        
        # Step 3: Posterior returns
        self.compute_posterior_returns(cov_matrix, P, Q, Omega)
        print(f"📈 Posterior returns: {self.posterior_returns.mean():.2%} avg")
        
        # Step 4: Optimal weights
        weights = self.optimize_weights(cov_matrix)
        print(f"⚖️ Top 5 weights: {weights.nlargest(5).to_dict()}")
        
        return weights


# =============================================================================
# INTEGRATED SENTIMENT-AWARE ALLOCATOR
# =============================================================================

class SentimentAwareAllocator:
    """
    Complete pipeline: Sentiment Cache → Black-Litterman → Portfolio Weights
    
    Combines:
    1. SentimentCacheManager for robust data handling
    2. BlackLittermanSentiment for alpha integration
    3. Fallback to HRP if BL fails
    """
    
    def __init__(
        self,
        cache_file: str = "sentiment_cache.csv",
        cache_dir: str = ".",
        risk_aversion: float = 2.5,
        sentiment_scale: float = 0.10,
        max_cache_age_hours: int = 24
    ):
        self.cache_manager = SentimentCacheManager(
            cache_file=cache_file,
            cache_dir=cache_dir,
            max_age_hours=max_cache_age_hours
        )
        
        self.bl_model = BlackLittermanSentiment(
            risk_aversion=risk_aversion,
            sentiment_scale=sentiment_scale
        )
        
        self.weights = None
        self.method_used = None
    
    def load_sentiment(
        self,
        scraper_func=None,
        tickers: List[str] = None,
        force_update: bool = False
    ) -> pd.DataFrame:
        """Load sentiment data with caching"""
        if scraper_func is not None and tickers is not None:
            return self.cache_manager.update_with_timeout(
                scraper_func,
                tickers,
                force_update
            )
        else:
            return self.cache_manager.load_cache()
    
    def get_optimal_weights(
        self,
        returns: pd.DataFrame,
        sentiment_data: pd.DataFrame = None,
        date: datetime = None,
        method: str = "black_litterman"
    ) -> pd.Series:
        """
        Get optimal portfolio weights.
        
        Args:
            returns: Historical returns
            sentiment_data: Sentiment DataFrame (uses cache if None)
            date: Date for sentiment lookup (latest if None)
            method: "black_litterman", "hrp", or "equal_weight"
        
        Returns:
            Portfolio weights
        """
        tickers = list(returns.columns)
        
        # Get sentiment
        if sentiment_data is None:
            sentiment_data = self.cache_manager.sentiment_data
        
        if sentiment_data is None:
            print("⚠️ No sentiment data, falling back to equal weight")
            self.weights = pd.Series(1/len(tickers), index=tickers)
            self.method_used = "equal_weight_no_sentiment"
            return self.weights
        
        # Get sentiment for target date
        if date is None:
            date = returns.index[-1]
        
        # Build sentiment scores series
        sentiment_scores = pd.Series(index=tickers, dtype=float)
        for ticker in tickers:
            sentiment_scores[ticker] = self.cache_manager.get_sentiment(date, ticker)
        
        print(f"\n📊 Sentiment Summary:")
        print(f"   Positive (>0.1): {(sentiment_scores > 0.1).sum()}")
        print(f"   Negative (<-0.1): {(sentiment_scores < -0.1).sum()}")
        print(f"   Neutral: {((sentiment_scores >= -0.1) & (sentiment_scores <= 0.1)).sum()}")
        print(f"   Mean: {sentiment_scores.mean():.3f}")
        
        # Choose method
        if method == "black_litterman":
            try:
                self.weights = self.bl_model.fit(
                    returns,
                    sentiment_scores
                )
                self.method_used = "black_litterman"
            except Exception as e:
                print(f"⚠️ Black-Litterman failed: {e}")
                print("   Falling back to sentiment-weighted equal weight")
                
                # Fallback: sentiment-tilted equal weight
                base = pd.Series(1/len(tickers), index=tickers)
                tilt = 1 + 0.2 * sentiment_scores
                self.weights = base * tilt
                self.weights = self.weights / self.weights.sum()
                self.method_used = "sentiment_tilted_equal"
        
        elif method == "hrp":
            try:
                from topology_allocation import HRPAllocator, TopologyConfig
                
                hrp = HRPAllocator(TopologyConfig())
                hrp.fit(returns)
                hrp_weights = hrp.get_weights()
                
                # Apply sentiment tilt
                tilt = 1 + 0.15 * sentiment_scores
                self.weights = hrp_weights * tilt.loc[hrp_weights.index]
                self.weights = self.weights / self.weights.sum()
                self.method_used = "hrp_sentiment_tilted"
                
            except Exception as e:
                print(f"⚠️ HRP failed: {e}")
                self.weights = pd.Series(1/len(tickers), index=tickers)
                self.method_used = "equal_weight_fallback"
        
        else:
            self.weights = pd.Series(1/len(tickers), index=tickers)
            self.method_used = "equal_weight"
        
        return self.weights
    
    def get_allocation_report(self) -> Dict:
        """Get detailed allocation report"""
        if self.weights is None:
            return {}
        
        return {
            'method': self.method_used,
            'cache_status': self.cache_manager.update_status,
            'n_assets': len(self.weights),
            'top_5': self.weights.nlargest(5).to_dict(),
            'bottom_5': self.weights.nsmallest(5).to_dict(),
            'concentration': (self.weights ** 2).sum(),  # HHI
            'effective_n': 1 / (self.weights ** 2).sum(),
            'max_weight': self.weights.max(),
            'min_weight': self.weights.min(),
        }


# =============================================================================
# SIMPLE FINBERT SCRAPER WRAPPER
# =============================================================================

def create_simple_sentiment_scraper(timeout: int = 60):
    """
    Create a simple sentiment scraping function.
    
    Returns a function that can be passed to SentimentCacheManager.
    """
    def scrape_sentiment(tickers: List[str]) -> pd.DataFrame:
        """Simple sentiment scraper with FinBERT"""
        try:
            # Try to import the multi-source scraper
            from multi_source_news_scraper import MultiSourceNewsScraper, ScraperConfig
            
            config = ScraperConfig(trading_days=30)
            scraper = MultiSourceNewsScraper(config)
            
            # Test sources
            working_sources = scraper.test_all_sources()
            
            if not working_sources:
                raise RuntimeError("No news sources available")
            
            # Scrape headlines
            headlines_df = scraper.scrape_all_tickers(tickers)
            
            if headlines_df.empty:
                raise RuntimeError("No headlines scraped")
            
            # Aggregate by ticker and date
            sentiment_by_ticker = headlines_df.groupby(
                ['date', 'ticker']
            )['sentiment'].mean().unstack()
            
            # Add market sentiment
            sentiment_by_ticker['MARKET_SENTIMENT'] = sentiment_by_ticker.mean(axis=1)
            
            return sentiment_by_ticker
            
        except ImportError:
            print("⚠️ multi_source_news_scraper not available")
            raise
        except Exception as e:
            print(f"❌ Scraping error: {e}")
            raise
    
    return scrape_sentiment


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SENTIMENT-AWARE ALLOCATOR TEST")
    print("=" * 70)
    
    # Generate sample data
    np.random.seed(42)
    n_assets = 10
    n_days = 252
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
               'NVDA', 'JPM', 'V', 'JNJ', 'PG']
    
    # Simulated returns
    returns = pd.DataFrame(
        np.random.randn(n_days, n_assets) * 0.02,
        columns=tickers,
        index=pd.date_range(end='2025-12-05', periods=n_days, freq='B')
    )
    
    # Simulated sentiment (with some signal)
    sentiment_scores = pd.Series({
        'AAPL': 0.3,   # Strong positive
        'MSFT': 0.2,   # Positive
        'GOOGL': 0.0,  # Neutral
        'AMZN': -0.1,  # Slight negative
        'META': 0.15,  # Positive
        'NVDA': 0.4,   # Very positive
        'JPM': -0.2,   # Negative
        'V': 0.1,      # Slight positive
        'JNJ': 0.0,    # Neutral
        'PG': -0.05    # Neutral
    })
    
    # Test Black-Litterman
    print("\n[Black-Litterman with Sentiment]")
    bl = BlackLittermanSentiment(sentiment_scale=0.15)
    
    # Market weights (equal for simplicity)
    market_weights = pd.Series(1/n_assets, index=tickers)
    
    weights = bl.fit(returns, sentiment_scores, market_weights)
    
    print(f"\nOptimal Weights:")
    for ticker, w in weights.nlargest(10).items():
        sent = sentiment_scores.get(ticker, 0)
        print(f"  {ticker}: {w:.1%} (sentiment: {sent:+.2f})")
    
    print(f"\n✅ Effective N: {1/(weights**2).sum():.1f}")
    print(f"✅ Max weight: {weights.max():.1%}")
