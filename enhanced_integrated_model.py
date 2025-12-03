"""
Enhanced Integrated Model - Weakness Compensation Framework
============================================================

Master's Thesis Implementation

Layers:
1. Dynamic Factor Loading (Kalman Filter - Time-varying Beta)
2. Sentiment Integration (FinBERT-based signals)
3. GARCH-MIDAS Volatility Component
4. Optimal Portfolio Weight Generation

Uses ONLY REAL DATA loaded from CSV files.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy import stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for the integrated model"""
    # Kalman Filter parameters
    kalman_observation_noise: float = 0.01
    kalman_process_noise: float = 0.001
    
    # Sentiment parameters
    sentiment_decay: float = 0.9
    sentiment_weight: float = 0.2
    
    # GARCH-MIDAS parameters
    garch_omega: float = 0.00001
    garch_alpha: float = 0.1
    garch_beta: float = 0.85
    midas_k: int = 22  # Monthly lag
    
    # Portfolio parameters
    risk_aversion: float = 2.0
    max_position: float = 0.15
    min_position: float = 0.0


# =============================================================================
# LAYER 1: DYNAMIC FACTOR LOADING (KALMAN FILTER)
# =============================================================================

class KalmanFilterBeta:
    """
    Time-varying beta estimation using Kalman Filter
    
    State equation: beta_t = beta_{t-1} + w_t, w_t ~ N(0, Q)
    Observation equation: r_t = alpha + beta_t * r_m,t + v_t, v_t ~ N(0, R)
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.beta_history = []
        self.beta_var_history = []
        
        # Kalman state
        self.beta = 1.0  # Initial beta estimate
        self.P = 1.0     # Initial variance estimate
        
        # Noise parameters
        self.Q = config.kalman_process_noise   # Process noise
        self.R = config.kalman_observation_noise  # Observation noise
    
    def update(self, stock_return: float, market_return: float) -> float:
        """
        Single Kalman filter update step
        
        Args:
            stock_return: Individual stock return
            market_return: Market (benchmark) return
        
        Returns:
            Updated beta estimate
        """
        # Prediction step
        beta_pred = self.beta
        P_pred = self.P + self.Q
        
        # Update step
        if abs(market_return) > 1e-8:
            # Innovation
            y = stock_return - beta_pred * market_return
            S = P_pred * market_return**2 + self.R
            
            # Kalman gain
            K = P_pred * market_return / S
            
            # State update
            self.beta = beta_pred + K * y
            self.P = (1 - K * market_return) * P_pred
        
        # Store history
        self.beta_history.append(self.beta)
        self.beta_var_history.append(self.P)
        
        return self.beta
    
    def fit(self, stock_returns: np.ndarray, market_returns: np.ndarray) -> np.ndarray:
        """
        Fit Kalman filter to entire series
        
        Returns:
            Array of time-varying betas
        """
        self.reset()
        
        betas = []
        for r_stock, r_market in zip(stock_returns, market_returns):
            beta = self.update(r_stock, r_market)
            betas.append(beta)
        
        return np.array(betas)
    
    def reset(self):
        """Reset filter state"""
        self.beta = 1.0
        self.P = 1.0
        self.beta_history = []
        self.beta_var_history = []


class DynamicFactorLoading:
    """
    Layer 1: Dynamic Factor Loading using Kalman Filter
    
    Estimates time-varying betas for all assets in the universe.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.filters: Dict[str, KalmanFilterBeta] = {}
        self.betas: Dict[str, np.ndarray] = {}
    
    def fit(self, returns: pd.DataFrame, market_returns: pd.Series) -> Dict[str, np.ndarray]:
        """
        Fit dynamic betas for all assets
        
        Args:
            returns: DataFrame of asset returns (columns = tickers)
            market_returns: Series of market/benchmark returns
        
        Returns:
            Dict mapping ticker to time-varying beta array
        """
        # Align data
        common_idx = returns.index.intersection(market_returns.index)
        returns = returns.loc[common_idx]
        market = market_returns.loc[common_idx].values
        
        self.betas = {}
        
        for ticker in returns.columns:
            # Create filter for this ticker
            filter_ = KalmanFilterBeta(self.config)
            
            # Fit
            stock_ret = returns[ticker].values
            betas = filter_.fit(stock_ret, market)
            
            self.betas[ticker] = betas
            self.filters[ticker] = filter_
        
        return self.betas
    
    def get_current_betas(self) -> Dict[str, float]:
        """Get most recent beta estimates"""
        return {ticker: betas[-1] for ticker, betas in self.betas.items()}


# =============================================================================
# LAYER 2: SENTIMENT INTEGRATION
# =============================================================================

class SentimentIntegration:
    """
    Layer 2: Sentiment signal integration
    
    Incorporates FinBERT sentiment scores into return predictions.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.sentiment_data: Optional[pd.DataFrame] = None
        self.smoothed_sentiment: Optional[pd.DataFrame] = None
    
    def load_sentiment(self, filepath: str = 'real_sentiment_data.csv') -> bool:
        """
        Load sentiment data from CSV
        
        Args:
            filepath: Path to sentiment CSV
        
        Returns:
            True if loaded successfully
        """
        try:
            self.sentiment_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            self._smooth_sentiment()
            print(f"✅ Loaded sentiment data: {len(self.sentiment_data)} days")
            return True
        except Exception as e:
            print(f"⚠️ Could not load sentiment: {e}")
            return False
    
    def _smooth_sentiment(self):
        """Apply exponential smoothing to sentiment"""
        if self.sentiment_data is None:
            return
        
        decay = self.config.sentiment_decay
        self.smoothed_sentiment = self.sentiment_data.ewm(alpha=1-decay).mean()
    
    def get_sentiment_signal(self, date, ticker: str = None) -> float:
        """
        Get sentiment signal for a given date and ticker
        
        Args:
            date: Date to get sentiment for
            ticker: Specific ticker (None for market sentiment)
        
        Returns:
            Sentiment signal (-1 to 1)
        """
        if self.smoothed_sentiment is None:
            return 0.0
        
        try:
            if ticker and ticker in self.smoothed_sentiment.columns:
                return self.smoothed_sentiment.loc[date, ticker]
            elif 'MARKET_SENTIMENT' in self.smoothed_sentiment.columns:
                return self.smoothed_sentiment.loc[date, 'MARKET_SENTIMENT']
            else:
                return 0.0
        except:
            return 0.0
    
    def adjust_expected_returns(
        self,
        base_returns: pd.Series,
        date
    ) -> pd.Series:
        """
        Adjust expected returns based on sentiment
        
        Args:
            base_returns: Base expected returns
            date: Current date
        
        Returns:
            Sentiment-adjusted expected returns
        """
        adjusted = base_returns.copy()
        weight = self.config.sentiment_weight
        
        for ticker in base_returns.index:
            sentiment = self.get_sentiment_signal(date, ticker)
            # Sentiment adjustment: positive sentiment increases expected return
            adjusted[ticker] = base_returns[ticker] * (1 + weight * sentiment)
        
        return adjusted


# =============================================================================
# LAYER 3: GARCH-MIDAS VOLATILITY
# =============================================================================

class GARCH_MIDAS:
    """
    Layer 3: GARCH-MIDAS volatility model
    
    Combines short-term GARCH dynamics with long-term (MIDAS) component.
    
    σ²_t = τ_t * g_t
    
    where:
    - g_t: Short-term GARCH component
    - τ_t: Long-term MIDAS component (monthly)
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
        # GARCH parameters
        self.omega = config.garch_omega
        self.alpha = config.garch_alpha
        self.beta = config.garch_beta
        
        # MIDAS parameters
        self.K = config.midas_k
        
        # State
        self.g_t = 0.0001  # Short-term variance
        self.tau_t = 0.0001  # Long-term variance
        self.volatility_history = []
    
    def _compute_midas_weights(self, k: int) -> np.ndarray:
        """Compute MIDAS beta weights"""
        weights = np.array([(1 - i/k)**2 for i in range(k)])
        return weights / weights.sum()
    
    def fit(self, returns: np.ndarray) -> np.ndarray:
        """
        Fit GARCH-MIDAS model
        
        Args:
            returns: Array of returns
        
        Returns:
            Array of volatility estimates
        """
        T = len(returns)
        volatilities = np.zeros(T)
        
        # Initialize
        self.g_t = np.var(returns[:min(20, T)])
        self.tau_t = np.var(returns[:min(60, T)])
        
        for t in range(T):
            # Update MIDAS (long-term) component monthly
            if t >= self.K:
                weights = self._compute_midas_weights(self.K)
                rv = returns[t-self.K:t]**2
                self.tau_t = np.dot(weights, rv)
            
            # Update GARCH (short-term) component
            if t > 0:
                self.g_t = (self.omega + 
                          self.alpha * returns[t-1]**2 + 
                          self.beta * self.g_t)
            
            # Total variance
            sigma2 = self.tau_t * (1 + self.g_t / self.tau_t) if self.tau_t > 0 else self.g_t
            volatilities[t] = np.sqrt(max(sigma2, 1e-8))
        
        self.volatility_history = volatilities
        return volatilities
    
    def forecast(self, h: int = 1) -> float:
        """
        Forecast volatility h steps ahead
        
        Args:
            h: Forecast horizon
        
        Returns:
            Volatility forecast
        """
        # Simple forecast using persistence
        g_forecast = self.g_t
        for _ in range(h):
            g_forecast = self.omega + (self.alpha + self.beta) * g_forecast
        
        sigma2 = self.tau_t * (1 + g_forecast / self.tau_t) if self.tau_t > 0 else g_forecast
        return np.sqrt(max(sigma2, 1e-8))


# =============================================================================
# MAIN INTEGRATED MODEL
# =============================================================================

class EnhancedIntegratedModel:
    """
    Weakness Compensation Framework - Complete Integration
    
    Combines:
    - Layer 1: Dynamic Factor Loading (Kalman Filter)
    - Layer 2: Sentiment Integration (FinBERT)
    - Layer 3: GARCH-MIDAS Volatility
    
    Outputs optimal portfolio weights.
    """
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        
        # Initialize layers
        self.layer1_factor = DynamicFactorLoading(self.config)
        self.layer2_sentiment = SentimentIntegration(self.config)
        self.layer3_volatility: Dict[str, GARCH_MIDAS] = {}
        
        # Data
        self.returns: Optional[pd.DataFrame] = None
        self.prices: Optional[pd.DataFrame] = None
        self.market_returns: Optional[pd.Series] = None
        
        # Results
        self.dynamic_betas: Dict[str, np.ndarray] = {}
        self.volatilities: Dict[str, np.ndarray] = {}
        self.fitted = False
    
    def load_data(
        self,
        prices_file: str = 'real_backtest_panel.csv',
        sentiment_file: str = 'real_sentiment_data.csv',
        benchmark_ticker: str = 'SPY'
    ) -> bool:
        """
        Load real data from CSV files
        
        Args:
            prices_file: Path to price/returns CSV
            sentiment_file: Path to sentiment CSV
            benchmark_ticker: Ticker to use as market benchmark
        
        Returns:
            True if loaded successfully
        """
        try:
            # Load price data
            data = pd.read_csv(prices_file, index_col=0, parse_dates=True)
            
            # Extract returns columns
            ret_cols = [c for c in data.columns if c.endswith('_ret')]
            self.returns = data[ret_cols].copy()
            self.returns.columns = [c.replace('_ret', '') for c in ret_cols]
            
            # Extract price columns
            price_cols = [c for c in data.columns if c.endswith('_price')]
            self.prices = data[price_cols].copy()
            self.prices.columns = [c.replace('_price', '') for c in price_cols]
            
            # Market returns
            bench_col = f'{benchmark_ticker}_ret'
            if bench_col.replace('_ret', '') in self.returns.columns:
                self.market_returns = self.returns[benchmark_ticker]
            else:
                # Use SPY or mean
                if 'SPY' in self.returns.columns:
                    self.market_returns = self.returns['SPY']
                else:
                    self.market_returns = self.returns.mean(axis=1)
            
            print(f"✅ Loaded price data: {len(self.returns)} days, {len(self.returns.columns)} assets")
            
            # Load sentiment
            self.layer2_sentiment.load_sentiment(sentiment_file)
            
            return True
        
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def fit(self) -> bool:
        """
        Fit all model components
        
        Returns:
            True if fitted successfully
        """
        if self.returns is None:
            print("❌ No data loaded. Call load_data() first.")
            return False
        
        print("\n" + "=" * 70)
        print("FITTING ENHANCED INTEGRATED MODEL")
        print("=" * 70)
        
        # Layer 1: Dynamic Betas
        print("\n[Layer 1] Fitting Dynamic Factor Loading (Kalman Filter)...")
        self.dynamic_betas = self.layer1_factor.fit(self.returns, self.market_returns)
        print(f"   Fitted betas for {len(self.dynamic_betas)} assets")
        
        # Layer 3: Volatility for each asset
        print("\n[Layer 3] Fitting GARCH-MIDAS Volatility...")
        for ticker in self.returns.columns:
            garch = GARCH_MIDAS(self.config)
            self.volatilities[ticker] = garch.fit(self.returns[ticker].values)
            self.layer3_volatility[ticker] = garch
        print(f"   Fitted volatility models for {len(self.volatilities)} assets")
        
        self.fitted = True
        print("\n✅ Model fitting complete!")
        print("=" * 70)
        
        return True
    
    def get_expected_returns(self, date=None) -> pd.Series:
        """
        Get expected returns adjusted by sentiment
        
        Args:
            date: Date for sentiment lookup (default: most recent)
        
        Returns:
            Series of expected returns
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Base expected returns (historical mean)
        base_returns = self.returns.mean()
        
        # Adjust by sentiment
        if date is None:
            date = self.returns.index[-1]
        
        adjusted = self.layer2_sentiment.adjust_expected_returns(base_returns, date)
        
        return adjusted
    
    def get_covariance_matrix(self, use_dynamic_beta: bool = True) -> pd.DataFrame:
        """
        Get covariance matrix using dynamic betas
        
        Args:
            use_dynamic_beta: Whether to use time-varying betas
        
        Returns:
            Covariance matrix DataFrame
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get current betas
        if use_dynamic_beta:
            betas = self.layer1_factor.get_current_betas()
        else:
            betas = {t: 1.0 for t in self.returns.columns}
        
        # Market variance
        market_var = self.market_returns.var()
        
        # Build covariance matrix using factor model
        # Cov(i,j) = beta_i * beta_j * var(market) + idiosyncratic
        tickers = list(self.returns.columns)
        n = len(tickers)
        
        cov_matrix = np.zeros((n, n))
        
        for i, ti in enumerate(tickers):
            for j, tj in enumerate(tickers):
                beta_i = betas.get(ti, 1.0)
                beta_j = betas.get(tj, 1.0)
                
                # Systematic covariance
                systematic = beta_i * beta_j * market_var
                
                if i == j:
                    # Add idiosyncratic variance
                    idio_var = self.returns[ti].var() - beta_i**2 * market_var
                    idio_var = max(idio_var, 0.0001)  # Ensure positive
                    cov_matrix[i, j] = systematic + idio_var
                else:
                    cov_matrix[i, j] = systematic
        
        return pd.DataFrame(cov_matrix, index=tickers, columns=tickers)
    
    def get_optimal_weights(
        self,
        method: str = 'mean_variance',
        risk_aversion: float = None
    ) -> pd.Series:
        """
        Calculate optimal portfolio weights
        
        Args:
            method: 'mean_variance', 'min_variance', or 'equal_weight'
            risk_aversion: Risk aversion parameter (default from config)
        
        Returns:
            Series of optimal weights
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if risk_aversion is None:
            risk_aversion = self.config.risk_aversion
        
        # Get inputs
        expected_returns = self.get_expected_returns()
        cov_matrix = self.get_covariance_matrix()
        
        tickers = list(expected_returns.index)
        n = len(tickers)
        
        if method == 'equal_weight':
            weights = pd.Series(1.0/n, index=tickers)
        
        elif method == 'min_variance':
            # Minimum variance portfolio
            if SCIPY_AVAILABLE:
                cov = cov_matrix.values
                
                def portfolio_var(w):
                    return w @ cov @ w
                
                constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
                bounds = [(self.config.min_position, self.config.max_position) for _ in range(n)]
                
                result = minimize(
                    portfolio_var,
                    x0=np.ones(n) / n,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                
                weights = pd.Series(result.x, index=tickers)
            else:
                weights = pd.Series(1.0/n, index=tickers)
        
        elif method == 'mean_variance':
            # Mean-variance optimization
            if SCIPY_AVAILABLE:
                mu = expected_returns.values
                cov = cov_matrix.values
                
                def neg_utility(w):
                    ret = w @ mu
                    var = w @ cov @ w
                    return -(ret - 0.5 * risk_aversion * var)
                
                constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
                bounds = [(self.config.min_position, self.config.max_position) for _ in range(n)]
                
                result = minimize(
                    neg_utility,
                    x0=np.ones(n) / n,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                
                weights = pd.Series(result.x, index=tickers)
            else:
                weights = pd.Series(1.0/n, index=tickers)
        
        else:
            weights = pd.Series(1.0/n, index=tickers)
        
        # Normalize
        weights = weights / weights.sum()
        
        return weights
    
    def get_dynamic_beta_series(self, ticker: str) -> pd.Series:
        """Get time-varying beta series for a ticker"""
        if ticker not in self.dynamic_betas:
            return pd.Series()
        return pd.Series(self.dynamic_betas[ticker], index=self.returns.index)
    
    def get_volatility_series(self, ticker: str) -> pd.Series:
        """Get volatility series for a ticker"""
        if ticker not in self.volatilities:
            return pd.Series()
        return pd.Series(self.volatilities[ticker], index=self.returns.index)
    
    def summary(self) -> Dict:
        """Get model summary"""
        if not self.fitted:
            return {'fitted': False}
        
        current_betas = self.layer1_factor.get_current_betas()
        
        return {
            'fitted': True,
            'n_assets': len(self.returns.columns),
            'n_days': len(self.returns),
            'date_range': (self.returns.index[0], self.returns.index[-1]),
            'current_betas': current_betas,
            'avg_beta': np.mean(list(current_betas.values())),
            'sentiment_loaded': self.layer2_sentiment.sentiment_data is not None
        }


def main():
    """Test the model"""
    print("\n" + "=" * 70)
    print("ENHANCED INTEGRATED MODEL - TEST")
    print("=" * 70)
    
    # Initialize
    config = ModelConfig()
    model = EnhancedIntegratedModel(config)
    
    # Load data
    print("\nLoading data...")
    if model.load_data():
        # Fit
        model.fit()
        
        # Get weights
        print("\nCalculating optimal weights...")
        weights = model.get_optimal_weights(method='mean_variance')
        
        print("\nTop 10 Weights:")
        for ticker, weight in weights.nlargest(10).items():
            print(f"   {ticker}: {weight:.2%}")
        
        # Summary
        print("\nModel Summary:")
        summary = model.summary()
        print(f"   Assets: {summary['n_assets']}")
        print(f"   Days: {summary['n_days']}")
        print(f"   Avg Beta: {summary['avg_beta']:.3f}")
    
    print("\n✅ Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
