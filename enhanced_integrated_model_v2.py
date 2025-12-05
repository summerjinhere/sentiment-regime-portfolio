"""
Enhanced Integrated Model with Statistical Validation
======================================================

Weakness Compensation Framework - Complete Implementation

Model Layers:
- Layer 1: Dynamic Factor Loading (Kalman Filter - Time-varying Beta)
- Layer 2: Sentiment Integration (FinBERT signals)
- Layer 3: GARCH-MIDAS Volatility

Statistical Validation:
- Alpha/Beta with HAC (Newey-West) standard errors
- Bootstrap confidence intervals (block bootstrap)
- Subperiod robustness analysis
- Transaction cost sensitivity
- Regime-conditional performance

NO SYNTHETIC DATA - Real market data only.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
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
    from statsmodels.regression.linear_model import OLS
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Complete model configuration"""
    
    # Kalman Filter parameters
    kalman_observation_noise: float = 0.01
    kalman_process_noise: float = 0.001
    kalman_initial_beta: float = 1.0
    
    # Sentiment parameters
    sentiment_decay: float = 0.9
    sentiment_weight: float = 0.2
    sentiment_threshold: float = 0.1
    
    # GARCH-MIDAS parameters
    garch_omega: float = 0.00001
    garch_alpha: float = 0.1
    garch_beta: float = 0.85
    midas_k: int = 22  # Monthly lag
    
    # Portfolio parameters
    risk_aversion: float = 2.0
    max_position: float = 0.15
    min_position: float = 0.0
    
    # Backtest parameters
    transaction_cost_bps: float = 10.0
    risk_free_rate: float = 0.02
    rebalance_frequency: str = 'monthly'  # daily, weekly, monthly
    
    # Statistical validation
    bootstrap_samples: int = 1000
    bootstrap_block_size: int = 22
    hac_maxlags: int = 5
    confidence_level: float = 0.95


# =============================================================================
# LAYER 1: DYNAMIC FACTOR LOADING (KALMAN FILTER)
# =============================================================================

class KalmanFilterBeta:
    """
    Time-varying beta estimation using Kalman Filter
    
    State equation: beta_t = beta_{t-1} + w_t, w_t ~ N(0, Q)
    Observation equation: r_t = beta_t * r_m,t + v_t, v_t ~ N(0, R)
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.beta = config.kalman_initial_beta
        self.P = 1.0  # State variance
        self.Q = config.kalman_process_noise
        self.R = config.kalman_observation_noise
        self.beta_history: List[float] = []
        self.P_history: List[float] = []
    
    def update(self, stock_return: float, market_return: float) -> float:
        """Single Kalman filter update step"""
        # Prediction
        beta_pred = self.beta
        P_pred = self.P + self.Q
        
        # Update (only if market moved)
        if abs(market_return) > 1e-8:
            innovation = stock_return - beta_pred * market_return
            S = P_pred * market_return**2 + self.R
            K = P_pred * market_return / S  # Kalman gain
            
            self.beta = beta_pred + K * innovation
            self.P = (1 - K * market_return) * P_pred
        
        self.beta_history.append(self.beta)
        self.P_history.append(self.P)
        
        return self.beta
    
    def fit(self, stock_returns: np.ndarray, market_returns: np.ndarray) -> np.ndarray:
        """Fit to entire series"""
        self.reset()
        for r_s, r_m in zip(stock_returns, market_returns):
            self.update(r_s, r_m)
        return np.array(self.beta_history)
    
    def reset(self):
        self.beta = self.config.kalman_initial_beta
        self.P = 1.0
        self.beta_history = []
        self.P_history = []
    
    def get_beta_std(self) -> np.ndarray:
        """Get standard deviation of beta estimates"""
        return np.sqrt(self.P_history)


class DynamicFactorLoading:
    """Layer 1: Dynamic Factor Loading Manager"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.filters: Dict[str, KalmanFilterBeta] = {}
        self.betas: Dict[str, np.ndarray] = {}
        self.beta_stds: Dict[str, np.ndarray] = {}
    
    def fit(self, returns: pd.DataFrame, market_returns: pd.Series) -> Dict[str, np.ndarray]:
        """Fit dynamic betas for all assets"""
        common_idx = returns.index.intersection(market_returns.index)
        returns = returns.loc[common_idx]
        market = market_returns.loc[common_idx].values
        
        self.betas = {}
        self.beta_stds = {}
        
        for ticker in returns.columns:
            filt = KalmanFilterBeta(self.config)
            betas = filt.fit(returns[ticker].values, market)
            
            self.betas[ticker] = betas
            self.beta_stds[ticker] = filt.get_beta_std()
            self.filters[ticker] = filt
        
        return self.betas
    
    def get_current_betas(self) -> Dict[str, float]:
        """Get most recent beta estimates"""
        return {t: b[-1] for t, b in self.betas.items()}
    
    def get_beta_series(self, ticker: str, index: pd.Index = None) -> pd.Series:
        """Get beta as pandas Series"""
        if ticker not in self.betas:
            return pd.Series()
        betas = self.betas[ticker]
        if index is not None and len(index) == len(betas):
            return pd.Series(betas, index=index)
        return pd.Series(betas)


# =============================================================================
# LAYER 2: SENTIMENT INTEGRATION
# =============================================================================

class SentimentIntegration:
    """Layer 2: Sentiment signal integration"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.sentiment_data: Optional[pd.DataFrame] = None
        self.smoothed_sentiment: Optional[pd.DataFrame] = None
        self.loaded = False
    
    def load(self, filepath: str = 'real_sentiment_data.csv') -> bool:
        """Load sentiment data from CSV"""
        try:
            self.sentiment_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            self._smooth_sentiment()
            self.loaded = True
            print(f"✅ Sentiment loaded: {len(self.sentiment_data)} days, "
                  f"{len(self.sentiment_data.columns)} tickers")
            return True
        except Exception as e:
            print(f"⚠️ Sentiment load failed: {e}")
            self.loaded = False
            return False
    
    def _smooth_sentiment(self):
        """Apply exponential smoothing"""
        if self.sentiment_data is not None:
            decay = self.config.sentiment_decay
            self.smoothed_sentiment = self.sentiment_data.ewm(alpha=1-decay).mean()
    
    def get_signal(self, date, ticker: str = None) -> float:
        """Get sentiment signal for date/ticker"""
        if self.smoothed_sentiment is None:
            return 0.0
        try:
            if ticker and ticker in self.smoothed_sentiment.columns:
                return float(self.smoothed_sentiment.loc[date, ticker])
            elif 'MARKET_SENTIMENT' in self.smoothed_sentiment.columns:
                return float(self.smoothed_sentiment.loc[date, 'MARKET_SENTIMENT'])
        except:
            pass
        return 0.0
    
    def get_all_signals(self, date) -> pd.Series:
        """Get all sentiment signals for a date"""
        if self.smoothed_sentiment is None:
            return pd.Series()
        try:
            return self.smoothed_sentiment.loc[date]
        except:
            return pd.Series()
    
    def adjust_expected_returns(self, base_returns: pd.Series, date) -> pd.Series:
        """Adjust expected returns based on sentiment"""
        adjusted = base_returns.copy()
        weight = self.config.sentiment_weight
        
        for ticker in base_returns.index:
            sent = self.get_signal(date, ticker)
            adjusted[ticker] = base_returns[ticker] * (1 + weight * sent)
        
        return adjusted


# =============================================================================
# LAYER 3: GARCH-MIDAS VOLATILITY
# =============================================================================

class GARCH_MIDAS:
    """
    Layer 3: GARCH-MIDAS volatility model
    
    σ²_t = τ_t * g_t
    where:
    - g_t: Short-term GARCH(1,1) component
    - τ_t: Long-term MIDAS component
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.omega = config.garch_omega
        self.alpha = config.garch_alpha
        self.beta = config.garch_beta
        self.K = config.midas_k
        
        self.g_t = 0.0001
        self.tau_t = 0.0001
        self.vol_history: List[float] = []
    
    def _midas_weights(self, k: int) -> np.ndarray:
        """Compute MIDAS beta polynomial weights"""
        weights = np.array([(1 - i/k)**2 for i in range(k)])
        return weights / weights.sum()
    
    def fit(self, returns: np.ndarray) -> np.ndarray:
        """Fit GARCH-MIDAS model"""
        T = len(returns)
        volatilities = np.zeros(T)
        
        # Initialize
        self.g_t = np.var(returns[:min(20, T)]) if T > 20 else 0.0001
        self.tau_t = np.var(returns[:min(60, T)]) if T > 60 else 0.0001
        
        for t in range(T):
            # Update MIDAS (monthly)
            if t >= self.K:
                weights = self._midas_weights(self.K)
                rv = returns[t-self.K:t]**2
                self.tau_t = np.dot(weights, rv)
            
            # Update GARCH (daily)
            if t > 0:
                self.g_t = self.omega + self.alpha * returns[t-1]**2 + self.beta * self.g_t
            
            # Total variance
            sigma2 = self.tau_t * (1 + self.g_t / max(self.tau_t, 1e-10))
            volatilities[t] = np.sqrt(max(sigma2, 1e-10))
        
        self.vol_history = volatilities
        return volatilities
    
    def forecast(self, h: int = 1) -> float:
        """Forecast volatility h steps ahead"""
        g = self.g_t
        for _ in range(h):
            g = self.omega + (self.alpha + self.beta) * g
        sigma2 = self.tau_t * (1 + g / max(self.tau_t, 1e-10))
        return np.sqrt(max(sigma2, 1e-10))


# =============================================================================
# STATISTICAL VALIDATOR
# =============================================================================

class StatisticalValidator:
    """Complete statistical validation suite"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def calculate_alpha_beta(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, Any]:
        """
        Calculate alpha and beta with HAC standard errors
        
        Returns:
            Dict with alpha, beta, t-stats, p-values, confidence intervals
        """
        # Align returns
        common = strategy_returns.index.intersection(benchmark_returns.index)
        y = strategy_returns.loc[common].values
        x = benchmark_returns.loc[common].values
        n = len(y)
        
        # Excess returns
        rf_daily = self.config.risk_free_rate / 252
        y_excess = y - rf_daily
        x_excess = x - rf_daily
        
        # OLS regression
        X = np.column_stack([np.ones(n), x_excess])
        
        if STATSMODELS_AVAILABLE:
            model = OLS(y_excess, X)
            # HAC standard errors (Newey-West)
            results = model.fit(cov_type='HAC', cov_kwds={'maxlags': self.config.hac_maxlags})
            
            alpha = results.params[0]
            beta = results.params[1]
            alpha_se = results.bse[0]
            beta_se = results.bse[1]
            alpha_tstat = results.tvalues[0]
            beta_tstat = results.tvalues[1]
            alpha_pval = results.pvalues[0]
            beta_pval = results.pvalues[1]
            r_squared = results.rsquared
        else:
            # Fallback: Simple OLS
            beta, alpha = np.polyfit(x_excess, y_excess, 1)
            residuals = y_excess - (alpha + beta * x_excess)
            
            # Simple standard errors
            mse = np.sum(residuals**2) / (n - 2)
            x_var = np.var(x_excess) * n
            alpha_se = np.sqrt(mse * (1/n + np.mean(x_excess)**2 / x_var))
            beta_se = np.sqrt(mse / x_var)
            
            alpha_tstat = alpha / alpha_se if alpha_se > 0 else 0
            beta_tstat = (beta - 1) / beta_se if beta_se > 0 else 0
            
            alpha_pval = 2 * (1 - stats.t.cdf(abs(alpha_tstat), n-2)) if SCIPY_AVAILABLE else 0.5
            beta_pval = 2 * (1 - stats.t.cdf(abs(beta_tstat), n-2)) if SCIPY_AVAILABLE else 0.5
            
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_excess - np.mean(y_excess))**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # Confidence intervals
        z = stats.norm.ppf((1 + self.config.confidence_level) / 2) if SCIPY_AVAILABLE else 1.96
        
        alpha_annual = alpha * 252
        alpha_se_annual = alpha_se * np.sqrt(252)
        
        return {
            'alpha_daily': alpha,
            'alpha_annual': alpha_annual,
            'alpha_se': alpha_se_annual,
            'alpha_tstat': alpha_tstat,
            'alpha_pval': alpha_pval,
            'alpha_ci_lower': alpha_annual - z * alpha_se_annual,
            'alpha_ci_upper': alpha_annual + z * alpha_se_annual,
            'beta': beta,
            'beta_se': beta_se,
            'beta_tstat': beta_tstat,
            'beta_pval': beta_pval,
            'beta_ci_lower': beta - z * beta_se,
            'beta_ci_upper': beta + z * beta_se,
            'r_squared': r_squared,
            'n_observations': n,
            'significant': alpha_pval < (1 - self.config.confidence_level)
        }
    
    def bootstrap_analysis(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, Any]:
        """
        Block bootstrap analysis for alpha and Sharpe ratio
        
        Uses overlapping block bootstrap to preserve autocorrelation
        """
        common = strategy_returns.index.intersection(benchmark_returns.index)
        y = strategy_returns.loc[common].values
        x = benchmark_returns.loc[common].values
        n = len(y)
        
        block_size = self.config.bootstrap_block_size
        n_samples = self.config.bootstrap_samples
        rf_daily = self.config.risk_free_rate / 252
        
        bootstrap_alphas = []
        bootstrap_sharpes = []
        
        for _ in range(n_samples):
            # Block bootstrap indices
            n_blocks = int(np.ceil(n / block_size))
            block_starts = np.random.randint(0, n - block_size + 1, n_blocks)
            
            indices = []
            for start in block_starts:
                indices.extend(range(start, min(start + block_size, n)))
            indices = np.array(indices[:n])
            
            # Bootstrap sample
            y_boot = y[indices]
            x_boot = x[indices]
            
            # Calculate alpha
            y_excess = y_boot - rf_daily
            x_excess = x_boot - rf_daily
            
            if len(x_excess) > 2:
                beta_boot, alpha_boot = np.polyfit(x_excess, y_excess, 1)
                bootstrap_alphas.append(alpha_boot * 252)  # Annualize
                
                # Calculate Sharpe
                excess_ret = y_boot - rf_daily
                sharpe_boot = np.mean(excess_ret) / np.std(excess_ret) * np.sqrt(252) if np.std(excess_ret) > 0 else 0
                bootstrap_sharpes.append(sharpe_boot)
        
        bootstrap_alphas = np.array(bootstrap_alphas)
        bootstrap_sharpes = np.array(bootstrap_sharpes)
        
        # Calculate confidence intervals
        ci = (1 - self.config.confidence_level) / 2
        
        return {
            'alpha_mean': np.mean(bootstrap_alphas),
            'alpha_std': np.std(bootstrap_alphas),
            'alpha_ci_lower': np.percentile(bootstrap_alphas, ci * 100),
            'alpha_ci_upper': np.percentile(bootstrap_alphas, (1 - ci) * 100),
            'alpha_zero_in_ci': (np.percentile(bootstrap_alphas, ci * 100) <= 0 <= 
                                np.percentile(bootstrap_alphas, (1 - ci) * 100)),
            'sharpe_mean': np.mean(bootstrap_sharpes),
            'sharpe_std': np.std(bootstrap_sharpes),
            'sharpe_ci_lower': np.percentile(bootstrap_sharpes, ci * 100),
            'sharpe_ci_upper': np.percentile(bootstrap_sharpes, (1 - ci) * 100),
            'sharpe_zero_in_ci': (np.percentile(bootstrap_sharpes, ci * 100) <= 0 <= 
                                 np.percentile(bootstrap_sharpes, (1 - ci) * 100)),
            'n_samples': n_samples,
            'bootstrap_alphas': bootstrap_alphas,
            'bootstrap_sharpes': bootstrap_sharpes
        }
    
    def subperiod_analysis(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        n_periods: int = 4
    ) -> List[Dict[str, Any]]:
        """Analyze performance across subperiods"""
        common = strategy_returns.index.intersection(benchmark_returns.index)
        y = strategy_returns.loc[common]
        x = benchmark_returns.loc[common]
        
        period_size = len(y) // n_periods
        results = []
        
        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < n_periods - 1 else len(y)
            
            y_period = y.iloc[start_idx:end_idx]
            x_period = x.iloc[start_idx:end_idx]
            
            # Calculate metrics
            alpha_beta = self.calculate_alpha_beta(y_period, x_period)
            
            total_ret = (1 + y_period).prod() - 1
            rf_daily = self.config.risk_free_rate / 252
            excess = y_period - rf_daily
            sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
            
            cum = (1 + y_period).cumprod()
            max_dd = (cum / cum.cummax() - 1).min()
            
            results.append({
                'period': i + 1,
                'start_date': y_period.index[0],
                'end_date': y_period.index[-1],
                'n_days': len(y_period),
                'total_return': total_ret,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'alpha_annual': alpha_beta['alpha_annual'],
                'alpha_pval': alpha_beta['alpha_pval'],
                'beta': alpha_beta['beta'],
                'significant': alpha_beta['significant']
            })
        
        return results
    
    def transaction_cost_sensitivity(
        self,
        strategy_returns: pd.Series,
        turnover: float = 0.30,
        cost_range: List[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze strategy sensitivity to transaction costs
        
        Args:
            turnover: Monthly turnover rate (default 30%)
            cost_range: List of transaction costs in bps
        """
        if cost_range is None:
            cost_range = [0, 5, 10, 15, 20, 25, 30, 40, 50]
        
        rf_daily = self.config.risk_free_rate / 252
        results = []
        
        for cost_bps in cost_range:
            # Daily cost impact (assuming monthly turnover)
            daily_cost = (cost_bps / 10000) * turnover / 21  # ~21 trading days/month
            
            # Adjusted returns
            adj_returns = strategy_returns - daily_cost
            
            # Calculate metrics
            total_ret = (1 + adj_returns).prod() - 1
            n_years = len(adj_returns) / 252
            cagr = (1 + total_ret) ** (1/n_years) - 1 if n_years > 0 else 0
            
            excess = adj_returns - rf_daily
            sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
            
            results.append({
                'cost_bps': cost_bps,
                'daily_cost': daily_cost,
                'total_return': total_ret,
                'cagr': cagr,
                'sharpe_ratio': sharpe,
                'sharpe_positive': sharpe > 0
            })
        
        # Find breakeven cost
        breakeven = None
        for i, r in enumerate(results):
            if r['sharpe_ratio'] <= 0:
                if i > 0:
                    # Interpolate
                    prev = results[i-1]
                    slope = (r['sharpe_ratio'] - prev['sharpe_ratio']) / (r['cost_bps'] - prev['cost_bps'])
                    breakeven = prev['cost_bps'] - prev['sharpe_ratio'] / slope if slope != 0 else prev['cost_bps']
                break
        
        return {
            'results': results,
            'breakeven_cost_bps': breakeven,
            'robust_at_50bps': results[-1]['sharpe_ratio'] > 0 if results else False
        }
    
    def regime_robustness(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        regimes: pd.Series
    ) -> Dict[str, Any]:
        """Analyze performance by market regime"""
        common = strategy_returns.index.intersection(benchmark_returns.index).intersection(regimes.index)
        
        strategy = strategy_returns.loc[common]
        benchmark = benchmark_returns.loc[common]
        regime = regimes.loc[common]
        
        results = {}
        rf_daily = self.config.risk_free_rate / 252
        
        for regime_name in regime.unique():
            mask = regime == regime_name
            y = strategy[mask]
            x = benchmark[mask]
            
            if len(y) < 30:
                continue
            
            # Calculate metrics
            alpha_beta = self.calculate_alpha_beta(y, x)
            
            excess = y - rf_daily
            sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
            
            total_ret = (1 + y).prod() - 1
            n_days = len(y)
            ann_ret = (1 + total_ret) ** (252/n_days) - 1 if n_days > 0 else 0
            
            results[regime_name] = {
                'n_days': n_days,
                'pct_of_total': n_days / len(strategy),
                'total_return': total_ret,
                'annualized_return': ann_ret,
                'sharpe_ratio': sharpe,
                'alpha_annual': alpha_beta['alpha_annual'],
                'alpha_pval': alpha_beta['alpha_pval'],
                'beta': alpha_beta['beta'],
                'significant': alpha_beta['significant']
            }
        
        return results


# =============================================================================
# PERFORMANCE CALCULATOR
# =============================================================================

class PerformanceCalculator:
    """Calculate comprehensive performance metrics"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate all performance metrics"""
        if returns.empty or len(returns) < 2:
            return {}
        
        rf_daily = self.config.risk_free_rate / 252
        excess = returns - rf_daily
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / 252
        cagr = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
        volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted
        sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
        
        downside = excess[excess < 0]
        sortino = excess.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0
        
        # Drawdown
        cum = (1 + returns).cumprod()
        running_max = cum.cummax()
        drawdown = cum / running_max - 1
        max_drawdown = drawdown.min()
        
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Distribution
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'n_days': len(returns),
            'n_years': n_years
        }


# =============================================================================
# MAIN INTEGRATED MODEL
# =============================================================================

class EnhancedIntegratedModel:
    """
    Weakness Compensation Framework - Complete Implementation
    
    Integrates:
    - Layer 1: Dynamic Factor Loading (Kalman Filter)
    - Layer 2: Sentiment Integration (FinBERT)
    - Layer 3: GARCH-MIDAS Volatility
    - Statistical Validation Suite
    """
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        
        # Model layers
        self.layer1_factor = DynamicFactorLoading(self.config)
        self.layer2_sentiment = SentimentIntegration(self.config)
        self.layer3_volatility: Dict[str, GARCH_MIDAS] = {}
        
        # Validators
        self.validator = StatisticalValidator(self.config)
        self.performance = PerformanceCalculator(self.config)
        
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
        benchmark: str = 'SPY',
        trading_days: int = None
    ) -> bool:
        """Load real data from CSV files"""
        try:
            print("\n" + "=" * 70)
            print("LOADING DATA")
            print("=" * 70)
            
            # Load price data
            data = pd.read_csv(prices_file, index_col=0, parse_dates=True)
            
            # Extract returns
            ret_cols = [c for c in data.columns if c.endswith('_ret')]
            self.returns = data[ret_cols].copy()
            self.returns.columns = [c.replace('_ret', '') for c in ret_cols]
            
            # Extract prices
            price_cols = [c for c in data.columns if c.endswith('_price')]
            self.prices = data[price_cols].copy()
            self.prices.columns = [c.replace('_price', '') for c in price_cols]
            
            # Filter by trading days
            if trading_days and len(self.returns) > trading_days:
                self.returns = self.returns.iloc[-trading_days:]
                self.prices = self.prices.iloc[-trading_days:]
                print(f"   Filtered to last {trading_days} trading days")
            
            # Market returns
            if benchmark in self.returns.columns:
                self.market_returns = self.returns[benchmark]
            elif 'SPY' in self.returns.columns:
                self.market_returns = self.returns['SPY']
            else:
                self.market_returns = self.returns.mean(axis=1)
            
            print(f"✅ Price data: {len(self.returns)} days, {len(self.returns.columns)} assets")
            print(f"   Date range: {self.returns.index[0].date()} to {self.returns.index[-1].date()}")
            print(f"   Benchmark: {benchmark}")
            
            # Load sentiment
            self.layer2_sentiment.load(sentiment_file)
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def fit(self) -> bool:
        """Fit all model components"""
        if self.returns is None:
            print("❌ No data loaded. Call load_data() first.")
            return False
        
        print("\n" + "=" * 70)
        print("FITTING MODEL COMPONENTS")
        print("=" * 70)
        
        # Layer 1: Dynamic Betas
        print("\n[Layer 1] Dynamic Factor Loading (Kalman Filter)...")
        self.dynamic_betas = self.layer1_factor.fit(self.returns, self.market_returns)
        avg_beta = np.mean([b[-1] for b in self.dynamic_betas.values()])
        print(f"   ✅ Fitted {len(self.dynamic_betas)} assets")
        print(f"   Average current beta: {avg_beta:.3f}")
        
        # Layer 2: Sentiment already loaded
        print("\n[Layer 2] Sentiment Integration...")
        if self.layer2_sentiment.loaded:
            print(f"   ✅ Sentiment data active")
        else:
            print(f"   ⚠️ Sentiment data not loaded")
        
        # Layer 3: Volatility
        print("\n[Layer 3] GARCH-MIDAS Volatility...")
        for ticker in self.returns.columns:
            garch = GARCH_MIDAS(self.config)
            self.volatilities[ticker] = garch.fit(self.returns[ticker].values)
            self.layer3_volatility[ticker] = garch
        print(f"   ✅ Fitted {len(self.volatilities)} volatility models")
        
        self.fitted = True
        print("\n" + "=" * 70)
        print("✅ MODEL FITTING COMPLETE")
        print("=" * 70)
        
        return True
    
    def get_optimal_weights(self, method: str = 'mean_variance') -> pd.Series:
        """Calculate optimal portfolio weights"""
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        # Get expected returns (sentiment-adjusted)
        base_returns = self.returns.mean()
        date = self.returns.index[-1]
        expected_returns = self.layer2_sentiment.adjust_expected_returns(base_returns, date)
        
        # Get covariance matrix
        cov_matrix = self.returns.cov()
        
        tickers = list(expected_returns.index)
        n = len(tickers)
        
        if method == 'equal_weight':
            weights = pd.Series(1.0/n, index=tickers)
        
        elif method == 'min_variance' and SCIPY_AVAILABLE:
            cov = cov_matrix.values
            
            def portfolio_var(w):
                return w @ cov @ w
            
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(self.config.min_position, self.config.max_position) for _ in range(n)]
            
            result = minimize(portfolio_var, x0=np.ones(n)/n, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            weights = pd.Series(result.x, index=tickers)
        
        elif method == 'mean_variance' and SCIPY_AVAILABLE:
            mu = expected_returns.values
            cov = cov_matrix.values
            gamma = self.config.risk_aversion
            
            def neg_utility(w):
                ret = w @ mu
                var = w @ cov @ w
                return -(ret - 0.5 * gamma * var)
            
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(self.config.min_position, self.config.max_position) for _ in range(n)]
            
            result = minimize(neg_utility, x0=np.ones(n)/n, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            weights = pd.Series(result.x, index=tickers)
        
        else:
            weights = pd.Series(1.0/n, index=tickers)
        
        return weights / weights.sum()
    
    def run_backtest(
        self,
        weights: pd.Series = None,
        rebalance: str = 'monthly'
    ) -> pd.Series:
        """Run backtest with given weights"""
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        if weights is None:
            weights = self.get_optimal_weights()
        
        # Get common tickers
        common = [t for t in weights.index if t in self.returns.columns]
        w = weights[common]
        w = w / w.sum()
        
        # Calculate portfolio returns
        portfolio_returns = (self.returns[common] * w).sum(axis=1)
        
        return portfolio_returns
    
    def validate(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series = None,
        run_bootstrap: bool = True,
        n_subperiods: int = 4
    ) -> Dict[str, Any]:
        """Run complete statistical validation"""
        if benchmark_returns is None:
            benchmark_returns = self.market_returns
        
        print("\n" + "=" * 70)
        print("STATISTICAL VALIDATION")
        print("=" * 70)
        
        results = {}
        
        # 1. Performance metrics
        print("\n[1] Performance Metrics...")
        results['performance'] = self.performance.calculate_metrics(strategy_returns)
        print(f"   Sharpe Ratio: {results['performance']['sharpe_ratio']:.3f}")
        print(f"   Total Return: {results['performance']['total_return']:.2%}")
        
        # 2. Alpha/Beta
        print("\n[2] Alpha/Beta (HAC Standard Errors)...")
        results['alpha_beta'] = self.validator.calculate_alpha_beta(
            strategy_returns, benchmark_returns
        )
        print(f"   Alpha (annual): {results['alpha_beta']['alpha_annual']:.2%}")
        print(f"   Alpha t-stat: {results['alpha_beta']['alpha_tstat']:.3f}")
        print(f"   Alpha p-value: {results['alpha_beta']['alpha_pval']:.4f}")
        print(f"   Significant: {'✅ YES' if results['alpha_beta']['significant'] else '❌ NO'}")
        
        # 3. Bootstrap
        if run_bootstrap:
            print(f"\n[3] Bootstrap Analysis ({self.config.bootstrap_samples} samples)...")
            results['bootstrap'] = self.validator.bootstrap_analysis(
                strategy_returns, benchmark_returns
            )
            print(f"   Alpha 95% CI: [{results['bootstrap']['alpha_ci_lower']:.2%}, "
                  f"{results['bootstrap']['alpha_ci_upper']:.2%}]")
            print(f"   Zero in CI: {'❌ YES' if results['bootstrap']['alpha_zero_in_ci'] else '✅ NO'}")
        
        # 4. Subperiod
        print(f"\n[4] Subperiod Analysis ({n_subperiods} periods)...")
        results['subperiods'] = self.validator.subperiod_analysis(
            strategy_returns, benchmark_returns, n_subperiods
        )
        positive_sharpe = sum(1 for p in results['subperiods'] if p['sharpe_ratio'] > 0)
        significant_alpha = sum(1 for p in results['subperiods'] if p['significant'])
        print(f"   Positive Sharpe: {positive_sharpe}/{n_subperiods}")
        print(f"   Significant Alpha: {significant_alpha}/{n_subperiods}")
        
        # 5. Transaction costs
        print("\n[5] Transaction Cost Sensitivity...")
        results['transaction_costs'] = self.validator.transaction_cost_sensitivity(
            strategy_returns
        )
        if results['transaction_costs']['breakeven_cost_bps']:
            print(f"   Breakeven cost: {results['transaction_costs']['breakeven_cost_bps']:.1f} bps")
        print(f"   Robust at 50bps: {'✅ YES' if results['transaction_costs']['robust_at_50bps'] else '❌ NO'}")
        
        print("\n" + "=" * 70)
        print("✅ VALIDATION COMPLETE")
        print("=" * 70)
        
        return results
    
    def get_dynamic_beta_series(self, ticker: str) -> pd.Series:
        """Get time-varying beta as Series"""
        return self.layer1_factor.get_beta_series(ticker, self.returns.index)
    
    def get_volatility_series(self, ticker: str) -> pd.Series:
        """Get volatility as Series"""
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
            'benchmark': 'SPY',
            'current_betas': current_betas,
            'avg_beta': np.mean(list(current_betas.values())),
            'sentiment_loaded': self.layer2_sentiment.loaded,
            'config': {
                'risk_free_rate': self.config.risk_free_rate,
                'transaction_cost_bps': self.config.transaction_cost_bps,
                'bootstrap_samples': self.config.bootstrap_samples
            }
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Test the model"""
    print("\n" + "=" * 70)
    print("ENHANCED INTEGRATED MODEL - TEST")
    print("Weakness Compensation Framework")
    print("=" * 70)
    
    # Initialize
    config = ModelConfig(
        bootstrap_samples=500,  # Faster for testing
        transaction_cost_bps=10
    )
    model = EnhancedIntegratedModel(config)
    
    # Load data
    if model.load_data():
        # Fit
        model.fit()
        
        # Get weights
        print("\n" + "=" * 70)
        print("PORTFOLIO WEIGHTS")
        print("=" * 70)
        weights = model.get_optimal_weights(method='mean_variance')
        print("\nTop 10 Holdings:")
        for ticker, weight in weights.nlargest(10).items():
            print(f"   {ticker}: {weight:.2%}")
        
        # Backtest
        strategy_returns = model.run_backtest(weights)
        
        # Validate
        validation = model.validate(strategy_returns, run_bootstrap=True)
        
        # Summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        
        perf = validation['performance']
        ab = validation['alpha_beta']
        
        print(f"""
Performance:
  Total Return:    {perf['total_return']:>10.2%}
  CAGR:            {perf['cagr']:>10.2%}
  Sharpe Ratio:    {perf['sharpe_ratio']:>10.3f}
  Max Drawdown:    {perf['max_drawdown']:>10.2%}

Statistical Significance:
  Alpha (annual):  {ab['alpha_annual']:>10.2%}
  Alpha t-stat:    {ab['alpha_tstat']:>10.3f}
  Alpha p-value:   {ab['alpha_pval']:>10.4f}
  Beta:            {ab['beta']:>10.3f}
  R-squared:       {ab['r_squared']:>10.2%}
  
  Verdict: {'✅ STATISTICALLY SIGNIFICANT' if ab['significant'] else '❌ NOT SIGNIFICANT'}
""")


if __name__ == "__main__":
    main()
