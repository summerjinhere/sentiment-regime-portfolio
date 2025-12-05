"""
Enhanced Integrated Model v3 - 4-Regime Support
================================================

Duke Feedback Incorporated:
1. 4-Regime Framework (Vol × Market Direction)
2. Train/Test Split Support
3. Realistic Transaction Costs (NASDAQ-calibrated)
4. FF5 + Momentum Factor Adjustment

Model Layers:
- Layer 1: Dynamic Factor Loading (Kalman Filter)
- Layer 2: Sentiment Integration (FinBERT)
- Layer 3: GARCH-MIDAS Volatility

NO SYNTHETIC DATA - Real market data only.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
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
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# =============================================================================
# ENUMS & CONFIGURATION
# =============================================================================

class MarketRegime(Enum):
    """4-Regime Classification based on Volatility × Market Direction"""
    LOW_VOL_BEAR = "LowVol_Bear"
    LOW_VOL_BULL = "LowVol_Bull"
    HIGH_VOL_BEAR = "HighVol_Bear"
    HIGH_VOL_BULL = "HighVol_Bull"


@dataclass
class ModelConfig:
    """Complete model configuration with Duke feedback incorporated"""
    
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
    midas_k: int = 22
    
    # Portfolio parameters
    risk_aversion: float = 2.0
    max_position: float = 0.15
    min_position: float = 0.0
    
    # Backtest parameters - NASDAQ calibrated (Duke feedback)
    transaction_cost_bps: float = 8.0  # Realistic for institutional
    risk_free_rate: float = 0.03
    rebalance_frequency: str = 'monthly'
    
    # Statistical validation
    bootstrap_samples: int = 1000
    bootstrap_block_size: int = 22
    hac_maxlags: int = 5
    confidence_level: float = 0.95
    
    # 4-Regime settings (Duke feedback)
    vol_threshold: float = 0.0
    market_direction_window: int = 20


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
        self.P = 1.0
        self.Q = config.kalman_process_noise
        self.R = config.kalman_observation_noise
        self.beta_history: List[float] = []
        self.P_history: List[float] = []
    
    def update(self, stock_return: float, market_return: float) -> float:
        """Single Kalman filter update step"""
        beta_pred = self.beta
        P_pred = self.P + self.Q
        
        if abs(market_return) > 1e-8:
            innovation = stock_return - beta_pred * market_return
            S = P_pred * market_return**2 + self.R
            K = P_pred * market_return / S
            
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
        return {t: b[-1] for t, b in self.betas.items()}
    
    def get_beta_series(self, ticker: str, index: pd.Index = None) -> pd.Series:
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
        if self.sentiment_data is not None:
            decay = self.config.sentiment_decay
            self.smoothed_sentiment = self.sentiment_data.ewm(alpha=1-decay).mean()
    
    def get_signal(self, date, ticker: str = None) -> float:
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
        if self.smoothed_sentiment is None:
            return pd.Series()
        try:
            return self.smoothed_sentiment.loc[date]
        except:
            return pd.Series()
    
    def adjust_expected_returns(self, base_returns: pd.Series, date) -> pd.Series:
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
        weights = np.array([(1 - i/k)**2 for i in range(k)])
        return weights / weights.sum()
    
    def fit(self, returns: np.ndarray) -> np.ndarray:
        T = len(returns)
        volatilities = np.zeros(T)
        
        self.g_t = np.var(returns[:min(20, T)]) if T > 20 else 0.0001
        self.tau_t = np.var(returns[:min(60, T)]) if T > 60 else 0.0001
        
        for t in range(T):
            if t >= self.K:
                weights = self._midas_weights(self.K)
                rv = returns[t-self.K:t]**2
                self.tau_t = np.dot(weights, rv)
            
            if t > 0:
                self.g_t = self.omega + self.alpha * returns[t-1]**2 + self.beta * self.g_t
            
            sigma2 = self.tau_t * (1 + self.g_t / max(self.tau_t, 1e-10))
            volatilities[t] = np.sqrt(max(sigma2, 1e-10))
        
        self.vol_history = volatilities
        return volatilities
    
    def forecast(self, h: int = 1) -> float:
        g = self.g_t
        for _ in range(h):
            g = self.omega + (self.alpha + self.beta) * g
        sigma2 = self.tau_t * (1 + g / max(self.tau_t, 1e-10))
        return np.sqrt(max(sigma2, 1e-10))


# =============================================================================
# 4-REGIME CLASSIFIER (Duke Feedback)
# =============================================================================

class RegimeClassifier:
    """
    4-Regime Classification based on Volatility × Market Direction
    
    Regimes:
    1. Low Vol + Bear (LB): Quiet decline
    2. Low Vol + Bull (LU): Quiet rally
    3. High Vol + Bear (HB): Crisis/Panic
    4. High Vol + Bull (HU): Recovery/Euphoria
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.market_direction: Optional[pd.Series] = None
        self.regimes: Optional[pd.DataFrame] = None
    
    def compute_market_direction(self, market_returns: pd.Series) -> pd.Series:
        """Compute market direction based on rolling returns"""
        window = self.config.market_direction_window
        rolling_ret = market_returns.rolling(window=window).mean()
        self.market_direction = (rolling_ret >= 0).astype(int) * 2 - 1
        return self.market_direction
    
    def classify(self, vol_series: pd.Series, market_returns: pd.Series) -> pd.DataFrame:
        """Classify each observation into one of 4 regimes."""
        if self.market_direction is None:
            self.compute_market_direction(market_returns)
        
        common_idx = vol_series.index.intersection(self.market_direction.index)
        vol = vol_series.loc[common_idx]
        direction = self.market_direction.loc[common_idx]
        
        vol_z = (vol - vol.mean()) / vol.std() if vol.std() > 0 else vol * 0
        
        low_vol = vol_z < self.config.vol_threshold
        high_vol = vol_z >= self.config.vol_threshold
        bear = direction < 0
        bull = direction >= 0
        
        regimes = pd.DataFrame(index=common_idx)
        regimes['vol_z'] = vol_z
        regimes['mkt_direction'] = direction
        regimes['reg_LB'] = (low_vol & bear).astype(int)
        regimes['reg_LU'] = (low_vol & bull).astype(int)
        regimes['reg_HB'] = (high_vol & bear).astype(int)
        regimes['reg_HU'] = (high_vol & bull).astype(int)
        
        conditions = [
            regimes['reg_LB'] == 1,
            regimes['reg_LU'] == 1,
            regimes['reg_HB'] == 1,
            regimes['reg_HU'] == 1,
        ]
        choices = [r.value for r in MarketRegime]
        regimes['regime'] = np.select(conditions, choices, default="Unknown")
        
        self.regimes = regimes
        return regimes


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
        """Calculate alpha and beta with HAC standard errors"""
        common = strategy_returns.index.intersection(benchmark_returns.index)
        y = strategy_returns.loc[common].values
        x = benchmark_returns.loc[common].values
        n = len(y)
        
        rf_daily = self.config.risk_free_rate / 252
        y_excess = y - rf_daily
        x_excess = x - rf_daily
        
        X = np.column_stack([np.ones(n), x_excess])
        
        if STATSMODELS_AVAILABLE:
            model = OLS(y_excess, X)
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
            beta, alpha = np.polyfit(x_excess, y_excess, 1)
            residuals = y_excess - (alpha + beta * x_excess)
            mse = np.sum(residuals**2) / (n - 2)
            x_var = np.var(x_excess) * n
            alpha_se = np.sqrt(mse * (1/n + np.mean(x_excess)**2 / x_var))
            beta_se = np.sqrt(mse / x_var)
            alpha_tstat = alpha / alpha_se if alpha_se > 0 else 0
            beta_tstat = (beta - 1) / beta_se if beta_se > 0 else 0
            alpha_pval = 0.5
            beta_pval = 0.5
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_excess - np.mean(y_excess))**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        z = 1.96
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
            'r_squared': r_squared,
            'significant': alpha_pval < (1 - self.config.confidence_level)
        }

    def compute_factor_alpha(
        self,
        returns: pd.Series,
        factors: pd.DataFrame,
        method: str = "OLS",
        nw_lags: int = 12,
        n_boot: int = 1000,
        block_len: int = 21,
        num_strategies: int = 10
    ) -> Dict[str, Any]:
        """
        Compute factor alpha using robust inference methods.
        Methods: OLS, NW (Newey-West), BlockBootstrap
        """
        # Align data
        common = returns.index.intersection(factors.index)
        y = returns.loc[common]
        X = factors.loc[common]
        
        # Excess returns (assuming returns are already excess or strategy LS)
        # If factors include RF, subtract it? Usually factors are excess returns.
        # We assume 'returns' is the strategy excess return.
        
        X = sm.add_constant(X)
        
        # 1. Base OLS fit
        model = OLS(y, X)
        
        if method == "NW":
            res = model.fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})
        else:
            res = model.fit()
            
        params = res.params
        tvalues = res.tvalues
        pvalues = res.pvalues
        conf_int = res.conf_int(alpha=0.05)
        
        # Robustly identify alpha (constant)
        if 'const' in params:
            alpha_val = params['const']
            alpha_t = tvalues['const']
            alpha_p = pvalues['const']
            alpha_ci_val = [conf_int.loc['const', 0], conf_int.loc['const', 1]]
        elif 'intercept' in params:
            alpha_val = params['intercept']
            alpha_t = tvalues['intercept']
            alpha_p = pvalues['intercept']
            alpha_ci_val = [conf_int.loc['intercept', 0], conf_int.loc['intercept', 1]]
        else:
            # Fallback: assume first term is constant if add_constant was used
            # But verify it's not a factor
            print(f"⚠️ 'const' not found in params: {params.index}. Using first param '{params.index[0]}'")
            first_param = params.index[0]
            alpha_val = params[first_param]
            alpha_t = tvalues[first_param]
            alpha_p = pvalues[first_param]
            alpha_ci_val = [conf_int.loc[first_param, 0], conf_int.loc[first_param, 1]]

        alpha_annual = alpha_val * 252
        alpha_tstat = alpha_t
        alpha_pval = alpha_p
        alpha_ci = [alpha_ci_val[0] * 252, alpha_ci_val[1] * 252]
        
        # 2. Block Bootstrap (if requested)
        if method == "BlockBootstrap":
            boot_alphas = []
            n = len(y)
            residuals = res.resid.values
            fitted = res.fittedvalues.values
            
            for _ in range(n_boot):
                # Circular block bootstrap
                starts = np.random.randint(0, n, int(np.ceil(n / block_len)))
                indices = []
                for s in starts:
                    indices.extend([ (s + i) % n for i in range(block_len) ])
                indices = indices[:n]
                
                # Resample residuals
                resid_boot = residuals[indices]
                y_boot = fitted + resid_boot
                
                # Re-estimate alpha
                # Fast OLS for alpha only: (X'X)^-1 X'y
                # Since X is constant, we can precompute (X'X)^-1 X'
                # But statsmodels is safer for now
                try:
                    # We just need the constant term
                    # Using simple mean of residuals + original alpha? 
                    # Better to run full regression
                    boot_model = OLS(y_boot, X)
                    boot_res = boot_model.fit()
                    
                    if 'const' in boot_res.params:
                        boot_alphas.append(boot_res.params['const'])
                    elif 'intercept' in boot_res.params:
                        boot_alphas.append(boot_res.params['intercept'])
                    else:
                        boot_alphas.append(boot_res.params[0])
                        
                except:
                    pass
            
            if len(boot_alphas) > 0:
                boot_alphas = np.array(boot_alphas)
                std_boot = np.std(boot_alphas)
                # t-stat based on bootstrap std error
                # Use the robustly identified alpha_val from above
                alpha_tstat = alpha_val / std_boot if std_boot > 0 else 0
                # p-value from distribution centered at 0
                # fraction of |boot_t| > |orig_t|? 
                # Or percentile method:
                # 2-sided p-value: 2 * min(mean(boot < 0), mean(boot > 0)) if we test H0: alpha=0
                # But we are bootstrapping residuals from the fitted model (H1).
                # To test H0, we should enforce alpha=0 in the bootstrap DGP.
                # Alternative: Percentile CI
                alpha_ci = [np.percentile(boot_alphas, 2.5) * 252, np.percentile(boot_alphas, 97.5) * 252]
                # P-value: Invert CI?
                # Simple approach: Bootstrap Standard Error
                alpha_pval = 2 * (1 - stats.norm.cdf(abs(alpha_tstat))) # Asymptotic normal using boot SE
        
        # 3. Deflated Sharpe / HLZ Adjustment
        # Approximate Deflated Sharpe P-value
        # Bailey, Lopez de Prado (2014)
        sharpe = y.mean() / y.std() * np.sqrt(252)
        skew = y.skew()
        kurt = y.kurtosis()
        
        # SR_0 (null hypothesis Sharpe) = 0
        # Variance of SR estimator
        var_sr = (1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2) / (len(y) / 252)
        # This formula is for annual SR? No, usually for period SR.
        # Let's use the standard HLZ approximation for t-stat deflation
        
        # Expected maximum t-stat among N strategies
        # E[max t] approx sqrt(2 * log(num_strategies))
        # This is a rough heuristic.
        
        # Deflated p-value
        # We adjust the t-stat threshold
        
        # Simple Bonferroni-like or HLZ logic:
        # We report the "Deflated Sharpe Probability"
        # For simplicity, we'll compute the "Prob(False Discovery)" using the DSR formula
        # DSR = CDF( (SR - E[maxSR]) / sigma_SR )
        
        # E[maxSR] approx sigma_SR * ((1-gamma)*Z^-1(1-1/N) + gamma*Z^-1(1-1/(N*e))) 
        # where gamma is Euler-Mascheroni constant.
        # This is complex to implement fully without a library.
        
        # Simplified HLZ:
        # t_adj = t_stat / sqrt(1 + log(num_strategies)) ? No.
        
        # Let's stick to a simple Bonferroni adjustment for the "Adj P-val" 
        # or a simplified deflation:
        # p_adj = 1 - (1 - p)^N
        adj_pval = 1 - (1 - alpha_pval)**num_strategies
        
        return {
            "alpha_annual": alpha_annual,
            "alpha_tstat": alpha_tstat,
            "alpha_pval": alpha_pval,
            "alpha_ci": alpha_ci,
            "adj_pval": adj_pval,
            "r_squared": res.rsquared,
            "n_obs": int(res.nobs),
            "method": method
        }
    
    def bootstrap_analysis(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, Any]:
        """Block bootstrap analysis"""
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
            n_blocks = int(np.ceil(n / block_size))
            block_starts = np.random.randint(0, max(1, n - block_size + 1), n_blocks)
            
            indices = []
            for start in block_starts:
                indices.extend(range(start, min(start + block_size, n)))
            indices = np.array(indices[:n])
            
            y_boot = y[indices]
            x_boot = x[indices]
            
            y_excess = y_boot - rf_daily
            x_excess = x_boot - rf_daily
            
            if len(x_excess) > 2 and np.std(x_excess) > 0:
                beta_boot, alpha_boot = np.polyfit(x_excess, y_excess, 1)
                bootstrap_alphas.append(alpha_boot * 252)
                
                excess_ret = y_boot - rf_daily
                sharpe_boot = np.mean(excess_ret) / np.std(excess_ret) * np.sqrt(252) if np.std(excess_ret) > 0 else 0
                bootstrap_sharpes.append(sharpe_boot)
        
        bootstrap_alphas = np.array(bootstrap_alphas)
        bootstrap_sharpes = np.array(bootstrap_sharpes)
        
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
    ) -> Dict[str, Any]:
        """Analyze strategy sensitivity to transaction costs"""
        if cost_range is None:
            cost_range = [0, 3, 5, 8, 10, 15, 20, 30, 50]
        
        rf_daily = self.config.risk_free_rate / 252
        results = []
        
        for cost_bps in cost_range:
            daily_cost = (cost_bps / 10000) * turnover / 21
            adj_returns = strategy_returns - daily_cost
            
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
        
        breakeven = None
        for i, r in enumerate(results):
            if r['sharpe_ratio'] <= 0:
                if i > 0:
                    prev = results[i-1]
                    slope = (r['sharpe_ratio'] - prev['sharpe_ratio']) / (r['cost_bps'] - prev['cost_bps'])
                    breakeven = prev['cost_bps'] - prev['sharpe_ratio'] / slope if slope != 0 else prev['cost_bps']
                break
        
        return {
            'results': results,
            'breakeven_cost_bps': breakeven if breakeven else float('inf'),
            'robust_at_50bps': results[-1]['sharpe_ratio'] > 0 if results else False
        }


# =============================================================================
# PERFORMANCE CALCULATOR
# =============================================================================

class PerformanceCalculator:
    """Calculate comprehensive performance metrics"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def calculate_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series = None
    ) -> Dict[str, Any]:
        rf_daily = self.config.risk_free_rate / 252
        
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / 252
        cagr = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
        
        volatility = returns.std() * np.sqrt(252)
        
        excess = returns - rf_daily
        sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
        
        cum = (1 + returns).cumprod()
        max_dd = (cum / cum.cummax() - 1).min()
        
        downside = returns[returns < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
        sortino = excess.mean() * 252 / downside_vol if downside_vol > 0 else 0
        
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'n_days': len(returns),
            'n_years': n_years
        }

    def apply_volatility_management(
        self,
        returns: pd.Series,
        target_vol_annual: float = 0.12,
        vol_window_days: int = 63,
        leverage_cap: float = 2.0
    ) -> pd.Series:
        """
        Apply volatility targeting (Barroso-Santa-Clara 2015).
        w_t = c / sigma_t
        """
        # Realized volatility (rolling std * sqrt(252))
        # Shift by 1 to use past data for current position
        realized_vol = returns.rolling(window=vol_window_days).std() * np.sqrt(252)
        realized_vol = realized_vol.shift(1)
        
        # Target weight
        # Avoid division by zero
        target_weight = target_vol_annual / realized_vol.replace(0, np.nan)
        
        # Cap leverage
        target_weight = target_weight.fillna(0).clip(0, leverage_cap)
        
        # Managed returns
        managed_returns = returns * target_weight
        
        return managed_returns

    def apply_crash_overlay(
        self,
        returns: pd.Series,
        market_vol_proxy: pd.Series,
        crash_quantile: float = 0.80,
        crash_scale: float = 0.0
    ) -> pd.Series:
        """
        Apply crash sensitivity overlay (Daniel-Moskowitz / Moreira-Muir).
        Scale down exposure when market vol is high.
        """
        # Determine threshold
        threshold = market_vol_proxy.quantile(crash_quantile)
        
        # Identify high vol regimes (using past data)
        # Shift by 1 to avoid lookahead bias
        is_high_vol = (market_vol_proxy.shift(1) >= threshold)
        
        # Apply scaling
        # If high vol, multiply by crash_scale (e.g. 0.5 or 0.0)
        # Else multiply by 1.0
        scale_factor = np.where(is_high_vol, crash_scale, 1.0)
        
        managed_returns = returns * scale_factor
        
        return managed_returns


# =============================================================================
# MAIN INTEGRATED MODEL
# =============================================================================

class EnhancedIntegratedModel:
    """
    Weakness Compensation Framework v3
    
    Integrates:
    - Layer 1: Dynamic Factor Loading (Kalman Filter)
    - Layer 2: Sentiment Integration (FinBERT)
    - Layer 3: GARCH-MIDAS Volatility
    - 4-Regime Classification (Duke feedback)
    """
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        
        self.layer1_factor = DynamicFactorLoading(self.config)
        self.layer2_sentiment = SentimentIntegration(self.config)
        self.layer3_volatility: Dict[str, GARCH_MIDAS] = {}
        
        self.regime_classifier = RegimeClassifier(self.config)
        
        self.validator = StatisticalValidator(self.config)
        self.performance = PerformanceCalculator(self.config)
        
        self.returns: Optional[pd.DataFrame] = None
        self.prices: Optional[pd.DataFrame] = None
        self.market_returns: Optional[pd.Series] = None
        
        self.dynamic_betas: Dict[str, np.ndarray] = {}
        self.volatilities: Dict[str, np.ndarray] = {}
        self.regimes: Optional[pd.DataFrame] = None
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
            
            data = pd.read_csv(prices_file, index_col=0, parse_dates=True)
            
            # Calculate returns
            ret_cols = [c for c in data.columns if c.endswith('_ret')]
            self.returns = data[ret_cols].copy()
            self.returns.columns = [c.replace('_ret', '') for c in ret_cols]
            
            # Filter out zero-variance columns (delisted stocks like ATVI)
            variances = self.returns.var()
            zero_var_cols = variances[variances < 1e-10].index.tolist()
            if zero_var_cols:
                print(f"⚠️ Dropping {len(zero_var_cols)} zero-variance assets: {zero_var_cols}")
                self.returns = self.returns.drop(columns=zero_var_cols)
            
            price_cols = [c for c in data.columns if c.endswith('_price')]
            self.prices = data[price_cols].copy()
            self.prices.columns = [c.replace('_price', '') for c in price_cols]
            
            # Keep only assets that exist in returns
            common_tickers = self.returns.columns.intersection(self.prices.columns)
            self.returns = self.returns[common_tickers]
            self.prices = self.prices[common_tickers]
            
            if trading_days and len(self.returns) > trading_days:
                self.returns = self.returns.iloc[-trading_days:]
                self.prices = self.prices.iloc[-trading_days:]
                print(f"   Filtered to last {trading_days} trading days")
            
            if benchmark in self.returns.columns:
                self.market_returns = self.returns[benchmark]
            elif 'SPY' in self.returns.columns:
                self.market_returns = self.returns['SPY']
            else:
                self.market_returns = self.returns.mean(axis=1)
            
            print(f"✅ Price data: {len(self.returns)} days, {len(self.returns.columns)} assets")
            print(f"   Date range: {self.returns.index[0].date()} to {self.returns.index[-1].date()}")
            print(f"   Benchmark: {benchmark}")
            
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
        
        print("\n[Layer 1] Dynamic Factor Loading (Kalman Filter)...")
        self.dynamic_betas = self.layer1_factor.fit(self.returns, self.market_returns)
        avg_beta = np.mean([b[-1] for b in self.dynamic_betas.values()])
        print(f"   ✅ Fitted {len(self.dynamic_betas)} assets, avg beta: {avg_beta:.3f}")
        
        print("\n[Layer 2] Sentiment Integration...")
        if self.layer2_sentiment.loaded:
            print(f"   ✅ Sentiment data active")
        else:
            print(f"   ⚠️ Sentiment data not loaded")
        
        print("\n[Layer 3] GARCH-MIDAS Volatility...")
        for ticker in self.returns.columns:
            garch = GARCH_MIDAS(self.config)
            self.volatilities[ticker] = garch.fit(self.returns[ticker].values)
            self.layer3_volatility[ticker] = garch
        print(f"   ✅ Fitted {len(self.volatilities)} volatility models")
        
        print("\n[Regime] 4-Regime Classification...")
        avg_vol = pd.Series({
            date: np.mean([self.volatilities[t][i] for t in self.returns.columns 
                          if i < len(self.volatilities[t])])
            for i, date in enumerate(self.returns.index)
        })
        self.regimes = self.regime_classifier.classify(avg_vol, self.market_returns)
        
        regime_counts = self.regimes['regime'].value_counts()
        print(f"   ✅ Regime distribution:")
        for regime, count in regime_counts.items():
            pct = count / len(self.regimes) * 100
            print(f"      {regime}: {count} days ({pct:.1f}%)")
        
        self.fitted = True
        print("\n✅ MODEL FITTING COMPLETE")
        
        return True
    
    def get_dynamic_beta_series(self, ticker: str) -> pd.Series:
        if ticker in self.dynamic_betas:
            return pd.Series(self.dynamic_betas[ticker], index=self.returns.index)
        return pd.Series()
    
    def get_volatility_series(self, ticker: str) -> pd.Series:
        if ticker in self.volatilities:
            return pd.Series(self.volatilities[ticker], index=self.returns.index)
        return pd.Series()
    
    def get_optimal_weights(self, method: str = 'black_litterman') -> pd.Series:
        """
        Get optimal portfolio weights.
        
        Methods:
        - 'black_litterman': BL model with sentiment views (recommended)
        - 'hrp': Hierarchical Risk Parity with sentiment tilt
        - 'herc': Hierarchical Equal Risk Contribution
        - 'network': Network/MST-based allocation
        - 'mean_variance': Traditional MVO
        - 'equal_weight': 1/N
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        tickers = list(self.returns.columns)
        n = len(tickers)
        date = self.returns.index[-1]
        
        # Get sentiment scores for current date
        sentiment_scores = pd.Series(index=tickers, dtype=float)
        for ticker in tickers:
            sentiment_scores[ticker] = self.layer2_sentiment.get_signal(date, ticker)
        
        print(f"\n📊 Sentiment Summary:")
        print(f"   Positive (>0.1): {(sentiment_scores > 0.1).sum()}")
        print(f"   Negative (<-0.1): {(sentiment_scores < -0.1).sum()}")
        print(f"   Mean: {sentiment_scores.mean():.3f}")
        
        # Black-Litterman with Sentiment Views (Primary Method)
        if method == 'black_litterman':
            try:
                from sentiment_bl_manager import BlackLittermanSentiment
                
                bl_model = BlackLittermanSentiment(
                    risk_aversion=self.config.risk_aversion,
                    sentiment_scale=0.15,  # 15% sentiment impact
                    view_confidence=0.6
                )
                
                # Market cap weights (equal weight as proxy)
                market_weights = pd.Series(1/n, index=tickers)
                
                weights = bl_model.fit(
                    self.returns,
                    sentiment_scores,
                    market_weights,
                    risk_free_rate=self.config.risk_free_rate
                )
                
                print(f"✅ Black-Litterman allocation applied")
                return weights
                
            except ImportError as e:
                print(f"⚠️ Black-Litterman not available: {e}")
                print("   Falling back to HRP with sentiment tilt")
                method = 'hrp'
            except Exception as e:
                print(f"⚠️ Black-Litterman failed: {e}")
                method = 'hrp'
        
        # Topology-based methods (HRP, HERC, Network) with sentiment tilt
        if method in ['hrp', 'herc', 'network']:
            try:
                from topology_allocation import TopologyAllocator, TopologyConfig
                
                config = TopologyConfig(method=method)
                allocator = TopologyAllocator(config)
                allocator.fit(self.returns)
                base_weights = allocator.get_weights()
                
                # Apply sentiment tilt (stronger than before: 20%)
                tilt = 1 + 0.20 * sentiment_scores.loc[base_weights.index]
                weights = base_weights * tilt
                weights = weights / weights.sum()
                
                print(f"✅ {method.upper()} + Sentiment tilt applied")
                return weights
                
            except ImportError:
                print(f"⚠️ topology_allocation not found, falling back to sentiment-tilted equal weight")
        
        # Sentiment-tilted Equal Weight (Fallback)
        if method == 'equal_weight' or method not in ['mean_variance']:
            base = pd.Series(1.0/n, index=tickers)
            tilt = 1 + 0.15 * sentiment_scores
            weights = base * tilt
            weights = weights / weights.sum()
            return weights
        
        # Mean-Variance with sentiment-adjusted returns
        if method == 'mean_variance' and SCIPY_AVAILABLE:
            base_returns = self.returns.mean()
            expected_returns = self.layer2_sentiment.adjust_expected_returns(base_returns, date)
            
            cov_matrix = self.returns.cov()
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
            return weights / weights.sum()
        
        # Final fallback
        return pd.Series(1.0/n, index=tickers)
    
    def run_backtest(self, weights: pd.Series = None, rebalance: str = 'monthly') -> pd.Series:
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        if weights is None:
            weights = self.get_optimal_weights()
        
        common_tickers = weights.index.intersection(self.returns.columns)
        weights = weights[common_tickers]
        weights = weights / weights.sum()
        
        returns = self.returns[common_tickers]
        portfolio_returns = (returns * weights).sum(axis=1)
        
        tc_daily = self.config.transaction_cost_bps / 10000 / 21
        portfolio_returns = portfolio_returns - tc_daily
        
        return portfolio_returns
    
    def validate(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        run_bootstrap: bool = True,
        n_subperiods: int = 4
    ) -> Dict[str, Any]:
        """Run complete validation suite"""
        results = {}
        
        results['alpha_beta'] = self.validator.calculate_alpha_beta(
            strategy_returns, benchmark_returns
        )
        
        if run_bootstrap:
            results['bootstrap'] = self.validator.bootstrap_analysis(
                strategy_returns, benchmark_returns
            )
        
        results['subperiods'] = self.validator.subperiod_analysis(
            strategy_returns, benchmark_returns, n_subperiods
        )
        
        results['transaction_costs'] = self.validator.transaction_cost_sensitivity(
            strategy_returns
        )
        
        results['performance'] = self.performance.calculate_metrics(
            strategy_returns, benchmark_returns
        )
        
        return results
    
    def summary(self) -> Dict[str, Any]:
        return {
            'n_assets': len(self.returns.columns) if self.returns is not None else 0,
            'n_days': len(self.returns) if self.returns is not None else 0,
            'avg_beta': np.mean([b[-1] for b in self.dynamic_betas.values()]) if self.dynamic_betas else 0,
            'sentiment_loaded': self.layer2_sentiment.loaded,
            'fitted': self.fitted
        }


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'ModelConfig',
    'MarketRegime',
    'EnhancedIntegratedModel',
    'StatisticalValidator',
    'PerformanceCalculator',
    'KalmanFilterBeta',
    'DynamicFactorLoading',
    'SentimentIntegration',
    'GARCH_MIDAS',
    'RegimeClassifier',
]
