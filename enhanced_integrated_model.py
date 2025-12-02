"""
Enhanced Integrated Model: Weakness Compensation Framework
===========================================================

This module addresses the key weaknesses identified in three high-impact papers:
1. FOGI (Kim et al., 2024) - Factor Overnight GARCH-Itô Models
2. FinBERT Sentiment (Kirtac & Germano, 2024) - Sentiment Trading with LLMs
3. GARCH-MIDAS + MCS (Conrad & Kleen, 2020) - Model Confidence Set

Key Compensations:
------------------
- Static Factor Loading → Dynamic Kalman Filter Loading
- Static Sentiment → AR Sentiment Model with Volatility Pathway
- Linear Long-run Volatility → Neural Network Nonlinear τ_t
- Single Variable → LASSO Multi-Variable Selection
- Post-hoc MCS → Rolling Adaptive MCS
- Overall Alpha Test → Regime-Conditional Alpha Testing

Author: Multi-Agent Portfolio System
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import warnings
from scipy import stats
from scipy.optimize import minimize
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION & DATA STRUCTURES
# =============================================================================

class MarketRegime(Enum):
    """Market regime classification"""
    BULL = 0
    TRANSITION = 1
    BEAR = 2


@dataclass
class ModelConfig:
    """Configuration for the Enhanced Integrated Model"""
    # Asset configuration
    n_assets: int = 10
    n_factors: int = 3
    
    # Dynamic Factor Loading
    loading_volatility: float = 0.01
    observation_noise: float = 0.1
    
    # Sentiment Model
    sentiment_ar_order: int = 2
    sentiment_decay: float = 0.9
    
    # Nonlinear GARCH-MIDAS
    n_predictors: int = 20
    hidden_units: List[int] = field(default_factory=lambda: [32, 16])
    lasso_alpha: float = 0.01
    
    # MCS Configuration
    mcs_window: int = 252
    mcs_update_frequency: int = 22
    mcs_confidence_level: float = 0.90
    mcs_bootstrap_replications: int = 8000
    
    # Validation
    min_observations_per_regime: int = 30
    newey_west_lags: int = 5
    bootstrap_replications: int = 10000


@dataclass
class SentimentSignal:
    """Sentiment analysis output"""
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    components: Dict[str, float] = field(default_factory=dict)
    timestamp: Optional[pd.Timestamp] = None


@dataclass
class VolatilityPrediction:
    """Volatility prediction output"""
    point_estimate: float
    confidence_interval: Tuple[float, float]
    components: Dict[str, float] = field(default_factory=dict)
    regime: Optional[MarketRegime] = None


@dataclass
class AlphaTestResult:
    """Alpha significance test result"""
    alpha_daily: float
    alpha_annual: float
    p_value: float
    is_significant: bool
    confidence_interval: Tuple[float, float]
    beta: float
    r_squared: float
    n_observations: int


# =============================================================================
# LAYER 1: DYNAMIC FACTOR LOADING (FOGI Weakness Compensation)
# =============================================================================

class DynamicFactorLoading:
    """
    Time-varying factor loading via Kalman Filter
    
    Compensates FOGI Weakness:
    - Static factor loading assumption
    - High-frequency data dependency
    - No regime conditioning
    
    Mathematical Framework:
    ----------------------
    State equation: Λ_t = Λ_{t-1} + η_t,  η_t ~ N(0, Q)
    Observation equation: r_t = Λ_t × f_t + ε_t,  ε_t ~ N(0, R)
    
    References:
    -----------
    Kim, Oh, Song & Wang (2024) - Factor Overnight GARCH-Itô Models
    """
    
    def __init__(self, n_assets: int, n_factors: int, config: Optional[ModelConfig] = None):
        self.n_assets = n_assets
        self.n_factors = n_factors
        self.config = config or ModelConfig()
        
        # State dimension: vectorized factor loading (p × r)
        self.state_dim = n_assets * n_factors
        
        # Kalman Filter matrices
        self.transition_matrix: Optional[np.ndarray] = None
        self.transition_cov: Optional[np.ndarray] = None
        self.observation_cov: Optional[np.ndarray] = None
        
        # Current state
        self.current_state: Optional[np.ndarray] = None
        self.current_cov: Optional[np.ndarray] = None
        
        # History for analysis
        self.loading_history: List[np.ndarray] = []
        
        # Regime-specific adjustments
        self.regime_adjustments = self._initialize_regime_adjustments()
    
    def _initialize_regime_adjustments(self) -> Dict[MarketRegime, np.ndarray]:
        """Initialize regime-specific factor loading adjustments"""
        base = np.ones((self.n_assets, self.n_factors))
        
        return {
            MarketRegime.BULL: base * np.array([1.1, 1.0, 0.9]),  # Market factor up
            MarketRegime.TRANSITION: base * np.array([1.0, 1.0, 1.0]),  # Neutral
            MarketRegime.BEAR: base * np.array([0.9, 1.0, 1.1])  # Defensive factor up
        }
    
    def initialize(
        self,
        initial_loading: np.ndarray,
        loading_volatility: Optional[float] = None,
        observation_noise: Optional[float] = None
    ) -> 'DynamicFactorLoading':
        """
        Initialize Kalman Filter with prior estimates
        
        Parameters:
        -----------
        initial_loading : np.ndarray
            Initial factor loading matrix (n_assets × n_factors)
        loading_volatility : float
            Innovation variance for factor loading random walk
        observation_noise : float
            Observation noise variance
        """
        loading_vol = loading_volatility or self.config.loading_volatility
        obs_noise = observation_noise or self.config.observation_noise
        
        # Transition: random walk for factor loadings
        self.transition_matrix = np.eye(self.state_dim)
        self.transition_cov = np.eye(self.state_dim) * loading_vol
        
        # Initial state
        self.current_state = initial_loading.flatten()
        self.current_cov = np.eye(self.state_dim) * 0.1
        
        # Observation noise
        self.observation_cov = np.eye(self.n_assets) * obs_noise
        
        logger.info(f"Initialized DynamicFactorLoading: {self.n_assets} assets, {self.n_factors} factors")
        
        return self
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman Filter prediction step
        
        Returns:
        --------
        state_pred : np.ndarray
            Predicted state (vectorized loading)
        cov_pred : np.ndarray
            Predicted covariance
        """
        if self.current_state is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        # Random walk prediction
        state_pred = self.transition_matrix @ self.current_state
        cov_pred = self.transition_matrix @ self.current_cov @ self.transition_matrix.T + self.transition_cov
        
        return state_pred, cov_pred
    
    def update(
        self,
        returns: np.ndarray,
        factor_returns: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman Filter update step
        
        Parameters:
        -----------
        returns : np.ndarray
            Asset returns (n_assets,)
        factor_returns : np.ndarray
            Factor returns (n_factors,)
            
        Returns:
        --------
        loading_updated : np.ndarray
            Updated factor loading matrix (n_assets × n_factors)
        loading_uncertainty : np.ndarray
            Diagonal of covariance (uncertainty per element)
        """
        # Predict step
        state_pred, cov_pred = self.predict()
        
        # Observation matrix: H_t = I_p ⊗ f_t^T
        # r_t = H_t × vec(Λ_t)
        H_t = np.kron(np.eye(self.n_assets), factor_returns.reshape(1, -1))
        
        # Innovation
        innovation = returns - H_t @ state_pred
        
        # Innovation covariance
        S = H_t @ cov_pred @ H_t.T + self.observation_cov
        
        # Kalman gain
        try:
            K = cov_pred @ H_t.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Regularization if singular
            K = cov_pred @ H_t.T @ np.linalg.pinv(S + 1e-6 * np.eye(S.shape[0]))
        
        # Update state
        state_updated = state_pred + K @ innovation
        
        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(self.state_dim) - K @ H_t
        cov_updated = I_KH @ cov_pred @ I_KH.T + K @ self.observation_cov @ K.T
        
        # Store updated state
        self.current_state = state_updated
        self.current_cov = cov_updated
        
        # Reshape to matrix
        loading_updated = state_updated.reshape(self.n_assets, self.n_factors)
        loading_uncertainty = np.sqrt(np.diag(cov_updated)).reshape(self.n_assets, self.n_factors)
        
        # Store history
        self.loading_history.append(loading_updated.copy())
        
        return loading_updated, loading_uncertainty
    
    def apply_regime_conditioning(
        self,
        base_loading: np.ndarray,
        regime: MarketRegime
    ) -> np.ndarray:
        """
        Apply regime-conditional adjustment to factor loading
        
        Λ_t^{regime} = Λ_t ⊙ adjustment_{regime}
        
        Parameters:
        -----------
        base_loading : np.ndarray
            Base factor loading matrix
        regime : MarketRegime
            Current market regime
            
        Returns:
        --------
        adjusted_loading : np.ndarray
            Regime-adjusted factor loading
        """
        adjustment = self.regime_adjustments.get(regime, np.ones_like(base_loading))
        return base_loading * adjustment
    
    def get_current_loading(self, regime: Optional[MarketRegime] = None) -> np.ndarray:
        """Get current factor loading, optionally regime-adjusted"""
        if self.current_state is None:
            raise ValueError("Model not initialized.")
        
        loading = self.current_state.reshape(self.n_assets, self.n_factors)
        
        if regime is not None:
            loading = self.apply_regime_conditioning(loading, regime)
        
        return loading
    
    def estimate_from_returns(
        self,
        returns: np.ndarray,
        n_factors: Optional[int] = None
    ) -> np.ndarray:
        """
        Estimate initial factor loading from historical returns using PCA
        
        Parameters:
        -----------
        returns : np.ndarray
            Historical returns (T × n_assets)
        n_factors : int
            Number of factors (default: self.n_factors)
            
        Returns:
        --------
        loading : np.ndarray
            Estimated factor loading matrix
        """
        n_factors = n_factors or self.n_factors
        
        # Standardize returns
        returns_std = (returns - returns.mean(axis=0)) / returns.std(axis=0)
        
        # PCA via SVD
        U, S, Vt = np.linalg.svd(returns_std, full_matrices=False)
        
        # Factor loading: first n_factors principal components
        loading = Vt[:n_factors].T * S[:n_factors]
        
        # Normalize
        loading = loading / np.sqrt((loading ** 2).sum(axis=0))
        
        return loading


# =============================================================================
# LAYER 2: DYNAMIC SENTIMENT INTEGRATION (FinBERT Weakness Compensation)
# =============================================================================

class DynamicSentimentModel:
    """
    Dynamic Sentiment Integration with AR Structure and Volatility Pathway
    
    Compensates FinBERT Weakness:
    - Static sentiment score → AR(p) dynamic model
    - Returns-only focus → Sentiment-to-Volatility pathway
    - Single frequency → Multi-frequency aggregation
    - No context → Regime-conditional impact
    
    Mathematical Framework:
    ----------------------
    Sentiment dynamics: S_t = μ + Σ φ_i × S_{t-i} + ε_t
    Volatility impact: σ²_adj = σ² × g(S_t, ΔS_t, σ_S, regime)
    
    References:
    -----------
    Kirtac & Germano (2024) - Sentiment Trading with Large Language Models
    """
    
    def __init__(self, ar_order: int = 2, config: Optional[ModelConfig] = None):
        self.ar_order = ar_order
        self.config = config or ModelConfig()
        
        # AR model parameters
        self.ar_params: Optional[Dict[str, Any]] = None
        
        # Sentiment history
        self.sentiment_history: List[float] = []
        self.sentiment_vol_history: List[float] = []
        
        # Volatility impact parameters (asymmetric)
        self.alpha_neg = 0.3  # Negative sentiment impact on volatility
        self.alpha_pos = 0.1  # Positive sentiment impact on volatility
        self.kappa = 0.15  # Sentiment volatility impact
        
        # Regime multipliers
        self.regime_multipliers = {
            MarketRegime.BULL: 0.8,      # Sentiment impact reduced in bull
            MarketRegime.TRANSITION: 1.0,
            MarketRegime.BEAR: 1.5       # Sentiment impact amplified in bear
        }
    
    def fit_ar_model(self, sentiment_history: np.ndarray) -> 'DynamicSentimentModel':
        """
        Fit AR(p) model to sentiment time series
        
        Parameters:
        -----------
        sentiment_history : np.ndarray
            Historical sentiment scores
            
        Returns:
        --------
        self : DynamicSentimentModel
            Fitted model
        """
        if len(sentiment_history) < self.ar_order + 10:
            warnings.warn("Insufficient data for AR fitting. Using default parameters.")
            self.ar_params = {
                'const': 0.0,
                'ar_coefs': np.ones(self.ar_order) * 0.3 / self.ar_order,
                'sigma': 0.1
            }
            return self
        
        # Simple OLS estimation for AR(p)
        T = len(sentiment_history)
        Y = sentiment_history[self.ar_order:]
        X = np.column_stack([
            np.ones(T - self.ar_order),
            *[sentiment_history[self.ar_order - i - 1:-i-1 if i > 0 else T-i-1] 
              for i in range(self.ar_order)]
        ])
        
        try:
            # OLS: β = (X'X)^{-1} X'Y
            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
            residuals = Y - X @ beta
            sigma = np.std(residuals)
            
            self.ar_params = {
                'const': beta[0],
                'ar_coefs': beta[1:],
                'sigma': sigma
            }
        except np.linalg.LinAlgError:
            logger.warning("AR fitting failed. Using default parameters.")
            self.ar_params = {
                'const': 0.0,
                'ar_coefs': np.ones(self.ar_order) * 0.3 / self.ar_order,
                'sigma': 0.1
            }
        
        logger.info(f"Fitted AR({self.ar_order}) model: φ = {self.ar_params['ar_coefs']}")
        
        return self
    
    def predict_sentiment(
        self,
        recent_sentiment: np.ndarray,
        horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict future sentiment and uncertainty
        
        Parameters:
        -----------
        recent_sentiment : np.ndarray
            Last ar_order sentiment observations
        horizon : int
            Prediction horizon
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted sentiment values
        uncertainties : np.ndarray
            Prediction uncertainties (growing with horizon)
        """
        if self.ar_params is None:
            raise ValueError("Model not fitted. Call fit_ar_model() first.")
        
        predictions = []
        uncertainties = []
        
        current = recent_sentiment.copy() if len(recent_sentiment) >= self.ar_order else \
                  np.pad(recent_sentiment, (self.ar_order - len(recent_sentiment), 0))
        
        cumulative_var = 0
        
        for h in range(horizon):
            # AR prediction
            pred = self.ar_params['const'] + np.dot(
                self.ar_params['ar_coefs'],
                current[-self.ar_order:][::-1]
            )
            
            # Clip to valid range
            pred = np.clip(pred, -1, 1)
            predictions.append(pred)
            
            # Uncertainty grows with horizon
            cumulative_var += self.ar_params['sigma'] ** 2
            uncertainties.append(np.sqrt(cumulative_var))
            
            # Update for next step
            current = np.append(current, pred)
        
        return np.array(predictions), np.array(uncertainties)
    
    def multi_frequency_aggregation(
        self,
        intraday_sentiments: List[float],
        weights: str = 'recency',
        timestamps: Optional[List[pd.Timestamp]] = None
    ) -> Tuple[float, float]:
        """
        Aggregate multi-frequency sentiment signals
        
        Parameters:
        -----------
        intraday_sentiments : List[float]
            Intraday sentiment scores from news
        weights : str
            Weighting scheme: 'recency', 'equal', 'volume'
        timestamps : List[pd.Timestamp]
            Optional timestamps for time-weighted aggregation
            
        Returns:
        --------
        agg_sentiment : float
            Aggregated sentiment score
        sentiment_vol : float
            Sentiment volatility (dispersion)
        """
        if not intraday_sentiments:
            return 0.0, 0.0
        
        sentiments = np.array(intraday_sentiments)
        n = len(sentiments)
        
        if weights == 'recency':
            # Exponential recency weights
            decay = self.config.sentiment_decay
            w = np.array([decay ** (n - 1 - i) for i in range(n)])
        elif weights == 'equal':
            w = np.ones(n)
        elif weights == 'time' and timestamps is not None:
            # Time-based weights (more recent = higher weight)
            times = pd.to_datetime(timestamps)
            time_diffs = (times.max() - times).total_seconds()
            w = np.exp(-time_diffs / 3600)  # 1-hour decay
        else:
            w = np.ones(n)
        
        w = w / w.sum()  # Normalize
        
        # Weighted mean
        agg_sentiment = np.dot(w, sentiments)
        
        # Weighted volatility
        sentiment_vol = np.sqrt(np.dot(w, (sentiments - agg_sentiment) ** 2))
        
        return agg_sentiment, sentiment_vol
    
    def sentiment_to_volatility_impact(
        self,
        sentiment: float,
        sentiment_change: float,
        sentiment_uncertainty: float,
        regime: MarketRegime
    ) -> Dict[str, float]:
        """
        Compute sentiment impact on volatility
        
        σ²_adj = σ² × (1 + level_impact + momentum_impact + uncertainty_impact) × regime_mult
        
        Parameters:
        -----------
        sentiment : float
            Current aggregate sentiment (-1 to 1)
        sentiment_change : float
            Sentiment change from previous period
        sentiment_uncertainty : float
            Sentiment volatility/uncertainty
        regime : MarketRegime
            Current market regime
            
        Returns:
        --------
        impact_dict : Dict[str, float]
            Volatility adjustment components
        """
        # Level impact (asymmetric)
        # Negative sentiment → volatility increase
        # Positive sentiment → slight volatility decrease
        threshold = 0.0
        if sentiment < threshold:
            level_impact = -sentiment * self.alpha_neg
        else:
            level_impact = -sentiment * self.alpha_pos
        
        # Momentum impact
        # Large sentiment changes → volatility spike
        momentum_impact = abs(sentiment_change) * 0.2
        
        # Uncertainty impact
        # High sentiment uncertainty → higher volatility
        uncertainty_impact = sentiment_uncertainty * self.kappa
        
        # Regime conditioning
        regime_mult = self.regime_multipliers.get(regime, 1.0)
        
        # Total adjustment
        total_impact = (level_impact + momentum_impact + uncertainty_impact) * regime_mult
        
        return {
            'total_vol_adjustment': 1 + total_impact,
            'level_component': level_impact,
            'momentum_component': momentum_impact,
            'uncertainty_component': uncertainty_impact,
            'regime_multiplier': regime_mult
        }
    
    def update(self, sentiment: float, sentiment_vol: float = 0.0) -> None:
        """Update sentiment history"""
        self.sentiment_history.append(sentiment)
        self.sentiment_vol_history.append(sentiment_vol)
    
    def get_sentiment_momentum(self, lookback: int = 5) -> float:
        """Calculate sentiment momentum"""
        if len(self.sentiment_history) < 2:
            return 0.0
        
        recent = self.sentiment_history[-lookback:]
        if len(recent) < 2:
            return recent[-1] - recent[0] if len(recent) == 2 else 0.0
        
        # Linear regression slope
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        
        return slope


class FinBERTSentimentAnalyzer:
    """
    FinBERT-based sentiment analysis wrapper
    
    Note: Actual FinBERT model loading requires transformers library.
    This is a structured wrapper that can be extended.
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
    
    def load_model(self) -> 'FinBERTSentimentAnalyzer':
        """Load FinBERT model (requires transformers)"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._is_loaded = True
            logger.info(f"Loaded FinBERT model: {self.model_name}")
        except ImportError:
            logger.warning("transformers not available. Using mock sentiment.")
            self._is_loaded = False
        except Exception as e:
            logger.warning(f"Failed to load FinBERT: {e}. Using mock sentiment.")
            self._is_loaded = False
        
        return self
    
    def analyze(self, text: str) -> SentimentSignal:
        """
        Analyze sentiment of financial text
        
        Parameters:
        -----------
        text : str
            Financial news headline or text
            
        Returns:
        --------
        signal : SentimentSignal
            Sentiment analysis result
        """
        if not self._is_loaded:
            # Mock sentiment for testing
            return self._mock_sentiment(text)
        
        import torch
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
        
        # FinBERT: [positive, negative, neutral]
        pos, neg, neu = probs
        
        # Score: positive - negative (range: -1 to 1)
        score = pos - neg
        
        # Confidence: max probability
        confidence = max(probs)
        
        return SentimentSignal(
            score=score,
            confidence=confidence,
            components={'positive': pos, 'negative': neg, 'neutral': neu},
            timestamp=pd.Timestamp.now()
        )
    
    def _mock_sentiment(self, text: str) -> SentimentSignal:
        """Mock sentiment for testing without model"""
        # Simple keyword-based mock
        positive_words = ['up', 'gain', 'rise', 'profit', 'growth', 'strong', 'beat']
        negative_words = ['down', 'loss', 'fall', 'decline', 'weak', 'miss', 'crash']
        
        text_lower = text.lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        
        total = pos_count + neg_count + 1
        score = (pos_count - neg_count) / total
        
        return SentimentSignal(
            score=score,
            confidence=0.5,
            components={'positive': pos_count/total, 'negative': neg_count/total},
            timestamp=pd.Timestamp.now()
        )
    
    def batch_analyze(self, texts: List[str]) -> List[SentimentSignal]:
        """Analyze multiple texts"""
        return [self.analyze(text) for text in texts]


# =============================================================================
# LAYER 3: NONLINEAR LONG-RUN VOLATILITY (GARCH-MIDAS Weakness Compensation)
# =============================================================================

class NonlinearGARCHMIDAS:
    """
    Nonlinear GARCH-MIDAS with Neural Network Long-run Component
    
    Compensates GARCH-MIDAS Weakness:
    - Linear τ_t → Neural network nonlinear τ_t
    - Single variable → LASSO multi-variable selection
    - Fixed model → Adaptive model selection
    
    Mathematical Framework:
    ----------------------
    r_t = √(g_t × τ_t) × ε_t
    
    g_t = (1 - α - β) + α × (r²_{t-1} / τ_{t-1}) + β × g_{t-1}  [Short-term GARCH]
    τ_t = NN(X_t; θ)  [Long-term Neural Network]
    
    References:
    -----------
    Conrad & Kleen (2020) - Two Are Better Than One: GARCH-MIDAS
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        
        # LASSO selected variables
        self.selected_variables: List[str] = []
        self.lasso_coefs: Dict[str, float] = {}
        
        # Neural network for τ_t
        self.neural_tau = None
        self.scaler = None
        
        # Short-term GARCH parameters
        self.omega = 1e-5
        self.alpha = 0.1
        self.beta = 0.85
        
        # Fitted flag
        self._is_fitted = False
    
    def variable_selection_lasso(
        self,
        target: np.ndarray,
        predictors: pd.DataFrame,
        alpha: Optional[float] = None
    ) -> List[str]:
        """
        LASSO variable selection for long-run volatility predictors
        
        Parameters:
        -----------
        target : np.ndarray
            Log long-run volatility (target)
        predictors : pd.DataFrame
            Candidate predictor variables
        alpha : float
            LASSO regularization parameter
            
        Returns:
        --------
        selected_vars : List[str]
            Selected variable names
        """
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import StandardScaler
        
        # Align and clean data
        valid_idx = ~(np.isnan(target) | np.isinf(target))
        target_clean = target[valid_idx]
        predictors_clean = predictors.iloc[valid_idx].dropna(axis=1)
        
        if len(target_clean) < 50:
            logger.warning("Insufficient data for LASSO. Using all variables.")
            self.selected_variables = predictors_clean.columns.tolist()
            return self.selected_variables
        
        # Standardize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(predictors_clean)
        
        # LASSO with cross-validation
        alphas = alpha if alpha else np.logspace(-4, 0, 50)
        lasso = LassoCV(cv=5, alphas=alphas if isinstance(alphas, np.ndarray) else [alphas])
        
        try:
            lasso.fit(X_scaled, target_clean)
            
            # Select non-zero coefficients
            selected_mask = np.abs(lasso.coef_) > 1e-6
            self.selected_variables = predictors_clean.columns[selected_mask].tolist()
            self.lasso_coefs = dict(zip(
                predictors_clean.columns[selected_mask],
                lasso.coef_[selected_mask]
            ))
            
            logger.info(f"LASSO selected {len(self.selected_variables)} variables: {self.selected_variables}")
            
        except Exception as e:
            logger.warning(f"LASSO failed: {e}. Using all variables.")
            self.selected_variables = predictors_clean.columns.tolist()
        
        return self.selected_variables
    
    def build_neural_tau(self, input_dim: int) -> None:
        """
        Build neural network for nonlinear τ_t
        
        Architecture:
        - Input: Selected macro/sentiment predictors
        - Hidden: 2 layers with ReLU, BatchNorm, Dropout
        - Output: log(τ_t) for positivity
        """
        try:
            import torch
            import torch.nn as nn
            
            class NeuralTau(nn.Module):
                def __init__(self, input_dim, hidden_units):
                    super().__init__()
                    
                    layers = []
                    prev_dim = input_dim
                    
                    for hidden_dim in hidden_units:
                        layers.append(nn.Linear(prev_dim, hidden_dim))
                        layers.append(nn.ReLU())
                        layers.append(nn.BatchNorm1d(hidden_dim))
                        layers.append(nn.Dropout(0.2))
                        prev_dim = hidden_dim
                    
                    # Output: log(τ_t)
                    layers.append(nn.Linear(prev_dim, 1))
                    
                    self.network = nn.Sequential(*layers)
                
                def forward(self, x):
                    log_tau = self.network(x)
                    tau = torch.exp(torch.clamp(log_tau, -5, 5))  # Bounded
                    return tau
            
            self.neural_tau = NeuralTau(input_dim, self.config.hidden_units)
            logger.info(f"Built neural tau network: {input_dim} → {self.config.hidden_units} → 1")
            
        except ImportError:
            logger.warning("PyTorch not available. Using linear fallback.")
            self.neural_tau = None
    
    def fit(
        self,
        returns: np.ndarray,
        predictors: pd.DataFrame,
        n_epochs: int = 100,
        lr: float = 0.001
    ) -> 'NonlinearGARCHMIDAS':
        """
        Fit joint GARCH + Neural τ_t model
        
        Parameters:
        -----------
        returns : np.ndarray
            Asset returns
        predictors : pd.DataFrame
            Macro/sentiment predictor variables
        n_epochs : int
            Training epochs
        lr : float
            Learning rate
        """
        # 1. Compute realized volatility as target
        realized_vol = pd.Series(returns).rolling(22).std() * np.sqrt(252)
        valid_idx = ~realized_vol.isna()
        
        # 2. Variable selection
        self.variable_selection_lasso(
            np.log(realized_vol[valid_idx].values),
            predictors.loc[valid_idx]
        )
        
        if not self.selected_variables:
            logger.warning("No variables selected. Using simple GARCH.")
            self._is_fitted = True
            return self
        
        # 3. Build neural network
        self.build_neural_tau(len(self.selected_variables))
        
        if self.neural_tau is None:
            # Fallback to linear model
            self._fit_linear(returns, predictors)
            return self
        
        # 4. PyTorch training
        import torch
        import torch.optim as optim
        
        selected_data = predictors[self.selected_variables].loc[valid_idx]
        X = torch.FloatTensor(self.scaler.transform(selected_data))
        y = torch.FloatTensor(returns[valid_idx])
        
        optimizer = optim.Adam(self.neural_tau.parameters(), lr=lr)
        
        self.neural_tau.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Neural tau
            tau = self.neural_tau(X).squeeze()
            
            # Short-term GARCH component
            g = torch.ones_like(y)
            for t in range(1, len(y)):
                g[t] = (1 - self.alpha - self.beta) + \
                       self.alpha * (y[t-1] ** 2 / (tau[t-1] + 1e-8)) + \
                       self.beta * g[t-1]
            
            # Total variance
            sigma2 = g * tau
            
            # Gaussian log-likelihood
            ll = -0.5 * (torch.log(sigma2 + 1e-8) + y ** 2 / (sigma2 + 1e-8))
            loss = -ll.mean()
            
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        self.neural_tau.eval()
        self._is_fitted = True
        
        logger.info("Fitted NonlinearGARCHMIDAS model")
        
        return self
    
    def _fit_linear(self, returns: np.ndarray, predictors: pd.DataFrame) -> None:
        """Fallback linear fitting when PyTorch unavailable"""
        from sklearn.linear_model import Ridge
        
        realized_vol = pd.Series(returns).rolling(22).std() * np.sqrt(252)
        valid_idx = ~realized_vol.isna()
        
        X = predictors[self.selected_variables].loc[valid_idx]
        y = np.log(realized_vol[valid_idx])
        
        self.linear_model = Ridge(alpha=1.0)
        self.linear_model.fit(self.scaler.transform(X), y)
        
        self._is_fitted = True
    
    def predict_long_run_vol(self, predictors: pd.DataFrame) -> np.ndarray:
        """
        Predict long-run volatility component
        
        Parameters:
        -----------
        predictors : pd.DataFrame
            Predictor values
            
        Returns:
        --------
        tau : np.ndarray
            Predicted long-run volatility
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if not self.selected_variables:
            # No predictors, return constant
            return np.ones(len(predictors)) * 0.15  # 15% baseline
        
        X = predictors[self.selected_variables]
        X_scaled = self.scaler.transform(X)
        
        if self.neural_tau is not None:
            import torch
            with torch.no_grad():
                tau = self.neural_tau(torch.FloatTensor(X_scaled)).numpy().squeeze()
        else:
            log_tau = self.linear_model.predict(X_scaled)
            tau = np.exp(log_tau)
        
        return tau
    
    def predict_total_volatility(
        self,
        returns: np.ndarray,
        predictors: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict total volatility (short-term × long-term)
        
        Parameters:
        -----------
        returns : np.ndarray
            Recent returns for GARCH component
        predictors : pd.DataFrame
            Predictors for long-term component
            
        Returns:
        --------
        sigma : np.ndarray
            Predicted total volatility
        """
        # Long-term component
        tau = self.predict_long_run_vol(predictors)
        
        # Short-term GARCH
        g = np.ones_like(returns)
        for t in range(1, len(returns)):
            g[t] = (1 - self.alpha - self.beta) + \
                   self.alpha * (returns[t-1] ** 2 / (tau[t-1] + 1e-8)) + \
                   self.beta * g[t-1]
        
        # Total volatility
        sigma = np.sqrt(g * tau)
        
        return sigma


# =============================================================================
# LAYER 4: ADAPTIVE MODEL CONFIDENCE SET (MCS Weakness Compensation)
# =============================================================================

class AdaptiveMCSSelector:
    """
    Rolling MCS for Adaptive Model Selection
    
    Compensates MCS Weakness:
    - Post-hoc validation → Real-time model selection
    - Static model set → Dynamic adaptation
    - Single loss → Multiple loss functions
    
    Mathematical Framework:
    ----------------------
    MCS at confidence level α: Set of models with equivalent predictive ability
    T_max test statistic with bootstrap distribution
    
    References:
    -----------
    Hansen, Lunde & Nason (2011) - The Model Confidence Set
    """
    
    def __init__(
        self,
        models: Dict[str, Callable],
        config: Optional[ModelConfig] = None
    ):
        self.models = models
        self.config = config or ModelConfig()
        
        # Current MCS
        self.current_mcs: List[str] = list(models.keys())
        
        # History
        self.mcs_history: List[Dict] = []
        self.loss_history: Dict[str, List[float]] = {m: [] for m in models.keys()}
        
        # Last update
        self.last_update_idx = 0
    
    def compute_losses(
        self,
        predictions: Dict[str, np.ndarray],
        actuals: np.ndarray,
        loss_type: str = 'QLIKE'
    ) -> Dict[str, np.ndarray]:
        """
        Compute loss for each model
        
        Parameters:
        -----------
        predictions : Dict[str, np.ndarray]
            Model predictions (variance forecasts)
        actuals : np.ndarray
            Actual squared returns (realized variance proxy)
        loss_type : str
            Loss function: 'MSE', 'MAE', 'QLIKE'
        """
        losses = {}
        
        for model_name, pred in predictions.items():
            pred = np.maximum(pred, 1e-8)  # Ensure positivity
            
            if loss_type == 'MSE':
                losses[model_name] = (actuals - pred) ** 2
            elif loss_type == 'MAE':
                losses[model_name] = np.abs(actuals - pred)
            elif loss_type == 'QLIKE':
                # QLIKE: robust to fat tails
                losses[model_name] = np.log(pred) + actuals / pred
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
        
        return losses
    
    def mcs_test(
        self,
        losses: Dict[str, np.ndarray],
        alpha: float = 0.10,
        n_bootstrap: int = 1000
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Model Confidence Set test
        
        Parameters:
        -----------
        losses : Dict[str, np.ndarray]
            Model losses
        alpha : float
            Significance level (1 - confidence)
        n_bootstrap : int
            Bootstrap replications
            
        Returns:
        --------
        included : List[str]
            Models in MCS
        p_values : Dict[str, float]
            P-values for each model
        """
        model_names = list(losses.keys())
        n_models = len(model_names)
        T = len(list(losses.values())[0])
        
        if n_models <= 1:
            return model_names, {m: 1.0 for m in model_names}
        
        # Loss matrix
        loss_matrix = np.column_stack([losses[m] for m in model_names])
        
        # Pairwise loss differences
        mean_losses = loss_matrix.mean(axis=0)
        
        # Bootstrap for distribution
        p_values = {}
        included = model_names.copy()
        
        # Sequential elimination
        while len(included) > 1:
            # Compute t-statistics for remaining models
            remaining_idx = [model_names.index(m) for m in included]
            remaining_losses = loss_matrix[:, remaining_idx]
            remaining_means = remaining_losses.mean(axis=0)
            
            # Max-t statistic: worst relative performer
            avg_loss = remaining_means.mean()
            t_stats = (remaining_means - avg_loss) / (remaining_losses.std(axis=0) / np.sqrt(T) + 1e-8)
            
            worst_idx = np.argmax(t_stats)
            worst_model = included[worst_idx]
            max_t = t_stats[worst_idx]
            
            # Bootstrap p-value
            boot_max_t = []
            for _ in range(n_bootstrap):
                boot_idx = np.random.choice(T, T, replace=True)
                boot_losses = remaining_losses[boot_idx]
                boot_means = boot_losses.mean(axis=0)
                boot_avg = boot_means.mean()
                boot_t = (boot_means - boot_avg) / (boot_losses.std(axis=0) / np.sqrt(T) + 1e-8)
                boot_max_t.append(np.max(boot_t))
            
            p_value = np.mean(np.array(boot_max_t) >= max_t)
            p_values[worst_model] = p_value
            
            if p_value < alpha:
                # Eliminate worst model
                included.remove(worst_model)
            else:
                # Cannot reject, stop
                for m in included:
                    if m not in p_values:
                        p_values[m] = 1.0
                break
        
        # Last model always included
        if len(included) == 1 and included[0] not in p_values:
            p_values[included[0]] = 1.0
        
        return included, p_values
    
    def update_mcs(
        self,
        recent_predictions: Dict[str, np.ndarray],
        recent_actuals: np.ndarray
    ) -> List[str]:
        """
        Update MCS with recent data
        
        Parameters:
        -----------
        recent_predictions : Dict[str, np.ndarray]
            Recent model predictions
        recent_actuals : np.ndarray
            Recent actual values
            
        Returns:
        --------
        current_mcs : List[str]
            Updated MCS
        """
        # Compute losses
        losses = self.compute_losses(recent_predictions, recent_actuals, 'QLIKE')
        
        # Update loss history
        for model_name, loss in losses.items():
            self.loss_history[model_name].extend(loss.tolist())
        
        # Run MCS test
        alpha = 1 - self.config.mcs_confidence_level
        self.current_mcs, p_values = self.mcs_test(losses, alpha)
        
        # Record history
        self.mcs_history.append({
            'timestamp': pd.Timestamp.now(),
            'included': self.current_mcs.copy(),
            'excluded': [m for m in self.models.keys() if m not in self.current_mcs],
            'p_values': p_values
        })
        
        logger.info(f"MCS updated: {self.current_mcs}")
        
        return self.current_mcs
    
    def get_ensemble_weights(self, method: str = 'equal') -> Dict[str, float]:
        """
        Get ensemble weights based on MCS
        
        Parameters:
        -----------
        method : str
            Weighting method: 'equal', 'inverse_loss', 'softmax'
            
        Returns:
        --------
        weights : Dict[str, float]
            Model weights (sum to 1)
        """
        weights = {m: 0.0 for m in self.models.keys()}
        
        if not self.current_mcs:
            # Fallback: all models equal
            n = len(self.models)
            return {m: 1/n for m in self.models.keys()}
        
        if method == 'equal':
            n_mcs = len(self.current_mcs)
            for m in self.current_mcs:
                weights[m] = 1 / n_mcs
                
        elif method == 'inverse_loss':
            # Weight by inverse of recent loss
            recent_losses = {}
            for m in self.current_mcs:
                if self.loss_history[m]:
                    recent_losses[m] = np.mean(self.loss_history[m][-22:])  # Last month
                else:
                    recent_losses[m] = 1.0
            
            inv_losses = {m: 1 / (loss + 1e-8) for m, loss in recent_losses.items()}
            total = sum(inv_losses.values())
            
            for m in self.current_mcs:
                weights[m] = inv_losses[m] / total
                
        elif method == 'softmax':
            # Softmax weights
            recent_losses = {}
            for m in self.current_mcs:
                if self.loss_history[m]:
                    recent_losses[m] = np.mean(self.loss_history[m][-22:])
                else:
                    recent_losses[m] = 1.0
            
            neg_losses = np.array([-recent_losses[m] for m in self.current_mcs])
            exp_losses = np.exp(neg_losses - neg_losses.max())
            softmax = exp_losses / exp_losses.sum()
            
            for i, m in enumerate(self.current_mcs):
                weights[m] = softmax[i]
        
        return weights
    
    def adaptive_forecast(
        self,
        model_forecasts: Dict[str, float],
        method: str = 'equal'
    ) -> float:
        """
        Generate MCS-weighted ensemble forecast
        
        Parameters:
        -----------
        model_forecasts : Dict[str, float]
            Individual model forecasts
        method : str
            Weighting method
            
        Returns:
        --------
        ensemble_forecast : float
            Weighted ensemble forecast
        """
        weights = self.get_ensemble_weights(method)
        
        ensemble = sum(
            weights[m] * model_forecasts.get(m, 0)
            for m in self.models.keys()
        )
        
        return ensemble


# =============================================================================
# LAYER 5: REGIME-CONDITIONAL ALPHA TESTING
# =============================================================================

class RegimeConditionalAlphaTest:
    """
    Regime-Specific Alpha Testing
    
    Compensates Standard Alpha Test Weakness:
    - Overall alpha may be insignificant
    - But regime-specific alpha can be significant
    - Weighted alpha provides economic interpretation
    
    Mathematical Framework:
    ----------------------
    r_p,t = α_regime + β_regime × r_m,t + ε_t  for each regime
    
    Weighted α = Σ (regime_proportion × α_regime)
    """
    
    def __init__(self, regimes: np.ndarray):
        self.regimes = regimes
        self.regime_labels = {
            MarketRegime.BULL.value: 'Bull',
            MarketRegime.TRANSITION.value: 'Transition',
            MarketRegime.BEAR.value: 'Bear'
        }
    
    def test_regime_alpha(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        min_obs: int = 30
    ) -> Dict[str, AlphaTestResult]:
        """
        Test alpha for each regime
        
        Parameters:
        -----------
        portfolio_returns : np.ndarray
            Portfolio returns
        benchmark_returns : np.ndarray
            Benchmark returns
        min_obs : int
            Minimum observations per regime
            
        Returns:
        --------
        results : Dict[str, AlphaTestResult]
            Alpha test results for each regime
        """
        import statsmodels.api as sm
        
        results = {}
        
        # Test each regime
        for regime_val in [0, 1, 2]:
            mask = self.regimes == regime_val
            regime_name = self.regime_labels.get(regime_val, f'Regime_{regime_val}')
            
            if mask.sum() < min_obs:
                logger.warning(f"Insufficient observations for {regime_name}: {mask.sum()}")
                continue
            
            port_regime = portfolio_returns[mask]
            bench_regime = benchmark_returns[mask]
            
            # OLS with HAC standard errors
            X = sm.add_constant(bench_regime)
            model = sm.OLS(port_regime, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
            
            # Extract results
            alpha = model.params[0]
            alpha_pval = model.pvalues[0]
            beta = model.params[1]
            
            # Confidence interval
            ci = model.conf_int(alpha=0.05)
            alpha_ci = (ci[0, 0], ci[1, 0])
            
            # R-squared
            r_squared = model.rsquared
            
            results[regime_name] = AlphaTestResult(
                alpha_daily=alpha,
                alpha_annual=alpha * 252,
                p_value=alpha_pval,
                is_significant=alpha_pval < 0.05,
                confidence_interval=(alpha_ci[0] * 252, alpha_ci[1] * 252),
                beta=beta,
                r_squared=r_squared,
                n_observations=mask.sum()
            )
        
        # Overall alpha
        X_all = sm.add_constant(benchmark_returns)
        model_all = sm.OLS(portfolio_returns, X_all).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        
        ci_all = model_all.conf_int(alpha=0.05)
        
        results['Overall'] = AlphaTestResult(
            alpha_daily=model_all.params[0],
            alpha_annual=model_all.params[0] * 252,
            p_value=model_all.pvalues[0],
            is_significant=model_all.pvalues[0] < 0.05,
            confidence_interval=(ci_all[0, 0] * 252, ci_all[1, 0] * 252),
            beta=model_all.params[1],
            r_squared=model_all.rsquared,
            n_observations=len(portfolio_returns)
        )
        
        return results
    
    def weighted_alpha(
        self,
        regime_results: Dict[str, AlphaTestResult]
    ) -> Dict[str, Any]:
        """
        Compute regime-weighted alpha
        
        Parameters:
        -----------
        regime_results : Dict[str, AlphaTestResult]
            Results from test_regime_alpha
            
        Returns:
        --------
        weighted_results : Dict
            Weighted alpha and analysis
        """
        # Compute weights and weighted alpha
        total_obs = sum(
            r.n_observations for name, r in regime_results.items() 
            if name != 'Overall'
        )
        
        weighted_alpha = 0.0
        contributions = {}
        
        for name, result in regime_results.items():
            if name == 'Overall':
                continue
            
            weight = result.n_observations / total_obs
            contribution = weight * result.alpha_annual
            weighted_alpha += contribution
            contributions[name] = {
                'weight': weight,
                'alpha': result.alpha_annual,
                'contribution': contribution,
                'significant': result.is_significant
            }
        
        # Find best and worst regime
        regime_alphas = [
            (name, r.alpha_annual) for name, r in regime_results.items()
            if name != 'Overall'
        ]
        
        best_regime = max(regime_alphas, key=lambda x: x[1])[0] if regime_alphas else None
        worst_regime = min(regime_alphas, key=lambda x: x[1])[0] if regime_alphas else None
        
        # Check if any regime is significant
        any_significant = any(
            r.is_significant for name, r in regime_results.items()
            if name != 'Overall'
        )
        
        return {
            'weighted_alpha_annual': weighted_alpha,
            'regime_contributions': contributions,
            'best_regime': best_regime,
            'worst_regime': worst_regime,
            'any_regime_significant': any_significant,
            'overall_significant': regime_results.get('Overall', AlphaTestResult(
                0, 0, 1, False, (0, 0), 1, 0, 0
            )).is_significant
        }
    
    def generate_report(
        self,
        regime_results: Dict[str, AlphaTestResult],
        weighted_results: Dict[str, Any]
    ) -> str:
        """Generate human-readable alpha analysis report"""
        
        report = []
        report.append("=" * 60)
        report.append("REGIME-CONDITIONAL ALPHA ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall
        overall = regime_results.get('Overall')
        if overall:
            report.append("OVERALL ALPHA:")
            report.append(f"  Annual Alpha: {overall.alpha_annual:.2%}")
            report.append(f"  P-value: {overall.p_value:.4f}")
            report.append(f"  95% CI: [{overall.confidence_interval[0]:.2%}, {overall.confidence_interval[1]:.2%}]")
            report.append(f"  Significant: {'✓ YES' if overall.is_significant else '✗ NO'}")
            report.append(f"  Beta: {overall.beta:.3f}")
            report.append(f"  R²: {overall.r_squared:.2%}")
            report.append("")
        
        # Regime-specific
        report.append("REGIME-SPECIFIC ALPHA:")
        for name in ['Bull', 'Transition', 'Bear']:
            if name in regime_results:
                r = regime_results[name]
                report.append(f"\n  {name} Regime:")
                report.append(f"    Annual Alpha: {r.alpha_annual:.2%}")
                report.append(f"    P-value: {r.p_value:.4f}")
                report.append(f"    Significant: {'✓' if r.is_significant else '✗'}")
                report.append(f"    Observations: {r.n_observations}")
        
        report.append("")
        report.append("WEIGHTED ANALYSIS:")
        report.append(f"  Weighted Alpha: {weighted_results['weighted_alpha_annual']:.2%}")
        report.append(f"  Best Regime: {weighted_results['best_regime']}")
        report.append(f"  Worst Regime: {weighted_results['worst_regime']}")
        
        report.append("")
        report.append("CONCLUSION:")
        if weighted_results['overall_significant']:
            report.append("  ✓ Overall alpha is statistically significant")
        elif weighted_results['any_regime_significant']:
            report.append("  ⚠ Overall alpha not significant, but regime-specific alpha exists")
        elif weighted_results['weighted_alpha_annual'] > 0.02:
            report.append("  ⚠ Statistical significance lacking, but economic significance (>2%)")
        else:
            report.append("  ✗ No statistical or economic significance detected")
        
        report.append("=" * 60)
        
        return "\n".join(report)


# =============================================================================
# INTEGRATED MODEL: COMBINING ALL COMPONENTS
# =============================================================================

class EnhancedIntegratedModel:
    """
    Enhanced Integrated Model: Complete Weakness Compensation Framework
    
    Integrates:
    1. Dynamic Factor Loading (FOGI compensation)
    2. Dynamic Sentiment (FinBERT compensation)
    3. Nonlinear GARCH-MIDAS (long-run vol compensation)
    4. Adaptive MCS (model selection compensation)
    5. Regime-Conditional Alpha Testing (validation compensation)
    
    This class orchestrates all components for portfolio optimization.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        
        # Initialize components
        self.dynamic_loading = DynamicFactorLoading(
            n_assets=self.config.n_assets,
            n_factors=self.config.n_factors,
            config=self.config
        )
        
        self.sentiment_model = DynamicSentimentModel(
            ar_order=self.config.sentiment_ar_order,
            config=self.config
        )
        
        self.sentiment_analyzer = FinBERTSentimentAnalyzer()
        
        self.nonlinear_garch = NonlinearGARCHMIDAS(config=self.config)
        
        self.mcs_selector = None  # Initialized after models defined
        
        # State
        self.current_regime: MarketRegime = MarketRegime.TRANSITION
        self.regime_history: List[MarketRegime] = []
        
        # Fitted flag
        self._is_fitted = False
    
    def fit(
        self,
        returns: np.ndarray,
        factor_returns: np.ndarray,
        predictors: pd.DataFrame,
        sentiment_history: Optional[np.ndarray] = None,
        regimes: Optional[np.ndarray] = None
    ) -> 'EnhancedIntegratedModel':
        """
        Fit all model components
        
        Parameters:
        -----------
        returns : np.ndarray
            Asset returns (T × n_assets)
        factor_returns : np.ndarray
            Factor returns (T × n_factors)
        predictors : pd.DataFrame
            Macro/sentiment predictors
        sentiment_history : np.ndarray
            Historical sentiment scores
        regimes : np.ndarray
            Historical regime classifications
        """
        logger.info("Fitting EnhancedIntegratedModel...")
        
        # 1. Estimate initial factor loading
        initial_loading = self.dynamic_loading.estimate_from_returns(
            returns, self.config.n_factors
        )
        self.dynamic_loading.initialize(initial_loading)
        
        # 2. Fit sentiment AR model
        if sentiment_history is not None:
            self.sentiment_model.fit_ar_model(sentiment_history)
        
        # 3. Fit nonlinear GARCH-MIDAS
        # Use first asset for simplicity (or aggregate)
        if returns.ndim > 1:
            agg_returns = returns.mean(axis=1)
        else:
            agg_returns = returns
        
        self.nonlinear_garch.fit(agg_returns, predictors)
        
        # 4. Store regime history
        if regimes is not None:
            self.regime_history = [MarketRegime(r) for r in regimes]
        
        self._is_fitted = True
        logger.info("EnhancedIntegratedModel fitted successfully")
        
        return self
    
    def process_daily_update(
        self,
        returns: np.ndarray,
        factor_returns: np.ndarray,
        news_headlines: List[str],
        macro_predictors: pd.Series,
        regime: MarketRegime
    ) -> Dict[str, Any]:
        """
        Process daily update and generate signals
        
        Parameters:
        -----------
        returns : np.ndarray
            Daily asset returns
        factor_returns : np.ndarray
            Daily factor returns
        news_headlines : List[str]
            Today's news headlines
        macro_predictors : pd.Series
            Today's macro predictors
        regime : MarketRegime
            Today's detected regime
            
        Returns:
        --------
        signals : Dict[str, Any]
            Integrated signals for portfolio optimization
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        results = {}
        
        # 1. Update dynamic factor loading
        loading, loading_uncertainty = self.dynamic_loading.update(
            returns, factor_returns
        )
        
        # Apply regime conditioning
        regime_loading = self.dynamic_loading.apply_regime_conditioning(loading, regime)
        results['factor_loading'] = regime_loading
        results['loading_uncertainty'] = loading_uncertainty
        
        # 2. Process sentiment
        sentiment_signals = [self.sentiment_analyzer.analyze(h) for h in news_headlines]
        sentiment_scores = [s.score for s in sentiment_signals]
        
        agg_sentiment, sentiment_vol = self.sentiment_model.multi_frequency_aggregation(
            sentiment_scores, weights='recency'
        )
        
        # Predict future sentiment
        if len(self.sentiment_model.sentiment_history) >= self.config.sentiment_ar_order:
            recent = np.array(self.sentiment_model.sentiment_history[-self.config.sentiment_ar_order:])
            pred_sentiment, pred_uncertainty = self.sentiment_model.predict_sentiment(recent)
        else:
            pred_sentiment = np.array([agg_sentiment])
            pred_uncertainty = np.array([sentiment_vol])
        
        # Sentiment change
        prev_sentiment = self.sentiment_model.sentiment_history[-1] if self.sentiment_model.sentiment_history else 0
        sentiment_change = agg_sentiment - prev_sentiment
        
        # Volatility impact
        vol_impact = self.sentiment_model.sentiment_to_volatility_impact(
            agg_sentiment, sentiment_change, sentiment_vol, regime
        )
        
        self.sentiment_model.update(agg_sentiment, sentiment_vol)
        
        results['sentiment'] = {
            'current': agg_sentiment,
            'volatility': sentiment_vol,
            'predicted': pred_sentiment[0],
            'predicted_uncertainty': pred_uncertainty[0],
            'momentum': sentiment_change,
            'vol_adjustment': vol_impact
        }
        
        # 3. Long-run volatility
        predictors_df = macro_predictors.to_frame().T
        tau = self.nonlinear_garch.predict_long_run_vol(predictors_df)
        results['long_run_vol'] = tau[0]
        
        # 4. Combined volatility prediction
        # Short-term: from recent returns
        short_term_vol = np.std(returns) if len(returns) > 1 else 0.01
        
        # Apply sentiment adjustment
        adjusted_vol = short_term_vol * vol_impact['total_vol_adjustment']
        
        # Combine with long-run
        combined_vol = np.sqrt(adjusted_vol ** 2 * tau[0])
        
        results['predicted_volatility'] = {
            'short_term': short_term_vol,
            'sentiment_adjusted': adjusted_vol,
            'long_run': tau[0],
            'combined': combined_vol
        }
        
        # 5. Update regime
        self.current_regime = regime
        self.regime_history.append(regime)
        
        results['regime'] = regime
        
        return results
    
    def validate_performance(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> Dict[str, Any]:
        """
        Validate portfolio performance with regime-conditional alpha testing
        
        Parameters:
        -----------
        portfolio_returns : np.ndarray
            Portfolio returns
        benchmark_returns : np.ndarray
            Benchmark returns
            
        Returns:
        --------
        validation : Dict[str, Any]
            Comprehensive validation results
        """
        if not self.regime_history:
            logger.warning("No regime history. Running overall alpha test only.")
            regimes = np.zeros(len(portfolio_returns))
        else:
            regimes = np.array([r.value for r in self.regime_history[-len(portfolio_returns):]])
        
        # Regime-conditional alpha test
        alpha_tester = RegimeConditionalAlphaTest(regimes)
        regime_results = alpha_tester.test_regime_alpha(
            portfolio_returns, benchmark_returns
        )
        weighted_results = alpha_tester.weighted_alpha(regime_results)
        
        # Generate report
        report = alpha_tester.generate_report(regime_results, weighted_results)
        
        return {
            'regime_alpha': regime_results,
            'weighted_alpha': weighted_results,
            'report': report
        }
    
    def get_optimal_weights(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        risk_aversion: float = 1.0,
        regime: Optional[MarketRegime] = None
    ) -> np.ndarray:
        """
        Compute optimal portfolio weights
        
        Parameters:
        -----------
        expected_returns : np.ndarray
            Expected returns
        covariance : np.ndarray
            Covariance matrix
        risk_aversion : float
            Risk aversion parameter
        regime : MarketRegime
            Current regime for adjustment
            
        Returns:
        --------
        weights : np.ndarray
            Optimal portfolio weights
        """
        n_assets = len(expected_returns)
        
        # Regime adjustment
        regime = regime or self.current_regime
        regime_mult = {
            MarketRegime.BULL: 1.2,      # More aggressive
            MarketRegime.TRANSITION: 1.0,
            MarketRegime.BEAR: 0.7       # More defensive
        }.get(regime, 1.0)
        
        # Mean-variance optimization
        # max: μ'w - (λ/2) w'Σw
        # s.t.: Σw = 1, w ≥ 0
        
        def objective(w):
            ret = np.dot(expected_returns, w)
            risk = np.dot(w, np.dot(covariance, w))
            return -(ret - 0.5 * risk_aversion * risk)
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 1) for _ in range(n_assets)]
        
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = result.x * regime_mult
            weights = weights / weights.sum()  # Renormalize
            return weights
        else:
            logger.warning("Optimization failed. Returning equal weights.")
            return np.ones(n_assets) / n_assets


# =============================================================================
# STATISTICAL VALIDATION FRAMEWORK
# =============================================================================

class IntegratedStatisticalFramework:
    """
    Comprehensive Statistical Validation Framework
    
    Combines:
    - Model Confidence Set (MCS)
    - Superior Predictive Ability (SPA) test
    - Bootstrap alpha confidence intervals
    - Regime-conditional testing
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
    
    def run_mcs_validation(
        self,
        predictions: Dict[str, np.ndarray],
        actuals: np.ndarray,
        loss_types: List[str] = ['MSE', 'MAE', 'QLIKE']
    ) -> Dict[str, Dict]:
        """
        Run MCS validation with multiple loss functions
        
        Parameters:
        -----------
        predictions : Dict[str, np.ndarray]
            Model predictions
        actuals : np.ndarray
            Actual values
        loss_types : List[str]
            Loss functions to use
            
        Returns:
        --------
        results : Dict[str, Dict]
            MCS results for each loss type
        """
        mcs = AdaptiveMCSSelector(
            models={name: lambda x: x for name in predictions.keys()},
            config=self.config
        )
        
        results = {}
        
        for loss_type in loss_types:
            losses = mcs.compute_losses(predictions, actuals, loss_type)
            included, p_values = mcs.mcs_test(losses)
            
            results[loss_type] = {
                'included': included,
                'excluded': [m for m in predictions.keys() if m not in included],
                'p_values': p_values
            }
        
        return results
    
    def bootstrap_alpha_ci(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        n_bootstrap: int = 10000,
        confidence: float = 0.95
    ) -> Dict[str, Any]:
        """
        Bootstrap confidence interval for alpha
        
        Parameters:
        -----------
        portfolio_returns : np.ndarray
            Portfolio returns
        benchmark_returns : np.ndarray
            Benchmark returns
        n_bootstrap : int
            Bootstrap replications
        confidence : float
            Confidence level
            
        Returns:
        --------
        results : Dict
            Bootstrap alpha results
        """
        T = len(portfolio_returns)
        alphas = []
        
        for _ in range(n_bootstrap):
            # Block bootstrap (preserve autocorrelation)
            block_size = 22  # Monthly blocks
            n_blocks = T // block_size + 1
            
            blocks = np.random.choice(T - block_size + 1, n_blocks, replace=True)
            indices = np.concatenate([np.arange(b, min(b + block_size, T)) for b in blocks])[:T]
            
            port_boot = portfolio_returns[indices]
            bench_boot = benchmark_returns[indices]
            
            # OLS alpha
            X = np.column_stack([np.ones(T), bench_boot])
            try:
                beta = np.linalg.lstsq(X, port_boot, rcond=None)[0]
                alphas.append(beta[0])
            except:
                continue
        
        alphas = np.array(alphas)
        
        # Confidence interval
        lower = np.percentile(alphas, (1 - confidence) / 2 * 100)
        upper = np.percentile(alphas, (1 + confidence) / 2 * 100)
        
        # Point estimate
        X_full = np.column_stack([np.ones(T), benchmark_returns])
        beta_full = np.linalg.lstsq(X_full, portfolio_returns, rcond=None)[0]
        alpha_point = beta_full[0]
        
        return {
            'alpha_point': alpha_point * 252,
            'alpha_ci_lower': lower * 252,
            'alpha_ci_upper': upper * 252,
            'contains_zero': lower <= 0 <= upper,
            'n_bootstrap': len(alphas)
        }
    
    def full_validation(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        vol_predictions: Dict[str, np.ndarray],
        realized_vol: np.ndarray,
        regimes: np.ndarray
    ) -> Dict[str, Any]:
        """
        Run full validation suite
        
        Returns comprehensive validation results including:
        - MCS for volatility models
        - Bootstrap alpha CI
        - Regime-conditional alpha
        """
        results = {}
        
        # 1. MCS validation for volatility
        results['mcs'] = self.run_mcs_validation(vol_predictions, realized_vol)
        
        # 2. Bootstrap alpha CI
        results['bootstrap_alpha'] = self.bootstrap_alpha_ci(
            portfolio_returns, benchmark_returns
        )
        
        # 3. Regime-conditional alpha
        alpha_tester = RegimeConditionalAlphaTest(regimes)
        regime_results = alpha_tester.test_regime_alpha(
            portfolio_returns, benchmark_returns
        )
        weighted = alpha_tester.weighted_alpha(regime_results)
        
        results['regime_alpha'] = {
            'by_regime': {name: {
                'alpha': r.alpha_annual,
                'p_value': r.p_value,
                'significant': r.is_significant
            } for name, r in regime_results.items()},
            'weighted': weighted
        }
        
        # 4. Summary
        results['summary'] = {
            'mcs_best_model': results['mcs']['QLIKE']['included'][0] if results['mcs']['QLIKE']['included'] else None,
            'alpha_significant': not results['bootstrap_alpha']['contains_zero'],
            'any_regime_significant': weighted['any_regime_significant'],
            'weighted_alpha': weighted['weighted_alpha_annual']
        }
        
        return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def create_enhanced_model(
    n_assets: int = 10,
    n_factors: int = 3,
    **kwargs
) -> EnhancedIntegratedModel:
    """
    Factory function to create EnhancedIntegratedModel
    
    Parameters:
    -----------
    n_assets : int
        Number of assets
    n_factors : int
        Number of factors
    **kwargs : dict
        Additional configuration parameters
        
    Returns:
    --------
    model : EnhancedIntegratedModel
        Configured model instance
    """
    config = ModelConfig(
        n_assets=n_assets,
        n_factors=n_factors,
        **{k: v for k, v in kwargs.items() if hasattr(ModelConfig, k)}
    )
    
    return EnhancedIntegratedModel(config)


if __name__ == "__main__":
    # Example usage
    print("Enhanced Integrated Model: Weakness Compensation Framework")
    print("=" * 60)
    
    # Create model
    model = create_enhanced_model(n_assets=10, n_factors=3)
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    T = 500
    n_assets = 10
    n_factors = 3
    
    # Synthetic returns
    factor_returns = np.random.randn(T, n_factors) * 0.01
    factor_loading = np.random.randn(n_assets, n_factors) * 0.5
    idio_returns = np.random.randn(T, n_assets) * 0.005
    returns = factor_returns @ factor_loading.T + idio_returns
    
    # Synthetic predictors
    predictors = pd.DataFrame({
        'VIX': np.random.randn(T) * 5 + 20,
        'TERM_SPREAD': np.random.randn(T) * 0.5 + 1.5,
        'CREDIT_SPREAD': np.random.randn(T) * 0.3 + 2.0,
        'SENTIMENT': np.random.randn(T) * 0.2
    })
    
    # Synthetic regimes
    regimes = np.random.choice([0, 1, 2], T, p=[0.4, 0.3, 0.3])
    
    # Fit model
    print("\nFitting model...")
    model.fit(
        returns=returns,
        factor_returns=factor_returns,
        predictors=predictors,
        sentiment_history=predictors['SENTIMENT'].values,
        regimes=regimes
    )
    
    # Process daily update
    print("\nProcessing daily update...")
    daily_signals = model.process_daily_update(
        returns=returns[-1],
        factor_returns=factor_returns[-1],
        news_headlines=["Market rises on strong earnings", "Fed signals rate pause"],
        macro_predictors=predictors.iloc[-1],
        regime=MarketRegime.BULL
    )
    
    print(f"\nDaily Signals:")
    print(f"  Sentiment: {daily_signals['sentiment']['current']:.3f}")
    print(f"  Volatility Adjustment: {daily_signals['sentiment']['vol_adjustment']['total_vol_adjustment']:.3f}")
    print(f"  Long-run Vol: {daily_signals['long_run_vol']:.3f}")
    print(f"  Combined Vol: {daily_signals['predicted_volatility']['combined']:.3f}")
    
    # Validation
    print("\nRunning validation...")
    portfolio_returns = returns.mean(axis=1)  # Equal-weight for demo
    benchmark_returns = factor_returns[:, 0]  # First factor as benchmark
    
    validation = model.validate_performance(portfolio_returns, benchmark_returns)
    print(validation['report'])
    
    print("\n✓ Enhanced Integrated Model demonstration complete")
