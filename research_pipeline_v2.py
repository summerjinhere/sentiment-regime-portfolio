"""
Research Pipeline v2 - Duke Feedback Incorporated
==================================================

Key Improvements:
1. 4-Regime Framework (LowVol_Bear, LowVol_Bull, HighVol_Bear, HighVol_Bull)
2. Explicit Train/Test Split (2005-2014 design, 2015-2025 evaluation)
3. Realistic Transaction Costs (NASDAQ-calibrated: 8bps)
4. FF5 + Momentum Factor Adjustment
5. Multiple Testing Adjustment (Benjamini-Hochberg FDR)
6. Regime Threshold Sensitivity Analysis
7. Enhanced Ablation (8 configurations)
8. Strategic Benchmarks (Heston, Tetlock, Momentum, Risk-Only)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from enhanced_integrated_model_v3 import (
    ModelConfig,
    MarketRegime,
    EnhancedIntegratedModel,
    StatisticalValidator,
)


# =============================================================================
# RESEARCH DESIGN CONFIGURATION
# =============================================================================

@dataclass
class ResearchDesignV2:
    """Enhanced Research Configuration with Duke Feedback"""
    
    # Train/Test Split (Critical for Data-Mining Defense)
    design_start: str = "2005-11-22"
    design_end: str = "2014-12-31"
    eval_start: str = "2015-01-01"
    eval_end: str = "2025-12-03"
    full_sample_mode: bool = False
    
    # 4-Regime Framework
    use_4regime: bool = True
    vol_threshold: float = 0.0
    market_direction_window: int = 20
    vol_threshold_alt: List[float] = field(default_factory=lambda: [-0.5, -0.25, 0.0, 0.25, 0.5])
    
    # Layer Flags (Ablation)
    use_dynamic_beta: bool = True
    use_sentiment: bool = True
    use_midas_vol: bool = True
    
    # Portfolio Construction
    n_deciles: int = 10
    min_assets_per_date: int = 10
    min_history_days: int = 252
    
    # Transaction Costs (NASDAQ-calibrated)
    effective_tc_bps: float = 8.0
    tc_sensitivity_range: List[float] = field(default_factory=lambda: [0, 3, 5, 8, 10, 15, 20, 30, 50])
    
    # Factor Model
    factor_file: Optional[str] = None
    use_ff5: bool = True
    use_momentum: bool = True
    
    # Statistical Inference
    use_zscore_fm: bool = True
    bootstrap_samples: int = 1000
    bootstrap_block_size: int = 22
    apply_multiple_testing: bool = True
    fdr_level: float = 0.10
    
    # Benchmarks
    use_heston_sentiment: bool = True
    use_tetlock_sentiment: bool = True
    use_gambarelli_sentiment: bool = True
    use_risk_benchmark: bool = True
    use_momentum_benchmark: bool = True


# =============================================================================
# CROSS-SECTIONAL TESTER WITH 4-REGIME
# =============================================================================

class CrossSectionalTesterV2:
    """Enhanced Cross-Sectional Tester with 4-Regime Framework"""
    
    def __init__(self, model: EnhancedIntegratedModel, design: ResearchDesignV2):
        self.model = model
        self.design = design
        self.market_direction = None
    
    def _get_date_mask(self, idx: pd.DatetimeIndex, period: str = "full") -> pd.DatetimeIndex:
        if self.design.full_sample_mode or period == "full":
            start = pd.to_datetime(self.design.design_start)
            end = pd.to_datetime(self.design.eval_end)
        elif period == "design":
            start = pd.to_datetime(self.design.design_start)
            end = pd.to_datetime(self.design.design_end)
        elif period == "eval":
            start = pd.to_datetime(self.design.eval_start)
            end = pd.to_datetime(self.design.eval_end)
        else:
            start = idx[0]
            end = idx[-1]
        
        return idx[(idx >= start) & (idx <= end)]
    
    def compute_market_direction(self) -> pd.Series:
        if self.model.market_returns is None:
            raise ValueError("Market returns not available")
        
        window = self.design.market_direction_window
        rolling_ret = self.model.market_returns.rolling(window=window).mean()
        direction = (rolling_ret >= 0).astype(int) * 2 - 1
        self.market_direction = direction
        return direction
    
    def compute_4regimes(self, panel: pd.DataFrame) -> pd.DataFrame:
        df = panel.copy()
        
        if "vol_midas" not in df.columns:
            return df
        
        df["vol_midas_z"] = df.groupby(level=0)["vol_midas"].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
        )
        
        if self.market_direction is None:
            self.compute_market_direction()
        
        dates = df.index.get_level_values(0)
        df["mkt_direction"] = dates.map(lambda d: self.market_direction.get(d, 0))
        
        vol_th = self.design.vol_threshold
        
        low_vol = df["vol_midas_z"] < vol_th
        high_vol = df["vol_midas_z"] >= vol_th
        bear = df["mkt_direction"] < 0
        bull = df["mkt_direction"] >= 0
        
        df["reg_LB"] = (low_vol & bear).astype(int)
        df["reg_LU"] = (low_vol & bull).astype(int)
        df["reg_HB"] = (high_vol & bear).astype(int)
        df["reg_HU"] = (high_vol & bull).astype(int)
        
        conditions = [df["reg_LB"] == 1, df["reg_LU"] == 1, df["reg_HB"] == 1, df["reg_HU"] == 1]
        choices = [r.value for r in MarketRegime]
        df["regime"] = np.select(conditions, choices, default="Unknown")
        
        return df
    
    def add_4regime_interactions(self, panel: pd.DataFrame, standardize: bool = True) -> pd.DataFrame:
        df = panel.copy()
        
        if "sentiment" not in df.columns:
            return df
        
        sent_col = "sentiment"
        if standardize:
            df["sentiment_z"] = df.groupby(level=0)["sentiment"].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
            )
            sent_col = "sentiment_z"
        
        for reg in ["LB", "LU", "HB", "HU"]:
            reg_col = f"reg_{reg}"
            if reg_col in df.columns:
                df[f"sent_{reg}"] = df[sent_col] * df[reg_col]
        
        return df
    
    def build_panel(self, period: str = "full") -> pd.DataFrame:
        if self.model.returns is None:
            raise ValueError("Model has no returns.")
        if not self.model.fitted:
            raise ValueError("Model not fitted.")
        
        rets = self.model.returns.copy()
        dates = rets.index
        tickers = rets.columns
        
        mask_idx = self._get_date_mask(dates, period)
        rets = rets.loc[mask_idx]
        
        ret_fwd = rets.shift(-1)
        
        beta_df = pd.DataFrame(
            {tic: self.model.get_dynamic_beta_series(tic).reindex(rets.index) for tic in tickers},
            index=rets.index
        )
        
        vol_df = pd.DataFrame(
            {tic: self.model.get_volatility_series(tic).reindex(rets.index) for tic in tickers},
            index=rets.index
        )
        
        if (self.model.layer2_sentiment is not None and 
            self.model.layer2_sentiment.smoothed_sentiment is not None):
            sent_df = self.model.layer2_sentiment.smoothed_sentiment.reindex(rets.index)
        else:
            sent_df = pd.DataFrame(index=rets.index, columns=tickers)
        
        panel = pd.DataFrame({
            "ret_excess": rets.stack(dropna=True),
            "ret_fwd": ret_fwd.stack(dropna=True),
            "beta_dyn": beta_df.stack(dropna=True),
            "vol_midas": vol_df.stack(dropna=True),
        })
        
        sent_long = sent_df.stack(dropna=False).rename("sentiment")
        panel = panel.join(sent_long, how="left")
        
        panel = panel.dropna(subset=["ret_fwd"])
        
        counts = panel.groupby(level=0)["ret_fwd"].size()
        good_dates = counts[counts >= self.design.min_assets_per_date].index
        panel = panel.loc[panel.index.get_level_values(0).isin(good_dates)]
        
        if not self.design.use_dynamic_beta:
            panel["beta_dyn"] = np.nan
        if not self.design.use_sentiment:
            panel["sentiment"] = np.nan
        if not self.design.use_midas_vol:
            panel["vol_midas"] = np.nan
        
        panel.index.names = ["Date", "Ticker"]
        
        if self.design.use_4regime:
            panel = self.compute_4regimes(panel)
            panel = self.add_4regime_interactions(panel, standardize=self.design.use_zscore_fm)
        
        return panel
    
    def fama_macbeth(
        self,
        panel: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        min_assets: int = 10,
    ) -> Dict[str, Any]:
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required")
        
        if feature_cols is None:
            if self.design.use_4regime:
                feature_cols = ["beta_dyn", "vol_midas", "sent_LB", "sent_LU", "sent_HB", "sent_HU"]
            else:
                feature_cols = ["beta_dyn", "sentiment", "vol_midas"]
        
        feature_cols = [c for c in feature_cols if c in panel.columns and panel[c].notna().any()]
        if not feature_cols:
            raise ValueError("No valid feature columns")
        
        coef_list = []
        
        for date, df_t in panel.groupby(level=0):
            df_t = df_t.dropna(subset=["ret_fwd"] + feature_cols)
            if len(df_t) < min_assets:
                continue
            
            y = df_t["ret_fwd"].values
            X = df_t[feature_cols].values
            X = sm.add_constant(X)
            
            model = OLS(y, X)
            res = model.fit()
            coef_list.append(pd.Series(res.params, index=["const"] + feature_cols))
        
        coef_mat = pd.DataFrame(coef_list)
        T = len(coef_mat)
        
        if T == 0:
            raise ValueError("No valid regressions")
        
        mean_coefs = coef_mat.mean()
        tstats = mean_coefs / (coef_mat.std(ddof=1) / np.sqrt(T))
        
        if SCIPY_AVAILABLE:
            pvalues = 2 * (1 - stats.t.cdf(np.abs(tstats), T - 1))
        else:
            pvalues = pd.Series(np.nan, index=tstats.index)
        
        return {
            "mean_coefs": mean_coefs,
            "tstats": tstats,
            "pvalues": pvalues,
            "T": T,
            "coef_time_series": coef_mat,
        }


# =============================================================================
# FACTOR VALIDATOR (FF5 + MOM)
# =============================================================================

class FactorValidatorV2:
    """Factor validation with FF5 + Momentum"""
    
    def __init__(self, factor_file: Optional[str] = None, model: EnhancedIntegratedModel = None):
        self.factor_data = None
        self.model = model
        
        if factor_file:
            self._load_factors(factor_file)
        elif model is not None:
            self._construct_proxy_factors()
    
    def _load_factors(self, filepath: str):
        try:
            self.factor_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            if self.factor_data.abs().mean().mean() > 1:
                self.factor_data /= 100
            print(f"✅ FF factors loaded: {len(self.factor_data)} days")
        except Exception as e:
            print(f"⚠️ Could not load factors: {e}")
    
    def _construct_proxy_factors(self):
        if self.model is None or self.model.returns is None:
            return
        
        rets = self.model.returns
        dates = rets.index
        
        mkt_rf = rets['SPY'] if 'SPY' in rets.columns else rets.mean(axis=1)
        
        vol_rank = rets.std().rank(pct=True)
        small = rets.columns[vol_rank > 0.5]
        large = rets.columns[vol_rank <= 0.5]
        smb = rets[small].mean(axis=1) - rets[large].mean(axis=1)
        
        if self.model.fitted:
            beta_dict = self.model.layer1_factor.get_current_betas()
            beta_rank = pd.Series(beta_dict).rank(pct=True)
            value = [t for t in rets.columns if t in beta_rank and beta_rank[t] <= 0.3]
            growth = [t for t in rets.columns if t in beta_rank and beta_rank[t] > 0.7]
            hml = rets[value].mean(axis=1) - rets[growth].mean(axis=1) if value and growth else pd.Series(0, index=dates)
        else:
            hml = pd.Series(0, index=dates)
        
        mom_signal = rets.rolling(252).mean().shift(21)
        mom_rank = mom_signal.rank(axis=1, pct=True)
        
        mom_series = []
        for date in dates:
            if date not in mom_rank.index:
                mom_series.append(0)
                continue
            ranks = mom_rank.loc[date].dropna()
            if len(ranks) < 10:
                mom_series.append(0)
                continue
            winners = ranks[ranks > 0.7].index
            losers = ranks[ranks < 0.3].index
            if len(winners) > 0 and len(losers) > 0:
                mom_ret = rets.loc[date, winners].mean() - rets.loc[date, losers].mean()
                mom_series.append(mom_ret if not np.isnan(mom_ret) else 0)
            else:
                mom_series.append(0)
        
        self.factor_data = pd.DataFrame({
            'Mkt-RF': mkt_rf,
            'SMB': smb,
            'HML': hml,
            'MOM': pd.Series(mom_series, index=dates),
            'RF': 0.03/252
        }, index=dates)
        
        print("✅ Proxy factors constructed")
    
    def run_regression(
        self,
        strategy_returns: pd.Series,
        model_type: str = "FF5+MOM",
        market_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        if not STATSMODELS_AVAILABLE:
            return {"error": "statsmodels not installed"}
        
        y = strategy_returns.copy()
        
        if self.factor_data is not None:
            factors = self.factor_data.reindex(y.index).dropna()
            valid_idx = factors.index.intersection(y.index)
            y = y.loc[valid_idx]
            factors = factors.loc[valid_idx]
            
            rf = factors['RF'] if 'RF' in factors.columns else 0
            y_excess = y - rf
            
            factor_names = {
                "CAPM": ["Mkt-RF"],
                "FF3": ["Mkt-RF", "SMB", "HML"],
                "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
                "FF5+MOM": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"],
            }
            
            cols = [c for c in factor_names.get(model_type, ["Mkt-RF"]) if c in factors.columns]
            X = factors[cols]
            
        elif market_returns is not None:
            valid_idx = market_returns.index.intersection(y.index)
            y = y.loc[valid_idx]
            mkt = market_returns.loc[valid_idx]
            y_excess = y
            X = pd.DataFrame({'Mkt-RF': mkt}, index=valid_idx)
            cols = ['Mkt-RF']
        else:
            return {"error": "No factor data"}
        
        if len(y_excess) < 30:
            return {"error": "Insufficient observations"}
        
        X = sm.add_constant(X)
        model = OLS(y_excess, X)
        res = model.fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        
        return {
            "alpha_annual": res.params['const'] * 252,
            "alpha_tstat": res.tvalues['const'],
            "alpha_pval": res.pvalues['const'],
            "r_squared": res.rsquared,
            "params": res.params.to_dict(),
            "tvalues": res.tvalues.to_dict(),
            "pvalues": res.pvalues.to_dict(),
            "n_obs": int(res.nobs),
            "model_type": model_type,
        }


# =============================================================================
# DECILE PORTFOLIO TESTER
# =============================================================================

class DecilePortfolioTesterV2:
    """Decile portfolio with 4-regime and strategic benchmarks"""
    
    def __init__(self, model: EnhancedIntegratedModel, design: ResearchDesignV2):
        self.model = model
        self.design = design
    
    def decile_backtest(self, panel: pd.DataFrame, score_col: str = "score") -> Tuple[Dict[int, pd.Series], pd.Series]:
        if score_col not in panel.columns:
            raise ValueError(f"{score_col} not in panel")
        
        df = panel.copy()
        dates = df.index.get_level_values(0).unique().sort_values()
        
        decile_returns = {d: [] for d in range(1, self.design.n_deciles + 1)}
        decile_dates = []
        
        for date in dates:
            df_t = df.xs(date, level=0).dropna(subset=[score_col, "ret_fwd"])
            if len(df_t) < self.design.min_assets_per_date:
                continue
            
            df_t = df_t.sort_values(score_col)
            n = len(df_t)
            bucket = np.floor(np.linspace(0, self.design.n_deciles, n, endpoint=False)).astype(int) + 1
            df_t["decile"] = bucket
            
            for d in range(1, self.design.n_deciles + 1):
                subset = df_t[df_t["decile"] == d]
                decile_returns[d].append(subset["ret_fwd"].mean() if len(subset) > 0 else np.nan)
            
            decile_dates.append(date)
        
        decile_ts = {}
        for d in range(1, self.design.n_deciles + 1):
            decile_ts[d] = pd.Series(decile_returns[d], index=pd.DatetimeIndex(decile_dates)).dropna()
        
        if self.design.n_deciles >= 2:
            ls = decile_ts[self.design.n_deciles].align(decile_ts[1], join="inner")
            long_short = ls[0] - ls[1]
        else:
            long_short = pd.Series(dtype=float)
        
        return decile_ts, long_short
    
    def regime_conditional_backtest(self, panel: pd.DataFrame, score_col: str = "score") -> Dict[str, Dict[str, Any]]:
        results = {}
        
        if "regime" not in panel.columns:
            return {"error": "No regime column"}
        
        for regime in [r.value for r in MarketRegime]:
            sub_panel = panel[panel["regime"] == regime]
            
            if len(sub_panel) < 100:
                results[regime] = {"error": "Insufficient data", "n_obs": len(sub_panel)}
                continue
            
            try:
                _, long_short = self.decile_backtest(sub_panel, score_col)
                
                if len(long_short) < 30:
                    results[regime] = {"error": "Insufficient trading days", "n_days": len(long_short)}
                    continue
                
                sharpe = long_short.mean() / long_short.std() * np.sqrt(252) if long_short.std() > 0 else 0
                total_ret = (1 + long_short).prod() - 1
                
                results[regime] = {
                    "long_short": long_short,
                    "sharpe": sharpe,
                    "total_return": total_ret,
                    "n_days": len(long_short),
                }
            except Exception as e:
                results[regime] = {"error": str(e)}
        
        return results
    
    # Strategic benchmarks
    def build_heston_sentiment_ls(self, panel: pd.DataFrame) -> Dict[str, pd.Series]:
        df = panel.copy()
        if "sentiment" not in df.columns:
            return {}
        df["sentiment_lag"] = df.groupby(level=1)["sentiment"].shift(1)
        return self._build_daily_ls(df, "sentiment_lag", 10, ascending=True)
    
    def build_tetlock_ls(self, panel: pd.DataFrame) -> Dict[str, pd.Series]:
        df = panel.copy()
        if "sentiment" not in df.columns:
            return {}
        df["neg"] = df["sentiment"].apply(lambda x: max(-x, 0) if pd.notnull(x) else np.nan)
        df = df.sort_index(level=[1, 0])
        df["neg_roll"] = df.groupby(level=1)["neg"].transform(
            lambda x: x.rolling(window=60, min_periods=20).mean()
        )
        df["abn_neg"] = df["neg"] - df["neg_roll"]
        df["abn_neg_lag"] = df.groupby(level=1)["abn_neg"].shift(1)
        return self._build_daily_ls(df, "abn_neg_lag", 10, ascending=True)
    
    def build_momentum_ls(self, panel: pd.DataFrame) -> Dict[str, pd.Series]:
        df = panel.copy()
        df["ret_lag"] = df.groupby(level=1)["ret_fwd"].shift(1)
        lookback = 12 * 21
        skip = 1 * 21
        window = lookback - skip
        df["mom"] = df.groupby(level=1)["ret_lag"].transform(
            lambda x: x.shift(skip).rolling(window=window, min_periods=window//2).sum()
        )
        return self._build_daily_ls(df, "mom", 10, ascending=True)
    
    def build_risk_ls(self, panel: pd.DataFrame) -> Dict[str, pd.Series]:
        df = panel.copy()
        cols = ["beta_dyn", "vol_midas"]
        if not all(c in df.columns for c in cols):
            return {}
        for c in cols:
            df[c+"_z"] = df.groupby(level=0)[c].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
            )
        df["risk_score"] = df[[c+"_z" for c in cols]].mean(axis=1)
        return self._build_daily_ls(df, "risk_score", 10, ascending=True)
    
    def _build_daily_ls(self, df: pd.DataFrame, score_col: str, n_deciles: int, ascending: bool = True) -> Dict[str, pd.Series]:
        ls_returns = {}
        for date, group in df.groupby(level=0):
            if len(group) < self.design.min_assets_per_date:
                continue
            valid = group.dropna(subset=[score_col, "ret_excess"])
            if len(valid) < self.design.min_assets_per_date:
                continue
            sorted_group = valid.sort_values(score_col, ascending=ascending)
            n = len(sorted_group)
            n_bucket = max(1, int(n / n_deciles))
            shorts = sorted_group.iloc[:n_bucket]
            longs = sorted_group.iloc[-n_bucket:]
            ls_returns[date] = longs["ret_excess"].mean() - shorts["ret_excess"].mean()
        return {"ls": pd.Series(ls_returns).sort_index()}


# =============================================================================
# ABLATION TESTER
# =============================================================================

class AblationTesterV2:
    """Enhanced ablation with 8 configurations"""
    
    def __init__(self, model: EnhancedIntegratedModel, design: ResearchDesignV2):
        self.model = model
        self.design = design
    
    def run_ablation(self, period: str = "eval") -> pd.DataFrame:
        configs = [
            ("Baseline (Market Only)", False, False, False),
            ("Beta Only", True, False, False),
            ("Sentiment Only", False, True, False),
            ("MIDAS Vol Only", False, False, True),
            ("Beta + Sentiment", True, True, False),
            ("Beta + Vol", True, False, True),
            ("Sentiment + Vol", False, True, True),
            ("Full Model", True, True, True),
        ]
        
        results = []
        
        for name, use_beta, use_sent, use_vol in configs:
            design_copy = ResearchDesignV2(
                design_start=self.design.design_start,
                design_end=self.design.design_end,
                eval_start=self.design.eval_start,
                eval_end=self.design.eval_end,
                use_dynamic_beta=use_beta,
                use_sentiment=use_sent,
                use_midas_vol=use_vol,
                use_4regime=self.design.use_4regime and use_sent,
            )
            
            try:
                tester = CrossSectionalTesterV2(self.model, design_copy)
                panel = tester.build_panel(period=period)
                
                feats = []
                if use_beta: feats.append("beta_dyn")
                if use_sent: feats.append("sentiment")
                if use_vol: feats.append("vol_midas")
                
                if not feats:
                    results.append({"Configuration": name, "Alpha": np.nan, "T-Stat": np.nan, "P-Value": np.nan, "Sharpe": np.nan})
                    continue
                
                for f in feats:
                    if f in panel.columns:
                        mu = panel[f].mean()
                        sigma = panel[f].std()
                        if sigma > 0:
                            panel[f+"_z"] = (panel[f] - mu) / sigma
                        else:
                            panel[f+"_z"] = 0.0
                
                z_cols = [f+"_z" for f in feats if f+"_z" in panel.columns]
                panel["score"] = panel[z_cols].mean(axis=1)
                
                dpt = DecilePortfolioTesterV2(self.model, design_copy)
                _, long_short = dpt.decile_backtest(panel, score_col="score")
                
                validator = StatisticalValidator(self.model.config)
                val = validator.calculate_alpha_beta(long_short, self.model.market_returns.loc[long_short.index])
                
                sharpe = long_short.mean() / long_short.std() * np.sqrt(252) if long_short.std() > 0 else 0
                
                results.append({
                    "Configuration": name,
                    "Alpha": val['alpha_annual'],
                    "T-Stat": val['alpha_tstat'],
                    "P-Value": val['alpha_pval'],
                    "Sharpe": sharpe,
                })
                
            except Exception as e:
                print(f"Ablation error for {name}: {e}")
                results.append({"Configuration": name, "Alpha": np.nan, "T-Stat": np.nan, "P-Value": np.nan, "Sharpe": np.nan})
        
        return pd.DataFrame(results)


# =============================================================================
# MULTIPLE TESTING ADJUSTMENT
# =============================================================================

class MultipleTestingAdjustment:
    """FDR control for multiple testing"""
    
    @staticmethod
    def benjamini_hochberg(pvalues: pd.Series, alpha: float = 0.10) -> pd.DataFrame:
        n = len(pvalues)
        sorted_idx = pvalues.argsort()
        sorted_pvals = pvalues.iloc[sorted_idx]
        
        adjusted = np.zeros(n)
        for i, (orig_idx, pval) in enumerate(zip(sorted_idx, sorted_pvals)):
            rank = i + 1
            adjusted[orig_idx] = min(pval * n / rank, 1.0)
        
        adjusted_sorted = adjusted[sorted_idx]
        for i in range(n-2, -1, -1):
            adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i+1])
        adjusted[sorted_idx] = adjusted_sorted
        
        return pd.DataFrame({
            'original_pval': pvalues,
            'adjusted_pval': adjusted,
            'significant_fdr': adjusted < alpha
        })


# =============================================================================
# REGIME SENSITIVITY
# =============================================================================

class RegimeSensitivityAnalyzer:
    """Analyze sensitivity to regime thresholds"""
    
    def __init__(self, model: EnhancedIntegratedModel, base_design: ResearchDesignV2):
        self.model = model
        self.base_design = base_design
    
    def run_threshold_sensitivity(
        self,
        vol_thresholds: List[float] = [-0.5, -0.25, 0.0, 0.25, 0.5],
        period: str = "eval"
    ) -> pd.DataFrame:
        results = []
        
        for vol_th in vol_thresholds:
            design = ResearchDesignV2(
                design_start=self.base_design.design_start,
                design_end=self.base_design.design_end,
                eval_start=self.base_design.eval_start,
                eval_end=self.base_design.eval_end,
                vol_threshold=vol_th,
                use_4regime=True,
            )
            
            try:
                tester = CrossSectionalTesterV2(self.model, design)
                panel = tester.build_panel(period=period)
                fm_result = tester.fama_macbeth(panel)
                
                results.append({
                    'vol_threshold': vol_th,
                    'beta_coef': fm_result['mean_coefs'].get('beta_dyn', np.nan),
                    'beta_tstat': fm_result['tstats'].get('beta_dyn', np.nan),
                    'sent_LB_tstat': fm_result['tstats'].get('sent_LB', np.nan),
                    'sent_LU_tstat': fm_result['tstats'].get('sent_LU', np.nan),
                    'sent_HB_tstat': fm_result['tstats'].get('sent_HB', np.nan),
                    'sent_HU_tstat': fm_result['tstats'].get('sent_HU', np.nan),
                    'T': fm_result['T'],
                })
            except Exception as e:
                print(f"Error at vol_threshold={vol_th}: {e}")
        
        return pd.DataFrame(results)


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_research_v2(
    prices_file: str = "real_backtest_panel.csv",
    sentiment_file: str = "real_sentiment_data.csv",
    factor_file: Optional[str] = None,
    benchmark: str = "SPY",
    model_config: Optional[ModelConfig] = None,
    design: Optional[ResearchDesignV2] = None,
) -> Dict[str, Any]:
    """Complete research pipeline with Duke feedback"""
    
    config = model_config or ModelConfig()
    design = design or ResearchDesignV2()
    
    print("=" * 70)
    print("RESEARCH PIPELINE V2 - Duke Feedback Incorporated")
    print("=" * 70)
    
    model = EnhancedIntegratedModel(config)
    if not model.load_data(prices_file=prices_file, sentiment_file=sentiment_file, benchmark=benchmark):
        raise RuntimeError("Failed to load data")
    if not model.fit():
        raise RuntimeError("Failed to fit model")
    
    results = {"model": model, "design": design}
    
    print("\n[1/8] Building panels...")
    tester = CrossSectionalTesterV2(model, design)
    panel_full = tester.build_panel(period="full")
    panel_design = tester.build_panel(period="design")
    panel_eval = tester.build_panel(period="eval")
    
    results["panel_full"] = panel_full
    results["panel_design"] = panel_design
    results["panel_eval"] = panel_eval
    print(f"   Full: {len(panel_full)}, Design: {len(panel_design)}, Eval: {len(panel_eval)}")
    
    print("\n[2/8] Fama-MacBeth regressions...")
    fm_design = tester.fama_macbeth(panel_design)
    fm_eval = tester.fama_macbeth(panel_eval)
    fm_full = tester.fama_macbeth(panel_full)
    results["fama_macbeth_design"] = fm_design
    results["fama_macbeth_eval"] = fm_eval
    results["fama_macbeth_full"] = fm_full
    print(f"   Design T={fm_design['T']}, Eval T={fm_eval['T']}")
    
    print("\n[3/8] Decile backtest...")
    for f in ["beta_dyn", "sentiment", "vol_midas"]:
        if f in panel_eval.columns:
            mu = panel_eval[f].mean()
            sigma = panel_eval[f].std()
            if sigma > 0:
                panel_eval[f+"_z"] = (panel_eval[f] - mu) / sigma
    z_cols = [c for c in panel_eval.columns if c.endswith("_z") and not c.startswith("sent_")]
    panel_eval["score"] = panel_eval[z_cols].mean(axis=1)
    
    dpt = DecilePortfolioTesterV2(model, design)
    deciles, long_short = dpt.decile_backtest(panel_eval, score_col="score")
    results["decile_returns"] = deciles
    results["long_short"] = long_short
    print(f"   Long-Short: {len(long_short)} days")
    
    print("\n[4/8] Factor regressions...")
    fv = FactorValidatorV2(factor_file=factor_file, model=model)
    val_capm = fv.run_regression(long_short, model_type="CAPM", market_returns=model.market_returns)
    val_ff5mom = fv.run_regression(long_short, model_type="FF5+MOM", market_returns=model.market_returns)
    results["validation_capm"] = val_capm
    results["validation_ff5mom"] = val_ff5mom
    if "alpha_annual" in val_capm:
        print(f"   CAPM: {val_capm['alpha_annual']:.2%} (t={val_capm['alpha_tstat']:.2f})")
    if "alpha_annual" in val_ff5mom:
        print(f"   FF5+MOM: {val_ff5mom['alpha_annual']:.2%} (t={val_ff5mom['alpha_tstat']:.2f})")
    
    print("\n[5/8] Ablation study...")
    ablation_tester = AblationTesterV2(model, design)
    ablation_results = ablation_tester.run_ablation(period="eval")
    results["ablation"] = ablation_results
    print(ablation_results.to_string(index=False))
    
    print("\n[6/8] Regime-conditional analysis...")
    regime_results = dpt.regime_conditional_backtest(panel_eval, score_col="score")
    results["regime_conditional"] = regime_results
    for regime, res in regime_results.items():
        if "sharpe" in res:
            print(f"   {regime}: Sharpe={res['sharpe']:.3f}, N={res['n_days']}")
    
    print("\n[7/8] Regime sensitivity...")
    sensitivity = RegimeSensitivityAnalyzer(model, design)
    sensitivity_results = sensitivity.run_threshold_sensitivity(period="eval")
    results["regime_sensitivity"] = sensitivity_results
    
    print("\n[8/8] Strategic benchmarks...")
    strategic = {}
    strategic["Heston"] = dpt.build_heston_sentiment_ls(panel_eval)
    strategic["Tetlock"] = dpt.build_tetlock_ls(panel_eval)
    strategic["Momentum"] = dpt.build_momentum_ls(panel_eval)
    strategic["Risk"] = dpt.build_risk_ls(panel_eval)
    results["strategic_benchmarks"] = strategic
    
    print("\n" + "=" * 70)
    print("RESEARCH PIPELINE V2 COMPLETE")
    print("=" * 70)
    
    return results


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'ResearchDesignV2',
    'CrossSectionalTesterV2',
    'FactorValidatorV2',
    'DecilePortfolioTesterV2',
    'AblationTesterV2',
    'MultipleTestingAdjustment',
    'RegimeSensitivityAnalyzer',
    'run_research_v2',
]


if __name__ == "__main__":
    config = ModelConfig(transaction_cost_bps=8)
    design = ResearchDesignV2(use_4regime=True)
    
    results = run_research_v2(
        prices_file="real_backtest_panel.csv",
        sentiment_file="real_sentiment_data.csv",
        benchmark="SPY",
        model_config=config,
        design=design,
    )
