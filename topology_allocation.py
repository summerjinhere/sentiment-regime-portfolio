"""
Topology-Based Asset Allocation
================================

Correlation → Distance → Hierarchical Clustering → HRP

Methods:
1. HRP (Hierarchical Risk Parity) - Lopez de Prado (2016)
2. HERC (Hierarchical Equal Risk Contribution)
3. Network Clustering (MST-based)

References:
- Lopez de Prado, M. (2016). Building Diversified Portfolios that Outperform Out of Sample
- Mantegna, R.N. (1999). Hierarchical Structure in Financial Markets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, fcluster
    from scipy.spatial.distance import squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False


@dataclass
class TopologyConfig:
    """Configuration for topology-based allocation"""
    method: str = "hrp"  # hrp, herc, network
    distance_metric: str = "correlation"  # correlation, angular, abs_correlation
    linkage_method: str = "ward"  # single, complete, average, ward
    risk_measure: str = "variance"  # variance, std, mad
    min_cluster_size: int = 2
    max_clusters: Optional[int] = None
    

# =============================================================================
# DISTANCE METRICS
# =============================================================================

def correlation_to_distance(corr: pd.DataFrame, method: str = "correlation") -> pd.DataFrame:
    """
    Convert correlation matrix to distance matrix.
    
    Methods:
    - correlation: d = sqrt(0.5 * (1 - corr))  [Lopez de Prado]
    - angular: d = sqrt(2 * (1 - corr))  [Angular distance]
    - abs_correlation: d = sqrt(1 - |corr|)  [Absolute correlation]
    """
    if method == "correlation":
        # Lopez de Prado distance
        dist = np.sqrt(0.5 * (1 - corr))
    elif method == "angular":
        # Angular distance
        dist = np.sqrt(2 * (1 - corr))
    elif method == "abs_correlation":
        # Absolute correlation distance
        dist = np.sqrt(1 - np.abs(corr))
    else:
        raise ValueError(f"Unknown distance method: {method}")
    
    # Ensure diagonal is 0
    np.fill_diagonal(dist.values, 0)
    
    return pd.DataFrame(dist, index=corr.index, columns=corr.columns)


# =============================================================================
# HIERARCHICAL CLUSTERING
# =============================================================================

def get_quasi_diagonal(link: np.ndarray) -> List[int]:
    """
    Sort clustered items by distance (quasi-diagonalization).
    Returns the order of assets that minimizes off-diagonal correlations.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for clustering")
    
    return list(leaves_list(link))


def get_cluster_variance(cov: pd.DataFrame, cluster_items: List[str]) -> float:
    """Calculate variance of an equal-weight portfolio within a cluster."""
    cov_slice = cov.loc[cluster_items, cluster_items]
    n = len(cluster_items)
    w = np.ones(n) / n
    return float(w @ cov_slice.values @ w)


def get_inverse_variance_weights(cov: pd.DataFrame, assets: List[str]) -> pd.Series:
    """Calculate inverse variance weights for a set of assets."""
    variances = np.diag(cov.loc[assets, assets].values)
    inv_var = 1 / variances
    weights = inv_var / inv_var.sum()
    return pd.Series(weights, index=assets)


# =============================================================================
# HRP (HIERARCHICAL RISK PARITY)
# =============================================================================

class HRPAllocator:
    """
    Hierarchical Risk Parity (Lopez de Prado, 2016)
    
    Steps:
    1. Compute correlation matrix
    2. Convert to distance matrix
    3. Hierarchical clustering
    4. Quasi-diagonalization
    5. Recursive bisection for weights
    """
    
    def __init__(self, config: TopologyConfig = None):
        self.config = config or TopologyConfig()
        self.corr = None
        self.cov = None
        self.dist = None
        self.link = None
        self.sorted_assets = None
        self.weights = None
    
    def fit(self, returns: pd.DataFrame) -> 'HRPAllocator':
        """Fit the HRP model to return data."""
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required")
        
        # Filter out zero-variance columns (e.g., delisted stocks like ATVI)
        variances = returns.var()
        zero_var_cols = variances[variances < 1e-10].index.tolist()
        if zero_var_cols:
            print(f"⚠️ Dropping {len(zero_var_cols)} zero-variance assets: {zero_var_cols}")
            returns = returns.drop(columns=zero_var_cols)
        
        # Also drop columns with too many NaNs
        nan_pct = returns.isna().mean()
        high_nan_cols = nan_pct[nan_pct > 0.5].index.tolist()
        if high_nan_cols:
            print(f"⚠️ Dropping {len(high_nan_cols)} high-NaN assets: {high_nan_cols}")
            returns = returns.drop(columns=high_nan_cols)
        
        # Fill remaining NaNs
        returns = returns.fillna(0)
        
        self.valid_assets = returns.columns.tolist()
        
        # Compute covariance and correlation
        self.cov = returns.cov()
        self.corr = returns.corr()
        
        # Check for NaN in correlation
        if self.corr.isna().any().any():
            print("⚠️ NaN in correlation matrix, replacing with 0")
            self.corr = self.corr.fillna(0)
        
        # Convert to distance
        self.dist = correlation_to_distance(self.corr, self.config.distance_metric)
        
        # Hierarchical clustering
        dist_condensed = squareform(self.dist.values)
        self.link = linkage(dist_condensed, method=self.config.linkage_method)
        
        # Quasi-diagonalization
        sort_idx = get_quasi_diagonal(self.link)
        self.sorted_assets = [returns.columns[i] for i in sort_idx]
        
        return self
    
    def get_weights(self) -> pd.Series:
        """Calculate HRP weights using recursive bisection."""
        if self.sorted_assets is None:
            raise ValueError("Must call fit() first")
        
        weights = pd.Series(1.0, index=self.sorted_assets)
        cluster_items = [self.sorted_assets]
        
        while len(cluster_items) > 0:
            # Bisect each cluster
            new_clusters = []
            for cluster in cluster_items:
                if len(cluster) > 1:
                    # Split in half
                    mid = len(cluster) // 2
                    left = cluster[:mid]
                    right = cluster[mid:]
                    
                    # Calculate cluster variances
                    var_left = get_cluster_variance(self.cov, left)
                    var_right = get_cluster_variance(self.cov, right)
                    
                    # Avoid division by zero
                    total_var = var_left + var_right
                    if total_var < 1e-10:
                        alpha = 0.5
                    else:
                        alpha = 1 - var_left / total_var
                    
                    # Update weights
                    weights[left] *= alpha
                    weights[right] *= (1 - alpha)
                    
                    # Add to new clusters if further splitting needed
                    if len(left) > 1:
                        new_clusters.append(left)
                    if len(right) > 1:
                        new_clusters.append(right)
            
            cluster_items = new_clusters
        
        self.weights = weights / weights.sum()
        return self.weights
    
    def get_cluster_info(self) -> Dict:
        """Return clustering information for visualization."""
        return {
            'sorted_assets': self.sorted_assets,
            'linkage': self.link,
            'distance_matrix': self.dist,
            'correlation_matrix': self.corr
        }


# =============================================================================
# HERC (HIERARCHICAL EQUAL RISK CONTRIBUTION)
# =============================================================================

class HERCAllocator:
    """
    Hierarchical Equal Risk Contribution
    
    Similar to HRP but aims for equal risk contribution
    at each level of the hierarchy.
    """
    
    def __init__(self, config: TopologyConfig = None):
        self.config = config or TopologyConfig()
        self.hrp = HRPAllocator(config)
        self.cluster_labels = None
    
    def fit(self, returns: pd.DataFrame, n_clusters: int = None) -> 'HERCAllocator':
        """Fit HERC model."""
        self.hrp.fit(returns)
        
        # Determine number of clusters
        if n_clusters is None:
            n_clusters = max(2, len(returns.columns) // 5)
        
        # Get cluster assignments
        self.cluster_labels = fcluster(self.hrp.link, n_clusters, criterion='maxclust')
        
        return self
    
    def get_weights(self) -> pd.Series:
        """Calculate HERC weights."""
        assets = self.hrp.sorted_assets
        cov = self.hrp.cov
        
        # Group assets by cluster
        clusters = {}
        for i, asset in enumerate(assets):
            label = self.cluster_labels[list(self.hrp.corr.columns).index(asset)]
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(asset)
        
        # Calculate intra-cluster weights (inverse variance)
        intra_weights = {}
        cluster_variances = {}
        
        for label, cluster_assets in clusters.items():
            w = get_inverse_variance_weights(cov, cluster_assets)
            intra_weights[label] = w
            
            # Cluster variance
            cluster_variances[label] = get_cluster_variance(cov, cluster_assets)
        
        # Inter-cluster weights (inverse variance)
        total_inv_var = sum(1/v for v in cluster_variances.values())
        inter_weights = {label: (1/v) / total_inv_var 
                        for label, v in cluster_variances.items()}
        
        # Combine
        final_weights = pd.Series(index=assets, dtype=float)
        for label, cluster_assets in clusters.items():
            for asset in cluster_assets:
                final_weights[asset] = inter_weights[label] * intra_weights[label][asset]
        
        return final_weights / final_weights.sum()


# =============================================================================
# NETWORK-BASED ALLOCATION (MST)
# =============================================================================

class NetworkAllocator:
    """
    Network-based allocation using Minimum Spanning Tree.
    
    Based on Mantegna (1999) - Hierarchical Structure in Financial Markets
    """
    
    def __init__(self, config: TopologyConfig = None):
        self.config = config or TopologyConfig()
        self.mst = None
        self.centrality = None
    
    def fit(self, returns: pd.DataFrame) -> 'NetworkAllocator':
        """Build MST from correlation matrix."""
        if not NX_AVAILABLE:
            raise ImportError("networkx required for network allocation")
        
        corr = returns.corr()
        dist = correlation_to_distance(corr, self.config.distance_metric)
        
        # Create complete graph
        G = nx.Graph()
        assets = list(corr.columns)
        
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i < j:
                    G.add_edge(asset1, asset2, weight=dist.loc[asset1, asset2])
        
        # Minimum Spanning Tree
        self.mst = nx.minimum_spanning_tree(G)
        
        # Calculate centrality measures
        self.centrality = {
            'degree': nx.degree_centrality(self.mst),
            'betweenness': nx.betweenness_centrality(self.mst),
            'closeness': nx.closeness_centrality(self.mst),
        }
        
        self.cov = returns.cov()
        
        return self
    
    def get_weights(self, centrality_type: str = 'betweenness') -> pd.Series:
        """
        Calculate weights based on network centrality.
        
        Less central = more diversified = higher weight
        """
        centrality = self.centrality[centrality_type]
        
        # Inverse centrality (peripheral assets get more weight)
        inv_centrality = {k: 1 / (v + 0.01) for k, v in centrality.items()}
        
        # Also consider variance
        variances = np.diag(self.cov.values)
        inv_var = 1 / variances
        
        # Combine: peripheral + low variance
        assets = list(self.cov.columns)
        combined = pd.Series(index=assets, dtype=float)
        
        for i, asset in enumerate(assets):
            combined[asset] = inv_centrality[asset] * inv_var[i]
        
        return combined / combined.sum()
    
    def get_mst_edges(self) -> List[Tuple[str, str, float]]:
        """Return MST edges for visualization."""
        return [(u, v, d['weight']) for u, v, d in self.mst.edges(data=True)]


# =============================================================================
# UNIFIED TOPOLOGY ALLOCATOR
# =============================================================================

class TopologyAllocator:
    """
    Unified interface for topology-based allocation methods.
    """
    
    def __init__(self, config: TopologyConfig = None):
        self.config = config or TopologyConfig()
        self.allocator = None
        self.weights = None
        self.returns = None
    
    def fit(self, returns: pd.DataFrame) -> 'TopologyAllocator':
        """Fit the allocator based on config method."""
        self.returns = returns
        
        if self.config.method == "hrp":
            self.allocator = HRPAllocator(self.config)
        elif self.config.method == "herc":
            self.allocator = HERCAllocator(self.config)
        elif self.config.method == "network":
            self.allocator = NetworkAllocator(self.config)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")
        
        self.allocator.fit(returns)
        return self
    
    def get_weights(self) -> pd.Series:
        """Get portfolio weights."""
        self.weights = self.allocator.get_weights()
        return self.weights
    
    def get_portfolio_stats(self, weights: pd.Series = None) -> Dict:
        """Calculate portfolio statistics."""
        if weights is None:
            weights = self.weights
        
        cov = self.returns.cov()
        expected_ret = self.returns.mean()
        
        port_ret = (weights * expected_ret).sum() * 252
        port_vol = np.sqrt(weights @ cov @ weights) * np.sqrt(252)
        sharpe = port_ret / port_vol if port_vol > 0 else 0
        
        # Diversification ratio
        asset_vols = np.sqrt(np.diag(cov.values)) * np.sqrt(252)
        weighted_vol = (weights * asset_vols).sum()
        div_ratio = weighted_vol / port_vol if port_vol > 0 else 1
        
        # Effective N (concentration)
        hhi = (weights ** 2).sum()
        eff_n = 1 / hhi if hhi > 0 else len(weights)
        
        return {
            'expected_return': port_ret,
            'volatility': port_vol,
            'sharpe_ratio': sharpe,
            'diversification_ratio': div_ratio,
            'effective_n': eff_n,
            'max_weight': weights.max(),
            'min_weight': weights.min(),
            'n_assets': len(weights),
        }
    
    def compare_methods(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compare all allocation methods."""
        results = []
        
        for method in ['hrp', 'herc', 'network']:
            try:
                config = TopologyConfig(method=method)
                allocator = TopologyAllocator(config)
                allocator.fit(returns)
                weights = allocator.get_weights()
                stats = allocator.get_portfolio_stats(weights)
                stats['method'] = method
                results.append(stats)
            except Exception as e:
                print(f"Error with {method}: {e}")
        
        # Add equal weight benchmark
        n = len(returns.columns)
        eq_weights = pd.Series(1/n, index=returns.columns)
        self.returns = returns
        eq_stats = self.get_portfolio_stats(eq_weights)
        eq_stats['method'] = 'equal_weight'
        results.append(eq_stats)
        
        return pd.DataFrame(results).set_index('method')


# =============================================================================
# INTEGRATION WITH EXISTING MODEL
# =============================================================================

def integrate_with_model(model, returns: pd.DataFrame, method: str = "hrp") -> pd.Series:
    """
    Integrate topology allocation with existing 3-layer model.
    
    Uses model signals to adjust topology-based weights.
    """
    # Get base topology weights
    config = TopologyConfig(method=method)
    allocator = TopologyAllocator(config)
    allocator.fit(returns)
    base_weights = allocator.get_weights()
    
    # Get model signals (if available)
    if hasattr(model, 'layer1_factor') and model.layer1_factor.betas:
        # Adjust by dynamic beta (lower beta = higher weight in risk parity sense)
        betas = model.layer1_factor.get_current_betas()
        beta_adj = pd.Series({t: 1/max(abs(betas.get(t, 1)), 0.1) for t in base_weights.index})
        beta_adj = beta_adj / beta_adj.sum()
        
        # Blend: 70% topology, 30% beta adjustment
        adjusted_weights = 0.7 * base_weights + 0.3 * beta_adj
    else:
        adjusted_weights = base_weights
    
    # Apply sentiment tilt (if available)
    if hasattr(model, 'layer2_sentiment') and model.layer2_sentiment.loaded:
        date = returns.index[-1]
        for ticker in adjusted_weights.index:
            sent = model.layer2_sentiment.get_signal(date, ticker)
            # Small tilt based on sentiment
            adjusted_weights[ticker] *= (1 + 0.1 * sent)
    
    # Normalize
    adjusted_weights = adjusted_weights / adjusted_weights.sum()
    
    return adjusted_weights


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_assets = 20
    n_days = 500
    
    # Correlated returns
    base = np.random.randn(n_days, 3)
    returns_data = np.zeros((n_days, n_assets))
    
    for i in range(n_assets):
        factor_idx = i % 3
        returns_data[:, i] = 0.5 * base[:, factor_idx] + 0.5 * np.random.randn(n_days)
    
    returns_data *= 0.02  # Scale to realistic daily returns
    
    tickers = [f"ASSET_{i}" for i in range(n_assets)]
    returns = pd.DataFrame(returns_data, columns=tickers)
    
    print("=" * 60)
    print("TOPOLOGY-BASED ALLOCATION TEST")
    print("=" * 60)
    
    # Test HRP
    print("\n[HRP] Hierarchical Risk Parity")
    hrp = TopologyAllocator(TopologyConfig(method="hrp"))
    hrp.fit(returns)
    hrp_weights = hrp.get_weights()
    hrp_stats = hrp.get_portfolio_stats()
    print(f"   Top 5 weights: {hrp_weights.nlargest(5).to_dict()}")
    print(f"   Sharpe: {hrp_stats['sharpe_ratio']:.3f}")
    print(f"   Effective N: {hrp_stats['effective_n']:.1f}")
    
    # Compare all methods
    print("\n[COMPARISON] All Methods")
    comparison = hrp.compare_methods(returns)
    print(comparison[['expected_return', 'volatility', 'sharpe_ratio', 'effective_n']].round(3))
