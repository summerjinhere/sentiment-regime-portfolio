import pandas as pd
import numpy as np

def check_data():
    print("Loading real_backtest_panel.csv...")
    try:
        df = pd.read_csv('real_backtest_panel.csv', index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print(f"Data shape: {df.shape}")
    
    ret_cols = [c for c in df.columns if c.endswith('_ret')]
    returns = df[ret_cols]
    print(f"Returns shape: {returns.shape}")
    
    # Check for NaNs
    nan_counts = returns.isna().sum()
    if nan_counts.sum() > 0:
        print("NaNs found in returns:")
        print(nan_counts[nan_counts > 0])
        returns = returns.fillna(0)
        print("Filled NaNs with 0")
    
    # Check for Infs
    inf_counts = np.isinf(returns).sum()
    if inf_counts.sum() > 0:
        print("Infs found in returns:")
        print(inf_counts[inf_counts > 0])
        returns = returns.replace([np.inf, -np.inf], 0)
        print("Replaced Infs with 0")

    # Check for zero variance
    stds = returns.std()
    zero_var_cols = stds[stds == 0].index
    if len(zero_var_cols) > 0:
        print("Columns with zero variance found:")
        print(zero_var_cols)
        # Drop them for testing
        returns = returns.drop(columns=zero_var_cols)
        print(f"Dropped {len(zero_var_cols)} columns. New shape: {returns.shape}")

    # Calculate covariance
    print("Calculating covariance matrix...")
    cov = returns.cov()
    
    if cov.isna().sum().sum() > 0:
        print("NaNs found in covariance matrix!")
        print(cov.isna().sum()[cov.isna().sum() > 0])
    
    # Check symmetry
    if not np.allclose(cov, cov.T, equal_nan=True):
        print("Covariance matrix is NOT symmetric!")
    else:
        print("Covariance matrix is symmetric.")

    # Calculate correlation
    print("Calculating correlation matrix...")
    corr = returns.corr()
    if corr.isna().sum().sum() > 0:
        print("NaNs found in correlation matrix!")
        print(corr.isna().sum()[corr.isna().sum() > 0])
        # Fill NaNs for testing
        corr = corr.fillna(0)
        
    # Test squareform
    print("Testing squareform on distance matrix...")
    try:
        from scipy.spatial.distance import squareform
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist.values, 0)
        
        # Check dist symmetry
        if not np.allclose(dist, dist.T, equal_nan=True):
            print("Distance matrix is NOT symmetric!")
            diff = dist - dist.T
            print(f"Max asymmetry: {np.max(np.abs(diff))}")
            
        squareform(dist.values)
        print("squareform passed.")
    except Exception as e:
        print(f"squareform failed: {e}")

if __name__ == "__main__":
    check_data()
