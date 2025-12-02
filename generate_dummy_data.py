import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_data():
    np.random.seed(42)
    n_days = 500
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')
    
    # 1. Generate real_backtest_panel.csv
    # Columns: date, ret_asset_0..9, factor_0..2, VIX, TERM_SPREAD, CREDIT_SPREAD, sentiment
    
    data = {'date': dates}
    
    # Assets
    for i in range(10):
        data[f'ret_asset_{i}'] = np.random.normal(0.0005, 0.015, n_days)
        
    # Factors
    for i in range(3):
        data[f'factor_{i}'] = np.random.normal(0.0004, 0.012, n_days)
        
    # Macro
    data['VIX'] = np.random.normal(20, 5, n_days)
    data['TERM_SPREAD'] = np.random.normal(1.5, 0.5, n_days)
    data['CREDIT_SPREAD'] = np.random.normal(2.0, 0.3, n_days)
    data['sentiment'] = np.random.normal(0, 0.5, n_days)
    
    df_panel = pd.DataFrame(data)
    df_panel.to_csv('real_backtest_panel.csv', index=False)
    print("Generated real_backtest_panel.csv")
    
    # 2. Generate real_backtest_results.csv
    # Columns: date, strategy_ret, benchmark_ret
    
    df_results = pd.DataFrame({
        'date': dates,
        'strategy_ret': np.random.normal(0.0008, 0.010, n_days),
        'benchmark_ret': np.random.normal(0.0004, 0.012, n_days)
    })
    
    df_results.to_csv('real_backtest_results.csv', index=False)
    print("Generated real_backtest_results.csv")

if __name__ == "__main__":
    generate_data()
