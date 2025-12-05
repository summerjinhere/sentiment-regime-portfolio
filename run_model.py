from enhanced_integrated_model_v3 import EnhancedIntegratedModel, ModelConfig
import pandas as pd
import numpy as np

def main():
    print("="*70)
    print("RUNNING ENHANCED INTEGRATED MODEL")
    print("="*70)
    
    # Config
    config = ModelConfig(
        transaction_cost_bps=8.0,
        risk_free_rate=0.03,
        vol_threshold=0.0,
        market_direction_window=20
    )
    
    # Initialize
    model = EnhancedIntegratedModel(config)
    
    # Load Data
    print("\nLoading data...")
    success = model.load_data(
        prices_file='real_backtest_panel.csv',
        sentiment_file='real_sentiment_data.csv', # Might be missing, handled by model
        trading_days=252*2 # 2 years
    )
    
    if not success:
        print("❌ Failed to load data")
        return
    
    # Fit
    print("\nFitting model...")
    model.fit()
    
    # Backtest
    print("\nRunning backtest...")
    weights = model.get_optimal_weights(method='hrp')
    returns = model.run_backtest(weights)
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    total_ret = (1 + returns).prod() - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    
    print(f"Total Return: {total_ret:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    
    summary = model.summary()
    print("\nModel Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
        
    print("\n✅ Model execution successful")

if __name__ == "__main__":
    main()
