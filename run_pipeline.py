import os
import sys
import subprocess
import time

def run_command(command, description):
    print(f"\n{'='*70}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*70}")
    
    try:
        subprocess.check_call(command, shell=True)
        print(f"✅ {description} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}.")
        return False

def main():
    print("AI FINANCE PIPELINE RUNNER")
    print("==========================")
    
    # 1. Fetch Price Data
    if not os.path.exists("real_backtest_panel.csv"):
        if not run_command("python fetch_real_data.py", "Fetch Real Price Data"):
            return
    else:
        print("✅ Price data already exists.")
        
    # 2. Fetch Sentiment Data
    if not os.path.exists("real_sentiment_data.csv"):
        if not run_command("python multi_source_news_scraper.py", "Fetch Real Sentiment Data"):
            return
    else:
        print("✅ Sentiment data already exists.")
        
    # 3. Run Research Pipeline
    run_command("python research_pipeline_v2.py", "Research Pipeline")
    
    # 4. Launch Dashboard (Optional)
    print("\nTo view the dashboard, run:")
    print("streamlit run enhanced_dashboard_v4.py")

if __name__ == "__main__":
    main()
