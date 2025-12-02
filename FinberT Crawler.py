import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, timedelta
import time
import random

# =============================================================================
# 0. ASSET UNIVERSE DEFINITION (Dashboard와 동일하게 설정)
# =============================================================================
ASSET_UNIVERSE = {
    'Technology': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'AMD', 'QCOM', 'INTC'],
    'Healthcare': ['LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'TMO', 'ABT', 'PFE', 'AMGN', 'DHR'],
    'Financials': ['JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'BLK', 'C', 'AXP'],
    'Consumer': ['AMZN', 'TSLA', 'HD', 'PG', 'COST', 'WMT', 'KO', 'PEP', 'MCD', 'NKE'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL'],
    'Utilities': ['NEE', 'SO', 'DUK', 'CEG', 'AEP', 'SRE', 'PEG', 'ED', 'EXC', 'XEL'],
    'Bonds': ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'BND', 'AGG', 'MUB', 'TIP', 'BNDX'],
    'Commodities': ['GLD', 'SLV', 'USO', 'UNG', 'DBC', 'DBA', 'GSG', 'PALL', 'PPLT', 'CPER'],
    'Indices': ['SPY', 'QQQ', 'IWM', 'DIA'] # 벤치마크 ETF 추가
}

# 딕셔너리를 평탄화(Flatten)하여 전체 티커 리스트 생성
ALL_TICKERS = [ticker for sector in ASSET_UNIVERSE.values() for ticker in sector]

# =============================================================================
# 1. SETUP & CONFIGURATION
# =============================================================================

print("Loading FinBERT model... (GPU 가속이 가능하면 자동으로 사용합니다)")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
# device=0 (GPU) if available, else -1 (CPU)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def get_finviz_news(ticker):
    """
    Finviz.com에서 특정 티커의 뉴스 헤드라인을 BeautifulSoup으로 크롤링
    """
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    
    # User-Agent 랜덤화 (차단 회피 확률 증가)
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0'
    ]
    
    headers = {'User-Agent': random.choice(user_agents)}
    
    try:
        req = requests.get(url, headers=headers, timeout=10) # 타임아웃 설정
        if req.status_code != 200:
            return []
            
        soup = BeautifulSoup(req.content, 'html.parser')
        news_table = soup.find(id='news-table')
        
        if not news_table:
            return []
        
        parsed_data = []
        
        for row in news_table.findAll('tr'):
            title_tag = row.a
            if not title_tag: continue
            title = title_tag.text
            
            date_data = row.td.text.split()
            
            if len(date_data) == 1:
                time_str = date_data[0]
                date_str = 'same_day' # placeholder
            else:
                date_str = date_data[0]
                time_str = date_data[1]
                
            parsed_data.append([ticker, date_str, time_str, title])
            
        return parsed_data

    except Exception as e:
        print(f"⚠️ Error parsing {ticker}: {e}")
        return []

def process_news_data(parsed_news):
    """데이터 정리 및 날짜 포맷팅"""
    if not parsed_news:
        return pd.DataFrame()
        
    df = pd.DataFrame(parsed_news, columns=['ticker', 'date', 'time', 'title'])
    
    current_date = datetime.now().strftime('%b-%d-%y')
    
    dates = []
    for i in df['date']:
        if i == 'today':
            dates.append(datetime.now().strftime('%b-%d-%y'))
            current_date = datetime.now().strftime('%b-%d-%y')
        elif i == 'same_day':
            dates.append(current_date)
        elif len(i.split('-')) == 3:
            dates.append(i)
            current_date = i
        else:
            dates.append(current_date)
            
    df['date'] = dates
    
    # 시간 변환
    try:
        df['parsed_date'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%b-%d-%y %I:%M%p', errors='coerce')
    except:
        df['parsed_date'] = pd.to_datetime(df['date'], errors='coerce')
        
    return df.dropna(subset=['parsed_date'])

def apply_finbert(df):
    """FinBERT 감성 분석"""
    print(f"📊 Running FinBERT on {len(df)} headlines across {df['ticker'].nunique()} assets...")
    
    results = []
    # 속도를 위해 progress bar 없이 배치 처리가 좋지만, 
    # 사용자 피드백을 위해 간단히 진행상황 출력
    total = len(df)
    
    titles = df['title'].tolist()
    
    # HuggingFace pipeline은 리스트를 입력으로 받으면 배치 처리 효율이 좋음
    # (단, CPU/GPU 메모리에 따라 배치 사이즈 조절 필요할 수 있음)
    try:
        model_outputs = nlp(titles, batch_size=8, truncation=True, max_length=512)
        
        for output in model_outputs:
            label = output['label']
            score = output['score']
            
            if label == 'positive':
                final_score = score
            elif label == 'negative':
                final_score = -score
            else:
                final_score = 0
            results.append(final_score)
            
    except Exception as e:
        print(f"Error in batch processing: {e}. Falling back to loop.")
        for title in titles:
            try:
                res = nlp(title[:512])[0]
                label = res['label']
                score = res['score']
                if label == 'positive': val = score
                elif label == 'negative': val = -score
                else: val = 0
                results.append(val)
            except:
                results.append(0)

    df['sentiment_score'] = results
    return df

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print(f"🚀 Starting Large-Scale Scraping for {len(ALL_TICKERS)} tickers...")
    print(f"sectors: {list(ASSET_UNIVERSE.keys())}")
    
    all_news = []
    
    # 티커별 크롤링 루프
    for idx, ticker in enumerate(ALL_TICKERS):
        # 진행률 표시
        print(f"[{idx+1}/{len(ALL_TICKERS)}] Scraping {ticker}...", end="\r")
        
        raw_data = get_finviz_news(ticker)
        all_news.extend(raw_data)
        
        # ⚠️ 중요: Finviz 차단 방지를 위한 지연 (1.5 ~ 3초 랜덤)
        # 종목이 많으므로 너무 빠르면 IP 차단당함
        time.sleep(random.uniform(1.5, 3.0))
    
    print("\n✅ Scraping complete.")
    
    if not all_news:
        print("❌ No news collected. Finviz might be blocking requests.")
        return

    # 데이터 구조화
    df = process_news_data(all_news)
    print(f"📚 Collected {len(df)} total headlines.")
    
    # FinBERT 분석
    if not df.empty:
        df_analyzed = apply_finbert(df)
        
        # 1. 일별 통합 감성 (Market Sentiment)
        daily_sentiment = df_analyzed.groupby(df_analyzed['parsed_date'].dt.date)['sentiment_score'].mean().reset_index()
        daily_sentiment.columns = ['date', 'sentiment_score']
        daily_sentiment.sort_values('date', inplace=True)
        daily_sentiment.to_csv("real_sentiment_data.csv", index=False)
        
        # 2. 섹터별 감성 (Sector Sentiment) - 심화 분석용
        # 티커에 섹터 정보 매핑
        ticker_to_sector = {t: s for s, tickers in ASSET_UNIVERSE.items() for t in tickers}
        df_analyzed['sector'] = df_analyzed['ticker'].map(ticker_to_sector)
        
        sector_sentiment = df_analyzed.groupby([df_analyzed['parsed_date'].dt.date, 'sector'])['sentiment_score'].mean().reset_index()
        sector_sentiment.to_csv("real_sector_sentiment.csv", index=False)
        
        # 3. 원본 저장
        df_analyzed.to_csv("real_news_raw_full.csv", index=False)
        
        print("\n🎉 ALL DONE!")
        print("Files created:")
        print("  - real_sentiment_data.csv (For Model Input)")
        print("  - real_sector_sentiment.csv (For Sector Analysis)")
        print("  - real_news_raw_full.csv (Raw Data)")
    else:
        print("No valid news data to analyze.")

if __name__ == "__main__":
    main()