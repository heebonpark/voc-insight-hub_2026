import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

def extract_keywords(texts, top_n=10):
    """TF-IDF 기반 핵심 키워드 추출"""
    # 불용어(stop words) 설정이 필요할 수 있음
    vectorizer = TfidfVectorizer(max_features=100)
    try:
        X = vectorizer.fit_transform(texts)
        indices = np.argsort(vectorizer.idf_)[::-1]
        features = vectorizer.get_feature_names_out()
        top_features = [features[i] for i in indices[:top_n]]
        return top_features
    except Exception:
        return []

def cluster_vocs(texts, n_clusters=3):
    """K-Means 군집화로 불만 유형 분류"""
    if len(texts) < n_clusters:
        return ["Unknown"] * len(texts)
        
    vectorizer = TfidfVectorizer(max_features=100)
    try:
        X = vectorizer.fit_transform(texts)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)
        return [f"유형-{lbl+1}" for lbl in kmeans.labels_]
    except Exception:
        return ["Error"] * len(texts)

def detect_anomalies(df, date_col='접수일시'):
    """Isolation Forest를 활용한 일별 VOC 이상 급증 탐지"""
    if date_col not in df.columns:
        return pd.DataFrame()
        
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        daily_counts = df.groupby(df[date_col].dt.date).size().reset_index(name='count')
        daily_counts.dropna(inplace=True)
        
        if len(daily_counts) < 5:
            return daily_counts
            
        model = IsolationForest(contamination=0.1, random_state=42)
        daily_counts['anomaly'] = model.fit_predict(daily_counts[['count']])
        # -1 indicates anomaly
        daily_counts['is_anomaly'] = daily_counts['anomaly'] == -1
        return daily_counts
    except Exception:
        return pd.DataFrame()

# 딥러닝 모델 (HuggingFace Transformers)
# 초기 로드 시 시간이 걸릴 수 있으므로 캐싱 사용이 권장됨
from transformers import pipeline

def load_sentiment_model():
    """감성 분석 모델 로드 (KoBERT 등 한국어 모델)"""
    # 여기서는 범용 다국어 모델 사용 (실 서비스 시 한국어 전용 모델 사용 권장: 'monologg/koelectra-base-v3-discriminator' 등)
    # 메모리 문제 방지를 위해 경량 모델 사용
    try:
        sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
        return sentiment_analyzer
    except Exception as e:
        print(f"Model load error: {e}")
        return None

def analyze_sentiment(texts, model):
    """VOC 내용 딥러닝 감성 분석"""
    if not model or len(texts) == 0:
        return ["Neutral"] * len(texts)
    
    results = []
    # 배치 처리
    try:
        outputs = model(list(texts), truncation=True, max_length=512)
        for out in outputs:
            # nlptown model returns 1 star to 5 stars
            label = out['label']
            if '1 star' in label or '2 star' in label:
                results.append('Negative')
            elif '3 star' in label:
                results.append('Neutral')
            else:
                results.append('Positive')
        return results
    except Exception:
        return ["Error"] * len(texts)
