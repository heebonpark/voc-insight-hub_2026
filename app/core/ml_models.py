import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import IsolationForest

# VOC 도메인 감성 어휘
_NEG = [
    '불만', '불편', '짜증', '화남', '최악', '열받', '기다', '안됨', '안되', '늦음', '느림',
    '고장', '오류', '에러', '버그', '먹통', '끊김', '끊기', '중단', '미흡', '형편없',
    '엉망', '황당', '실망', '답답', '억울', '피해', '손해', '하자', '결함', '불량',
    '문제', '취소', '환불', '배상', '보상', '항의', '클레임', '신고', '처리안', '해결안',
    '거짓', '허위', '위반', '불법', '강요', '폭언', '무시', '불친절', '무례', '불성실',
    '나쁨', '나쁘', '심각', '과금', '오청구', '연결안', '전화안', '연락안', '미처리',
    '지연', '지체', '늑장', '무응답', '불이행', '약속위반',
]
_POS = [
    '감사', '고맙', '친절', '빠름', '해결', '완료', '만족', '좋음', '좋아', '훌륭',
    '최고', '칭찬', '잘됨', '성실', '정확', '신속', '완벽', '편리', '도움',
]


def _tokenize(texts: list) -> list[list[str]]:
    """kiwipiepy 명사 추출 (없으면 공백 분리로 fallback)"""
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        result = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                result.append([])
                continue
            analyzed = kiwi.analyze(text[:500])
            nouns = [t.form for t in analyzed[0][0]
                     if t.tag in ('NNG', 'NNP') and len(t.form) >= 2]
            result.append(nouns)
        return result
    except Exception:
        return [t.split() if isinstance(t, str) else [] for t in texts]


def _vectorize(texts: list, max_features: int = 300, min_df: int = 2):
    tokenized = _tokenize(texts)
    joined = [' '.join(t) for t in tokenized]
    non_empty = [t for t in joined if t.strip()]
    if len(non_empty) < 3:
        return None, None, None
    vec = TfidfVectorizer(max_features=max_features, min_df=min_df)
    X = vec.fit_transform(non_empty)
    return vec, X, non_empty


def extract_keywords(texts: list, top_n: int = 20) -> list[tuple]:
    """TF-IDF 평균 점수 기반 핵심 키워드 추출"""
    if not texts:
        return []
    try:
        vec, X, _ = _vectorize(texts)
        if vec is None:
            return []
        mean_scores = X.mean(axis=0).A1
        features = vec.get_feature_names_out()
        idx = np.argsort(mean_scores)[::-1]
        return [(features[i], round(float(mean_scores[i]), 4)) for i in idx[:top_n]]
    except Exception:
        return []


def extract_topics_lda(texts: list, n_topics: int = 5) -> list[dict]:
    """LDA 토픽 모델링"""
    if len(texts) < 10:
        return []
    try:
        vec, X, _ = _vectorize(texts, max_features=500, min_df=2)
        if vec is None or X.shape[0] < 10:
            return []
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=30)
        lda.fit(X)
        features = vec.get_feature_names_out()
        topics = []
        for i, comp in enumerate(lda.components_):
            top_words = [features[j] for j in comp.argsort()[-10:][::-1]]
            topics.append({
                'topic': f'토픽 {i + 1}',
                'keywords': top_words,
                'label': ' / '.join(top_words[:3]),
            })
        return topics
    except Exception:
        return []


def cluster_vocs(texts: list, n_clusters: int = 4) -> list[str]:
    """K-Means 군집화"""
    if len(texts) < n_clusters:
        return ['미분류'] * len(texts)
    try:
        vec, X, non_empty = _vectorize(texts, min_df=1)
        if vec is None:
            return ['미분류'] * len(texts)
        n = min(n_clusters, X.shape[0])
        km = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        label_map = {i: f'유형-{i + 1}' for i in range(n)}
        result = []
        ne_idx = 0
        for text in texts:
            if isinstance(text, str) and text.strip():
                result.append(label_map.get(labels[ne_idx], '미분류'))
                ne_idx += 1
            else:
                result.append('미분류')
        return result
    except Exception:
        return ['오류'] * len(texts)


def detect_anomalies(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """IQR + Isolation Forest 이중 이상 탐지"""
    if date_col not in df.columns:
        return pd.DataFrame()
    try:
        tmp = df.copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors='coerce')
        daily = tmp.groupby(tmp[date_col].dt.date).size().reset_index(name='count')
        daily = daily.dropna()
        daily[date_col] = pd.to_datetime(daily[date_col])

        if len(daily) < 5:
            daily['is_anomaly'] = False
            return daily

        q1, q3 = daily['count'].quantile([0.25, 0.75])
        iqr_upper = q3 + 1.5 * (q3 - q1)
        iqr_flag = daily['count'] > iqr_upper

        iso = IsolationForest(contamination=0.1, random_state=42)
        iso_flag = iso.fit_predict(daily[['count']]) == -1

        daily['is_anomaly'] = iqr_flag | iso_flag
        return daily
    except Exception:
        return pd.DataFrame()


def analyze_sentiment_rule(texts: list) -> list[str]:
    """규칙 기반 한국어 감성 분석 (빠름, 의존성 없음)"""
    results = []
    for text in texts:
        if not isinstance(text, str):
            results.append('중립')
            continue
        neg = sum(1 for w in _NEG if w in text)
        pos = sum(1 for w in _POS if w in text)
        if neg > pos:
            results.append('부정')
        elif pos > neg:
            results.append('긍정')
        else:
            results.append('중립')
    return results


def load_sentiment_model():
    """HuggingFace 감성 모델 로드 (lazy import)"""
    try:
        from transformers import pipeline  # noqa: lazy
        return pipeline(
            'sentiment-analysis',
            model='nlptown/bert-base-multilingual-uncased-sentiment',
        )
    except Exception as e:
        print(f'Model load error: {e}')
        return None


def analyze_sentiment(texts: list, model) -> list[str]:
    """딥러닝 감성 분석"""
    if not model or not texts:
        return ['중립'] * len(texts)
    try:
        outputs = model(list(texts), truncation=True, max_length=512)
        mapping = {'1 star': '부정', '2 stars': '부정', '3 stars': '중립',
                   '4 stars': '긍정', '5 stars': '긍정'}
        return [mapping.get(o['label'], '중립') for o in outputs]
    except Exception:
        return ['오류'] * len(texts)


import re as _re

# 리스크·긴급도 판단용 패턴
_RISK_EMOTION = _re.compile(r'감성|불만|해지|항의')
_RISK_URGENT  = _re.compile(r'빠른연락|빠른|긴급|즉시|지금당장')
_EM_HIGH      = _re.compile(r'감성불만|감성')
_EM_MED       = _re.compile(r'불만|항의')
_EM_CHURN     = _re.compile(r'해지|미연락|미방문')
_EM_URGENT    = _re.compile(r'긴급|비일|즉시')

_TERM_STATUS  = {'일반해지', '명의해지', '직권해지', '해지'}


def compute_em_score(text: str) -> int:
    """감성/긴급도 점수 (0–10). HTML v3의 emScore() 이식"""
    if not isinstance(text, str):
        return 0
    score = 0
    if _EM_HIGH.search(text):
        score += 4
    elif _EM_MED.search(text):
        score += 2
    if _RISK_URGENT.search(text):
        score += 2
    if _EM_CHURN.search(text):
        score += 2
    if _EM_URGENT.search(text):
        score += 1
    return min(score, 10)


def compute_risk_score(row: dict) -> int:
    """VOC 리스크 점수 (0–30). HTML v3의 calcRisk() 이식"""
    score = 0
    text = str(row.get('등록내용', '') or '')

    if _RISK_EMOTION.search(text):
        score += 6
    if _RISK_URGENT.search(text):
        score += 3

    if row.get('상태') == '미접수':
        score += 4
    vtype = str(row.get('VOC유형대', '') or '')
    if vtype == '청구 미/이의':
        score += 5
    elif vtype == '해지':
        score += 4
    if str(row.get('접수횟수', '0') or '0') == '1':
        score += 4

    cstatus = str(row.get('_cStatusM', '') or '')
    if cstatus == '일시정지':
        score += 2
    elif cstatus in _TERM_STATUS:
        score += 6

    return min(score, 30)


def add_scores_to_df(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """감성 점수·리스크 점수를 DataFrame에 컬럼으로 추가"""
    df = df.copy()
    df['_emScore']   = df[text_col].apply(compute_em_score) if text_col in df.columns else 0
    df['_riskScore'] = df.apply(lambda r: compute_risk_score(r.to_dict()), axis=1)
    return df
