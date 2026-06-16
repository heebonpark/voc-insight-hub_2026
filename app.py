import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from app.ui.components import load_css, render_glass_card, render_metric
from app.core.handlers import load_and_preprocess_data
from app.core.ml_models import extract_keywords, cluster_vocs, detect_anomalies, load_sentiment_model, analyze_sentiment

# 페이지 기본 설정
st.set_page_config(
    page_title="Data Intel PRO - VOC Analytics",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 로드
load_css()

# Session State 초기화
if "matched_df" not in st.session_state:
    st.session_state.matched_df = None
if "sentiment_model" not in st.session_state:
    st.session_state.sentiment_model = None

# Sidebar
with st.sidebar:
    st.markdown("### 🌌 Data Intel PRO")
    st.markdown("---")
    st.markdown("#### 📁 파일 업로드")
    voc_file = st.file_uploader("1. VOC 접수 파일 (CSV)", type=['csv'])
    fac_file = st.file_uploader("2. 시설 정보 파일 (CSV)", type=['csv'])
    
    if st.button("데이터 분석 실행", use_container_width=True, type="primary"):
        if voc_file and fac_file:
            with st.spinner("데이터 매칭 및 전처리 중..."):
                try:
                    df = load_and_preprocess_data(voc_file, fac_file)
                    st.session_state.matched_df = df
                    st.success(f"매칭 성공! (총 {len(df)}건)")
                except Exception as e:
                    st.error(f"오류: {e}")
        else:
            st.warning("두 파일을 모두 업로드해주세요.")

# Main Dashboard
st.title("Data Intel PRO : VOC AI Analytics")

if st.session_state.matched_df is not None:
    df = st.session_state.matched_df
    
    # 1. KPI Metrics
    st.markdown("### 📊 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric("총 VOC 건수", f"{len(df):,}건")
    with col2:
        matched = df[df['_matchType'] != ""]
        match_rate = len(matched) / len(df) * 100 if len(df) > 0 else 0
        render_metric("시설 매칭률", f"{match_rate:.1f}%")
    with col3:
        if '_bizZone' in df.columns:
            top_zone = df['_bizZone'].value_counts().index[0] if len(df['_bizZone'].value_counts()) > 0 else "-"
            render_metric("최다 발생 구역", top_zone)
        else:
            render_metric("최다 발생 구역", "-")
    with col4:
        st.markdown(f'<div class="metric-container" style="cursor:pointer;" title="딥러닝 감성분석 로드"><div class="metric-value">🧠</div><div class="metric-label">AI 모델 로드하기</div></div>', unsafe_allow_html=True)

    # 2. Advanced Visualizations
    st.markdown("---")
    st.markdown("### 📈 Visual Analytics")
    vcol1, vcol2 = st.columns(2)
    
    with vcol1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 구역별 불만 현황")
        if '_bizZone' in df.columns and len(df) > 0:
            zone_counts = df[df['_bizZone'] != ""]['_bizZone'].value_counts().reset_index()
            zone_counts.columns = ['구역', '건수']
            fig = px.pie(zone_counts, values='건수', names='구역', hole=0.4,
                         color_discrete_sequence=px.colors.sequential.Blues_r)
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with vcol2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 일별 VOC 접수 추이 (이상 탐지)")
        # '접수일자' 또는 '접수일시' 컬럼이 있다고 가정 (없으면 더미 처리)
        date_col = '접수일자' if '접수일자' in df.columns else ('접수일시' if '접수일시' in df.columns else None)
        if date_col:
            anomaly_df = detect_anomalies(df.copy(), date_col)
            if not anomaly_df.empty:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=anomaly_df[date_col], y=anomaly_df['count'], mode='lines', name='정상'))
                anomalies = anomaly_df[anomaly_df['is_anomaly']]
                fig2.add_trace(go.Scatter(x=anomalies[date_col], y=anomalies['count'], mode='markers', 
                                          marker=dict(color='red', size=10), name='이상 급증'))
                fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("시계열 데이터를 분석할 수 없습니다.")
        else:
            st.info("날짜 컬럼(접수일자/접수일시)을 찾을 수 없습니다.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Machine Learning & Deep Learning AI Analysis
    st.markdown("---")
    st.markdown("### 🤖 Deep Learning / Machine Learning Insights")
    
    # 텍스트 내용 컬럼명 추정 ('상담내용', '불만내용', '내용' 등)
    text_cols = [c for c in df.columns if '내용' in c or '상담' in c or 'VOC' in c]
    text_col = text_cols[0] if text_cols else None
    
    if text_col:
        st.info(f"선택된 분석 대상 텍스트 컬럼: `{text_col}`")
        
        tab1, tab2, tab3 = st.tabs(["🔑 키워드 & 군집 (ML)", "🎭 감성 분석 (DL)", "📋 상세 데이터"])
        
        valid_texts = df[text_col].dropna().astype(str).tolist()
        
        with tab1:
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.markdown("#### TF-IDF 주요 키워드")
                keywords = extract_keywords(valid_texts, top_n=10)
                for i, kw in enumerate(keywords):
                    st.markdown(f"`{i+1}. {kw}`")
            with mcol2:
                st.markdown("#### 불만 유형 군집화 (K-Means)")
                if len(valid_texts) > 0:
                    sample_texts = valid_texts[:100] # 성능을 위해 100개만 샘플링
                    clusters = cluster_vocs(sample_texts, n_clusters=4)
                    sample_df = pd.DataFrame({'Text': sample_texts, 'Cluster': clusters})
                    st.dataframe(sample_df)
                    
        with tab2:
            st.markdown("#### HuggingFace 기반 감성 분석")
            if st.button("감성 분석 모델 로드 및 실행 (Time Consuming)"):
                with st.spinner("모델을 로드하고 데이터를 분석 중입니다..."):
                    if st.session_state.sentiment_model is None:
                        st.session_state.sentiment_model = load_sentiment_model()
                    
                    if st.session_state.sentiment_model:
                        # 샘플 50건 분석
                        sample_for_dl = valid_texts[:50]
                        sentiments = analyze_sentiment(sample_for_dl, st.session_state.sentiment_model)
                        s_df = pd.DataFrame({'Text': sample_for_dl, 'Sentiment': sentiments})
                        
                        # 시각화
                        s_count = s_df['Sentiment'].value_counts().reset_index()
                        s_count.columns = ['감성', '건수']
                        fig3 = px.bar(s_count, x='감성', y='건수', color='감성',
                                      color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'})
                        fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        st.plotly_chart(fig3, use_container_width=True)
                        st.dataframe(s_df)
                    else:
                        st.error("모델 로드에 실패했습니다.")
        
        with tab3:
            st.dataframe(df.head(100))
    else:
        st.warning("분석할 텍스트 컬럼('내용' 등)을 찾지 못했습니다.")

else:
    st.info("👈 좌측 사이드바에서 데이터 파일을 업로드해주세요.")
