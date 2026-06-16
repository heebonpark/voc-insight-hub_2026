import io
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.ui.components import load_css, render_metric
from app.core.handlers import load_and_preprocess_data, load_voc_only
from app.core.ml_models import (
    extract_keywords, extract_topics_lda, cluster_vocs,
    detect_anomalies, analyze_sentiment_rule,
    load_sentiment_model, analyze_sentiment,
)

st.set_page_config(
    page_title="VOC Insight Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()

for _k in ['matched_df', 'sentiment_model']:
    if _k not in st.session_state:
        st.session_state[_k] = None

# ── 캐시 래퍼 ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_keywords(texts_tuple, top_n):
    return extract_keywords(list(texts_tuple), top_n)

@st.cache_data(show_spinner=False)
def cached_lda(texts_tuple, n_topics):
    return extract_topics_lda(list(texts_tuple), n_topics)

@st.cache_data(show_spinner=False)
def cached_cluster(texts_tuple, n_clusters):
    return cluster_vocs(list(texts_tuple), n_clusters)

@st.cache_data(show_spinner=False)
def cached_sentiment_rule(texts_tuple):
    return analyze_sentiment_rule(list(texts_tuple))

@st.cache_data(show_spinner=False)
def cached_anomalies(df_json, date_col):
    return detect_anomalies(pd.read_json(io.StringIO(df_json)), date_col)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 VOC Insight Hub")
    st.markdown("---")
    st.markdown("#### 📁 파일 업로드")
    voc_file = st.file_uploader("1. VOC 접수 파일 (CSV / Excel)", type=['csv', 'xlsx', 'xls'])
    fac_file = st.file_uploader("2. 시설 정보 파일 (CSV / Excel)", type=['csv', 'xlsx', 'xls'])

    if st.button("🔍 데이터 분석 실행", use_container_width=True, type="primary"):
        if not voc_file:
            st.warning("VOC 접수 파일을 업로드해주세요.")
        elif voc_file and fac_file:
            with st.spinner("데이터 매칭 및 전처리 중..."):
                try:
                    df_result = load_and_preprocess_data(voc_file, fac_file)
                    st.session_state.matched_df = df_result
                    st.success(f"✅ 매칭 완료! 총 {len(df_result):,}건")
                except Exception as e:
                    st.error(f"❌ {e}")
        else:
            with st.spinner("VOC 파일 로드 중..."):
                try:
                    df_result = load_voc_only(voc_file)
                    st.session_state.matched_df = df_result
                    st.info(f"📄 VOC 파일만 로드됨 ({len(df_result):,}건) — 시설 파일 없이 텍스트 분석만 가능합니다.")
                except Exception as e:
                    st.error(f"❌ {e}")

    if st.session_state.matched_df is not None:
        st.markdown("---")
        st.markdown("#### 🔎 필터")
        _df_all = st.session_state.matched_df

        zones = ['전체'] + sorted(
            _df_all.loc[_df_all['_bizZone'] != '', '_bizZone'].unique().tolist()
        ) if '_bizZone' in _df_all.columns else ['전체']
        sel_zone = st.selectbox("영업구역", zones)

        match_opts = {'전체': None, '서비스번호 매칭': 'svc',
                      '계약번호 매칭': 'cno', '고객번호 매칭': 'cust', '미매칭': ''}
        sel_match = st.selectbox("매칭 유형", list(match_opts.keys()))

        date_col = next((c for c in _df_all.columns if '접수일' in c), None)
        date_range = None
        if date_col:
            try:
                _dates = pd.to_datetime(_df_all[date_col], errors='coerce').dropna()
                if not _dates.empty:
                    date_range = st.date_input(
                        "접수 기간",
                        value=(_dates.min().date(), _dates.max().date()),
                        min_value=_dates.min().date(),
                        max_value=_dates.max().date(),
                    )
            except Exception:
                pass

# ── Main ─────────────────────────────────────────────────────────────────────
st.title("📊 VOC Insight Hub")

if st.session_state.matched_df is None:
    st.info("👈 좌측 사이드바에서 파일을 업로드하고 **데이터 분석 실행**을 눌러주세요.")
    st.markdown("""
    **지원 기능 요약**
    | 기능 | 설명 |
    |---|---|
    | 📁 파일 | CSV·Excel 자동 인코딩 감지 |
    | 🔗 매칭 | 서비스번호→계약번호→고객번호 3단계 자동 매칭 |
    | 🔑 키워드 | 한국어 형태소 분석 + TF-IDF |
    | 📚 토픽 | LDA 토픽 모델링 |
    | 🎭 감성 | 규칙 기반 / 딥러닝 감성 분석 |
    | 🗂️ 군집 | K-Means 불만 유형 분류 |
    | 🚨 이상 | IQR + Isolation Forest 급증 탐지 |
    | 📥 내보내기 | Excel 다운로드 |
    """)
    st.stop()

# ── 필터 적용 ─────────────────────────────────────────────────────────────────
df = st.session_state.matched_df.copy()

if sel_zone != '전체' and '_bizZone' in df.columns:
    df = df[df['_bizZone'] == sel_zone]

mval = match_opts.get(sel_match)
if mval is not None:
    df = df[df['_matchType'] == mval]

if date_range and date_col and len(date_range) == 2:
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df[(df[date_col].dt.date >= date_range[0]) &
                (df[date_col].dt.date <= date_range[1])]
    except Exception:
        pass

if df.empty:
    st.warning("선택한 필터 조건에 해당하는 데이터가 없습니다.")
    st.stop()

# ── KPI ───────────────────────────────────────────────────────────────────────
st.markdown("### 📈 핵심 지표")
total   = len(df)
matched = (df['_matchType'] != '').sum()
unmatch = total - matched
rate    = matched / total * 100 if total else 0

top_zone = '-'
if '_bizZone' in df.columns:
    vc = df[df['_bizZone'] != '']['_bizZone'].value_counts()
    top_zone = vc.index[0] if not vc.empty else '-'

avg_day = '-'
if date_col and date_col in df.columns:
    try:
        n_days = df[date_col].dt.date.nunique()
        avg_day = f"{total / n_days:.1f}건" if n_days else '-'
    except Exception:
        pass

c1, c2, c3, c4, c5 = st.columns(5)
with c1: render_metric("총 VOC 건수",    f"{total:,}건")
with c2: render_metric("시설 매칭률",    f"{rate:.1f}%")
with c3: render_metric("미매칭 건수",    f"{unmatch:,}건")
with c4: render_metric("최다 발생 구역", top_zone)
with c5: render_metric("일 평균 접수",   avg_day)

# ── 현황 차트 ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 현황 분석")
ch1, ch2 = st.columns(2)

with ch1:
    st.markdown("#### 구역별 VOC 현황")
    if '_bizZone' in df.columns:
        zd = df[df['_bizZone'] != '']['_bizZone'].value_counts().reset_index()
        zd.columns = ['구역', '건수']
        if not zd.empty:
            fig = px.pie(zd, values='건수', names='구역', hole=0.4,
                         color_discrete_sequence=px.colors.sequential.Blues_r)
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                              paper_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='white'), margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

with ch2:
    st.markdown("#### 일별 VOC 추이 (이상 탐지)")
    dc = next((c for c in df.columns if '접수일' in c), None)
    if dc:
        adf = cached_anomalies(df.to_json(), dc)
        if not adf.empty:
            fig2 = go.Figure()
            norm = adf[~adf['is_anomaly']]
            anom = adf[adf['is_anomaly']]
            fig2.add_trace(go.Scatter(x=norm[dc], y=norm['count'], mode='lines+markers',
                                      name='정상', line=dict(color='#2563eb')))
            fig2.add_trace(go.Scatter(x=anom[dc], y=anom['count'], mode='markers',
                                      name='이상 급증',
                                      marker=dict(color='red', size=12, symbol='star')))
            fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                               paper_bgcolor='rgba(0,0,0,0)',
                               font=dict(color='white'), margin=dict(t=10),
                               legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("날짜 컬럼(접수일자/접수일시)을 찾을 수 없습니다.")

# 매칭 유형 분포
st.markdown("#### 매칭 유형 분포")
mlabels = {'svc': '서비스번호', 'cno': '계약번호', 'cust': '고객번호', '': '미매칭'}
md = df['_matchType'].map(mlabels).value_counts().reset_index()
md.columns = ['유형', '건수']
fig3 = px.bar(md, x='유형', y='건수', color='유형',
              color_discrete_sequence=['#2563eb', '#16a34a', '#d97706', '#dc2626'])
fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                   font=dict(color='white'), showlegend=False, margin=dict(t=10))
st.plotly_chart(fig3, use_container_width=True)

# ── AI 분석 ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🤖 AI 텍스트 분석")

text_col = next(
    (c for c in df.columns if any(k in c for k in ['내용', '상담', 'VOC', '불만', '접수내용'])),
    None,
)
if not text_col:
    st.warning("분석할 텍스트 컬럼('내용', '상담내용' 등)을 찾을 수 없습니다.")
    st.stop()

st.info(f"분석 대상 컬럼: **{text_col}**")
all_texts = df[text_col].dropna().astype(str).tolist()
valid_texts = [t for t in all_texts if t.strip() and t != 'nan']

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🔑 키워드", "📚 토픽 모델링", "🎭 감성 분석", "🗂️ 군집화", "📋 데이터 & 내보내기"]
)

# ── Tab 1: 키워드 ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown("#### TF-IDF 핵심 키워드 (한국어 형태소 분석 적용)")
    with st.spinner("키워드 추출 중..."):
        kws = cached_keywords(tuple(valid_texts), 20)

    if kws:
        kw_df = pd.DataFrame(kws, columns=['키워드', '점수'])

        col_a, col_b = st.columns(2)
        with col_a:
            fig_bar = px.bar(kw_df.head(15), x='점수', y='키워드', orientation='h',
                             color='점수', color_continuous_scale='Blues')
            fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                                  paper_bgcolor='rgba(0,0,0,0)',
                                  font=dict(color='white'),
                                  yaxis={'categoryorder': 'total ascending'},
                                  margin=dict(t=10))
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b:
            fig_tree = px.treemap(kw_df, path=['키워드'], values='점수',
                                  color='점수', color_continuous_scale='Blues')
            fig_tree.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                                   paper_bgcolor='rgba(0,0,0,0)',
                                   font=dict(color='white'), margin=dict(t=10))
            st.plotly_chart(fig_tree, use_container_width=True)
    else:
        st.info("키워드를 추출할 수 없습니다. 텍스트 데이터가 충분한지 확인해주세요.")

# ── Tab 2: 토픽 모델링 ────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### LDA 토픽 모델링")
    n_topics = st.slider("토픽 수", 3, 8, 5, key="lda_n")
    with st.spinner("토픽 분석 중..."):
        topics = cached_lda(tuple(valid_texts), n_topics)

    if topics:
        for t in topics:
            with st.expander(f"**{t['topic']}** — {t['label']}"):
                st.write(" · ".join(t['keywords']))

        rows = [{'토픽': t['topic'], '키워드': kw, '값': 1}
                for t in topics for kw in t['keywords'][:5]]
        if rows:
            fig_sun = px.sunburst(pd.DataFrame(rows), path=['토픽', '키워드'], values='값')
            fig_sun.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                                  paper_bgcolor='rgba(0,0,0,0)',
                                  font=dict(color='white'))
            st.plotly_chart(fig_sun, use_container_width=True)
    else:
        st.info("토픽 모델링에 필요한 데이터가 부족합니다 (최소 10건 이상 필요).")

# ── Tab 3: 감성 분석 ──────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### 감성 분석")
    rule_tab, dl_tab = st.tabs(["⚡ 규칙 기반 (즉시 실행)", "🧠 딥러닝 (고정밀)"])

    with rule_tab:
        with st.spinner("감성 분석 중..."):
            sents = cached_sentiment_rule(tuple(valid_texts))

        s_df = pd.DataFrame({'텍스트': valid_texts, '감성': sents})
        s_cnt = s_df['감성'].value_counts().reset_index()
        s_cnt.columns = ['감성', '건수']
        color_map = {'부정': '#dc2626', '중립': '#94a3b8', '긍정': '#16a34a'}

        sa, sb = st.columns(2)
        with sa:
            fig_pie = px.pie(s_cnt, values='건수', names='감성', hole=0.4,
                             color='감성', color_discrete_map=color_map)
            fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                                  paper_bgcolor='rgba(0,0,0,0)',
                                  font=dict(color='white'))
            st.plotly_chart(fig_pie, use_container_width=True)
        with sb:
            fig_bar2 = px.bar(s_cnt, x='감성', y='건수', color='감성',
                              color_discrete_map=color_map)
            fig_bar2.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                                   paper_bgcolor='rgba(0,0,0,0)',
                                   font=dict(color='white'), showlegend=False)
            st.plotly_chart(fig_bar2, use_container_width=True)

        st.dataframe(s_df.head(100), use_container_width=True)

    with dl_tab:
        st.info("최초 실행 시 모델 다운로드로 수 분이 소요됩니다.")
        if st.button("딥러닝 감성 분석 실행 (최대 50건 샘플)"):
            with st.spinner("모델 로드 중..."):
                if st.session_state.sentiment_model is None:
                    st.session_state.sentiment_model = load_sentiment_model()
            if st.session_state.sentiment_model:
                sample = valid_texts[:50]
                with st.spinner(f"{len(sample)}건 분석 중..."):
                    dl_sents = analyze_sentiment(sample, st.session_state.sentiment_model)
                st.dataframe(pd.DataFrame({'텍스트': sample, '감성': dl_sents}),
                             use_container_width=True)
            else:
                st.error("모델 로드에 실패했습니다.")

# ── Tab 4: 군집화 ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown("#### 불만 유형 군집화 (K-Means)")
    n_cl = st.slider("군집 수", 2, 8, 4, key="km_n")
    sample_size = min(300, len(valid_texts))
    sample_texts = valid_texts[:sample_size]

    with st.spinner("군집화 중..."):
        clusters = cached_cluster(tuple(sample_texts), n_cl)

    cl_df = pd.DataFrame({'내용': sample_texts, '유형': clusters})
    cl_cnt = cl_df['유형'].value_counts().reset_index()
    cl_cnt.columns = ['유형', '건수']

    fig_cl = px.bar(cl_cnt, x='유형', y='건수', color='유형')
    fig_cl.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                         paper_bgcolor='rgba(0,0,0,0)',
                         font=dict(color='white'), showlegend=False)
    st.plotly_chart(fig_cl, use_container_width=True)
    st.dataframe(cl_df, use_container_width=True)

# ── Tab 5: 데이터 & 내보내기 ──────────────────────────────────────────────────
with tab5:
    st.markdown(f"#### 전체 데이터 — {len(df):,}건")

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='VOC분석결과')
    buf.seek(0)
    st.download_button(
        label="📥 Excel 다운로드",
        data=buf,
        file_name="voc_분석결과.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    search_q = st.text_input("🔍 내용 검색", placeholder="검색어를 입력하세요...")
    view_df = df[df[text_col].str.contains(search_q, na=False)] if search_q else df
    st.dataframe(view_df, use_container_width=True)
