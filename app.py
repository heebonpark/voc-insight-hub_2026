import io
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.ui.components import load_css, render_metric
from app.core.handlers import (
    load_and_preprocess_data, load_voc_only, get_excel_sheets,
)
from app.core.ml_models import (
    extract_keywords, extract_topics_lda, cluster_vocs,
    detect_anomalies, analyze_sentiment_rule,
    load_sentiment_model, analyze_sentiment,
    add_scores_to_df,
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
if 'sidebar_open' not in st.session_state:
    st.session_state.sidebar_open = True

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

@st.cache_data(show_spinner=False)
def cached_scores(df_json, text_col):
    return add_scores_to_df(pd.read_json(io.StringIO(df_json)), text_col)


# ── Sankey 워크플로우 빌더 ─────────────────────────────────────────────────────
def _build_sankey(df: pd.DataFrame, stage_cols: list, max_cat: int = 10) -> go.Figure | None:
    """다중 단계 Sankey 다이어그램. stage_cols 순서대로 노드-링크 구성."""
    STEP_COLORS = [
        'rgba(37,99,235,0.45)',   # blue
        'rgba(22,163,74,0.45)',   # green
        'rgba(217,119,6,0.45)',   # amber
        'rgba(124,58,237,0.45)',  # purple
        'rgba(220,38,38,0.45)',   # red
    ]
    label_list: list[str] = []
    label_idx: dict[str, int] = {}
    sources, targets, values, link_colors = [], [], [], []

    def node_idx(stage: str, val: str) -> int:
        # 동일 값이 다른 단계에 있어도 별개 노드로 처리
        key = f"{stage}::{val}"
        if key not in label_idx:
            label_idx[key] = len(label_list)
            label_list.append(str(val))
        return label_idx[key]

    for step, (src_col, tgt_col) in enumerate(zip(stage_cols[:-1], stage_cols[1:])):
        grp = (
            df[[src_col, tgt_col]].fillna('(없음)').astype(str)
            .groupby([src_col, tgt_col]).size()
            .reset_index(name='n')
        )
        top_src = grp.groupby(src_col)['n'].sum().nlargest(max_cat).index
        top_tgt = grp.groupby(tgt_col)['n'].sum().nlargest(max_cat).index
        grp = grp[grp[src_col].isin(top_src) & grp[tgt_col].isin(top_tgt)]

        color = STEP_COLORS[step % len(STEP_COLORS)]
        for _, row in grp.iterrows():
            sources.append(node_idx(src_col, row[src_col]))
            targets.append(node_idx(tgt_col, row[tgt_col]))
            values.append(int(row['n']))
            link_colors.append(color)

    if not sources:
        return None

    fig = go.Figure(go.Sankey(
        arrangement='snap',
        node=dict(
            label=label_list,
            pad=20, thickness=18,
            color='#334155',
            line=dict(color='rgba(148,163,184,0.3)', width=0.5),
        ),
        link=dict(source=sources, target=targets, value=values, color=link_colors),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f1f5f9', size=11),
        margin=dict(t=10, l=10, r=10, b=10),
        height=500,
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 VOC Insight Hub")
    st.markdown("---")

    # ── 파일 입력: 업로드 OR 로컬 경로 ──────────────────────────────────────
    st.markdown("#### 📁 파일 입력")
    voc_file = st.file_uploader("1. VOC 접수 파일 (CSV / Excel)",
                                 type=['csv', 'xlsx', 'xls'], key="voc_upload")
    fac_file = st.file_uploader("2. 시설 정보 파일 (CSV / Excel)",
                                 type=['csv', 'xlsx', 'xls'], key="fac_upload")

    with st.expander("📂 로컬 경로 직접 입력 (Excel 열린 상태 포함)"):
        st.caption("파일을 저장 없이 열린 채로 경로만 입력해도 읽습니다.")
        voc_path = st.text_input("VOC 파일 경로", placeholder="/Users/…/voc.xlsx", key="voc_path")
        fac_path = st.text_input("시설 파일 경로", placeholder="/Users/…/fac.xlsx", key="fac_path")
        if voc_path:
            if os.path.exists(voc_path):
                voc_file = voc_path
                st.success(f"✓ {os.path.basename(voc_path)}")
            else:
                st.warning("경로를 찾을 수 없습니다.")
        if fac_path:
            if os.path.exists(fac_path):
                fac_file = fac_path
                st.success(f"✓ {os.path.basename(fac_path)}")
            else:
                st.warning("경로를 찾을 수 없습니다.")

    # ── 시트 선택 ─────────────────────────────────────────────────────────────
    voc_sheet, fac_sheet = 0, 0

    if voc_file is not None:
        voc_sheets = get_excel_sheets(voc_file)
        if len(voc_sheets) > 1:
            voc_sheet = st.selectbox(
                "📄 VOC 시트 선택", voc_sheets, key="voc_sheet_sel",
                help="Excel 파일에 시트가 여러 개일 때 선택",
            )
        elif len(voc_sheets) == 1:
            voc_sheet = voc_sheets[0]
            st.caption(f"VOC 시트: **{voc_sheets[0]}**")

    if fac_file is not None:
        fac_sheets = get_excel_sheets(fac_file)
        if len(fac_sheets) > 1:
            fac_sheet = st.selectbox(
                "📄 시설 시트 선택", fac_sheets, key="fac_sheet_sel",
            )
        elif len(fac_sheets) == 1:
            fac_sheet = fac_sheets[0]
            st.caption(f"시설 시트: **{fac_sheets[0]}**")

    st.markdown("")
    if st.button("🔍 데이터 분석 실행", use_container_width=True, type="primary"):
        if not voc_file:
            st.warning("VOC 접수 파일을 업로드하거나 경로를 입력해주세요.")
        elif voc_file and fac_file:
            with st.spinner("데이터 매칭 및 전처리 중..."):
                try:
                    df_result = load_and_preprocess_data(
                        voc_file, fac_file,
                        voc_sheet=voc_sheet, fac_sheet=fac_sheet,
                    )
                    st.session_state.matched_df = df_result
                    n_matched = (df_result['_matchType'] != '').sum()
                    st.success(f"✅ 매칭 완료! 총 {len(df_result):,}건 / 매칭 {n_matched:,}건")
                except Exception as e:
                    st.error(f"❌ {e}")
        else:
            with st.spinner("VOC 파일 로드 중..."):
                try:
                    df_result = load_voc_only(voc_file, sheet_name=voc_sheet)
                    st.session_state.matched_df = df_result
                    st.info(f"📄 VOC 파일만 로드됨 ({len(df_result):,}건) — 텍스트 분석만 가능합니다.")
                except Exception as e:
                    st.error(f"❌ {e}")

    # ── 필터 ──────────────────────────────────────────────────────────────────
    if st.session_state.matched_df is not None:
        st.markdown("---")
        st.markdown("#### 🔎 필터")
        _df_all = st.session_state.matched_df

        zones = ['전체']
        if '_bizZone' in _df_all.columns:
            zones += sorted(_df_all.loc[_df_all['_bizZone'] != '', '_bizZone'].unique().tolist())
        sel_zone = st.selectbox("영업구역", zones)

        match_opts = {
            '전체':         None,
            '서비스번호 매칭': 'svc',
            '계약번호 매칭':  'cno',
            '고객번호 매칭':  'cust',
            '상호명 매칭':   'name',
            '미매칭':        '',
            '🔶 정지 시설':  'stop',
            '🔴 해지 시설':  'term',
        }
        sel_match = st.selectbox("매칭/계약 유형", list(match_opts.keys()))

        state_col = next((c for c in _df_all.columns if c in ('상태', '처리상태', 'status')), None)
        if state_col:
            states = ['전체'] + sorted(_df_all[state_col].dropna().unique().tolist())
            sel_state = st.selectbox("처리 상태", states)
        else:
            sel_state = '전체'

        vtype_col = next((c for c in _df_all.columns if 'VOC유형대' in c or c == 'VOC유형'), None)
        if vtype_col:
            vtypes = ['전체'] + sorted(_df_all[vtype_col].dropna().unique().tolist())
            sel_vtype = st.selectbox("VOC 유형", vtypes)
        else:
            sel_vtype = '전체'

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


# ── 사이드바 표시/숨김 ─────────────────────────────────────────────────────────
if not st.session_state.sidebar_open:
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] { display: none !important; min-width: 0 !important; }
    .main .block-container { max-width: 100% !important; padding-left: 2rem !important; }
    </style>
    """, unsafe_allow_html=True)

# ── 헤더 ──────────────────────────────────────────────────────────────────────
btn_col, title_col = st.columns([0.04, 0.96])
with btn_col:
    icon = "✕" if st.session_state.sidebar_open else "☰"
    if st.button(icon, key="toggle_sidebar", help="사이드바 접기/펼치기"):
        st.session_state.sidebar_open = not st.session_state.sidebar_open
        st.rerun()
with title_col:
    st.title("📊 VOC Insight Hub")

if st.session_state.matched_df is None:
    st.info("👈 좌측 사이드바에서 파일을 업로드하거나 로컬 경로를 입력한 뒤 **데이터 분석 실행**을 눌러주세요.")
    st.markdown("""
    **지원 기능**
    | 기능 | 설명 |
    |---|---|
    | 📁 파일 | CSV·Excel 자동 인코딩 감지, **열린 파일 경로 직접 입력**, **다중 시트 선택** |
    | 🔗 매칭 | 서비스번호→계약번호→고객번호→상호명 4단계 자동 매칭 |
    | 🔀 워크플로우 | 등록내용 기준 Sankey 시각화, 다중조건 동적 필터 |
    | ⚠️ 리스크 | calcRisk 이식 — 위험/주의/관찰/정상 등급 + TOP20 |
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
    if mval == 'stop':
        df = df[df['_cStatusM'] == '일시정지'] if '_cStatusM' in df.columns else df.iloc[0:0]
    elif mval == 'term':
        _TERM = {'일반해지', '명의해지', '직권해지', '해지'}
        df = df[df['_cStatusM'].isin(_TERM)] if '_cStatusM' in df.columns else df.iloc[0:0]
    else:
        df = df[df['_matchType'] == mval]

if sel_state != '전체' and state_col and state_col in df.columns:
    df = df[df[state_col] == sel_state]

if sel_vtype != '전체' and vtype_col and vtype_col in df.columns:
    df = df[df[vtype_col] == sel_vtype]

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

# ── KPI 8개 ───────────────────────────────────────────────────────────────────
st.markdown("### 📈 핵심 지표")
total   = len(df)
matched = (df['_matchType'] != '').sum()
rate    = matched / total * 100 if total else 0

state_col_cur = next((c for c in df.columns if c in ('상태', '처리상태')), None)
unprocessed = int((df[state_col_cur] == '미접수').sum()) if state_col_cur else 0

stop_count = int((df['_cStatusM'] == '일시정지').sum()) if '_cStatusM' in df.columns else 0
_TERM_SET = {'일반해지', '명의해지', '직권해지', '해지'}
term_count = int(df['_cStatusM'].isin(_TERM_SET).sum()) if '_cStatusM' in df.columns else 0

vtype_cur = next((c for c in df.columns if 'VOC유형대' in c), None)
billing = int((df[vtype_cur] == '청구 미/이의').sum()) if vtype_cur else 0

text_col = next(
    (c for c in df.columns if any(k in c for k in ['등록내용', '내용', '상담', 'VOC', '불만', '접수내용'])),
    None,
)
urgent = 0
if text_col:
    urgent = int(df[text_col].fillna('').str.contains('감성|불만|빠른연락|긴급').sum())

avg_day = '-'
date_col = next((c for c in df.columns if '접수일' in c), None)
if date_col:
    try:
        _ts = pd.to_datetime(df[date_col], errors='coerce')
        n_days = _ts.dt.date.nunique()
        avg_day = f"{total / n_days:.1f}건" if n_days else '-'
    except Exception:
        pass

c1, c2, c3, c4 = st.columns(4)
c5, c6, c7, c8 = st.columns(4)
with c1: render_metric("총 VOC 건수",    f"{total:,}건")
with c2: render_metric("시설 매칭률",    f"{rate:.1f}%")
with c3: render_metric("미접수 건수",    f"{unprocessed:,}건")
with c4: render_metric("정지 시설 VOC",  f"{stop_count:,}건")
with c5: render_metric("해지 시설 VOC",  f"{term_count:,}건")
with c6: render_metric("청구·이의 민원", f"{billing:,}건")
with c7: render_metric("감성·긴급 VOC",  f"{urgent:,}건")
with c8: render_metric("일 평균 접수",   avg_day)

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
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='white'), margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("영업구역 정보가 없습니다.")

with ch2:
    st.markdown("#### 일별 VOC 추이 (이상 탐지)")
    if date_col:
        adf = cached_anomalies(df.to_json(), date_col)
        if not adf.empty:
            norm = adf[~adf['is_anomaly']]
            anom = adf[adf['is_anomaly']]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=norm[date_col], y=norm['count'], mode='lines+markers',
                                      name='정상', line=dict(color='#2563eb')))
            fig2.add_trace(go.Scatter(x=anom[date_col], y=anom['count'], mode='markers',
                                      name='이상 급증', marker=dict(color='red', size=12, symbol='star')))
            fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                               font=dict(color='white'), margin=dict(t=10),
                               legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("날짜 컬럼(접수일자)을 찾을 수 없습니다.")

_mlabels = {'svc': '서비스번호', 'cno': '계약번호', 'cust': '고객번호', 'name': '상호명', '': '미매칭'}
st.markdown("#### 매칭 유형 분포")
md = df['_matchType'].map(_mlabels).value_counts().reset_index()
md.columns = ['유형', '건수']
fig3 = px.bar(md, x='유형', y='건수', color='유형',
              color_discrete_sequence=['#2563eb', '#16a34a', '#d97706', '#7c3aed', '#dc2626'])
fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                   font=dict(color='white'), showlegend=False, margin=dict(t=10))
st.plotly_chart(fig3, use_container_width=True)

# ── AI 분석 ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🤖 AI 텍스트 분석")

if not text_col:
    st.warning("분석할 텍스트 컬럼('등록내용', '내용' 등)을 찾을 수 없습니다.")
    st.stop()

st.info(f"분석 대상 컬럼: **{text_col}**")
all_texts  = df[text_col].dropna().astype(str).tolist()
valid_texts = [t for t in all_texts if t.strip() and t != 'nan']

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔀 워크플로우", "🔑 키워드", "📚 토픽 모델링",
    "🎭 감성 분석", "⚠️ 리스크 분석", "🗂️ 군집화", "📋 데이터 & 내보내기",
])

# ── Tab 1: 워크플로우 Sankey ───────────────────────────────────────────────────
with tab1:
    st.markdown("#### 🔀 VOC 처리 흐름 시각화 (다중조건 동적 Sankey)")
    st.caption("등록내용 기준 파생 컬럼(_감성, _위험등급)을 포함해 단계를 자유롭게 조합하세요.")

    # 등록내용에서 감성·위험등급 파생 컬럼 생성
    wf_df = df.copy()
    with st.spinner("등록내용 감성·위험등급 계산 중..."):
        sents_wf = cached_sentiment_rule(
            tuple(wf_df[text_col].fillna('').astype(str).tolist())
        )
        wf_df['_감성'] = sents_wf

        scored_wf = cached_scores(df.to_json(), text_col)
        if '_riskScore' in scored_wf.columns:
            def _grade(s):
                if s >= 20: return '🔴위험'
                if s >= 12: return '🟠주의'
                if s >= 6:  return '🟡관찰'
                return '🟢정상'
            wf_df['_위험등급'] = scored_wf['_riskScore'].apply(_grade)

    # 선택 가능한 단계 컬럼 목록 (카테고리가 적당한 컬럼만)
    _skip_prefixes = ('Norm_', )
    _candidate_cols = [
        c for c in wf_df.columns
        if not any(c.startswith(p) for p in _skip_prefixes)
        and wf_df[c].dtype == object
        and 1 < wf_df[c].nunique() <= 30
        and c not in (text_col,)
    ]
    # 파생 컬럼을 맨 앞에 배치
    _derived = [c for c in ['_감성', '_위험등급', '_matchType'] if c in _candidate_cols]
    _rest    = [c for c in _candidate_cols if c not in _derived]
    _all_stage_cols = _derived + _rest

    # 기본 단계: 등록내용 감성 → VOC유형대 → 처리상태
    _default_stages = []
    for c in ['_감성', 'VOC유형대', '담당자 팀', '상태']:
        if c in _all_stage_cols:
            _default_stages.append(c)
    _default_stages = _default_stages[:3]

    stages = st.multiselect(
        "워크플로우 단계 선택 (2~5개, 순서대로 연결됩니다)",
        _all_stage_cols,
        default=_default_stages,
        max_selections=5,
        key="wf_stages",
    )

    # ── 다중조건 동적 필터 ────────────────────────────────────────────────────
    flt_df = wf_df.copy()
    with st.expander("🔧 다중조건 필터 (선택한 단계별 값 제한)", expanded=False):
        filter_applied = False
        if stages:
            cols_per_row = 3
            for i in range(0, len(stages), cols_per_row):
                row_cols = st.columns(cols_per_row)
                for j, col_name in enumerate(stages[i:i + cols_per_row]):
                    with row_cols[j]:
                        uniq_vals = sorted(
                            wf_df[col_name].fillna('(없음)').astype(str).unique().tolist()
                        )
                        sel_vals = st.multiselect(
                            f"{col_name}", uniq_vals, key=f"wf_flt_{col_name}",
                        )
                        if sel_vals:
                            flt_df = flt_df[flt_df[col_name].fillna('(없음)').astype(str).isin(sel_vals)]
                            filter_applied = True
        if filter_applied:
            st.caption(f"필터 적용 후: **{len(flt_df):,}건**")

    max_cat = st.slider("단계당 최대 카테고리 수", 5, 20, 10, key="wf_maxcat")

    if len(stages) >= 2:
        fig_sankey = _build_sankey(flt_df, stages, max_cat=max_cat)
        if fig_sankey:
            st.plotly_chart(fig_sankey, use_container_width=True)

            # 단계별 집계 테이블
            with st.expander("📊 단계별 집계 보기"):
                for src_c, tgt_c in zip(stages[:-1], stages[1:]):
                    st.markdown(f"**{src_c} → {tgt_c}**")
                    pivot = (
                        flt_df.fillna('(없음)').groupby([src_c, tgt_c])
                        .size().reset_index(name='건수')
                        .sort_values('건수', ascending=False)
                        .head(20)
                    )
                    st.dataframe(pivot, use_container_width=True, hide_index=True)
        else:
            st.info("선택한 단계 조합에 표시할 데이터가 없습니다.")
    else:
        st.info("워크플로우 단계를 **2개 이상** 선택해주세요.")

# ── Tab 2: 키워드 ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### TF-IDF 핵심 키워드 (한국어 형태소 분석)")
    with st.spinner("키워드 추출 중..."):
        kws = cached_keywords(tuple(valid_texts), 20)

    if kws:
        kw_df = pd.DataFrame(kws, columns=['키워드', '점수'])
        col_a, col_b = st.columns(2)
        with col_a:
            fig_bar = px.bar(kw_df.head(15), x='점수', y='키워드', orientation='h',
                             color='점수', color_continuous_scale='Blues')
            fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                  font=dict(color='white'), yaxis={'categoryorder': 'total ascending'},
                                  margin=dict(t=10))
            st.plotly_chart(fig_bar, use_container_width=True)
        with col_b:
            fig_tree = px.treemap(kw_df, path=['키워드'], values='점수',
                                  color='점수', color_continuous_scale='Blues')
            fig_tree.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                   font=dict(color='white'), margin=dict(t=10))
            st.plotly_chart(fig_tree, use_container_width=True)
    else:
        st.info("키워드를 추출할 수 없습니다. 데이터가 충분한지 확인해주세요.")

# ── Tab 3: 토픽 모델링 ────────────────────────────────────────────────────────
with tab3:
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
            fig_sun.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                  font=dict(color='white'))
            st.plotly_chart(fig_sun, use_container_width=True)
    else:
        st.info("토픽 모델링에 필요한 데이터가 부족합니다 (최소 10건 이상 필요).")

# ── Tab 4: 감성 분석 ──────────────────────────────────────────────────────────
with tab4:
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
            fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                  font=dict(color='white'))
            st.plotly_chart(fig_pie, use_container_width=True)
        with sb:
            fig_bar2 = px.bar(s_cnt, x='감성', y='건수', color='감성',
                              color_discrete_map=color_map)
            fig_bar2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
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

# ── Tab 5: 리스크 분석 ────────────────────────────────────────────────────────
with tab5:
    st.markdown("#### ⚠️ VOC 리스크 스코어링")
    st.caption("감성 키워드·미접수·청구민원·해지/정지 시설 복합 점수 (0~30점)")

    with st.spinner("리스크 점수 계산 중..."):
        scored_df = cached_scores(df.to_json(), text_col)

    if '_riskScore' in scored_df.columns and scored_df['_riskScore'].max() > 0:
        def _grade(s):
            if s >= 20: return '🔴 위험'
            if s >= 12: return '🟠 주의'
            if s >= 6:  return '🟡 관찰'
            return '🟢 정상'

        scored_df['위험등급'] = scored_df['_riskScore'].apply(_grade)
        grade_cnt = scored_df['위험등급'].value_counts().reset_index()
        grade_cnt.columns = ['등급', '건수']
        grade_colors = {'🔴 위험': '#dc2626', '🟠 주의': '#f97316',
                        '🟡 관찰': '#eab308', '🟢 정상': '#16a34a'}

        r1, r2 = st.columns(2)
        with r1:
            fig_grade = px.bar(grade_cnt, x='등급', y='건수', color='등급',
                               color_discrete_map=grade_colors, title='위험도 등급 분포')
            fig_grade.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='white'), showlegend=False, margin=dict(t=40))
            st.plotly_chart(fig_grade, use_container_width=True)
        with r2:
            fig_hist = px.histogram(scored_df, x='_riskScore', nbins=20,
                                    color_discrete_sequence=['#2563eb'], title='리스크 점수 분포')
            fig_hist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                   font=dict(color='white'), margin=dict(t=40),
                                   xaxis_title='점수', yaxis_title='건수')
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### 🔺 고위험 VOC TOP 20")
        disp_cols = ['_riskScore', '_emScore', '위험등급']
        for c in ['상태', 'VOC유형대', '접수일자', '_cStatusM', '_bizZone', text_col]:
            if c in scored_df.columns:
                disp_cols.append(c)
        top20 = scored_df.nlargest(20, '_riskScore')[disp_cols].reset_index(drop=True)
        top20.columns = [c.replace('_riskScore', '리스크점수').replace('_emScore', '감성점수')
                         .replace('_cStatusM', '계약상태').replace('_bizZone', '영업구역')
                         .replace(text_col, 'VOC내용') for c in top20.columns]
        st.dataframe(top20, use_container_width=True)
    else:
        st.info("리스크 점수 계산 가능 컬럼이 부족합니다. 시설 파일을 함께 업로드하면 정확도가 향상됩니다.")

# ── Tab 6: 군집화 ─────────────────────────────────────────────────────────────
with tab6:
    st.markdown("#### 불만 유형 군집화 (K-Means)")
    n_cl = st.slider("군집 수", 2, 8, 4, key="km_n")
    sample_texts = valid_texts[:min(300, len(valid_texts))]

    with st.spinner("군집화 중..."):
        clusters = cached_cluster(tuple(sample_texts), n_cl)

    cl_df = pd.DataFrame({'내용': sample_texts, '유형': clusters})
    cl_cnt = cl_df['유형'].value_counts().reset_index()
    cl_cnt.columns = ['유형', '건수']
    fig_cl = px.bar(cl_cnt, x='유형', y='건수', color='유형')
    fig_cl.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                         font=dict(color='white'), showlegend=False)
    st.plotly_chart(fig_cl, use_container_width=True)
    st.dataframe(cl_df, use_container_width=True)

# ── Tab 7: 데이터 & 내보내기 ──────────────────────────────────────────────────
with tab7:
    st.markdown(f"#### 전체 데이터 — {len(df):,}건")
    col_rename = {
        '_matchType': '매칭유형', '_bizZone': '영업구역', '_techZone': '기술구역',
        '_tel': '전화번호', '_cStatus': '계약상태(대)', '_cStatusM': '계약상태(중)',
        '_sStatusM': '서비스상태', '_stopDate': '정지일', '_termDate': '해지일',
        '_facAddr': '설치주소', '_mgr': '담당자', '_salesName': '영업사원',
    }
    display_df = df.rename(columns=col_rename)
    if '매칭유형' in display_df.columns:
        display_df['매칭유형'] = display_df['매칭유형'].map(
            {'svc': '서비스번호', 'cno': '계약번호', 'cust': '고객번호', 'name': '상호명', '': '미매칭'}
        )

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        display_df.to_excel(writer, index=False, sheet_name='VOC분석결과')
    buf.seek(0)
    st.download_button(
        label="📥 Excel 다운로드",
        data=buf,
        file_name="voc_분석결과.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    search_q = st.text_input("🔍 내용 검색", placeholder="검색어를 입력하세요...")
    view_df = display_df[display_df[text_col].str.contains(search_q, na=False)] if search_q else display_df
    st.dataframe(view_df, use_container_width=True)
