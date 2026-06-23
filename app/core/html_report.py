import json
from datetime import datetime

import pandas as pd


def _pick_col(df: pd.DataFrame, candidates: list) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# 지사 표시 순서 (요청 고정 순서)
_BRANCH_ORDER = ['중앙지사', '강북지사', '서대문지사', '고양지사', '의정부지사', '남양주지사', '강릉지사', '원주지사']


def _sort_branches(values) -> list:
    def _key(v):
        try:
            return (0, _BRANCH_ORDER.index(v))
        except ValueError:
            return (1, v)
    return sorted(values, key=_key)


def generate_html_report(df: pd.DataFrame) -> str:
    """지사·구역·담당 필터 + 지사별 요약 대시보드를 포함한 정적 HTML 리포트 생성.
    지사별 미접수/접수/처리완료/기타 분포와 지사별 점유율을 기본 노출,
    나머지 상세(VOC유형·감성·리스크, 전체 데이터)는 접힌 상태로 구성한다.
    """
    branch_col = _pick_col(df, ['관리지사'])
    zone_col = (
        '_bizZone' if ('_bizZone' in df.columns and (df['_bizZone'] != '').any())
        else _pick_col(df, ['군구', '관리본부'])
    )
    mgr_col = _pick_col(df, ['담당자']) or ('_mgr' if '_mgr' in df.columns else None)
    state_col = _pick_col(df, ['상태', '처리상태'])
    vtype_col = _pick_col(df, ['VOC유형대'])
    risk_col = '_riskScore' if '_riskScore' in df.columns else None
    sent_col = '_감성' if '_감성' in df.columns else None
    date_col = next((c for c in df.columns if '접수일' in c), None)
    text_col = _pick_col(df, ['등록내용'])

    def _val(row, col, default='(미지정)'):
        if not col:
            return default
        v = str(row.get(col, '') or '').strip()
        return v if v else default

    records = []
    for _, row in df.iterrows():
        state_raw = _val(row, state_col, '')
        state_norm = state_raw if state_raw in ('미접수', '접수', '처리완료') else '기타'
        records.append({
            'branch': _val(row, branch_col),
            'zone': _val(row, zone_col),
            'mgr': _val(row, mgr_col),
            'state': state_norm,
            'vtype': _val(row, vtype_col, ''),
            'risk': int(row.get(risk_col, 0) or 0) if risk_col else 0,
            'sent': _val(row, sent_col, ''),
            'date': _val(row, date_col, ''),
            'content': (str(row.get(text_col, '') or '')[:80]) if text_col else '',
        })

    data_json = json.dumps(records, ensure_ascii=False).replace('</script>', '<\\/script>')
    branch_order_json = json.dumps(_BRANCH_ORDER, ensure_ascii=False)
    branches = _sort_branches({r['branch'] for r in records})
    zones = sorted({r['zone'] for r in records})
    mgrs = sorted({r['mgr'] for r in records})
    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    total = len(records)

    def _opts(values):
        return ''.join(f'<option value="{v}">{v}</option>' for v in values)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>VOC 지사별 요약 리포트</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  :root {{ --bg:#0f172a; --card:rgba(30,41,59,0.7); --accent:#2563eb; --text:#f8fafc; --muted:#94a3b8; --border:rgba(255,255,255,0.1); }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text); font-family:'Pretendard',-apple-system,sans-serif; padding:24px; }}
  h1 {{ font-size:1.6rem; margin-bottom:4px; }}
  .sub {{ color:var(--muted); margin-bottom:20px; font-size:0.85rem; }}
  .filters {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px; background:var(--card); padding:16px; border-radius:12px; border:1px solid var(--border); }}
  .filters label {{ display:block; font-size:0.8rem; color:var(--muted); margin-bottom:4px; }}
  select {{ background:#1e293b; color:var(--text); border:1px solid var(--border); border-radius:8px; padding:6px 10px; min-width:160px; }}
  .kpi-row {{ display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-bottom:20px; }}
  .kpi {{ background:linear-gradient(135deg, rgba(37,99,235,0.2) 0%, rgba(15,23,42,0.8) 100%); border:1px solid rgba(37,99,235,0.3); border-radius:12px; padding:14px; text-align:center; }}
  .kpi .v {{ font-size:1.6rem; font-weight:bold; }}
  .kpi .l {{ font-size:0.8rem; color:var(--muted); margin-top:4px; }}
  .charts {{ display:grid; grid-template-columns:1.4fr 1fr; gap:16px; margin-bottom:24px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:16px; padding:16px; }}
  details {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:12px 16px; margin-bottom:12px; }}
  summary {{ cursor:pointer; font-weight:600; padding:4px 0; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.82rem; margin-top:10px; }}
  th, td {{ border-bottom:1px solid var(--border); padding:6px 8px; text-align:left; }}
  th {{ color:var(--muted); position:sticky; top:0; background:#0f172a; }}
  .tablewrap {{ max-height:480px; overflow:auto; }}
  ul {{ margin:6px 0; padding-left:18px; font-size:0.85rem; }}
  @media (max-width:900px) {{ .charts {{ grid-template-columns:1fr; }} .kpi-row {{ grid-template-columns:repeat(2,1fr); }} }}
</style>
</head>
<body>
  <h1>📊 VOC 지사별 요약 리포트</h1>
  <div class="sub">생성일시: {generated_at} · 전체 {total:,}건</div>

  <div class="filters">
    <div><label>지사</label>
      <select id="f-branch"><option value="">전체</option>{_opts(branches)}</select>
    </div>
    <div><label>담당구역</label>
      <select id="f-zone"><option value="">전체</option>{_opts(zones)}</select>
    </div>
    <div><label>담당</label>
      <select id="f-mgr"><option value="">전체</option>{_opts(mgrs)}</select>
    </div>
  </div>

  <div class="kpi-row" id="kpi-row"></div>

  <div class="charts">
    <div class="card"><div id="chart-status"></div></div>
    <div class="card"><div id="chart-share"></div></div>
  </div>

  <div id="mgr-section" style="display:none;">
    <h2 style="font-size:1.15rem; margin:4px 0 12px;">👤 담당별 상세 분석 — <span id="mgr-section-branch"></span></h2>
    <div class="charts">
      <div class="card"><div id="chart-mgr-status"></div></div>
      <div class="card"><div id="chart-mgr-volume"></div></div>
    </div>
    <div class="charts">
      <div class="card"><div id="chart-mgr-share"></div></div>
      <div class="card"><div id="chart-mgr-risk"></div></div>
    </div>
    <div class="card" style="margin-bottom:24px;"><div id="chart-mgr-rate"></div></div>
  </div>

  <details>
    <summary>📋 VOC 유형 · 감성 · 리스크 요약 (펼치기)</summary>
    <div id="extra-summary"></div>
  </details>

  <details>
    <summary>📄 전체 데이터 (펼치기, 최대 2000건 표시)</summary>
    <div class="tablewrap">
      <table id="data-table">
        <thead><tr><th>지사</th><th>담당구역</th><th>담당</th><th>상태</th><th>VOC유형</th><th>감성</th><th>리스크</th><th>접수일시</th><th>내용</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </details>

<script>
const DATA = {data_json};
const BRANCH_ORDER = {branch_order_json};

function sortBranches(arr) {{
  return arr.slice().sort((a, b) => {{
    const ia = BRANCH_ORDER.indexOf(a);
    const ib = BRANCH_ORDER.indexOf(b);
    if (ia === -1 && ib === -1) return a.localeCompare(b, 'ko');
    if (ia === -1) return 1;
    if (ib === -1) return -1;
    return ia - ib;
  }});
}}

function applyFilters() {{
  const b = document.getElementById('f-branch').value;
  const z = document.getElementById('f-zone').value;
  const m = document.getElementById('f-mgr').value;
  return DATA.filter(r => (!b || r.branch === b) && (!z || r.zone === z) && (!m || r.mgr === m));
}}

// 선택된 지사·담당구역에 속한 담당자만 동적으로 옵션 구성
function updateMgrOptions() {{
  const b = document.getElementById('f-branch').value;
  const z = document.getElementById('f-zone').value;
  const pool = DATA.filter(r => (!b || r.branch === b) && (!z || r.zone === z));
  const mgrs = [...new Set(pool.map(r => r.mgr))].sort((a, c) => a.localeCompare(c, 'ko'));
  const sel = document.getElementById('f-mgr');
  const prev = sel.value;
  sel.innerHTML = '<option value="">전체</option>' + mgrs.map(m => `<option value="${{m}}">${{m}}</option>`).join('');
  sel.value = mgrs.includes(prev) ? prev : '';
}}

function render() {{
  const filtered = applyFilters();
  const total = filtered.length;

  const counts = {{'미접수':0,'접수':0,'처리완료':0,'기타':0}};
  filtered.forEach(r => {{ counts[r.state] = (counts[r.state]||0) + 1; }});
  const kpiHtml = [['전체', total], ['미접수', counts['미접수']], ['접수', counts['접수']],
                    ['처리완료', counts['처리완료']], ['기타', counts['기타']]]
    .map(([l,v]) => `<div class="kpi"><div class="v">${{v.toLocaleString()}}</div><div class="l">${{l}}</div></div>`).join('');
  document.getElementById('kpi-row').innerHTML = kpiHtml;

  const branches = sortBranches([...new Set(filtered.map(r => r.branch))]);
  const states = ['미접수','접수','처리완료','기타'];
  const colors = {{'미접수':'#dc2626','접수':'#d97706','처리완료':'#16a34a','기타':'#64748b'}};
  const traces = states.map(s => ({{
    x: branches,
    y: branches.map(b => filtered.filter(r => r.branch===b && r.state===s).length),
    name: s, type: 'bar', marker: {{color: colors[s]}}
  }}));
  Plotly.newPlot('chart-status', traces, {{
    barmode:'stack', title:'지사별 처리 상태 분포',
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    font:{{color:'#f8fafc'}}, legend:{{orientation:'h', y:-0.2}}, margin:{{t:40}}
  }}, {{displayModeBar:false, responsive:true}});

  const shareCounts = branches.map(b => filtered.filter(r => r.branch===b).length);
  Plotly.newPlot('chart-share', [{{
    labels: branches, values: shareCounts, type:'pie', hole:0.45, textinfo:'label+percent',
    marker:{{colors:['#2563eb','#16a34a','#d97706','#7c3aed','#dc2626','#0891b2','#65a30d','#c026d3']}}
  }}], {{
    title:'지사별 점유율', paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    font:{{color:'#f8fafc'}}, margin:{{t:40}}, showlegend:false
  }}, {{displayModeBar:false, responsive:true}});

  const vtypeCounts = {{}};
  filtered.forEach(r => {{ const k = r.vtype || '(미지정)'; vtypeCounts[k] = (vtypeCounts[k]||0)+1; }});
  const sentCounts = {{}};
  filtered.forEach(r => {{ const k = r.sent || '(없음)'; sentCounts[k] = (sentCounts[k]||0)+1; }});
  const highRisk = filtered.filter(r => r.risk >= 20).length;
  const rows = Object.entries(vtypeCounts).map(([k,v]) => `<li>${{k}}: ${{v}}건</li>`).join('');
  const srows = Object.entries(sentCounts).map(([k,v]) => `<li>${{k}}: ${{v}}건</li>`).join('');
  document.getElementById('extra-summary').innerHTML = `
    <div style="display:flex; gap:32px; flex-wrap:wrap; margin-top:10px;">
      <div><b>VOC 유형</b><ul>${{rows}}</ul></div>
      <div><b>감성 분포</b><ul>${{srows}}</ul></div>
      <div><b>고위험(20점↑)</b><div style="font-size:1.4rem;color:#dc2626;">${{highRisk.toLocaleString()}}건</div></div>
    </div>`;

  const tbody = document.querySelector('#data-table tbody');
  tbody.innerHTML = filtered.slice(0, 2000).map(r => `
    <tr><td>${{r.branch}}</td><td>${{r.zone}}</td><td>${{r.mgr}}</td><td>${{r.state}}</td>
    <td>${{r.vtype}}</td><td>${{r.sent}}</td><td>${{r.risk}}</td><td>${{r.date}}</td><td>${{r.content}}</td></tr>`
  ).join('');

  renderMgrCharts();
}}

// 지사 선택 시 해당 지사 내 담당자별 5종 시각화 표시
function renderMgrCharts() {{
  const branch = document.getElementById('f-branch').value;
  const zone = document.getElementById('f-zone').value;
  const section = document.getElementById('mgr-section');

  if (!branch) {{ section.style.display = 'none'; return; }}
  section.style.display = 'block';
  document.getElementById('mgr-section-branch').textContent = branch;

  const pool = DATA.filter(r => r.branch === branch && (!zone || r.zone === zone));
  const mgrs = [...new Set(pool.map(r => r.mgr))].sort((a, b) => a.localeCompare(b, 'ko'));
  const states = ['미접수','접수','처리완료','기타'];
  const colors = {{'미접수':'#dc2626','접수':'#d97706','처리완료':'#16a34a','기타':'#64748b'}};
  const plotCfg = {{displayModeBar:false, responsive:true}};
  const baseLayout = {{
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{{color:'#f8fafc'}}, margin:{{t:40}}
  }};

  // 1. 담당별 처리상태 누적바
  const traces1 = states.map(s => ({{
    x: mgrs, y: mgrs.map(m => pool.filter(r => r.mgr===m && r.state===s).length),
    name: s, type:'bar', marker:{{color: colors[s]}}
  }}));
  Plotly.newPlot('chart-mgr-status', traces1, {{
    ...baseLayout, barmode:'stack', title:'담당별 처리 상태', legend:{{orientation:'h', y:-0.3}}
  }}, plotCfg);

  // 2. 담당별 총 건수 (내림차순 가로 바)
  const volumes = mgrs.map(m => ({{m, n: pool.filter(r => r.mgr===m).length}})).sort((a,b) => b.n - a.n);
  Plotly.newPlot('chart-mgr-volume', [{{
    x: volumes.map(v=>v.n), y: volumes.map(v=>v.m), type:'bar', orientation:'h', marker:{{color:'#2563eb'}}
  }}], {{
    ...baseLayout, title:'담당별 총 건수', margin:{{t:40, l:90}}, yaxis:{{autorange:'reversed'}}
  }}, plotCfg);

  // 3. 담당별 점유율 도넛
  Plotly.newPlot('chart-mgr-share', [{{
    labels: mgrs, values: mgrs.map(m => pool.filter(r=>r.mgr===m).length),
    type:'pie', hole:0.45, textinfo:'label+percent',
    marker:{{colors:['#2563eb','#16a34a','#d97706','#7c3aed','#dc2626','#0891b2','#65a30d','#c026d3','#e11d48','#0ea5e9']}}
  }}], {{
    ...baseLayout, title:'담당별 점유율', showlegend:false
  }}, plotCfg);

  // 4. 담당별 평균 리스크 점수
  const riskAvg = mgrs.map(m => {{
    const rows = pool.filter(r => r.mgr===m);
    const avg = rows.reduce((s,r)=>s+r.risk,0) / (rows.length||1);
    return {{m, avg}};
  }}).sort((a,b) => b.avg - a.avg);
  Plotly.newPlot('chart-mgr-risk', [{{
    x: riskAvg.map(v=>v.m), y: riskAvg.map(v=>Math.round(v.avg*10)/10), type:'bar', marker:{{color:'#dc2626'}}
  }}], {{
    ...baseLayout, title:'담당별 평균 리스크 점수'
  }}, plotCfg);

  // 5. 담당별 처리완료율(%)
  const rate = mgrs.map(m => {{
    const rows = pool.filter(r=>r.mgr===m);
    const done = rows.filter(r=>r.state==='처리완료').length;
    return {{m, pct: rows.length ? (done/rows.length*100) : 0}};
  }}).sort((a,b) => b.pct - a.pct);
  Plotly.newPlot('chart-mgr-rate', [{{
    x: rate.map(v=>v.m), y: rate.map(v=>Math.round(v.pct*10)/10), type:'bar', marker:{{color:'#16a34a'}}
  }}], {{
    ...baseLayout, title:'담당별 처리완료율(%)', yaxis:{{ticksuffix:'%'}}
  }}, plotCfg);
}}

['f-branch', 'f-zone'].forEach(id => document.getElementById(id).addEventListener('change', () => {{
  updateMgrOptions();
  render();
}}));
document.getElementById('f-mgr').addEventListener('change', render);
updateMgrOptions();
render();
</script>
</body>
</html>"""
    return html
