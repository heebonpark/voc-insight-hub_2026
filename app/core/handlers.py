import os
import pandas as pd
import re
from typing import Optional

VOC_COL_ALIASES = {
    '서비스번호': ['서비스번호', '서비스_번호', 'service_no', 'svcno', '서비스번'],
    '계약번호':  ['계약번호',  '계약_번호',  'contract_no', 'cno',   '계약번'],
    '고객번호':  ['고객번호',  '고객_번호',  'customer_no', 'custno', '고객번'],
}

FAC_COL_ALIASES = {
    '서비스번호':   ['서비스번호', '서비스_번호', 'service_no', 'svcno'],
    '계약번호':    ['계약번호',  '계약_번호',  'contract_no', 'cno'],
    '고객번호':    ['고객번호',  '고객_번호',  'customer_no', 'custno'],
    '영업구역정보': ['영업구역정보', '영업구역', '구역', 'biz_zone', '영업구역명', '영업구역번호'],
    '기술구역정보': ['기술구역정보', '기술구역', 'tech_zone', '기술구역번호'],
    '접속전화번호': ['접속전화번호', '전화번호', 'tel', 'phone', '연락처', '청구전화번호', '휴대폰', '이동전화'],
    '계약상태(대)': ['계약상태(대)', '계약상태대', '계약상태_대'],
    '계약상태(중)': ['계약상태(중)', '계약상태', 'contract_status'],
    '서비스상태(중)': ['서비스상태(중)', '서비스상태', 'service_status'],
    '정지시작일자': ['정지시작일자', '정지일', 'stop_date', '정지시작'],
    '해지일자':    ['해지일자', '해지일', 'term_date'],
    '설치주소':    ['설치주소', '주소', 'address', '설치지주소'],
    '담당자':      ['담당자', '구역담당영업사원', '구역담당', 'manager'],
    '영업사원명':   ['영업사원명', '영업사원', 'sales_name'],
    '상호':        ['상호', '업체명', '상호명'],
}

# 4단계 상호명 매칭용 컬럼 후보
_VOC_NAME_COLS = ['상호', '업체명', '고객명', '상호명']
_FAC_NAME_COLS = ['상호', '업체명', '고객명', '상호명']

# 결과 DataFrame에 추가할 시설 컬럼 목록
_OUTPUT_COLS = [
    '_matchType', '_bizZone', '_techZone', '_tel',
    '_cStatus', '_cStatusM', '_sStatusM',
    '_stopDate', '_termDate', '_facAddr', '_mgr', '_salesName',
]


def _detect_encoding(file) -> str:
    raw = file.read(8192)
    file.seek(0)
    for enc in ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']:
        try:
            raw.decode(enc)
            return enc
        except (UnicodeDecodeError, AttributeError):
            continue
    return 'cp949'


def _find_column(df: pd.DataFrame, aliases: list) -> Optional[str]:
    norm = {
        c.lower().replace(' ', '').replace('_', '').replace('(', '').replace(')', ''): c
        for c in df.columns
    }
    for alias in aliases:
        key = alias.lower().replace(' ', '').replace('_', '').replace('(', '').replace(')', '')
        if key in norm:
            return norm[key]
    return None


def _norm_num(series: pd.Series) -> pd.Series:
    return series.fillna('').astype(str).apply(lambda x: re.sub(r'\D', '', x))


def _norm_name(val: str) -> str:
    """상호명 정규화: 서비스명·법인구분·특수문자 제거 후 소문자"""
    val = str(val or '').strip()
    if not val:
        return ''
    val = re.sub(r'GiGAeyes.*|OCT.*|ktt.*', '', val, flags=re.IGNORECASE)
    val = re.sub(r'\(주\)|\(재\)|\(유\)|주식회사|유한회사', '', val)
    val = re.sub(r'[\(\)\[\]\s\-_\.\·/,]', '', val)
    return val.lower()


def get_excel_sheets(file) -> list:
    """Excel 파일의 시트 이름 목록 반환. CSV면 빈 리스트."""
    name = os.path.basename(file).lower() if isinstance(file, str) else getattr(file, 'name', '').lower()
    if not name.endswith(('.xlsx', '.xls')):
        return []
    try:
        xl = pd.ExcelFile(file)
        sheets = xl.sheet_names
        if hasattr(file, 'seek'):
            file.seek(0)
        return sheets
    except Exception:
        if hasattr(file, 'seek'):
            file.seek(0)
        return []


def _load_file(file, sheet_name=0) -> pd.DataFrame:
    """파일 로드.
    file: Streamlit 업로드 객체 또는 로컬 경로 문자열
    sheet_name: Excel 시트 이름/인덱스 (CSV는 무시)
    """
    if isinstance(file, str):
        name = os.path.basename(file).lower()
    else:
        name = getattr(file, 'name', '').lower()

    if name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file, sheet_name=sheet_name, dtype=str)

    # CSV: 경로 문자열이면 인코딩 직접 감지
    if isinstance(file, str):
        with open(file, 'rb') as fh:
            raw = fh.read(8192)
        for enc in ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']:
            try:
                raw.decode(enc)
                return pd.read_csv(file, encoding=enc, dtype=str)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(file, encoding='cp949', dtype=str)

    enc = _detect_encoding(file)
    return pd.read_csv(file, encoding=enc, dtype=str)


def load_voc_only(voc_file, sheet_name=0) -> pd.DataFrame:
    """시설 파일 없이 VOC 파일만 로드 (시설 컬럼은 빈 값으로 초기화)"""
    df = _load_file(voc_file, sheet_name=sheet_name).fillna('')
    for col in _OUTPUT_COLS:
        df[col] = ''
    return df


def load_and_preprocess_data(voc_file, fac_file, voc_sheet=0, fac_sheet=0) -> pd.DataFrame:
    df_voc = _load_file(voc_file, sheet_name=voc_sheet).fillna('')
    df_fac = _load_file(fac_file, sheet_name=fac_sheet).fillna('')

    # 3가지 숫자 키 컬럼 감지 및 정규화
    norm_keys = {'서비스번호': 'Norm_Svc', '계약번호': 'Norm_Cno', '고객번호': 'Norm_Cust'}
    key_map = {}
    for key in norm_keys:
        vc = _find_column(df_voc, VOC_COL_ALIASES[key])
        fc = _find_column(df_fac, FAC_COL_ALIASES[key])
        if not vc:
            raise ValueError(
                f"VOC 파일에서 '{key}' 컬럼을 찾을 수 없습니다.\n"
                f"현재 컬럼: {list(df_voc.columns)}"
            )
        if not fc:
            raise ValueError(
                f"시설 파일에서 '{key}' 컬럼을 찾을 수 없습니다.\n"
                f"현재 컬럼: {list(df_fac.columns)}"
            )
        key_map[key] = {'voc': vc, 'fac': fc}

    for key, nk in norm_keys.items():
        df_voc[nk] = _norm_num(df_voc[key_map[key]['voc']])
        df_fac[nk] = _norm_num(df_fac[key_map[key]['fac']])

    # 4단계: 상호명 정규화 키 생성
    voc_name_col = _find_column(df_voc, _VOC_NAME_COLS)
    fac_name_col = _find_column(df_fac, _FAC_NAME_COLS)
    has_name = bool(voc_name_col and fac_name_col)
    if has_name:
        df_voc['Norm_Name'] = df_voc[voc_name_col].apply(_norm_name)
        df_fac['Norm_Name'] = df_fac[fac_name_col].apply(_norm_name)

    # 출력 컬럼 (시설 → 결과) 매핑
    output_map = {
        '_bizZone':   _find_column(df_fac, FAC_COL_ALIASES['영업구역정보']),
        '_techZone':  _find_column(df_fac, FAC_COL_ALIASES['기술구역정보']),
        '_tel':       _find_column(df_fac, FAC_COL_ALIASES['접속전화번호']),
        '_cStatus':   _find_column(df_fac, FAC_COL_ALIASES['계약상태(대)']),
        '_cStatusM':  _find_column(df_fac, FAC_COL_ALIASES['계약상태(중)']),
        '_sStatusM':  _find_column(df_fac, FAC_COL_ALIASES['서비스상태(중)']),
        '_stopDate':  _find_column(df_fac, FAC_COL_ALIASES['정지시작일자']),
        '_termDate':  _find_column(df_fac, FAC_COL_ALIASES['해지일자']),
        '_facAddr':   _find_column(df_fac, FAC_COL_ALIASES['설치주소']),
        '_mgr':       _find_column(df_fac, FAC_COL_ALIASES['담당자']),
        '_salesName': _find_column(df_fac, FAC_COL_ALIASES['영업사원명']),
    }

    result = df_voc.copy()
    result['_matchType'] = ''
    for out_col in output_map:
        result[out_col] = ''

    # 벡터화 4단계 매칭 (pandas map 사용, iterrows 없음)
    match_levels = [
        ('Norm_Svc',  'svc'),
        ('Norm_Cno',  'cno'),
        ('Norm_Cust', 'cust'),
    ]
    if has_name:
        match_levels.append(('Norm_Name', 'name'))

    for nk, mtype in match_levels:
        unmatched_mask = result['_matchType'] == ''
        if not unmatched_mask.any():
            break
        if nk not in result.columns or nk not in df_fac.columns:
            continue

        fac_idx = (
            df_fac[df_fac[nk] != '']
            .drop_duplicates(subset=[nk])
            .set_index(nk)
        )
        unmatched_keys = result.loc[unmatched_mask, nk]
        hit_mask = unmatched_keys.isin(fac_idx.index)
        hit_idx = unmatched_keys[hit_mask].index

        if len(hit_idx) > 0:
            result.loc[hit_idx, '_matchType'] = mtype
            for out_col, fac_col in output_map.items():
                if fac_col and fac_col in fac_idx.columns:
                    result.loc[hit_idx, out_col] = (
                        unmatched_keys[hit_mask].map(fac_idx[fac_col]).values
                    )

    return result
