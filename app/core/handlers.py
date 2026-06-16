import pandas as pd
import re
from typing import Optional

VOC_COL_ALIASES = {
    '서비스번호': ['서비스번호', '서비스_번호', 'service_no', 'svcno', '서비스번'],
    '계약번호':  ['계약번호',  '계약_번호',  'contract_no', 'cno',   '계약번'],
    '고객번호':  ['고객번호',  '고객_번호',  'customer_no', 'custno','고객번'],
}

FAC_COL_ALIASES = {
    '서비스번호':   ['서비스번호', '서비스_번호', 'service_no', 'svcno'],
    '계약번호':    ['계약번호',  '계약_번호',  'contract_no', 'cno'],
    '고객번호':    ['고객번호',  '고객_번호',  'customer_no', 'custno'],
    '영업구역정보': ['영업구역정보', '영업구역', '구역', 'biz_zone', '영업구역명'],
    '접속전화번호': ['접속전화번호', '전화번호', 'tel', 'phone', '연락처'],
    '계약상태(중)': ['계약상태(중)', '계약상태', 'contract_status', '상태'],
    '설치주소':    ['설치주소', '주소', 'address', '설치지주소'],
}


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
    norm = {c.lower().replace(' ', '').replace('_', '').replace('(', '').replace(')', ''): c
            for c in df.columns}
    for alias in aliases:
        key = alias.lower().replace(' ', '').replace('_', '').replace('(', '').replace(')', '')
        if key in norm:
            return norm[key]
    return None


def _norm_num(series: pd.Series) -> pd.Series:
    return series.fillna('').astype(str).apply(lambda x: re.sub(r'\D', '', x))


def _load_file(file) -> pd.DataFrame:
    name = getattr(file, 'name', '').lower()
    if name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file, dtype=str)
    enc = _detect_encoding(file)
    return pd.read_csv(file, encoding=enc, dtype=str)


def load_voc_only(voc_file) -> pd.DataFrame:
    """시설 파일 없이 VOC 파일만 로드"""
    df = _load_file(voc_file).fillna('')
    df['_matchType'] = ''
    df['_bizZone']   = ''
    df['_tel']       = ''
    df['_cStatusM']  = ''
    df['_facAddr']   = ''
    return df


def load_and_preprocess_data(voc_file, fac_file) -> pd.DataFrame:
    df_voc = _load_file(voc_file).fillna('')
    df_fac = _load_file(fac_file).fillna('')

    # 키 컬럼 감지
    norm_keys = {'서비스번호': 'Norm_Svc', '계약번호': 'Norm_Cno', '고객번호': 'Norm_Cust'}
    key_map = {}
    for key in norm_keys:
        vc = _find_column(df_voc, VOC_COL_ALIASES[key])
        fc = _find_column(df_fac, FAC_COL_ALIASES[key])
        if not vc:
            raise ValueError(
                f"VOC 파일에서 '{key}' 컬럼을 찾을 수 없습니다.\n"
                f"현재 컬럼 목록: {list(df_voc.columns)}"
            )
        if not fc:
            raise ValueError(
                f"시설 파일에서 '{key}' 컬럼을 찾을 수 없습니다.\n"
                f"현재 컬럼 목록: {list(df_fac.columns)}"
            )
        key_map[key] = {'voc': vc, 'fac': fc}

    # 정규화 키 생성
    for key, nk in norm_keys.items():
        df_voc[nk] = _norm_num(df_voc[key_map[key]['voc']])
        df_fac[nk] = _norm_num(df_fac[key_map[key]['fac']])

    # 출력 컬럼 감지
    output_map = {
        '_bizZone':  _find_column(df_fac, FAC_COL_ALIASES['영업구역정보']),
        '_tel':      _find_column(df_fac, FAC_COL_ALIASES['접속전화번호']),
        '_cStatusM': _find_column(df_fac, FAC_COL_ALIASES['계약상태(중)']),
        '_facAddr':  _find_column(df_fac, FAC_COL_ALIASES['설치주소']),
    }

    result = df_voc.copy()
    result['_matchType'] = ''
    for out_col in output_map:
        result[out_col] = ''

    # 벡터화 3단계 매칭 (iterrows 제거 → pandas map 사용)
    for nk, mtype in [('Norm_Svc', 'svc'), ('Norm_Cno', 'cno'), ('Norm_Cust', 'cust')]:
        unmatched_mask = result['_matchType'] == ''
        if not unmatched_mask.any():
            break

        fac_idx = df_fac[df_fac[nk] != ''].drop_duplicates(subset=[nk]).set_index(nk)
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
