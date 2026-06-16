import pandas as pd
import re

def strict_norm_num(val) -> str:
    if pd.isna(val) or val is None:
        return ""
    return re.sub(r'\D', '', str(val))

def load_and_preprocess_data(voc_file, fac_file):
    try:
        # 1. 파일 로드
        df_voc = pd.read_csv(voc_file, encoding='cp949', dtype=str)
        df_fac = pd.read_csv(fac_file, encoding='cp949', dtype=str)
        
        # 2. 결측치 처리
        df_voc = df_voc.fillna("")
        df_fac = df_fac.fillna("")

        # 3. 정규화 키 생성
        df_voc['Norm_Svc'] = df_voc['서비스번호'].apply(strict_norm_num)
        df_voc['Norm_Cno'] = df_voc['계약번호'].apply(strict_norm_num)
        df_voc['Norm_Cust'] = df_voc['고객번호'].apply(strict_norm_num)
        
        df_fac['Norm_Svc'] = df_fac['서비스번호'].apply(strict_norm_num)
        df_fac['Norm_Cno'] = df_fac['계약번호'].apply(strict_norm_num)
        df_fac['Norm_Cust'] = df_fac['고객번호'].apply(strict_norm_num)
        
        # 4. FAC 데이터를 딕셔너리로 인덱싱 (빠른 매칭)
        # 서비스번호 기준
        fac_by_svc = df_fac[df_fac['Norm_Svc'] != ""].set_index('Norm_Svc').to_dict('index')
        fac_by_cno = df_fac[df_fac['Norm_Cno'] != ""].set_index('Norm_Cno').to_dict('index')
        fac_by_cust = df_fac[df_fac['Norm_Cust'] != ""].drop_duplicates(subset=['Norm_Cust']).set_index('Norm_Cust').to_dict('index')
        
        # 5. 매칭 수행
        matched_results = []
        for _, row in df_voc.iterrows():
            target = None
            mtype = ""
            
            v_svc = row['Norm_Svc']
            v_cno = row['Norm_Cno']
            v_cust = row['Norm_Cust']
            
            if v_svc and v_svc in fac_by_svc:
                target = fac_by_svc[v_svc]
                mtype = "svc"
            elif v_cno and v_cno in fac_by_cno:
                target = fac_by_cno[v_cno]
                mtype = "cno"
            elif v_cust and v_cust in fac_by_cust:
                target = fac_by_cust[v_cust]
                mtype = "cust"
                
            res = row.to_dict()
            res['_matchType'] = mtype
            if target:
                res['_bizZone'] = target.get('영업구역정보', '')
                res['_tel'] = target.get('접속전화번호', '')
                res['_cStatusM'] = target.get('계약상태(중)', '')
                res['_facAddr'] = target.get('설치주소', '')
            else:
                res['_bizZone'] = ""
                res['_tel'] = ""
                res['_cStatusM'] = ""
                res['_facAddr'] = ""
                
            matched_results.append(res)
            
        return pd.DataFrame(matched_results)
        
    except Exception as e:
        raise Exception(f"데이터 처리 중 오류 발생: {str(e)}")
