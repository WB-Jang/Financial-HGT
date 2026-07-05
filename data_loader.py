
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import re
import random
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import openpyxl
from collections import defaultdict

def convert_legal_citation(text):
    if not isinstance(text, str):
        return text
    
    # 2. 원문자 매핑을 위한 문자열 (인덱스 활용)
    circles = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
    
    # 3. 정규표현식 패턴
    # (제X조) + (원문자) + [선택: 제Y호 or 제Y의Z호] + [선택: X목]
    pattern = r'(제\s*\d+\s*조)\s*([①-⑳])(?:제\s*\d+\s*(?:의\s*\d+\s*)?호)?(?:[가-힣]\s*목)?'
    
    def replacer(match):
        article = match.group(1)          # "제3조" 부분
        circled_char = match.group(2)     # "①" 부분
        
        # 원문자의 인덱스를 찾아 숫자로 변환 (+1 필요)
        paragraph_num = circles.index(circled_char) + 1 
        
        return f"{article} 제{paragraph_num}항"
    
    # 정규식 패턴에 매칭되는 부분을 찾아 replacer 함수의 결과로 치환
    return re.sub(pattern, replacer, text)


def normalize_johang_key(law_nm, article_raw, hang_num=None):
    """
    법령명 + 원본 조항 표기(및 항 번호)를 '법령명 제X조[의Y] 제X항' 기본 포맷으로 정규화합니다.

    nodes_df(조/항 단위)와 triplets_df(호/목 단위까지 세분화, 원문자 표기 혼재)가
    서로 다른 세밀도와 공백 표기를 사용해 조항 키가 어긋나던 문제(약 30~40% 매칭 실패)를
    막기 위해, 두 데이터프레임의 new_johang 키를 이 함수 하나로 통일해서 생성합니다.
    """
    law_nm = (law_nm or '').strip()
    if not isinstance(article_raw, str):
        article_raw = ''

    # 1. 원문자(①②...)를 '제N항'으로 변환 (조 뒤에 바로 붙어있는 경우)
    circles = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"

    def circle_repl(m):
        return f"{m.group(1)} 제{circles.index(m.group(2)) + 1}항"

    text = re.sub(r'(제\s*\d+(?:-\d+)?\s*조(?:의\s*\d+)?)\s*([①-⑳])', circle_repl, article_raw)

    # 2. 조(+의Y) 추출 - 내부 공백 제거로 표기 통일 ('제 1 조' -> '제1조')
    jo_match = re.search(r'제\s*\d+(?:-\d+)?\s*조(?:\s*의\s*\d+)?', text)
    jo = re.sub(r'\s+', '', jo_match.group(0)) if jo_match else re.sub(r'\s+', '', text.strip())

    # 3. 항 추출 - 조 뒤에 명시된 '제N항'만 인정하고, 그 뒤의 호/목 세부 표기는 버림
    rest = text[jo_match.end():] if jo_match else ''
    hang_match = re.search(r'제\s*\d+\s*항', rest)
    if hang_match:
        hang = re.sub(r'\s+', '', hang_match.group(0))
    elif hang_num is not None and pd.notna(hang_num):
        hang = f"제{int(hang_num)}항"
    else:
        hang = ''

    return ' '.join(p for p in [law_nm, jo, hang] if p)


def split_fsc_train_test(fsc, test_size=100, random_state=42, strata_col='# of laws_clean', query_col='jilui'):
    """
    질의(jilui) 단위로 층화추출을 수행하여 fsc 데이터프레임에 'split'('train'/'test') 컬럼을 부여합니다.
    strata_col(참조 법률 수) 값의 분포 비율을 test set에서도 유지하고,
    동일 질의가 중복 행으로 여러 번 등장하더라도 train/test에 동시에 섞이지 않도록(leakage 방지)
    질의 단위로 먼저 중복을 제거한 뒤 표본을 추출합니다.
    """
    rng = np.random.default_rng(random_state)
    fsc = fsc.copy()
    fsc['split'] = 'train'

    unique_queries = fsc.drop_duplicates(subset=query_col)[[query_col, strata_col]]
    total = len(unique_queries)

    # 층(stratum)별 목표 test 표본 수를 비율대로 배분하고,
    # 최대잔여법(largest remainder method)으로 반올림 오차를 보정해 정확히 test_size에 맞춥니다.
    group_sizes = unique_queries[strata_col].value_counts()
    quotas = group_sizes / total * test_size
    floor_quotas = np.floor(quotas).astype(int)
    remainder = int(test_size - floor_quotas.sum())
    frac_order = (quotas - floor_quotas).sort_values(ascending=False).index
    final_quotas = floor_quotas.copy()
    for stratum in frac_order[:remainder]:
        final_quotas[stratum] += 1

    test_queries = []
    for stratum, quota in final_quotas.items():
        pool = unique_queries.loc[unique_queries[strata_col] == stratum, query_col].tolist()
        quota = min(quota, len(pool))
        if quota <= 0:
            continue
        chosen = rng.choice(pool, size=quota, replace=False)
        test_queries.extend(chosen.tolist())

    fsc.loc[fsc[query_col].isin(test_queries), 'split'] = 'test'
    return fsc


def fsc_dataset_preprocessing(file,nodes_df):
    raw_file = pd.read_excel(file, sheet_name=['법령O+조항O'])
    fsc = raw_file['법령O+조항O']
    fsc['new_johang_clean_split'] = fsc['new_johang_clean'].str.split('|')
    # 1. 리스트들을 개별 행으로 펼치기 (index는 원본을 유지함)
    # 예: [A, B] -> 행1: A, 행2: B (둘 다 원래 인덱스 0을 가짐)
    exploded = fsc['new_johang_clean_split'].explode()
    
    # 2. 고속 벡터화 정규식 추출 (.str.extract 사용)
    pattern = r'^(.+?제\s*\d+(?:-\d+)?\s*조(?:의\s*\d+)?)(?:\s*(제\s*\d+\s*항))?'
    extracted = exploded.str.extract(pattern)
    
    # 3. 조(0열)와 항(1열) 문자열 결합
    # 정규식에 매칭되지 않은 텍스트(NaN)는 원본(exploded) 유지
    law_and_jo = extracted[0].fillna(exploded)
    hang = extracted[1].fillna('') # 항이 없으면 빈 문자열
    
    cleaned_series = (law_and_jo + " " + hang).str.strip()
    
    # 4. 펼쳐졌던 행들을 원래 인덱스(level=0) 기준으로 다시 리스트로 묶기
    fsc['new_johang_clean_split_'] = cleaned_series.groupby(level=0).agg(list)
    # 가정: 
    # df1 = 현재 작업 중인 데이터프레임 (리스트가 들어있는 컬럼: 'new_column_name')
    # df2 = 텍스트를 가져올 다른 데이터프레임 (키 컬럼: 'law_jo_hang', 텍스트 컬럼: 'full_text')

    # 1. df2에서 키를 기준으로 full_text를 리스트로 묶어 딕셔너리로 변환
    # 동일한 조항에 여러 full_text가 있으면 리스트 안에 모두 들어갑니다.

    text_mapping_dict = nodes_df.groupby('new_johang')['full_text'].apply(list).to_dict()
    # 2. '조' 단위로 묶어놓을 딕셔너리 생성 (항이 명시되지 않은 경우 사용)
    article_match_dict = defaultdict(list)

    # df2의 키들을 순회하면서 '조' 단위 키를 추출하여 누적합니다.
    for key, texts in text_mapping_dict.items():
        # 정규식: 끝에 " 제X항"이 있으면 해당 부분을 잘라냅니다.
        # 예: '지배구조법 제25조 제1항' -> '지배구조법 제25조'
        # 예: '지배구조법 제25조' -> '지배구조법 제25조' (변화 없음)
        article_key = re.sub(r'\s*제\s*\d+\s*항$', '', key).strip()
        
        # 추출한 '조' 키에 텍스트들을 전부 이어 붙입니다. (제1항, 제2항... 순차적으로 쌓임)
        article_match_dict[article_key].extend(texts)

   # 3. 매핑 함수 정의
    def fetch_full_texts(key_list):
        if not isinstance(key_list, list):
            return []
            
        fetched_texts = []
        for key in key_list:
            key = key.strip()
            
            # 조건: 검색하려는 키(key)의 끝에 '항'이 포함되어 있는가?
            if re.search(r'제\s*\d+\s*항$', key):
                # '지배구조법 제25조 제1항' 처럼 항을 딱 꼬집어 검색한 경우
                # text_mapping_dict 에서 해당 항의 텍스트만 가져옵니다.
                fetched_texts.extend(text_mapping_dict.get(key, []))
            else:
                # '지배구조법 제25조' 처럼 조까지만 검색한 경우
                # article_match_dict 에서 제1항~제6항까지 누적된 텍스트를 모두 가져옵니다.
                fetched_texts.extend(article_match_dict.get(key, []))
                
        return fetched_texts

    # 3. df1에 적용하여 새로운 컬럼 생성
    fsc['full_text_matched'] = fsc['new_johang_clean_split_'].apply(fetch_full_texts)

    fsc = split_fsc_train_test(fsc, test_size=100, strata_col='# of laws_clean', query_col='jilui')

    col_list = ['row','jilui','# of laws_clean','new_johang_clean_split_','full_text_matched','split']
    fsc = fsc[col_list]
    return fsc
def build_fsc_qa_dataset_hard_negative(fsc_df, nodes_df, encoder, device='cuda', num_negatives=2):
    """
    BGE-M3 임베딩을 활용하여 질의와 의미적으로 유사하지만 정답이 아닌
    'Hard Negative' 조항들을 추출해 학습용 데이터셋으로 구성합니다.
    """
    print("🎯 Hard Negative 추출을 위한 전체 조항 임베딩 생성 중...")
    
    # 1. 탐색 대상이 될 전체 조항 리스트와 텍스트 준비
    unique_nodes = nodes_df.dropna(subset=['new_johang']).drop_duplicates(subset=['new_johang'])
    clause_list = unique_nodes['new_johang'].tolist()
    clause_texts = unique_nodes['full_text'].astype(str).tolist()
    
    # 2. 전체 조항 텍스트를 한 번에 임베딩 (GPU 활용 속도 최적화)
    # convert_to_tensor=True 를 사용하여 PyTorch 텐서로 바로 반환받습니다.
    clause_embs = encoder.encode(clause_texts, convert_to_tensor=True, show_progress_bar=True).to(device)
    clause_embs = F.normalize(clause_embs, p=2, dim=-1) # 코사인 유사도를 위한 L2 정규화
    
    qa_dataset = []
    
    print("🔍 질의별 Hard Negative 매핑 및 데이터셋 조립 중...")
    for _, row in fsc_df.iterrows():
        query = str(row['jilui']).strip()
        pos_clauses = row['new_johang_clean_split_']
        
        # 유효하지 않은 행 건너뛰기
        if not query or not isinstance(pos_clauses, list):
            continue
            
        pos_set = set(pos_clauses)
        
        # 3. 질의(Query) 임베딩 및 유사도 계산
        query_emb = encoder.encode([query], convert_to_tensor=True).to(device)
        query_emb = F.normalize(query_emb, p=2, dim=-1)
        
        # 내적(Dot product) 연산으로 코사인 유사도 일괄 계산
        sims = torch.matmul(query_emb, clause_embs.T).squeeze(0) # shape: (num_clauses,)
        
        # 유사도가 높은 순서대로 인덱스 내림차순 정렬
        sorted_indices = torch.argsort(sims, descending=True)
        
        # 4. 정답을 제외한 최상위 유사도 조항(Hard Negative) 추출
        hard_negatives_pool = []
        for idx in sorted_indices:
            candidate_clause = clause_list[idx.item()]
            
            # 정답 집합(pos_set)에 없는 조항만 오답 풀에 추가
            if candidate_clause not in pos_set:
                hard_negatives_pool.append(candidate_clause)
            
            # 필요한 오답 개수를 넉넉히 확보하면 탐색 중단 (시간 단축)
            if len(hard_negatives_pool) >= num_negatives * 2:
                break
                
        # 5. 수집된 Hard Negative 풀에서 가상 노드 형태(List of Lists)로 묶기
        hard_negatives = []
        for _ in range(num_negatives):
            sample_size = random.choice([1, 2]) # 1개 또는 2개의 조항 묶음
            
            if len(hard_negatives_pool) >= sample_size:
                # 앞에서부터(유사도가 가장 높은 것부터) 꺼내서 조합
                neg_sample = [hard_negatives_pool.pop(0) for _ in range(sample_size)]
                hard_negatives.append(neg_sample)
            else:
                # 풀이 부족할 경우 방어 코드
                hard_negatives.append(hard_negatives_pool)
                
        # 6. 결과 딕셔너리 조립
        qa_dataset.append({
            "query": query,
            "positive_clauses": pos_clauses,
            "hard_negative_clauses": hard_negatives
        })
        
    print(f"✅ 총 {len(qa_dataset)}건의 QA 데이터셋 구성 완료!")
    return qa_dataset

def load_and_build_graph(nodes_path, triplets_path, use_dummy_emb=False):
    """
    CSV 파일들을 읽어 PyTorch Geometric의 HeteroData 객체로 변환합니다.
    """
    print("데이터 로딩 중...")
    nodes_df = pd.read_csv(nodes_path)
    triplets_df = pd.read_csv(triplets_path)

    # nodes_df/triplets_df가 서로 다른 세밀도(조/항 vs 호/목)와 공백 표기를 쓰던 문제를 막기 위해
    # 두 데이터프레임 모두 normalize_johang_key()로 '법령명 제X조[의Y] 제X항' 포맷의 키를 생성합니다.
    nodes_df['new_johang'] = [
        normalize_johang_key(law_nm, article_number, hang_number)
        for law_nm, article_number, hang_number in zip(
            nodes_df['law_nm'], nodes_df['article_number'], nodes_df['hang_number']
        )
    ]

    fsc = fsc_dataset_preprocessing(file='./data/for_review_corrected.xlsx',nodes_df=nodes_df)


    triplets_df['new_johang'] = [
        normalize_johang_key(law_nm, article_number)
        for law_nm, article_number in zip(triplets_df['law_nm'], triplets_df['article_number'])
    ]

    # 1. 노드 딕셔너리 및 인덱스 매핑 구축
    clause_list = nodes_df['new_johang'].dropna().unique().tolist()
    clause_to_idx = {clause: i for i, clause in enumerate(clause_list)}
    
    # Entity는 triplets의 subject와 object에서 고유값을 추출
    subjects = triplets_df['subject'].dropna().unique().tolist()
    objects = triplets_df['object'].dropna().unique().tolist()
    entity_list = list(set(subjects + objects))
    entity_to_idx = {entity: i for i, entity in enumerate(entity_list)}

    # 2. BGE-M3 모델을 이용한 초기 텍스트 임베딩 (1024d)
    if use_dummy_emb:
        print("더미 임베딩 사용 (빠른 테스트용)")
        clause_embs = torch.randn((len(clause_list), 1024))
        entity_embs = torch.randn((len(entity_list), 1024))
    else:
        print("BGE-M3 임베딩 추출 중... (시간이 소요될 수 있습니다)")
        encoder = SentenceTransformer('BAAI/bge-m3')
        
        # Clause는 full_text를 인코딩
        clause_texts = []
        for clause in clause_list:
            text = nodes_df[nodes_df['new_johang'] == clause]['full_text'].iloc[0]
            clause_texts.append(str(text))
        clause_embs = torch.tensor(encoder.encode(clause_texts, show_progress_bar=True))
        
        # Entity는 명칭 자체를 인코딩 (필요시 주변 context 결합 가능)
        entity_embs = torch.tensor(encoder.encode(entity_list, show_progress_bar=True))

    # 3. HeteroData 객체 생성 및 노드 피처 할당 (BGE-M3 임베딩을 실제 노드 피처로 사용)
    data = HeteroData()
    data['clause'].x = clause_embs.float()
    data['entity'].x = entity_embs.float()

    # 4. 엣지 인덱스(Edge Indices) 구축
    # 4.1 Clause -> Entity (HAS_SUBJECT, HAS_OBJECT)
    has_subject_edges = [[], []]
    has_object_edges = [[], []]
    
    for _, row in triplets_df.iterrows():
        clause_key = row['new_johang']
        if pd.isna(clause_key) or clause_key not in clause_to_idx: continue
        
        clause_idx = clause_to_idx[clause_key]
        
        # 주어 연결
        if pd.notna(row['subject']) and row['subject'] in entity_to_idx:
            has_subject_edges[0].append(clause_idx)
            has_subject_edges[1].append(entity_to_idx[row['subject']])
            
        # 목적어 연결
        if pd.notna(row['object']) and row['object'] in entity_to_idx:
            has_object_edges[0].append(clause_idx)
            has_object_edges[1].append(entity_to_idx[row['object']])

    data['clause', 'has_subject', 'entity'].edge_index = torch.tensor(has_subject_edges, dtype=torch.long)
    data['clause', 'has_object', 'entity'].edge_index = torch.tensor(has_object_edges, dtype=torch.long)

    # 4.2 Entity -> Entity (동적 관계 추출)
    
    # 1. 결측치를 제외한 고유 관계 카테고리 추출
    unique_relations = triplets_df['relation_category'].dropna().unique()
    
    # 2. 각 관계 카테고리별로 [[], []] 형태의 빈 리스트를 딕셔너리에 초기화
    entity_edges_dict = {rel: [[], []] for rel in unique_relations}
    
    # 3. 데이터프레임을 순회하며 엣지 쌓기
    for _, row in triplets_df.iterrows():
        sub = row['subject']
        obj = row['object']
        rel_cat = row['relation_category']
        
        # subject, object, relation_category가 모두 유효하고 인덱스에 존재하는 경우
        if pd.notna(rel_cat) and pd.notna(sub) and pd.notna(obj):
            if sub in entity_to_idx and obj in entity_to_idx:
                sub_idx = entity_to_idx[sub]
                obj_idx = entity_to_idx[obj]
                
                # 해당 카테고리의 리스트에 출발지(sub)와 도착지(obj) 인덱스 추가
                entity_edges_dict[rel_cat][0].append(sub_idx)
                entity_edges_dict[rel_cat][1].append(obj_idx)

    # 4. 딕셔너리에 쌓인 엣지들을 순회하며 HeteroData에 일괄 할당
    for rel_cat, edges in entity_edges_dict.items():
        # PyG 관례에 따라 관계 이름은 소문자로 변환하여 사용
        edge_type_name = str(rel_cat).lower()
        
        # 연결된 엣지가 1개 이상 존재하는 경우에만 텐서로 변환하여 그래프에 추가
        if len(edges[0]) > 0:
            data['entity', edge_type_name, 'entity'].edge_index = torch.tensor(edges, dtype=torch.long)

    print(f"그래프 구축 완료: {data}")

    fsc_train = fsc[fsc['split'] == 'train'].reset_index(drop=True)
    fsc_test = fsc[fsc['split'] == 'test'].reset_index(drop=True)
    print(f"fsc 질의 분할 완료: train {len(fsc_train)}건 / test {len(fsc_test)}건")

    fsc_qa_dataset_train = build_fsc_qa_dataset_hard_negative(fsc_df=fsc_train, nodes_df=nodes_df, encoder=encoder, device='cuda', num_negatives=3)
    fsc_qa_dataset_test = build_fsc_qa_dataset_hard_negative(fsc_df=fsc_test, nodes_df=nodes_df, encoder=encoder, device='cuda', num_negatives=3)
    print(f"fsc 학습용/평가용 데이터 셋 구축 완료")
    return data, clause_to_idx, entity_to_idx, fsc_qa_dataset_train, fsc_qa_dataset_test
    
# 단독 실행 테스트용
if __name__ == "__main__":
    # print('---스크립트 실행 시작---')
    # fsc = fsc_dataset_preprocessing('./data/for_review_corrected.xlsx')
    # print(fsc.head())

