"""
retrieval_common.py

evaluate_baseline.py(베이스라인 평가)와 train_query_encoder.py(Stage 2 학습)가
공유하는 데이터 구성/지표 계산 로직.

- build_clause_index: 조항 리스트/텍스트 (data_loader와 동일한 첫 등장 순서 -> 캐시 적중)
- build_retrieval_items: (질의, 정답 조항 인덱스 집합, num_laws) 목록 구성
  (조 단위 표기 -> 항 단위 노드 확장 로직 포함, train.py 평가와 동일)
- compute_metric_rows / summarize_metrics: Recall@K(비율형), Hit@K, MRR 계산
"""

import os
import re
from collections import defaultdict

import pandas as pd

K_VALUES = [1, 3, 5, 10, 15, 30]


def emb_tag(clause_emb_path):
    """조항 임베딩 출처를 결과 파일명에 넣기 위한 짧은 태그.

    None(원본 BGE 캐시) -> 'origEmb', 평활화 파일 -> 'smoothEmb', 그 외 -> 파일 stem.
    """
    if not clause_emb_path:
        return "origEmb"
    stem = os.path.splitext(os.path.basename(clause_emb_path))[0]
    if "smooth" in stem.lower():
        return "smoothEmb"
    return re.sub(r'[^0-9A-Za-z]+', '-', stem).strip('-') or "customEmb"


def article_key_of(clause_key):
    """항 단위 키에서 조 단위 키 추출: '지배구조법 제25조 제6항' -> '지배구조법 제25조'"""
    return re.sub(r'\s*제\s*\d+\s*항$', '', clause_key).strip()


def build_clause_adjacency(clause_list, triplets_df, max_entity_df=20,
                           sibling_weight=1.0, entity_weight=1.0, verbose=True):
    """조항 사이의 무방향 가중 인접 그래프 구성. (i, j) -> weight 딕셔너리 (i < j).

    1) 형제 항: 같은 조(article)에 속한 항들끼리 연결
    2) 공유 엔터티: 같은 엔터티를 언급하는 조항끼리 연결 (triplets 활용)
       - 허브 엔터티(max_entity_df개 초과 조항과 연결)는 제외
    (build_smoothed_clause_emb.py와 rerank/PPR이 공유)
    """
    clause_to_idx = {c: i for i, c in enumerate(clause_list)}
    edge_w = defaultdict(float)

    # 1) 형제 항
    article_groups = defaultdict(list)
    for key, idx in clause_to_idx.items():
        article_groups[article_key_of(key)].append(idx)
    n_sibling_pairs = 0
    for members in article_groups.values():
        if len(members) < 2:
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = sorted((members[a], members[b]))
                edge_w[(i, j)] += sibling_weight
                n_sibling_pairs += 1

    # 2) 공유 엔터티
    ent_clauses = defaultdict(set)
    for _, row in triplets_df.iterrows():
        ck = row['new_johang']
        if pd.isna(ck) or ck not in clause_to_idx:
            continue
        ci = clause_to_idx[ck]
        for col in ('subject', 'object'):
            v = row[col]
            if pd.notna(v):
                ent_clauses[str(v)].add(ci)

    n_entity_pairs, n_used_entities = 0, 0
    for ent, cls in ent_clauses.items():
        if len(cls) < 2 or len(cls) > max_entity_df:
            continue
        n_used_entities += 1
        members = sorted(cls)
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                edge_w[(members[a], members[b])] += entity_weight
                n_entity_pairs += 1

    if verbose:
        print(f"인접 구성: 형제 항 쌍 {n_sibling_pairs:,}개, "
              f"공유 엔터티 쌍 {n_entity_pairs:,}개 (사용 엔터티 {n_used_entities:,}개, df<={max_entity_df})")
        print(f"고유 엣지: {len(edge_w):,}개")
    return edge_w


def build_clause_index(nodes_df):
    """조항 키 리스트와 각 조항의 full_text를 반환. (new_johang 컬럼이 이미 있어야 함)"""
    clause_list = nodes_df['new_johang'].dropna().unique().tolist()
    clause_to_text = nodes_df.drop_duplicates('new_johang').set_index('new_johang')['full_text']
    clause_texts = [str(clause_to_text[c]) for c in clause_list]
    return clause_list, clause_texts


def build_article_expander(clause_list):
    """'법령명 제X조'(조 단위 키)를 그 조의 모든 항 단위 노드로 확장하는 함수를 반환."""
    clause_set = set(clause_list)
    article_to_paragraphs = defaultdict(list)
    for key in clause_list:
        article_key = re.sub(r'\s*제\s*\d+\s*항$', '', key).strip()
        if article_key != key:
            article_to_paragraphs[article_key].append(key)

    def expand(key):
        if key in clause_set:
            return [key]
        return article_to_paragraphs.get(key, [])

    return expand


def build_retrieval_items(fsc_df, clause_list):
    """fsc 행들을 (query, pos_idxs 집합, num_laws) 아이템 목록으로 변환.

    정답 조항이 그래프에 하나도 매칭되지 않는 질의는 제외.
    반환: (items, skipped_count)
    """
    clause_to_idx = {c: i for i, c in enumerate(clause_list)}
    expand = build_article_expander(clause_list)

    items, skipped = [], 0
    for _, row in fsc_df.iterrows():
        query = str(row['jilui']).strip()
        pos_raw = row['new_johang_clean_split_']
        if not query or not isinstance(pos_raw, list):
            skipped += 1
            continue
        pos_clauses = list(dict.fromkeys(sum((expand(c) for c in pos_raw), [])))
        if not pos_clauses:
            skipped += 1
            continue
        num_laws = row['# of laws_clean']
        items.append({
            "query": query,
            "pos_idxs": set(clause_to_idx[c] for c in pos_clauses),
            "num_laws": int(num_laws) if pd.notna(num_laws) else None,
        })
    return items, skipped


def _metric_row(ranked, pos_set, k_values):
    """단일 질의의 recall/hit/MRR 계산. ranked와 pos_set의 원소는 인덱스든 문자열이든 무방."""
    max_k = max(k_values)
    row = {}
    for k in k_values:
        hit_cnt = len(set(ranked[:k]) & pos_set)
        row[f"recall@{k}"] = hit_cnt / len(pos_set)          # 비율형 recall
        row[f"hit@{k}"] = 1.0 if hit_cnt > 0 else 0.0        # 상위 K에 정답 1개 이상
    rr = 0.0
    for rank, x in enumerate(ranked[:max_k], 1):
        if x in pos_set:
            rr = 1.0 / rank
            break
    row[f"mrr@{max_k}"] = rr
    return row


def compute_metric_rows(ranked_lists, items, k_values=K_VALUES):
    """질의별 '항 단위' 지표 행 계산 (순수 파이썬 - 단위 테스트 가능).

    ranked_lists: 질의별 상위 노드 인덱스 리스트 (길이 >= max(k_values) 권장)
    items: build_retrieval_items가 만든 아이템 목록 (같은 순서)
    """
    mrr_col = f"mrr@{max(k_values)}"
    rows = []
    for ranked, it in zip(ranked_lists, items):
        row = {
            "query": it["query"],
            "num_laws": it["num_laws"],
            "num_positive_clauses": len(it["pos_idxs"]),
        }
        row.update(_metric_row(ranked, it["pos_idxs"], k_values))
        rows.append(row)
    return rows, mrr_col


def compute_article_metric_rows(full_ranked_lists, items, clause_list, k_values=K_VALUES):
    """질의별 '조(article) 단위' 지표 행 계산.

    항 단위 랭킹을 조 단위로 접는다: 랭킹 상위부터 각 항의 조 키를 뽑아
    처음 등장하는 순서대로 조 랭킹을 만들고(= 조 점수를 소속 항 최고 점수로 정의),
    정답도 조 단위로 접는다. 원본 fsc 인용이 조 단위였던 것들은 여기서 원래
    세밀도로 복원되므로, 조 단위 검색인 KG-search 프로젝트와 공정 비교가 가능하다.

    full_ranked_lists: 질의별 '전체' 노드 인덱스 랭킹 (조 단위로 접으면 길이가 줄어드므로
                       max(k_values)보다 훨씬 깊은 랭킹 필요 - 전체 argsort 권장)
    """
    max_k = max(k_values)
    mrr_col = f"mrr@{max_k}"
    rows = []
    for ranked, it in zip(full_ranked_lists, items):
        pos_articles = {article_key_of(clause_list[i]) for i in it["pos_idxs"]}
        seen, article_ranking = set(), []
        for idx in ranked:
            ak = article_key_of(clause_list[idx])
            if ak not in seen:
                seen.add(ak)
                article_ranking.append(ak)
                if len(article_ranking) >= max_k:
                    break
        row = {
            "query": it["query"],
            "num_laws": it["num_laws"],
            "num_positive_articles": len(pos_articles),
        }
        row.update(_metric_row(article_ranking, pos_articles, k_values))
        rows.append(row)
    return rows, mrr_col


def summarize_metrics(eval_df, k_values=K_VALUES, mrr_col=None):
    """참조 법률 개수(num_laws)별 평균 + overall 행을 붙인 요약 DataFrame 반환."""
    if mrr_col is None:
        mrr_col = f"mrr@{max(k_values)}"
    recall_cols = [f"recall@{k}" for k in k_values]
    hit_cols = [f"hit@{k}" for k in k_values]
    metric_cols = recall_cols + hit_cols + [mrr_col]

    by_num_laws = eval_df.groupby("num_laws")[metric_cols].mean()
    by_num_laws["num_queries"] = eval_df.groupby("num_laws").size()
    by_num_laws = by_num_laws.reset_index().sort_values("num_laws")

    overall_row = {"num_laws": "overall", "num_queries": len(eval_df)}
    overall_row.update(eval_df[metric_cols].mean().to_dict())

    summary_df = pd.concat([by_num_laws, pd.DataFrame([overall_row])], ignore_index=True)
    return summary_df, by_num_laws, overall_row, recall_cols, hit_cols
