"""
retrieval_common.py

evaluate_baseline.py(베이스라인 평가)와 train_query_encoder.py(Stage 2 학습)가
공유하는 데이터 구성/지표 계산 로직.

- build_clause_index: 조항 리스트/텍스트 (data_loader와 동일한 첫 등장 순서 -> 캐시 적중)
- build_retrieval_items: (질의, 정답 조항 인덱스 집합, num_laws) 목록 구성
  (조 단위 표기 -> 항 단위 노드 확장 로직 포함, train.py 평가와 동일)
- compute_metric_rows / summarize_metrics: Recall@K(비율형), Hit@K, MRR 계산
"""

import re
from collections import defaultdict

import pandas as pd

K_VALUES = [1, 3, 5, 10, 15, 30]


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


def compute_metric_rows(ranked_lists, items, k_values=K_VALUES):
    """질의별 지표 행 계산 (순수 파이썬 - 단위 테스트 가능).

    ranked_lists: 질의별 상위 노드 인덱스 리스트 (길이 >= max(k_values) 권장)
    items: build_retrieval_items가 만든 아이템 목록 (같은 순서)
    """
    max_k = max(k_values)
    mrr_col = f"mrr@{max_k}"
    rows = []
    for ranked, it in zip(ranked_lists, items):
        pos = it["pos_idxs"]
        row = {
            "query": it["query"],
            "num_laws": it["num_laws"],
            "num_positive_clauses": len(pos),
        }
        for k in k_values:
            hit_cnt = len(set(ranked[:k]) & pos)
            row[f"recall@{k}"] = hit_cnt / len(pos)          # 비율형 recall (train.py와 동일 정의)
            row[f"hit@{k}"] = 1.0 if hit_cnt > 0 else 0.0    # 상위 K에 정답 1개 이상 (KG-search 비교용)
        rr = 0.0
        for rank, idx in enumerate(ranked[:max_k], 1):
            if idx in pos:
                rr = 1.0 / rank
                break
        row[mrr_col] = rr
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
