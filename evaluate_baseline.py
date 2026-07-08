"""
evaluate_baseline.py

학습 없이 '순수 BGE-M3 코사인 유사도'만으로 조항 검색 성능을 측정하는 베이스라인 스크립트.

목적:
  train.py로 학습한 HGT 모델의 Recall@K와 직접 비교하기 위한 기준선.
  - test 분할(층화추출 시드 42), 조->항 정답 확장, Recall@K 정의 모두 train.py 평가와 동일
  - 추가로 Hit@K(상위 K에 정답 1개 이상 포함 비율)와 MRR도 보고
    (KG-search_PPR_GNN_Transformer 프로젝트의 "Recall@K"는 실제로는 Hit@K 정의라서,
     그쪽 수치와 비교할 때는 Hit@K 열을 봐야 공정한 비교가 됩니다)

실행:
  python evaluate_baseline.py
  (GPU 불필요 - 전부 CPU로 동작. emb_cache/가 있으면 임베딩 재인코딩 없이 수 분 내 완료)

결과:
  콘솔 출력 + eval_results/baseline_eval_*.csv / .json 저장
"""

import os
import re
import json
from collections import defaultdict
from datetime import datetime

import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from data_loader import normalize_johang_key, fsc_dataset_preprocessing, encode_texts_cached

K_VALUES = [1, 3, 5, 10, 15, 30]
NODES_CSV = './data/nodes.csv'
FSC_XLSX = './data/for_review_corrected.xlsx'


def build_test_items(nodes_df, fsc_test, clause_list):
    """test 질의를 (query, 정답 조항 인덱스 집합, num_laws)로 변환.

    조 단위 표기('법령명 제X조')를 그 조에 속한 모든 항 단위 노드로 확장하는 로직은
    data_loader.build_fsc_qa_dataset_hard_negative / train.py 평가와 동일하다.
    """
    clause_set = set(clause_list)
    clause_to_idx = {c: i for i, c in enumerate(clause_list)}

    article_to_paragraphs = defaultdict(list)
    for key in clause_list:
        article_key = re.sub(r'\s*제\s*\d+\s*항$', '', key).strip()
        if article_key != key:
            article_to_paragraphs[article_key].append(key)

    def expand(key):
        if key in clause_set:
            return [key]
        return article_to_paragraphs.get(key, [])

    items, skipped = [], 0
    for _, row in fsc_test.iterrows():
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


def main():
    print("=== 순수 BGE-M3 베이스라인 평가 (학습 없음) ===")

    # 1. 노드 로드 + 조항 키 정규화 (data_loader와 동일)
    nodes_df = pd.read_csv(NODES_CSV)
    nodes_df['new_johang'] = [
        normalize_johang_key(law_nm, article_number, hang_number)
        for law_nm, article_number, hang_number in zip(
            nodes_df['law_nm'], nodes_df['article_number'], nodes_df['hang_number']
        )
    ]

    # 2. fsc 전처리 + train/test 분할 (train.py와 동일한 시드/층화 로직)
    fsc = fsc_dataset_preprocessing(file=FSC_XLSX, nodes_df=nodes_df)
    fsc_test = fsc[fsc['split'] == 'test'].reset_index(drop=True)
    print(f"test 질의(분할 기준): {len(fsc_test)}건")

    # 3. 조항 리스트/텍스트 (data_loader와 동일한 첫 등장 순서 -> 임베딩 캐시 적중)
    clause_list = nodes_df['new_johang'].dropna().unique().tolist()
    clause_to_text = nodes_df.drop_duplicates('new_johang').set_index('new_johang')['full_text']
    clause_texts = [str(clause_to_text[c]) for c in clause_list]
    print(f"검색 대상 조항 노드: {len(clause_list):,}개")

    # 4. 평가 대상 test 질의 구성
    items, skipped = build_test_items(nodes_df, fsc_test, clause_list)
    print(f"평가 가능 질의: {len(items)}건 (그래프 미매칭 제외 {skipped}건)")
    if not items:
        print("평가 가능한 질의가 없습니다.")
        return

    # 5. 임베딩 준비 (emb_cache/ 재사용 - train.py 실행 시 만든 캐시와 동일 파일)
    encoder = SentenceTransformer('BAAI/bge-m3', device='cpu')
    clause_embs = encode_texts_cached(encoder, clause_texts, 'clause_embs')
    # 캐시 이름을 hard negative 단계와 동일하게 두어, train.py를 이미 실행했다면 그 캐시가 적중됨
    query_embs = encode_texts_cached(encoder, [it["query"] for it in items], 'fsc_query_embs')

    clause_embs = F.normalize(clause_embs.float(), dim=-1)   # (N, 1024)
    query_embs = F.normalize(query_embs.float(), dim=-1)     # (Q, 1024)

    # 6. 전체 코퍼스 코사인 랭킹 -> 지표 계산
    sims = query_embs @ clause_embs.T                        # (Q, N)
    max_k = min(max(K_VALUES), sims.size(1))
    topk = sims.topk(max_k, dim=1).indices                   # (Q, max_k)

    results = []
    for i, it in enumerate(items):
        pos = it["pos_idxs"]
        ranked = topk[i].tolist()
        row = {
            "query": it["query"],
            "num_laws": it["num_laws"],
            "num_positive_clauses": len(pos),
        }
        for k in K_VALUES:
            hit_cnt = len(set(ranked[:k]) & pos)
            row[f"recall@{k}"] = hit_cnt / len(pos)          # 비율형 recall (train.py와 동일 정의)
            row[f"hit@{k}"] = 1.0 if hit_cnt > 0 else 0.0    # KG-search 비교용
        rr = 0.0
        for rank, idx in enumerate(ranked, 1):
            if idx in pos:
                rr = 1.0 / rank
                break
        row[f"mrr@{max_k}"] = rr
        results.append(row)

    eval_df = pd.DataFrame(results)
    mrr_col = f"mrr@{max_k}"
    recall_cols = [f"recall@{k}" for k in K_VALUES]
    hit_cols = [f"hit@{k}" for k in K_VALUES]
    metric_cols = recall_cols + hit_cols + [mrr_col]

    # 참조 법률 개수별 + 전체 요약
    by_num_laws = eval_df.groupby("num_laws")[metric_cols].mean()
    by_num_laws["num_queries"] = eval_df.groupby("num_laws").size()
    by_num_laws = by_num_laws.reset_index().sort_values("num_laws")

    overall_row = {"num_laws": "overall", "num_queries": len(eval_df)}
    overall_row.update(eval_df[metric_cols].mean().to_dict())
    summary_df = pd.concat([by_num_laws, pd.DataFrame([overall_row])], ignore_index=True)

    pd.set_option("display.width", 200)
    print("\n[베이스라인: 비율형 Recall@K - train.py 평가와 동일 정의]")
    print(summary_df[["num_laws", "num_queries"] + recall_cols].to_string(index=False))
    print("\n[베이스라인: Hit@K / MRR - KG-search 프로젝트 수치와 비교용]")
    print(summary_df[["num_laws", "num_queries"] + hit_cols + [mrr_col]].to_string(index=False))

    # 7. 결과 저장 (train.py의 eval_results/와 같은 폴더, baseline 접두어)
    eval_dir = os.path.join(os.path.dirname(__file__), "eval_results")
    os.makedirs(eval_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    detailed_csv = os.path.join(eval_dir, f"baseline_eval_detailed_{timestamp}.csv")
    summary_csv = os.path.join(eval_dir, f"baseline_eval_summary_{timestamp}.csv")
    summary_json = os.path.join(eval_dir, f"baseline_eval_summary_{timestamp}.json")

    eval_df.to_csv(detailed_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "method": "pure BGE-M3 cosine (no training)",
            "k_values": K_VALUES,
            "num_test_queries_evaluated": len(eval_df),
            "by_num_laws": by_num_laws.to_dict(orient="records"),
            "overall": overall_row,
            "per_query": eval_df.to_dict(orient="records"),
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 베이스라인 평가 저장 완료!")
    print(f"   📄 세부(질의별): {detailed_csv}")
    print(f"   📄 요약: {summary_csv}")
    print(f"   📄 요약(JSON): {summary_json}")
    print("\n비교 방법: train.py 실행 후 eval_results/test_eval_summary_*.csv의 recall@K와")
    print("이 파일의 recall@K를 나란히 비교하세요. 베이스라인이 더 높다면, HGT 학습이")
    print("BGE 공간의 검색 능력을 훼손하고 있다는 뜻입니다 (Stage 1/2 분리 개조의 근거).")


if __name__ == "__main__":
    main()
