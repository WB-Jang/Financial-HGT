"""
evaluate_baseline.py

학습 없이 '순수 BGE-M3 코사인 유사도'만으로 조항 검색 성능을 측정하는 베이스라인 스크립트.

목적:
  train.py(HGT) / train_query_encoder.py(Stage 2)로 학습한 모델과 직접 비교하기 위한 기준선.
  - test 분할(층화추출 시드 42), 조->항 정답 확장, Recall@K 정의 모두 동일
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
import json
import argparse
from datetime import datetime

import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from safetensors.torch import load_file

from data_loader import normalize_johang_key, fsc_dataset_preprocessing, encode_texts_cached
from retrieval_common import (
    K_VALUES, build_clause_index, build_retrieval_items,
    compute_metric_rows, summarize_metrics, emb_tag,
)

NODES_CSV = './data/nodes.csv'
FSC_XLSX = './data/for_review_corrected.xlsx'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clause_emb", default=None,
                        help="조항 임베딩 safetensors 파일 (예: data/clause_emb_smooth.safetensors). "
                             "미지정 시 원본 BGE 임베딩(캐시) 사용")
    parser.add_argument("--test_size", type=int, default=100,
                        help="test 질의 수 (평가가능 질의에서 층화추출). 다른 스크립트와 동일 값 사용 필수")
    args = parser.parse_args()

    method_name = "pure BGE-M3 cosine (no training)"
    if args.clause_emb:
        method_name = f"BGE-M3 cosine + clause emb from {os.path.basename(args.clause_emb)}"
    print(f"=== 베이스라인 평가 (학습 없음): {method_name} ===")

    # 1. 노드 로드 + 조항 키 정규화 (data_loader와 동일)
    nodes_df = pd.read_csv(NODES_CSV)
    nodes_df['new_johang'] = [
        normalize_johang_key(law_nm, article_number, hang_number)
        for law_nm, article_number, hang_number in zip(
            nodes_df['law_nm'], nodes_df['article_number'], nodes_df['hang_number']
        )
    ]

    # 2. fsc 전처리 + train/test 분할 (평가가능 질의에서 층화추출)
    fsc = fsc_dataset_preprocessing(file=FSC_XLSX, nodes_df=nodes_df, test_size=args.test_size)
    fsc_test = fsc[fsc['split'] == 'test'].reset_index(drop=True)
    print(f"test 질의(분할 기준): {len(fsc_test)}건")

    # 3. 조항 인덱스 + 평가 아이템 구성 (공통 모듈)
    clause_list, clause_texts = build_clause_index(nodes_df)
    print(f"검색 대상 조항 노드: {len(clause_list):,}개")
    items, skipped = build_retrieval_items(fsc_test, clause_list)
    print(f"평가 가능 질의: {len(items)}건 (그래프 미매칭 제외 {skipped}건)")
    if not items:
        print("평가 가능한 질의가 없습니다.")
        return

    # 4. 임베딩 준비 (emb_cache/ 재사용 - train.py 실행 시 만든 캐시와 동일 파일)
    encoder = SentenceTransformer('BAAI/bge-m3', device='cpu')
    if args.clause_emb:
        clause_embs = load_file(args.clause_emb)['embeddings']
        assert clause_embs.size(0) == len(clause_list), \
            f"조항 임베딩 크기 불일치: {clause_embs.size(0)} != {len(clause_list)} (데이터가 바뀌었으면 build_smoothed_clause_emb.py를 다시 실행하세요)"
        print(f"조항 임베딩 로드: {args.clause_emb}")
    else:
        clause_embs = encode_texts_cached(encoder, clause_texts, 'clause_embs')
    # 캐시 이름을 hard negative 단계와 동일하게 두어, train.py를 이미 실행했다면 그 캐시가 적중됨
    query_embs = encode_texts_cached(encoder, [it["query"] for it in items], 'fsc_query_embs')

    clause_embs = F.normalize(clause_embs.float(), dim=-1)   # (N, 1024)
    query_embs = F.normalize(query_embs.float(), dim=-1)     # (Q, 1024)

    # 5. 전체 코퍼스 코사인 랭킹 -> 지표 계산 (공통 모듈)
    sims = query_embs @ clause_embs.T                        # (Q, N)
    max_k = min(max(K_VALUES), sims.size(1))
    topk = sims.topk(max_k, dim=1).indices                   # (Q, max_k)
    ranked_lists = [row.tolist() for row in topk]

    rows, mrr_col = compute_metric_rows(ranked_lists, items, K_VALUES)
    eval_df = pd.DataFrame(rows)
    summary_df, by_num_laws, overall_row, recall_cols, hit_cols = summarize_metrics(eval_df, K_VALUES, mrr_col)

    pd.set_option("display.width", 200)
    print("\n[베이스라인: 비율형 Recall@K - train.py 평가와 동일 정의]")
    print(summary_df[["num_laws", "num_queries"] + recall_cols].to_string(index=False))
    print("\n[베이스라인: Hit@K / MRR - KG-search 프로젝트 수치와 비교용]")
    print(summary_df[["num_laws", "num_queries"] + hit_cols + [mrr_col]].to_string(index=False))

    # 6. 결과 저장. 파일명 규칙: baseline_{origEmb|smoothEmb}_{summary|detailed}_{ts}
    eval_dir = os.path.join(os.path.dirname(__file__), "eval_results")
    os.makedirs(eval_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = emb_tag(args.clause_emb)

    detailed_csv = os.path.join(eval_dir, f"baseline_{tag}_detailed_{timestamp}.csv")
    summary_csv = os.path.join(eval_dir, f"baseline_{tag}_summary_{timestamp}.csv")
    summary_json = os.path.join(eval_dir, f"baseline_{tag}_summary_{timestamp}.json")

    eval_df.to_csv(detailed_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "method": method_name,
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
    print("\n비교 방법: train_query_encoder.py 실행 후 eval_results/stage2_eval_summary_*.csv와")
    print("이 파일을 나란히 비교하세요. 같은 test 분할, 같은 지표입니다.")


if __name__ == "__main__":
    main()
