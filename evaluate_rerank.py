"""
evaluate_rerank.py

통합 검색 평가 하네스. 하나의 스크립트로 여러 아키텍처 조합을 평가하고,
설정을 그대로 드러내는 파일명으로 결과를 저장한다. '항 단위'와 '조 단위' 지표를 모두 보고.

축(axis)별 옵션:
  - 질의 인코딩:  Stage2 QueryEncoder(기본) | --no_query_encoder(순수 BGE)
  - 조항 임베딩:  원본 BGE(기본) | --clause_emb data/clause_emb_smooth.safetensors(평활화)
  - 기본 검색:    dense 코사인(기본) | --hybrid(dense + BM25 어휘, RRF 융합)
  - 재랭킹:       --rerank none(기본) | ppr(그래프 PPR) | cross(BGE-reranker cross-encoder)

재랭킹 방식:
  ppr   : 상위 seeds개를 시드로 조항 인접 그래프(형제 항+공유 엔터티)에서 PPR 전파,
          최종 = minmax(기본점수) + beta*(PPR/maxPPR). 구조 정보를 추론 시점에 주입.
  cross : 기본 상위 rerank_topk개를 (질의,조항) 쌍으로 BGE-reranker-v2-m3(cross-encoder)에
          넣어 재정렬. bi-encoder가 놓치는 정밀한 관련성을 잡는다(학습 불필요).

실행 예:
  python evaluate_rerank.py                              # Stage2 + dense, 재랭킹 없음
  python evaluate_rerank.py --rerank ppr --beta 0.5      # + PPR
  python evaluate_rerank.py --rerank cross               # + cross-encoder 재랭킹
  python evaluate_rerank.py --hybrid                     # dense+BM25 하이브리드
  python evaluate_rerank.py --no_query_encoder           # 순수 BGE (베이스라인)

결과: 콘솔 + eval_results/rerank_{설정}_{para|article|summary}_{ts}.csv/.json
"""

import argparse
import os
import json
from datetime import datetime

import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from data_loader import (
    normalize_johang_key, fsc_dataset_preprocessing, encode_texts_cached, make_bge_encoder,
)
from query_encoder import QueryEncoder
from retrieval_common import (
    K_VALUES, build_clause_index, build_retrieval_items, build_clause_adjacency,
    compute_metric_rows, compute_article_metric_rows, summarize_metrics, emb_tag,
)
from ranking_methods import compute_base_scores, compute_ppr_scores, compute_cross_rerank

NODES_CSV = './data/nodes.csv'
TRIPLETS_CSV = './data/triplets.csv'
FSC_XLSX = './data/for_review_corrected.xlsx'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_encoder", default="query_encoder_best.safetensors",
                        help="Stage 2 체크포인트 경로")
    parser.add_argument("--no_query_encoder", action="store_true",
                        help="QueryEncoder 없이 순수 BGE 질의 임베딩 사용")
    parser.add_argument("--clause_emb", default=None,
                        help="조항 임베딩 safetensors 파일 (미지정 시 원본 BGE 캐시)")
    parser.add_argument("--rerank", choices=["none", "ppr", "cross"], default="none",
                        help="재랭킹 방식: none | ppr(그래프) | cross(cross-encoder)")
    parser.add_argument("--ppr", action="store_true", help="(별칭) --rerank ppr 와 동일")
    parser.add_argument("--hybrid", action="store_true", help="dense + BM25 어휘 하이브리드(RRF)")
    parser.add_argument("--beta", type=float, default=0.3, help="PPR 점수 혼합 비중")
    parser.add_argument("--restart", type=float, default=0.5, help="PPR 재시작 확률")
    parser.add_argument("--iters", type=int, default=10, help="PPR 전파 반복 수")
    parser.add_argument("--seeds", type=int, default=20, help="PPR 시드 조항 수")
    parser.add_argument("--rerank_topk", type=int, default=50, help="cross-encoder 재랭킹 후보 수")
    parser.add_argument("--cross_model", default="BAAI/bge-reranker-v2-m3",
                        help="cross-encoder 재랭커 모델")
    parser.add_argument("--max_entity_df", type=int, default=20)
    parser.add_argument("--test_size", type=int, default=100,
                        help="test 질의 수 (평가가능 질의에서 층화추출). 다른 스크립트와 동일 값 사용 필수")
    args = parser.parse_args()

    if args.ppr and args.rerank == "none":   # --ppr 별칭 호환
        args.rerank = "ppr"

    method_parts = ["BGE query" if args.no_query_encoder else "Stage2 QueryEncoder"]
    if args.clause_emb:
        method_parts.append(f"clause_emb={os.path.basename(args.clause_emb)}")
    method_parts.append("hybrid(dense+BM25)" if args.hybrid else "dense")
    if args.rerank == "ppr":
        method_parts.append(f"PPR(beta={args.beta}, restart={args.restart}, seeds={args.seeds})")
    elif args.rerank == "cross":
        method_parts.append(f"cross-encoder({args.cross_model}, topk={args.rerank_topk})")
    method_name = " + ".join(method_parts)
    print(f"=== 평가: {method_name} ===")

    # 1. 데이터 구성 (기존 평가들과 동일한 분할·확장)
    nodes_df = pd.read_csv(NODES_CSV)
    nodes_df['new_johang'] = [
        normalize_johang_key(law_nm, article_number, hang_number)
        for law_nm, article_number, hang_number in zip(
            nodes_df['law_nm'], nodes_df['article_number'], nodes_df['hang_number']
        )
    ]
    fsc = fsc_dataset_preprocessing(file=FSC_XLSX, nodes_df=nodes_df, test_size=args.test_size)
    fsc_test = fsc[fsc['split'] == 'test'].reset_index(drop=True)

    clause_list, clause_texts = build_clause_index(nodes_df)
    items, skipped = build_retrieval_items(fsc_test, clause_list)
    print(f"조항 노드 {len(clause_list):,}개 | 평가 가능 질의 {len(items)}건 (제외 {skipped}건)")

    # 2. 임베딩 준비 (인코딩은 GPU가 있으면 GPU, 끝나면 즉시 해제)
    encoder = make_bge_encoder()
    if args.clause_emb:
        clause_embs = load_file(args.clause_emb)['embeddings']
        assert clause_embs.size(0) == len(clause_list), "조항 임베딩 크기 불일치"
        print(f"조항 임베딩 로드: {args.clause_emb}")
    else:
        clause_embs = encode_texts_cached(encoder, clause_texts, 'clause_embs')
    query_embs = encode_texts_cached(encoder, [it["query"] for it in items], 'fsc_query_embs')
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    clause_embs = F.normalize(clause_embs.float(), dim=-1)
    query_embs = F.normalize(query_embs.float(), dim=-1)

    # 3. 질의 인코딩 (Stage 2 or 순수 BGE)
    if args.no_query_encoder:
        q = query_embs
    else:
        model = QueryEncoder(dim=clause_embs.size(1))
        model.load_state_dict(load_file(args.query_encoder))
        model.eval()
        with torch.no_grad():
            q = model(query_embs)
        print(f"QueryEncoder 로드: {args.query_encoder}")

    # 4. 기본 점수 (dense 또는 dense+BM25 하이브리드) — ranking_methods 공유 함수
    base_scores = compute_base_scores(q, clause_embs, items, clause_texts, args.hybrid)

    # 5. 재랭킹 — ranking_methods 공유 함수 (assemble_context.py와 동일 구현)
    if args.rerank == "ppr":
        triplets_df = pd.read_csv(TRIPLETS_CSV)
        triplets_df['new_johang'] = [
            normalize_johang_key(law_nm, article_number)
            for law_nm, article_number in zip(triplets_df['law_nm'], triplets_df['article_number'])
        ]
        edge_w = build_clause_adjacency(clause_list, triplets_df, args.max_entity_df)
        final_scores = compute_ppr_scores(base_scores, edge_w, len(clause_list),
                                          seeds=args.seeds, beta=args.beta,
                                          restart=args.restart, iters=args.iters)
        full_ranking = final_scores.argsort(dim=1, descending=True)
        full_ranked_lists = [row.tolist() for row in full_ranking]

    elif args.rerank == "cross":
        full_ranked_lists, _ = compute_cross_rerank(base_scores, items, clause_texts,
                                                    cross_model=args.cross_model, topk=args.rerank_topk)

    else:  # none
        full_ranking = base_scores.argsort(dim=1, descending=True)
        full_ranked_lists = [row.tolist() for row in full_ranking]

    # 6. 전체 랭킹 -> 항 단위 / 조 단위 지표
    max_k = max(K_VALUES)
    top_ranked_lists = [r[:max_k] for r in full_ranked_lists]

    para_rows, mrr_col = compute_metric_rows(top_ranked_lists, items, K_VALUES)
    para_df = pd.DataFrame(para_rows)
    para_summary, para_by, para_overall, recall_cols, hit_cols = summarize_metrics(para_df, K_VALUES, mrr_col)

    art_rows, _ = compute_article_metric_rows(full_ranked_lists, items, clause_list, K_VALUES)
    art_df = pd.DataFrame(art_rows)
    art_summary, art_by, art_overall, _, _ = summarize_metrics(art_df, K_VALUES, mrr_col)

    pd.set_option("display.width", 220)
    print(f"\n[항(paragraph) 단위 - 기존 평가들과 동일 정의]")
    print(para_summary[["num_laws", "num_queries"] + recall_cols + [mrr_col]].to_string(index=False))
    print(f"\n[조(article) 단위 - KG-search 프로젝트와 비교 가능한 세밀도]")
    print(art_summary[["num_laws", "num_queries"] + recall_cols + [mrr_col]].to_string(index=False))
    print(f"\n[조 단위 Hit@K]")
    print(art_summary[["num_laws", "num_queries"] + hit_cols].to_string(index=False))

    # 7. 저장. 파일명 규칙:
    #   rerank_{origEmb|smoothEmb}_{stage2|bgeq}_{dense|hybrid}_{none|ppr-b0.5|cross-k50}_{para|article|summary}_{ts}
    eval_dir = os.path.join(os.path.dirname(__file__), "eval_results")
    os.makedirs(eval_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    q_tag = "bgeq" if args.no_query_encoder else "stage2"
    retr_tag = "hybrid" if args.hybrid else "dense"
    if args.rerank == "ppr":
        rr_tag = f"ppr-b{args.beta:g}"
    elif args.rerank == "cross":
        rr_tag = f"cross-k{args.rerank_topk}"
    else:
        rr_tag = "none"
    cfg = f"{emb_tag(args.clause_emb)}_{q_tag}_{retr_tag}_{rr_tag}"
    para_df.to_csv(os.path.join(eval_dir, f"rerank_{cfg}_paragraph_{ts}.csv"), index=False, encoding="utf-8-sig")
    art_df.to_csv(os.path.join(eval_dir, f"rerank_{cfg}_article_{ts}.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(eval_dir, f"rerank_{cfg}_summary_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": ts,
            "method": method_name,
            "hyperparams": vars(args),
            "k_values": K_VALUES,
            "num_test_queries_evaluated": len(para_df),
            "paragraph_level": {
                "by_num_laws": para_by.to_dict(orient="records"),
                "overall": para_overall,
            },
            "article_level": {
                "by_num_laws": art_by.to_dict(orient="records"),
                "overall": art_overall,
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 저장 완료: eval_results/rerank_{cfg}_summary_{ts}.json")


if __name__ == "__main__":
    main()
