"""
evaluate_rerank.py

최종 검색 파이프라인 평가 스크립트: Stage 2 QueryEncoder 랭킹 위에
(선택) PPR(Personalized PageRank) 그래프 재랭킹을 얹고,
'항 단위'와 '조 단위' 지표를 모두 보고한다.

PPR 재랭킹 원리:
  1. QueryEncoder 코사인 랭킹 상위 seeds개 조항을 시드로 선택
  2. 조항 인접 그래프(형제 항 + 공유 엔터티) 위에서 시드로부터 확률 질량을 전파
     p <- restart * seed + (1 - restart) * P^T p   (iters회 반복)
  3. 최종 점수 = 코사인 + beta * (PPR / max(PPR))  로 전체 재랭킹
  임베딩을 건드리지 않고 검색 시점에만 그래프를 쓰므로, 평활화와 달리
  Stage 2 학습과 중복되지 않는 방식으로 구조 정보를 주입한다.

조 단위 지표:
  항 단위 랭킹을 조(article) 단위로 접어서(조 점수 = 소속 항 최고 점수) 계산.
  조 단위 검색인 KG-search_PPR_GNN_Transformer와 공정 비교가 가능한 수치.

실행:
  python evaluate_rerank.py                       # Stage 2만 (PPR 없음) + 양쪽 지표
  python evaluate_rerank.py --ppr                 # Stage 2 + PPR 재랭킹
  python evaluate_rerank.py --ppr --beta 0.5      # PPR 비중 상향
  python evaluate_rerank.py --no_query_encoder    # 순수 BGE 기반 (베이스라인 확인용)

결과: 콘솔 + eval_results/rerank_eval_*.csv / .json
"""

import argparse
import os
import json
from datetime import datetime

import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from safetensors.torch import load_file

from data_loader import normalize_johang_key, fsc_dataset_preprocessing, encode_texts_cached
from query_encoder import QueryEncoder
from retrieval_common import (
    K_VALUES, build_clause_index, build_retrieval_items, build_clause_adjacency,
    compute_metric_rows, compute_article_metric_rows, summarize_metrics, emb_tag,
)

NODES_CSV = './data/nodes.csv'
TRIPLETS_CSV = './data/triplets.csv'
FSC_XLSX = './data/for_review_corrected.xlsx'


def build_ppr_operator(edge_w, n):
    """PPR 전파에 필요한 텐서(src, dst, 전이확률)를 준비한다."""
    pairs = torch.tensor(list(edge_w.keys()), dtype=torch.long)
    w = torch.tensor(list(edge_w.values()), dtype=torch.float32)
    src = torch.cat([pairs[:, 0], pairs[:, 1]])
    dst = torch.cat([pairs[:, 1], pairs[:, 0]])
    ww = torch.cat([w, w])

    w_out = torch.zeros(n).index_add_(0, src, ww)      # 노드별 나가는 가중치 합
    trans = ww / w_out[src].clamp(min=1e-12)            # 엣지별 전이 확률 w_ij / sum_j w_ij
    isolated = w_out == 0                                # 이웃 없는 노드 (질량 자기 유지)
    return src, dst, trans, isolated


def personalized_pagerank(seed_vec, src, dst, trans, isolated, restart=0.5, iters=10):
    """단일 질의의 PPR 점수 벡터 계산. seed_vec: (N,) 합=1"""
    p = seed_vec.clone()
    for _ in range(iters):
        spread = torch.zeros_like(p).index_add_(0, dst, p[src] * trans)
        spread = spread + p * isolated.float()           # 고립 노드는 질량 유지
        p = restart * seed_vec + (1 - restart) * spread
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_encoder", default="query_encoder_best.safetensors",
                        help="Stage 2 체크포인트 경로")
    parser.add_argument("--no_query_encoder", action="store_true",
                        help="QueryEncoder 없이 순수 BGE 질의 임베딩 사용")
    parser.add_argument("--clause_emb", default=None,
                        help="조항 임베딩 safetensors 파일 (미지정 시 원본 BGE 캐시)")
    parser.add_argument("--ppr", action="store_true", help="PPR 그래프 재랭킹 활성화")
    parser.add_argument("--beta", type=float, default=0.3, help="PPR 점수 혼합 비중")
    parser.add_argument("--restart", type=float, default=0.5, help="PPR 재시작 확률")
    parser.add_argument("--iters", type=int, default=10, help="PPR 전파 반복 수")
    parser.add_argument("--seeds", type=int, default=20, help="PPR 시드 조항 수")
    parser.add_argument("--max_entity_df", type=int, default=20)
    args = parser.parse_args()

    method_parts = []
    method_parts.append("BGE query" if args.no_query_encoder else "Stage2 QueryEncoder")
    if args.clause_emb:
        method_parts.append(f"clause_emb={os.path.basename(args.clause_emb)}")
    if args.ppr:
        method_parts.append(f"PPR(beta={args.beta}, restart={args.restart}, seeds={args.seeds})")
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
    fsc = fsc_dataset_preprocessing(file=FSC_XLSX, nodes_df=nodes_df)
    fsc_test = fsc[fsc['split'] == 'test'].reset_index(drop=True)

    clause_list, clause_texts = build_clause_index(nodes_df)
    items, skipped = build_retrieval_items(fsc_test, clause_list)
    print(f"조항 노드 {len(clause_list):,}개 | 평가 가능 질의 {len(items)}건 (제외 {skipped}건)")

    # 2. 임베딩 준비
    encoder = SentenceTransformer('BAAI/bge-m3', device='cpu')
    if args.clause_emb:
        clause_embs = load_file(args.clause_emb)['embeddings']
        assert clause_embs.size(0) == len(clause_list), "조항 임베딩 크기 불일치"
        print(f"조항 임베딩 로드: {args.clause_emb}")
    else:
        clause_embs = encode_texts_cached(encoder, clause_texts, 'clause_embs')
    query_embs = encode_texts_cached(encoder, [it["query"] for it in items], 'fsc_query_embs')
    del encoder

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

    sims = q @ clause_embs.T                              # (Q, N) 기본 점수

    # 4. (선택) PPR 재랭킹
    if args.ppr:
        triplets_df = pd.read_csv(TRIPLETS_CSV)
        triplets_df['new_johang'] = [
            normalize_johang_key(law_nm, article_number)
            for law_nm, article_number in zip(triplets_df['law_nm'], triplets_df['article_number'])
        ]
        edge_w = build_clause_adjacency(clause_list, triplets_df, args.max_entity_df)
        src, dst, trans, isolated = build_ppr_operator(edge_w, len(clause_list))

        final_scores = torch.zeros_like(sims)
        for i in range(sims.size(0)):
            seed_idx = sims[i].topk(args.seeds).indices
            seed_scores = sims[i][seed_idx]
            seed_scores = (seed_scores - seed_scores.min()).clamp(min=1e-6)
            seed_vec = torch.zeros(sims.size(1))
            seed_vec[seed_idx] = seed_scores / seed_scores.sum()

            ppr = personalized_pagerank(seed_vec, src, dst, trans, isolated,
                                        restart=args.restart, iters=args.iters)
            ppr_max = ppr.max()
            ppr_n = ppr / ppr_max if ppr_max > 0 else ppr
            final_scores[i] = sims[i] + args.beta * ppr_n
        print(f"PPR 재랭킹 완료 ({sims.size(0)}개 질의)")
    else:
        final_scores = sims

    # 5. 전체 랭킹 -> 항 단위 / 조 단위 지표
    full_ranking = final_scores.argsort(dim=1, descending=True)   # (Q, N) 전체 랭킹
    full_ranked_lists = [row.tolist() for row in full_ranking]
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

    # 6. 저장. 파일명 규칙: rerank_{origEmb|smoothEmb}_{stage2|bgeq}_{noppr|ppr-b0.5}_{para|article|summary}_{ts}
    eval_dir = os.path.join(os.path.dirname(__file__), "eval_results")
    os.makedirs(eval_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    q_tag = "bgeq" if args.no_query_encoder else "stage2"
    ppr_tag = f"ppr-b{args.beta:g}" if args.ppr else "noppr"
    cfg = f"{emb_tag(args.clause_emb)}_{q_tag}_{ppr_tag}"
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
