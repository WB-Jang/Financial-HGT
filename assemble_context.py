"""
assemble_context.py

'폭 우선(breadth-first) 컨텍스트 조립' — 이 프로젝트의 실제 목표 구현.

아이디어:
  질의와 관련된 조항을 검색(=관련성 필터)한 뒤, 그중에서 '구조적으로 가장 넓은 범위를
  커버하는(다른 법률과 가장 많이 연결된)' 골격 조항을 컨텍스트 맨 앞에 세운다. 이 골격이
  하위 질문 답변의 넓은 배경 지식이 된다.

검색 recall이 아니라 "앞단 조항이 질의의 법적 범위를 얼마나 넓게 감싸는가"로 성패를 본다.

'관련성 필터' 자체는 여러 방식으로 만들 수 있다 — evaluate_rerank.py와 동일한 플래그로
선택한다(같은 ranking_methods.py 구현 재사용, 두 스크립트가 같은 후보 풀을 다른 지표로 봄):
  --no_query_encoder      : 순수 BGE dense
  (기본)                   : Stage2 QueryEncoder dense
  --hybrid                 : dense + BM25 어휘 (RRF 융합)
  --rerank ppr             : + PPR 그래프 재랭킹 (breadth의 kg_degree와 같은 인접 그래프 사용)
  --rerank cross            : + cross-encoder(BGE-reranker) 재랭킹

두 가지 모드:
  1) 단건 조립:  --query "질의문"
     → 선택한 방식으로 관련 top-N 검색 → 그 후보 안에서 관련도-폭 블렌드로 리드 선택

  2) 배치 평가:  (옵션 없이 실행, --alpha_sweep으로 alpha 스윕)
     test 셋 전체에 대해 alpha별 리드 조항의 폭/커버리지를 측정:
       - lead_cross_laws : 리드 조항이 참조하는 타법 수 (폭이 실제로 넓어졌는가)
       - coverage        : 리드 조항의 (자법 ∪ 참조 graph법)이 질의의 정답 법령을 덮는 비율
                            (= 앞단이 질의의 법적 범위를 구조적으로 감싸는 정도)
     결과: eval_results/breadth_assembly_sweep_{method태그}_{ts}.csv

실행:
  python assemble_context.py --query "준법감시인의 자격요건은?"
  python assemble_context.py --test_size 300 --pool_n 20                     # Stage2 dense (기본)
  python assemble_context.py --test_size 300 --rerank ppr --beta 0.5         # + PPR
  python assemble_context.py --test_size 300 --rerank cross                  # + cross-encoder
  python assemble_context.py --test_size 300 --hybrid                        # + BM25 하이브리드
  # 네 방식의 breadth_assembly_sweep_*.csv를 나란히 놓으면 "어떤 재랭킹이 폭 지표에
  # 가장 유리한가"를 직접 비교할 수 있다.
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
    build_clause_index, build_retrieval_items, build_clause_adjacency, law_of,
)
from ranking_methods import compute_base_scores, compute_ppr_scores, compute_cross_rerank
from breadth import compute_breadth, kg_degree_from_adjacency

NODES_CSV = './data/nodes.csv'
TRIPLETS_CSV = './data/triplets.csv'
FSC_XLSX = './data/for_review_corrected.xlsx'


def load_common(args):
    """nodes/clause/breadth/embeddings/model 로드 (두 모드 공용)."""
    nodes_df = pd.read_csv(NODES_CSV)
    nodes_df['new_johang'] = [
        normalize_johang_key(l, a, h)
        for l, a, h in zip(nodes_df['law_nm'], nodes_df['article_number'], nodes_df['hang_number'])
    ]
    clause_list, clause_texts = build_clause_index(nodes_df)

    triplets_df = pd.read_csv(TRIPLETS_CSV)
    triplets_df['new_johang'] = [
        normalize_johang_key(l, a)
        for l, a in zip(triplets_df['law_nm'], triplets_df['article_number'])
    ]
    edge_w = build_clause_adjacency(clause_list, triplets_df, args.max_entity_df, verbose=False)
    kg_deg = kg_degree_from_adjacency(edge_w, len(clause_list))

    breadth, comp = compute_breadth(nodes_df, clause_list, kg_degree=kg_deg,
                                    weights=tuple(args.breadth_weights))
    breadth = torch.tensor(breadth, dtype=torch.float32)
    ref_graph_laws = list(comp['ref_graph_laws'])   # 조항별 참조 graph 법령 집합
    clause_law = [law_of(c) for c in clause_list]

    # 조항 임베딩 (캐시)
    encoder = make_bge_encoder()
    clause_embs = F.normalize(encode_texts_cached(encoder, clause_texts, 'clause_embs').float(), dim=-1)
    return dict(nodes_df=nodes_df, clause_list=clause_list, clause_texts=clause_texts,
                breadth=breadth, ref_graph_laws=ref_graph_laws, clause_law=clause_law,
                clause_embs=clause_embs, encoder=encoder,
                edge_w=edge_w)   # PPR 재랭킹이 breadth의 kg_degree와 동일 인접그래프를 재사용


def method_tag(args):
    """실행 설정을 드러내는 짧은 태그 (결과 파일명/로그용) — evaluate_rerank.py와 동일 규칙."""
    q_tag = "bgeq" if args.no_query_encoder else "stage2"
    retr_tag = "hybrid" if args.hybrid else "dense"
    if args.rerank == "ppr":
        rr_tag = f"ppr-b{args.beta:g}"
    elif args.rerank == "cross":
        rr_tag = f"cross-k{args.rerank_topk}"
    else:
        rr_tag = "none"
    return f"{q_tag}_{retr_tag}_{rr_tag}"


def compute_final_scores(args, ctx, q, items):
    """선택한 방식(dense/hybrid + none/ppr/cross)으로 (Q,N) 최종 점수 또는
    (cross 전용) 전체 랭킹+topk점수를 계산한다.
    반환: ('scores', (Q,N) tensor) 또는 ('cross', (full_ranked_lists, topk_scores))
    """
    base_scores = compute_base_scores(q, ctx['clause_embs'], items, ctx['clause_texts'], args.hybrid)
    if args.rerank == "ppr":
        final = compute_ppr_scores(base_scores, ctx['edge_w'], len(ctx['clause_list']),
                                   seeds=args.seeds, beta=args.beta,
                                   restart=args.restart, iters=args.iters)
        return 'scores', final
    elif args.rerank == "cross":
        full_ranked_lists, topk_scores = compute_cross_rerank(
            base_scores, items, ctx['clause_texts'],
            cross_model=args.cross_model, topk=args.rerank_topk)
        return 'cross', (full_ranked_lists, topk_scores)
    else:
        return 'scores', base_scores


def build_pools(kind, payload, pool_n):
    """방식에 무관하게 질의별 (pool_indices, pool_scores) 리스트로 통일."""
    pools = []
    if kind == 'cross':
        full_ranked_lists, topk_scores = payload
        for idx_list, score_list in zip(full_ranked_lists, topk_scores):
            n = min(pool_n, len(score_list))
            pools.append((idx_list[:n], score_list[:n]))
    else:
        final = payload
        for i in range(final.size(0)):
            k = min(pool_n, final.size(1))
            top = final[i].topk(k)
            pools.append((top.indices.tolist(), top.values.tolist()))
    return pools


def load_query_encoder(args, dim):
    if args.no_query_encoder:
        return None
    model = QueryEncoder(dim=dim)
    model.load_state_dict(load_file(args.query_encoder))
    model.eval()
    return model


def _minmax(v):
    v = torch.as_tensor(v, dtype=torch.float32)
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if (hi - lo) > 1e-12 else v * 0.0


def pick_leads_from_pool(pool_idx, pool_scores, breadth, alpha=1.0):
    """이미 구성된 후보 풀(어떤 방식으로 만들어졌든 무관: dense/hybrid/ppr/cross)에서
    관련도-리드와 (관련도·폭 블렌드) 리드를 선택.

    블렌드 점수 = (1-alpha)*관련도_norm + alpha*폭_norm  (풀 내부 min-max 정규화)
      alpha=0 -> 순수 관련도(=풀의 1번째, 이미 방식별 점수로 정렬돼 있음), alpha=1 -> 순수 폭.
    반환: (relevance_lead_idx, blend_lead_idx, pool_indices, blend_lead_rank)
    """
    rel_lead = pool_idx[0]
    rel_norm = _minmax(pool_scores)
    brd_norm = _minmax(breadth[torch.as_tensor(pool_idx, dtype=torch.long)])
    blend = (1 - alpha) * rel_norm + alpha * brd_norm
    b_pos = int(torch.argmax(blend).item())
    return rel_lead, pool_idx[b_pos], pool_idx, b_pos


# ── 모드 1: 단건 조립 ─────────────────────────────────────────────────────────

def run_single(args, ctx):
    tag = method_tag(args)
    model = load_query_encoder(args, ctx['clause_embs'].size(1))
    with torch.no_grad():
        q_bge = F.normalize(torch.tensor(ctx['encoder'].encode([args.query])).float(), dim=-1)
        q = q_bge if model is None else model(q_bge)
        q = F.normalize(q, dim=-1)
    del ctx['encoder']
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    items = [{"query": args.query}]
    kind, payload = compute_final_scores(args, ctx, q, items)
    pool_idx, pool_scores = build_pools(kind, payload, args.pool_n)[0]
    rel_lead, breadth_lead, pool, _ = pick_leads_from_pool(pool_idx, pool_scores, ctx['breadth'], args.alpha)

    cl, texts, breadth = ctx['clause_list'], ctx['clause_texts'], ctx['breadth']
    score_of = dict(zip(pool_idx, pool_scores))
    print(f"\n질의: {args.query}   (방식={tag}, alpha={args.alpha})")
    print("=" * 78)
    print(f"[골격(framework) 조항 - 앞단 컨텍스트]  breadth={breadth[breadth_lead]:.3f}")
    print(f"  {cl[breadth_lead]}  (타법 참조법: {', '.join(sorted(ctx['ref_graph_laws'][breadth_lead])) or '-'})")
    print(f"  {texts[breadth_lead][:220]}...")
    print(f"\n[참고: 관련도 1위 조항]  {cl[rel_lead]}  breadth={breadth[rel_lead]:.3f}")
    print("\n[상세(detail) 조항 - 관련도순, 골격 제외]")
    rank = 0
    for idx in pool:
        if idx == breadth_lead:
            continue
        rank += 1
        if rank > args.detail_k:
            break
        print(f"  [{rank}] {cl[idx]}  (score={score_of[idx]:.3f}, breadth={breadth[idx]:.3f})")
    print("=" * 78)
    print("→ 골격 조항을 맨 앞에 두고, 이 넓은 배경 위에서 하위 질문들을 답변하도록 LLM에 전달")


# ── 모드 2: 배치 평가 (폭 우선 vs 관련도 우선) ────────────────────────────────

def run_eval(args, ctx):
    tag = method_tag(args)
    if args.rerank == "cross" and args.pool_n > args.rerank_topk:
        print(f"경고: pool_n({args.pool_n}) > rerank_topk({args.rerank_topk}) — "
              f"cross-encoder가 채점한 후보보다 풀이 커서 topk까지만 사용합니다.")

    fsc = fsc_dataset_preprocessing(file=FSC_XLSX, nodes_df=ctx['nodes_df'], test_size=args.test_size)
    fsc_test = fsc[fsc['split'] == 'test'].reset_index(drop=True)
    items, skipped = build_retrieval_items(fsc_test, ctx['clause_list'])
    print(f"평가 가능 질의 {len(items)}건 (제외 {skipped}건) | 방식={tag}")

    model = load_query_encoder(args, ctx['clause_embs'].size(1))
    q_bge = F.normalize(encode_texts_cached(ctx['encoder'], [it['query'] for it in items],
                                            'fsc_query_embs').float(), dim=-1)
    del ctx['encoder']
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        q = q_bge if model is None else model(q_bge)
        q = F.normalize(q, dim=-1)

    # 선택한 방식(dense/hybrid + none/ppr/cross)으로 후보 풀을 한 번만 구성.
    # alpha 스윕은 이 풀 위에서 블렌드만 다시 계산하므로(재검색 없음) 저렴하다.
    kind, payload = compute_final_scores(args, ctx, q, items)
    pools = build_pools(kind, payload, args.pool_n)

    breadth, ref_graph_laws, clause_law = ctx['breadth'], ctx['ref_graph_laws'], ctx['clause_law']
    num_laws = [it['num_laws'] for it in items]
    targets = [{clause_law[p] for p in it['pos_idxs']} for it in items]

    def reach(idx):
        return {clause_law[idx]} | set(ref_graph_laws[idx])

    def coverage(lead_idx, i):
        t = targets[i]
        return len(t & reach(lead_idx)) / len(t) if t else None

    # alpha 스윕: 0(순수 관련도) ~ 1(순수 폭). 관련성-폭 트레이드오프의 최적점 탐색.
    alphas = args.alpha_sweep if args.alpha_sweep else [args.alpha]
    single = [i for i, n in enumerate(num_laws) if n == 1]
    multi = [i for i, n in enumerate(num_laws) if n and n >= 3]   # 다법(교차) 질의

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else float('nan')

    print(f"\n=== alpha 스윕: 관련도(alpha=0) ↔ 폭(alpha=1) 블렌드  [방식={tag}] ===")
    print("alpha | 리드타법수 | 커버리지(전체) | 커버리지(단일법) | 커버리지(다법≥3) | 폭리드 관련도순위")
    sweep_rows = []
    best = None
    for a in alphas:
        leads = [pick_leads_from_pool(pools[i][0], pools[i][1], breadth, a) for i in range(len(items))]
        lead_idx = [L[1] for L in leads]
        relranks = [L[3] for L in leads]
        cov_all = [coverage(lead_idx[i], i) for i in range(len(items))]
        cross = [len(ref_graph_laws[lead_idx[i]]) for i in range(len(items))]
        r = {
            'alpha': a,
            'lead_cross_laws': mean(cross),
            'coverage_all': mean(cov_all),
            'coverage_single': mean([cov_all[i] for i in single]),
            'coverage_multi3': mean([cov_all[i] for i in multi]),
            'blend_lead_relrank': mean(relranks),
        }
        sweep_rows.append(r)
        print(f" {a:.2f} |   {r['lead_cross_laws']:.2f}    |     {r['coverage_all']:.3f}    |"
              f"     {r['coverage_single']:.3f}     |     {r['coverage_multi3']:.3f}     |    {r['blend_lead_relrank']:.1f}")
        if best is None or r['coverage_all'] > best['coverage_all']:
            best = r

    print(f"\n권장: 전체 커버리지 최대 alpha={best['alpha']:.2f} "
          f"(단일법은 관련도우선이 유리, 다법일수록 폭이 유리 — 목표에 따라 조정)")
    print("주: coverage는 '리드가 질의의 정답 법령을 덮는 비율' 프록시. 최종 판단은 종단(LLM 답변) 평가로.")
    print(f"방식 간 비교: 이 sweep 표를 다른 --hybrid/--rerank 조합 실행 결과와 나란히 놓고,")
    print(f"같은 alpha에서 lead_cross_laws/coverage가 어느 방식이 가장 높은지 비교하세요.")

    eval_dir = os.path.join(os.path.dirname(__file__), "eval_results")
    os.makedirs(eval_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(sweep_rows).to_csv(
        os.path.join(eval_dir, f"breadth_assembly_sweep_{tag}_{ts}.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(eval_dir, f"breadth_assembly_summary_{tag}_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump({'timestamp': ts, 'method_tag': tag, 'pool_n': args.pool_n,
                   'breadth_weights': args.breadth_weights,
                   'n_single_law': len(single), 'n_multi_law(>=3)': len(multi),
                   'sweep': sweep_rows}, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 저장: eval_results/breadth_assembly_sweep_{tag}_{ts}.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=None, help="단건 조립 모드: 질의문")
    parser.add_argument("--query_encoder", default="query_encoder_best.safetensors")
    parser.add_argument("--no_query_encoder", action="store_true", help="순수 BGE 검색")
    # 관련성 필터 방식 — evaluate_rerank.py와 동일 플래그 (ranking_methods.py 공유 구현)
    parser.add_argument("--hybrid", action="store_true", help="dense + BM25 어휘 하이브리드(RRF)")
    parser.add_argument("--rerank", choices=["none", "ppr", "cross"], default="none",
                        help="후보 풀 재랭킹: none | ppr(그래프) | cross(cross-encoder)")
    parser.add_argument("--beta", type=float, default=0.3, help="PPR 점수 혼합 비중")
    parser.add_argument("--restart", type=float, default=0.5, help="PPR 재시작 확률")
    parser.add_argument("--iters", type=int, default=10, help="PPR 전파 반복 수")
    parser.add_argument("--seeds", type=int, default=20, help="PPR 시드 조항 수")
    parser.add_argument("--rerank_topk", type=int, default=50, help="cross-encoder 재랭킹 후보 수")
    parser.add_argument("--cross_model", default="BAAI/bge-reranker-v2-m3", help="cross-encoder 모델")
    parser.add_argument("--pool_n", type=int, default=20, help="관련도 top-N 후보 풀 (이 안에서 리드 선택)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="리드 선택 시 폭 비중 (0=순수 관련도, 1=순수 폭). 단건 모드/스윕 미지정 시 사용")
    parser.add_argument("--alpha_sweep", type=float, nargs="*", default=[0.0, 0.25, 0.5, 0.75, 1.0],
                        help="배치 평가에서 스윕할 alpha 목록 (빈 값이면 --alpha 단일)")
    parser.add_argument("--detail_k", type=int, default=8, help="단건 모드: 상세 조항 출력 수")
    parser.add_argument("--breadth_weights", type=float, nargs=4, default=[0.5, 0.2, 0.1, 0.2],
                        help="breadth 가중치 (n_cross_laws, n_cross_refs, n_intra_refs, kg_degree)")
    parser.add_argument("--max_entity_df", type=int, default=20)
    parser.add_argument("--test_size", type=int, default=100)
    args = parser.parse_args()

    ctx = load_common(args)
    if args.query:
        run_single(args, ctx)
    else:
        run_eval(args, ctx)


if __name__ == "__main__":
    main()
