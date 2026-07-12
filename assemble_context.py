"""
assemble_context.py

'폭 우선(breadth-first) 컨텍스트 조립' — 이 프로젝트의 실제 목표 구현.

아이디어:
  질의와 관련된 조항을 검색(학습된 QueryEncoder = 관련성 필터)한 뒤, 그중에서
  '구조적으로 가장 넓은 범위를 커버하는(다른 법률과 가장 많이 연결된)' 골격 조항을
  컨텍스트 맨 앞에 세운다. 이 골격이 하위 질문 답변의 넓은 배경 지식이 된다.

검색 recall이 아니라 "앞단 조항이 질의의 법적 범위를 얼마나 넓게 감싸는가"로 성패를 본다.

두 가지 모드:
  1) 단건 조립:  --query "질의문"
     → 관련 top-N 검색 → 최고 breadth 조항을 골격으로 승격 → 정렬된 컨텍스트 출력

  2) 배치 평가:  (옵션 없이 실행)
     test 셋 전체에 대해 '폭 우선' vs '관련도 우선(baseline)' 리드 조항을 비교:
       - lead_n_cross_laws : 리드 조항이 참조하는 타법 수 (폭이 실제로 넓어졌는가)
       - reach_coverage    : 리드 조항의 (자법 ∪ 참조 graph법)이 질의의 정답 법령을 덮는 비율
                             (= 앞단이 질의의 법적 범위를 구조적으로 감싸는 정도)
     결과: eval_results/breadth_assembly_{ts}.csv / .json

실행:
  python assemble_context.py --query "준법감시인의 자격요건은?"
  python assemble_context.py --test_size 300 --pool_n 20
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
                clause_embs=clause_embs, encoder=encoder)


def load_query_encoder(args, dim):
    if args.no_query_encoder:
        return None
    model = QueryEncoder(dim=dim)
    model.load_state_dict(load_file(args.query_encoder))
    model.eval()
    return model


def rank_query(q_bge, model, clause_embs):
    """질의 BGE 임베딩 -> (Stage2) -> 조항 유사도 벡터."""
    q = q_bge if model is None else model(q_bge)
    return (F.normalize(q, dim=-1) @ clause_embs.T).squeeze(0)


def _minmax(v):
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if (hi - lo) > 1e-12 else v * 0.0


def pick_leads(sims, breadth, pool_n, alpha=1.0):
    """관련도 top-N 후보에서 관련도-리드와 (관련도·폭 블렌드) 리드를 선택.

    블렌드 점수 = (1-alpha)*관련도_norm + alpha*폭_norm  (풀 내부 min-max 정규화)
      alpha=0 -> 순수 관련도(=관련도 1위), alpha=1 -> 순수 폭.
    반환: (relevance_lead_idx, blend_lead_idx, pool_indices, blend_lead_rank)
    """
    pool = sims.topk(min(pool_n, sims.numel())).indices
    rel_lead = pool[0].item()
    rel_norm = _minmax(sims[pool])
    brd_norm = _minmax(breadth[pool])
    blend = (1 - alpha) * rel_norm + alpha * brd_norm
    b_pos = int(torch.argmax(blend).item())
    return rel_lead, pool[b_pos].item(), pool.tolist(), b_pos


# ── 모드 1: 단건 조립 ─────────────────────────────────────────────────────────

def run_single(args, ctx):
    model = load_query_encoder(args, ctx['clause_embs'].size(1))
    with torch.no_grad():
        q_bge = F.normalize(torch.tensor(ctx['encoder'].encode([args.query])).float(), dim=-1)
    del ctx['encoder']
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        sims = rank_query(q_bge, model, ctx['clause_embs'])
    rel_lead, breadth_lead, pool, _ = pick_leads(sims, ctx['breadth'], args.pool_n, args.alpha)

    cl, texts, breadth = ctx['clause_list'], ctx['clause_texts'], ctx['breadth']
    print(f"\n질의: {args.query}   (alpha={args.alpha})")
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
        print(f"  [{rank}] {cl[idx]}  (sim={sims[idx]:.3f}, breadth={breadth[idx]:.3f})")
    print("=" * 78)
    print("→ 골격 조항을 맨 앞에 두고, 이 넓은 배경 위에서 하위 질문들을 답변하도록 LLM에 전달")


# ── 모드 2: 배치 평가 (폭 우선 vs 관련도 우선) ────────────────────────────────

def run_eval(args, ctx):
    fsc = fsc_dataset_preprocessing(file=FSC_XLSX, nodes_df=ctx['nodes_df'], test_size=args.test_size)
    fsc_test = fsc[fsc['split'] == 'test'].reset_index(drop=True)
    items, skipped = build_retrieval_items(fsc_test, ctx['clause_list'])
    print(f"평가 가능 질의 {len(items)}건 (제외 {skipped}건)")

    model = load_query_encoder(args, ctx['clause_embs'].size(1))
    q_bge = F.normalize(encode_texts_cached(ctx['encoder'], [it['query'] for it in items],
                                            'fsc_query_embs').float(), dim=-1)
    del ctx['encoder']
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        q = q_bge if model is None else model(q_bge)
        q = F.normalize(q, dim=-1)
        sims_all = q @ ctx['clause_embs'].T           # (Q, N)

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

    print("\n=== alpha 스윕: 관련도(alpha=0) ↔ 폭(alpha=1) 블렌드 ===")
    print("alpha | 리드타법수 | 커버리지(전체) | 커버리지(단일법) | 커버리지(다법≥3) | 폭리드 관련도순위")
    sweep_rows = []
    best = None
    for a in alphas:
        leads = [pick_leads(sims_all[i], breadth, args.pool_n, a) for i in range(len(items))]
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

    eval_dir = os.path.join(os.path.dirname(__file__), "eval_results")
    os.makedirs(eval_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(sweep_rows).to_csv(
        os.path.join(eval_dir, f"breadth_assembly_sweep_{ts}.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(eval_dir, f"breadth_assembly_summary_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump({'timestamp': ts, 'pool_n': args.pool_n, 'breadth_weights': args.breadth_weights,
                   'n_single_law': len(single), 'n_multi_law(>=3)': len(multi),
                   'sweep': sweep_rows}, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 저장: eval_results/breadth_assembly_sweep_{ts}.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=None, help="단건 조립 모드: 질의문")
    parser.add_argument("--query_encoder", default="query_encoder_best.safetensors")
    parser.add_argument("--no_query_encoder", action="store_true", help="순수 BGE 검색")
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
