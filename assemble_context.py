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


def pick_leads(sims, breadth, pool_n):
    """관련도 top-N 후보에서 관련도-리드와 폭-리드를 선택.
    반환: (relevance_lead_idx, breadth_lead_idx, pool_indices, breadth_lead_rank)"""
    pool = sims.topk(min(pool_n, sims.numel())).indices
    rel_lead = pool[0].item()
    pool_breadth = breadth[pool]
    b_pos = int(torch.argmax(pool_breadth).item())
    breadth_lead = pool[b_pos].item()
    return rel_lead, breadth_lead, pool.tolist(), b_pos


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
    rel_lead, breadth_lead, pool, _ = pick_leads(sims, ctx['breadth'], args.pool_n)

    cl, texts, breadth = ctx['clause_list'], ctx['clause_texts'], ctx['breadth']
    print(f"\n질의: {args.query}")
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
    rows = []
    for i, it in enumerate(items):
        target_laws = {clause_law[p] for p in it['pos_idxs']}       # 질의의 정답 법령 집합
        rel_lead, breadth_lead, pool, b_rank = pick_leads(sims_all[i], breadth, args.pool_n)

        def reach(idx):
            return {clause_law[idx]} | set(ref_graph_laws[idx])

        def cov(idx):
            if not target_laws:
                return None
            return len(target_laws & reach(idx)) / len(target_laws)

        rows.append({
            'query': it['query'],
            'num_laws': it['num_laws'],
            'rel_lead_cross_laws': len(ref_graph_laws[rel_lead]),
            'breadth_lead_cross_laws': len(ref_graph_laws[breadth_lead]),
            'rel_lead_coverage': cov(rel_lead),
            'breadth_lead_coverage': cov(breadth_lead),
            'breadth_lead_relrank': b_rank,     # 폭-리드가 관련도 몇 위였는지 (0=관련도 1위와 동일)
        })

    df = pd.DataFrame(rows)
    ov = {
        'num_queries': len(df),
        'rel_lead_cross_laws(mean)': df['rel_lead_cross_laws'].mean(),
        'breadth_lead_cross_laws(mean)': df['breadth_lead_cross_laws'].mean(),
        'rel_lead_coverage(mean)': df['rel_lead_coverage'].mean(),
        'breadth_lead_coverage(mean)': df['breadth_lead_coverage'].mean(),
        'breadth_lead_relrank(mean)': df['breadth_lead_relrank'].mean(),
    }

    print("\n=== 폭 우선(breadth-first) vs 관련도 우선(relevance-first) 리드 조항 ===")
    print(f"리드 조항의 평균 타법 참조 수:  관련도우선 {ov['rel_lead_cross_laws(mean)']:.2f}"
          f"  →  폭우선 {ov['breadth_lead_cross_laws(mean)']:.2f}")
    print(f"질의 법적범위 도달 커버리지:    관련도우선 {ov['rel_lead_coverage(mean)']:.3f}"
          f"  →  폭우선 {ov['breadth_lead_coverage(mean)']:.3f}")
    print(f"폭-리드의 평균 관련도 순위(0=1위): {ov['breadth_lead_relrank(mean)']:.1f}  (낮을수록 관련성도 유지)")

    print("\n[참조 법률 수(num_laws)별 도달 커버리지]")
    by = df.groupby('num_laws')[['rel_lead_coverage', 'breadth_lead_coverage']].mean()
    by['num_queries'] = df.groupby('num_laws').size()
    print(by.reset_index().to_string(index=False))

    eval_dir = os.path.join(os.path.dirname(__file__), "eval_results")
    os.makedirs(eval_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(os.path.join(eval_dir, f"breadth_assembly_detailed_{ts}.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(eval_dir, f"breadth_assembly_summary_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump({'timestamp': ts, 'pool_n': args.pool_n,
                   'breadth_weights': args.breadth_weights, 'overall': ov,
                   'by_num_laws': by.reset_index().to_dict(orient='records')},
                  f, ensure_ascii=False, indent=2)
    print(f"\n✅ 저장: eval_results/breadth_assembly_summary_{ts}.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=None, help="단건 조립 모드: 질의문")
    parser.add_argument("--query_encoder", default="query_encoder_best.safetensors")
    parser.add_argument("--no_query_encoder", action="store_true", help="순수 BGE 검색")
    parser.add_argument("--pool_n", type=int, default=20, help="관련도 top-N 후보 풀 (이 안에서 폭 최대 선택)")
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
