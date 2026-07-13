"""
ranking_methods.py

evaluate_rerank.py(recall 지표 비교)와 assemble_context.py(구조적 폭 지표 비교)가
공유하는 검색/재랭킹 스코어링 로직. 같은 재랭킹 방식(dense/hybrid/ppr/cross)으로 만든
후보 풀을 두 스크립트가 서로 다른 목적(recall@K / breadth coverage)으로 평가할 수 있도록
한 곳에만 구현한다.

제공 함수:
  compute_base_scores  : dense 또는 dense+BM25(RRF) 하이브리드 (Q,N) 점수
  compute_ppr_scores   : base_scores 위에 PPR 그래프 재랭킹을 얹은 (Q,N) 점수
  compute_cross_rerank : base_scores 상위 topk를 cross-encoder로 재정렬
                        (전체 랭킹 + 재정렬된 topk의 원시 점수)
"""

import torch

from retrieval_common import BM25, korean_tokenize, build_clause_adjacency


def rrf_fuse(dense_scores, bm25_scores, k=60):
    """두 점수 벡터를 Reciprocal Rank Fusion으로 융합 (스케일 무관)."""
    dense = dense_scores if torch.is_tensor(dense_scores) else torch.tensor(dense_scores)
    bm25 = torch.tensor(bm25_scores, dtype=torch.float32)
    dr = torch.empty_like(dense); dr[dense.argsort(descending=True)] = torch.arange(len(dense), dtype=dense.dtype)
    br = torch.empty_like(bm25);  br[bm25.argsort(descending=True)]  = torch.arange(len(bm25), dtype=bm25.dtype)
    return 1.0 / (k + dr) + 1.0 / (k + br)


def build_ppr_operator(edge_w, n):
    """PPR 전파에 필요한 텐서(src, dst, 전이확률)를 준비한다."""
    pairs = torch.tensor(list(edge_w.keys()), dtype=torch.long)
    w = torch.tensor(list(edge_w.values()), dtype=torch.float32)
    src = torch.cat([pairs[:, 0], pairs[:, 1]])
    dst = torch.cat([pairs[:, 1], pairs[:, 0]])
    ww = torch.cat([w, w])
    w_out = torch.zeros(n).index_add_(0, src, ww)
    trans = ww / w_out[src].clamp(min=1e-12)
    isolated = w_out == 0
    return src, dst, trans, isolated


def personalized_pagerank(seed_vec, src, dst, trans, isolated, restart=0.5, iters=10):
    """단일 질의의 PPR 점수 벡터 계산. seed_vec: (N,) 합=1"""
    p = seed_vec.clone()
    for _ in range(iters):
        spread = torch.zeros_like(p).index_add_(0, dst, p[src] * trans)
        spread = spread + p * isolated.float()
        p = restart * seed_vec + (1 - restart) * spread
    return p


def compute_base_scores(q, clause_embs, items, clause_texts, hybrid):
    """dense 코사인, 또는 --hybrid 시 dense+BM25(RRF) 융합 (Q,N) 점수."""
    dense_sims = q @ clause_embs.T
    if not hybrid:
        return dense_sims
    print("BM25 어휘 인덱스 구축 중...", flush=True)
    bm25 = BM25([korean_tokenize(t) for t in clause_texts])
    base = torch.zeros_like(dense_sims)
    for i, it in enumerate(items):
        bm = bm25.scores(korean_tokenize(it["query"]))
        base[i] = rrf_fuse(dense_sims[i], bm)
    print("dense+BM25 하이브리드(RRF) 융합 완료")
    return base


def compute_ppr_scores(base_scores, edge_w, n_clauses, seeds=20, beta=0.3, restart=0.5, iters=10):
    """base_scores 상위 seeds개를 시드로 PPR 전파 후 base와 합산한 (Q,N) 점수.

    edge_w: build_clause_adjacency(...)의 결과 (조항 인접 그래프, 호출부에서 미리 구성해 전달
    -> assemble_context.py가 breadth의 kg_degree와 동일 그래프를 재사용하도록).
    """
    src, dst, trans, isolated = build_ppr_operator(edge_w, n_clauses)
    final = torch.zeros_like(base_scores)
    for i in range(base_scores.size(0)):
        k = min(seeds, base_scores.size(1))
        seed_idx = base_scores[i].topk(k).indices
        seed_scores = base_scores[i][seed_idx]
        seed_scores = (seed_scores - seed_scores.min()).clamp(min=1e-6)
        seed_vec = torch.zeros(base_scores.size(1))
        seed_vec[seed_idx] = seed_scores / seed_scores.sum()
        ppr = personalized_pagerank(seed_vec, src, dst, trans, isolated, restart=restart, iters=iters)
        ppr_n = ppr / ppr.max().clamp(min=1e-9)
        b = base_scores[i]
        b_n = (b - b.min()) / (b.max() - b.min()).clamp(min=1e-9)
        final[i] = b_n + beta * ppr_n
    print(f"PPR 재랭킹 완료 ({base_scores.size(0)}개 질의)")
    return final


def compute_cross_rerank(base_scores, items, clause_texts, cross_model="BAAI/bge-reranker-v2-m3",
                         topk=50, device=None):
    """base_scores 상위 topk를 (질의,조항) cross-encoder로 재정렬.

    반환: (full_ranked_lists, topk_scores)
      full_ranked_lists[i] : 재정렬된 topk + 그 뒤는 base 순서 그대로 (길이 N, evaluate_rerank용)
      topk_scores[i]       : full_ranked_lists[i][:topk]에 대응하는 cross-encoder 원시 점수(내림차순,
                             assemble_context가 그 안에서 breadth 블렌드할 때 사용)
    """
    from sentence_transformers import CrossEncoder
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"cross-encoder 로드: {cross_model} (device={device}, 첫 실행 시 다운로드)", flush=True)
    ce = CrossEncoder(cross_model, device=device, max_length=512)
    base_ranking = base_scores.argsort(dim=1, descending=True)
    topk = min(topk, base_scores.size(1))
    full_ranked_lists, topk_scores = [], []
    for i, it in enumerate(items):
        base_order = base_ranking[i].tolist()
        cand = base_order[:topk]
        pairs = [[it["query"], clause_texts[c]] for c in cand]
        ce_scores = ce.predict(pairs, batch_size=32, show_progress_bar=False)
        order = sorted(range(len(cand)), key=lambda k: -ce_scores[k])
        full_ranked_lists.append([cand[k] for k in order] + base_order[topk:])
        topk_scores.append([float(ce_scores[k]) for k in order])
        if (i + 1) % 50 == 0:
            print(f"  cross-encoder [{i+1}/{len(items)}]", flush=True)
    print(f"cross-encoder 재랭킹 완료 (질의당 상위 {topk}개)")
    return full_ranked_lists, topk_scores
