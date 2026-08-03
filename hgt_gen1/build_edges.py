"""
build_edges.py — Stage 0: 링크예측 학습용 엣지와 회귀 타깃 생성.

KG-search(Gen1) src/data/graph_builder.py의 엣지 의미를 Financial-HGT(Gen2) 데이터에
옮긴다. BGE가 필요 없다 — nodes.csv와 triplets.csv만 읽는다.

엣지 3종:
  INTRA    같은 법 안의 조항 쌍 (형제 항 + 공유 엔터티)
  SIBLING  법 ↔ 그 시행령 사이의 조항 쌍
  CROSS    그 밖의 서로 다른 법 사이의 조항 쌍

회귀 타깃 (Gen1 train_multilaw.py와 동일):
  INTRA          target = shared_count / max_shared_count
  SIBLING/CROSS  target = confidence = co_occurrence / max_co_occurrence

⚠️ insight-agent의 export_fhgt_graph.py는 SIBLING/CROSS의 confidence를 1.0으로
고정한다. 그 값을 회귀 타깃으로 쓰면 "모든 cross 쌍의 코사인을 1로" 만드는 목표가 되어
표현이 붕괴한다. 여기서는 Gen1처럼 최댓값 정규화를 복원한다.

엣지 출처는 조항 텍스트와 트리플뿐이고 FSC 질의를 쓰지 않는다. Gen1의 CROSS는 FSC
co-occurrence였기 때문에 test 질의를 빼는 --exclude_queries가 필요했지만, 여기서는
질의 데이터가 개입하지 않으므로 누수가 구조적으로 불가능하다.

사용법:
    python hgt_gen1/build_edges.py                    # -> hgt_gen1/graph/
    python hgt_gen1/build_edges.py --max_entity_df 20 --cite_weight 1.0
"""

import argparse
import os
import re
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval_common import build_clause_adjacency, build_clause_index  # noqa: E402

SCOPES = {'INTRA': 0, 'SIBLING': 1, 'CROSS': 2}


def law_of(clause_key):
    """'자본시장법 제94조 제1항' -> '자본시장법'."""
    m = re.search(r'\s*제\s*\d', clause_key)
    return (clause_key[:m.start()] if m else clause_key).strip()


def article_of(clause_key):
    """'자본시장법 제94조 제1항' -> '제94조'."""
    m = re.search(r'제\s*\d+(?:-\d+)?\s*조(?:\s*의\s*\d+)?', clause_key)
    return re.sub(r'\s+', '', m.group(0)) if m else ''


def is_sibling(law_a, law_b):
    """법 ↔ 그 시행령/시행규칙 관계인가."""
    a, b = re.sub(r'\s+', '', law_a), re.sub(r'\s+', '', law_b)
    if a == b:
        return False
    for suffix in ('시행령', '시행규칙'):
        if a == b + suffix or b == a + suffix:
            return True
    return False


_FULL_TO_SHORT_CACHE = {}


def to_short_law(full_name, graph_laws):
    """cross_law_refs의 공식 법령명을 그래프 short명으로 정규화 (breadth.py 규칙 재사용)."""
    if full_name in _FULL_TO_SHORT_CACHE:
        return _FULL_TO_SHORT_CACHE[full_name]
    from breadth import to_graph_law
    out = to_graph_law(full_name, graph_laws)
    _FULL_TO_SHORT_CACHE[full_name] = out
    return out


_CITE_RE = re.compile(r'「([^」]+)」\s*((?:제\s*\d+(?:-\d+)?\s*조(?:\s*의\s*\d+)?)?)')


def parse_citations(raw):
    """cross_law_refs 문자열 -> [(법령명, '제X조' | '')]. 조가 없으면 빈 문자열."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    return [(m.group(1).strip(), re.sub(r'\s+', '', m.group(2)))
            for m in _CITE_RE.finditer(raw)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nodes', default='data/nodes.csv')
    ap.add_argument('--triplets', default='data/triplets.csv')
    ap.add_argument('--out_dir', default='hgt_gen1/graph')
    ap.add_argument('--max_entity_df', type=int, default=20,
                    help='이 개수를 넘는 조항과 연결된 허브 엔터티는 제외')
    ap.add_argument('--cite_weight', type=float, default=1.0,
                    help='명시 인용 1건이 co-occurrence 몇 단위에 해당하는가')
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nodes_df = pd.read_csv(os.path.join(root, args.nodes))
    triplets_df = pd.read_csv(os.path.join(root, args.triplets))

    from data_loader import normalize_johang_key
    for df, has_hang in ((nodes_df, True), (triplets_df, False)):
        df['new_johang'] = [
            normalize_johang_key(l, a, h)
            for l, a, h in zip(df['law_nm'], df['article_number'],
                               df['hang_number'] if has_hang else [None] * len(df))
        ]

    clause_list, _ = build_clause_index(nodes_df)
    idx_of = {c: i for i, c in enumerate(clause_list)}
    laws = [law_of(c) for c in clause_list]
    arts = [article_of(c) for c in clause_list]
    graph_laws = set(laws)
    print(f'조항 노드 {len(clause_list):,}개 | 법령 {len(graph_laws)}종')

    # ── 1) 형제 항 + 공유 엔터티 인접 (레포 공용 함수 재사용) ────────────────
    edge_w = build_clause_adjacency(clause_list, triplets_df,
                                    max_entity_df=args.max_entity_df)

    # ── 2) cross_law_refs 명시 인용 추가 (조 번호가 있는 것만) ───────────────
    art_index = defaultdict(list)
    for i, (lw, at) in enumerate(zip(laws, arts)):
        art_index[(lw, at)].append(i)

    cite_series = (nodes_df.drop_duplicates('new_johang')
                   .set_index('new_johang')['cross_law_refs'])
    n_cite, n_cite_skip = 0, 0
    for c, i in idx_of.items():
        for law_full, art in parse_citations(cite_series.get(c)):
            if not art:
                n_cite_skip += 1
                continue          # 조문 미명시 → 정밀도 위해 링크 생략
            tgt = to_short_law(law_full, graph_laws)
            if tgt not in graph_laws:
                continue          # 그래프 밖 법령
            for j in art_index.get((tgt, art), []):
                if i == j:
                    continue
                edge_w[(min(i, j), max(i, j))] += args.cite_weight
                n_cite += 1
    print(f'명시 인용 엣지: {n_cite:,}건 추가 (조문 미명시로 생략 {n_cite_skip:,}건)')

    # ── 3) 법 관계로 스코프 분류 ────────────────────────────────────────────
    rows = []
    for (i, j), w in edge_w.items():
        la, lb = laws[i], laws[j]
        scope = 'INTRA' if la == lb else ('SIBLING' if is_sibling(la, lb) else 'CROSS')
        rows.append((i, j, scope, w))

    # ── 4) 회귀 타깃: 스코프군별 최댓값 정규화 (Gen1 규칙) ───────────────────
    max_intra = max((w for _, _, s, w in rows if s == 'INTRA'), default=1.0)
    max_cross = max((w for _, _, s, w in rows if s != 'INTRA'), default=1.0)
    out = []
    for i, j, scope, w in rows:
        target = w / (max_intra if scope == 'INTRA' else max_cross)
        out.append({
            'src_idx': i, 'dst_idx': j,
            'src_node': clause_list[i], 'dst_node': clause_list[j],
            'edge_scope': scope, 'scope_id': SCOPES[scope],
            'raw_weight': round(w, 4), 'target': round(min(target, 1.0), 6),
        })
    edges = pd.DataFrame(out).sort_values(['edge_scope', 'target'], ascending=[True, False])

    out_dir = os.path.join(root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    edges.to_csv(os.path.join(out_dir, 'edges.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame({'node_idx': range(len(clause_list)), 'clause': clause_list,
                  'law': laws, 'article': arts}).to_csv(
        os.path.join(out_dir, 'nodes_index.csv'), index=False, encoding='utf-8-sig')

    print()
    for scope in ('INTRA', 'SIBLING', 'CROSS'):
        sub = edges[edges.edge_scope == scope]
        if len(sub) == 0:
            print(f'  {scope:8s} 0건')
            continue
        print(f'  {scope:8s} {len(sub):>7,}건 | target 평균 {sub.target.mean():.4f} '
              f'중앙값 {sub.target.median():.4f} 최대 {sub.target.max():.4f}')
    covered = set(edges.src_idx) | set(edges.dst_idx)
    print(f'\n엣지에 등장하는 노드: {len(covered):,}/{len(clause_list):,} '
          f'({len(covered) / len(clause_list) * 100:.1f}%)')
    print(f'-> {out_dir}/edges.csv, nodes_index.csv')


if __name__ == '__main__':
    main()
