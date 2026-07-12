"""
breadth.py

조항별 '구조적 폭(structural breadth)' 점수를 계산한다.
사용자 목표: 질의와 관련된 조항 중에서 '가장 넓은 범위를 커버하는(=다른 조항/법률과
관계를 가장 많이 가진)' 골격 조항을 컨텍스트 맨 앞에 세우기 위한 신호.

폭 신호 4가지 (nodes.csv에 이미 주석되어 있음 + KG 구조):
  - n_cross_laws : cross_law_refs가 참조하는 '서로 다른 법률의 수'  (타법 연결 폭, 최우선)
  - n_cross_refs : 타법 참조 '건수'
  - n_intra_refs : intra_law_refs (자법 내 참조) 건수
  - kg_degree    : KG 인접(형제 항+공유 엔터티) 이웃 수 (관계 밀도)

각 성분을 조항 전체에 대해 min-max 정규화 후 가중합 → breadth ∈ [0, 1].
"""

import re
from collections import defaultdict

import pandas as pd

# 그래프 short 법령명 <- cross_law_refs의 공식 full명 (약칭 매핑; data에서 실측해 구성)
_FULL_TO_SHORT = {
    "자본시장과 금융투자업에 관한 법률": "자본시장법",
    "자본시장과 금융투자업에 관한 법률 시행령": "자본시장법 시행령",
    "금융회사의 지배구조에 관한 법률": "지배구조법",
    "금융회사의 지배구조에 관한 법률 시행령": "지배구조법 시행령",
    "신용정보의 이용 및 보호에 관한 법률": "신용정보법",
    "신용정보의 이용 및 보호에 관한 법률 시행령": "신용정보법 시행령",
    "금융소비자 보호에 관한 법률": "금융소비자보호법",
    "금융산업의 구조개선에 관한 법률": "금산법",
    "금융실명거래 및 비밀보장에 관한 법률": "금융실명법",
    "특정 금융거래정보의 보고 및 이용 등에 관한 법률": "특정금융정보법",
    "상호저축은행법": "저축은행법",
}


def _norm(s):
    return re.sub(r'\s+', '', str(s)).strip()


_CANON = {_norm(k): v for k, v in _FULL_TO_SHORT.items()}


def to_graph_law(name, graph_law_set):
    """공식/약칭 법령명을 그래프 short명으로 정규화 (그래프에 없으면 원본 반환)."""
    n = _norm(name)
    if n in _CANON:
        return _CANON[n]
    for gl in graph_law_set:
        if _norm(gl) == n:
            return gl
    return name  # 그래프 밖 법령 (상법·민법 등)


def parse_ref_laws(s):
    """cross_law_refs 문자열에서 「...」로 감싼 법령명 리스트 추출."""
    if not isinstance(s, str) or not s.strip():
        return []
    return [m.strip() for m in re.findall(r'「([^」]+)」', s)]


def _count_refs(s):
    if not isinstance(s, str) or not s.strip():
        return 0
    return len([x for x in s.split(';') if x.strip()])


def _minmax(series):
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-12:
        return series * 0.0
    return (series - lo) / (hi - lo)


def compute_breadth(nodes_df, clause_list, kg_degree=None, weights=(0.5, 0.2, 0.1, 0.2)):
    """clause_list 순서에 정렬된 breadth 점수와 구성요소 DataFrame을 반환.

    nodes_df: 'new_johang' 컬럼이 이미 계산되어 있어야 함.
    kg_degree: clause_list 인덱스별 KG 인접 이웃 수 (list/array, 선택).
    weights: (n_cross_laws, n_cross_refs, n_intra_refs, kg_degree) 가중치.

    Returns: (breadth_array, comp_df)
      breadth_array: len(clause_list) 길이의 float 리스트 ([0,1])
      comp_df: 조항별 폭 구성요소 + 참조 graph 법령 집합
    """
    graph_law_set = set(nodes_df['law_nm'].dropna().unique())

    # new_johang(=조항) 단위 집계: 같은 조항의 여러 행(항)에 걸친 참조를 합침
    agg = defaultdict(lambda: {'ref_laws': set(), 'ref_graph_laws': set(),
                               'cross_refs': 0, 'intra_refs': 0})
    for nj, cross, intra in zip(nodes_df['new_johang'], nodes_df.get('cross_law_refs'),
                                nodes_df.get('intra_law_refs')):
        if pd.isna(nj):
            continue
        a = agg[nj]
        for full in parse_ref_laws(cross):
            a['ref_laws'].add(full)
            gl = to_graph_law(full, graph_law_set)
            if gl in graph_law_set:
                a['ref_graph_laws'].add(gl)
        a['cross_refs'] += _count_refs(cross)
        a['intra_refs'] += _count_refs(intra)

    rows = []
    for i, c in enumerate(clause_list):
        a = agg.get(c, {'ref_laws': set(), 'ref_graph_laws': set(), 'cross_refs': 0, 'intra_refs': 0})
        rows.append({
            'clause': c,
            'n_cross_laws': len(a['ref_laws']),
            'n_cross_refs': a['cross_refs'],
            'n_intra_refs': a['intra_refs'],
            'kg_degree': int(kg_degree[i]) if kg_degree is not None else 0,
            'ref_graph_laws': a['ref_graph_laws'],
        })
    comp_df = pd.DataFrame(rows)

    w1, w2, w3, w4 = weights
    breadth = (w1 * _minmax(comp_df['n_cross_laws'])
               + w2 * _minmax(comp_df['n_cross_refs'])
               + w3 * _minmax(comp_df['n_intra_refs'])
               + w4 * _minmax(comp_df['kg_degree']))
    comp_df['breadth'] = breadth
    return breadth.tolist(), comp_df


def kg_degree_from_adjacency(edge_w, n):
    """build_clause_adjacency 결과(edge_w)에서 조항별 이웃 수(degree) 배열 계산."""
    deg = [0] * n
    for (i, j) in edge_w:
        deg[i] += 1
        deg[j] += 1
    return deg
