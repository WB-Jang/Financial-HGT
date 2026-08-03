"""
model.py — Gen1(KG-search) HGT를 Financial-HGT 데이터에 맞춰 이식.

원본: KG-search_PPR_GNN_Transformer/src/models/{node_encoder,graph_transformer}.py
(Hu et al., "Heterogeneous Graph Transformer", WWW 2020 — PyG 없이 순수 PyTorch)

Gen2 데이터에 맞춘 변경 2가지:
  1. 노드 타입 = law_id(30종)가 아니라 law_type(법률/시행령/시행규칙/규정/기타 5종).
     Gen1은 법령 9종·노드 6,103개라 법령당 678노드였지만 Gen2는 30종·9,311개로
     법령당 310개다. 작은 법령은 타입별 Q/K/V가 학습되지 않는다. law_type은
     법↔시행령의 서술 방식 차이라는 실제 이질성을 유지하면서 타입당 데이터를 확보한다.
     법령 정체성은 NodeEncoder의 law_id 임베딩으로 그대로 들어간다.
  2. entity_type / cross_law_ref_count / law_position_ratio를 Gen2 nodes.csv 컬럼에서 유도.

HGT 레이어의 잔차(x = norm(x + out))가 원본 그대로 남아 있다. 텍스트 신호가 층을
통과해 보존되는 경로이며, 구 train.py의 PyG 경로에는 이것이 없었다.
"""

import math

import torch
import torch.nn as nn

EDGE_SCOPES = {"INTRA": 0, "SIBLING": 1, "CROSS": 2}


class NodeEncoder(nn.Module):
    """텍스트 임베딩 + 범주형/수치형 피처 -> hidden_dim."""

    def __init__(self, hidden_dim, text_dim=1024, n_law_ids=32, n_law_types=6,
                 n_entity_types=24, law_emb_dim=16, type_emb_dim=8, dropout=0.1):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.law_id_emb = nn.Embedding(n_law_ids, law_emb_dim)
        self.law_type_emb = nn.Embedding(n_law_types, type_emb_dim)
        self.entity_type_emb = nn.Embedding(n_entity_types, type_emb_dim)
        cat_dim = hidden_dim + law_emb_dim + 2 * type_emb_dim + 2
        self.proj = nn.Sequential(
            nn.Linear(cat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, text_emb, law_id, law_type, entity_type, scalars):
        h = torch.cat([
            self.text_proj(text_emb),
            self.law_id_emb(law_id),
            self.law_type_emb(law_type),
            self.entity_type_emb(entity_type),
            scalars,
        ], dim=-1)
        return self.proj(h)


def _scatter_softmax(scores, dst_idx, n_nodes):
    """목적지 노드별 softmax. scores (E,H) -> (E,H)."""
    max_vals = torch.full((n_nodes, scores.size(1)), float("-inf"), device=scores.device)
    dst_expanded = dst_idx.unsqueeze(1).expand_as(scores)
    max_vals.scatter_reduce_(0, dst_expanded, scores, reduce="amax", include_self=True)
    max_vals = max_vals.clamp(min=-1e30)          # 이웃 없는 노드의 -inf 방어
    exp_s = (scores - max_vals[dst_idx]).exp()
    sum_exp = torch.zeros(n_nodes, scores.size(1), device=scores.device)
    sum_exp.scatter_add_(0, dst_expanded, exp_s)
    return exp_s / (sum_exp[dst_idx] + 1e-8)


class HGTLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_node_types, n_edge_scopes=3, dropout=0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim, self.n_heads = hidden_dim, n_heads
        self.head_dim = hidden_dim // n_heads
        mk = lambda bias: nn.ModuleList(  # noqa: E731
            [nn.Linear(hidden_dim, hidden_dim, bias=bias) for _ in range(n_node_types)])
        self.W_Q, self.W_K, self.W_V = mk(False), mk(False), mk(False)
        self.W_O = mk(True)
        self.edge_bias = nn.Embedding(n_edge_scopes, n_heads)
        self.norm1, self.norm2 = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_scope, node_type):
        N, E = x.size(0), edge_index.size(1)
        src_idx, dst_idx = edge_index[0], edge_index[1]
        Q = x.new_zeros(N, self.n_heads, self.head_dim)
        K = x.new_zeros(N, self.n_heads, self.head_dim)
        V = x.new_zeros(N, self.n_heads, self.head_dim)
        for t in range(len(self.W_Q)):
            mask = node_type == t
            if mask.any():
                shape = (-1, self.n_heads, self.head_dim)
                Q[mask] = self.W_Q[t](x[mask]).view(shape)
                K[mask] = self.W_K[t](x[mask]).view(shape)
                V[mask] = self.W_V[t](x[mask]).view(shape)

        attn = (Q[dst_idx] * K[src_idx]).sum(-1) / math.sqrt(self.head_dim)
        attn = attn + self.edge_bias(edge_scope)
        attn = self.dropout(_scatter_softmax(attn, dst_idx, N))

        msg = (attn.unsqueeze(-1) * V[src_idx]).view(E, self.hidden_dim)
        agg = x.new_zeros(N, self.hidden_dim)
        agg.scatter_add_(0, dst_idx.unsqueeze(1).expand_as(msg), msg)

        out = x.new_zeros(N, self.hidden_dim)
        for t in range(len(self.W_O)):
            mask = node_type == t
            if mask.any():
                out[mask] = self.W_O[t](agg[mask])

        x = self.norm1(x + self.dropout(out))          # 잔차 — 텍스트 신호 보존 경로
        return self.norm2(x + self.dropout(self.ffn(x)))


class HGT(nn.Module):
    """NodeEncoder + 다층 HGT. forward는 L2 정규화된 노드 임베딩을 돌려준다."""

    def __init__(self, hidden_dim=256, n_layers=2, n_heads=4, n_node_types=6,
                 text_dim=1024, n_law_ids=32, n_entity_types=24, dropout=0.1):
        super().__init__()
        self.encoder = NodeEncoder(hidden_dim, text_dim=text_dim, n_law_ids=n_law_ids,
                                   n_law_types=n_node_types, n_entity_types=n_entity_types,
                                   dropout=dropout)
        self.layers = nn.ModuleList([
            HGTLayer(hidden_dim, n_heads, n_node_types, len(EDGE_SCOPES), dropout)
            for _ in range(n_layers)
        ])

    def forward(self, feats, edge_index, edge_scope):
        x = self.encoder(feats["text"], feats["law_id"], feats["law_type"],
                         feats["entity_type"], feats["scalars"])
        node_type = feats["law_type"]
        for layer in self.layers:
            x = layer(x, edge_index, edge_scope, node_type)
        return torch.nn.functional.normalize(x, dim=-1)


class QueryEncoder256(nn.Module):
    """Gen1 Stage 2 질의 인코더: BGE 1024d -> HGT 공간(hidden_dim).

    Gen2의 잔차 인코더(x + MLP(x))는 여기서 쓸 수 없다 — 입력 1024d와 출력 256d의
    차원이 달라 잔차 연결 자체가 성립하지 않는다. 따라서 '학습 시작 = 순수 BGE
    베이스라인' 보장도 없다. Gen1과 같은 조건이며, 결과 해석 시 명시해야 한다.
    """

    def __init__(self, hidden_dim=256, text_dim=1024, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(512, hidden_dim),
        )

    def forward(self, bge_emb):
        return torch.nn.functional.normalize(self.mlp(bge_emb), dim=-1)
