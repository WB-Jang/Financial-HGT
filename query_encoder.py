"""
query_encoder.py

Stage 2: 질의 BGE-M3 임베딩(1024d)을 고정된 조항 BGE 임베딩 공간(1024d)으로
투영하는 경량 잔차 MLP.

구조:
  out = L2_normalize( x + MLP(x) )
  MLP: Linear(1024->512) -> LayerNorm -> GELU -> Dropout -> Linear(512->1024)

설계 포인트:
- 잔차 연결 + 마지막 레이어 0 초기화: 학습 시작 시점에 out == normalize(x),
  즉 '순수 BGE 코사인 베이스라인'과 완전히 동일한 상태에서 출발한다.
  학습은 베이스라인 위에 보정만 얹으므로 성능이 베이스라인 아래로
  무너진 채 시작하는 문제(기존 HGT 방식의 실패 원인)가 원천 차단된다.
- 파라미터 약 105만 개: 학습 질의 ~2.8천 건 규모에 적정한 용량.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryEncoder(nn.Module):
    def __init__(self, dim: int = 1024, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        # 마지막 레이어를 0으로 초기화 -> 초기 forward가 항등(=베이스라인)과 일치
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, bge_emb: torch.Tensor) -> torch.Tensor:
        """bge_emb: (B, 1024) -> (B, 1024) L2-normalized"""
        return F.normalize(bge_emb + self.mlp(bge_emb), dim=-1)
