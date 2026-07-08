"""
build_smoothed_clause_emb.py

Stage 1 그래프 평활화: 조항 BGE 임베딩을 KG 이웃과 가중 평균하여
'구조 정보가 주입된' 고정 조항 임베딩을 생성한다. (학습 없음 - label propagation)

KG-search_PPR_GNN_Transformer의 build_smoothed_node_emb.py 방식을 Financial-HGT
그래프에 맞게 이식한 것. 조항(clause) 사이에 직접 엣지가 없으므로 인접을 두 가지로 정의:

  1) 형제 항: 같은 조(article)에 속한 항들끼리 연결
     (예: 지배구조법 제25조 제1항 <-> 제25조 제6항)
  2) 공유 엔터티: 같은 엔터티를 언급하는 조항끼리 연결 (triplets.csv 활용)
     - 조항A -has_subject-> [준법감시인] <-has_object- 조항B  =>  A<->B
     - 공유 엔터티 수만큼 엣지 가중치 누적
     - 허브 엔터티('금융위원회'는 1,451개 조항과 연결)는 모든 조항을 뭉개므로
       max_entity_df(기본 20)개 초과 조항과 연결된 엔터티는 제외

수식 (hop마다):
  new_emb[i] = L2_normalize( (1-alpha) * emb[i] + alpha * (이웃 가중 평균) )
  이웃이 없는 조항은 원본 유지.

실행:
  python build_smoothed_clause_emb.py                     # 기본값 alpha=0.3, 1 hop
  python build_smoothed_clause_emb.py --alpha 0.2 --max_entity_df 10

출력: ./data/clause_emb_smooth.safetensors

사용:
  python evaluate_baseline.py    --clause_emb data/clause_emb_smooth.safetensors
  python train_query_encoder.py --clause_emb data/clause_emb_smooth.safetensors
"""

import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from safetensors.torch import save_file

from data_loader import normalize_johang_key, encode_texts_cached
from retrieval_common import build_clause_index, build_clause_adjacency

NODES_CSV = './data/nodes.csv'
TRIPLETS_CSV = './data/triplets.csv'


def smooth(emb, edge_w, alpha, hops):
    """가중 이웃 평균과 (1-alpha):alpha 혼합을 hops회 반복."""
    n = emb.size(0)
    if not edge_w:
        print("경고: 엣지가 없어 원본 임베딩을 그대로 반환합니다.")
        return emb

    pairs = torch.tensor(list(edge_w.keys()), dtype=torch.long)      # (E, 2)
    w = torch.tensor(list(edge_w.values()), dtype=torch.float32)     # (E,)
    # 무방향 -> 양방향 확장
    src = torch.cat([pairs[:, 0], pairs[:, 1]])
    dst = torch.cat([pairs[:, 1], pairs[:, 0]])
    ww = torch.cat([w, w])

    w_sum = torch.zeros(n).index_add_(0, dst, ww)                    # 노드별 가중치 합
    has_neighbor = w_sum > 0
    print(f"이웃 보유 조항: {int(has_neighbor.sum()):,}/{n:,}개, "
          f"평균 가중 차수 {w_sum[has_neighbor].mean():.1f}")

    cur = F.normalize(emb.float(), dim=-1)
    for _ in range(hops):
        neigh_sum = torch.zeros_like(cur).index_add_(0, dst, cur[src] * ww.unsqueeze(1))
        neigh_mean = torch.zeros_like(cur)
        neigh_mean[has_neighbor] = neigh_sum[has_neighbor] / w_sum[has_neighbor].unsqueeze(1)
        nxt = cur.clone()
        nxt[has_neighbor] = (1 - alpha) * cur[has_neighbor] + alpha * neigh_mean[has_neighbor]
        cur = F.normalize(nxt, dim=-1)
    return cur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.3, help="이웃 혼합 비율 (0=원본 유지)")
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--max_entity_df", type=int, default=20,
                        help="이보다 많은 조항과 연결된 엔터티는 허브로 간주하고 제외")
    parser.add_argument("--sibling_weight", type=float, default=1.0)
    parser.add_argument("--entity_weight", type=float, default=1.0)
    parser.add_argument("--out", default="./data/clause_emb_smooth.safetensors")
    args = parser.parse_args()

    # 1. 조항 인덱스 (data_loader와 동일 순서 -> 캐시 적중)
    nodes_df = pd.read_csv(NODES_CSV)
    nodes_df['new_johang'] = [
        normalize_johang_key(law_nm, article_number, hang_number)
        for law_nm, article_number, hang_number in zip(
            nodes_df['law_nm'], nodes_df['article_number'], nodes_df['hang_number']
        )
    ]
    clause_list, clause_texts = build_clause_index(nodes_df)
    print(f"조항 노드: {len(clause_list):,}개")

    triplets_df = pd.read_csv(TRIPLETS_CSV)
    triplets_df['new_johang'] = [
        normalize_johang_key(law_nm, article_number)
        for law_nm, article_number in zip(triplets_df['law_nm'], triplets_df['article_number'])
    ]

    # 2. 원본 BGE 임베딩 (캐시 재사용)
    encoder = SentenceTransformer('BAAI/bge-m3', device='cpu')
    clause_embs = encode_texts_cached(encoder, clause_texts, 'clause_embs')
    del encoder

    # 3. 인접 구성 + 평활화 (retrieval_common의 공용 인접 빌더 사용)
    edge_w = build_clause_adjacency(clause_list, triplets_df,
                                    args.max_entity_df, args.sibling_weight, args.entity_weight)
    smoothed = smooth(clause_embs, edge_w, args.alpha, args.hops)

    # 원본과의 평균 코사인 유사도 (얼마나 움직였는지 확인용)
    orig = F.normalize(clause_embs.float(), dim=-1)
    mean_cos = (orig * smoothed).sum(dim=-1).mean().item()
    print(f"평활화 후 원본과의 평균 코사인: {mean_cos:.4f} (1.0이면 변화 없음)")

    save_file({'embeddings': smoothed.contiguous()}, args.out)
    print(f"\n✅ 저장 완료: {args.out}  shape={tuple(smoothed.shape)}")
    print(f"   평가:  python evaluate_baseline.py --clause_emb {args.out}")
    print(f"   학습:  python train_query_encoder.py --clause_emb {args.out}")


if __name__ == "__main__":
    main()
