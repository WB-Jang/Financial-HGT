import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear

class FinancialHGT(nn.Module):
    def __init__(self, metadata, in_channels=1024, hidden_channels=256, out_channels=1024, num_heads=4, num_layers=2):
        super(FinancialHGT, self).__init__()
        
        # 1. 노드 타입별 차원 축소 (BGE-M3 1024d -> HGT 256d)
        # 보고서 2장: '초기 노드 피처 주입 전략' 구현
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(in_channels, hidden_channels)
            # 노드 종류별로 다른 가중치 종류를 사용해서 256d로 압축함
            
        # 2. HGT 레이어 구성
        # 보고서 3장: '정보 집계 (Message Passing)' 구현
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # group='sum' 인자는 구버전 PyG 전용으로, 최신 torch-geometric에서는 제거됨 (기본 집계가 sum)
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)
            
        # 3. 가상 노드(Virtual Node) 집계를 위한 Attention Pooling
        # 질의와 관련된 여러 조항(Clause) 벡터를 가중합하여 하나의 벡터로 만듭니다.
        self.vnode_attention = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.Tanh(),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        # 4. Projection 레이어 (256d -> 1024d)
        # 보고서 4장: 대조 학습을 위해 Query 벡터와 차원을 맞춥니다.
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels, 512),
            nn.ReLU(),
            nn.Linear(512, out_channels)
        )

    def forward(self, x_dict, edge_index_dict):
        """ 전체 KG에 대해 Message Passing을 수행하여 노드 임베딩을 업데이트합니다. """
        # 차원 축소(1024 -> 256)
        out_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }
        
        # HGT Message Passing
        for conv in self.convs:
            out_dict = conv(out_dict, edge_index_dict)
            '''
            edge_index_dict는 이렇게 생김
            {
                # (출발타입, 관계명, 도착타입) : [출발지 인덱스 배열, 도착지 인덱스 배열]
                ('clause', 'has_subject', 'entity'): tensor([[0, 0, 1...], [10, 15, 8...]]),
                ('clause', 'has_object', 'entity'): tensor([[0, 1, 2...], [3, 9, 12...]]),
                ('entity', 'logical', 'entity'): tensor([[...], [...]])
            }
            '''
            #edge_index_dict에 따라서 0번 clause("제1조")의 full_text_emb를 10번 엔터티와 15번 엔터티에 보내는데, clause가 K,V이고 entity가 Q라서,
            #clause와 entity의 attention을 계산하여서 V(attention_score * clause_full_text)에 곱하고 이 정보를 entity에 sum한다. 이 때 attention 가중치는 metadata의 관계 종류 별로 다른 것이 사용됨 
        return out_dict

    def aggregate_virtual_node(self, clause_embeddings, target_indices):
        """
        특정 질의에 필요한 조항(Clause)들의 인덱스를 받아,
        가상 노드(Virtual Node)의 1024d 임베딩을 생성합니다.
        
        Args:
            clause_embeddings: HGT를 통과한 전체 Clause 노드 벡터 (Num_clauses x 256)
            target_indices: 가상 노드로 묶을 타겟 조항들의 인덱스 리스트 (List[int])
        Returns:
            가상 노드의 1024d 벡터 (1 x 1024)
        """
        # 타겟 조항들의 벡터 추출 (N x 256)
        target_embs = clause_embeddings[target_indices]
        
        # Attention Score 계산 및 정규화
        attn_scores = self.vnode_attention(target_embs) # (N x 1)
        attn_weights = torch.softmax(attn_scores, dim=0)
        
        # 가중합(Weighted Sum)을 통한 256d 가상 노드 집계
        vnode_256d = torch.sum(attn_weights * target_embs, dim=0, keepdim=True) # (1 x 256)
        
        # Projection (256d -> 1024d)
        vnode_1024d = self.projection(vnode_256d) # (1 x 1024)
        
        return vnode_1024d
