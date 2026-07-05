import torch
import torch.nn.functional as F
import torch.optim as optim
from data_loader import load_and_build_graph
from hgt_model import FinancialHGT
from sentence_transformers import SentenceTransformer
import random
# safetensors 라이브러리에서 PyTorch용 저장 함수 임포트
from safetensors.torch import save_file 

# ==========================================
# 1. InfoNCE Loss (Contrastive Learning) 정의
# ==========================================
def info_nce_loss(query_emb, pos_vnode_emb, neg_vnode_embs, temperature=0.07):
    """
    질의(query)와 정답 가상노드(pos) 간의 유사도는 높이고, 오답(neg)들과의 유사도는 낮춥니다.
    """
    # 벡터 정규화
    query_emb = F.normalize(query_emb, p=2, dim=-1)
    pos_vnode_emb = F.normalize(pos_vnode_emb, p=2, dim=-1)
    neg_vnode_embs = F.normalize(neg_vnode_embs, p=2, dim=-1) # (Num_negs, 1024)
    
    # 유사도 계산 (Dot product after normalization = Cosine Similarity)
    pos_sim = torch.sum(query_emb * pos_vnode_emb, dim=-1) / temperature # (1,)
    neg_sims = torch.matmul(query_emb, neg_vnode_embs.T).squeeze() / temperature # (Num_negs,)
    
    # InfoNCE Loss 수식 적용: -log( exp(pos) / (exp(pos) + sum(exp(neg))) )
    logits = torch.cat([pos_sim.unsqueeze(0), neg_sims.unsqueeze(0) if neg_sims.dim()==0 else neg_sims])
    labels = torch.zeros(1, dtype=torch.long, device=logits.device) # 정답(positive)은 0번째 인덱스
    
    loss = F.cross_entropy(logits.unsqueeze(0), labels)
    return loss


# ==========================================
# 2. 메인 학습 루프
# ==========================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 기기: {device}")

    # 1. 데이터 로드 (빠른 테스트를 위해 dummy 사용, 실제로는 False)
    graph_data, clause_to_idx, entity_to_idx,fsc_qa_dataset = load_and_build_graph('./data/nodes_20260626_171919.csv', './data/triplets_20260626_171919.csv', use_dummy_emb=False)
    graph_data = graph_data.to(device)
    
    # 2. 텍스트 쿼리 인코더 (BGE-M3) - 실제 환경에서는 메모리에 올립니다.
    # text_encoder = SentenceTransformer('BAAI/bge-m3').to(device)
    
    # 3. HGT 모델 초기화
    model = FinancialHGT(metadata=graph_data.metadata(), in_channels=1024, hidden_channels=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 4. FSC 학습 데이터 모사 (가상의 질의와 정답 조항 리스트)
    # fsc_qa_dataset = [
    #     {
    #         "query": "사외이사의 자격요건과 이를 위반했을 때의 과태료 처분 절차는 어떻게 되나요?",
    #         "positive_clauses": ["제1조", "제18조"],
    #         "hard_negative_clauses": [["제20조"], ["제41조", "제2조"]] # 오답 가상 노드들용 리스트
    #     },
    #     # ... 추가 데이터 ...
    # ]

    model.train()
    epochs = 10
    
    for epoch in range(epochs):
        total_loss = 0
        
        # 매 Epoch마다 전체 그래프의 노드 임베딩을 한 번 업데이트합니다. (Graph is static)
        updated_node_embs = model(graph_data.x_dict, graph_data.edge_index_dict)
        clause_embs = updated_node_embs['clause'] # (Num_clauses x 256)
        
        for item in fsc_qa_dataset:
            optimizer.zero_grad()
            
            # (1) Query 벡터화 (실제로는 text_encoder 활용)
            # q_text = item["query"]
            # q_emb = torch.tensor(text_encoder.encode([q_text])).to(device) 
            q_emb = torch.randn((1, 1024)).to(device) # 더미 대체
            
            # (2) Positive Virtual Node 생성
            pos_indices = [clause_to_idx[c] for c in item["positive_clauses"] if c in clause_to_idx]
            if not pos_indices: continue
            pos_vnode = model.aggregate_virtual_node(clause_embs, pos_indices)
            
            # (3) Negative Virtual Nodes 생성
            neg_vnodes_list = []
            for neg_group in item["hard_negative_clauses"]:
                neg_indices = [clause_to_idx[c] for c in neg_group if c in clause_to_idx]
                if neg_indices:
                    neg_vnodes_list.append(model.aggregate_virtual_node(clause_embs, neg_indices))
            
            if not neg_vnodes_list:
                # 하드 네거티브가 없으면 랜덤 샘플링 대체 로직 필요
                neg_vnodes_list.append(torch.randn((1, 1024)).to(device))
                
            neg_vnodes = torch.cat(neg_vnodes_list, dim=0) # (Num_negs x 1024)
            
            # (4) Loss 계산 및 역전파
            loss = info_nce_loss(q_emb, pos_vnode, neg_vnodes)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss/len(fsc_qa_dataset):.4f}")

    # ==========================================
    # 5. Safetensors 포맷으로 모델 저장
    # ==========================================
    save_path = "./financial_hgt_model.safetensors"
    
    # 모델의 가중치(state_dict)를 추출
    state_dict = model.state_dict()
    
    # Safetensors는 메모리 연속성(contiguous)을 요구하므로 변환 처리 (에러 방지용)
    contiguous_state_dict = {k: v.contiguous() for k, v in state_dict.items()}
    
    # 모델 저장 실행
    save_file(contiguous_state_dict, save_path)
    print(f"\n✅ 학습 완료! 모델이 안전한 safetensors 포맷으로 '{save_path}'에 저장되었습니다.")

if __name__ == "__main__":
    train()