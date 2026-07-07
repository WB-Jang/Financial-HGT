import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import json
import pandas as pd
from datetime import datetime
from data_loader import load_and_build_graph
from hgt_model import FinancialHGT
from sentence_transformers import SentenceTransformer
import random
from safetensors.torch import save_file

# Recall@K 평가 시 사용할 K 값 목록
K_VALUES = [1, 3, 5, 10, 20]

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
# 2. Recall@K 평가 (Test 셋 조항 검색 성능)
# ==========================================
def evaluate_recall_at_k(model, graph_data, clause_to_idx, fsc_qa_dataset_test, query_embs_test, k_values=K_VALUES):
    """
    학습된 모델로 test 셋에 대해 조항 검색(retrieval) 성능을 Recall@K로 평가합니다.
    전체 조항을 개별적으로 1024d 투영 공간에 임베딩해 검색 후보 풀로 삼고,
    질의 임베딩과의 코사인 유사도 top-k 안에 실제 정답 조항이 얼마나 포함되는지 측정합니다.
    (개별 조항 하나짜리 가상노드는 attention softmax(1개)=1이라 projection(clause_256d)와 동일합니다.)

    Recall@K = (top-k 안에 포함된 정답 조항 수) / (해당 질의의 전체 정답 조항 수)
    """
    model.eval()
    with torch.no_grad():
        updated_node_embs = model(graph_data.x_dict, graph_data.edge_index_dict)
        clause_embs = updated_node_embs['clause']  # (Num_clauses x 256)

        all_clause_proj = model.projection(clause_embs)  # (Num_clauses x 1024)
        all_clause_proj = F.normalize(all_clause_proj, p=2, dim=-1)
        query_embs_norm = F.normalize(query_embs_test, p=2, dim=-1)

        max_k = max(k_values)
        results = []

        for i, item in enumerate(fsc_qa_dataset_test):
            pos_indices = set(clause_to_idx[c] for c in item["positive_clauses"] if c in clause_to_idx)
            if not pos_indices:
                continue

            q_emb = query_embs_norm[i].unsqueeze(0)  # (1 x 1024)
            sims = torch.matmul(q_emb, all_clause_proj.T).squeeze(0)  # (Num_clauses,)
            topk_indices = torch.topk(sims, k=min(max_k, sims.shape[0])).indices.tolist()

            row = {
                "query": item["query"],
                "num_laws": item["num_laws"],
                "num_positive_clauses": len(pos_indices),
            }
            for k in k_values:
                hit = len(set(topk_indices[:k]) & pos_indices)
                row[f"recall@{k}"] = hit / len(pos_indices)
            results.append(row)

    model.train()
    return results


# ==========================================
# 3. 메인 학습 루프
# ==========================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 기기: {device}")

    # 1. 데이터 로드 (빠른 테스트를 위해 dummy 사용, 실제로는 False)
    graph_data, clause_to_idx, entity_to_idx, fsc_qa_dataset, fsc_qa_dataset_test = load_and_build_graph('./data/nodes_20260704_231120.csv', './data/triplets_20260704_231120.csv', use_dummy_emb=False)
    graph_data = graph_data.to(device)
    
    # 2. 텍스트 쿼리 인코더 (BGE-M3)
    # 주의: 처음 실행 시 BGE-M3 모델(약 2.3GB)이 다운로드됨
    # Windows: C:\Users\<username>\.cache\huggingface\hub\models--BAAI--bge-m3\
    # Linux/Mac: ~/.cache/huggingface/hub/models--BAAI--bge-m3/
    # CPU에서 실행하여 VRAM 절약 (8GB GPU의 OOM 방지)
    print("BGE-M3 모델 로드 중... (첫 실행 시 ~2.3GB 다운로드될 수 있습니다)")
    text_encoder = SentenceTransformer('BAAI/bge-m3', device='cpu')

    # 3. HGT 모델 초기화
    model = FinancialHGT(metadata=graph_data.metadata(), in_channels=1024, hidden_channels=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 4. 질의 임베딩 사전 계산 (배치 처리로 메모리 절약)
    print(f"질의 임베딩 사전 계산 중... (CPU 인코딩이라 시간이 걸릴 수 있습니다)", flush=True)
    query_texts = [item["query"] for item in fsc_qa_dataset]
    batch_size_encode = 64  # 더 작은 배치 크기 (메모리 절약)
    query_embs_list = []

    for i in range(0, len(query_texts), batch_size_encode):
        batch_texts = query_texts[i:i+batch_size_encode]
        batch_embs = torch.tensor(text_encoder.encode(batch_texts, show_progress_bar=False))
        query_embs_list.append(batch_embs)
        done = min(i + batch_size_encode, len(query_texts))
        print(f"  [{done:,}/{len(query_texts):,}] 처리 완료", flush=True)

    query_embs = torch.cat(query_embs_list, dim=0).to(device)
    print(f"✓ {len(query_embs):,}개 질의 임베딩 계산 완료 (배치 크기: {batch_size_encode})")

    # 4-1. Test 질의 임베딩 사전 계산 (Recall@K 평가용, text_encoder 삭제 전에 미리 계산)
    print(f"Test 질의 임베딩 사전 계산 중...", flush=True)
    test_query_texts = [item["query"] for item in fsc_qa_dataset_test]
    test_query_embs_list = []

    for i in range(0, len(test_query_texts), batch_size_encode):
        batch_texts = test_query_texts[i:i+batch_size_encode]
        batch_embs = torch.tensor(text_encoder.encode(batch_texts, show_progress_bar=False))
        test_query_embs_list.append(batch_embs)
        done = min(i + batch_size_encode, len(test_query_texts))
        print(f"  [{done:,}/{len(test_query_texts):,}] 처리 완료", flush=True)

    test_query_embs = torch.cat(test_query_embs_list, dim=0).to(device)
    print(f"✓ {len(test_query_embs):,}개 test 질의 임베딩 계산 완료")

    del text_encoder  # 메모리 정리
    torch.cuda.empty_cache()

    model.train()
    epochs = 10

    for epoch in range(epochs):
        total_loss = 0

        # 매 Epoch마다 전체 그래프의 노드 임베딩을 한 번 업데이트합니다. (Graph is static)
        print(f"Epoch {epoch+1}/{epochs} - 그래프 노드 임베딩 계산 중...")
        updated_node_embs = model(graph_data.x_dict, graph_data.edge_index_dict)
        clause_embs = updated_node_embs['clause'] # (Num_clauses x 256)
        print(f"Epoch {epoch+1}/{epochs} - 학습 중...")

        for i, item in enumerate(fsc_qa_dataset):
            optimizer.zero_grad()

            # (1) Query 벡터화 (사전 계산된 BGE-M3 임베딩 사용)
            q_emb = query_embs[i].unsqueeze(0)

            # (2) Positive Virtual Node 생성
            pos_indices = [clause_to_idx[c] for c in item["positive_clauses"] if c in clause_to_idx]
            if not pos_indices:
                continue
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

            # 메모리 정리 (매 50 스텝마다)
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()
                print(f"  [{i+1:,}/{len(fsc_qa_dataset):,}] Loss: {total_loss/(i+1):.4f}")
            
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss/len(fsc_qa_dataset):.4f}")

    # ==========================================
    # 5. Safetensors 포맷으로 모델 저장 (프로젝트 폴더 내)
    # ==========================================
    save_path = os.path.join(os.path.dirname(__file__), "financial_hgt_model.safetensors")

    # 모델의 가중치(state_dict)를 추출
    state_dict = model.state_dict()

    # Safetensors는 메모리 연속성(contiguous)을 요구하므로 변환 처리 (에러 방지용)
    contiguous_state_dict = {k: v.contiguous() for k, v in state_dict.items()}

    # 모델 저장 실행 (safetensors 형식은 매우 안전하고 torch 취약점 영향 없음)
    save_file(contiguous_state_dict, save_path)
    model_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"\n✅ 학습 완료!")
    print(f"   📁 저장 경로: {os.path.abspath(save_path)}")
    print(f"   💾 파일 크기: {model_size_mb:.2f} MB")
    print(f"   🔒 형식: safetensors (안전한 가중치 직렬화 형식)")

    # ==========================================
    # 6. Test 셋 평가 (Recall@K) - 참조 법률 개수별 분석
    # ==========================================
    print("\n📊 Test 셋 Recall@K 평가 중...")
    eval_results = evaluate_recall_at_k(model, graph_data, clause_to_idx, fsc_qa_dataset_test, test_query_embs, k_values=K_VALUES)

    if not eval_results:
        print("⚠️ 평가 가능한 test 질의가 없습니다 (정답 조항이 그래프에 매칭되지 않음).")
    else:
        eval_df = pd.DataFrame(eval_results)
        recall_cols = [f"recall@{k}" for k in K_VALUES]

        # 참조 법률 개수(num_laws)별 평균 Recall@K
        by_num_laws = eval_df.groupby("num_laws")[recall_cols].mean()
        by_num_laws["num_queries"] = eval_df.groupby("num_laws").size()
        by_num_laws = by_num_laws.reset_index().sort_values("num_laws")

        overall = eval_df[recall_cols].mean()
        overall_row = {"num_laws": "overall", "num_queries": len(eval_df)}
        overall_row.update(overall.to_dict())

        summary_df = pd.concat([by_num_laws, pd.DataFrame([overall_row])], ignore_index=True)

        print("\n[Recall@K 평가 결과 - 참조 법률 개수별]")
        print(summary_df.to_string(index=False))

        # 결과 파일 저장 (재확인 가능한 형식: 질의별 상세 CSV + 요약 CSV/JSON)
        eval_dir = os.path.join(os.path.dirname(__file__), "eval_results")
        os.makedirs(eval_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        detailed_csv_path = os.path.join(eval_dir, f"test_eval_detailed_{timestamp}.csv")
        summary_csv_path = os.path.join(eval_dir, f"test_eval_summary_{timestamp}.csv")
        summary_json_path = os.path.join(eval_dir, f"test_eval_summary_{timestamp}.json")

        eval_df.to_csv(detailed_csv_path, index=False, encoding="utf-8-sig")
        summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": timestamp,
                "k_values": K_VALUES,
                "num_test_queries_evaluated": len(eval_df),
                "by_num_laws": by_num_laws.to_dict(orient="records"),
                "overall": overall_row,
                "per_query": eval_df.to_dict(orient="records"),
            }, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Recall@K 평가 결과 저장 완료!")
        print(f"   📄 세부 결과(질의별): {detailed_csv_path}")
        print(f"   📄 요약(참조 법률 개수별): {summary_csv_path}")
        print(f"   📄 요약(JSON): {summary_json_path}")

if __name__ == "__main__":
    train()