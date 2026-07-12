"""
train_query_encoder.py

Stage 2: QueryEncoder(소형 잔차 MLP)만 학습하는 검색 학습 스크립트.
KG-search_PPR_GNN_Transformer의 train_retrieval.py 레시피를 이식한 것.

구조 (Stage 1/2 분리):
  Stage 1 (고정): 조항 BGE-M3 임베딩 (emb_cache/ 재사용, 학습 없음)
  Stage 2 (학습): 질의 1024d -> 1024d 잔차 MLP 하나만 학습

핵심 설계 (기존 train.py의 HGT 방식과의 차이):
  - 조항(문서) 쪽은 절대 건드리지 않음 -> 사전학습 BGE 공간 보존
  - InfoNCE 분모가 '전체 조항 7천여 개' (기존: 정답1+오답3의 4지선다)
  - multi-positive: 한 질의의 여러 정답 조항을 동시에 반영
  - hard negative를 warmup 후 매 interval epoch마다 현재 모델로 재채굴
    + margin hinge (기존: 학습 전 1회 고정 -> 몇 epoch 뒤 gradient 소실)
  - 검증 분리 + 매 epoch Hit@15/Recall@15 측정 + best 체크포인트 저장
  - 잔차+0초기화로 시작 성능 == 순수 BGE 베이스라인 (아래로 못 떨어짐)

실행:
  python train_query_encoder.py            # 기본 100 epochs
  python train_query_encoder.py --epochs 50 --lr 3e-4

학습 후 자동으로 test 셋 평가를 수행하고 eval_results/stage2_eval_*.csv/.json 저장.
(evaluate_baseline.py 결과와 같은 포맷 -> 나란히 비교)
"""

import argparse
import os
import json
import random
from collections import defaultdict
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from safetensors.torch import save_file, load_file

from data_loader import normalize_johang_key, fsc_dataset_preprocessing, encode_texts_cached
from query_encoder import QueryEncoder
from retrieval_common import (
    K_VALUES, build_clause_index, build_retrieval_items, build_clause_adjacency,
    compute_metric_rows, summarize_metrics, emb_tag,
)

NODES_CSV = './data/nodes.csv'
TRIPLETS_CSV = './data/triplets.csv'
FSC_XLSX = './data/for_review_corrected.xlsx'


# ── 손실 함수 ────────────────────────────────────────────────────────────────

def infonce_multi_positive(q_emb, clause_embs, pos_idxs_batch, hard_neg_batch, temp, margin,
                           neighbor_batch=None):
    """전체 코퍼스 분모의 multi-positive InfoNCE + hard negative margin hinge.

    q_emb: (B, D) L2-normalized
    clause_embs: (N, D) L2-normalized (고정)
    pos_idxs_batch: 질의별 정답 인덱스 리스트
    hard_neg_batch: 질의별 hard negative 인덱스 리스트 (빈 리스트면 hinge 생략)
    neighbor_batch: 질의별 '정답의 그래프 이웃' 인덱스 리스트. 지정 시 분모(logsumexp)에서
        제외하여 거짓 음성(형제 항/공유 엔터티 조항을 오답으로 벌주는 것)을 방지. (정답은 이미 제외됨)
    """
    sims = q_emb @ clause_embs.T          # (B, N) 코사인 유사도
    logits = sims / temp

    loss = q_emb.new_zeros(())
    for i, pos_list in enumerate(pos_idxs_batch):
        row = logits[i]
        if neighbor_batch is not None and neighbor_batch[i]:
            # 이웃 조항을 분모에서 제외 (masked_fill로 -inf -> exp=0)
            row = row.clone()
            row[neighbor_batch[i]] = float('-inf')
        log_denom = torch.logsumexp(row, dim=0)
        log_pos = logits[i][pos_list].mean()
        step_loss = log_denom - log_pos

        hard_negs = hard_neg_batch[i]
        if hard_negs:
            best_pos_sim = sims[i][pos_list].max()
            hard_neg_sims = sims[i][hard_negs]
            step_loss = step_loss + F.relu(hard_neg_sims - best_pos_sim + margin).mean()

        loss = loss + step_loss
    return loss / len(pos_idxs_batch)


@torch.no_grad()
def mine_hard_negatives(model, train_qemb, clause_embs, samples, k):
    """현재 모델로 전체 train 질의를 랭킹 -> 상위 비정답을 hard negative로 갱신.
    정답 및 정답의 그래프 이웃(sample['neighbor_set'])은 제외 -> 숨은 정답을 오답으로
    가르치는 것을 방지."""
    model.eval()
    q_all = model(train_qemb)                       # (n_train, D)
    sims = q_all @ clause_embs.T                    # (n_train, N)
    fetch = min(k * 8, sims.size(1))                # 이웃 제외로 후보가 줄 수 있어 넉넉히 확보
    top_idx = sims.topk(fetch, dim=1).indices
    for i, sample in enumerate(samples):
        forbidden = sample["pos_idxs"] | sample.get("neighbor_set", set())
        sample["hard_neg_idxs"] = [
            idx.item() for idx in top_idx[i] if idx.item() not in forbidden
        ][:k]
    model.train()


# ── 평가 ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def rank_all(model, qemb, clause_embs, max_k):
    """질의 임베딩 -> 모델 통과 -> 전체 조항 랭킹 상위 max_k 인덱스 리스트."""
    model.eval()
    q = model(qemb)
    sims = q @ clause_embs.T
    topk = sims.topk(min(max_k, sims.size(1)), dim=1).indices
    model.train()
    return [row.tolist() for row in topk]


def quick_val_metrics(model, val_qemb, clause_embs, val_items, k=15):
    """검증용 Hit@k / 비율형 Recall@k (체크포인트 선택 기준)."""
    ranked_lists = rank_all(model, val_qemb, clause_embs, k)
    hits, recall_sum = 0, 0.0
    for ranked, it in zip(ranked_lists, val_items):
        pos = it["pos_idxs"]
        inter = len(set(ranked[:k]) & pos)
        hits += 1 if inter > 0 else 0
        recall_sum += inter / len(pos)
    n = max(len(val_items), 1)
    return hits / n, recall_sum / n


# ── 메인 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard_neg_k", type=int, default=10, help="0이면 hard negative 비활성")
    parser.add_argument("--hard_neg_warmup", type=int, default=10)
    parser.add_argument("--hard_neg_interval", type=int, default=5)
    parser.add_argument("--hard_neg_margin", type=float, default=0.1)
    parser.add_argument("--clause_emb", default=None,
                        help="조항 임베딩 safetensors 파일 (예: data/clause_emb_smooth.safetensors). "
                             "미지정 시 원본 BGE 임베딩(캐시) 사용")
    parser.add_argument("--test_size", type=int, default=100,
                        help="test 질의 수 (평가가능 질의에서 층화추출). baseline/rerank와 동일 값 사용 필수")
    parser.add_argument("--exclude_neighbors", type=int, default=1,
                        help="1이면 정답의 그래프 이웃(형제 항/공유 엔터티 조항)을 hard negative와 "
                             "InfoNCE 분모에서 제외 (거짓 음성 방지). 0이면 비활성")
    parser.add_argument("--max_entity_df", type=int, default=20,
                        help="이웃 그래프 구성 시 허브 엔터티 컷오프 (초과 연결 엔터티 제외)")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 기기: {device}")

    # 1. 데이터 구성 (train.py / evaluate_baseline.py와 동일한 분할·확장 로직)
    nodes_df = pd.read_csv(NODES_CSV)
    nodes_df['new_johang'] = [
        normalize_johang_key(law_nm, article_number, hang_number)
        for law_nm, article_number, hang_number in zip(
            nodes_df['law_nm'], nodes_df['article_number'], nodes_df['hang_number']
        )
    ]
    fsc = fsc_dataset_preprocessing(file=FSC_XLSX, nodes_df=nodes_df, test_size=args.test_size)
    fsc_train = fsc[fsc['split'] == 'train'].reset_index(drop=True)
    fsc_test = fsc[fsc['split'] == 'test'].reset_index(drop=True)

    clause_list, clause_texts = build_clause_index(nodes_df)
    train_items, tr_skip = build_retrieval_items(fsc_train, clause_list)
    test_items, te_skip = build_retrieval_items(fsc_test, clause_list)
    print(f"조항 노드 {len(clause_list):,}개 | train 질의 {len(train_items)}건(제외 {tr_skip}) | test 질의 {len(test_items)}건(제외 {te_skip})")

    # 2. 임베딩 준비 (전부 캐시 재사용 - Stage 1은 고정 BGE 임베딩 or 평활화 버전)
    encoder = SentenceTransformer('BAAI/bge-m3', device='cpu')
    if args.clause_emb:
        clause_embs = load_file(args.clause_emb)['embeddings']
        assert clause_embs.size(0) == len(clause_list), \
            f"조항 임베딩 크기 불일치: {clause_embs.size(0)} != {len(clause_list)} (데이터가 바뀌었으면 build_smoothed_clause_emb.py를 다시 실행하세요)"
        print(f"조항 임베딩 로드: {args.clause_emb}")
    else:
        clause_embs = encode_texts_cached(encoder, clause_texts, 'clause_embs')
    train_qemb_all = encode_texts_cached(encoder, [it["query"] for it in train_items], 'fsc_query_embs')
    test_qemb = encode_texts_cached(encoder, [it["query"] for it in test_items], 'fsc_query_embs')
    del encoder

    clause_embs = F.normalize(clause_embs.float(), dim=-1).to(device)   # (N, 1024) 고정
    train_qemb_all = F.normalize(train_qemb_all.float(), dim=-1).to(device)
    test_qemb = F.normalize(test_qemb.float(), dim=-1).to(device)

    # 3. train/val 분리 (질의 단위)
    idxs = list(range(len(train_items)))
    random.shuffle(idxs)
    n_val = max(1, int(len(idxs) * args.val_ratio))
    val_ids, tr_ids = idxs[:n_val], idxs[n_val:]
    val_items = [train_items[i] for i in val_ids]
    tr_items = [train_items[i] for i in tr_ids]
    val_qemb = train_qemb_all[val_ids]
    tr_qemb = train_qemb_all[tr_ids]
    print(f"학습 {len(tr_items)}건 / 검증 {len(val_items)}건")

    for s in tr_items:
        s["hard_neg_idxs"] = []
        s["pos_list"] = sorted(s["pos_idxs"])   # pos_idxs를 리스트로도 (텐서 인덱싱용)
        s["neighbor_set"] = set()
        s["neighbor_list"] = []

    # 3-1. (선택) 정답의 그래프 이웃 인덱스 구성 -> hard negative / InfoNCE 분모에서 제외
    if args.exclude_neighbors:
        triplets_df = pd.read_csv(TRIPLETS_CSV)
        triplets_df['new_johang'] = [
            normalize_johang_key(law_nm, article_number)
            for law_nm, article_number in zip(triplets_df['law_nm'], triplets_df['article_number'])
        ]
        edge_w = build_clause_adjacency(clause_list, triplets_df, args.max_entity_df)
        neighbors = defaultdict(set)
        for (i, j) in edge_w:
            neighbors[i].add(j)
            neighbors[j].add(i)
        for s in tr_items:
            nbr = set()
            for p in s["pos_idxs"]:
                nbr |= neighbors.get(p, set())
            nbr -= s["pos_idxs"]                 # 정답은 이웃 목록에서 제외 (분모에 남아야 함)
            s["neighbor_set"] = nbr
            s["neighbor_list"] = sorted(nbr)
        avg_nbr = sum(len(s["neighbor_list"]) for s in tr_items) / max(len(tr_items), 1)
        print(f"이웃 제외 활성: 질의당 평균 제외 조항 {avg_nbr:.1f}개")

    # 4. 모델/옵티마이저 (학습 대상은 QueryEncoder 하나뿐)
    model = QueryEncoder(dim=clause_embs.size(1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 학습 전 성능 = 베이스라인과 동일해야 함 (잔차+0초기화 확인용)
    hit15, rec15 = quick_val_metrics(model, val_qemb, clause_embs, val_items)
    print(f"[epoch 0 = 베이스라인] val Hit@15={hit15:.3f} Recall@15={rec15:.3f}")

    ckpt_path = os.path.join(os.path.dirname(__file__), "query_encoder_best.safetensors")
    best_val = hit15  # 시작점(베이스라인)보다 나빠진 모델은 저장하지 않음
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, ckpt_path)

    # 5. 학습 루프
    model.train()
    use_hard_neg = args.hard_neg_k > 0
    for epoch in range(1, args.epochs + 1):
        if use_hard_neg and epoch > args.hard_neg_warmup:
            if (epoch - args.hard_neg_warmup) % args.hard_neg_interval == 1:
                mine_hard_negatives(model, tr_qemb, clause_embs, tr_items, args.hard_neg_k)
                print(f"  epoch {epoch}: hard negatives 재채굴 (k={args.hard_neg_k})")

        order = list(range(len(tr_items)))
        random.shuffle(order)
        epoch_loss, n_batches = 0.0, 0

        for start in range(0, len(order), args.batch_size):
            batch_ids = order[start:start + args.batch_size]
            q = model(tr_qemb[batch_ids])
            pos_batch = [tr_items[i]["pos_list"] for i in batch_ids]
            hard_batch = [tr_items[i]["hard_neg_idxs"] for i in batch_ids]
            nbr_batch = [tr_items[i]["neighbor_list"] for i in batch_ids] if args.exclude_neighbors else None

            loss = infonce_multi_positive(q, clause_embs, pos_batch, hard_batch,
                                          args.temp, args.hard_neg_margin, neighbor_batch=nbr_batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        hit15, rec15 = quick_val_metrics(model, val_qemb, clause_embs, val_items)
        marker = ""
        if hit15 > best_val:
            best_val = hit15
            save_file({k: v.contiguous() for k, v in model.state_dict().items()}, ckpt_path)
            marker = "  <- best 저장"
        print(f"epoch {epoch:3d} | loss={epoch_loss/max(n_batches,1):.4f} | val Hit@15={hit15:.3f} Recall@15={rec15:.3f}{marker}")

    print(f"\n학습 완료. best val Hit@15={best_val:.4f} ({ckpt_path})")

    # 6. best 모델로 test 셋 최종 평가 (evaluate_baseline.py와 동일 지표/포맷)
    model.load_state_dict(load_file(ckpt_path))
    max_k = max(K_VALUES)
    ranked_lists = rank_all(model, test_qemb, clause_embs, max_k)
    rows, mrr_col = compute_metric_rows(ranked_lists, test_items, K_VALUES)
    eval_df = pd.DataFrame(rows)
    summary_df, by_num_laws, overall_row, recall_cols, hit_cols = summarize_metrics(eval_df, K_VALUES, mrr_col)

    pd.set_option("display.width", 200)
    print("\n[Stage 2: 비율형 Recall@K - baseline_eval_summary와 동일 정의]")
    print(summary_df[["num_laws", "num_queries"] + recall_cols].to_string(index=False))
    print("\n[Stage 2: Hit@K / MRR]")
    print(summary_df[["num_laws", "num_queries"] + hit_cols + [mrr_col]].to_string(index=False))

    # 파일명 규칙: stage2_{origEmb|smoothEmb}_{summary|detailed}_{ts}
    eval_dir = os.path.join(os.path.dirname(__file__), "eval_results")
    os.makedirs(eval_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = emb_tag(args.clause_emb)
    eval_df.to_csv(os.path.join(eval_dir, f"stage2_{tag}_detailed_{ts}.csv"), index=False, encoding="utf-8-sig")
    summary_df.to_csv(os.path.join(eval_dir, f"stage2_{tag}_summary_{ts}.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(eval_dir, f"stage2_{tag}_summary_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": ts,
            "method": f"Stage2 QueryEncoder (clause index: {tag})",
            "hyperparams": vars(args),
            "best_val_hit15": best_val,
            "k_values": K_VALUES,
            "num_test_queries_evaluated": len(eval_df),
            "by_num_laws": by_num_laws.to_dict(orient="records"),
            "overall": overall_row,
            "per_query": eval_df.to_dict(orient="records"),
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Stage 2 평가 저장 완료: eval_results/stage2_{tag}_summary_{ts}.csv")
    print("   evaluate_baseline.py 결과와 나란히 비교하세요 - 같은 test 분할, 같은 지표입니다.")


if __name__ == "__main__":
    main()
