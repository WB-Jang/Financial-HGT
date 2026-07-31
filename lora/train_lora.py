"""
lora/train_lora.py

BGE-M3 LoRA 파인튜닝 (질의 측만). QueryEncoder(MLP)의 대조군.

train_query_encoder.py와 '데이터·분할·손실·hard negative 일정·체크포인트 규칙'을
전부 동일하게 맞추고, 학습 대상만 잔차 MLP -> BGE LoRA로 바꾼다.
비교에서 달라지는 변수를 하나로 묶기 위한 설계다.

동일하게 유지하는 것:
  - fsc_dataset_preprocessing(test_size=300) 층화 분할, seed 42
  - build_retrieval_items의 조->항 정답 확장 (multi-positive)
  - 전체 코퍼스(9,311개) InfoNCE 분모  <- train_query_encoder에서 그대로 import
  - hard negative 재채굴 + margin hinge, 이웃 제외
  - val Hit@15 기준 best 체크포인트, best_val을 '베이스라인 점수'로 초기화
    (베이스라인보다 나쁜 모델은 저장하지 않음)

다르게 하는 것 (근거는 lora/README.md):
  - epochs 100 -> 12  (사전학습 백본 위 LoRA는 훨씬 빨리 수렴. MLP는 캐시 임베딩만
    곱하지만 LoRA는 매 step BGE forward+backward가 필요하다)
  - lr 3e-4 -> 1e-4   (사전학습 가중치를 건드리므로 보수적으로)
  - batch 32 -> micro 8 x accum 4
    이 손실은 질의마다 독립항이고 분모가 전체 코퍼스라 in-batch negative가 없다.
    따라서 누적 gradient가 batch 32와 '수학적으로 동일'하다.

실행:
  python lora/train_lora.py --test_size 300
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import (  # noqa: E402
    normalize_johang_key, fsc_dataset_preprocessing, encode_texts_cached, make_bge_encoder,
)
from retrieval_common import (  # noqa: E402
    K_VALUES, build_clause_index, build_retrieval_items, build_clause_adjacency,
    compute_metric_rows, compute_article_metric_rows, summarize_metrics,
)
from train_query_encoder import infonce_multi_positive  # noqa: E402  <- 손실은 재구현하지 않는다

from model import build_lora_bge, encode_queries_lora  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES_CSV = os.path.join(ROOT, 'data/nodes.csv')
TRIPLETS_CSV = os.path.join(ROOT, 'data/triplets.csv')
FSC_XLSX = os.path.join(ROOT, 'data/for_review_corrected.xlsx')


# ── 평가 ────────────────────────────────────────────────────────────────────

def rank_all_lora(model, tok, texts, clause_embs, max_k, max_len, batch_size):
    """질의 텍스트 -> LoRA 인코딩 -> 전체 조항 랭킹 상위 max_k."""
    was_training = model.training
    model.eval()
    q = encode_queries_lora(model, tok, texts, max_len=max_len, batch_size=batch_size)
    sims = q @ clause_embs.T
    topk = sims.topk(min(max_k, sims.size(1)), dim=1).indices
    if was_training:
        model.train()
    return [row.tolist() for row in topk]


def quick_val_metrics_lora(model, tok, val_items, clause_embs, max_len, batch_size, k=15):
    """검증용 Hit@k / 비율형 Recall@k (quick_val_metrics와 같은 정의)."""
    ranked_lists = rank_all_lora(model, tok, [it["query"] for it in val_items],
                                 clause_embs, k, max_len, batch_size)
    hits, recall_sum = 0, 0.0
    for ranked, it in zip(ranked_lists, val_items):
        pos = it["pos_idxs"]
        inter = len(set(ranked[:k]) & pos)
        hits += 1 if inter > 0 else 0
        recall_sum += inter / len(pos)
    n = max(len(val_items), 1)
    return hits / n, recall_sum / n


@torch.no_grad()
def mine_hard_negatives_lora(model, tok, samples, clause_embs, k, max_len, batch_size):
    """현재 모델로 전체 train 질의를 랭킹 -> 상위 비정답을 hard negative로 갱신.

    mine_hard_negatives(train_query_encoder.py:90-105)와 같은 로직. 질의를 텍스트에서
    다시 인코딩해야 해서 시그니처만 다르다.
    """
    was_training = model.training
    model.eval()
    q_all = encode_queries_lora(model, tok, [s["query"] for s in samples],
                                max_len=max_len, batch_size=batch_size)
    sims = q_all @ clause_embs.T
    fetch = min(k * 8, sims.size(1))          # 이웃 제외로 후보가 줄 수 있어 넉넉히
    top_idx = sims.topk(fetch, dim=1).indices
    for i, sample in enumerate(samples):
        forbidden = sample["pos_idxs"] | sample.get("neighbor_set", set())
        sample["hard_neg_idxs"] = [
            idx.item() for idx in top_idx[i] if idx.item() not in forbidden
        ][:k]
    if was_training:
        model.train()


# ── 게이트 ──────────────────────────────────────────────────────────────────

def check_pooling_parity(model, tok, texts, ref_embs, max_len, batch_size, tol=0.999):
    """게이트 ①: 어댑터가 항등(B=0)인 상태에서 우리 CLS 경로가 SentenceTransformer
    임베딩과 같은 벡터를 내는지 확인.

    조항 임베딩은 SentenceTransformer가 만들었고 학습은 AutoModel을 직접 쓰므로,
    풀링/정규화/truncation이 어긋나면 질의와 조항이 서로 다른 공간에 놓인다.
    그 상태로 학습하면 이후 비교가 전부 무의미해지므로 하드 게이트로 둔다.
    """
    ours = encode_queries_lora(model, tok, texts, max_len=max_len, batch_size=batch_size)
    ref = F.normalize(ref_embs.float(), dim=-1).to(ours.device)
    cos = (ours * ref).sum(dim=-1)
    print(f"[게이트 ①] 풀링 정합성: 코사인 min={cos.min():.6f} mean={cos.mean():.6f} (기준 {tol})")
    if cos.min() < tol:
        raise SystemExit(
            f"❌ 풀링 불일치 (min cos={cos.min():.6f} < {tol}).\n"
            f"   질의와 조항이 다른 공간에 있습니다. max_len({max_len}) 절단이 원인일 수 있으니"
            f" --max_len을 늘려 재시도하세요. 이 상태로 학습하면 비교가 무효입니다."
        )
    print("   통과 — 질의/조항이 같은 공간에 있습니다.")


# ── 메인 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--micro_batch", type=int, default=8, help="GPU에 한번에 올릴 질의 수")
    parser.add_argument("--accum", type=int, default=4, help="micro_batch x accum = 유효 배치(=32)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard_neg_k", type=int, default=10, help="0이면 hard negative 비활성")
    parser.add_argument("--hard_neg_warmup", type=int, default=2)
    parser.add_argument("--hard_neg_interval", type=int, default=2)
    parser.add_argument("--hard_neg_margin", type=float, default=0.1)
    parser.add_argument("--test_size", type=int, default=300,
                        help="test 질의 수. MLP arm과 반드시 같은 값이어야 분할이 일치한다")
    parser.add_argument("--exclude_neighbors", type=int, default=1)
    parser.add_argument("--max_entity_df", type=int, default=20)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=256, help="질의 토큰 절단 길이")
    parser.add_argument("--eval_batch", type=int, default=32, help="추론 시 배치(학습보다 크게)")
    parser.add_argument("--amp", type=int, default=1, help="1이면 fp16 autocast (T4 권장)")
    parser.add_argument("--out_dir", default=os.path.join(ROOT, "lora/checkpoints"))
    parser.add_argument("--results_dir", default=os.path.join(ROOT, "lora/results"))
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = bool(args.amp) and device.type == 'cuda'   # autocast('cuda')는 GPU에서만
    print(f"사용 기기: {device} | fp16 autocast: {use_amp}")

    # 1. 데이터 구성 — train_query_encoder.py와 완전히 동일한 경로를 탄다
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
    print(f"조항 노드 {len(clause_list):,}개 | train {len(train_items)}건(제외 {tr_skip}) | "
          f"test {len(test_items)}건(제외 {te_skip})")

    # 2. 조항 임베딩 — 학습하지 않는다. MLP arm과 같은 캐시 파일을 그대로 쓴다.
    encoder = make_bge_encoder()
    clause_embs = encode_texts_cached(encoder, clause_texts, 'clause_embs')
    ref_qemb = encode_texts_cached(encoder, [it["query"] for it in train_items], 'fsc_query_embs')
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    clause_embs = F.normalize(clause_embs.float(), dim=-1).to(device)

    # 3. train/val 분리 (train_query_encoder.py:203-211과 같은 규칙, 같은 seed)
    idxs = list(range(len(train_items)))
    random.shuffle(idxs)
    n_val = max(1, int(len(idxs) * args.val_ratio))
    val_ids, tr_ids = idxs[:n_val], idxs[n_val:]
    val_items = [train_items[i] for i in val_ids]
    tr_items = [train_items[i] for i in tr_ids]
    print(f"학습 {len(tr_items)}건 / 검증 {len(val_items)}건")

    for s in tr_items:
        s["hard_neg_idxs"] = []
        s["pos_list"] = sorted(s["pos_idxs"])
        s["neighbor_set"] = set()
        s["neighbor_list"] = []

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
            nbr -= s["pos_idxs"]
            s["neighbor_set"] = nbr
            s["neighbor_list"] = sorted(nbr)
        avg_nbr = sum(len(s["neighbor_list"]) for s in tr_items) / max(len(tr_items), 1)
        print(f"이웃 제외 활성: 질의당 평균 제외 조항 {avg_nbr:.1f}개")

    # 4. 모델
    model, tok = build_lora_bge(args.lora_r, args.lora_alpha, args.lora_dropout, device)

    # 게이트 ① — 어댑터가 아직 항등(B=0)이므로 순수 BGE와 같은 벡터가 나와야 한다
    n_probe = min(32, len(train_items))
    check_pooling_parity(model, tok, [train_items[i]["query"] for i in range(n_probe)],
                         ref_qemb[:n_probe], args.max_len, args.eval_batch)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # 게이트 ② — B=0이므로 학습 전 성능은 순수 BGE 베이스라인과 같아야 한다
    hit15, rec15 = quick_val_metrics_lora(model, tok, val_items, clause_embs,
                                          args.max_len, args.eval_batch)
    print(f"[게이트 ②] epoch 0 = 베이스라인: val Hit@15={hit15:.3f} Recall@15={rec15:.3f}")
    print("   evaluate_baseline.py의 값과 일치해야 합니다 (LoRA B=0 -> W_eff=W_0).")

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = hit15   # 베이스라인보다 나빠진 모델은 저장하지 않음 (MLP arm과 같은 규칙)
    model.save_pretrained(args.out_dir)

    # 5. 학습 루프
    model.train()
    use_hard_neg = args.hard_neg_k > 0
    for epoch in range(1, args.epochs + 1):
        if use_hard_neg and epoch > args.hard_neg_warmup:
            if (epoch - args.hard_neg_warmup) % args.hard_neg_interval == 1:
                mine_hard_negatives_lora(model, tok, tr_items, clause_embs,
                                         args.hard_neg_k, args.max_len, args.eval_batch)
                print(f"  epoch {epoch}: hard negatives 재채굴 (k={args.hard_neg_k})")

        order = list(range(len(tr_items)))
        random.shuffle(order)
        epoch_loss, n_micro = 0.0, 0
        optimizer.zero_grad()

        for step, start in enumerate(range(0, len(order), args.micro_batch)):
            batch_ids = order[start:start + args.micro_batch]
            q = encode_queries_lora(model, tok, [tr_items[i]["query"] for i in batch_ids],
                                    max_len=args.max_len, batch_size=len(batch_ids),
                                    grad=True, amp=use_amp)
            pos_batch = [tr_items[i]["pos_list"] for i in batch_ids]
            hard_batch = [tr_items[i]["hard_neg_idxs"] for i in batch_ids]
            nbr_batch = [tr_items[i]["neighbor_list"] for i in batch_ids] if args.exclude_neighbors else None

            loss = infonce_multi_positive(q, clause_embs, pos_batch, hard_batch,
                                          args.temp, args.hard_neg_margin, neighbor_batch=nbr_batch)
            epoch_loss += loss.item()
            n_micro += 1

            # 질의별 독립항이므로 micro-batch 평균을 accum으로 나눠 더하면 batch 32와 동일
            scaler.scale(loss / args.accum).backward()

            if (step + 1) % args.accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # 남은 micro-batch 처리
        if n_micro % args.accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        scheduler.step()
        hit15, rec15 = quick_val_metrics_lora(model, tok, val_items, clause_embs,
                                              args.max_len, args.eval_batch)
        marker = ""
        if hit15 > best_val:
            best_val = hit15
            model.save_pretrained(args.out_dir)
            marker = "  <- best 저장"
        print(f"epoch {epoch:3d} | loss={epoch_loss/max(n_micro,1):.4f} | "
              f"val Hit@15={hit15:.3f} Recall@15={rec15:.3f}{marker}")

    print(f"\n학습 완료. best val Hit@15={best_val:.4f} ({args.out_dir})")

    # 6. best 어댑터 재로드 후 test 평가 + 질의 임베딩 export
    from peft import PeftModel
    from transformers import AutoModel
    from model import MODEL_ID
    base = AutoModel.from_pretrained(MODEL_ID)
    model = PeftModel.from_pretrained(base, args.out_dir).to(device)
    model.eval()

    test_texts = [it["query"] for it in test_items]
    q_test = encode_queries_lora(model, tok, test_texts, max_len=args.max_len,
                                 batch_size=args.eval_batch)

    # 검색 평가는 GPU 없이도 돌 수 있도록 질의 임베딩을 파일로 남긴다
    from safetensors.torch import save_file
    emb_dir = os.path.join(ROOT, 'emb_cache')
    os.makedirs(emb_dir, exist_ok=True)
    emb_path = os.path.join(emb_dir, 'fsc_query_embs_lora.safetensors')
    save_file({'embeddings': q_test.cpu().contiguous()}, emb_path)
    print(f"💾 LoRA 질의 임베딩 저장: {emb_path} ({len(test_texts)}건)")

    sims = q_test @ clause_embs.T
    full_ranking = sims.argsort(dim=1, descending=True)
    full_ranked_lists = [row.tolist() for row in full_ranking]
    max_k = max(K_VALUES)

    # 주 지표 = 항(paragraph) 단위
    para_rows, mrr_col = compute_metric_rows([r[:max_k] for r in full_ranked_lists],
                                             test_items, K_VALUES)
    para_df = pd.DataFrame(para_rows)
    para_summary, para_by, para_overall, recall_cols, hit_cols = summarize_metrics(
        para_df, K_VALUES, mrr_col)

    # 서브 지표 = 조(article) 단위
    art_rows, _ = compute_article_metric_rows(full_ranked_lists, test_items, clause_list, K_VALUES)
    art_df = pd.DataFrame(art_rows)
    art_summary, art_by, art_overall, _, _ = summarize_metrics(art_df, K_VALUES, mrr_col)

    pd.set_option("display.width", 200)
    print("\n=== [주 지표] 항(paragraph) 단위 ===")
    print(para_summary[["num_laws", "num_queries"] + recall_cols].to_string(index=False))
    print(para_summary[["num_laws", "num_queries"] + hit_cols + [mrr_col]].to_string(index=False))
    print("\n=== [서브 지표] 조(article) 단위 ===")
    print(art_summary[["num_laws", "num_queries"] + recall_cols].to_string(index=False))

    os.makedirs(args.results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    para_df.to_csv(os.path.join(args.results_dir, f"lora_paragraph_{ts}.csv"),
                   index=False, encoding="utf-8-sig")
    art_df.to_csv(os.path.join(args.results_dir, f"lora_article_{ts}.csv"),
                  index=False, encoding="utf-8-sig")
    with open(os.path.join(args.results_dir, f"lora_summary_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": ts,
            "method": f"BGE-M3 LoRA r={args.lora_r} (query-side only)",
            "hyperparams": vars(args),
            "trainable_params": sum(p.numel() for p in trainable),
            "best_val_hit15": best_val,
            "k_values": K_VALUES,
            "num_test_queries_evaluated": len(para_df),
            "paragraph_level": {"by_num_laws": para_by.to_dict(orient="records"),
                                "overall": para_overall},
            "article_level": {"by_num_laws": art_by.to_dict(orient="records"),
                              "overall": art_overall},
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 저장 완료: lora/results/lora_summary_{ts}.json")


if __name__ == "__main__":
    main()
