"""
train_stage2.py — Gen1 Stage 2: 동결된 HGT 노드 임베딩에 질의를 정렬한다.

원본: KG-search_PPR_GNN_Transformer/src/training/train_retrieval.py

Stage 1이 만든 256d 노드 임베딩을 **동결**하고 QueryEncoder256(1024→512→256)만
학습한다. 이 방향이 Gen1이 붕괴하지 않은 이유의 핵심이다 — 문서 공간이 어떤 모양이든
질의 쪽이 거기에 맞춰 간다.

손실·hard negative 재채굴·이웃 제외는 train_query_encoder.py에서 그대로 import한다.
MLP arm과 학습 신호를 비트 단위로 같게 유지하기 위해서다. 분할(seed 42, test_size),
정답 확장, 이웃 집합도 레포 공용 함수를 써서 자동으로 일치한다.

⚠️ MLP arm과 다른 점: 출력이 256d라 잔차 연결(x + MLP(x))이 성립하지 않는다. 따라서
'학습 시작 = 순수 BGE 베이스라인' 보장이 없고, best 체크포인트의 baseline seeding도
할 수 없다(다른 공간이라 비교 자체가 불가). Gen1과 같은 조건이며 결과에 명시할 것.

사용법:
    python hgt_gen1/train_stage2.py --test_size 300 --epochs 100
"""

import argparse
import json
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from data_loader import (encode_texts_cached, fsc_dataset_preprocessing,  # noqa: E402
                         make_bge_encoder, normalize_johang_key)
from hgt_gen1.model import QueryEncoder256  # noqa: E402
from retrieval_common import (build_clause_adjacency, build_clause_index,  # noqa: E402
                              build_retrieval_items)
from train_query_encoder import infonce_multi_positive, mine_hard_negatives  # noqa: E402


def evaluate(model, qemb, clause_embs, items, k=15):
    """항 단위 Hit@k / Recall@k."""
    model.eval()
    with torch.no_grad():
        sims = model(qemb) @ clause_embs.T
        top = sims.topk(k, dim=1).indices.tolist()
    hit = rec = 0.0
    for row, item in zip(top, items):
        pos = item["pos_idxs"]
        found = len(pos & set(row))
        hit += 1.0 if found else 0.0
        rec += found / len(pos)
    model.train()
    return hit / len(items), rec / len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node_emb", default="hgt_gen1/checkpoints/node_embeddings_best.safetensors")
    ap.add_argument("--out_dir", default="hgt_gen1/checkpoints")
    ap.add_argument("--test_size", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--hard_neg_k", type=int, default=10)
    ap.add_argument("--hard_neg_warmup", type=int, default=10)
    ap.add_argument("--hard_neg_interval", type=int, default=5)
    ap.add_argument("--exclude_neighbors", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 기기: {device}")

    nodes_df = pd.read_csv(os.path.join(ROOT, "data/nodes.csv"))
    triplets_df = pd.read_csv(os.path.join(ROOT, "data/triplets.csv"))
    for df, hang in ((nodes_df, True), (triplets_df, False)):
        df["new_johang"] = [
            normalize_johang_key(l, a, h)
            for l, a, h in zip(df["law_nm"], df["article_number"],
                               df["hang_number"] if hang else [None] * len(df))
        ]
    clause_list, clause_texts = build_clause_index(nodes_df)

    # 문서 공간 = Stage 1의 동결 HGT 임베딩
    clause_embs = load_file(os.path.join(ROOT, args.node_emb))["node_embeddings"]
    clause_embs = F.normalize(clause_embs.float(), dim=-1).to(device)
    if clause_embs.size(0) != len(clause_list):
        raise ValueError(f"노드 수 불일치: emb {clause_embs.size(0)} vs clause {len(clause_list)}")
    print(f"동결 노드 임베딩: {tuple(clause_embs.shape)}")

    fsc = fsc_dataset_preprocessing(file=os.path.join(ROOT, "data/for_review_corrected.xlsx"),
                                    nodes_df=nodes_df, test_size=args.test_size)
    train_items, _ = build_retrieval_items(fsc[fsc.split == "train"].reset_index(drop=True), clause_list)
    test_items, _ = build_retrieval_items(fsc[fsc.split == "test"].reset_index(drop=True), clause_list)

    encoder = make_bge_encoder()
    train_qemb = encode_texts_cached(encoder, [it["query"] for it in train_items], "fsc_query_embs").to(device)
    test_qemb = encode_texts_cached(encoder, [it["query"] for it in test_items], "fsc_query_embs").to(device)
    del encoder
    torch.cuda.empty_cache()

    neighbors = {}
    if args.exclude_neighbors:
        edge_w = build_clause_adjacency(clause_list, triplets_df)
        adj = {}
        for (i, j) in edge_w:
            adj.setdefault(i, set()).add(j)
            adj.setdefault(j, set()).add(i)
        for it in train_items:
            nb = set()
            for p in it["pos_idxs"]:
                nb |= adj.get(p, set())
            it["neighbor_set"] = nb - it["pos_idxs"]
        neighbors = {"on": True}

    rng = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(train_items), generator=rng).tolist()
    n_val = max(1, int(len(idx) * 0.1))
    val_items = [train_items[i] for i in idx[:n_val]]
    val_qemb = train_qemb[idx[:n_val]]
    fit_items = [train_items[i] for i in idx[n_val:]]
    fit_qemb = train_qemb[idx[n_val:]]
    print(f"학습 {len(fit_items)}건 / 검증 {len(val_items)}건 / test {len(test_items)}건")

    model = QueryEncoder256(hidden_dim=clause_embs.size(1)).to(device)
    print(f"학습 파라미터: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for it in fit_items:
        it.setdefault("hard_neg_idxs", [])
    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    hit0, rec0 = evaluate(model, val_qemb, clause_embs, val_items)
    print(f"[학습 전] val Hit@15={hit0:.3f} Recall@15={rec0:.3f} "
          f"(무작위 초기화 — 잔차가 없어 베이스라인 출발 보장 없음)")

    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        if epoch > args.hard_neg_warmup and (epoch - args.hard_neg_warmup) % args.hard_neg_interval == 1:
            print(f"  epoch {epoch}: hard negatives 재채굴 (k={args.hard_neg_k})")
            mine_hard_negatives(model, fit_qemb, clause_embs, fit_items, args.hard_neg_k)

        model.train()
        order = torch.randperm(len(fit_items), generator=rng).tolist()
        total = 0.0
        for s in range(0, len(order), args.batch_size):
            batch = order[s:s + args.batch_size]
            opt.zero_grad()
            q = model(fit_qemb[batch])
            loss = infonce_multi_positive(
                q, clause_embs,
                [fit_items[i]["pos_idxs"] for i in batch],
                [fit_items[i]["hard_neg_idxs"] for i in batch],
                args.temp, args.margin,
                [fit_items[i].get("neighbor_set", set()) for i in batch] if neighbors else None,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * len(batch)
        sched.step()

        hit, rec = evaluate(model, val_qemb, clause_embs, val_items)
        mark = ""
        if hit > best_val:
            best_val, mark = hit, "  <- best 저장"
            save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
                      os.path.join(out_dir, "query_encoder_hgt_best.safetensors"))
        print(f"epoch {epoch:3d} | loss={total / len(order):.4f} | "
              f"val Hit@15={hit:.3f} Recall@15={rec:.3f}{mark}")

    with open(os.path.join(out_dir, "stage2_config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "best_val_hit15": best_val,
                   "n_train": len(fit_items), "n_val": len(val_items),
                   "n_test": len(test_items)}, f, ensure_ascii=False, indent=2)
    print(f"\n최적 val Hit@15 {best_val:.3f} -> {out_dir}/query_encoder_hgt_best.safetensors")


if __name__ == "__main__":
    main()
