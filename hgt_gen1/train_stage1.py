"""
train_stage1.py — Gen1 방식 HGT 학습 (링크예측 회귀).

원본: KG-search_PPR_GNN_Transformer/src/training/train_multilaw.py

목표:
  양성 = 엣지, target = build_edges.py가 계산한 정규화 co-occurrence
      INTRA          shared_count / max_shared_count
      SIBLING/CROSS  co_occurrence / max_co_occurrence
  음성 = 랜덤 노드쌍, target = 0 (양성 대비 neg_ratio배, epoch마다 재추출)
  손실 = MSE_intra + lambda_cross * MSE_cross      (cosine(emb_i, emb_j) ↔ target)

구 train.py(Gen2 HGT)와의 차이가 여기 전부 있다:
  - 질의를 전혀 쓰지 않는다. 동결 BGE 질의 공간에 문서를 맞추라고 요구하지 않는다.
    문서 공간은 자기완결적으로 만들어지고, 질의는 Stage 2에서 이 공간에 맞춘다.
  - 손실이 개별 노드쌍의 코사인을 직접 제약한다. attention 풀링된 centroid가 아니다.
  - 엣지 기반이라 엣지에 등장하는 모든 노드가 매 epoch 감독을 받는다 (실측 95.3%).

사용법:
    python hgt_gen1/train_stage1.py --epochs 100 --hidden 256 --layers 2
"""

import argparse
import json
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from data_loader import encode_texts_cached, make_bge_encoder, normalize_johang_key  # noqa: E402
from hgt_gen1.model import HGT  # noqa: E402
from retrieval_common import build_clause_index  # noqa: E402

LAW_TYPES = ["법률", "시행령", "시행규칙", "규정", "기타"]


def law_type_of(law_name):
    for i, suffix in enumerate(LAW_TYPES[1:-1], start=1):
        if law_name.endswith(suffix):
            return i
    return 3 if law_name.endswith("규정") or "규정" in law_name else 0


def build_features(nodes_df, clause_list, clause_embs, device):
    """노드별 (텍스트, law_id, law_type, entity_type, 스칼라 2종) 텐서."""
    first = nodes_df.drop_duplicates("new_johang").set_index("new_johang")
    laws, ents, xrefs, pos = [], [], [], []
    for c in clause_list:
        row = first.loc[c] if c in first.index else None
        law = str(row["law_nm"]).strip() if row is not None else ""
        laws.append(law)
        ents.append(str(row["entity_type"]) if row is not None and pd.notna(row.get("entity_type")) else "<UNK>")
        raw = row.get("cross_law_refs") if row is not None else None
        xrefs.append(float(str(raw).count("「")) if isinstance(raw, str) else 0.0)
        si = row.get("structural_index") if row is not None else None
        pos.append(float(si) if pd.notna(si) and str(si).replace(".", "", 1).isdigit() else 0.0)

    law_vocab = {v: i for i, v in enumerate(sorted(set(laws)))}
    ent_vocab = {v: i for i, v in enumerate(sorted(set(ents)))}
    xr = torch.tensor(xrefs)
    pr = torch.tensor(pos)
    scalars = torch.stack([
        xr / (xr.max() + 1e-8),
        pr / (pr.max() + 1e-8),
    ], dim=-1)
    feats = {
        "text": clause_embs.to(device),
        "law_id": torch.tensor([law_vocab[x] for x in laws], device=device),
        "law_type": torch.tensor([law_type_of(x) for x in laws], device=device),
        "entity_type": torch.tensor([ent_vocab[x] for x in ents], device=device),
        "scalars": scalars.to(device),
    }
    return feats, len(law_vocab), len(ent_vocab)


def sample_negatives(n_nodes, n_neg, edge_set, device, generator):
    """엣지가 아닌 랜덤 노드쌍. 충돌은 한 번만 재추출한다(잔여 충돌은 무시할 수준)."""
    src = torch.randint(0, n_nodes, (n_neg,), device=device, generator=generator)
    dst = torch.randint(0, n_nodes, (n_neg,), device=device, generator=generator)
    bad = src == dst
    if bad.any():
        dst[bad] = (dst[bad] + 1) % n_nodes
    keys = (torch.minimum(src, dst) * n_nodes + torch.maximum(src, dst)).tolist()
    clash = torch.tensor([k in edge_set for k in keys], device=device)
    if clash.any():
        dst[clash] = torch.randint(0, n_nodes, (int(clash.sum()),), device=device,
                                   generator=generator)
    return src, dst


def mse_on(emb, src, dst, target):
    if src.numel() == 0:
        return emb.new_zeros(())
    return F.mse_loss((emb[src] * emb[dst]).sum(-1), target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default="hgt_gen1/graph/edges.csv")
    ap.add_argument("--nodes", default="data/nodes.csv")
    ap.add_argument("--out_dir", default="hgt_gen1/checkpoints")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--neg_ratio", type=int, default=3)
    ap.add_argument("--lambda_cross", type=float, default=1.0)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(args.seed)
    print(f"사용 기기: {device}")

    nodes_df = pd.read_csv(os.path.join(ROOT, args.nodes))
    nodes_df["new_johang"] = [
        normalize_johang_key(l, a, h)
        for l, a, h in zip(nodes_df["law_nm"], nodes_df["article_number"], nodes_df["hang_number"])
    ]
    clause_list, clause_texts = build_clause_index(nodes_df)
    encoder = make_bge_encoder()
    clause_embs = encode_texts_cached(encoder, clause_texts, "clause_embs")
    del encoder
    torch.cuda.empty_cache()

    feats, n_laws, n_ents = build_features(nodes_df, clause_list, clause_embs, device)
    N = len(clause_list)

    edges = pd.read_csv(os.path.join(ROOT, args.edges))
    src = torch.tensor(edges["src_idx"].values, device=device)
    dst = torch.tensor(edges["dst_idx"].values, device=device)
    scope = torch.tensor(edges["scope_id"].values, device=device)
    target = torch.tensor(edges["target"].values, dtype=torch.float32, device=device)
    edge_set = set((torch.minimum(src, dst) * N + torch.maximum(src, dst)).tolist())
    print(f"노드 {N:,} | 엣지 {len(edges):,} "
          f"(INTRA {int((scope == 0).sum())}, SIBLING {int((scope == 1).sum())}, "
          f"CROSS {int((scope == 2).sum())})")

    # 메시지 패싱은 무방향 — 양방향 복제
    mp_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    mp_scope = torch.cat([scope, scope])

    perm = torch.randperm(len(edges), generator=gen, device=device)
    n_val = int(len(edges) * args.val_frac)
    val_e, train_e = perm[:n_val], perm[n_val:]
    print(f"엣지 분할: train {len(train_e):,} / val {len(val_e):,}")

    model = HGT(hidden_dim=args.hidden, n_layers=args.layers, n_heads=args.heads,
                n_node_types=len(LAW_TYPES) + 1, n_law_ids=n_laws + 1,
                n_entity_types=n_ents + 1, dropout=args.dropout).to(device)
    print(f"학습 파라미터: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    best_val, bad_epochs = float("inf"), 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        emb = model(feats, mp_index, mp_scope)

        s, d, sc, tg = src[train_e], dst[train_e], scope[train_e], target[train_e]
        n_neg = len(train_e) * args.neg_ratio
        ns, nd = sample_negatives(N, n_neg, edge_set, device, gen)
        zeros = torch.zeros(n_neg, device=device)

        intra = sc == 0
        loss_intra = mse_on(emb, s[intra], d[intra], tg[intra])
        loss_cross = mse_on(emb, s[~intra], d[~intra], tg[~intra])
        loss_neg = mse_on(emb, ns, nd, zeros)
        loss = loss_intra + args.lambda_cross * loss_cross + loss_neg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            emb_v = model(feats, mp_index, mp_scope)
            vs, vd, vt = src[val_e], dst[val_e], target[val_e]
            val = float(mse_on(emb_v, vs, vd, vt))
            # 구조 학습 게이트: 이웃 코사인이 무작위쌍 코사인보다 높아야 한다
            rs, rd = sample_negatives(N, len(val_e), edge_set, device, gen)
            pos_cos = float((emb_v[vs] * emb_v[vd]).sum(-1).mean())
            rnd_cos = float((emb_v[rs] * emb_v[rd]).sum(-1).mean())

        mark = ""
        if val < best_val:
            best_val, bad_epochs, mark = val, 0, "  <- best 저장"
            save_file({"node_embeddings": emb_v.detach().cpu().contiguous()},
                      os.path.join(out_dir, "node_embeddings_best.safetensors"))
            save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
                      os.path.join(out_dir, "hgt_best.safetensors"))
        else:
            bad_epochs += 1

        if epoch % 5 == 0 or mark or epoch == 1:
            print(f"epoch {epoch:3d} | loss={float(loss):.5f} "
                  f"(intra {float(loss_intra):.5f} cross {float(loss_cross):.5f} "
                  f"neg {float(loss_neg):.5f}) | val_mse={val:.5f} | "
                  f"cos 이웃 {pos_cos:+.3f} vs 무작위 {rnd_cos:+.3f}{mark}")
        if bad_epochs >= args.patience:
            print(f"early stopping (patience {args.patience})")
            break

    with open(os.path.join(out_dir, "stage1_config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "n_nodes": N, "n_edges": len(edges),
                   "n_law_ids": n_laws, "n_entity_types": n_ents,
                   "best_val_mse": best_val}, f, ensure_ascii=False, indent=2)
    print(f"\n최적 val MSE {best_val:.5f} -> {out_dir}/node_embeddings_best.safetensors")


if __name__ == "__main__":
    main()
