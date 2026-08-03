"""
evaluate_hgt.py — Gen1식 HGT arm의 test 검색 성능.

항(paragraph) 단위가 주 지표, 조(article) 단위는 서브 지표다. 두 지표 모두
retrieval_common의 공용 함수를 호출하므로 MLP·BGE arm과 계산 경로가 동일하다.

evaluate_rerank.py를 재사용하지 않는 이유: 그쪽은 조항 임베딩이 1024d BGE라고 가정하는데
이 arm은 질의·문서가 모두 256d HGT 공간이다. 재랭킹(dense/hybrid/ppr/cross) 없이
순수 코사인만 측정한다 — Gen1 Stage 2가 학습한 것이 정확히 그것이기 때문이다.

사용법:
    python hgt_gen1/evaluate_hgt.py --test_size 300
"""

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from data_loader import (encode_texts_cached, fsc_dataset_preprocessing,  # noqa: E402
                         make_bge_encoder, normalize_johang_key)
from hgt_gen1.model import QueryEncoder256  # noqa: E402
from retrieval_common import (K_VALUES, apply_num_laws_ref, build_clause_index,  # noqa: E402
                              build_retrieval_items, compute_article_metric_rows,
                              compute_metric_rows, load_num_laws_ref, summarize_by_bucket)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node_emb", default="hgt_gen1/checkpoints/node_embeddings_best.safetensors")
    ap.add_argument("--ckpt", default="hgt_gen1/checkpoints/query_encoder_hgt_best.safetensors")
    ap.add_argument("--out_dir", default="hgt_gen1/results")
    ap.add_argument("--test_size", type=int, default=300)
    ap.add_argument("--num_laws_ref", default=None,
                    help="기존 런의 answer_details jsonl. 지정 시 num_laws를 질의 텍스트로 "
                         "매칭해 덮어써 층화 축을 기존 17런(250/46/5)과 맞춘다")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nodes_df = pd.read_csv(os.path.join(ROOT, "data/nodes.csv"))
    nodes_df["new_johang"] = [
        normalize_johang_key(l, a, h)
        for l, a, h in zip(nodes_df["law_nm"], nodes_df["article_number"], nodes_df["hang_number"])
    ]
    clause_list, _ = build_clause_index(nodes_df)

    clause_embs = load_file(os.path.join(ROOT, args.node_emb))["node_embeddings"]
    clause_embs = F.normalize(clause_embs.float(), dim=-1).to(device)

    fsc = fsc_dataset_preprocessing(file=os.path.join(ROOT, "data/for_review_corrected.xlsx"),
                                    nodes_df=nodes_df, test_size=args.test_size)
    items, _ = build_retrieval_items(fsc[fsc.split == "test"].reset_index(drop=True), clause_list)
    if args.num_laws_ref:
        apply_num_laws_ref(items, load_num_laws_ref(os.path.join(ROOT, args.num_laws_ref)))

    encoder = make_bge_encoder()
    qemb = encode_texts_cached(encoder, [it["query"] for it in items], "fsc_query_embs").to(device)
    del encoder

    model = QueryEncoder256(hidden_dim=clause_embs.size(1)).to(device)
    model.load_state_dict(load_file(os.path.join(ROOT, args.ckpt)))
    model.eval()
    with torch.no_grad():
        sims = model(qemb) @ clause_embs.T
        ranked = sims.topk(max(K_VALUES), dim=1).indices.tolist()

    para_rows, mrr_col = compute_metric_rows(ranked, items)
    art_rows, _ = compute_article_metric_rows(ranked, items, clause_list)

    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {}
    for level, rows in (("paragraph", para_rows), ("article", art_rows)):
        df = pd.DataFrame(rows)
        path = os.path.join(out_dir, f"hgt_gen1_{level}_{stamp}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        by_bucket = summarize_by_bucket(df, mrr_col=mrr_col)
        summary[level] = by_bucket.to_dict(orient="records")
        head = "=== [주 지표] 항(paragraph) 단위 ===" if level == "paragraph" \
            else "=== [서브 지표] 조(article) 단위 ==="
        print(f"\n{head}")
        print(by_bucket.to_string(index=False))
        print(f"  -> {path}")

    with open(os.path.join(out_dir, f"hgt_gen1_summary_{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump({"n_test": len(items), "emb_dim": clause_embs.size(1),
                   "node_emb": args.node_emb, "ckpt": args.ckpt,
                   "num_laws_ref": args.num_laws_ref, "summary": summary}, f,
                  ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
