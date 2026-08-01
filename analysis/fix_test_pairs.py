"""
fix_test_pairs.py

insight-agent의 test_pairs.jsonl을 기존 301문항 런의 정답셋에 맞춰 되돌린다.

현재 파일은 positives가 'KG 안 조항만 남기고 조→항으로 확장된' 형태이고, laws가 그
필터링된 positives에서 다시 유도돼 있다. 이는 retrieval_eval.py:55-61이 명시한 규칙과
반대다 — laws는 질의가 참조하는 법령 전체여야 하며 positives의 접두사에서 유도하면
안 된다(positives는 KG 안 조항만 담기 때문). 그 결과 stratified_sample이 쓰는
category_from_laws 버킷이 289/11/1로 무너져 있다(기존 런은 250/46/5).

기존 런의 answer_details jsonl은 eval_answers.py:342-344에서 test_pairs의
num_laws/laws/positives를 그대로 복사해 기록하므로, 그 파일이 곧 원래 test_pairs의
사본이다. 이 스크립트는 그것을 기준으로 네 필드를 되돌린다.

사용법:
    python analysis/fix_test_pairs.py \
        --test_pairs test_pairs.jsonl --ref <기존_런>-answer_details_f.jsonl \
        --out test_pairs_fixed.jsonl
"""

import argparse
import collections
import json
import re


def norm(s):
    return re.sub(r'\s+', '', str(s))


def bucket_of(laws_field):
    """eval_answers.py가 층화에 쓰는 category_from_laws와 같은 규칙."""
    n = len({x.strip() for x in str(laws_field or '').split('|') if x.strip()})
    return '1-2' if n <= 2 else ('3-4' if n <= 4 else '5+')


def load(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def counts(rows, field):
    return dict(sorted(collections.Counter(bucket_of(r[field]) for r in rows).items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_pairs', required=True)
    ap.add_argument('--ref', required=True, help='기존 런의 answer_details jsonl')
    ap.add_argument('--out', required=True)
    ap.add_argument('--keep_positives', action='store_true',
                    help='positives는 현 파일 값을 유지하고 laws/num_laws만 되돌린다')
    args = ap.parse_args()

    tp, ref = load(args.test_pairs), load(args.ref)
    if len(tp) != len(ref):
        raise ValueError(f'행 수 불일치: test_pairs {len(tp)} vs ref {len(ref)}')

    tp_by_q = {}
    for r in tp:
        tp_by_q.setdefault(norm(r['query']), []).append(r)
    missing = [r['idx'] for r in ref if norm(r['query']) not in tp_by_q]
    if missing:
        raise ValueError(f'test_pairs에 없는 참조 질의 {len(missing)}건: {missing[:5]}')

    print(f'입력 {len(tp)}행 / 참조 {len(ref)}행')
    print(f'  현재 laws 버킷    : {counts(tp, "laws")}')
    print(f'  참조 laws 버킷    : {counts(ref, "laws")}')
    d_laws = sum(1 for r in ref if tp_by_q[norm(r['query'])][0]['laws'] != r['laws'])
    d_nl = sum(1 for r in ref if tp_by_q[norm(r['query'])][0]['num_laws'] != r['num_laws'])
    d_pos = sum(1 for r in ref
                if tp_by_q[norm(r['query'])][0]['positives'] != r['gold_positives'])
    print(f'  차이: laws {d_laws}건 / num_laws {d_nl}건 / positives {d_pos}건')

    out = []
    for r in ref:
        src = tp_by_q[norm(r['query'])][0]
        out.append({
            'query': src['query'],                 # 원문 표기는 현 파일 것을 유지
            'positives': src['positives'] if args.keep_positives else r['gold_positives'],
            'laws': r['laws'],
            'num_laws': r['num_laws'],
        })

    def n_laws(field):
        return len({x.strip() for x in str(field or '').split('|') if x.strip()})

    bad = [i for i, r in enumerate(out) if n_laws(r['laws']) != r['num_laws']]
    print(f'  정정본 내적 정합(서로 다른 laws 수 == num_laws) 위반: {len(bad)}건')
    print(f'  정정본 laws 버킷  : {counts(out, "laws")}')

    with open(args.out, 'w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
