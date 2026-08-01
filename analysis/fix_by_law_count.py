"""
fix_by_law_count.py

answer_metrics.json의 by_law_count를 정정한다.

eval_answers.py는 관련법 버킷을 나눌 때 문항의 '참조 법률 개수'가 아니라
gold_positives에 실제로 남아 있는 서로 다른 법령 수를 세고 있다. gold_positives는
KG에 존재하는 조항만 남기므로, 인용된 법령 중 일부가 KG 밖이면 그 법령이 통째로
사라지고 문항이 낮은 버킷으로 밀린다. 결과적으로 3-4법 버킷이 실제보다 훨씬 작게
집계된다(관측된 사례: n=11, 실제 47).

정답은 answer_details.jsonl의 num_laws 필드다. 이 값은 심사자가 정리한
for_review_corrected.xlsx의 '# of laws_clean'과 같아야 하며, 이 스크립트가 그것을
검증한 뒤 by_law_count를 다시 집계한다.

전역 지표(RAGAS/ARES/REFERENCE/answer_cite_recall)는 버킷과 무관하므로 건드리지 않고,
details로부터 재계산해 원본과 일치하는지만 확인한다.

사용법:
    python analysis/fix_by_law_count.py \
        --details answer_details.jsonl --metrics answer_metrics.json \
        --out answer_metrics_f.json
"""

import argparse
import json
import re

import numpy as np
import pandas as pd

GLOBAL_METRICS = [
    'faithfulness', 'answer_relevancy', 'context_recall', 'context_precision',
    'context_relevance', 'answer_faithfulness', 'answer_relevance',
    'answer_correctness', 'answer_completeness',
]


def bucket(n):
    return '1-2' if n <= 2 else ('3-4' if n <= 4 else '5+')


def norm(s):
    return re.sub(r'\s+', '', str(s))


def verify_num_laws(rows, fsc_xlsx, sheet):
    """details의 num_laws가 심사자 원본 '# of laws_clean'과 같은지 확인."""
    df = pd.read_excel(fsc_xlsx, sheet_name=sheet)
    lut = {}
    for _, r in df.iterrows():
        lut.setdefault(norm(r['jilui']), int(r['# of laws_clean']))
    missing = [r['idx'] for r in rows if norm(r['query']) not in lut]
    bad = [(r['idx'], r['num_laws'], lut[norm(r['query'])])
           for r in rows if norm(r['query']) in lut and r['num_laws'] != lut[norm(r['query'])]]
    print(f'원본 대조: 질의 매칭 {len(rows) - len(missing)}/{len(rows)}, num_laws 불일치 {len(bad)}건')
    for idx, got, want in bad[:10]:
        print(f'  idx={idx} details={got} xlsx={want}')
    return len(bad) == 0 and not missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--details', required=True)
    ap.add_argument('--metrics', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fsc_xlsx', default='data/for_review_corrected.xlsx')
    ap.add_argument('--fsc_sheet', default='법령O+조항O')
    args = ap.parse_args()

    with open(args.details, encoding='utf-8') as f:
        rows = sorted((json.loads(line) for line in f), key=lambda r: r['idx'])
    with open(args.metrics, encoding='utf-8') as f:
        m = json.load(f)

    if len(rows) != m['n']:
        raise ValueError(f'문항 수 불일치: details {len(rows)} vs metrics {m["n"]}')

    verified = verify_num_laws(rows, args.fsc_xlsx, args.fsc_sheet)

    # 전역 지표는 정정 대상이 아니다 — 재계산해 원본과 같은지만 확인한다.
    print('\n전역 지표 재현 확인 (정정 대상 아님)')
    for k in GLOBAL_METRICS:
        got = float(np.nanmean(np.array([r['scores'].get(k) for r in rows], dtype=float)))
        want = m['metric_stats'][k]['mean']
        flag = 'OK' if abs(got - want) < 5e-5 else '불일치!'
        print(f'  {k:22s} {got:.4f} vs {want:.4f}  {flag}')

    old = m.get('by_law_count', {})
    new = {}
    for b in ('1-2', '3-4', '5+'):
        sel = [r for r in rows if bucket(r['num_laws']) == b]
        if not sel:
            continue
        cor = np.array([r['scores'].get('answer_correctness') for r in sel], dtype=float)
        com = np.array([r['scores'].get('answer_completeness') for r in sel], dtype=float)
        cite = np.array([r['answer_cite_recall'] for r in sel], dtype=float)
        new[b] = {
            'n': len(sel),
            'REFERENCE': {
                'answer_correctness': round(float(np.nanmean(cor)), 4),
                'answer_completeness': round(float(np.nanmean(com)), 4),
            },
            'answer_cite_recall': round(float(np.nanmean(cite)), 4),
        }

    print('\nby_law_count 정정')
    for b in ('1-2', '3-4', '5+'):
        o, n_ = old.get(b), new.get(b)
        if not n_:
            continue
        on = o['n'] if o else 0
        print(f'  {b:4s} n {on:3d} -> {n_["n"]:3d}   '
              f'correctness {o["REFERENCE"]["answer_correctness"] if o else float("nan"):.4f} -> '
              f'{n_["REFERENCE"]["answer_correctness"]:.4f}   '
              f'completeness {o["REFERENCE"]["answer_completeness"] if o else float("nan"):.4f} -> '
              f'{n_["REFERENCE"]["answer_completeness"]:.4f}   '
              f'cite {o["answer_cite_recall"] if o else float("nan"):.4f} -> {n_["answer_cite_recall"]:.4f}')

    m['by_law_count'] = new
    m['by_law_count_source'] = {
        'field': 'answer_details.num_laws',
        'verified_against': f'{args.fsc_xlsx} > {args.fsc_sheet} > # of laws_clean',
        'verified': bool(verified),
        'note': 'eval_answers.py는 gold_positives의 서로 다른 법령 수로 버킷을 나눠 '
                'KG 밖 법령이 누락되면 버킷이 낮아진다. 이 파일은 num_laws로 다시 집계한 정정본이다.',
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(f'\n-> {args.out}')


if __name__ == '__main__':
    main()
