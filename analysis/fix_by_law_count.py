"""
fix_by_law_count.py

answer_metrics.json의 by_law_count를 정정한다.

eval_answers.py는 관련법 버킷을 나눌 때 문항의 '참조 법률 개수'가 아니라
gold_positives에 실제로 남아 있는 서로 다른 법령 수를 세고 있다. gold_positives는
KG에 존재하는 조항만 남기므로, 인용된 법령 중 일부가 KG 밖이면 그 법령이 통째로
사라지고 문항이 낮은 버킷으로 밀린다. 결과적으로 3-4법 버킷이 실제보다 훨씬 작게
집계된다(관측된 사례: n=11, 실제 47).

버킷은 answer_details.jsonl의 num_laws 필드로 다시 집계한다.

num_laws의 기준 축은 --num_laws_ref로 지정한다. 기존 301문항 런들(17구성)이 공유하는
층화가 이미 마스터 파일 ⑤ 시트와 대응표본 분석 전체의 기준이므로, 신규 런을 그 표에
얹으려면 같은 축을 써야 한다. 지정하지 않으면 details 자신의 num_laws를 쓴다.

for_review_corrected.xlsx의 '# of laws_clean'과의 차이는 참고용으로 보고만 한다.
기준 축 선택은 기존 결과와의 비교 가능성 문제이지 이 스크립트가 판정할 문제가 아니다.

전역 지표(RAGAS/ARES/REFERENCE/answer_cite_recall)는 버킷과 무관하므로 건드리지 않고,
details로부터 재계산해 원본과 일치하는지만 확인한다.

사용법:
    python analysis/fix_by_law_count.py \
        --details answer_details.jsonl --metrics answer_metrics.json \
        --num_laws_ref <기존_런>-answer_details_f.jsonl \
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


def load_ref_num_laws(path):
    """기준 축이 될 기존 런의 num_laws를 idx -> num_laws로 읽는다."""
    with open(path, encoding='utf-8') as f:
        return {r['idx']: r['num_laws'] for r in (json.loads(line) for line in f)}


def compare_with_source(rows, num_laws, fsc_xlsx, sheet):
    """채택한 num_laws가 심사자 원본 '# of laws_clean'과 얼마나 다른지 참고 보고."""
    df = pd.read_excel(fsc_xlsx, sheet_name=sheet)
    lut = {}
    for _, r in df.iterrows():
        lut.setdefault(norm(r['jilui']), int(r['# of laws_clean']))
    diff = [(r['idx'], num_laws[r['idx']], lut[norm(r['query'])])
            for r in rows if norm(r['query']) in lut and num_laws[r['idx']] != lut[norm(r['query'])]]
    print(f'참고: 채택한 num_laws와 {sheet}의 # of laws_clean 차이 {len(diff)}건')
    for idx, got, want in diff[:10]:
        print(f'  idx={idx} 채택={got} xlsx={want}')
    return len(diff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--details', required=True)
    ap.add_argument('--metrics', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--num_laws_ref', default=None,
                    help='num_laws 기준 축으로 삼을 기존 런의 answer_details jsonl')
    ap.add_argument('--fsc_xlsx', default='data/for_review_corrected.xlsx')
    ap.add_argument('--fsc_sheet', default='법령O+조항O')
    args = ap.parse_args()

    with open(args.details, encoding='utf-8') as f:
        rows = sorted((json.loads(line) for line in f), key=lambda r: r['idx'])
    with open(args.metrics, encoding='utf-8') as f:
        m = json.load(f)

    if len(rows) != m['n']:
        raise ValueError(f'문항 수 불일치: details {len(rows)} vs metrics {m["n"]}')

    if args.num_laws_ref:
        ref = load_ref_num_laws(args.num_laws_ref)
        missing = [r['idx'] for r in rows if r['idx'] not in ref]
        if missing:
            raise ValueError(f'기준 파일에 없는 idx {len(missing)}건: {missing[:5]}')
        num_laws = {r['idx']: ref[r['idx']] for r in rows}
        moved = sum(1 for r in rows if bucket(r['num_laws']) != bucket(num_laws[r['idx']]))
        print(f'num_laws 기준 축: {args.num_laws_ref}')
        print(f'  자체 num_laws 대비 값이 다른 문항 '
              f'{sum(1 for r in rows if r["num_laws"] != num_laws[r["idx"]])}건, '
              f'그중 버킷이 바뀌는 문항 {moved}건')
    else:
        num_laws = {r['idx']: r['num_laws'] for r in rows}
        print('num_laws 기준 축: details 자체 필드')
    n_diff = compare_with_source(rows, num_laws, args.fsc_xlsx, args.fsc_sheet)

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
        sel = [r for r in rows if bucket(num_laws[r['idx']]) == b]
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
        'num_laws_ref': args.num_laws_ref or 'self',
        'diff_vs_fsc_clean': n_diff,
        'note': 'eval_answers.py는 gold_positives의 서로 다른 법령 수로 버킷을 나눠 '
                'KG 밖 법령이 누락되면 버킷이 낮아진다. 이 파일은 num_laws로 다시 집계한 정정본이며, '
                '기존 301문항 런들과 같은 층화를 쓰도록 num_laws_ref의 축을 따랐다.',
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(f'\n-> {args.out}')


if __name__ == '__main__':
    main()
