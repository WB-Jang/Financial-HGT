"""
lora/paired_retrieval_analysis.py

MLP(QueryEncoder) vs LoRA 검색 성능의 문항 페어링 검정.

analysis/paired_answer_analysis.py와 같은 방법론을 검색 지표에 적용한 것이다.
두 arm이 '같은 test 301문항'에 대해 평가되었으므로, 구성별 평균만 비교하는 대신
문항 단위로 짝지어 차이의 분포를 보면 분산이 크게 줄어 실제 효과가 드러난다.

주 지표는 '항(paragraph) 단위'다. 조(article) 단위는 --level article로 서브 확인.
두 세밀도에서 순위가 뒤집히는 경우가 실제로 있으므로(README §1 각주) 둘을 섞지 않는다.

입력: evaluate_rerank.py가 남기는 질의별 상세 CSV
      eval_results/rerank_{...}_paragraph_{ts}.csv  (또는 _article_)
      lora/results/lora_paragraph_{ts}.csv

실행:
  python lora/paired_retrieval_analysis.py \
      --mlp  eval_results/rerank_origEmb_stage2_hybrid_none_paragraph_*.csv \
      --lora eval_results/rerank_origEmb_lora_hybrid_none_paragraph_*.csv \
      --baseline eval_results/rerank_origEmb_bgeq_hybrid_none_paragraph_*.csv
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

N_BOOT = 20000
METRICS = ['recall@15', 'hit@15', 'recall@1', 'hit@1', 'recall@30', 'mrr@30']


def load_arm(pattern, name):
    """glob 패턴으로 CSV 하나를 읽는다. 여러 개면 가장 최근 것(파일명 timestamp 순)."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f'{name}: 일치하는 파일 없음 -> {pattern}')
    if len(paths) > 1:
        print(f'  [{name}] {len(paths)}개 일치, 최신 사용: {os.path.basename(paths[-1])}')
    df = pd.read_csv(paths[-1])
    print(f'  [{name}] {os.path.basename(paths[-1])} ({len(df)}행)')
    return df


def align(arms):
    """arm들을 query 기준으로 정렬·검증. 문항 집합이 다르면 페어링 불가이므로 중단."""
    ref = arms[0][1]
    base_q = list(ref['query'])
    for name, df in arms[1:]:
        if list(df['query']) != base_q:
            raise SystemExit(
                f'❌ {name}의 문항 순서/구성이 기준과 다릅니다 — 페어링 불가.\n'
                f'   모든 arm을 같은 --test_size로 실행했는지 확인하세요.'
            )
    return np.asarray(ref['num_laws'].fillna(0), dtype=int)


def paired_test(x, y, mask, rng):
    """x-y의 문항 페어링 검정 (paired_answer_analysis.paired_test와 동일 정의)."""
    m = mask & ~np.isnan(x) & ~np.isnan(y)
    d = (x - y)[m]
    n = len(d)
    if n == 0:
        return {'n': 0, 'delta': float('nan'), 'ci_low': float('nan'),
                'ci_high': float('nan'), 'p_wilcoxon': float('nan'),
                'win': 0, 'loss': 0, 'tie': 0}
    boot = d[rng.integers(0, n, (N_BOOT, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    try:
        p = stats.wilcoxon(d, zero_method='wilcox').pvalue
    except ValueError:      # 전부 동점이면 검정 불가
        p = float('nan')
    return {'n': n, 'delta': d.mean(), 'ci_low': lo, 'ci_high': hi, 'p_wilcoxon': p,
            'win': int((d > 0).sum()), 'loss': int((d < 0).sum()), 'tie': int((d == 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mlp', required=True, help='MLP arm 상세 CSV (glob 가능)')
    ap.add_argument('--lora', required=True, help='LoRA arm 상세 CSV (glob 가능)')
    ap.add_argument('--baseline', default=None, help='순수 BGE arm 상세 CSV (선택)')
    ap.add_argument('--level', choices=['paragraph', 'article'], default='paragraph',
                    help='보고용 라벨. 실제 세밀도는 입력 CSV가 결정한다')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  'results'))
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print(f'=== MLP vs LoRA 검색 성능 대응표본 검정 ({args.level} 단위) ===')
    arms = [('MLP', load_arm(args.mlp, 'MLP')), ('LoRA', load_arm(args.lora, 'LoRA'))]
    if args.baseline:
        arms.append(('baseline', load_arm(args.baseline, 'baseline')))

    num_laws = align(arms)
    by = dict(arms)
    buckets = [('all', np.ones(len(num_laws), dtype=bool)),
               ('1-2', (num_laws >= 1) & (num_laws <= 2)),
               ('3-4', num_laws >= 3)]
    rng = np.random.default_rng(args.seed)

    rows = []
    for metric in METRICS:
        if metric not in by['MLP'].columns:
            print(f'  (건너뜀: {metric} 컬럼 없음)')
            continue
        lora = np.asarray(by['LoRA'][metric], dtype=float)
        mlp = np.asarray(by['MLP'][metric], dtype=float)
        for bname, bmask in buckets:
            r = paired_test(lora, mlp, bmask, rng)
            rows.append(['LoRA-MLP', metric, bname, r['n'], r['delta'], r['ci_low'],
                         r['ci_high'], r['p_wilcoxon'], r['win'], r['loss'], r['tie']])
        if 'baseline' in by:
            base = np.asarray(by['baseline'][metric], dtype=float)
            for tag, arr in (('LoRA-base', lora), ('MLP-base', mlp)):
                r = paired_test(arr, base, buckets[0][1], rng)
                rows.append([tag, metric, 'all', r['n'], r['delta'], r['ci_low'],
                             r['ci_high'], r['p_wilcoxon'], r['win'], r['loss'], r['tie']])

    out_df = pd.DataFrame(rows, columns=['comparison', 'metric', 'bucket', 'n', 'delta',
                                         'ci_low', 'ci_high', 'p_wilcoxon',
                                         'win', 'loss', 'tie'])
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f'paired_retrieval_{args.level}.csv')
    out_df.to_csv(path, index=False, float_format='%.4f')

    pd.set_option('display.width', 200)
    print(f'\n{out_df.to_string(index=False, float_format=lambda v: f"{v:.4f}")}')
    print(f'\n✅ 저장: {path}')

    main_row = out_df[(out_df.comparison == 'LoRA-MLP') &
                      (out_df.metric == 'recall@15') & (out_df.bucket == 'all')]
    if not main_row.empty:
        r = main_row.iloc[0]
        verdict = ('LoRA 우세' if r.ci_low > 0 else
                   'MLP 우세' if r.ci_high < 0 else '차이 불명확 (CI가 0을 포함)')
        print(f'\n[주 판정] {args.level} 단위 recall@15, LoRA-MLP = {r.delta:+.4f} '
              f'(95% CI [{r.ci_low:+.4f}, {r.ci_high:+.4f}], p={r.p_wilcoxon:.4f}) -> {verdict}')


if __name__ == '__main__':
    main()
