"""
paired_answer_analysis.py

답변 품질(answer_details_f.jsonl)에 대한 대응표본 통계 검정.

evaluate_rerank.py가 측정하는 것은 '검색이 정답 조항을 찾는가'(recall/hit)이고,
이 스크립트가 측정하는 것은 '그 검색 결과로 만든 답변이 실제로 좋은가'이다.
모든 구성이 동일한 301문항에 답했으므로 독립표본 비교가 아니라 문항 페어링 검정을 쓴다.

핵심 비교 4가지:
  1. breadth - plain          : 폭 우선 조립 성분의 순효과 (같은 검색기끼리 짝지어서)
  2. 교호작용                  : 그 순효과가 다법 질의(3-4법)에서 단일/2법보다 큰가
  3. alpha 0.80 - alpha 0.25  : 관련도-폭 블렌드 비중의 최적점
  4. hybrid alpha 용량-반응    : alpha 0/.25/.50/.60/.80/.90 6점 곡선 (hybrid 계열만 전 구간 보유)

지표: REFERENCE_avg = (answer_correctness + answer_completeness) / 2
      (심판 모델이 golden 정답과 비교해 매긴 점수. answer_metrics_f.json과 동일 정의)

사용법:
    python analysis/paired_answer_analysis.py --details_dir eval_results/answer_details
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

# 구성 이름 -> answer_details jsonl 파일명 접두사
# (파일명이 해시로 시작하므로 접두사로 매칭. 접미사는 런마다 달라 <접두사>-*.jsonl로 glob한다)
CONFIGS = {
    'bge_m3': '0b779d79',          # 베이스라인: 학습 없는 순수 BGE-M3 top15
    # plain = 폭 우선 조립 없음 (Stage2 QueryEncoder + 각 재랭킹)
    'pl_hybrid': 'aa160411',
    'pl_cross': 'a05e1734',
    'pl_ppr': '7e3bca9e',
    'pl_dense': 'af47c00b',
    # breadth alpha=0.80
    'br80_hybrid': '94396eb6',
    'br80_cross': '2f96729a',
    'br80_ppr': 'e303e8c0',
    'br80_dense': 'b40930fd',
    # breadth alpha=0.25
    'br25_hybrid': '66f22c36',
    'br25_cross': 'fbdd0d8c',
    'br25_ppr': '12b74700',
    'br25_dense': '65f60dd3',
    # breadth alpha=0.50 / 0.60 / 0.90 — hybrid 계열만 추가 실행 (용량-반응 곡선용)
    'br50_hybrid': '9f43c101',
    'br60_hybrid': 'bce946aa',
    'br90_hybrid': 'c6132a56',
}

FAMILIES = ['hybrid', 'cross', 'ppr', 'dense']

# hybrid 계열 alpha 용량-반응. plain = alpha 0 (blend가 관련도 그대로이므로 수학적으로 동일)
ALPHA_SWEEP = [('pl_hybrid', 0.00), ('br25_hybrid', 0.25), ('br50_hybrid', 0.50),
               ('br60_hybrid', 0.60), ('br80_hybrid', 0.80), ('br90_hybrid', 0.90)]

N_BOOT = 20000


def load_details(details_dir):
    """구성별 jsonl을 idx 순으로 정렬해 로드. 모든 구성의 문항 정렬이 같은지 검증한다."""
    data = {}
    for name, prefix in CONFIGS.items():
        hits = sorted(glob.glob(os.path.join(details_dir, f'{prefix}-*.jsonl')))
        if len(hits) != 1:
            raise FileNotFoundError(f'{name}: {details_dir}/{prefix}-*.jsonl 이 {len(hits)}개 매칭 (1개여야 함)')
        path = hits[0]
        with open(path, encoding='utf-8') as f:
            rows = [json.loads(line) for line in f]
        data[name] = sorted(rows, key=lambda r: r['idx'])

    ref_idx = [r['idx'] for r in data['bge_m3']]
    ref_nl = [r['num_laws'] for r in data['bge_m3']]
    for name, rows in data.items():
        if [r['idx'] for r in rows] != ref_idx:
            raise ValueError(f'{name}: 문항 idx가 베이스라인과 불일치 — 페어링 불가')
        if [r['num_laws'] for r in rows] != ref_nl:
            raise ValueError(f'{name}: num_laws가 베이스라인과 불일치')
    return data, np.array(ref_nl)


def reference_avg(rows):
    """(answer_correctness + answer_completeness) / 2. 둘 중 하나라도 None이면 NaN."""
    out = []
    for r in rows:
        s = r['scores']
        a, b = s.get('answer_correctness'), s.get('answer_completeness')
        out.append(np.nan if (a is None or b is None) else (a + b) / 2)
    return np.array(out, dtype=float)


def paired_test(x, y, mask, rng):
    """x-y의 문항 페어링 검정. 어느 한쪽이 결측인 문항은 제외(쌍별 제외).

    p를 두 개 낸다. 둘은 서로 다른 추정량에 대한 검정이므로 결론이 갈릴 수 있다:
      p_wilcoxon — 부호+크기 '순위'. 동점을 버리므로 유효표본이 n보다 훨씬 작다.
                   크기가 들쭉날쭉해도 순위로 압축되어 이상치에 강하다.
      p_boot     — '평균'. ci_low/ci_high와 같은 추정량이라 CI와 항상 정합적이다.
                   크기를 그대로 쓰므로 이상치가 분산을 키워 보수적으로 나오는 경향.
    """
    m = mask & ~np.isnan(x) & ~np.isnan(y)
    d = (x - y)[m]
    n = len(d)
    delta = d.mean()
    boot = d[rng.integers(0, n, (N_BOOT, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # 귀무(평균 0) 중심 재표집. (d - delta)[idx].mean() == d[idx].mean() - delta 이므로
    # 같은 재표집을 평행이동하면 된다 — 난수를 추가로 소비하지 않아 기존 CI가 그대로 보존된다.
    # (1+k)/(1+B): 재표집 p의 하한은 1/(B+1)이다. 0을 보고하지 않기 위한 표준 보정.
    p_boot = (1 + int((np.abs(boot - delta) >= abs(delta)).sum())) / (1 + N_BOOT)
    try:
        p = stats.wilcoxon(d, zero_method='wilcox').pvalue
    except ValueError:  # 전부 동점이면 검정 불가
        p = float('nan')
    return {
        'n': n, 'delta': delta, 'ci_low': lo, 'ci_high': hi, 'p_wilcoxon': p, 'p_boot': p_boot,
        'win': int((d > 0).sum()), 'loss': int((d < 0).sum()), 'tie': int((d == 0).sum()),
    }


def interaction_test(d, m12, m34, rng):
    """(3-4법에서의 효과) - (1-2법에서의 효과). 두 버킷은 서로 다른 문항이므로 비페어링 부트스트랩."""
    a = d[m12 & ~np.isnan(d)]
    b = d[m34 & ~np.isnan(d)]
    boot = (b[rng.integers(0, len(b), (N_BOOT, len(b)))].mean(axis=1)
            - a[rng.integers(0, len(a), (N_BOOT, len(a)))].mean(axis=1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {'delta_34': b.mean(), 'delta_12': a.mean(), 'interaction': b.mean() - a.mean(),
            'ci_low': lo, 'ci_high': hi, 'p_boot': p}


def holm(pvals):
    """Holm-Bonferroni 보정. 입력 순서대로 보정된 p를 돌려준다.

    검정 불가(NaN — paired_test가 전부 동점일 때 반환)는 가족에서 빼고 유효 검정 수로만
    보정한다. NaN을 그대로 넣으면 argsort가 뒤로 보내고 max(running, nan)이 nan을 무시해
    (nan > running 이 False), 실패한 검정이 다른 검정의 보정 p를 물려받아 정상값처럼
    기록된다. 게다가 승수 m이 NaN까지 세어 나머지가 과보정된다.
    """
    pvals = np.asarray(pvals, dtype=float)
    adj = np.full(len(pvals), np.nan)
    valid = np.flatnonzero(~np.isnan(pvals))
    running = 0.0
    for rank, i in enumerate(valid[np.argsort(pvals[valid])]):
        running = max(running, pvals[i] * (len(valid) - rank))
        adj[i] = min(1.0, running)
    return adj


def sign_test(values):
    """양측 부호검정. 0은 시행에서 제외한다. -> (양수 개수, 유효 시행 수, p)

    ⚠️ 시행 독립을 전제하는 검정인데 여기 묶이는 것들은 독립이 아니다. 계열 4종은 같은
    301문항·같은 plain 기준선을 공유하고, alpha 5종은 같은 계열에 강도만 바꿔 건 개입이라
    상관이 거의 1이다. 따라서 이 p는 증거를 과대평가한다 — 방향 일관성의 탐색적 지표로만
    읽고, 유의성 주장의 근거로 쓰지 말 것.
    """
    pos = sum(1 for v in values if v > 0)
    neg = sum(1 for v in values if v < 0)
    m = pos + neg
    if m == 0:
        return pos, m, float('nan')
    return pos, m, min(1.0, 2 * stats.binom.cdf(min(pos, neg), m, 0.5))


def write_csv(path, header, rows):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(','.join(header) + '\n')
        for r in rows:
            f.write(','.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in r) + '\n')
    print(f'  -> {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--details_dir', default='eval_results/answer_details',
                    help='answer_details_f.jsonl 파일들이 있는 디렉터리')
    ap.add_argument('--out_dir', default='analysis/results')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    data, num_laws = load_details(args.details_dir)
    R = {name: reference_avg(rows) for name, rows in data.items()}

    buckets = [('all', num_laws >= 1), ('1-2', num_laws <= 2), ('3-4', (num_laws >= 3) & (num_laws <= 4))]
    print(f'문항 수: {len(num_laws)} (1-2법 {int((num_laws <= 2).sum())}, '
          f'3-4법 {int(((num_laws >= 3) & (num_laws <= 4)).sum())}, '
          f'5+법 {int((num_laws >= 5).sum())})')
    for name in CONFIGS:
        print(f'  {name:12s} mean={np.nanmean(R[name]):.4f}  결측={int(np.isnan(R[name]).sum())}')

    # ── 1. breadth 순효과 (alpha별) ───────────────────────────────────────
    for tag, alpha in [('br80', '0.80'), ('br25', '0.25')]:
        print(f'\n[breadth alpha={alpha}] - plain')
        rows = []
        for fam in FAMILIES:
            for bname, bmask in buckets:
                r = paired_test(R[f'{tag}_{fam}'], R[f'pl_{fam}'], bmask, rng)
                rows.append([fam, bname, r['n'], r['delta'], r['ci_low'], r['ci_high'],
                             r['p_wilcoxon'], r['p_boot'], r['win'], r['loss'], r['tie']])
                print(f'  {fam:7s} {bname:4s} n={r["n"]:3d} delta={r["delta"]:+.4f} '
                      f'CI[{r["ci_low"]:+.4f},{r["ci_high"]:+.4f}] '
                      f'p_w={r["p_wilcoxon"]:.3f} p_b={r["p_boot"]:.3f} '
                      f'{r["win"]}/{r["loss"]}/{r["tie"]}')
        write_csv(os.path.join(args.out_dir, f'net_effect_alpha{alpha.replace(".", "")}.csv'),
                  ['family', 'bucket', 'n', 'delta', 'ci_low', 'ci_high', 'p_wilcoxon', 'p_boot',
                   'win', 'loss', 'tie'], rows)

    # ── 2. 교호작용 (다법에서 효과가 더 큰가) ─────────────────────────────
    m12 = num_laws <= 2
    m34 = (num_laws >= 3) & (num_laws <= 4)
    print('\n[교호작용] delta(3-4) - delta(1-2)')
    inter_rows, pvals = [], {}
    for tag, alpha in [('br80', '0.80'), ('br25', '0.25')]:
        ps = []
        for fam in FAMILIES:
            r = interaction_test(R[f'{tag}_{fam}'] - R[f'pl_{fam}'], m12, m34, rng)
            inter_rows.append([alpha, fam, r['delta_12'], r['delta_34'], r['interaction'],
                               r['ci_low'], r['ci_high'], r['p_boot']])
            ps.append(r['p_boot'])
        pvals[alpha] = holm(np.array(ps))
        for i, fam in enumerate(FAMILIES):
            idx = len(inter_rows) - len(FAMILIES) + i
            inter_rows[idx].append(pvals[alpha][i])
            print(f'  a={alpha} {fam:7s} interaction={inter_rows[idx][4]:+.4f} '
                  f'CI[{inter_rows[idx][5]:+.4f},{inter_rows[idx][6]:+.4f}] '
                  f'p={inter_rows[idx][7]:.3f} holm={pvals[alpha][i]:.3f}')
        pos, m_sign, p_sign = sign_test(
            [inter_rows[len(inter_rows) - len(FAMILIES) + i][4] for i in range(len(FAMILIES))])
        print(f'    부호일치: {pos}/{m_sign} 양수 (0 제외 {len(FAMILIES) - m_sign}) -> '
              f'부호검정 p={p_sign:.4f}  ⚠️ 계열 간 상관 높음 — 탐색적 지표')
    write_csv(os.path.join(args.out_dir, 'interaction.csv'),
              ['alpha', 'family', 'delta_1_2', 'delta_3_4', 'interaction',
               'ci_low', 'ci_high', 'p_boot', 'p_holm'], inter_rows)

    # ── 3. alpha 0.80 vs 0.25 ────────────────────────────────────────────
    print('\n[alpha 비교] 0.80 - 0.25  (양수 = 0.80 우세)')
    rows = []
    for fam in FAMILIES:
        for bname, bmask in buckets:
            r = paired_test(R[f'br80_{fam}'], R[f'br25_{fam}'], bmask, rng)
            rows.append([fam, bname, r['n'], r['delta'], r['ci_low'], r['ci_high'],
                         r['p_wilcoxon'], r['p_boot'], r['win'], r['loss'], r['tie']])
            print(f'  {fam:7s} {bname:4s} n={r["n"]:3d} delta={r["delta"]:+.4f} '
                  f'CI[{r["ci_low"]:+.4f},{r["ci_high"]:+.4f}] '
                  f'p_w={r["p_wilcoxon"]:.3f} p_b={r["p_boot"]:.3f}')
    write_csv(os.path.join(args.out_dir, 'alpha_comparison.csv'),
              ['family', 'bucket', 'n', 'delta', 'ci_low', 'ci_high', 'p_wilcoxon', 'p_boot',
               'win', 'loss', 'tie'], rows)

    # ── 4. 검정력 진단: 동점률이 유효표본을 얼마나 깎는가 ────────────────
    print('\n[동점률 진단] 3-4법 버킷')
    rows = []
    for fam in FAMILIES:
        d = (R[f'br80_{fam}'] - R[f'pl_{fam}'])[m34]
        d = d[~np.isnan(d)]
        tie_rate = (d == 0).mean()
        eff = int((d != 0).sum())
        rows.append([fam, len(d), int((d == 0).sum()), tie_rate, eff])
        print(f'  {fam:7s} n={len(d)} 동점={int((d == 0).sum())} ({tie_rate * 100:.0f}%) 유효표본={eff}')
    write_csv(os.path.join(args.out_dir, 'tie_diagnostics.csv'),
              ['family', 'n', 'ties', 'tie_rate', 'effective_n'], rows)

    # ── 5. hybrid alpha 용량-반응 곡선 ────────────────────────────────────
    # breadth_assembly_retriever는 pool(20건)에서 blend 최댓값 1건만 맨 앞으로 올리고
    # 나머지는 base 관련도순을 유지한다(breadth_assembly_retriever.py:89-92). 다만 컨텍스트는
    # 15건으로 잘리므로(eval_answers.py:336,341), lead가 pool 15~19위에서 나오면 base 14위가
    # 밀려나 '검색 집합 자체'가 바뀐다. 실측 집합변경: a=.25 0건 / .50 10 / .60 34 / .80 49 / .90 54.
    # 그 교체가 전부 비정답끼리였기 때문에(정답 출입 0회) recall@15/hit@15가 전 alpha에서
    # 비트 단위로 같은 것이지, 집합이 불변이어서가 아니다. 이 구분이 중요하다 — 정답이 한 번이라도
    # 드나들면 alpha 차이를 '맥락 순서 효과'로 귀속시킬 수 없다. gold_churn 열이 그 방어선이다.
    print('\n[검색 불변성] alpha가 검색 집합을 바꾸는가')
    inv_rows = []
    pl_lead = [r['retrieved'][0]['node'] for r in data['pl_hybrid']]
    pl_set = [set(x['node'] for x in r['retrieved']) for r in data['pl_hybrid']]
    gold = [set(r['gold_positives']) for r in data['pl_hybrid']]
    for name, alpha in ALPHA_SWEEP:
        got = [set(x['node'] for x in r['retrieved']) for r in data[name]]
        lead = [r['retrieved'][0]['node'] for r in data[name]]
        rec = float(np.mean([len(ret & gld) / len(gld) for ret, gld in zip(got, gold)]))
        hit = float(np.mean([1.0 if ret & gld else 0.0 for ret, gld in zip(got, gold)]))
        promo = sum(1 for a, b in zip(lead, pl_lead) if a != b)
        lead_hit = sum(1 for l, gld in zip(lead, gold) if l in gld)
        changed = sum(1 for ret, base in zip(got, pl_set) if ret != base)
        churn = sum(len((base - ret) & gld) + len((ret - base) & gld)
                    for ret, base, gld in zip(got, pl_set, gold))
        inv_rows.append([alpha, rec, hit, changed, churn, promo, promo / len(lead),
                         lead_hit, lead_hit / len(lead)])
        print(f'  a={alpha:.2f} recall@15={rec:.6f} hit@15={hit:.6f} '
              f'집합변경={changed:3d} 정답출입={churn:2d} '
              f'1위교체={promo:3d}/{len(lead)} 1위정답={lead_hit:3d}({lead_hit / len(lead) * 100:.1f}%)')
    if any(r[4] for r in inv_rows):
        print('  ⚠️ 정답출입 != 0 — 검색 품질이 실제로 달라졌다. alpha 효과를 순서 효과로 해석할 수 없다')
    write_csv(os.path.join(args.out_dir, 'alpha_sweep_retrieval.csv'),
              ['alpha', 'recall@15', 'hit@15', 'set_changed', 'gold_churn',
               'lead_changed', 'lead_changed_rate', 'lead_is_gold', 'lead_is_gold_rate'], inv_rows)

    sweep = [(n, a) for n, a in ALPHA_SWEEP if a > 0]
    print('\n[alpha 용량-반응] alpha - plain (양수 = breadth 우세)')
    rows = []
    for bname, bmask in buckets:
        res = [paired_test(R[n], R['pl_hybrid'], bmask, rng) for n, _ in sweep]
        adj = holm(np.array([r['p_wilcoxon'] for r in res]))
        adj_b = holm(np.array([r['p_boot'] for r in res]))
        for (n, a), r, h, hb in zip(sweep, res, adj, adj_b):
            rows.append([a, bname, r['n'], r['delta'], r['ci_low'], r['ci_high'],
                         r['p_wilcoxon'], h, r['p_boot'], hb, r['win'], r['loss'], r['tie']])
            print(f'  {bname:4s} a={a:.2f} n={r["n"]:3d} delta={r["delta"]:+.4f} '
                  f'CI[{r["ci_low"]:+.4f},{r["ci_high"]:+.4f}] '
                  f'p_w={r["p_wilcoxon"]:.4f}/holm={h:.4f} '
                  f'p_b={r["p_boot"]:.4f}/holm={hb:.4f} {r["win"]}/{r["loss"]}/{r["tie"]}')
    write_csv(os.path.join(args.out_dir, 'alpha_sweep_vs_plain.csv'),
              ['alpha', 'bucket', 'n', 'delta', 'ci_low', 'ci_high',
               'p_wilcoxon', 'p_wilcoxon_holm', 'p_boot', 'p_boot_holm',
               'win', 'loss', 'tie'], rows)

    print('\n[alpha 용량-반응 교호작용] delta(3-4) - delta(1-2)')
    rows = []
    for n, a in sweep:
        r = interaction_test(R[n] - R['pl_hybrid'], m12, m34, rng)
        rows.append([a, r['delta_12'], r['delta_34'], r['interaction'],
                     r['ci_low'], r['ci_high'], r['p_boot']])
        print(f'  a={a:.2f} d(1-2)={r["delta_12"]:+.4f} d(3-4)={r["delta_34"]:+.4f} '
              f'교호={r["interaction"]:+.4f} CI[{r["ci_low"]:+.4f},{r["ci_high"]:+.4f}] p={r["p_boot"]:.4f}')
    pos, m_sign, p_sign = sign_test([r[3] for r in rows])
    print(f'    부호일치: {pos}/{m_sign} 양수 (0 제외 {len(rows) - m_sign}) -> '
          f'부호검정 p={p_sign:.4f}  ⚠️ alpha 간 상관 매우 높음 — 탐색적 지표')
    write_csv(os.path.join(args.out_dir, 'alpha_sweep_interaction.csv'),
              ['alpha', 'delta_1_2', 'delta_3_4', 'interaction', 'ci_low', 'ci_high', 'p_boot'], rows)


if __name__ == '__main__':
    main()
