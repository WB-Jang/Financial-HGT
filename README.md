# Financial-HGT: 금융 법령 조항 검색 (KG 기반 Retrieval)

금융위원회(FSC) 법령해석 질의에 대해, 지식그래프(KG)로 구축된 금융 법령 조항(항 단위 9,311개 노드, 30개 법령)에서
관련 조항을 검색하는 파이프라인. **고정 BGE-M3 임베딩 + 경량 QueryEncoder 학습**을 축으로,
그래프 재랭킹(PPR)·의미 재랭킹(cross-encoder)·어휘 하이브리드(BM25)를 선택적으로 얹는다.

---

## 1. 결과 비교

test 셋: 평가가능 질의(정답이 그래프에 존재)에서 층화추출(시드 42)한 **301건**. n=301의 95% 신뢰구간 반폭 ≈ ±0.05.
모든 방법이 **동일 test 분할·동일 지표**로 측정. 아래는 **조(article) 단위 overall**(실사용/타 프로젝트 비교의 기준).

| 구성 | R@1 | R@5 | R@30 | Hit@1 | Hit@10 | Hit@30 | MRR@30 |
|---|---|---|---|---|---|---|---|
| [A] BGE 베이스라인 (학습 없음) | .251 | .462 | .670 | .399 | .718 | .801 | .500 |
| [B] +Stage 2 학습 | .320 | .615 | .812 | .478 | .847 | .927 | .599 |
| [C] +PPR 재랭킹 | .321 | .605 | .820 | .482 | .844 | .930 | .602 |
| **[D] +Cross-encoder** (정밀 최고) | **.364** | .617 | .820 | **.548** | .844 | .937 | **.654** |
| **[E] +Hybrid BM25** (커버리지 최고) | .342 | **.636** | **.851** | .518 | **.860** | **.947** | .637 |
| [F] +Hybrid+Cross | .349 | .592 | .849 | .518 | .834 | .944 | .621 |

핵심:
- **[A]→[B] 학습이 최대 상승** (MRR .500→.599, R@30 .670→.812). 파이프라인의 토대.
- **[C] PPR은 학습 위에서 거의 무효** ([B]와 동일 수준). 그래프 구조 신호가 학습된 텍스트 표현에 이미 포화.
- **[D] Cross-encoder = top 정밀도 최고** (Hit@1 +0.07, MRR +0.055). 단 상위 50만 재랭킹해 중위 recall은 소폭 하락.
- **[E] Hybrid(BM25) = 커버리지 최고** (R@30 .851, Hit@30 .947). 법률 용어·조문번호 정확 일치를 dense가 놓치는 부분을 보완. 저비용.
- **[F] Hybrid+Cross 스택은 시너지 없음** — 각 단독보다 못하거나 비슷. 목적에 따라 하나만 선택.

목적별 권장 구성:
- **LLM 컨텍스트 공급**(top-K 안에 정답 포함이 목표) → **[E] Hybrid** (최고 커버리지, cross-encoder 불필요, 저비용)
- **정밀 랭킹**(rank-1 정확도/MRR 중요) → **[D] Cross-encoder**

### 지표 정의 주의

- **recall@K (비율형)**: top-K에 포함된 정답 수 / 전체 정답 수. 정답이 K개보다 많으면 1.0 불가능.
- **hit@K**: top-K에 정답이 1개라도 있으면 1. 타 프로젝트(예: KG-search)의 "Recall@K"는 대개 이 정의이므로 프로젝트 간 비교엔 hit@K 사용.
- 위 표는 조 단위. 항(paragraph) 단위 overall은 더 낮게 나온다([B] R@30 .659, [E] .674, [D] MRR .563) — 후보가 더 잘게 쪼개져 랭킹 난이도가 높기 때문.

### 애블레이션·이전 대비

- **이웃 제외(exclude_neighbors) 효과 없음**: on/off 재학습 결과가 신뢰구간 내 동일(항 단위 Hit@15 .824 vs .841, MRR .550 vs .548). 거짓 음성 우려는 전체 코퍼스 InfoNCE 분모가 이미 흡수. 기본값 유지/해제 무방.
- 이전(74건·22법령) 대비 커버리지 확장(30법령)+평가셋 4배로 수치가 전반적으로 상승했으나 **test 분할이 달라 직접 비교는 불가**.

---

## 1-b. 폭 우선(breadth-first) 컨텍스트 조립 — 프로젝트의 실제 목표

위 §1은 **검색 커버리지**(질의의 정답 조항을 찾는가)를 측정한다. 그러나 이 프로젝트의 최종 목표는
다르다: **질의와 관련된 조항 중 "구조적으로 가장 넓은 범위를 커버하는(=다른 법률과 가장 많이 연결된)
골격 조항"을 컨텍스트 맨 앞에 세우고**, 이 넓은 배경 지식 위에서 하위 질문들을 답변해 일반 RAG보다
통찰 있는 답을 만드는 것이다. 그래서 데이터에 `cross_law_refs`(타법 참조)가 주석되어 있다.

이 관점에서 **학습은 무의미하지 않다** — "관련성 필터"라는 필수 절반을 담당한다(무관하게 넓기만 한
조항을 앞세울 수는 없으므로). 다만 §1의 recall/hit 지표는 이 목표를 측정하지 못한다. 그래서 다음을 추가:

- `breadth.py`: 조항별 **구조적 폭 점수** = `cross_law_refs`의 타법 종수(최우선) + 참조 건수 +
  자법 참조 + KG 인접 이웃 수를 정규화·가중합. (폭 top 조항은 전부 정의·적용제외 같은 골격 조항으로 검증됨)
- `assemble_context.py`: **폭 우선 조립** — 학습된 검색기로 관련 top-N을 뽑고, 그중 최고 폭 조항을
  골격으로 승격해 맨 앞에 배치, 나머지는 관련도순으로 뒤에.

리드 선택은 **관련도·폭 블렌드**로 한다: `score = (1-alpha)*관련도 + alpha*폭` (후보 풀 내 min-max).
`alpha=0`은 순수 관련도, `alpha=1`은 순수 폭.

```bash
# 단건: 질의 -> 골격 조항 + 상세 조항으로 조립된 컨텍스트 출력
python assemble_context.py --query "준법감시인의 자격요건은?" --alpha 0.5

# 배치 평가: alpha 스윕(0~1)으로 관련성-폭 트레이드오프의 최적점 탐색 (같은 test 분할)
python assemble_context.py --test_size 300 --pool_n 20
```

배치 평가 지표 (검색 recall이 아니라 **앞단이 질의의 법적 범위를 얼마나 넓게 감싸는가**):
- **lead_cross_laws**: 리드 조항이 참조하는 타법 수 — 폭을 키우면 커짐.
- **reach_coverage**: 리드의 (자법 ∪ 참조 graph법)이 질의의 정답 법령을 덮는 비율. 전체/단일법/다법(≥3) 분리 보고.
- **blend_lead_relrank**: 리드가 관련도 몇 위였는지 — 낮을수록 관련성 유지.

**실측 결과(test 300, pool_n=20)**: 순수 폭(alpha=1)은 리드 타법 참조를 0.49→1.62로 늘리지만 관련성을
과희생(평균 관련도 7위)해 **전체 커버리지는 0.641→0.557로 하락**. 다만 **num_laws별로 갈린다**:
단일법(0.76→0.63)·2법은 하락, **3법(0.44→0.46)·4법(0.30→0.40)은 상승** — 폭의 이득은 **다법 교차 질의에
국한**된다(사용자 설계의 핵심 use case와 정확히 일치). 따라서 blanket 폭 최대화가 아니라 alpha 블렌드로
관련성을 유지하며 다법 질의에서 폭을 취하는 것이 맞다. alpha 스윕으로 목표에 맞는 지점을 고른다.

> 최종(본질) 평가는 폭-우선 RAG vs 일반 RAG의 답변 통찰을 LLM 심판으로 비교하는 종단 실험이며, 위
> reach_coverage는 그 전 단계의 가벼운 프록시다(단일법 질의는 애초에 넓은 배경이 불필요하므로 프록시가
> 폭을 과소평가하는 한계가 있음 — 종단 평가는 다법·통찰형 질의에 초점을 둘 것).

---

## 2. 최종 아키텍처

```
[오프라인 - Stage 1]
  조항 full_text ── BGE-M3 (고정) ──> 조항 임베딩 7,702 x 1024  (emb_cache/에 캐시)

[학습 - Stage 2]  ... python train_query_encoder.py
  질의 ── BGE-M3 (고정) ──> 1024d ── QueryEncoder(잔차 MLP, ~105만 파라미터, 학습) ──> 1024d
  손실: 전체 코퍼스 분모 multi-positive InfoNCE
        + 주기적 hard negative 재채굴(warmup 10, 매 5 epoch) + margin hinge
  검증 분리(10%) + val Hit@15 기준 best 체크포인트 저장

[검색/평가]  ... python evaluate_rerank.py --ppr --beta 0.5
  1. 코사인 랭킹: QueryEncoder(질의) x 조항 임베딩
  2. PPR 재랭킹: 상위 20개 시드 -> 조항 인접 그래프에서 Personalized PageRank 전파
     (인접 = 같은 조의 형제 항 + 공유 엔터티. 허브 엔터티는 df<=20 컷오프)
  3. 최종 점수 = 코사인 + 0.5 x 정규화된 PPR
```

핵심 설계 원칙: **문서(조항) 쪽 임베딩은 절대 학습하지 않는다.** 사전학습된 BGE 공간을 보존하고,
질의 쪽 소형 MLP만 학습하며, 그래프 구조는 추론 시점(PPR)에 주입한다.
QueryEncoder는 잔차 연결 + 마지막 레이어 0 초기화로 **학습 시작 성능 = 순수 BGE 베이스라인**이 보장된다.

### 기존 HGT 방식이 실패한 이유 (교훈)

구 `train.py`는 무작위 초기화된 Linear+HGT+MLP로 조항 임베딩 공간 전체를 재구축하면서
"정답 1 vs 고정 오답 3"의 4지선다 신호(총 ~900 gradient step)로 학습했다. 결과:
- 사전학습 BGE 정렬이 파괴되고, 3천 건 미만의 질의로는 복원 불가능
- 학습에 등장하지 않는 수천 개 조항의 벡터 위치가 아무 제약 없이 표류 -> 평가 랭킹 오염
- 측정 결과 학습 안 한 베이스라인보다 21배 낮은 성능 (recall@30: 0.018 vs 0.394)

---

## 3. 실행 방법

### 설치

```bash
pip install -r requirements.txt
```

필요 데이터 (`data/`): `nodes.csv`, `triplets.csv`, `for_review_corrected.xlsx`

### 실험 재현 — 이 순서대로 실행 (모두 `--test_size 300` 고정)

아래 순서대로 실행하면, 각 명령이 설정을 드러내는 이름의 결과 파일 하나씩을 `eval_results/`에
남긴다. `--test_size`는 **모든 명령에서 동일(300)** 해야 test 분할이 같아 비교가 성립한다.
(GPU가 있으면 임베딩이 자동으로 GPU에서 계산된다. 데이터가 바뀌면 STEP 1에서 캐시가 자동 재생성됨)

```bash
# ── STEP 1. Stage 2 학습 (최종 운영 모델) ───────────────────────────────────
#   첫 실행: BGE-M3 다운로드(~2.3GB) + 조항 9천여개 GPU 인코딩 후 emb_cache/에 저장.
#   이후 실행은 캐시 재사용으로 수 분. 끝나면 test 지표(항 단위)를 스스로 출력.
python train_query_encoder.py --test_size 300
#   → query_encoder_best.safetensors  (이후 STEP 3~ 에서 사용)
#   → eval_results/stage2_origEmb_summary_*.csv

# ── STEP 2. 베이스라인 (학습 없음, 순수 BGE dense) ──────────────────────────
python evaluate_rerank.py --no_query_encoder --test_size 300
#   → rerank_origEmb_bgeq_dense_none_*        [A] 아무것도 안 한 바닥선

# ── STEP 3. Stage 2, dense, 재랭킹 없음 ─────────────────────────────────────
python evaluate_rerank.py --test_size 300
#   → rerank_origEmb_stage2_dense_none_*      [B] 학습 효과 (A 대비)

# ── STEP 4. + PPR 그래프 재랭킹 ─────────────────────────────────────────────
python evaluate_rerank.py --rerank ppr --beta 0.5 --test_size 300
#   → rerank_origEmb_stage2_dense_ppr-b0.5_*  [C] 그래프 구조 주입 효과 (B 대비)

# ── STEP 5. + Cross-encoder 재랭킹 (1순위 신규) ─────────────────────────────
#   첫 실행 시 BGE-reranker-v2-m3(~2.3GB) 다운로드. 상위 50개만 재랭킹.
python evaluate_rerank.py --rerank cross --test_size 300
#   → rerank_origEmb_stage2_dense_cross-k50_* [D] 의미 재랭킹 효과 (B, C 대비)

# ── STEP 6. Dense+BM25 하이브리드 (2순위 신규) ──────────────────────────────
python evaluate_rerank.py --hybrid --test_size 300
#   → rerank_origEmb_stage2_hybrid_none_*     [E] 어휘 신호 효과 (B 대비)

# ── STEP 7. 하이브리드 + Cross-encoder (스택 최강 후보) ─────────────────────
python evaluate_rerank.py --hybrid --rerank cross --test_size 300
#   → rerank_origEmb_stage2_hybrid_cross-k50_* [F] 최종 조합

# ── STEP 8. 애블레이션: 이웃 제외 on/off (3순위) ────────────────────────────
#   기본(STEP 1)은 이웃 제외 ON. OFF 모델을 다른 이름으로 학습해 끝의 test 표를 비교.
python train_query_encoder.py --test_size 300 --exclude_neighbors 0 --ckpt query_encoder_noNbr.safetensors
#   두 학습의 마지막 "[Stage 2 ...]" 표를 직접 비교 (ON=STEP1 vs OFF=여기)
```

읽는 법: `[A]→[B]` 학습 효과, `[B]→[C]` 그래프(PPR), `[B]→[D]` cross-encoder,
`[B]→[E]` 어휘(하이브리드), `[F]` 최종 스택. 각 파일의 **조 단위** 표가 프로젝트 간 비교용,
**항 단위** 표가 기존 실험과의 연속 비교용이다.

> 평활화(선택): `python build_smoothed_clause_emb.py` 후 위 명령들에 `--clause_emb
> data/clause_emb_smooth.safetensors`를 붙이면 된다(파일명이 `smoothEmb`로 바뀜). 단, Stage 2와
> 결합 시 이득이 사라진다는 이전 결과가 있어 우선순위는 낮다.

- 산출물: `query_encoder_best.safetensors` (모델), `eval_results/*.csv|json` (평가 기록)
- 임베딩 캐시: `emb_cache/` — 텍스트 내용+순서의 해시가 키라서 데이터가 바뀌면 자동 재계산
- GPU: 8GB VRAM이면 충분 (임베딩·cross-encoder는 GPU에서 순차 실행 후 즉시 해제)

### 평가 결과 파일명 규칙 (`eval_results/`)

파일명이 실행 설정을 그대로 드러내므로, 파일명만으로 어떤 조건의 결과인지 식별된다.
`evaluate_rerank.py`의 파일명 형식:

```
rerank_{origEmb|smoothEmb}_{stage2|bgeq}_{dense|hybrid}_{none|ppr-b0.5|cross-k50}_{paragraph|article|summary}_TS
       └ 조항임베딩          └ 질의인코딩     └ 기본검색       └ 재랭킹방식               └ 지표세밀도
```

| 실행 | 생성 파일 (핵심 부분) |
|---|---|
| `evaluate_rerank.py --no_query_encoder` | `rerank_origEmb_bgeq_dense_none_*` (베이스라인) |
| `evaluate_rerank.py` | `rerank_origEmb_stage2_dense_none_*` |
| `evaluate_rerank.py --rerank ppr --beta 0.5` | `rerank_origEmb_stage2_dense_ppr-b0.5_*` |
| `evaluate_rerank.py --rerank cross` | `rerank_origEmb_stage2_dense_cross-k50_*` |
| `evaluate_rerank.py --hybrid` | `rerank_origEmb_stage2_hybrid_none_*` |
| `evaluate_rerank.py --hybrid --rerank cross` | `rerank_origEmb_stage2_hybrid_cross-k50_*` |
| `train_query_encoder.py` | `stage2_origEmb_summary_*` (학습 시 자체 평가) |
| `evaluate_baseline.py` | `baseline_origEmb_summary_*` |

각 rerank 실행은 `_paragraph_`, `_article_` CSV와 `_summary_` JSON을 함께 남긴다.
`eval_results/`는 git 추적 대상이 아니므로(로컬 전용) 규칙 적용 전 옛 파일은 그대로 남는다.

### 주요 하이퍼파라미터

| 파라미터 | 위치 | 기본값 | 설명 |
|---|---|---|---|
| `--epochs` | train_query_encoder | 100 | 학습 epoch |
| `--temp` | train_query_encoder | 0.1 | InfoNCE 온도 |
| `--hard_neg_k` | train_query_encoder | 10 | 질의당 hard negative 수 (0=비활성) |
| `--exclude_neighbors` | train_query_encoder | 1 | 정답의 그래프 이웃을 hard negative·InfoNCE 분모에서 제외(거짓 음성 방지) |
| `--ckpt` | train_query_encoder | query_encoder_best.safetensors | 모델 저장 파일명 (애블레이션 시 다른 이름) |
| `--test_size` | 공통(baseline/stage2/rerank) | 100 | test 질의 수. **모든 스크립트에 같은 값 필수** |
| `--rerank` | evaluate_rerank | none | 재랭킹: none / ppr(그래프) / cross(cross-encoder) |
| `--hybrid` | evaluate_rerank | off | dense + BM25 어휘 하이브리드(RRF 융합) |
| `--beta` | evaluate_rerank | 0.3 (권장 0.5) | PPR 점수 혼합 비중 |
| `--rerank_topk` | evaluate_rerank | 50 | cross-encoder 재랭킹 후보 수 |
| `--seeds` | evaluate_rerank | 20 | PPR 시드 조항 수 |
| `--max_entity_df` | 공통 | 20 | 허브 엔터티 컷오프 (초과 연결 엔터티 제외) |

### 재랭킹·하이브리드 아키텍처 (성능 평가 축)

- **Cross-encoder 재랭킹** (`--rerank cross`): bi-encoder가 뽑은 상위 K개를 (질의,조항) 쌍으로
  BGE-reranker-v2-m3에 넣어 정밀 재정렬. 학습 불필요, 상위 K만 처리해 저렴. 재랭킹 중 기대효과 최대.
- **Dense+BM25 하이브리드** (`--hybrid`): BGE dense 랭킹과 BM25 어휘 랭킹을 RRF로 융합.
  법률 텍스트의 정확한 용어/조문번호 일치를 dense가 놓칠 때 보완. BM25는 의존성 없이 내장 구현
  (한글 2-gram 토크나이저).
- **exclude_neighbors 애블레이션**: `--exclude_neighbors 0` 학습본과 기본(1) 학습본의 최종 test 표를
  비교해 거짓 음성 제거의 실제 효과를 측정.

### 평가셋 구성 및 거짓 음성 처리 (데이터 품질 개선)

- **평가가능 질의 기준 층화추출**: test 표본을 "정답이 그래프에 존재하는(평가가능)" 질의에서만
  뽑는다. `--test_size N`이 곧 평가가능 test 질의 수가 되어(기존에는 100건 추출 후 26건이
  평가불가로 버려짐) 목표 크기를 정확히 통제한다. 통계적 유의성을 위해 `--test_size 300` 권장
  (n=74에서 95% CI ±0.10 → n=300에서 ±0.05).
- **거짓 음성(false negative) 제외**: 정답 조항의 그래프 이웃(형제 항·공유 엔터티 조항)은 실제로
  관련 조항일 확률이 높으므로, hard negative 채굴과 InfoNCE 분모에서 제외한다(`--exclude_neighbors 1`).
- 주의: 위 두 변경은 **test 분할 자체를 바꾸므로 이전 실행 수치와 직접 비교 불가**. baseline·stage2·rerank를
  모두 같은 `--test_size`로 다시 실행해 재-베이스라인해야 한다.

---

## 4. 파일 구조

```
├── data_loader.py               # 데이터 로드, 조항 키 정규화, fsc 전처리/층화 분할,
│                                #   HeteroData 그래프 구축, 임베딩 캐시(encode_texts_cached)
├── retrieval_common.py          # 공유 로직: 평가 아이템 구성, 조항 인접 그래프, 지표 계산
├── query_encoder.py             # Stage 2 모델 (잔차 MLP, 0 초기화)
├── train_query_encoder.py       # Stage 2 학습 + test 평가  ★ 메인 학습 스크립트
├── evaluate_rerank.py           # 최종 평가 (PPR 재랭킹 + 항/조 단위 지표)  ★ 메인 평가 스크립트
├── evaluate_baseline.py         # 순수 BGE 베이스라인 측정
├── build_smoothed_clause_emb.py # (실험) 그래프 평활화 임베딩 생성
├── train.py                     # (레거시) HGT 전체 학습 방식 - 비교 보존용, 사용 비권장
├── hgt_model.py                 # (레거시) HGT 모델 정의
└── data/
    ├── nodes.csv                # 조항 노드 (22개 법령, 항 단위)
    ├── triplets.csv             # SPO 트리플릿 (엔터티 관계)
    └── for_review_corrected.xlsx  # FSC 법령해석 질의 (정답 조항 인용 포함)
```

## 5. 데이터 참고 사항

- 조항 키는 `법령명 제X조[의Y] 제X항` 포맷으로 정규화 (`normalize_johang_key`).
  조 단위로만 인용된 정답은 해당 조의 모든 항으로 확장하여 매칭.
- test 질의 100건 중 26건은 참조 법령이 그래프에 없어 평가 제외 (데이터 커버리지 한계).
- train/test 분할은 질의(jilui) 단위 층화추출('# of laws_clean' 기준, 시드 42) — 누수 없음.
