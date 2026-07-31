# BGE-M3 LoRA 파인튜닝 — QueryEncoder(MLP)의 대조군

## 왜 만드는가

이 레포의 검색 성능 향상은 전부 **QueryEncoder(잔차 MLP, 105만 파라미터)** 에서 나온다
(README §1: 조 단위 MRR .500 → .599, PPR은 그 위에서 +.003).

그런데 **"BGE-M3 fine-tuning보다 MLP가 효과적"이라는 주장에는 대조군이 없다**
(`analysis/NOTION_UPDATE.md:151`). 이 레포와 KG-search 레포 전수 검색 결과 `lora`/`peft`는
0건이고, BGE는 설계상 항상 동결이다. 지금까지 "임베딩 fine-tuning은 실패"로 불려온 결과는
사실 **HGT로 문서 임베딩 공간을 재구축한 실험**(recall@30 0.018 vs 0.394)이지 BGE
파인튜닝이 아니다.

`lora/`는 그 **빠진 대조군**이다.

---

## 1. 두 방식이 같은 출발점에서 시작한다는 것의 원리

비교의 공정성이 여기에 걸려 있으므로 정확히 기록한다.

### MLP 쪽 (`query_encoder.py:24-40`)

```python
forward:  out = normalize(x + MLP(x))
MLP    :  Linear1(1024→512) → LayerNorm → GELU → Dropout → Linear2(512→1024)
init   :  nn.init.zeros_(Linear2.weight);  nn.init.zeros_(Linear2.bias)
```

**(1) Δ=0이 항등식이다.** `Linear2(h) = W₂h + b₂`이고 `W₂=0, b₂=0`이므로 **h가 무엇이든**
출력이 정확히 0이다. Linear1·LayerNorm·GELU가 무엇을 내놓든 상관없다. 따라서

```
out = normalize(x + 0) = normalize(x)
```

이는 순수 BGE 코사인 베이스라인과 **부동소수점 수준까지 동일**하다. "가중치가 작아서
비슷하다"가 아니라 **대수적 항등**이므로 초기화 시드·데이터·차원과 무관하게 성립한다.
이것이 "구조적 보장"의 의미다.

**(2) 그런데 gradient는 죽지 않는다.** `y = W₂h + b₂`에서

- `∂y/∂W₂ = hᵀ ≠ 0` (Linear1이 랜덤 초기화라 h≠0) → **W₂는 첫 step부터 움직인다**
- `∂y/∂h = W₂ = 0` → **첫 step에서 Linear1로 가는 gradient는 0**

즉 step 1에서는 마지막 층만 학습되고, W₂가 0을 벗어난 step 2부터 Linear1이 따라 학습된다.
정지가 아니라 **웜스타트**다.

> Linear1까지 0으로 뒀다면 h=0이 되어 `∂y/∂W₂=0`, 영구 정지한다.
> **마지막 층만** 0으로 두는 것이 핵심이다.

**(3) 시작점 보장이지 종료점 보장이 아니다.** 학습이 진행되면 얼마든지 베이스라인 아래로
갈 수 있다. 최종 산출물을 보호하는 건 **두 번째 장치**다 — `train_query_encoder.py:251`이
`best_val`을 **베이스라인 점수로 초기화**해 두어, 어떤 epoch도 베이스라인을 못 넘으면
저장된 체크포인트가 항등 모델(=베이스라인)로 남는다. 두 겹이며, 논문에 쓸 때 구분해야 한다.

### LoRA 쪽 — 정확히 같은 구조

LoRA는 원 가중치를 `W_eff = W₀ + (α/r)·B·A`로 대체한다. PEFT 기본 초기화는
**A ~ Kaiming 랜덤, B = 0**이다.

**(1) Δ=0이 항등식이다.** `B=0`이므로 `B·A = 0` (영행렬 × 임의행렬). A가 무엇이든 상관없다.
따라서 `W_eff = W₀`, 즉 어댑터를 붙인 모델이 **원본 BGE-M3와 완전히 같은 함수**다.
질의 임베딩이 순수 BGE 임베딩과 동일하므로 검색 성능도 베이스라인과 같다.

**(2) gradient도 같은 방식으로 산다.**

- `∂L/∂B = (∂L/∂y)(Ax)ᵀ ≠ 0` (A가 랜덤이라 Ax≠0) → **B가 첫 step부터 움직인다**
- `∂L/∂A = Bᵀ(∂L/∂y)xᵀ = 0` → **첫 step에서 A는 정지**

A와 B를 둘 다 0으로 두면 영구 정지하므로 한쪽만 0으로 둔다.

### 동형(isomorphism)

두 방식 모두 *"잔차 경로의 마지막 곱셈 인자를 0으로 두어 Δ=0을 만들되, 그 인자에 대한
gradient는 살려 둔다"* 는 동일한 트릭이다.

| | MLP | LoRA |
|---|---|---|
| 잔차 경로 | `x + MLP(x)` | `W₀x + (α/r)BAx` |
| 0으로 두는 인자 | `W₂` (마지막 Linear) | `B` |
| 랜덤으로 두는 인자 | `W₁` (첫 Linear) | `A` |
| step 0 출력 | `normalize(x)` = 베이스라인 | `W₀x` = 베이스라인 |
| step 1에 움직이는 것 | `W₂`만 | `B`만 |

**어느 한쪽도 초기 조건에서 유리/불리하지 않다.** 이것이 MLP vs LoRA 비교의 공정성 근거다.

`train_lora.py`는 두 번째 장치(best_val 베이스라인 초기화)도 동일하게 적용한다.
이걸 빼면 보호 장치가 MLP에만 있어 비교가 비대칭이 된다.

**검증**: 학습 전 val Hit@15가 `evaluate_baseline.py` 값과 정확히 일치해야 한다(게이트 ②).
불일치하면 위 항등식이 깨진 것 — 대개 풀링/정규화 불일치가 원인이다(게이트 ①이 먼저 잡는다).

---

## 2. 설정과 근거

| | MLP (기존) | LoRA (신규) |
|---|---|---|
| 학습 대상 | 질의 임베딩 위 잔차 MLP | BGE-M3 attention의 q·v 투영 |
| 문서 측 | 동결 | **같은 캐시 파일 그대로 재사용** |
| 파라미터 | 1,051,136 | **1,572,864** (r=16: 24층 × 2모듈 × 2×1024×16) |
| epochs | 100 | 12 |
| lr | 3e-4 | 1e-4 |
| batch | 32 | micro 8 × accum 4 |

**epochs 100 → 12**: MLP는 캐시된 임베딩만 곱해서 100 epoch가 수 분이지만, LoRA는 매 step
BGE forward+backward가 필요하다. 사전학습 백본 위의 LoRA는 훨씬 빨리 수렴한다.
hard-negative 일정도 비례 축소(warmup 10→2, interval 5→2).

**lr 3e-4 → 1e-4**: 사전학습 가중치를 직접 건드리므로 보수적으로.

**micro 8 × accum 4**: 이 손실은 질의마다 독립항이고 InfoNCE 분모가 전체 코퍼스(9,311개)라
**in-batch negative가 없다.** 따라서 누적 gradient가 배치 32와 **수학적으로 동일**하다.
대조학습에서 흔한 "작은 배치가 성능을 깎는다" 문제가 여기선 발생하지 않는다.

**QLoRA를 쓰지 않는 이유**: BGE-M3(XLM-RoBERTa-large, 568M)를 fp16으로 올리면 1.14GB다.
T4 16GB에서 4-bit 양자화는 이득 없이 속도만 느려지고 불안정성만 추가된다.
T4는 Turing이라 **bf16 미지원** → fp16 autocast + GradScaler를 쓴다.

**나머지는 전부 MLP와 동일**: `fsc_dataset_preprocessing(test_size=300)` 층화 분할(seed 42),
조→항 정답 확장, 전체 코퍼스 InfoNCE 분모, hard negative 재채굴 + margin hinge, 이웃 제외,
val Hit@15 기준 best 체크포인트. 손실 함수는 `train_query_encoder.infonce_multi_positive`를
**그대로 import**한다 — 재구현하지 않는다.

---

## 3. Colab에 무엇을 가져가는가

### 결론: 코드·데이터는 업로드할 것이 없다. 체크포인트 1개만 가져가면 된다.

`data/`가 git에 추적되고 있어(`git ls-files data/`) `git clone` 한 번이면 코드와 데이터가
모두 들어온다. 레포 전체 pack이 15.4MiB뿐이다.

**A. git clone으로 자동 확보 (업로드 불필요)**

- `data/nodes.csv` (8.3MB) · `data/triplets.csv` (7.1MB) · `data/for_review_corrected.xlsx` (5.8MB)
- `data_loader.py` · `retrieval_common.py` · `query_encoder.py` · `train_query_encoder.py` ·
  `ranking_methods.py` · `evaluate_baseline.py` · `evaluate_rerank.py` · `lora/` 일체

**B. Colab에서 재생성 (업로드 불필요, 시간만 소요)**

| 산출물 | 방법 | 비용 |
|---|---|---|
| BGE-M3 가중치 (~2.3GB) | HF 자동 다운로드 | 3~5분 |
| `emb_cache/clause_embs_<md5>.safetensors` (~36MB) | `encode_texts_cached`, 조항 9,311건 | T4로 3~5분 |
| `emb_cache/fsc_query_embs_<md5>.safetensors` (~12MB) | 동일, 질의 ~3천건 | 1~2분 |

캐시 키가 텍스트 내용+순서의 md5(`data_loader.py:46-53`)라 **CSV가 바이트 동일하면
로컬/Colab 캐시가 서로 호환된다.** 로컬에 `emb_cache/`가 있으면 올려서 시간을 아낄 수 있다.

**C. ⭐ 로컬에서 반드시 가져올 것**

| 파일 | 크기 | 이유 |
|---|---|---|
| `query_encoder_best.safetensors` | ~4.2MB | 기존 `pl_hybrid` 답변 결과(REF_avg 0.1710)를 만든 **바로 그 모델**. 비교 기준선 |

`.gitignore`(`*.safetensors`) 대상이라 git에 없다. **재학습으로 대체하면 안 된다** — 다른
체크포인트가 나오고(SentenceTransformer 버전 차이로 임베딩이 미세하게 달라질 수 있음)
기존에 아카이브된 답변 결과와의 연결이 끊긴다. 재학습은 **README 수치 재현 검증** 용도로만
쓴다(노트북 4-b 셀).

**D. 답변 파이프라인(경로 1) 비교에 추가로 필요한 것** — insight-agent 레포 쪽

`data/kg/fhgt_graph/{node_emb.pt, test_pairs.jsonl, nodes.csv}`,
`data/kg/checkpoints/query_encoder_fhgt_best.pt`.
`export_fhgt_graph.py`로 재생성 가능하지만 **`test_pairs.jsonl`은 반드시 기존 것을 쓴다** —
재생성 시 301문항의 순서·구성이 달라지면 `analysis/paired_answer_analysis.py:65-71`의
페어링 검사에서 `ValueError`가 나고 기존 13개 구성과 비교가 불가능해진다.

**E. Colab → 로컬로 가져올 산출물**

| 파일 | 크기 | 용도 |
|---|---|---|
| `lora/checkpoints/` | ~6MB | LoRA 어댑터 |
| `emb_cache/fsc_query_embs_lora.safetensors` | ~12MB | **LoRA 질의 임베딩** — 이후 검색 평가를 GPU 없이 로컬에서 돌릴 수 있다 |
| `lora/results/` · `eval_results/*paragraph*.csv` | 수십 KB | 지표 |

---

## 4. 실행 순서

```bash
# Colab (T4). 노트북이 아래를 순서대로 실행한다.
lora/train_lora_colab.ipynb

# 또는 CLI로 직접
python lora/train_lora.py --test_size 300 --epochs 12 --max_len 256

# 검색 평가 3 arm — LoRA는 --query_emb_file로 넣으면 이후 경로가 MLP arm과 동일
python evaluate_rerank.py --no_query_encoder --hybrid --test_size 300
python evaluate_rerank.py --hybrid --test_size 300
python evaluate_rerank.py --query_emb_file emb_cache/fsc_query_embs_lora.safetensors \
                          --hybrid --test_size 300

# 대응표본 검정 (주 지표 = 항 단위)
python lora/paired_retrieval_analysis.py \
    --mlp  "eval_results/rerank_origEmb_stage2_hybrid_none_paragraph_*.csv" \
    --lora "eval_results/rerank_origEmb_lora_hybrid_none_paragraph_*.csv" \
    --baseline "eval_results/rerank_origEmb_bgeq_hybrid_none_paragraph_*.csv"
```

---

## 5. 비교 프로토콜

### 주 비교축은 항(paragraph) 단위

조 단위는 서브 지표로만 확인한다. 두 세밀도에서 순위가 뒤집히는 경우가 실제로 있다
(README §1 각주: PPR의 Recall@15 1위는 항 단위에서만 성립, 조 단위에선 3위).

지표 정의 주의(README §1): `recall@K`는 **비율형**(top-K 정답 수 / 전체 정답 수),
`hit@K`는 **이진**. 타 프로젝트의 "Recall@K"는 대개 `hit@K`다.

### 기존 MLP 항 단위 데이터가 남아 있는 위치 — 경로가 둘이고 수치가 다르다

**경로 1 — 답변 파이프라인** (원자료 보존됨).
`insight-agent/analysis/answer_details/*.jsonl`의 `retrieved`(top-15)를 `gold_positives`와
대조해 문항 단위로 계산. insight-agent에는 조 단위 접기가 없어 전부 항 단위 node_id 정확매칭.

| 구성 | R@1 | Hit@1 | **R@15** | Hit@15 | MRR@15 | REF_avg |
|---|---|---|---|---|---|---|
| `bge_m3` | .1309 | .2458 | **.3255** | .5316 | .3137 | .1319 |
| **`pl_hybrid` (MLP+hybrid)** | .1729 | .3322 | **.4352** | .6678 | .4276 | **.1710** |
| `pl_ppr` | .1798 | .3355 | .4281 | .6844 | .4300 | .1605 |
| `pl_dense` | .1830 | .3355 | .4224 | .6744 | .4299 | .1560 |
| `pl_cross` | .1930 | .3455 | .4129 | .6445 | .4267 | .1598 |

**경로 2 — `evaluate_rerank.py`** (CSV 소실, README 파편만).
`eval_results/`가 `.gitignore` 대상이고 디렉터리 자체가 없다. 남은 것:
README §1 각주 [B] R@30 .659 / [E] R@30 .674 / [D] MRR@30 .563, 애블레이션 [B] 항 Hit@15
.824(neighbors on) · .841(off), MRR .550 · .548. **항 R@15은 기록이 없다** → MLP arm 재실행 필요.

**⚠️ 두 경로를 같은 표에 넣지 말 것.** 같은 "항 단위 Hit@15"인데 경로 1은 .668, 경로 2는 .824다.
경로 2는 원질문 1회로 전체 9,311개를 랭킹하고, 경로 1은 답변 파이프라인의 검색 호출
결과(top-15 고정) + in-KG 필터를 쓴다. 비교는 항상 같은 경로 안에서만 한다.

### Stage B — 답변 품질 (LoRA arm만 신규 실행)

MLP 결과는 이미 있으므로 `lo_pl_hybrid` 하나만 생성해 `pl_hybrid`와 문항 단위로 대응시킨다.

```
ANSWER_MODEL=google/gemma-4-26b-a4b
EVAL_MODEL=<gemini-2.5-flash OpenRouter 슬러그>
TEST_PAIRS=data/kg/fhgt_graph/test_pairs.jsonl     # 301건, 기존 파일 그대로
```

metric·details 필드는 `eval_answers.py`를 손대지 않으므로 자동으로 기존과 동일하다 —
judge 9종(`llm_judge.py:27-32`) + `answer_cite_recall`, jsonl 필드 13종
(`eval_answers.py:339-354`). `REFERENCE_avg`는 분석 단계에서
`(answer_correctness + answer_completeness)/2`.

**⚠️ 시작 전 judge 정합성 확인.** 기존 301문항 런이 어떤 judge를 썼는지 레포에 기록이 없다
(`EVAL_MODEL`은 `.env`로만 주입되는데 `.env`가 없고, 레포 전체에 `gemini` 문자열 0건,
`eval_answers.py:25` 기본값은 `openai/gpt-4o-mini`). judge가 arm 간 다르면 LoRA−MLP 차이가
검색기 때문인지 judge 때문인지 분리되지 않는다 — 기존 효과크기가 Δ≈0.02~0.04라 judge 교체만으로
뒤집힐 수 있다.

1. 기존 런의 `EVAL_MODEL` 슬러그를 노션/로컬 `.env`/`runs/*/answer_metrics.json`에서 특정
2. 다르면 **답변 재생성 없이 judge만 재실행** — 아카이브된 `pl_hybrid.jsonl`에
   `query`·`context`·`final_answer`·`ground_truth`가 전부 있어 judge만 301회 다시 돌리면
   교란이 완전히 제거된다 (full arm의 절반 이하 비용)
3. 재채점을 안 하면 결과에 "judge가 arm 간 다름 — 인과 해석 불가"를 명시

**검정력 사전 고지**: 3-4법 버킷 동점률 63~72%, 필요표본 hybrid ≈61건인데 현재 46건이다.
n=301 전체에서는 Δ≈0.03을 잡을 수 있지만 **3-4법 버킷은 애초에 결론을 낼 수 없다.**

insight-agent 변경분은 `lora/patches/insight_agent_lora.diff`에 보존한다
(해당 레포는 이 세션의 GitHub 접근 범위 밖이라 푸시 불가).

---

## 6. 최종 산출 표 (항 단위, 경로 1 기준)

| arm | 파라미터 | R@1 | **R@15** | Hit@15 | MRR@15 | REF_avg |
|---|---|---|---|---|---|---|
| BGE 베이스라인 | 0 | .1309 | .3255 | .5316 | .3137 | .1319 |
| MLP (`pl_hybrid`) | 1.05M | .1729 | **.4352** | .6678 | .4276 | **.1710** |
| **LoRA r=16** | 1.57M | | | | | |

각 셀에 MLP 대비 대응표본 Δ와 95% CI를 병기한다. 조 단위 표는 부록.

---

## 파일

```
lora/
├── README.md                      이 문서
├── model.py                       build_lora_bge() / encode_queries_lora()
├── train_lora.py                  학습 + test 평가 + 질의 임베딩 export
├── train_lora_colab.ipynb         Colab T4 드라이버
├── paired_retrieval_analysis.py   MLP vs LoRA 대응표본 검정
├── checkpoints/                   adapter_config.json + adapter_model.safetensors
├── results/                       지표 CSV/JSON
└── patches/                       insight-agent 변경분 보존
```

기존 파일 수정은 3곳뿐이다: `evaluate_rerank.py`(`--query_emb_file` 플래그),
`requirements.txt`(peft/transformers 명시), `.gitignore`(어댑터 예외).
