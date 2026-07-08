# Financial-HGT: 금융 법령 조항 검색 (KG 기반 Retrieval)

금융위원회(FSC) 법령해석 질의에 대해, 지식그래프(KG)로 구축된 금융 법령 조항(항 단위 7,702개 노드, 22개 법령)에서
관련 조항을 검색하는 파이프라인. BGE-M3 임베딩 + 경량 QueryEncoder 학습 + PPR 그래프 재랭킹으로 구성된다.

---

## 1. 결과 비교

test 셋: 층화추출(시드 42)로 분리한 질의 100건 중 그래프에 정답이 존재하는 74건.
모든 방법이 **동일한 test 분할, 동일한 지표**로 측정됨.

### 방법별 성능 (항 단위, overall)

| 방법 | recall@1 | recall@10 | recall@30 | MRR@30 |
|---|---|---|---|---|
| HGT 전체 학습 (구 `train.py`) | 0.014 | 0.017 | 0.018 | — |
| 순수 BGE-M3 코사인 (학습 없음) | 0.123 | 0.273 | 0.394 | 0.338 |
| BGE + 그래프 평활화 (학습 없음) | 0.146 | 0.323 | 0.422 | 0.360 |
| BGE + Stage 2 QueryEncoder | 0.181 | 0.430 | 0.531 | 0.436 |
| **BGE + Stage 2 + PPR 재랭킹 (최종)** | **0.174** | **0.420** | **0.554** | **0.441** |

- 최종 구성은 기존 HGT 방식 대비 **recall@30 기준 31배**, 순수 BGE 대비 **+41%**.
- 그래프 평활화(임베딩 수정)는 단독으로는 유효했으나 Stage 2 학습과 결합 시 이득이 사라짐
  (QueryEncoder가 같은 정보를 학습으로 흡수). 반면 PPR(추론 시점 재랭킹)은 Stage 2와 중복되지 않아 순증.

### 최종 구성의 조(article) 단위 성능

항 단위 랭킹을 조 단위로 접은 것(조 점수 = 소속 항 최고 점수). 조 단위 검색 시스템과 비교할 때는 이 표를 사용.

| 지표 | @1 | @3 | @5 | @10 | @15 | @30 |
|---|---|---|---|---|---|---|
| recall (비율형) | 0.246 | 0.464 | 0.530 | 0.606 | 0.645 | 0.732 |
| **hit (정답 1개 이상 포함)** | 0.392 | 0.595 | 0.662 | 0.730 | 0.743 | **0.797** |

MRR@30 = 0.504. 질의 10건 중 8건은 조 단위 top-30 안에 정답 조가 포함된다.

### 지표 정의 주의

- **recall@K (비율형)**: top-K에 포함된 정답 수 / 전체 정답 수. 정답이 K개보다 많으면 1.0 불가능
  (이 test 셋의 이론적 상한: recall@1 = 0.56).
- **hit@K**: top-K에 정답이 1개라도 있으면 1. 타 프로젝트(예: KG-search)의 "Recall@K"는 대개 이 정의이므로
  프로젝트 간 비교에는 hit@K를 사용할 것.

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

### 파이프라인

```bash
# 1. Stage 2 학습 (첫 실행 시 BGE-M3 다운로드 ~2.3GB + 조항 임베딩 인코딩 후 캐시.
#    이후 실행은 emb_cache/ 재사용으로 수 분 내 완료. 학습 후 test 평가 자동 수행)
python train_query_encoder.py

# 2. 최종 평가 (PPR 재랭킹 포함, 항 단위 + 조 단위 지표)
python evaluate_rerank.py --ppr --beta 0.5

# 참고: 베이스라인 측정 (학습 없음)
python evaluate_baseline.py

# 참고: 그래프 평활화 임베딩 실험 (최종 구성에는 미사용)
python build_smoothed_clause_emb.py
python evaluate_rerank.py --clause_emb data/clause_emb_smooth.safetensors
```

- 산출물: `query_encoder_best.safetensors` (모델), `eval_results/*.csv|json` (평가 기록)
- 임베딩 캐시: `emb_cache/` — 텍스트 내용+순서의 해시가 키라서 데이터가 바뀌면 자동 재계산
- GPU: 8GB VRAM이면 충분 (인코딩은 CPU, 학습은 행렬곱만이라 가벼움)

### 주요 하이퍼파라미터

| 파라미터 | 위치 | 기본값 | 설명 |
|---|---|---|---|
| `--epochs` | train_query_encoder | 100 | 학습 epoch |
| `--temp` | train_query_encoder | 0.1 | InfoNCE 온도 |
| `--hard_neg_k` | train_query_encoder | 10 | 질의당 hard negative 수 (0=비활성) |
| `--beta` | evaluate_rerank | 0.3 (권장 0.5) | PPR 점수 혼합 비중 |
| `--seeds` | evaluate_rerank | 20 | PPR 시드 조항 수 |
| `--max_entity_df` | 공통 | 20 | 허브 엔터티 컷오프 (초과 연결 엔터티 제외) |

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
