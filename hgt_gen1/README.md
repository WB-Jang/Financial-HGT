# Gen1 방식 HGT 재학습 (Gen2 데이터)

구 `train.py`의 HGT는 학습 안 한 베이스라인의 **1/21**(recall@30 0.018 vs 0.394)로 붕괴했다.
같은 데이터를 **Gen1(KG-search) 방식**으로 다시 학습해, 붕괴가 "HGT라는 모델" 때문인지
"학습 방식" 때문인지 가른다.

## 붕괴 원인과 이 설계의 대응

| 구 train.py의 붕괴 원인 | 여기서의 대응 |
|---|---|
| 동결 BGE 질의 공간에 문서가 맞춰야 함 (흡수 장치 없음) | **역전** — 문서 공간을 먼저 만들고 Stage 2에서 질의가 맞춤 |
| 조항의 68.8%가 손실에 한 번도 등장하지 않음 | 엣지 기반이라 **95.3%**가 매 epoch 감독받음 (실측) |
| attention 풀링된 centroid만 제약 (질의 70.3%가 다중 정답) | 링크예측이 **개별 노드쌍 코사인**을 직접 제약. 풀링 없음 |
| 4지선다 × ~900 step | 엣지 32,735개 × (1+3 음성) × 최대 100 epoch |

## 실행

```bash
python hgt_gen1/build_edges.py                                  # Stage 0 (BGE 불필요, 수 분)
python hgt_gen1/train_stage1.py --epochs 100 --hidden 256        # Stage 1
python hgt_gen1/train_stage2.py --test_size 300 --epochs 100     # Stage 2
python hgt_gen1/evaluate_hgt.py --test_size 300                  # Stage 3
```

## Stage 0 — 엣지와 회귀 타깃 (실행 완료, `graph/`에 산출됨)

`retrieval_common.build_clause_adjacency`(형제 항 + 공유 엔터티)에 `nodes.csv`의
`cross_law_refs` 명시 인용을 더한 뒤, 법 관계로 스코프를 나눈다.

```
조항 노드 9,311개 | 법령 30종
형제 항 18,514쌍 + 공유 엔터티 15,358쌍 = 고유 30,495 엣지
명시 인용 2,787건 추가 (조문 미명시로 생략 1,143건)

  INTRA     23,592건 | target 평균 0.1235 중앙값 0.1111
  SIBLING    2,400건 | target 평균 0.0871 중앙값 0.0833
  CROSS      6,743건 | target 평균 0.0980 중앙값 0.0833

엣지에 등장하는 노드: 8,877/9,311 (95.3%)
```

### Gen1과 다른 점 2가지 (의도적)

**① CROSS 엣지의 출처.** Gen1은 FSC 질의 행의 `연관키워드집합` co-occurrence를 썼는데
`data/for_review_corrected.xlsx`에 그 컬럼이 없다. 대신 `cross_law_refs`의 명시 인용을 쓴다.

이건 손실이 아니라 개선이다 — Gen1은 FSC 행을 쓰므로 test 질의 누수를 막는
`--exclude_queries` 플래그가 필요했지만, 여기서는 **질의 데이터가 전혀 개입하지 않아
누수가 구조적으로 불가능**하다. 법리적 인용이라 "같이 언급됐다"보다 정밀하기도 하다.

**② confidence 정규화 복원.** insight-agent의 `export_fhgt_graph.py:238,240`은
SIBLING/CROSS의 confidence를 **1.0으로 하드코딩**한다. 그 값을 회귀 타깃으로 쓰면
"모든 cross 쌍의 코사인을 1로(=동일 벡터로)" 만드는 목표가 되어 그 자체로 표현이
붕괴한다. Gen1처럼 `co_occurrence / max_co_occurrence`를 복원했다(위 표의 평균 ~0.09).

## Stage 1 — HGT 링크예측 회귀

```
양성 = 엣지, target = 위에서 만든 정규화 co-occurrence
음성 = 랜덤 노드쌍, target = 0 (neg_ratio 3, epoch마다 재추출)
손실 = MSE_intra + λ·MSE_cross + MSE_neg
AdamW lr 1e-3 / wd 1e-2, cosine, clip 1.0, 100 epoch, early stopping patience 5
```

**게이트**: 매 epoch `cos(이웃) vs cos(무작위쌍)`을 출력한다. 학습이 진행되는데도 두 값이
벌어지지 않으면 구조를 못 배우는 것이므로 중단하고 원인을 봐야 한다.

노드 타입은 `law_id`(30종)가 아니라 **`law_type`(법률/시행령/시행규칙/규정/기타)** 을 쓴다.
Gen1은 법령 9종·노드 6,103개로 법령당 678개였지만 Gen2는 30종·9,311개로 310개다.
작은 법령은 타입별 Q/K/V가 학습되지 않는다. 법령 정체성은 `NodeEncoder`의 `law_id`
임베딩으로 그대로 들어간다.

## Stage 2 — 질의 정렬

Stage 1의 256d 노드 임베딩을 **동결**하고 `QueryEncoder256`(1024→512→256)만 학습한다.
손실·hard negative 재채굴·이웃 제외는 `train_query_encoder.py`에서 **import해서** 쓰므로
MLP arm과 학습 신호가 비트 단위로 같다. 분할(seed 42)·정답 확장·이웃 집합도 레포 공용
함수를 써서 자동으로 일치한다.

**⚠️ MLP arm과 다른 조건**: 출력이 256d라 잔차 연결(`x + MLP(x)`)이 차원상 성립하지 않는다.
따라서 "학습 시작 = 순수 BGE 베이스라인" 보장이 없고, best 체크포인트의 baseline seeding도
불가능하다(다른 공간이라 비교 자체가 성립하지 않음). Gen1과 같은 조건이며 **결과 보고 시
반드시 명시할 것.**

## Stage 3 — 평가

`compute_metric_rows`(항 단위, 주 지표) / `compute_article_metric_rows`(조 단위, 서브)를
그대로 호출하므로 MLP·BGE arm과 계산 경로가 같다. 재랭킹(dense/hybrid/ppr/cross)은
적용하지 않는다 — Stage 2가 학습한 것이 순수 코사인이기 때문이다.

## 비교 대상

| arm | 항 recall@30 | 비고 |
|---|---|---|
| 순수 BGE | .394 | 학습 없음 |
| MLP (잔차 1024d) | .659 | 현행 최고 |
| 구 HGT (Gen2 방식) | **.018** | 붕괴 |
| **Gen1식 HGT** | ? | 이번 작업 |

BGE 베이스라인을 넘지 못해도 결과다 — "문서 공간 학습은 이 데이터 규모에서 부적합"이
결론이 되고, 그 역시 논문에 필요한 대조군이다. 중요한 건 **.018과 같은 붕괴가 재현되는지**다.
붕괴가 사라지면 원인이 모델이 아니라 학습 방식이었다는 것이 확정된다.

## 실행 환경

로컬에 torch가 없다. LoRA 작업과 동일하게 Colab에서 돌린다 —
레포를 clone하면 `data/`가 함께 들어오고, `emb_cache/`는 첫 실행에서 재생성된다
(조항 9,311건 인코딩 T4로 3~5분). Stage 0은 pandas만 쓰므로 로컬에서도 실행된다.
