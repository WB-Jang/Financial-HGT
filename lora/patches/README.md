# insight-agent LoRA arm 패치

`insight_agent_lora.diff`는 insight-agent 레포(`final/additional-test-results` 기준)에
`fhgt_<sub>_lora` 모드를 추가한다. 그 레포는 이 세션의 GitHub 범위 밖이라 변경분을
여기 보존한다.

## 적용

```bash
cd <insight-agent>
git apply /path/to/insight_agent_lora.diff
poetry install --with gnn        # peft>=0.11 추가됨
```

## 어댑터 배치

`lora/checkpoints/`의 두 파일을 그대로 복사한다. 디렉터리 이름이 곧 `LORA_PATH`다.

```bash
mkdir -p data/kg/checkpoints/lora_bge_m3
cp <Financial-HGT>/lora/checkpoints/adapter_config.json      data/kg/checkpoints/lora_bge_m3/
cp <Financial-HGT>/lora/checkpoints/adapter_model.safetensors data/kg/checkpoints/lora_bge_m3/
```

## 실행

```bash
MODE=fhgt_dense_lora \
ANSWER_MODEL=google/gemma-4-26b-a4b \
EVAL_MODEL=google/gemini-2.5-flash-lite \
TEST_PAIRS=data/kg/fhgt_graph/test_pairs.jsonl \
NO_DECOMPOSE=1 \
  poetry run python scripts/eval_answers.py 301
```

`LORA_PATH`로 다른 어댑터를, `LORA_MAX_LEN`으로 절단 길이를 바꿀 수 있다.
기본 448은 학습 시 값이며(질의 토큰 99분위 438 실측), **바꾸면 학습/추론 분포가
어긋나므로 다른 어댑터를 쓸 때만 함께 조정한다.**

## 무엇이 바뀌는가

| | 질의 인코딩 | 조항 임베딩 |
|---|---|---|
| `fhgt_dense` | BGE-M3 → 잔차 QueryEncoder | `node_emb.pt` (동결 BGE) |
| `fhgt_dense_raw` | BGE-M3 | `node_emb.pt` (동결 BGE) |
| `fhgt_dense_lora` | **LoRA를 얹은 BGE-M3** | `node_emb.pt` (동결 BGE) |

세 arm이 **같은 문서 공간**을 쓴다. 어댑터는 질의 측만 학습했으므로 `node_emb.pt`를
다시 만들면 안 된다 — 재생성하면 비교 자체가 무너진다.

`lora_path`를 주면 `use_query_encoder`는 자동으로 꺼진다. 질의 측 학습을 두 겹으로
얹지 않기 위해서다.

## 검증

실행 로그에 다음 두 줄이 보여야 한다.

```
[fhgt] LoRA adapter loaded: data/kg/checkpoints/lora_bge_m3 (max_len=448)
[fhgt] mode=dense | 9311 nodes, emb dim=1024, device=...
```

`answer_metrics.json`의 `run_config`에 `lora_path`와 `lora_max_len`이 기록된다.
