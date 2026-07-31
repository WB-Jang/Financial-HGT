"""
lora/model.py

BGE-M3에 LoRA 어댑터를 붙여 '질의 측만' 학습하기 위한 최소 래퍼.

설계 원칙 (기존 QueryEncoder(MLP)와의 대칭성):
- 문서(조항) 임베딩은 절대 건드리지 않는다. emb_cache/clause_embs_*.safetensors를
  그대로 재사용하므로, MLP arm과 LoRA arm이 '완전히 같은 조항 인덱스'를 공유한다.
- LoRA 기본 초기화는 A ~ 랜덤 / B = 0 이므로 B·A = 0, 즉 W_eff = W_0 이다.
  학습 시작 시점의 모델은 원본 BGE-M3와 '같은 함수'이고, 따라서 질의 임베딩도
  순수 BGE 임베딩과 동일하다. QueryEncoder가 마지막 Linear를 0으로 초기화해
  out == normalize(x)로 시작하는 것과 정확히 같은 구조다.
  (자세한 유도는 lora/README.md 참조)

풀링: 조항 임베딩은 SentenceTransformer('BAAI/bge-m3')가 만들었고 여기서는
AutoModel을 직접 쓰므로, 같은 공간에 놓이려면 CLS 풀링 + L2 정규화를 정확히
재현해야 한다. train_lora.py가 학습 전에 이를 코사인으로 검증한다(게이트 ①).
"""

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer

MODEL_ID = 'BAAI/bge-m3'


def build_lora_bge(r=16, alpha=32, dropout=0.1, device=None, dtype=torch.float32):
    """LoRA 어댑터를 붙인 BGE-M3와 토크나이저를 반환.

    target_modules=["query","value"]는 LoRA 원논문의 기본 선택.
    r=16 기준 학습 파라미터는 24층 x 2모듈 x 2 x 1024 x 16 = 1,572,864개로
    QueryEncoder(MLP)의 1,051,136개와 같은 자릿수다.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModel.from_pretrained(MODEL_ID, torch_dtype=dtype)

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["query", "value"],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(base, cfg).to(device)
    model.print_trainable_parameters()
    return model, tok


def encode_queries_lora(model, tok, texts, max_len=256, batch_size=8, device=None,
                        grad=False, amp=False):
    """질의 텍스트 -> (N, 1024) L2 정규화 임베딩.

    CLS 풀링(last_hidden_state[:, 0, :]) + L2 정규화 — SentenceTransformer의
    bge-m3 파이프라인과 동일한 처리다.

    grad=True면 학습용(그래프 유지), False면 추론용(no_grad).
    amp=True면 fp16 autocast (T4에서 학습 시 사용. T4는 Turing이라 bf16 미지원).
    """
    device = device or next(model.parameters()).device
    outs = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for i in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[i:i + batch_size]]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=max_len, return_tensors='pt').to(device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                out = model(**enc).last_hidden_state[:, 0, :]
            outs.append(F.normalize(out.float(), dim=-1))
    return torch.cat(outs, dim=0)
