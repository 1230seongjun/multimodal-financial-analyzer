"""
LLM(Qwen2.5-7B-Instruct)으로 본문 텍스트 + VLM 차트/표 분석 결과를 종합해
사용자 질문에 답하는 최종 재무 판단 텍스트를 생성한다.

설치:
    pip install torch transformers accelerate bitsandbytes --break-system-packages
"""

import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessor, LogitsProcessorList


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# 이전에는 본문 텍스트가 12000~16000자로 길어서 7B 모델이 OOM을 일으켰다.
# 이제 DART 원문을 1500자로 강하게 제한하고 정형 재무 요약 위주로 구성해
# 실제 입력 길이가 훨씬 짧아졌으므로(수천 자 이내), 7B로 복귀해 지침 준수
# 능력(한국어만 사용 등)을 개선한다. 그래도 안전 마진을 위해 제한은 유지한다.
MAX_BODY_TEXT_CHARS = 10000

SYNTHESIS_SYSTEM_PROMPT = """당신은 엄격한 재무 데이터 추출 보조 도구입니다.
아래 제공된 [리포트 본문]과 [차트/표 분석 결과]에 있는 내용만 사용하여 질문에 답하세요.

[절대 지켜야 할 규칙]
1. 없는 사실 생성 금지: 제공된 자료에 없는 기업명, 원인, 배경 설명은 절대 지어내지 마세요.
2. 단위 및 수치 엄수: 자료에 표기된 숫자와 단위(원, 조, 억원 등)를 그대로 복사해서 사용하세요. 달러 변환 등 임의로 계산하거나 단위를 바꾸지 마세요.
3. 연도 표기: 자료에 있는 연도(예: "2025년")를 정확히 그대로 기재하세요. 쉼표를 넣거나 다른 방식으로 풀어 쓰지 마세요.
4. 객관적 서술: 자료의 수치 변화(증가/감소액 등)만 건조하게 서술하고, 경영 결정이나 재무 건강에 대한 자의적 평가는 넣지 마세요.
5. 이 답변은 참고용 분석이며 투자 조언이 아닙니다. 단정적인 투자 권유 표현은 피하세요.
6. 반드시 한국어로만, 인사말 없이 결론만 간결하게 작성하세요.
"""


class _BanCJKLogitsProcessor(LogitsProcessor):
    """
    한자(중국어/일본어 한자, 유니코드 U+4E00~U+9FFF)가 포함된 토큰의 확률을
    -inf로 만들어 생성 자체를 원천 차단한다. 한글(Hangul, U+AC00~U+D7A3)은
    완전히 다른 유니코드 영역이라 한국어 출력에는 영향이 없다.
    """

    def __init__(self, banned_token_ids: torch.Tensor):
        self.banned_token_ids = banned_token_ids

    def __call__(self, input_ids, scores):
        scores[:, self.banned_token_ids] = -float("inf")
        return scores


class LLMSynthesizer:
    """
    VLM과 마찬가지로, T4 16GB 환경에서 두 모델을 동시에 GPU에 올리면 메모리가
    부족하므로 필요할 때만 로드하고 사용 후 언로드하는 방식을 쓴다.
    """

    def __init__(self, model_name: str = MODEL_NAME, load_in_4bit: bool = True):
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.model = None
        # tokenizer는 가볍고 GPU를 쓰지 않으므로 항상 유지한다.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 한자 포함 토큰 ID 목록을 미리 계산해둔다 (모델은 요청마다 언로드/재로드되지만
        # 토크나이저는 계속 유지되므로, 이 계산은 서버 시작 시 한 번만 수행하면 된다).
        self._cjk_banned_ids = self._compute_cjk_banned_ids()

    def _compute_cjk_banned_ids(self) -> torch.Tensor:
        banned = []
        for token_id in range(self.tokenizer.vocab_size):
            token_str = self.tokenizer.decode([token_id])
            if any("\u4e00" <= ch <= "\u9fff" for ch in token_str):
                banned.append(token_id)
        return torch.tensor(banned, dtype=torch.long)

    def _ensure_loaded(self):
        if self.model is None:
            quantization_config = (
                BitsAndBytesConfig(load_in_4bit=True) if self.load_in_4bit else None
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=quantization_config,
            )

    def unload(self):
        """GPU 메모리를 비워야 할 때(VLM 추론 직전 등) 호출한다."""
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()

    # 본문에서 재무제표 관련 내용이 처음 등장하는 지점을 찾기 위한 키워드.
    # DART 원문이 하나의 거대한 문서로 통째로 들어있는 경우가 많아, 단순히 앞에서부터
    # 자르면 회사 개요/사업 내용만 남고 정작 필요한 숫자(재무제표)는 잘려나간다.
    _FINANCE_SECTION_KEYWORDS = (
        "재무상태표", "손익계산서", "부채총계", "자본총계", "현금흐름표", "재무제표"
    )

    def _build_context(self, body_text: str, vlm_summaries: dict[str, str]) -> str:
        if len(body_text) > MAX_BODY_TEXT_CHARS:
            # 재무 키워드가 처음 등장하는 위치를 찾아, 그 지점(약간 앞부터)부터 자른다.
            earliest_idx = -1
            for keyword in self._FINANCE_SECTION_KEYWORDS:
                pos = body_text.find(keyword)
                if pos != -1 and (earliest_idx == -1 or pos < earliest_idx):
                    earliest_idx = pos

            if earliest_idx != -1:
                start = max(0, earliest_idx - 300)  # 표 제목 등 문맥을 위해 약간 앞부터 포함
                body_text = (
                    "...(앞부분 생략)...\n"
                    + body_text[start:start + MAX_BODY_TEXT_CHARS]
                )
            else:
                # 재무 키워드를 못 찾으면 기존처럼 앞에서부터 자른다.
                body_text = body_text[:MAX_BODY_TEXT_CHARS] + "\n\n...(본문이 길어 이하 생략됨)"

        chart_section = "\n\n".join(
            f"[페이지 이미지: {path}]\n{summary}"
            for path, summary in vlm_summaries.items()
        )
        return (
            f"--- 리포트 본문 텍스트 ---\n{body_text}\n\n"
            f"--- 차트/표 분석 결과 (VLM) ---\n{chart_section}"
        )

    def answer(self, body_text: str, vlm_summaries: dict[str, str], question: str, unload_after: bool = True) -> str:
        """
        Args:
            body_text: PDF에서 추출한 본문 텍스트 (여러 페이지 합친 것)
            vlm_summaries: {페이지 이미지 경로: VLM 분석 요약} 딕셔너리
            question: 사용자 질문 (예: "부채 추이가 어떻게 돼?")
            unload_after: 답변 생성 후 GPU 메모리를 비울지 여부
        """
        self._ensure_loaded()
        context = self._build_context(body_text, vlm_summaries)

        # 디버그용: 실제로 모델에 어떤 내용이 들어가는지 서버 로그로 확인한다.
        # (모델이 헛소리를 하는 게 프롬프트/데이터 문제인지, 모델 자체의 한계인지 구분하기 위함)
        print("=" * 60)
        print("[LLM 입력 컨텍스트 (디버그)]")
        print(context)
        print("=" * 60)

        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\n--- 질문 ---\n{question}"},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # T4(Turing 아키텍처)는 Flash Attention을 지원하지 않으므로 PyTorch가 자동으로
        # 적절한(대개 math) 커널로 폴백한다. Qwen은 GQA 구조라 특정 커널을 강제하면
        # "No available kernel" 에러가 날 수 있으므로 강제하지 않고 자동 선택에 맡긴다.
        #
        # repetition_penalty/no_repeat_ngram_size는 한국어처럼 음절 단위 토큰화가
        # 촘촘한 언어에서, 이미 나온 토큰을 억지로 피하게 하다가 전혀 다른 글자로
        # 튀는 부작용("증가"의 "증"이 페널티를 받아 "즧" 같은 존재하지 않는 글자로
        # 대체되는 현상)이 있어 완전히 껐다. 대신 무한 반복 루프가 재발하면
        # repetition_penalty를 1.0보다 살짝 높은 값(1.05~1.1)으로 절충한다.
        logits_processor = LogitsProcessorList([
            _BanCJKLogitsProcessor(self._cjk_banned_ids.to(self.model.device))
        ])

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=400,
            repetition_penalty=1.0,  # 완전 비활성화
            do_sample=False,
            logits_processor=logits_processor,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )[0]

        if unload_after:
            self.unload()

        return output_text.strip()