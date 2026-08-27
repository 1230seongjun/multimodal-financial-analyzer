"""
VLM(Qwen2-VL-2B)으로 차트/표 이미지를 분석해 자연어 요약을 생성한다.

역할: 이미지 -> 텍스트 요약 (숫자 추출 + 추세 서술)만 담당한다.
      최종 재무 판단은 LLM(llm_synthesizer.py)이 담당한다.

설치:
    pip install torch transformers accelerate bitsandbytes qwen-vl-utils pillow --break-system-packages
"""

import gc

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info


MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

CHART_ANALYSIS_PROMPT = """이 이미지는 증권사 리포트 또는 IR 자료의 한 페이지입니다.
페이지 안에 있는 차트(막대그래프, 선그래프 등)나 표를 찾아 다음을 서술하세요.

1. 어떤 지표를 나타내는 차트/표인지 (예: 매출액, 영업이익, 부채비율 등)
2. 표/차트 근처에 표기된 단위(예: "단위: 원", "단위: 백만원", "단위: 억원")를 반드시 확인해서 명시
3. 연도/분기별 수치를 가능한 한 구체적으로 (축의 눈금과 라벨을 참고하여 추정)
4. 전체적인 추세 (증가/감소/변동)

차트나 표가 없는 페이지라면 "분석 대상 차트/표 없음"이라고만 답하세요.
숫자를 추정할 때는 실제 라벨/눈금에 근거하고, 근거 없는 숫자를 지어내지 마세요.
단위를 찾지 못했다면 "단위 확인 불가"라고 명시하세요 (임의로 단위를 지어내지 마세요).
"""


class VLMChartAnalyzer:
    """
    GPU 메모리가 빠듯한 환경(T4 16GB 등)을 고려해, 모델을 항상 GPU에 올려두지 않는다.
    필요할 때(_ensure_loaded)만 로드하고, 사용 후(unload) 메모리에서 내린다.
    (bitsandbytes 4bit 모델은 .to()로 device 이동이 불가능하므로 로드/언로드 방식을 사용한다.)
    """

    def __init__(self, model_name: str = MODEL_NAME, load_in_4bit: bool = True):
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.model = None
        # processor는 가볍고 GPU를 거의 쓰지 않으므로 항상 유지한다.
        self.processor = AutoProcessor.from_pretrained(model_name)

    def _ensure_loaded(self):
        if self.model is None:
            quantization_config = (
                BitsAndBytesConfig(load_in_4bit=True) if self.load_in_4bit else None
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=quantization_config,
            )

    def unload(self):
        """GPU 메모리를 비워야 할 때(LLM 추론 직전 등) 호출한다."""
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()

    def analyze(self, image_path: str, custom_prompt: str | None = None) -> str:
        """이미지 하나를 분석해 자연어 요약 텍스트를 반환한다."""
        self._ensure_loaded()
        prompt = custom_prompt or CHART_ANALYSIS_PROMPT

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # T4는 Flash Attention 미지원이라 PyTorch가 자동으로 적절한 커널로 폴백한다.
        # 특정 커널을 강제하면 GQA 구조에서 "No available kernel" 에러가 날 수 있으므로
        # 강제하지 않는다.
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=False,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text.strip()

    def analyze_pages(self, image_paths: list[str], unload_after: bool = True) -> dict[str, str]:
        """
        여러 페이지 이미지를 순차적으로 분석해 {경로: 요약} 딕셔너리를 반환한다.
        unload_after=True면 분석이 끝난 뒤 GPU 메모리를 비운다 (LLM 실행 공간 확보용).
        """
        results = {}
        for path in image_paths:
            results[path] = self.analyze(path)
        if unload_after:
            self.unload()
        return results