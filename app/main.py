"""
FastAPI 엔트리포인트: PDF 리포트 업로드 -> 분석 -> 질문 응답

실행:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

주의:
    모델 로딩(VLM + LLM)은 서버 시작 시 한 번만 수행한다 (요청마다 로딩하면 매우 느림).
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.pdf_processor import extract_pdf_content, filter_chart_candidate_pages
from app.models.vlm_analyzer import VLMChartAnalyzer
from app.models.llm_synthesizer import LLMSynthesizer
from app.document_collector import collect_documents

app = FastAPI(title="재무 리포트 멀티모달 분석 API")

# 서버 시작 시 모델을 한 번만 로드한다 (전역 상태)
vlm_analyzer: VLMChartAnalyzer | None = None
llm_synthesizer: LLMSynthesizer | None = None


@app.on_event("startup")
def load_models():
    global vlm_analyzer, llm_synthesizer
    print("VLM 로딩 중...")
    vlm_analyzer = VLMChartAnalyzer()
    print("LLM 로딩 중...")
    llm_synthesizer = LLMSynthesizer()
    print("모델 로딩 완료.")


class AnalyzeResponse(BaseModel):
    answer: str
    analyzed_pages: list[int]


class CompanyAnalyzeRequest(BaseModel):
    company_name: str
    question: str


class CompanyAnalyzeResponse(BaseModel):
    answer: str
    sources: list[str]        # 어떤 문서(dart/report)를 사용했는지
    analyzed_pages: dict[str, list[int]]  # 문서별 분석 페이지 번호


def _run_pipeline(pdf_path: str, question: str, tmp_dir: str) -> tuple[str, list[int]]:
    """PDF 경로 하나를 받아 (본문+차트 분석 후) 답변을 생성하는 공통 로직."""
    pages = extract_pdf_content(pdf_path, output_dir=str(Path(tmp_dir) / "pages"))
    chart_candidates = filter_chart_candidate_pages(pages)

    image_paths = [p.image_path for p in chart_candidates]
    vlm_summaries = vlm_analyzer.analyze_pages(image_paths)

    body_text = "\n\n".join(p.text for p in pages if p.text)
    answer = llm_synthesizer.answer(
        body_text=body_text,
        vlm_summaries=vlm_summaries,
        question=question,
    )
    return answer, [p.page_number for p in chart_candidates]


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_report(
    file: UploadFile = File(..., description="분석할 재무 리포트 PDF"),
    question: str = Form(..., description="예: 부채 추이가 어떻게 돼?"),
):
    if vlm_analyzer is None or llm_synthesizer is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로딩 중입니다.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 지원합니다.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = Path(tmp_dir) / file.filename
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        answer, analyzed_pages = _run_pipeline(str(pdf_path), question, tmp_dir)

    return AnalyzeResponse(answer=answer, analyzed_pages=analyzed_pages)


@app.post("/analyze_by_company", response_model=CompanyAnalyzeResponse)
async def analyze_by_company(req: CompanyAnalyzeRequest):
    """회사명을 입력하면 DART 사업보고서 + 증권사 리포트를 자동 수집해 분석한다."""
    if vlm_analyzer is None or llm_synthesizer is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로딩 중입니다.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        documents = collect_documents(req.company_name, output_dir=tmp_dir)

        if not documents:
            raise HTTPException(
                status_code=404,
                detail=f"'{req.company_name}'에 대한 문서를 찾지 못했습니다.",
            )

        # 문서별로 (본문 텍스트, VLM 요약)을 모아서 하나의 컨텍스트로 합친다
        combined_body_text = []
        combined_vlm_summaries = {}
        analyzed_pages_by_source: dict[str, list[int]] = {}

        for doc in documents:
            if doc.content_type == "pdf":
                pages = extract_pdf_content(doc.pdf_path, output_dir=str(Path(tmp_dir) / doc.source))
                chart_candidates = filter_chart_candidate_pages(pages)

                image_paths = [p.image_path for p in chart_candidates]
                vlm_summaries = vlm_analyzer.analyze_pages(image_paths)

                combined_body_text.append(
                    f"[{doc.source}: {doc.title}]\n" + "\n\n".join(p.text for p in pages if p.text)
                )
                combined_vlm_summaries.update(vlm_summaries)
                analyzed_pages_by_source[doc.source] = [p.page_number for p in chart_candidates]
            else:
                # content_type == "text": 이미지/차트 없이 텍스트만 있는 문서
                # (예: dart_financials의 정형 재무 요약, dart의 사업보고서 원문)
                text = doc.text or ""

                if doc.source == "dart":
                    # DART 사업보고서 원문은 매우 길고 지저분해서(표 구조 깨짐, 회사 개요/
                    # 지배구조 등 무관한 내용) LLM에 그대로 다 넣으면 정확한 재무 수치
                    # (dart_financials)를 밀어내거나 헛소리를 유발한다.
                    # 최소한의 맥락(회사 개요 정도)만 남기고 짧게 자른다.
                    RAW_REPORT_MAX_CHARS = 1500
                    if len(text) > RAW_REPORT_MAX_CHARS:
                        text = text[:RAW_REPORT_MAX_CHARS] + "\n...(이하 생략)"

                combined_body_text.append(f"[{doc.source}: {doc.title}]\n{text}")
                analyzed_pages_by_source[doc.source] = []

        answer = llm_synthesizer.answer(
            body_text="\n\n".join(combined_body_text),
            vlm_summaries=combined_vlm_summaries,
            question=req.question,
        )

    return CompanyAnalyzeResponse(
        answer=answer,
        sources=[d.source for d in documents],
        analyzed_pages=analyzed_pages_by_source,
    )


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "vlm_loaded": vlm_analyzer is not None,
        "llm_loaded": llm_synthesizer is not None,
    }


# 웹 UI 정적 파일 서빙. 다른 API 라우트(/analyze_by_company, /health 등)가
# 먼저 매칭되도록, 반드시 그 라우트들을 정의한 뒤 마지막에 마운트해야 한다.
# 루트("/")에 마운트하므로 http://<서버 주소>:8000/ 으로 바로 접속 가능하다.
app.mount("/", StaticFiles(directory="static", html=True), name="ui")