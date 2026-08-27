"""
PDF 리포트를 (1) 본문 텍스트 (2) 페이지 이미지로 분리한다.

- 본문 텍스트: PyMuPDF(fitz)로 바로 추출 (VLM 불필요, 빠르고 정확)
- 페이지 이미지: 각 페이지를 PNG로 렌더링 (차트/표가 있는 페이지를 VLM에 넣기 위함)

설치:
    pip install pymupdf pillow --break-system-packages
"""

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class PageContent:
    page_number: int          # 1부터 시작
    text: str                 # 해당 페이지의 본문 텍스트
    image_path: str           # 렌더링된 페이지 이미지 경로
    has_visual_content: bool  # 이미지/그래픽 객체가 있는지 여부 (차트 후보 판단용)


def extract_pdf_content(pdf_path: str, output_dir: str, dpi: int = 200) -> list[PageContent]:
    """
    PDF를 페이지 단위로 분해한다.

    Args:
        pdf_path: 입력 PDF 경로
        output_dir: 페이지 이미지를 저장할 디렉토리
        dpi: 렌더링 해상도 (차트 디테일이 필요하면 200~300 권장)

    Returns:
        페이지별 PageContent 리스트
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    pdf_stem = Path(pdf_path).stem
    results: list[PageContent] = []

    zoom = dpi / 72  # PDF 기본 72 DPI 기준 배율
    matrix = fitz.Matrix(zoom, zoom)

    for i, page in enumerate(doc):
        page_number = i + 1

        # 1) 본문 텍스트 추출
        text = page.get_text().strip()

        # 2) 페이지를 이미지로 렌더링
        pix = page.get_pixmap(matrix=matrix)
        image_path = out_dir / f"{pdf_stem}_page{page_number:03d}.png"
        pix.save(str(image_path))

        # 3) 이 페이지에 이미지/그래픽 객체가 있는지 (차트 후보 판단용 힌트)
        #    - 삽입된 이미지가 있거나
        #    - 벡터 드로잉(선/도형) 개수가 일정 이상이면 차트일 가능성이 높음
        has_embedded_image = len(page.get_images()) > 0
        drawing_count = len(page.get_drawings())
        has_visual_content = has_embedded_image or drawing_count > 10

        results.append(
            PageContent(
                page_number=page_number,
                text=text,
                image_path=str(image_path),
                has_visual_content=has_visual_content,
            )
        )

    doc.close()
    return results


def filter_chart_candidate_pages(pages: list[PageContent]) -> list[PageContent]:
    """차트/표가 있을 것으로 추정되는 페이지만 필터링한다 (VLM에 넣을 대상)."""
    return [p for p in pages if p.has_visual_content]


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="분석할 PDF 경로")
    parser.add_argument("--out", default="./output_pages", help="페이지 이미지 저장 디렉토리")
    args = parser.parse_args()

    pages = extract_pdf_content(args.pdf, args.out)
    candidates = filter_chart_candidate_pages(pages)

    print(f"전체 페이지: {len(pages)}장")
    print(f"차트/표 후보 페이지: {len(candidates)}장 → {[p.page_number for p in candidates]}")

    summary = [
        {
            "page": p.page_number,
            "text_length": len(p.text),
            "has_visual_content": p.has_visual_content,
            "image_path": p.image_path,
        }
        for p in pages
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
