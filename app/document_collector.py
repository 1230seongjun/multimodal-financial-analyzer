"""
회사명을 입력받아 (1) DART 사업보고서 (2) 증권사 리포트를 실시간으로 수집해
PDF 파일 경로 리스트로 반환한다.

설치:
    pip install requests beautifulsoup4 --break-system-packages

환경변수:
    DART_API_KEY - https://opendart.fss.or.kr 에서 발급받은 API 키
"""

import io
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

DART_API_KEY = os.environ.get("DART_API_KEY", "")
DART_BASE_URL = "https://opendart.fss.or.kr/api"


@dataclass
class CollectedDocument:
    source: str          # "dart" 또는 "report"
    title: str
    content_type: str    # "pdf" (VLM+텍스트 파이프라인 대상) 또는 "text" (텍스트만, VLM 불필요)
    pdf_path: str | None = None   # content_type == "pdf"일 때 사용
    text: str | None = None       # content_type == "text"일 때 사용


# ---------------------------------------------------------------------------
# 1) DART 사업보고서 수집
# ---------------------------------------------------------------------------

def _get_corp_code(company_name: str) -> str | None:
    """
    DART는 회사 고유번호(corp_code)로 조회하므로, 전체 회사 목록 XML에서
    이름이 일치하는 corp_code를 찾는다.

    전체 목록은 자주 바뀌지 않으므로 실제 서비스에서는 캐싱을 권장한다.
    """
    if not DART_API_KEY:
        raise RuntimeError("DART_API_KEY 환경변수가 설정되지 않았습니다.")

    resp = requests.get(
        f"{DART_BASE_URL}/corpCode.xml",
        params={"crtfc_key": DART_API_KEY},
        timeout=10,
    )
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        xml_bytes = zf.read("CORPCODE.xml")

    soup = BeautifulSoup(xml_bytes, "lxml-xml")
    for item in soup.find_all("list"):
        name = item.find("corp_name").text.strip()
        if name == company_name:
            return item.find("corp_code").text.strip()

    return None


def _format_krw(amount_str: str) -> str:
    """
    DART가 주는 원 단위 숫자 문자열(예: "136230000000000")을
    사람이 읽기 쉬운 "OO조 OO억원" 형태로 변환한다.
    LLM이 직접 단위를 계산하게 두면 잘못된 단위 변환을 하는 경우가 많아, 코드에서 미리 변환한다.
    """
    try:
        amount = int(str(amount_str).replace(",", "").strip())
    except (TypeError, ValueError):
        return str(amount_str) if amount_str else "정보 없음"

    is_negative = amount < 0
    amount = abs(amount)

    trillion, remainder = divmod(amount, 1_0000_0000_0000)  # 1조 = 1e12원
    eok = remainder // 1_0000_0000  # 1억 = 1e8원

    parts = []
    if trillion:
        parts.append(f"{trillion}조")
    if eok:
        parts.append(f"{eok}억")
    if not parts:
        parts.append(f"{amount:,}")

    result = " ".join(parts) + "원"
    return f"-{result}" if is_negative else result


def fetch_dart_financial_summary(company_name: str) -> CollectedDocument | None:
    """
    DART의 단일회사 주요계정 API(fnlttSinglAcnt.json)로 최근 사업연도의
    부채총계/자산총계/자본총계 등 핵심 재무 수치를 정형 데이터로 받아온다.

    거대한 사업보고서 원문(HTML/XML)을 통째로 파싱해 키워드로 자르는 방식은
    표 구조가 깨지거나 인코딩이 섞여 LLM이 엉뚱한 답을 만드는 문제가 있었다.
    이 API는 계정명/당기/전기/전전기 금액을 이미 깔끔하게 정리해서 주므로
    훨씬 안정적이다.
    """
    corp_code = _get_corp_code(company_name)
    if corp_code is None:
        print(f"[DART 재무정보] '{company_name}'에 해당하는 corp_code를 찾지 못했습니다.")
        return None

    from datetime import datetime

    # 올해 사업보고서는 아직 제출 전일 수 있으므로, 가장 최근 확정된 사업연도(작년)를 조회한다.
    target_year = str(datetime.now().year - 1)

    resp = requests.get(
        f"{DART_BASE_URL}/fnlttSinglAcnt.json",
        params={
            "crtfc_key": DART_API_KEY,
            "corp_code": corp_code,
            "bsns_year": target_year,
            "reprt_code": "11011",  # 사업보고서(연간)
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "000" or not data.get("list"):
        print(f"[DART 재무정보] '{company_name}' 재무정보를 찾지 못했습니다: {data.get('message')}")
        return None

    interesting_accounts = ("부채총계", "자산총계", "자본총계", "매출액", "영업이익", "당기순이익")

    # 실제 연도 숫자를 명시해서 모델이 "21년", "23년" 같은 임의의 연도를 지어내지
    # 못하도록 한다. thstrm(당기)=target_year, frmtrm(전기)=1년 전, bfefrmtrm(전전기)=2년 전.
    target_year_int = int(target_year)
    year_labels = {
        "당기": f"{target_year_int}년",
        "전기": f"{target_year_int - 1}년",
        "전전기": f"{target_year_int - 2}년",
    }

    # DART는 연결재무제표(CFS, 계열사 포함 전체)와 별도재무제표(OFS, 회사 단독)를
    # 같은 계정명으로 함께 내려준다. 구분 없이 다 넣으면 "부채총계"가 두 번 다른
    # 숫자로 나와 LLM이 혼란스러워하며 엉뚱한 내용을 지어내는 문제가 있었다.
    # 일반적으로 "회사의 부채/실적"이라고 물을 때 기대하는 연결재무제표(CFS)만 사용한다.
    seen_accounts = set()
    lines = []
    for item in data["list"]:
        if item.get("fs_div") != "CFS":  # CFS: 연결재무제표만 사용, OFS(별도)는 제외
            continue

        account = (item.get("account_nm") or "").strip()
        if account in seen_accounts:
            continue  # 같은 계정이 중복으로 오는 경우(재무상태표/손익계산서 등 중복 집계) 방지
        if any(keyword in account for keyword in interesting_accounts):
            lines.append(
                f"{account}: {year_labels['당기']} {_format_krw(item.get('thstrm_amount'))}, "
                f"{year_labels['전기']} {_format_krw(item.get('frmtrm_amount'))}, "
                f"{year_labels['전전기']} {_format_krw(item.get('bfefrmtrm_amount'))}"
            )
            seen_accounts.add(account)

    if not lines:
        print(f"[DART 재무정보] '{company_name}' 응답에서 주요 계정을 찾지 못했습니다.")
        return None

    summary_text = (
        f"[{target_year}년 사업보고서 기준 연결재무제표 주요 수치 (DART 공식 데이터, 단위: 조/억원)]\n"
        + "\n".join(lines)
    )

    return CollectedDocument(
        source="dart_financials",
        title=f"{target_year}년 주요 재무 수치",
        content_type="text",
        text=summary_text,
    )


def fetch_dart_report(company_name: str, output_dir: str) -> CollectedDocument | None:
    """회사명으로 최신 사업보고서 원문 PDF를 받아온다."""
    corp_code = _get_corp_code(company_name)
    if corp_code is None:
        print(f"[DART] '{company_name}'에 해당하는 corp_code를 찾지 못했습니다.")
        return None

    # 최근 공시 목록에서 사업보고서(A001) 문서 찾기
    # bgn_de/end_de를 명시하지 않으면 조회 범위가 좁아 결과가 비는 경우가 많으므로
    # 최근 3년치를 명시적으로 조회한다.
    from datetime import datetime, timedelta

    end_de = datetime.now().strftime("%Y%m%d")
    bgn_de = (datetime.now() - timedelta(days=3 * 365)).strftime("%Y%m%d")

    resp = requests.get(
        f"{DART_BASE_URL}/list.json",
        params={
            "crtfc_key": DART_API_KEY,
            "corp_code": corp_code,
            "bgn_de": bgn_de,
            "end_de": end_de,
            "pblntf_detail_ty": "A001",  # 사업보고서
            "sort": "date",
            "sort_mth": "desc",
            "page_count": 10,
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "000" or not data.get("list"):
        print(f"[DART] '{company_name}' 사업보고서를 찾지 못했습니다: {data.get('message')}")
        return None

    latest = data["list"][0]
    rcept_no = latest["rcept_no"]

    # 공시 원문(ZIP)을 받는다. 대부분 HTML 형식이고, 드물게 PDF가 포함된 경우도 있다.
    doc_resp = requests.get(
        f"{DART_BASE_URL}/document.xml",
        params={"crtfc_key": DART_API_KEY, "rcept_no": rcept_no},
        timeout=30,
    )
    doc_resp.raise_for_status()

    os.makedirs(output_dir, exist_ok=True)
    title = latest.get("report_nm", "사업보고서")

    with zipfile.ZipFile(io.BytesIO(doc_resp.content)) as zf:
        pdf_names = [n for n in zf.namelist() if n.lower().endswith(".pdf")]

        if pdf_names:
            # PDF가 있는 경우: 기존처럼 PDF 파이프라인(VLM 포함)으로 처리
            pdf_path = os.path.join(output_dir, f"dart_{company_name}_{rcept_no}.pdf")
            with open(pdf_path, "wb") as f:
                f.write(zf.read(pdf_names[0]))
            return CollectedDocument(
                source="dart", title=title, content_type="pdf", pdf_path=pdf_path
            )

        # PDF가 없는 경우(대부분): HTML/XML 원문에서 텍스트만 추출한다.
        # DART 원문은 여러 개의 html 조각 파일로 구성되며, 보통 회사 개요/사업 내용이
        # 앞부분에, 재무제표(부채/자산 등 숫자)는 뒷부분에 위치한다.
        # 뒤에서 잘려나가는 것을 막기 위해, 재무 관련 키워드가 있는 조각을 앞으로 정렬한다.
        html_names = [n for n in zf.namelist() if n.lower().endswith((".html", ".htm", ".xml"))]
        if not html_names:
            print(f"[DART] '{company_name}' 원문에서 텍스트를 추출할 파일을 찾지 못했습니다.")
            return None

        FINANCE_KEYWORDS = ("재무상태표", "손익계산서", "재무제표", "부채총계", "자본총계", "현금흐름표")

        finance_parts = []
        other_parts = []
        for name in html_names:
            raw = zf.read(name)
            soup = BeautifulSoup(raw, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            if any(keyword in text for keyword in FINANCE_KEYWORDS):
                finance_parts.append(text)
            else:
                other_parts.append(text)

        # 재무 관련 조각을 먼저 배치해, 뒤이은 길이 제한(truncation)에서 살아남게 한다.
        combined_text_parts = finance_parts + other_parts

    combined_text = "\n\n".join(combined_text_parts)
    return CollectedDocument(
        source="dart", title=title, content_type="text", text=combined_text
    )


# ---------------------------------------------------------------------------
# 2) 증권사 리포트 크롤링 (네이버 금융)
# ---------------------------------------------------------------------------

NAVER_RESEARCH_URL = "https://finance.naver.com/research/company_list.naver"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_naver_research_report(company_name: str, output_dir: str) -> CollectedDocument | None:
    """네이버 금융 종목분석 리포트 목록에서 회사명으로 검색해 최신 리포트 PDF를 받아온다."""
    resp = requests.get(
        NAVER_RESEARCH_URL,
        params={"searchType": "keyword", "keyword": company_name},
        headers=HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    resp.encoding = "euc-kr"

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = soup.select("table.type_1 tr")

    pdf_link = None
    title = None
    for row in rows:
        link_tag = row.find("a", href=re.compile(r"\.pdf$"))
        if link_tag:
            pdf_link = link_tag["href"]
            title = row.get_text(strip=True)
            break

    if pdf_link is None:
        print(f"[증권사 리포트] '{company_name}' 리포트를 찾지 못했습니다.")
        return None

    pdf_resp = requests.get(pdf_link, headers=HEADERS, timeout=30)
    pdf_resp.raise_for_status()

    os.makedirs(output_dir, exist_ok=True)
    safe_name = re.sub(r"[^\w]", "_", company_name)
    pdf_path = os.path.join(output_dir, f"report_{safe_name}.pdf")
    with open(pdf_path, "wb") as f:
        f.write(pdf_resp.content)

    return CollectedDocument(source="report", title=title or "증권사 리포트", content_type="pdf", pdf_path=pdf_path)


# ---------------------------------------------------------------------------
# 통합 수집 함수
# ---------------------------------------------------------------------------

def collect_documents(company_name: str, output_dir: str | None = None) -> list[CollectedDocument]:
    """DART 재무 수치 + DART 사업보고서 원문 + 증권사 리포트를 모두 시도해서 성공한 것만 반환한다."""
    output_dir = output_dir or tempfile.mkdtemp(prefix="collected_")

    documents: list[CollectedDocument] = []

    # 가장 신뢰도 높고 깔끔한 소스(정형 재무 수치)를 먼저 시도하고, 리스트 맨 앞에 둔다.
    # (뒤에서 텍스트를 자를 때 이 부분이 가장 먼저 살아남도록 하기 위함)
    try:
        financial_doc = fetch_dart_financial_summary(company_name)
        if financial_doc:
            documents.append(financial_doc)
    except Exception as e:
        print(f"[DART 재무정보] 수집 실패: {e}")

    try:
        dart_doc = fetch_dart_report(company_name, output_dir)
        if dart_doc:
            documents.append(dart_doc)
    except Exception as e:
        print(f"[DART] 수집 실패: {e}")

    try:
        report_doc = fetch_naver_research_report(company_name, output_dir)
        if report_doc:
            documents.append(report_doc)
    except Exception as e:
        print(f"[증권사 리포트] 수집 실패: {e}")

    return documents


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--company", required=True, help="예: 삼성전자")
    args = parser.parse_args()

    docs = collect_documents(args.company)
    print(json.dumps(
        [
            {
                "source": d.source,
                "title": d.title,
                "content_type": d.content_type,
                "pdf_path": d.pdf_path,
                "text_preview": (d.text[:200] + "...") if d.text else None,
            }
            for d in docs
        ],
        ensure_ascii=False,
        indent=2,
    ))