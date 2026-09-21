"""추출 로직.

이 파일은 고칠 일이 거의 없습니다. 라벨·서류종류·확인번호는 전부
patterns.py 에 있습니다.

핵심 원칙: **못 뽑으면 빈 값으로 둡니다.** 추측해서 채우지 않습니다.
틀린 값이 화면에 떠 있으면 실무자가 그걸 믿고 넘어갈 수 있는데,
빈 값이면 반드시 직접 확인하게 됩니다.
"""


from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

from patterns import (
    DOC_TITLES,
    FIELD_BY_KEY,
    VERIFY_SPECS,
    schema_for,
)

MIN_TEXT_CHARS = 30
PDF_MAGIC = b"%PDF-"


@dataclass
class Value:
    key: str
    display: str
    value: str | None = None
    raw_line: str = ""        # 어느 줄에서 뽑았는지 — 실무자 대조용

    @property
    def found(self) -> bool:
        return bool(self.value)


@dataclass
class Result:
    path: str
    kind: str = "unreadable"      # born_digital / scanned / empty / unreadable
    doc_type: str | None = None
    text: str = ""
    text_chars: int = 0
    values: dict = field(default_factory=dict)
    verify_numbers: list = field(default_factory=list)
    problems: list = field(default_factory=list)

    @property
    def missing(self):
        return [k for k, v in self.values.items() if not v.found]

    @property
    def missing_required(self):
        return [k for k in self.missing if FIELD_BY_KEY[k].required]

    @property
    def ready(self) -> bool:
        return not self.missing_required


def normalize(text: str) -> str:
    """전각 공백·기호만 정리. 자간은 건드리지 않습니다.

    여기서 공백을 다 없애면 값(소속명 등)의 단어 경계까지 사라집니다.
    자간은 patterns.spaced() 가 정규식 쪽에서 흡수합니다.
    """
    text = text.replace("\u3000", " ").replace("：", ":")
    text = re.sub(r"[ \t]+", " ", text)
    return "\n".join(line.strip() for line in text.split("\n"))


def find_doc_type(text: str) -> str | None:
    """서류 제목. 제목도 자간이 벌어지므로 공백을 지우고 비교합니다."""
    flat_head = re.sub(r"\s+", "", "\n".join(text.split("\n")[:12]))
    flat_all = re.sub(r"\s+", "", text)
    for scope in (flat_head, flat_all):
        best = None
        for canonical, aliases in DOC_TITLES.items():
            for alias in aliases:
                flat = re.sub(r"\s+", "", alias)
                if flat in scope and (best is None or len(flat) > best[0]):
                    # 긴 별칭이 더 구체적 (사망진단서 > 진단서)
                    best = (len(flat), canonical)
        if best:
            return best[1]
    return None


def find_bare_issue_date(text: str) -> str | None:
    """발급일 라벨이 없는 서류 보정.

    대학·관공서 증명서는 '발급일자:' 없이 본문 끝에 날짜만 찍습니다.

        위의 내용을 증명합니다.
              2023년 06월 23일
            대구가톨릭대학교

    종결 문구 뒤에 오는 날짜를 발급일로 봅니다.
    """
    lines = text.split("\n")
    closing = re.compile(r"(증명합니다|증명함|확인합니다|확인함|위와\s*같이)")
    date_pat = re.compile(r"(\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일)")
    for i, line in enumerate(lines):
        if closing.search(line):
            for nxt in lines[i:i + 6]:
                m = date_pat.search(nxt)
                if m:
                    return re.sub(r"\s+", " ", m.group(1)).strip()
    return None


def extract(path: str | Path) -> Result:
    p = Path(path)
    out = Result(path=str(p))

    # --- 파일 검사 ---
    if not p.exists():
        out.problems.append("파일 없음")
        return out
    if p.stat().st_size == 0:
        out.kind = "empty"
        out.problems.append("빈 파일 (0 bytes)")
        return out
    with open(p, "rb") as f:
        if f.read(5) != PDF_MAGIC:
            out.problems.append("PDF 형식이 아님 — 확장자만 .pdf")
            return out

    # --- 텍스트 추출 ---
    try:
        with pdfplumber.open(str(p)) as pdf:
            pages = [(pg.extract_text() or "") for pg in pdf.pages]
    except Exception as e:  # noqa: BLE001
        out.problems.append(f"PDF 열기 실패: {e}")
        return out

    out.text = normalize("\n".join(pages))
    out.text_chars = len(out.text.strip())

    if out.text_chars < MIN_TEXT_CHARS:
        out.kind = "scanned"
        out.problems.append("텍스트 레이어 없음 — 스캔본. OCR 또는 LLM 필요")
        return out

    out.kind = "born_digital"

    # --- 서류 종류 ---
    out.doc_type = find_doc_type(out.text)
    if not out.doc_type:
        out.problems.append(
            "서류 종류 판별 실패 — patterns.py 의 DOC_TITLES 에 제목 추가 필요")

    # --- 확인번호 ---
    for spec in VERIFY_SPECS:
        m = spec.pattern.search(out.text)
        if m:
            out.verify_numbers.append((spec.name, m.group(1), spec.note))

    # --- 필드 ---
    for key in schema_for(out.doc_type):
        spec = FIELD_BY_KEY[key]
        v = Value(key, spec.display)
        m = spec.regex().search(out.text)
        if m:
            v.value = re.sub(r"\s+", " ", m.group(1)).strip()
            start = out.text.rfind("\n", 0, m.start()) + 1
            end = out.text.find("\n", m.end())
            v.raw_line = out.text[start:end if end != -1 else len(out.text)].strip()
        out.values[key] = v

    # 발급일 라벨이 없는 경우 보정
    if "issue_date" in out.values and not out.values["issue_date"].found:
        bare = find_bare_issue_date(out.text)
        if bare:
            out.values["issue_date"].value = bare
            out.values["issue_date"].raw_line = "(종결문구 뒤 날짜)"

    return out


# ---------------------------------------------------------------------
# 진단 — 못 뽑았을 때 patterns.py 에 뭘 추가해야 하는지 알려줍니다
# ---------------------------------------------------------------------


def diagnose(out: Result) -> list[str]:
    """실패한 필드에 대해, 원문에서 라벨처럼 보이는 줄을 찾아 보여줍니다.

    실무자가 PDF 를 열어보지 않고도 "아, 이 서류는 라벨을 이렇게 쓰는구나"
    를 알 수 있게 해서, patterns.py 수정을 쉽게 만드는 게 목적입니다.
    """
    if out.kind != "born_digital" or not out.missing:
        return []

    # 원문에서 "라벨 : 값" 형태인 줄을 전부 모읍니다
    labelled = []
    for line in out.text.split("\n"):
        if ":" in line and len(line) < 80:
            label = line.split(":", 1)[0].strip()
            if 1 <= len(label.replace(" ", "")) <= 12:
                labelled.append((label, line.strip()))

    # 이미 잡힌 줄, 그리고 patterns.py 에 이미 등록된 라벨은 제외합니다.
    # 안 그러면 "생년월일"처럼 이미 처리 중인 라벨이 후보로 나와
    # 힌트가 오히려 헷갈립니다.
    used_lines = {v.raw_line for v in out.values.values() if v.found}
    known = {re.sub(r"\s+", "", lb)
             for spec in FIELD_BY_KEY.values() for lb in spec.labels}

    unused = [
        (lb, ln) for lb, ln in labelled
        if ln not in used_lines and re.sub(r"\s+", "", lb) not in known
    ]

    tips = []
    for key in out.missing:
        spec = FIELD_BY_KEY[key]
        tips.append(f"[{spec.display}] patterns.py 의 '{key}' labels 에 추가 →")
        if unused:
            for lb, ln in unused[:6]:
                tips.append(f"      후보: \"{lb.replace(' ', '')}\"   (원문: {ln[:50]})")
        else:
            tips.append("      원문에 라벨 형태의 줄이 없습니다 — LLM 이 필요한 필드일 수 있습니다")
        break 
    return tips
