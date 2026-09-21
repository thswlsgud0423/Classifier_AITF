"""패턴 시험 도구 — patterns.py 를 고치기 전에 여기서 먼저 확인하십시오.

patterns.py 를 고치고 → run.py 돌리고 → 안 되면 다시 고치고… 를
반복하면 느립니다. 이 도구로 **고치기 전에** 될지 안 될지 바로 봅니다.

사용법
------
1) 라벨 하나가 먹히는지 시험

    python test_pattern.py --text samples/재학증명서.txt --label 소속
    python test_pattern.py --text samples/재학증명서.txt --label 출국일자

   → 먹히면 patterns.py 의 해당 labels 에 그 단어를 추가하십시오.

2) 문서에 라벨처럼 생긴 줄이 뭐가 있는지 전부 보기

    python test_pattern.py --text samples/재학증명서.txt --scan

   → 추출이 안 되는 필드가 있을 때 여기서 실제 라벨을 찾으십시오.

3) 정규식을 직접 시험 (VERIFY_SPECS 추가할 때)

    python test_pattern.py --text x.txt --regex "확인번호\\s*[:：]\\s*([A-Z]\\d{8})"

4) PDF 를 텍스트로 뽑아 저장 (샘플 만들 때)

    python test_pattern.py --pdf 내서류.pdf --save 내서류.txt
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from patterns import (
    FIELDS,
    P_DATE,
    P_NAME,
    P_NUM,
    P_TEXT,
    spaced,
)

VALUE_PATTERNS = {
    "date": (P_DATE, "날짜"),
    "name": (P_NAME, "한글 이름"),
    "text": (P_TEXT, "줄 나머지 전부"),
    "num": (P_NUM, "숫자"),
}


def load_text(args) -> str:
    if args.pdf:
        try:
            import pdfplumber
        except ImportError:
            sys.exit("pdfplumber 가 필요합니다:  pip install pdfplumber")
        with pdfplumber.open(args.pdf) as pdf:
            raw = "\n".join((p.extract_text() or "") for p in pdf.pages)
        text = raw.replace("\u3000", " ")
        text = "\n".join(line.strip() for line in text.split("\n"))
        if args.save:
            Path(args.save).write_text(text, encoding="utf-8")
            print(f"저장됨: {args.save}  ({len(text)}자)\n")
        return text
    if args.text:
        return Path(args.text).read_text(encoding="utf-8")
    sys.exit("--text 또는 --pdf 가 필요합니다")


def cmd_scan(text: str) -> None:
    """'라벨 : 값' 형태의 줄을 전부 찾아 보여줍니다."""
    print("문서에서 발견한 '라벨 : 값' 줄\n")
    line_pat = re.compile(r"^\s*([^\n:：]{1,20})\s*[:：]\s*(.+?)\s*$")
    known = {lbl for f in FIELDS for lbl in f.labels}

    found = False
    for line in text.split("\n"):
        m = line_pat.match(line)
        if not m:
            continue
        found = True
        label_raw = m.group(1).strip()
        label_flat = re.sub(r"\s+", "", label_raw)   # 자간 제거
        value = m.group(2).strip()
        mark = "등록됨" if label_flat in known else "★ 미등록"
        print(f"  [{mark}] {label_flat:14s} = {value[:40]}")
        if label_flat not in known:
            print(f"            patterns.py 의 적당한 필드 labels 에 \"{label_flat}\" 추가")

    if not found:
        print("  '라벨 : 값' 형태의 줄이 없습니다.")
        print("  → 라벨 없이 값만 있는 문서입니다. 정규식으로는 어렵고 LLM 대상입니다.")

    print("\n등록된 라벨 목록 (참고):")
    for f in FIELDS:
        print(f"  {f.key:16s} {', '.join(f.labels)}")


def cmd_label(text: str, label: str, value_kind: str) -> None:
    """라벨 하나가 이 문서에서 먹히는지 시험."""
    pattern, desc = VALUE_PATTERNS[value_kind]
    rx = re.compile(rf"(?:{spaced(label)})\s*[:：]\s*{pattern}", re.MULTILINE)

    print(f"시험할 라벨 : \"{label}\"   (값 모양: {desc})")
    print(f"생성된 정규식: {rx.pattern}\n")

    hits = list(rx.finditer(text))
    if not hits:
        print("  매칭 실패\n")
        # 왜 실패했는지 힌트
        flat = re.sub(r"\s+", "", text)
        if re.sub(r"\s+", "", label) in flat:
            print("  힌트: 라벨 글자는 문서에 있습니다. 값 모양이 안 맞는 것 같습니다.")
            print(f"        --as date / name / text / num 으로 바꿔 보십시오.")
            for line in text.split("\n"):
                if re.sub(r"\s+", "", label) in re.sub(r"\s+", "", line):
                    print(f"        해당 줄: {line.strip()}")
        else:
            print("  힌트: 라벨 글자 자체가 문서에 없습니다. --scan 으로 실제 라벨을 찾으십시오.")
        return

    print(f"  매칭 성공 — {len(hits)}건")
    for m in hits:
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        line = text[line_start:line_end if line_end != -1 else len(text)].strip()
        print(f"    값: {m.group(1)}")
        print(f"    원문: {line}")
    print(f"\n  → patterns.py 에서 이 라벨을 쓰는 필드의 labels 에 \"{label}\" 추가하십시오.")


def cmd_regex(text: str, pattern: str) -> None:
    """정규식 직접 시험 (VERIFY_SPECS 용)."""
    try:
        rx = re.compile(pattern, re.MULTILINE)
    except re.error as e:
        sys.exit(f"정규식 오류: {e}")

    hits = list(rx.finditer(text))
    print(f"정규식: {pattern}\n")
    if not hits:
        print("  매칭 실패")
        return
    print(f"  매칭 성공 — {len(hits)}건")
    for m in hits:
        print(f"    잡힌 값: {m.group(1) if m.groups() else m.group(0)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="텍스트 파일")
    ap.add_argument("--pdf", help="PDF 파일")
    ap.add_argument("--save", help="--pdf 와 함께. 텍스트로 저장")
    ap.add_argument("--scan", action="store_true", help="문서의 모든 라벨 보기")
    ap.add_argument("--label", help="시험할 라벨 (공백 없이)")
    ap.add_argument("--as", dest="value_kind", default="text",
                    choices=list(VALUE_PATTERNS), help="값 모양 (기본: text)")
    ap.add_argument("--regex", help="정규식 직접 시험")
    args = ap.parse_args()

    text = load_text(args)

    if args.scan:
        cmd_scan(text)
    elif args.label:
        cmd_label(text, args.label, args.value_kind)
    elif args.regex:
        cmd_regex(text, args.regex)
    else:
        print(f"{len(text)}자 읽음. 무엇을 할지 지정하십시오:\n")
        print("  --scan                문서의 라벨 전부 보기")
        print("  --label 소속           라벨 하나 시험")
        print("  --regex \"...\"          정규식 시험")


if __name__ == "__main__":
    main()
