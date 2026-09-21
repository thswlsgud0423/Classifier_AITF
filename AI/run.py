"""실행.

    python run.py                      # samples/ 폴더 전체
    python run.py 파일.pdf              # 파일 하나
    python run.py 파일.pdf --text       # 추출된 원문도 보기

못 뽑은 필드가 있으면 patterns.py 에 뭘 추가하면 되는지 알려줍니다.
"""
from __future__ import annotations

import sys
from pathlib import Path

from extract import diagnose, extract


def show(r) -> None:
    got = sum(1 for v in r.values.values() if v.found)
    total = len(r.values)
    print(f"\n{'─' * 70}")
    print(f"{Path(r.path).name}")
    print(f"  종류 {r.doc_type or '판별실패'} · {r.kind} · {r.text_chars}자 · 추출 {got}/{total}")

    if r.values:
        print()
        for v in r.values.values():
            need = "*" if v.key in ("name", "birth_date", "issue_date") else " "
            if v.found:
                print(f"  {need} {v.display:10s} {v.value}")
            else:
                print(f"  {need} {v.display:10s} — 실무자 직접 입력")

    if r.verify_numbers:
        print()
        for name, val, note in r.verify_numbers:
            print(f"    [{name}] {val}")
            print(f"       {note}")

    if r.problems:
        print()
        for p in r.problems:
            print(f"  ! {p}")

    tips = diagnose(r)
    if tips:
        print()
        print("  ┌─ patterns.py 수정 힌트 ──────────────────────────")
        for t in tips:
            print(f"  │ {t}")
        print("  └──────────────────────────────────────────────────")

    print()
    print(f"  필수 누락: {r.missing_required or '없음'}   "
          f"→ {'자동 처리 가능' if r.ready else '실무자 입력 필요'}")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_text = "--text" in sys.argv

    if args:
        files = [Path(a) for a in args]
    else:
        samples_dir = Path(__file__).resolve().parent / "samples"
        files = sorted(samples_dir.glob("*.pdf"))
        if not files:
            sys.exit(f"{samples_dir} 에 PDF가 없습니다. 파일 경로를 직접 넣어보세요.")

    stats = {"found": 0, "total": 0}
    for f in files:
        r = extract(f)
        if show_text:
            print("=" * 70)
            print(r.text)
        show(r)
        stats["found"] += sum(1 for v in r.values.values() if v.found)
        stats["total"] += len(r.values)

    if stats["total"]:
        pct = stats["found"] / stats["total"] * 100
        print(f"\n{'═' * 70}")
        print(f"전체 {stats['total']}개 필드 중 {stats['found']}개 추출 ({pct:.0f}%)")
        print(f"나머지 {stats['total'] - stats['found']}개는 patterns.py 보강 또는 LLM 대상")


if __name__ == "__main__":
    main()
