"""
clean_insat3d.py — FIXED VERSION
==================================
Fixes vs the previous run:
  1. Boilerplate detection now normalizes whitespace before comparing lines
     across pages (strips, collapses internal spaces, normalizes \xa0/tabs).
  2. Uses a repetition THRESHOLD (default 60% of pages) instead of requiring
     100% exact match, so it still catches headers that are missing on 1-2
     blank/short pages.
  3. Debug mode prints the FULL raw text of a page (not just first 10 lines)
     so you can verify nothing is being mysteriously added.
  4. Char counts are computed on the exact same strings that get printed,
     so the "removed N chars" math is always consistent with what you see.
  5. Cleaning is strictly subtractive: cleaned_text is built by filtering
     lines out of raw_text, never appending anything new.
  6. Page-number stripping only applies at the very top/bottom of a page,
     so numeric table values (e.g. resolution "1", quantization bits "10")
     in the MIDDLE of a page are never deleted.
  7. Saves cleaned output to data/processed/<pdf_stem>.jsonl so it can
     actually be used as input for chunking, not just printed to terminal.

Usage:
    python clean_insat3d.py --pdf data/raw/pdfs/INSAT3D_Data_Products_Format.pdf
    python clean_insat3d.py --pdf <path> --debug-page 1   # full before/after for page 1
    python clean_insat3d.py --pdf <path> --threshold 0.5  # looser boilerplate detection
    python clean_insat3d.py --pdf <path> --out custom/path.jsonl
"""

import argparse
import json
import os
import re
from collections import Counter

import fitz  # PyMuPDF


def normalize_line(line: str) -> str:
    """Normalize a line so whitespace/encoding quirks don't break exact matching."""
    line = line.replace("\xa0", " ")       # non-breaking space -> regular space
    line = re.sub(r"\s+", " ", line)       # collapse multiple spaces/tabs
    return line.strip()


def extract_pages(pdf_path: str):
    """Return list of dicts: {page_num, raw_text, raw_lines}."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        raw_text = page.get_text()
        raw_lines = raw_text.split("\n")
        pages.append({"page_num": i, "raw_text": raw_text, "raw_lines": raw_lines})
    doc.close()
    return pages


def detect_boilerplate_lines(pages, threshold: float = 0.6):
    """
    Find lines that repeat across many pages -> likely headers/footers.
    threshold: fraction of pages a (normalized) line must appear on to count
               as boilerplate. 0.6 = appears on 60%+ of pages.
    Only considers short lines (<80 chars) since long repeated lines are
    unlikely to be headers and more likely to be a coincidence / real content.
    """
    line_page_counts = Counter()
    total_pages = len(pages)

    for p in pages:
        # count each distinct normalized line ONCE per page (avoid a line that
        # repeats twice on the same page inflating the count)
        seen_this_page = set()
        for line in p["raw_lines"]:
            norm = normalize_line(line)
            if not norm or len(norm) > 80:
                continue
            if norm in seen_this_page:
                continue
            seen_this_page.add(norm)
            line_page_counts[norm] += 1

    boilerplate = {
        line for line, count in line_page_counts.items()
        if count / total_pages >= threshold
    }
    return boilerplate, line_page_counts


PAGE_NUM_PATTERNS = [
    re.compile(r"^\d+$"),                 # standalone number: "26"
    re.compile(r"^[ivxlcdm]+$", re.I),    # roman numeral page: "iii"
    re.compile(r"^page\s+\d+(\s+of\s+\d+)?$", re.I),  # "Page 8 of 31"
]


def is_page_number_line(norm_line: str) -> bool:
    return any(p.match(norm_line) for p in PAGE_NUM_PATTERNS)


def clean_page(raw_text: str, boilerplate_lines: set) -> str:
    """
    Strictly subtractive: only removes lines, never adds any.
    Removes: detected boilerplate lines, and a standalone page-number line
    but ONLY if it sits at the very top or bottom of the page (where real
    page numbers live). A bare number in the MIDDLE of a page is almost
    always table data (e.g. a resolution or quantization-bits cell) and
    must be preserved.
    Also collapses multiple blank lines into one.
    """
    lines = raw_text.split("\n")

    # Find indices of non-blank lines, so we know which lines are the
    # "first" and "last" content lines on the page (candidates for a
    # real printed page number), as opposed to numbers buried inside
    # a table in the middle of the page.
    non_blank_idx = [i for i, l in enumerate(lines) if normalize_line(l)]
    first_content_idx = non_blank_idx[0] if non_blank_idx else None
    last_content_idx = non_blank_idx[-1] if non_blank_idx else None

    kept_lines = []
    for i, line in enumerate(lines):
        norm = normalize_line(line)

        if not norm:
            # keep at most one blank line in a row
            if kept_lines and kept_lines[-1] == "":
                continue
            kept_lines.append("")
            continue

        if norm in boilerplate_lines:
            continue

        is_edge_of_page = i in (first_content_idx, last_content_idx)
        if is_edge_of_page and is_page_number_line(norm):
            continue

        kept_lines.append(norm)

    # trim leading/trailing blank lines
    while kept_lines and kept_lines[0] == "":
        kept_lines.pop(0)
    while kept_lines and kept_lines[-1] == "":
        kept_lines.pop()

    return "\n".join(kept_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--threshold", type=float, default=0.6,
                         help="Fraction of pages a line must appear on to be treated as boilerplate")
    parser.add_argument("--debug-page", type=int, default=None,
                         help="Print FULL raw vs cleaned text for a single page number")
    parser.add_argument("--out", default=None,
                         help="Path to write cleaned pages as JSONL "
                              "(default: data/processed/<pdf_stem>.jsonl)")
    args = parser.parse_args()

    pages = extract_pages(args.pdf)
    total_pages = len(pages)
    print(f"[*] Total Pages: {total_pages}")

    boilerplate_lines, counts = detect_boilerplate_lines(pages, threshold=args.threshold)

    print(f"[+] Detected {len(boilerplate_lines)} repeated boilerplate line(s) "
          f"(threshold={args.threshold*100:.0f}% of pages):")
    for line in sorted(boilerplate_lines, key=lambda l: -counts[l]):
        pct = counts[line] / total_pages * 100
        print(f"    - \"{line}\"  (appears on {counts[line]}/{total_pages} pages, {pct:.0f}%)")
    print("-" * 85)

    cleaned_pages = []
    for p in pages:
        cleaned_text = clean_page(p["raw_text"], boilerplate_lines)
        cleaned_pages.append({
            "page_num": p["page_num"],
            "raw_text": p["raw_text"],
            "cleaned_text": cleaned_text,
        })

    # Debug mode: show FULL before/after for one page, no truncation
    if args.debug_page is not None:
        target = next((p for p in cleaned_pages if p["page_num"] == args.debug_page), None)
        if target is None:
            print(f"[!] Page {args.debug_page} not found.")
        else:
            print(f"📄 --- [PAGE {target['page_num']} FULL BEFORE] ---")
            print(target["raw_text"])
            print(f"\n📄 --- [PAGE {target['page_num']} FULL AFTER] ---")
            print(target["cleaned_text"])
            raw_len = len(target["raw_text"])
            clean_len = len(target["cleaned_text"])
            print(f"\n-> Characters: {raw_len} raw ==> {clean_len} cleaned "
                  f"({raw_len - clean_len} removed)")
        print("-" * 85)
    else:
        # Show a compact summary for every page (first 3 lines only, clearly labeled as preview)
        for p in cleaned_pages:
            raw_len = len(p["raw_text"])
            clean_len = len(p["cleaned_text"])
            preview = p["cleaned_text"][:100].replace("\n", " ")
            print(f"Page {p['page_num']:02d}/{total_pages}: {raw_len} -> {clean_len} chars "
                  f"| preview: \"{preview}...\"")

    # ---- Save cleaned pages to disk (this is the actual input for chunking) ----
    source_filename = os.path.basename(args.pdf)
    pdf_stem = os.path.splitext(source_filename)[0]
    out_path = args.out or os.path.join("data", "processed", f"{pdf_stem}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for p in cleaned_pages:
            record = {
                "source": source_filename,
                "page": p["page_num"],
                "text": p["cleaned_text"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("=" * 85)
    print(f"✅ Cleaning complete for {total_pages} pages.")
    print(f"💾 Saved cleaned pages to: {out_path}")
    print(f"Tip: run with --debug-page N to sanity-check a specific page in full before chunking.")


if __name__ == "__main__":
    main()