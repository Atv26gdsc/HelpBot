"""
chunker.py — Table-aware, section-aware chunking for MOSDAC PDFs
====================================================================
Reads cleaned page-level JSONL files from data/processed/*.jsonl
(the output of clean_insat3d.py) and produces chunk-level records
ready for embedding.

Key design decisions (see conversation for rationale):
  1. TABLE PAGES: if a page contains a line starting with "Table",
     the ENTIRE page is kept as ONE chunk, never word-split. Splitting
     a table mid-row would separate a channel name from its own
     resolution/quantization values, corrupting the exact facts
     (specs, resolutions) your chatbot needs to answer correctly.
  2. PROSE PAGES: split on section headings (e.g. "1.1 Scope",
     "Chapter 1: Introduction", "Appendix- I") so each chunk stays
     on-topic. If a section runs long, it's further split by a
     ~300-500 word sliding window with overlap, always at paragraph
     boundaries (never mid-sentence).
  3. Every chunk carries metadata: chunk_id, source, page, section,
     is_table, word_count — this is what Step 7 of the guide asks for
     (filtering, citations, debugging retrieval later).

Usage:
    python chunker.py --input-dir data/processed --out data/processed/chunks.jsonl
    python chunker.py --input-dir data/processed --out data/processed/chunks.jsonl --max-words 400 --overlap-words 60
"""

import argparse
import glob
import json
import os
import re

TABLE_LINE_PATTERN = re.compile(r"^Table[:\s]", re.I)

HEADING_PATTERNS = [
    re.compile(r"^Chapter\s+\d+", re.I),          # "Chapter 1: Introduction"
    re.compile(r"^Appendix[\s\-]", re.I),          # "Appendix- I : ..."
    re.compile(r"^\d+(\.\d+)+\s+\S"),              # "1.1 Scope", "2.4 Geophysical..."
]


def is_heading(line: str) -> bool:
    return any(p.match(line) for p in HEADING_PATTERNS)


def is_table_page(text: str) -> bool:
    return any(TABLE_LINE_PATTERN.match(line) for line in text.split("\n"))


def find_table_title(text: str) -> str:
    """Return the first 'Table ...' line found, used as the section label."""
    for line in text.split("\n"):
        if TABLE_LINE_PATTERN.match(line):
            return line.strip()
    return "Table"


def split_into_paragraphs(text: str):
    """
    Split cleaned page text into paragraph-like blocks using blank lines
    and heading lines as natural break points. Returns list of (heading_or_None, block_text).

    A heading line is NOT emitted as its own block -- a heading immediately
    followed by a blank line (this doc's exact formatting, e.g. "1.1 Scope"
    then a blank line then its paragraph) must not be flushed prematurely.
    The heading text is prepended to the body only ONCE -- the first body
    block under that heading -- so a heading followed by several
    blank-line-separated paragraphs (e.g. "1.3 INSAT-3D Payloads" with many
    short paragraphs under it) doesn't get the heading repeated on every
    single paragraph.
    """
    lines = text.split("\n")
    blocks = []
    current_heading = None
    heading_already_used = False
    body_lines = []  # accumulated body content under current_heading (heading itself excluded)

    def flush():
        nonlocal heading_already_used
        if body_lines:
            body_text = "\n".join(body_lines).strip()
            if body_text:
                if current_heading and not heading_already_used:
                    combined = f"{current_heading}\n{body_text}"
                    heading_already_used = True
                else:
                    combined = body_text
                blocks.append((current_heading, combined))

    for line in lines:
        stripped = line.strip()
        if is_heading(stripped):
            # New heading found -> flush whatever body we had under the OLD heading,
            # then start fresh under the new one (heading itself withheld until body arrives).
            flush()
            current_heading = stripped
            heading_already_used = False
            body_lines = []
        elif not stripped:
            # Blank line: only a real paragraph break if we've already collected
            # body content. A blank line right after a bare heading is just
            # formatting, not a break -- don't flush yet.
            if body_lines:
                flush()
                body_lines = []
        else:
            body_lines.append(line)

    flush()
    return blocks


def word_count(text: str) -> int:
    return len(text.split())


def chunk_prose_blocks(blocks, max_words: int, overlap_words: int):
    """
    Merge small blocks and split large ones into ~max_words chunks with
    ~overlap_words of trailing context carried into the next chunk.
    Never splits mid-sentence: overlap is taken from whole trailing
    sentences/lines, not an arbitrary word cut.
    Returns list of (section_label, chunk_text).
    """
    chunks = []
    current_section = None
    current_text_parts = []
    current_words = 0

    def flush(next_section_hint=None):
        nonlocal current_text_parts, current_words
        if current_text_parts:
            full_text = "\n".join(current_text_parts).strip()
            if full_text:
                chunks.append((current_section, full_text))
        current_text_parts = []
        current_words = 0

    for heading, block_text in blocks:
        if heading is not None and heading != current_section:
            # Starting a genuinely new section -> flush the PREVIOUS section's
            # accumulated text, but do NOT emit the heading as its own chunk.
            # It gets folded into the start of the new section's first chunk
            # below, so titles stay attached to their content.
            flush()
            current_section = heading

        block_words = word_count(block_text)

        if current_words + block_words <= max_words or not current_text_parts:
            current_text_parts.append(block_text)
            current_words += block_words
        else:
            # would overflow -> flush current chunk, start new one
            # carry a small tail of the previous chunk forward for continuity
            tail_text = ""
            if current_text_parts:
                prev_full = "\n".join(current_text_parts)
                prev_words_list = prev_full.split()
                tail_words = prev_words_list[-overlap_words:] if overlap_words > 0 else []
                tail_text = " ".join(tail_words)
            flush()
            new_parts = [tail_text, block_text] if tail_text else [block_text]
            current_text_parts = new_parts
            current_words = word_count(block_text) + (overlap_words if tail_text else 0)

    flush()
    return chunks


def chunk_page(record: dict, max_words: int, overlap_words: int):
    """Return list of chunk dicts for a single page record {source, page, text}."""
    text = record["text"]
    source = record["source"]
    page = record["page"]

    if not text.strip():
        return []

    if is_table_page(text):
        # Whole page as one chunk -- never split a table mid-row.
        return [{
            "text": text,
            "source": source,
            "page": page,
            "section": find_table_title(text),
            "is_table": True,
            "word_count": word_count(text),
        }]

    blocks = split_into_paragraphs(text)
    prose_chunks = chunk_prose_blocks(blocks, max_words, overlap_words)

    results = []
    for section_label, chunk_text in prose_chunks:
        results.append({
            "text": chunk_text,
            "source": source,
            "page": page,
            "section": section_label or "(unlabeled)",
            "is_table": False,
            "word_count": word_count(chunk_text),
        })
    return results


def make_chunk_id(source: str, index: int) -> str:
    stem = os.path.splitext(source)[0].lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return f"{stem}_{index:04d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=os.path.join("data", "processed"),
                         help="Directory containing cleaned page-level .jsonl files")
    parser.add_argument("--out", default=os.path.join("data", "processed", "chunks.jsonl"),
                         help="Output path for chunk-level .jsonl")
    parser.add_argument("--max-words", type=int, default=450,
                         help="Target max words per prose chunk")
    parser.add_argument("--overlap-words", type=int, default=60,
                         help="Words of trailing context carried into the next chunk")
    args = parser.parse_args()

    input_files = sorted(glob.glob(os.path.join(args.input_dir, "*.jsonl")))
    # Don't re-chunk our own output if it's sitting in the same folder
    input_files = [f for f in input_files if os.path.abspath(f) != os.path.abspath(args.out)]

    if not input_files:
        print(f"[!] No .jsonl files found in {args.input_dir}. Run clean_insat3d.py first.")
        return

    print(f"[*] Found {len(input_files)} cleaned document file(s):")
    for f in input_files:
        print(f"    - {f}")
    print("-" * 85)

    all_chunks = []
    per_doc_stats = {}

    for filepath in input_files:
        doc_chunks_before = len(all_chunks)
        table_count = 0
        prose_count = 0

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                page_chunks = chunk_page(record, args.max_words, args.overlap_words)
                for c in page_chunks:
                    if c["is_table"]:
                        table_count += 1
                    else:
                        prose_count += 1
                all_chunks.extend(page_chunks)

        doc_name = os.path.basename(filepath)
        n_chunks = len(all_chunks) - doc_chunks_before
        per_doc_stats[doc_name] = (n_chunks, table_count, prose_count)
        print(f"  {doc_name}: {n_chunks} chunks ({prose_count} prose, {table_count} table)")

    # Assign chunk_ids per source document, in order
    counters = {}
    for c in all_chunks:
        src = c["source"]
        counters[src] = counters.get(src, 0) + 1
        c["chunk_id"] = make_chunk_id(src, counters[src])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for c in all_chunks:
            # order keys nicely for readability
            ordered = {
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "source": c["source"],
                "page": c["page"],
                "section": c["section"],
                "is_table": c["is_table"],
                "word_count": c["word_count"],
            }
            f.write(json.dumps(ordered, ensure_ascii=False) + "\n")

    print("-" * 85)
    total_table = sum(v[1] for v in per_doc_stats.values())
    total_prose = sum(v[2] for v in per_doc_stats.values())
    print(f"✅ Total chunks: {len(all_chunks)} ({total_prose} prose, {total_table} table)")
    print(f"💾 Saved to: {args.out}")
    print("Tip: open the file and spot-check a few table chunks and a few prose chunks "
          "before moving to embeddings.")


if __name__ == "__main__":
    main()