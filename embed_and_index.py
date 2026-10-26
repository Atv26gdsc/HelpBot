"""
embed_and_index.py — Step 8 + Step 9: Embeddings + Vector Database
=====================================================================
Reads chunks.jsonl (output of chunker.py), embeds every chunk's text
using BGE-M3 (or a fallback model), and builds a local FAISS index so
the chunks can be semantically searched.

Produces TWO files (keep them together, they're a matched pair):
  1. <out>.index        -- the FAISS vector index (just numbers, no text)
  2. <out>.metadata.jsonl -- one line per vector, in the SAME ORDER as
     the index, containing chunk_id/text/source/page/section/is_table.
     This is what lets you go from "FAISS found vector #42" back to
     "here's the actual text and its PDF source/page for citation."

Usage:
    python embed_and_index.py --chunks data/processed/chunks.jsonl
    python embed_and_index.py --chunks data/processed/chunks.jsonl --model small
    python embed_and_index.py --chunks data/processed/chunks.jsonl --test-query "resolution of SWIR channel"

Model options:
    bge-m3   (default) -- BAAI/bge-m3, best accuracy, ~2.2GB download
    small              -- BAAI/bge-small-en-v1.5, ~130MB, much faster, slightly lower accuracy
    minilm             -- sentence-transformers/all-MiniLM-L6-v2, ~80MB, fastest, for quick sanity tests only

First run will download the model from Hugging Face (needs internet).
Subsequent runs reuse the cached model, no re-download.
"""

import argparse
import json
import os
import sys

import numpy as np

MODEL_MAP = {
    "bge-m3": "BAAI/bge-m3",
    "small": "BAAI/bge-small-en-v1.5",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}


def load_chunks(path: str):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def embed_chunks(chunks, model_name: str, batch_size: int = 16):
    """Embed chunk texts. Import is inside the function so the script
    still gives a clear error message if sentence-transformers isn't
    installed, rather than failing at the top of the file."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[!] sentence-transformers not installed. Run:")
        print("    pip install sentence-transformers --break-system-packages")
        sys.exit(1)

    print(f"[*] Loading model: {model_name} (first run downloads it, may take a while)")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    print(f"[*] Embedding {len(texts)} chunks in batches of {batch_size}...")

    # normalize_embeddings=True -> vectors are unit length, so we can use
    # a simple inner-product FAISS index for cosine similarity search.
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype="float32"), model


def build_faiss_index(embeddings: np.ndarray):
    import faiss

    dim = embeddings.shape[1]
    # IndexFlatIP = exact search via inner product. Since embeddings are
    # normalized, inner product == cosine similarity. "Flat" means it
    # checks every vector -- fine and exact for a prototype's ~100-500
    # chunks; only matters to switch to an approximate index at much
    # larger scale (tens of thousands+ chunks).
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[*] Built FAISS index: {index.ntotal} vectors, dimension {dim}")
    return index


def save_metadata(chunks, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            record = {
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "source": c["source"],
                "page": c["page"],
                "section": c["section"],
                "is_table": c["is_table"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_test_query(query: str, model, index, chunks, top_k: int = 5):
    print("\n" + "=" * 85)
    print(f"🔍 Test query: \"{query}\"")
    print("=" * 85)
    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype="float32")

    scores, indices = index.search(query_vec, top_k)

    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if idx < 0:
            continue
        c = chunks[idx]
        preview = c["text"][:150].replace("\n", " ")
        table_flag = " [TABLE]" if c["is_table"] else ""
        print(f"\n#{rank}  score={score:.3f}  {c['source']} p.{c['page']}{table_flag}")
        print(f"     section: {c['section']}")
        print(f"     preview: {preview}...")
    print("\n" + "=" * 85)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, help="Path to chunks.jsonl from chunker.py")
    parser.add_argument("--model", default="bge-m3", choices=list(MODEL_MAP.keys()),
                         help="Which embedding model to use (default: bge-m3)")
    parser.add_argument("--out", default=os.path.join("vector_db", "mosdac"),
                         help="Output path prefix (default: vector_db/mosdac -> "
                              "vector_db/mosdac.index + vector_db/mosdac.metadata.jsonl)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-query", default=None,
                         help="Optional: run a test search after building the index")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    if not chunks:
        print(f"[!] No chunks found in {args.chunks}. Run chunker.py first.")
        sys.exit(1)
    print(f"[*] Loaded {len(chunks)} chunks from {args.chunks}")

    model_name = MODEL_MAP[args.model]
    embeddings, model = embed_chunks(chunks, model_name, batch_size=args.batch_size)

    index = build_faiss_index(embeddings)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    index_path = f"{args.out}.index"
    metadata_path = f"{args.out}.metadata.jsonl"

    import faiss
    faiss.write_index(index, index_path)
    save_metadata(chunks, metadata_path)

    print("=" * 85)
    print(f"✅ Index built and saved.")
    print(f"💾 Index:    {index_path}")
    print(f"💾 Metadata: {metadata_path}")
    print("Tip: use --test-query \"your question\" to sanity-check retrieval right now.")

    if args.test_query:
        run_test_query(args.test_query, model, index, chunks)


if __name__ == "__main__":
    main()