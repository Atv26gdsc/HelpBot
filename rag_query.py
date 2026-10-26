"""
rag_query.py — Step 10 + Step 11: Basic RAG with grounding safeguard
========================================================================
Ties everything together: takes a user question, retrieves the most
relevant chunks from your FAISS index, and asks a LOCAL LLM (via Ollama)
to answer USING ONLY those chunks -- with a mandatory "I don't know"
fallback if the retrieved evidence doesn't actually support an answer
(Step 11 of the guide: never let the bot invent an answer).

Uses Ollama (https://ollama.com) for fully local, free generation --
no API key, no per-token cost, no internet required after setup.

Setup (one-time):
    1. Install Ollama from https://ollama.com
    2. Pull a model:  ollama pull llama3.1:8b
    3. Make sure Ollama is running (it usually starts automatically,
       or run:  ollama serve)

Usage:
    python rag_query.py --index vector_db/mosdac.index --metadata vector_db/mosdac.metadata.jsonl --query "What is the resolution of the SWIR channel?"
    python rag_query.py --index vector_db/mosdac.index --metadata vector_db/mosdac.metadata.jsonl --interactive
    python rag_query.py --index vector_db/mosdac.index --metadata vector_db/mosdac.metadata.jsonl --interactive --gen-model mistral
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

# Minimum similarity score (cosine, since vectors are normalized) below
# which we treat retrieval as "nothing relevant found" and refuse to
# answer, rather than letting the LLM stretch a weak match into a guess.
MIN_RELEVANCE_SCORE = 0.35

SYSTEM_PROMPT = """You are a helpful assistant answering questions about MOSDAC satellite \
data products using ONLY the provided document excerpts.

STRICT RULES:
1. Answer using ONLY the information in the provided excerpts. Do not use outside knowledge.
2. If the excerpts do not contain enough information to answer confidently, say exactly: \
"I could not find sufficient information in the available MOSDAC documents to answer this question." \
Do not guess or fill gaps with general knowledge.
3. Every factual claim in your answer must be traceable to one of the excerpts.
4. After your answer, list the sources you used in the format: Source: <filename>, Page <number>
5. If an excerpt is marked [TABLE], treat its numbers as exact values from a structured \
table -- read column/row labels carefully before stating a figure.
6. Be concise. Do not repeat the question back or add unnecessary preamble."""


def load_metadata(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_faiss_index(path: str):
    import faiss
    return faiss.read_index(path)


def load_embedding_model(model_key: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_MAP[model_key])


def retrieve(query: str, embed_model, index, metadata, top_k: int = 5):
    query_vec = embed_model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype="float32")
    scores, indices = index.search(query_vec, top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0:
            continue
        chunk = metadata[idx]
        results.append({**chunk, "score": float(score)})
    return results


def build_context_block(results):
    """Format retrieved chunks into a labeled context block for the LLM prompt."""
    parts = []
    for i, r in enumerate(results, start=1):
        table_tag = " [TABLE]" if r["is_table"] else ""
        parts.append(
            f"--- Excerpt {i} (Source: {r['source']}, Page {r['page']}, "
            f"Section: {r['section']}{table_tag}) ---\n{r['text']}"
        )
    return "\n\n".join(parts)


def compute_confidence(results):
    """
    Confidence is a proxy for 'how sure are we this retrieval actually
    found relevant evidence', based on the FAISS similarity scores.
    NOT the same as accuracy -- accuracy requires comparing against a
    known correct answer (see Step 12/13: the evaluation set), which
    a single live query has no way to check itself against.
    """
    if not results:
        return 0.0, "Low"

    top_score = results[0]["score"]
    # average of top-3 (or fewer if less than 3 retrieved) -- a single
    # high top score with weak scores right behind it is less trustworthy
    # than several consistently strong matches.
    top_n = results[:3]
    avg_top = sum(r["score"] for r in top_n) / len(top_n)

    # Blend: mostly the top match, but penalized a bit if the rest of
    # the field is weak (suggests the top hit might be a fluke/edge case).
    combined = 0.7 * top_score + 0.3 * avg_top

    if combined >= 0.55:
        label = "High"
    elif combined >= MIN_RELEVANCE_SCORE:
        label = "Medium"
    else:
        label = "Low"

    return combined, label


def call_ollama(question: str, context_block: str, model: str = "llama3.1:8b",
                 host: str = "http://localhost:11434"):
    """Call a local Ollama server. Requires Ollama installed and running
    (https://ollama.com), and the model already pulled, e.g.:
        ollama pull llama3.1:8b
    """
    import requests

    user_message = f"""Document excerpts:

{context_block}

Question: {question}"""

    try:
        response = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
            },
            timeout=600,  # generous timeout -- first call loads the model into
                          # memory (slow on CPU-only machines) on top of generating
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Could not connect to Ollama at {host}. Is it running? "
            f"Start it with: ollama serve"
        )
    except requests.exceptions.ReadTimeout:
        raise RuntimeError(
            "Ollama took too long to respond (over 10 minutes). "
            "Try a smaller/faster model, e.g.: ollama pull llama3.2:3b, "
            "then re-run with --gen-model llama3.2:3b"
        )

    if response.status_code != 200:
        raise RuntimeError(f"Ollama returned HTTP {response.status_code}: {response.text}")

    data = response.json()
    return data["message"]["content"]


def answer_question(question: str, embed_model, index, metadata,
                     top_k: int = 5, gen_model: str = "llama3.1:8b",
                     ollama_host: str = "http://localhost:11434", verbose: bool = True):
    results = retrieve(question, embed_model, index, metadata, top_k=top_k)

    if verbose:
        print("\n" + "-" * 85)
        print("Retrieved chunks:")
        for r in results:
            table_tag = " [TABLE]" if r["is_table"] else ""
            print(f"  score={r['score']:.3f}  {r['source']} p.{r['page']}{table_tag}  "
                  f"({r['section']})")
        print("-" * 85)

    # Grounding safeguard: if even the TOP result is a weak match,
    # don't bother calling the LLM at all -- refuse locally. This is a
    # second line of defense on top of the LLM's own instruction to say
    # "I don't know", in case the LLM tries to answer anyway from a
    # borderline-relevant excerpt.
    if not results or results[0]["score"] < MIN_RELEVANCE_SCORE:
        confidence_score, confidence_label = compute_confidence(results)
        return ("I could not find sufficient information in the available MOSDAC "
                "documents to answer this question."), results, (confidence_score, confidence_label)

    context_block = build_context_block(results)
    answer = call_ollama(question, context_block, model=gen_model, host=ollama_host)

    # Don't rely on the LLM to remember to list sources -- smaller local
    # models are inconsistent about following that instruction. Append
    # the citation list programmatically from the actual retrieved
    # metadata instead, so it's guaranteed correct every time.
    seen = set()
    citation_lines = []
    for r in results:
        key = (r["source"], r["page"])
        if key not in seen:
            seen.add(key)
            citation_lines.append(f"Source: {r['source']}, Page {r['page']}")

    answer_with_citations = answer.strip() + "\n\n" + "\n".join(citation_lines)
    confidence_score, confidence_label = compute_confidence(results)
    return answer_with_citations, results, (confidence_score, confidence_label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="Path to FAISS .index file")
    parser.add_argument("--metadata", required=True, help="Path to matching .metadata.jsonl file")
    parser.add_argument("--query", default=None, help="A single question to ask")
    parser.add_argument("--interactive", action="store_true", help="Ask questions in a loop")
    parser.add_argument("--embed-model", default="bge-m3", choices=list(MODEL_MAP.keys()),
                         help="MUST match the model used in embed_and_index.py")
    parser.add_argument("--gen-model", default="llama3.1:8b",
                         help="Ollama model to use for answer generation "
                              "(must already be pulled, e.g. 'ollama pull llama3.1:8b')")
    parser.add_argument("--ollama-host", default="http://localhost:11434",
                         help="Ollama server address (default: local install)")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not args.query and not args.interactive:
        print("[!] Provide --query \"...\" or use --interactive")
        sys.exit(1)

    print(f"[*] Loading FAISS index: {args.index}")
    index = load_faiss_index(args.index)
    metadata = load_metadata(args.metadata)
    print(f"[*] Loaded {len(metadata)} chunk records")

    print(f"[*] Loading embedding model: {MODEL_MAP[args.embed_model]}")
    embed_model = load_embedding_model(args.embed_model)

    def ask(q):
        try:
            answer, results, (confidence_score, confidence_label) = answer_question(
                q, embed_model, index, metadata,
                top_k=args.top_k, gen_model=args.gen_model, ollama_host=args.ollama_host,
            )
        except RuntimeError as e:
            print(f"\n[!] {e}")
            return
        print("\n" + "=" * 85)
        print("ANSWER:")
        print(answer)
        print("-" * 85)
        print(f"Confidence: {confidence_label} ({confidence_score:.3f})  "
              f"[based on retrieval similarity, not verified accuracy -- "
              f"see the evaluation set for measured accuracy]")
        print("=" * 85)

    if args.interactive:
        print("\nInteractive mode. Type a question, or 'quit' to exit.")
        while True:
            q = input("\n> ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            if q:
                ask(q)
    else:
        ask(args.query)


if __name__ == "__main__":
    main()