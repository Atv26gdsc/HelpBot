"""
run_evaluation.py — Step 13: Measure your prototype's accuracy
==================================================================
Runs every question in eval_set.json through your RAG pipeline and
produces a results table you fill in by hand (source-correct / answer-
correct / grounded), matching the tracking table format from Step 13
of the guide. This does NOT auto-grade correctness -- judging whether
an answer is actually right still needs a human eye (that's the point:
"don't invent these numbers, your actual experiments should produce
them").

What this script DOES automate:
  - Running all 20-30 questions so you don't type them one by one
  - Recording what was retrieved (source/page) for each
  - Recording the confidence score for each
  - Flagging whether an out-of-scope question was correctly refused
    (this ONE column CAN be auto-checked, since "did it refuse" is
    objective)
  - Saving everything to a CSV you can open in Excel and fill in the
    remaining ✅/❌ columns by comparing against expected_information

Usage:
    python run_evaluation.py --eval-set data/evaluation/eval_set.json --index vector_db/mosdac.index --metadata vector_db/mosdac.metadata.jsonl
"""

import argparse
import csv
import json
import os
import sys

# reuse everything from rag_query.py rather than duplicating logic.
# Try same-folder import first (matches a flat project layout where all
# scripts sit together), falling back to the src/generation/ subfolder
# layout if that's how the project is organized instead.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "generation"))
from rag_query import (  # noqa: E402
    MODEL_MAP, load_metadata, load_faiss_index, load_embedding_model, answer_question,
)

REFUSAL_PHRASE = "I could not find sufficient information"


def score_answer(answer: str, expected_information, question_type: str, refused: bool):
    """
    Automated proxy accuracy score based on keyword/phrase presence.
    NOT a substitute for reading the answer -- this can be fooled by a
    fact appearing out of context, or miss a correct answer phrased
    differently than expected_information. Use it as a fast terminal
    signal, and still spot-check the actual answers, especially for
    comparison/multi-document questions where synthesis quality matters
    more than keyword presence.

    Matching is loose on purpose: each expected item is split into
    individual significant words (stopwords dropped), and counts as
    "found" if most of its words appear anywhere in the answer -- not
    requiring the exact phrase verbatim. E.g. expected "Imager payload"
    matches an answer containing "an Imager" and "payloads" separately.
    This trades some precision for not punishing correct answers that
    are just phrased differently than the expected_information string.

    Returns (score 0.0-1.0, matched_count, total_count).
    """
    if question_type == "out-of-scope":
        # For out-of-scope questions, "correct" literally means refused.
        return (1.0, 1, 1) if refused else (0.0, 0, 1)

    if not expected_information:
        return (0.0, 0, 0)

    STOPWORDS = {"a", "an", "the", "of", "in", "on", "is", "are", "and", "or",
                 "to", "for", "with", "at", "by"}

    answer_lower = answer.lower()
    matched = 0
    for phrase in expected_information:
        words = [w.strip(".,()") for w in phrase.lower().split()]
        significant_words = [w for w in words if w and w not in STOPWORDS]
        if not significant_words:
            continue
        hits = sum(1 for w in significant_words if w in answer_lower)
        # count as matched if at least 60% of the phrase's significant words appear
        if hits / len(significant_words) >= 0.6:
            matched += 1

    total = len(expected_information)
    return matched / total, matched, total


def load_eval_set(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default=os.path.join("data", "evaluation", "eval_set.json"))
    parser.add_argument("--index", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--embed-model", default="bge-m3", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--gen-model", default="llama3.1:8b")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default=os.path.join("data", "evaluation", "results.csv"))
    args = parser.parse_args()

    questions = load_eval_set(args.eval_set)
    print(f"[*] Loaded {len(questions)} evaluation questions")

    index = load_faiss_index(args.index)
    metadata = load_metadata(args.metadata)
    embed_model = load_embedding_model(args.embed_model)

    rows = []
    all_scores = []
    for i, q in enumerate(questions, start=1):
        print(f"\n[{i}/{len(questions)}] ({q['type']}) {q['question']}")

        try:
            answer, results, (conf_score, conf_label) = answer_question(
                q["question"], embed_model, index, metadata,
                top_k=args.top_k, gen_model=args.gen_model, ollama_host=args.ollama_host,
                verbose=False,
            )
        except RuntimeError as e:
            print(f"    [!] FAILED: {e}")
            rows.append({
                "id": q["id"], "type": q["type"], "question": q["question"],
                "expected_source": q.get("source") or "", "expected_page": q.get("page") or "",
                "expected_facts": "; ".join(q.get("expected_information", [])),
                "top_retrieved_source": "ERROR", "top_retrieved_page": "",
                "source_match_auto_hint": "", "confidence": "",
                "refused": "", "auto_check_refusal_correct": "",
                "auto_accuracy_score": "SKIPPED", "auto_accuracy_matched": "",
                "generated_answer": f"[FAILED: {e}]",
                "retrieved_correct_source": "", "answer_correct": "", "grounded": "",
                "notes": "Ollama error -- re-run this question individually later",
            })
            continue

        refused = REFUSAL_PHRASE in answer
        top_source = results[0]["source"] if results else ""
        top_page = results[0]["page"] if results else ""

        score, matched, total = score_answer(answer, q.get("expected_information", []),
                                              q["type"], refused)

        # The only column we can auto-check: for out-of-scope questions,
        # correct behavior IS refusing. For everything else, a human
        # needs to read the answer and compare to expected_information.
        if q["type"] == "out-of-scope":
            auto_correct_refusal = "YES" if refused else "NO (should have refused!)"
        else:
            auto_correct_refusal = "N/A"

        expected_source = q.get("source") or ""
        source_match_hint = ""
        if expected_source and expected_source != "multiple":
            source_match_hint = "YES" if top_source == expected_source else "CHECK"

        rows.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "expected_source": expected_source,
            "expected_page": q.get("page") or "",
            "expected_facts": "; ".join(q.get("expected_information", [])),
            "top_retrieved_source": top_source,
            "top_retrieved_page": top_page,
            "source_match_auto_hint": source_match_hint,
            "confidence": f"{conf_label} ({conf_score:.3f})",
            "refused": "YES" if refused else "NO",
            "auto_check_refusal_correct": auto_correct_refusal,
            "auto_accuracy_score": f"{score:.2f}",
            "auto_accuracy_matched": f"{matched}/{total}",
            "generated_answer": answer.replace("\n", " | "),
            # empty columns for YOU to fill in by reading the answer:
            "retrieved_correct_source": "",
            "answer_correct": "",
            "grounded": "",
            "notes": "",
        })

        print(f"    retrieved: {top_source} p.{top_page}  |  confidence: {conf_label} ({conf_score:.3f})")
        print(f"    refused: {'YES' if refused else 'NO'}"
              + (f"  [{auto_correct_refusal}]" if auto_correct_refusal != "N/A" else ""))
        print(f"    auto accuracy score: {score:.0%}  ({matched}/{total} expected facts found)")

        all_scores.append(score)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 85)
    print(f"✅ Ran {len(rows)} questions.")
    print(f"💾 Results saved to: {args.out}")
    print("\nNext steps:")
    print("  1. Open the CSV in Excel/Sheets.")
    print("  2. For each row, read 'generated_answer' against 'expected_facts'.")
    print("  3. Fill in: retrieved_correct_source (Y/N), answer_correct (Y/N), grounded (Y/N).")
    print("  4. Grounded = did the answer only use info actually in the cited excerpts,")
    print("     with no invented facts.")
    print("  5. Count your Y's per column to get your accuracy percentages -- that's your")
    print("     real, measured baseline (Step 13), not a guessed number.")

    out_of_scope_total = sum(1 for r in rows if r["type"] == "out-of-scope")
    out_of_scope_correct = sum(
        1 for r in rows if r["type"] == "out-of-scope" and r["auto_check_refusal_correct"] == "YES"
    )
    if out_of_scope_total:
        print(f"\nAuto-checked: out-of-scope refusal rate = "
              f"{out_of_scope_correct}/{out_of_scope_total} "
              f"({100 * out_of_scope_correct / out_of_scope_total:.0f}%)")

    # ---- Terminal accuracy summary (the number you asked for) ----
    print("\n" + "=" * 85)
    print("AUTO ACCURACY SCORE (keyword-match proxy -- not a substitute for reading answers)")
    print("=" * 85)

    by_type = {}
    for q, s in zip(questions, all_scores):
        by_type.setdefault(q["type"], []).append(s)

    for qtype, scores in by_type.items():
        avg = sum(scores) / len(scores)
        print(f"  {qtype:16s}: {avg:.0%}  ({len(scores)} questions)")

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print("-" * 85)
    print(f"  {'OVERALL':16s}: {overall:.0%}  ({len(all_scores)} questions)")
    print("=" * 85)
    print("Note: this score only checks whether expected keywords/phrases appear in the")
    print("answer text. It cannot judge synthesis quality (e.g. comparison questions that")
    print("state facts but never actually compare them) or catch a fact used out of context.")
    print("Treat it as a fast sanity signal, not your final reported accuracy.")


if __name__ == "__main__":
    main()