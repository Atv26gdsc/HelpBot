"""Quick diagnostic: for each cleaned page, show whether a 'Table' line was found,
and print the first few lines of any page that has "Table" (case-insensitive)
ANYWHERE in the text, so we can see the exact formatting your PDFs actually use."""
import json
import re
import sys

TABLE_LINE_PATTERN = re.compile(r"^Table[:\s]", re.I)

path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/INSAT3D_Data_Products_Format.jsonl"

with open(path, encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        text = record["text"]
        lines = text.split("\n")
        matched = [l for l in lines if TABLE_LINE_PATTERN.match(l)]
        contains_word_table = any("table" in l.lower() for l in lines)
        if matched:
            print(f"Page {record['page']}: MATCHED -> {matched}")
        elif contains_word_table:
            offending = [l for l in lines if "table" in l.lower()]
            print(f"Page {record['page']}: word 'table' present but pattern did NOT match -> {offending}")