import os
import re
import sys
import pymupdf
from tabulate import tabulate

# 6. Ensure Windows Unicode compatibility by configuring sys.stdout
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

PDF_DIR = "data/raw/pdfs"

def clean_text(text: str) -> str:
    """
    Cleans raw PDF text by removing MOSDAC website boilerplate
    and normalizing excessive whitespace while preserving technical terms.
    """
    # Remove MOSDAC website boilerplate (case-insensitive)
    cleaned = re.sub(r'www\.mosdac\.gov\.in|mosdac\.gov\.in', '', text, flags=re.IGNORECASE)
    
    # Normalize excessive horizontal whitespace (multiple spaces/tabs -> single space)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    
    # Normalize excessive vertical whitespace (multiple newlines -> single newline)
    cleaned = re.sub(r'\n+', '\n', cleaned)
    
    return cleaned.strip()

def verify_and_audit_pdfs():
    print("=" * 80)
    print("           MOSDAC PDF EXTRACTION & ACCURACY AUDIT")
    print("=" * 80)

    if not os.path.exists(PDF_DIR):
        print(f"[!] Directory not found: {PDF_DIR}")
        return

    # Find all .pdf files in the directory
    pdf_files = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')])

    if not pdf_files:
        print(f"[!] No PDF documents found in {PDF_DIR}")
        return

    audit_results = []

    for filename in pdf_files:
        filepath = os.path.join(PDF_DIR, filename)
        file_size_kb = os.path.getsize(filepath) / 1024
        size_str = f"{file_size_kb:.1f} KB"
        
        print(f"\n[*] Processing Document: {filename} ({size_str})")
        print("-" * 80)

        # 2. Check each file's first 4 bytes for the %PDF signature
        try:
            with open(filepath, "rb") as f:
                header = f.read(4)
                is_valid_pdf = header == b"%PDF"
        except Exception as e:
            print(f"  [-] Failed to read file signature: {e}")
            audit_results.append([filename, size_str, "0/0", "❌ File Error", "0.0%"])
            continue

        if not is_valid_pdf:
            print("  [❌] Invalid PDF format (missing %PDF signature). Possibly corrupt or HTML page.")
            audit_results.append([filename, size_str, "0/0", "❌ Corrupt / HTML Redirect", "0.0%"])
            continue

        # 3. Extract, clean, and preview text page-by-page
        try:
            doc = pymupdf.open(filepath)
            total_pages = len(doc)
            readable_pages = 0

            for i, page in enumerate(doc):
                page_num = i + 1
                raw_text = page.get_text()
                
                # Clean text
                cleaned_text = clean_text(raw_text)
                char_count = len(cleaned_text)

                # Determine status
                if char_count > 40:
                    readable_pages += 1
                    status_icon = "✅ Success"
                elif char_count > 0:
                    status_icon = "⚠️ Low Text"
                else:
                    status_icon = "❌ Empty"

                # Generate a single-line preview (remove newlines for clean printing)
                preview = cleaned_text.replace("\n", " ")
                if len(preview) > 80:
                    preview = preview[:77] + "..."
                elif not preview:
                    preview = "[No text extracted]"

                print(f"  Page {page_num:2d}: [{status_icon} | Char Count: {char_count:4d}] Preview: {preview}")

            # 4. Calculate extraction accuracy metric
            accuracy = (readable_pages / total_pages * 100) if total_pages > 0 else 0
            
            # Determine quality status
            if accuracy >= 80:
                quality = "✅ High Quality"
            elif accuracy > 0:
                quality = "⚠️ Low Text / Scanned"
            else:
                quality = "❌ Unreadable"

            audit_results.append([
                filename[:32] + ("..." if len(filename) > 32 else ""),
                size_str,
                f"{readable_pages}/{total_pages}",
                quality,
                f"{accuracy:.1f}%"
            ])
            
            doc.close()

        except Exception as e:
            print(f"  [❌] Error processing document: {e}")
            audit_results.append([filename[:32], size_str, "0/0", f"❌ Error: {str(e)[:15]}", "0.0%"])

    # 5. Print a final summary table using tabulate
    print("\n" + "=" * 80)
    print("                              FINAL AUDIT SUMMARY")
    print("=" * 80)
    headers = ["Filename", "Size", "Readable Pages", "Extraction Quality", "Accuracy %"]
    print(tabulate(audit_results, headers=headers, tablefmt="fancy_grid"))
    print("=" * 80 + "\n")

if __name__ == "__main__":
    verify_and_audit_pdfs()