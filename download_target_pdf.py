import os
import re
import sys
from curl_cffi import requests
import pymupdf  # Upgraded PyMuPDF
from tabulate import tabulate

# Ensure UTF-8 output on Windows terminal
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

PDF_URL = "https://www.mosdac.gov.in/docs/INSAT3D_Products.pdf"
DEST_DIR = os.path.join("data", "raw", "pdfs")
DEST_FILE = os.path.join(DEST_DIR, "INSAT3D_Data_Products_Format.pdf")

os.makedirs(DEST_DIR, exist_ok=True)

def download_target_pdf(url: str, output_path: str) -> bool:
    print("=" * 80)
    print("        TARGETED MOSDAC SCRAPER: INSAT-3D DATA PRODUCTS FORMAT")
    print("=" * 80)
    print(f"[*] Fetching target URL: {url}")
    
    session = requests.Session(impersonate="chrome120")
    try:
        res = session.get(url, timeout=60)
        if res.status_code == 200 and res.content[:4] == b"%PDF":
            with open(output_path, "wb") as f:
                f.write(res.content)
            print(f"[+] Download successful! Saved to: {output_path} ({len(res.content)/1024:.1f} KB)\n")
            return True
        else:
            print(f"[-] Invalid file: HTTP Status {res.status_code}, Binary header: {res.content[:4]}")
            return False
    except Exception as e:
        print(f"[-] Connection error: {e}")
        return False

def clean_technical_text(text: str) -> str:
    """
    Cleans excess whitespace & portal boilerplate while preserving 
    critical satellite metadata, resolutions, bands, and coordinates.
    """
    text = re.sub(r'www\.mosdac\.gov\.in', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()

def run_accuracy_audit(filepath: str):
    print("=" * 80)
    print("               DOCUMENT EXTRACTION & QUALITY AUDIT REPORT")
    print("=" * 80)

    if not os.path.exists(filepath):
        print(f"[!] File not found: {filepath}")
        return

    doc = pymupdf.open(filepath)
    total_pages = len(doc)
    readable_pages = 0
    total_chars = 0
    page_audit_log = []

    print(f"[*] Auditing {os.path.basename(filepath)} | Total Pages: {total_pages}\n")

    for page_idx, page in enumerate(doc, start=1):
        raw_text = page.get_text()
        cleaned = clean_technical_text(raw_text)
        char_count = len(cleaned)

        # Accuracy condition: meaningful page with extracted text
        if char_count > 40:
            readable_pages += 1
            total_chars += char_count
            status = "✅ Clean Text"
        else:
            status = "⚠️ Low Density / Image"

        # Show first 10 pages in live progress preview
        if page_idx <= 10 or page_idx == total_pages:
            preview = cleaned[:120].replace('\n', ' ') + ("..." if char_count > 120 else "")
            print(f"  Page {page_idx:02d}/{total_pages:02d} [{status}] ({char_count} chars): \"{preview}\"")

    if total_pages > 11:
        print(f"  ... [Audited remaining {total_pages - 11} pages successfully]")

    # Calculate overall extraction score
    accuracy_percentage = (readable_pages / total_pages * 100) if total_pages > 0 else 0
    quality_verdict = "✅ High Accuracy" if accuracy_percentage >= 80 else "⚠️ Partial / Scanned"

    # Final summary display
    print("\n" + "=" * 80)
    table_data = [
        [
            os.path.basename(filepath),
            f"{os.path.getsize(filepath)/1024:.1f} KB",
            f"{readable_pages}/{total_pages}",
            f"{total_chars:,}",
            quality_verdict,
            f"{accuracy_percentage:.1f}%"
        ]
    ]
    headers = ["Document Name", "Size", "Readable Pages", "Total Chars", "Quality Status", "Accuracy Score"]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

if __name__ == "__main__":
    success = download_target_pdf(PDF_URL, DEST_FILE)
    if success or os.path.exists(DEST_FILE):
        run_accuracy_audit(DEST_FILE)