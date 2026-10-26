import os
import re
import sys
import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# Ensure Windows console supports printing emojis and Unicode symbols
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

# Attempt to import curl_cffi for TLS Fingerprint impersonation (Anti-Blocking)
try:
    from curl_cffi import requests as curl_requests
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

# Attempt to import playwright for dynamic JS rendering fallback
try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

import requests  # Standard requests fallback

# Target portal configuration
BASE_URL = "https://mosdac.gov.in"
TARGET_PAGE = "https://mosdac.gov.in/announcements"
DOWNLOAD_DIR = "data/raw/pdfs"
TIMEOUT = 45  # Custom timeout in seconds
MAX_RETRIES = 5  # Custom retries

# Rich real-browser emulation headers for standard requests session fallback
REAL_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "Referer": "https://www.google.com/"
}

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def sanitize_filename(name: str) -> str:
    """Keep names clean and safe for file storage."""
    return re.sub(r'[^\w\-_\.]', '_', name)

def fetch_page_html(url: str) -> str:
    """
    Fetches targeting page HTML. Emulates a Chrome browser session via TLS 
    impersonation (curl_cffi), falling back to standard requests with rich headers,
    and finally trying Playwright if request is blocked.
    """
    # 1. Primary Method: curl_cffi for TLS Fingerprinting impersonation
    if HAS_CURL_CFFI:
        print("[*] Attempting TLS Impersonation via curl_cffi...")
        backoff = 2
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = curl_requests.get(url, impersonate="chrome120", timeout=TIMEOUT)
                if response.status_code == 200:
                    return response.text
                elif response.status_code in [500, 502, 503, 504]:
                    print(f"  [!] HTTP {response.status_code} server error (attempt {attempt}/{MAX_RETRIES}).")
                else:
                    print(f"  [!] HTTP {response.status_code} client error. Retrying...")
            except Exception as e:
                print(f"  [!] Connection failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                print(f"      Retrying in {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2

    # 2. Secondary Fallback: requests.Session with browser headers
    print("[*] Falling back to standard requests.Session with real-browser headers...")
    session = requests.Session()
    backoff = 2
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, headers=REAL_BROWSER_HEADERS, timeout=TIMEOUT)
            if response.status_code == 200:
                return response.text
            elif response.status_code in [500, 502, 503, 504]:
                print(f"  [!] HTTP {response.status_code} server error (attempt {attempt}/{MAX_RETRIES}).")
            else:
                print(f"  [!] HTTP {response.status_code} client error. Retrying...")
        except Exception as e:
            print(f"  [!] Connection failed (attempt {attempt}/{MAX_RETRIES}): {e}")
        if attempt < MAX_RETRIES:
            print(f"      Retrying in {backoff} seconds...")
            time.sleep(backoff)
            backoff *= 2

    # 3. Tertiary Fallback: Playwright Headless Browser
    if HAS_PLAYWRIGHT:
        print("[*] Attempting dynamic render fallback via Playwright...")
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
                page.goto(url, timeout=TIMEOUT * 1000)
                # Wait for some content to load
                page.wait_for_timeout(3000)
                html = page.content()
                browser.close()
                return html
        except Exception as e:
            print(f"  [-] Playwright execution failed: {e}")
            print("  [TIP] If you have missing browser binaries, run: playwright install chromium")
    
    raise ConnectionError("Failed to fetch target page HTML using all scraping fallbacks.")

def download_pdf(url: str, filepath: str):
    """
    Downloads a PDF document in 8 KB chunks with signature verification before saving, 
    and handles TLS fingerprinting/real-browser header emulation.
    """
    backoff = 2
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if HAS_CURL_CFFI:
                # curl_cffi doesn't directly support stream=True in standard iter_content, 
                # but we can get it as a streamed response
                res = curl_requests.get(url, impersonate="chrome120", timeout=TIMEOUT, stream=True)
            else:
                res = requests.get(url, headers=REAL_BROWSER_HEADERS, timeout=TIMEOUT, stream=True)
            
            if res.status_code == 200:
                # Read first chunk of 8 KB to check PDF signature
                first_chunk = next(res.iter_content(chunk_size=8192), b"")
                
                # Check first 4 bytes for '%PDF' signature
                if not first_chunk.startswith(b"%PDF"):
                    print(f"  [❌] Signature Check Failed for {url}: Invalid magic bytes. Rejecting file.")
                    return False
                
                # Signature check passed, write first chunk and stream the rest
                with open(filepath, "wb") as f:
                    f.write(first_chunk)
                    for chunk in res.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                print(f"  ✓ Downloaded successfully: {os.path.basename(filepath)}")
                return True
            else:
                print(f"  [!] HTTP {res.status_code} status on PDF request (attempt {attempt}/{MAX_RETRIES})")
        except Exception as e:
            print(f"  [!] PDF download failed (attempt {attempt}/{MAX_RETRIES}): {e}")
        
        if attempt < MAX_RETRIES:
            print(f"      Retrying download in {backoff} seconds...")
            time.sleep(backoff)
            backoff *= 2
            
    print(f"  [-] Failed to download PDF after {MAX_RETRIES} attempts.")
    return False

def scrape_pdfs(limit: int = 5):
    print("=" * 80)
    print("           MOSDAC RESILIENT ANTI-BLOCKING PDF SCRAPER")
    print("=" * 80)
    print(f"[*] Targeting Portal: {TARGET_PAGE}")
    print(f"[*] Destination Directory: {DOWNLOAD_DIR}")
    
    # Check dependencies availability and report status
    print(f"[*] TLS fingerprint impersonation (curl_cffi): {'Available' if HAS_CURL_CFFI else 'Unavailable (Falling back to requests)'}")
    print(f"[*] Headless browser emulation (Playwright): {'Available' if HAS_PLAYWRIGHT else 'Unavailable (Optional)'}")
    print("=" * 80)
    
    try:
        html = fetch_page_html(TARGET_PAGE)
    except Exception as e:
        print(f"\n[-] Critical Error fetching main page: {e}")
        print("\n[Installation Instructions to fix potential issues]:")
        print("  1. Run: pip install curl-cffi requests beautifulsoup4 playwright")
        print("  2. Run: playwright install chromium")
        return
        
    soup = BeautifulSoup(html, "html.parser")
    pdf_links = []
    
    # 3. Discover all PDF anchor links
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if ".pdf" in href.lower():
            # Resolve relative URLs
            full_url = urljoin(BASE_URL, href)
            label = a_tag.get_text(strip=True) or os.path.basename(urlparse(full_url).path)
            pdf_links.append((full_url, label))
            
    # Remove duplicate links
    unique_links = []
    seen_urls = set()
    for url, label in pdf_links:
        if url not in seen_urls:
            seen_urls.add(url)
            unique_links.append((url, label))
            
    print(f"[+] Discovered {len(unique_links)} unique PDF links. Starting download of top {min(limit, len(unique_links))}...")
    
    downloaded_count = 0
    for url, label in unique_links[:limit]:
        filename = sanitize_filename(os.path.basename(urlparse(url).path))
        if not filename.endswith(".pdf"):
            filename += ".pdf"
            
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        print(f"\n[*] Downloading PDF {downloaded_count+1}: {filename}")
        print(f"    Source URL: {url}")
        
        success = download_pdf(url, filepath)
        if success:
            downloaded_count += 1
            
    print("\n" + "=" * 80)
    print(f"Scraping completed! Successfully downloaded {downloaded_count} valid PDFs.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    scrape_pdfs(limit=5)
