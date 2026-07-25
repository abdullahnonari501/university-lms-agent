"""Phase 1 scraper: crawl giki.edu.pk's public site into data/raw/*.txt + manifest.json."""

import json
import re
import ssl
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser
from urllib.parse import urljoin, urlparse, urldefrag

import certifi
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter

BASE_URL = "https://giki.edu.pk/"
USER_AGENT = "university-lms-agent-bot/0.1 (+https://github.com/abdullahnonari501/university-lms-agent)"
MAX_DEPTH = 2
MAX_PAGES = 20
REQUEST_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 10

REPO_ROOT = Path(__file__).resolve().parent.parent
INTERMEDIATE_CERT = REPO_ROOT / "certs" / "rapidssl_intermediate.crt"
OUTPUT_DIR = REPO_ROOT / "data" / "raw"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

# giki.edu.pk's server doesn't send its intermediate cert (RapidSSL TLS RSA CA
# G1 -> DigiCert Global Root G2), so plain certifi verification fails even
# though the leaf cert is legitimate. Load certifi's roots plus the vendored
# intermediate so the chain verifies without disabling verification.
def build_session() -> requests.Session:
    ctx = ssl.create_default_context(cafile=certifi.where())
    ctx.load_verify_locations(cafile=str(INTERMEDIATE_CERT))

    class SSLContextAdapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            kwargs["ssl_context"] = ctx
            return super().init_poolmanager(*args, **kwargs)

    session = requests.Session()
    session.mount("https://", SSLContextAdapter())
    session.headers["User-Agent"] = USER_AGENT
    return session


def load_robots(session: requests.Session, base_url: str) -> robotparser.RobotFileParser:
    rp = robotparser.RobotFileParser()
    try:
        resp = session.get(urljoin(base_url, "/robots.txt"), timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        rp.parse(resp.text.splitlines())
    except requests.RequestException:
        rp.parse([])  # no robots.txt reachable -> default allow
    return rp


def slugify(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if not path:
        return "index"
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", path).strip("-").lower()
    return slug or "index"


def extract_title_and_text(soup: BeautifulSoup) -> tuple[str, str]:
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    # GIKI's WordPress theme (Kingster) duplicates the full nav menu in a
    # hidden mobile-menu <div> that sits outside any <nav>/<header> tag, so
    # tag-stripping alone lets it leak into every page's extracted text.
    # #kingster-page-wrapper holds the actual per-page content and is present
    # across every template checked (homepage, blog post, static page) --
    # prefer it, and fall back to tag-stripping the whole doc if it's absent.
    content = soup.find(id="kingster-page-wrapper") or soup

    for tag in content(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()

    text = content.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    return title, cleaned


def extract_same_domain_links(soup: BeautifulSoup, page_url: str, domain: str) -> list[str]:
    links = []
    for a in soup.find_all("a", href=True):
        absolute = urljoin(page_url, a["href"])
        absolute, _ = urldefrag(absolute)  # drop #fragments
        if urlparse(absolute).netloc == domain and absolute.startswith(("http://", "https://")):
            links.append(absolute)
    return links


def crawl() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session()
    domain = urlparse(BASE_URL).netloc
    robots = load_robots(session, BASE_URL)

    queue = deque([(BASE_URL, 0)])
    seen: set[str] = {BASE_URL}  # marked at discovery time to avoid re-queuing the same URL
    manifest = []

    while queue and len(manifest) < MAX_PAGES:
        url, depth = queue.popleft()

        if not robots.can_fetch(USER_AGENT, url):
            print(f"skip (robots disallow): {url}")
            continue

        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"skip (fetch error): {url} ({exc})")
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        title, text = extract_title_and_text(soup)

        slug = slugify(url)
        out_path = OUTPUT_DIR / f"{slug}.txt"
        out_path.write_text(text, encoding="utf-8")

        manifest.append(
            {
                "url": url,
                "title": title,
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "slug": slug,
            }
        )
        print(f"saved ({len(manifest)}/{MAX_PAGES}): {url} -> {out_path.name}")

        if depth < MAX_DEPTH:
            for link in extract_same_domain_links(soup, url, domain):
                if link not in seen:
                    seen.add(link)
                    queue.append((link, depth + 1))

        time.sleep(REQUEST_DELAY_SECONDS)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDone. {len(manifest)} pages saved to {OUTPUT_DIR}, manifest at {MANIFEST_PATH}")


if __name__ == "__main__":
    crawl()
