from __future__ import annotations

import logging
import time

import httpx

from tachyrag.config import CF_ACCOUNT_ID, CF_API_TOKEN, CF_CRAWL_POLL_INTERVAL, CF_CRAWL_POLL_MAX

log = logging.getLogger(__name__)

_BASE = "https://api.cloudflare.com/client/v4/accounts"


def _cf_configured() -> bool:
    return bool(CF_ACCOUNT_ID and CF_API_TOKEN)


def _headers() -> dict:
    return {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}


def _check_configured() -> None:
    if not _cf_configured():
        raise ValueError("CF_ACCOUNT_ID and CF_API_TOKEN must be set for Cloudflare crawling")


def start_crawl(
    url: str,
    *,
    limit: int = 50,
    depth: int = 3,
    render: bool = False,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> str:
    """Start a Cloudflare crawl job. Returns job ID."""
    _check_configured()
    body: dict = {
        "url": url,
        "limit": limit,
        "depth": depth,
        "render": render,
        "formats": ["markdown"],
        "crawlPurposes": ["search", "ai-input"],
    }
    opts: dict = {}
    if include_patterns:
        opts["includePatterns"] = include_patterns
    if exclude_patterns:
        opts["excludePatterns"] = exclude_patterns
    if opts:
        body["options"] = opts

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(f"{_BASE}/{CF_ACCOUNT_ID}/browser-rendering/crawl", headers=_headers(), json=body)
        resp.raise_for_status()
        data = resp.json()

    if not data.get("success"):
        raise RuntimeError(f"Cloudflare crawl start failed: {data}")

    job_id = data["result"]
    log.info("Crawl job started: %s for %s (limit=%d, depth=%d)", job_id, url, limit, depth)
    return job_id


def poll_crawl(job_id: str) -> dict:
    """Poll until crawl job completes. Returns full result dict."""
    _check_configured()
    url = f"{_BASE}/{CF_ACCOUNT_ID}/browser-rendering/crawl/{job_id}"

    with httpx.Client(timeout=30.0) as client:
        for attempt in range(CF_CRAWL_POLL_MAX):
            resp = client.get(f"{url}?limit=1", headers=_headers())
            resp.raise_for_status()
            data = resp.json()
            status = data["result"]["status"]

            if status != "running":
                break
            log.info("Crawl %s still running (attempt %d/%d)", job_id, attempt + 1, CF_CRAWL_POLL_MAX)
            time.sleep(CF_CRAWL_POLL_INTERVAL)
        else:
            raise TimeoutError(f"Crawl job {job_id} did not complete within {CF_CRAWL_POLL_MAX * CF_CRAWL_POLL_INTERVAL}s")

        if status != "completed":
            raise RuntimeError(f"Crawl job {job_id} ended with status: {status}")

        # Fetch all completed records (paginated)
        records = []
        cursor = None
        while True:
            params = "?status=completed&limit=100"
            if cursor:
                params += f"&cursor={cursor}"
            resp = client.get(f"{url}{params}", headers=_headers())
            resp.raise_for_status()
            page = resp.json()["result"]
            records.extend(page.get("records", []))
            cursor = page.get("cursor")
            if not cursor:
                break

    log.info("Crawl %s completed: %d pages", job_id, len(records))
    return {"job_id": job_id, "status": status, "records": records}


def crawl_and_collect(
    url: str,
    *,
    limit: int = 50,
    depth: int = 3,
    render: bool = False,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[dict]:
    """Crawl a URL and return pages. Uses Cloudflare if configured, else local scraper."""
    if _cf_configured():
        job_id = start_crawl(url, limit=limit, depth=depth, render=render,
                             include_patterns=include_patterns, exclude_patterns=exclude_patterns)
        result = poll_crawl(job_id)
        pages = []
        for r in result["records"]:
            md = r.get("markdown", "")
            if not md or len(md.strip()) < 50:
                continue
            pages.append({
                "url": r.get("url", url),
                "markdown": md,
                "title": r.get("metadata", {}).get("title", ""),
            })
        return pages
    else:
        return _local_crawl(url, limit=limit, depth=depth)


# ---------------------------------------------------------------------------
# Local scraper fallback (no Cloudflare dependency)
# ---------------------------------------------------------------------------

def _local_crawl(start_url: str, limit: int = 50, depth: int = 3) -> list[dict]:
    """Simple BFS web scraper using httpx + BeautifulSoup. No external API needed."""
    try:
        from bs4 import BeautifulSoup
        import html2text
    except ImportError:
        raise ValueError("Local crawl requires: pip install beautifulsoup4 html2text")

    from urllib.parse import urljoin, urlparse
    import urllib.robotparser

    parsed_start = urlparse(start_url)
    base_domain = parsed_start.netloc

    # Check robots.txt
    rp = urllib.robotparser.RobotFileParser()
    try:
        rp.set_url(f"{parsed_start.scheme}://{base_domain}/robots.txt")
        rp.read()
    except Exception:
        rp = None

    h2t = html2text.HTML2Text()
    h2t.ignore_links = False
    h2t.ignore_images = True
    h2t.body_width = 0

    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(start_url, 0)]
    pages: list[dict] = []

    with httpx.Client(timeout=15.0, follow_redirects=True, headers={"User-Agent": "TachyGraphCrawler/1.0"}) as client:
        while queue and len(pages) < limit:
            url, current_depth = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            if rp and not rp.can_fetch("TachyGraphCrawler", url):
                log.debug("Robots.txt disallows: %s", url)
                continue

            try:
                resp = client.get(url)
                if resp.status_code != 200 or "text/html" not in resp.headers.get("content-type", ""):
                    continue
            except Exception as e:
                log.debug("Failed to fetch %s: %s", url, e)
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.string.strip() if soup.title and soup.title.string else ""

            # Remove nav, footer, script, style
            for tag in soup.find_all(["nav", "footer", "script", "style", "header"]):
                tag.decompose()

            markdown = h2t.handle(str(soup)).strip()
            if len(markdown) >= 50:
                pages.append({"url": url, "markdown": markdown, "title": title})

            # Follow links (same domain only)
            if current_depth < depth:
                for a in soup.find_all("a", href=True):
                    href = urljoin(url, a["href"])
                    parsed = urlparse(href)
                    if parsed.netloc == base_domain and href not in visited and parsed.scheme in ("http", "https"):
                        clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                        if clean not in visited:
                            queue.append((clean, current_depth + 1))

    log.info("Local crawl of %s: %d pages collected", start_url, len(pages))
    return pages
