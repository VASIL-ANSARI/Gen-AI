import os, time, json, hashlib, logging
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import tldextract
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

os.makedirs("company_docs", exist_ok=True)
os.makedirs("scraped_pages", exist_ok=True)
os.makedirs("crawler_logs", exist_ok=True)

logger = logging.getLogger("site_crawler")
handler = logging.FileHandler("crawler_logs/crawler.log")
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def is_allowed_by_robots(base_url, user_agent="*"):
    try:
        robots_url = urljoin(base_url, "/robots.txt")
        r = requests.get(robots_url, timeout=5)
        if r.status_code != 200:
            return True
        from urllib.robotparser import RobotFileParser
        rp = RobotFileParser()
        rp.parse(r.text.splitlines())
        return rp.can_fetch(user_agent, base_url)
    except Exception as e:
        logger.warning(f"robots.txt check failed: {e}")
        return True


def fetch_sitemap_urls(base_url):
    urls = set()
    try:
        sitemap_url = urljoin(base_url, "/sitemap.xml")
        r = requests.get(sitemap_url, timeout=8)
        if r.status_code != 200:
            return urls
        try:
            soup = BeautifulSoup(r.text, "xml")   # requires lxml
        except Exception:
            soup = BeautifulSoup(r.text, "html.parser")
        for loc in soup.find_all("loc"):
            href = loc.get_text().strip()
            if href.startswith(base_url):
                urls.add(href)
        logger.info(f"Sitemap parsed: {len(urls)} URLs")
    except Exception as e:
        logger.info(f"sitemap fetch failed: {e}")
    return urls


def is_same_domain(url_a, url_b):
    ea = tldextract.extract(url_a)
    eb = tldextract.extract(url_b)
    return ea.domain == eb.domain and ea.suffix == eb.suffix


def extract_links_from_html(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    pdfs = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("#", "mailto:", "tel:")):
            continue
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        norm = parsed._replace(fragment="").geturl()
        if is_same_domain(norm, base_url):
            links.add(norm)
        if norm.lower().endswith(".pdf"):
            pdfs.add(norm)
    return links, pdfs


def render_page_playwright(url, timeout=20000, headless=True):
    rendered_html = ""
    rendered_text = ""
    pdfs = set()
    endpoints = set()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()
        def on_request(request):
            try:
                endpoint = request.url
                if endpoint.lower().endswith(".pdf"):
                    pdfs.add(endpoint)
                if request.resource_type in ("xhr", "fetch") or "api" in endpoint.lower():
                    endpoints.add(endpoint)
            except Exception:
                pass
        page.on("request", on_request)
        try:
            page.goto(url, wait_until="networkidle", timeout=timeout)
            page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(0.4)
            rendered_html = page.content()
            rendered_text = page.inner_text("body")
        except PlaywrightTimeout:
            logger.warning(f"Timeout loading {url}")
        except Exception as e:
            logger.warning(f"Playwright error on {url}: {e}")
        finally:
            try:
                context.close()
            except:
                pass
            browser.close()
    return rendered_html, rendered_text, pdfs, endpoints


def crawl_site(base_url, max_pages=300, delay=1.0):
    pages = {}
    to_crawl = []
    visited = set()
    sitemap_urls = fetch_sitemap_urls(base_url)
    if sitemap_urls:
        to_crawl.extend(list(sitemap_urls))
    else:
        to_crawl.append(base_url)
    if not is_allowed_by_robots(base_url):
        logger.warning("Crawling blocked by robots.txt")
        return pages
    while to_crawl and len(visited) < max_pages:
        url = to_crawl.pop(0)
        if url in visited:
            continue
        visited.add(url)
        logger.info(f"Crawling {url} ({len(visited)}/{max_pages})")
        try:
            rendered_html, rendered_text, pdfs_js, endpoints_js = render_page_playwright(url)
            pages[url] = {"html": rendered_html, "text": rendered_text, "pdfs": set(pdfs_js), "endpoints": set(endpoints_js)}
            links, pdfs_html = extract_links_from_html(rendered_html, base_url)
            pages[url]["pdfs"].update(pdfs_html)
            for link in links:
                if link not in visited and link not in to_crawl:
                    to_crawl.append(link)
        except Exception as e:
            logger.warning(f"Error crawling {url}: {e}")
        time.sleep(delay)
    return pages


def save_scraped_pages(pages, out_dir="scraped_pages"):
    os.makedirs(out_dir, exist_ok=True)
    manifest = {}
    for url, meta in pages.items():
        text = meta.get("text", "").strip()
        if not text:
            continue
        h = hashlib.sha1(url.encode()).hexdigest()[:10]
        filename = f"{h}.txt"
        filepath = os.path.join(out_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"URL: {url}\n\n{text}")
        manifest[url] = {"file": filepath, "pdfs": list(meta["pdfs"]), "endpoints": list(meta["endpoints"])}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved {len(manifest)} pages")
    return manifest


def download_pdfs(manifest, out_pdf_dir="company_docs"):
    os.makedirs(out_pdf_dir, exist_ok=True)
    downloaded = []
    for url, meta in manifest.items():
        for pdf_url in meta.get("pdfs", []):
            try:
                fname = os.path.basename(urlparse(pdf_url).path)
                if not fname:
                    fname = hashlib.sha1(pdf_url.encode()).hexdigest()[:10] + ".pdf"
                out_path = os.path.join(out_pdf_dir, fname)
                if os.path.exists(out_path):
                    continue
                r = requests.get(pdf_url, timeout=15)
                if r.status_code == 200 and "pdf" in r.headers.get("content-type", "").lower():
                    with open(out_path, "wb") as fh:
                        fh.write(r.content)
                    downloaded.append(out_path)
                    logger.info(f"Downloaded PDF: {pdf_url}")
            except Exception as e:
                logger.warning(f"Failed PDF download {pdf_url}: {e}")
    return downloaded


def aggregate_endpoints(manifest):
    endpoints = set()
    for meta in manifest.values():
        endpoints.update(meta.get("endpoints", []))
    return sorted(endpoints)


def discover_site(base_url, max_pages=200, delay=1.0):
    pages = crawl_site(base_url, max_pages=max_pages, delay=delay)
    manifest = save_scraped_pages(pages)
    pdfs = download_pdfs(manifest)
    endpoints = aggregate_endpoints(manifest)
    return {"manifest": manifest, "pdfs": pdfs, "endpoints": endpoints}
