# ==============================================================================
# searxng_scraper.py
# ==============================================================================
# Scrapes trading strategy documents using a local SearXNG metasearch instance.
#
# SearXNG aggregates results from Google, Bing, DuckDuckGo, etc. without
# tracking. Runs locally via Docker.
#
# Docker setup:
#     docker run -d --name searxng -p 8080:8080 \
#         -v ./searxng:/etc/searxng \
#         -e SEARXNG_BASE_URL=http://localhost:8080/ \
#         searxng/searxng:latest
#
# Usage:
#     from searxng_scraper import SearXNGScraper
#
#     scraper = SearXNGScraper()
#     results = scraper.search("momentum trading strategy backtest")
#     documents = scraper.fetch_documents(results)
#
# ==============================================================================

import time
import hashlib
import logging
import requests
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, urljoin

from discovery_config import DISCOVERY_CONFIG as cfg, CUSTOM_QUERIES_FILE
from research_db import ResearchDatabase

logger = logging.getLogger(__name__)


class SearXNGScraper:
    """
    Metasearch scraper using a local SearXNG instance.

    Flow:
        1. Send search queries to SearXNG JSON API
        2. Collect unique URLs from results
        3. Fetch full page content for each URL
        4. Store raw documents in research database
    """

    def __init__(self, db: Optional[ResearchDatabase] = None):
        self.base_url = cfg.searxng.base_url.rstrip("/")
        self.search_endpoint = f"{self.base_url}/search"
        self.max_results = cfg.searxng.max_results_per_query
        self.delay = cfg.searxng.request_delay
        self.timeout = cfg.searxng.timeout
        self.db = db or ResearchDatabase()

        # Track URLs we've already seen this session
        self._seen_urls: set = set()

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "TradingLab-Research/1.0",
            "Accept": "application/json",
        })

        # Stats
        self.stats = {
            "queries_run": 0,
            "results_found": 0,
            "documents_fetched": 0,
            "documents_saved": 0,
            "duplicates_skipped": 0,
            "errors": 0,
        }

    # ==========================================================================
    # SEARCH
    # ==========================================================================

    def search(self, query: str, source_type: str = "general",
               source_bias: str = "unknown") -> List[Dict[str, Any]]:
        """
        Search SearXNG for a query and return results.

        Args:
            query: Search query string
            source_type: Expected source type (arxiv, github, reddit, etc.)
            source_bias: Bias tag (academic, retail, institutional)

        Returns:
            List of result dicts with keys: url, title, snippet, source_type, source_bias
        """
        params = {
            "q": query,
            "format": cfg.searxng.output_format,
            "categories": cfg.searxng.categories,
            "language": cfg.searxng.language,
            "pageno": 1,
        }

        results = []
        try:
            logger.info(f"Searching: {query}")
            response = self.session.get(
                self.search_endpoint,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            for item in data.get("results", [])[:self.max_results]:
                url = item.get("url", "")
                if not url or url in self._seen_urls:
                    continue

                # Auto-detect source type from URL if not specified
                detected_type = self._detect_source_type(url)
                actual_type = detected_type if detected_type != "general" else source_type

                results.append({
                    "url": url,
                    "title": item.get("title", ""),
                    "snippet": item.get("content", ""),
                    "source_type": actual_type,
                    "source_bias": source_bias,
                    "search_query": query,
                    "engine": item.get("engine", "unknown"),
                })
                self._seen_urls.add(url)

            self.stats["queries_run"] += 1
            self.stats["results_found"] += len(results)
            logger.info(f"  Found {len(results)} new results")

        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to SearXNG at {self.base_url}. Is it running?")
            self.stats["errors"] += 1
        except requests.exceptions.Timeout:
            logger.error(f"SearXNG request timed out for: {query}")
            self.stats["errors"] += 1
        except Exception as e:
            logger.error(f"Search error for '{query}': {e}")
            self.stats["errors"] += 1

        return results

    def search_all_queries(self, queries: Optional[List[Dict[str, str]]] = None) -> List[Dict]:
        """
        Run all configured search queries.

        Args:
            queries: Optional list of query dicts. Defaults to SEARCH_QUERIES from config.

        Returns:
            Combined list of all search results.
        """
        queries = queries or cfg.search_queries

        # Also load custom queries from file if it exists
        custom = self._load_custom_queries()
        if custom:
            queries = queries + custom
            logger.info(f"Loaded {len(custom)} custom queries from {CUSTOM_QUERIES_FILE}")

        all_results = []
        total = len(queries)

        for i, q in enumerate(queries, 1):
            query_str = q.get("query", "")
            source_type = q.get("source_type", "general")
            bias = q.get("bias", "unknown")

            logger.info(f"[{i}/{total}] {query_str}")
            results = self.search(query_str, source_type=source_type, source_bias=bias)
            all_results.extend(results)

            # Rate limiting between queries
            if i < total:
                time.sleep(self.delay)

        logger.info(f"Search complete: {len(all_results)} total results from {total} queries")
        return all_results

    # ==========================================================================
    # FETCH DOCUMENTS
    # ==========================================================================

    def fetch_document(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Fetch full content from a URL.

        Args:
            result: Search result dict with url, title, source_type, etc.

        Returns:
            Document dict ready for database, or None on failure.
        """
        url = result.get("url", "")
        if not url:
            return None

        # Skip if already in database
        if self.db.document_exists(url=url):
            logger.debug(f"  Skipping (already in DB): {url}")
            self.stats["duplicates_skipped"] += 1
            return None

        # Skip known non-content URLs
        if self._should_skip_url(url):
            logger.debug(f"  Skipping (filtered): {url}")
            return None

        try:
            response = self.session.get(
                url,
                timeout=self.timeout,
                headers={"Accept": "text/html,application/xhtml+xml,text/plain"},
            )
            response.raise_for_status()

            # Extract text content
            content = self._extract_text(response)
            if not content:
                logger.debug(f"  No text content: {url}")
                return None

            # Length filter
            if len(content) < cfg.pipeline.min_document_length:
                logger.debug(f"  Too short ({len(content)} chars): {url}")
                return None

            if len(content) > cfg.pipeline.max_document_length:
                content = content[:cfg.pipeline.max_document_length]
                logger.debug(f"  Truncated to {cfg.pipeline.max_document_length} chars: {url}")

            document = {
                "url": url,
                "title": result.get("title", ""),
                "content": content,
                "source_type": result.get("source_type", "general"),
                "source_bias": result.get("source_bias", "unknown"),
                "search_query": result.get("search_query", ""),
            }

            self.stats["documents_fetched"] += 1
            return document

        except requests.exceptions.Timeout:
            logger.warning(f"  Timeout fetching: {url}")
            self.stats["errors"] += 1
        except requests.exceptions.HTTPError as e:
            logger.warning(f"  HTTP error {e.response.status_code}: {url}")
            self.stats["errors"] += 1
        except Exception as e:
            logger.warning(f"  Fetch error: {url} - {e}")
            self.stats["errors"] += 1

        return None

    def fetch_and_store(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        Fetch all search results and store in database.

        Args:
            results: List of search result dicts from search().

        Returns:
            List of doc_ids for successfully stored documents.
        """
        doc_ids = []
        total = len(results)

        for i, result in enumerate(results, 1):
            url = result.get("url", "")
            logger.info(f"[{i}/{total}] Fetching: {url[:80]}...")

            document = self.fetch_document(result)
            if document:
                # Check for content-level duplicates
                if self.db.document_exists(content=document["content"]):
                    logger.debug(f"  Content duplicate, skipping")
                    self.stats["duplicates_skipped"] += 1
                    continue

                doc_id = self.db.save_document(document)
                if doc_id:
                    doc_ids.append(doc_id)
                    self.stats["documents_saved"] += 1
                    logger.info(f"  Saved: {doc_id}")
                else:
                    self.stats["duplicates_skipped"] += 1

            # Rate limit between fetches
            if i < total:
                time.sleep(self.delay * 0.5)  # Faster than search rate limit

        return doc_ids

    # ==========================================================================
    # HELPERS
    # ==========================================================================

    @staticmethod
    def _detect_source_type(url: str) -> str:
        """Detect source type from URL domain."""
        domain = urlparse(url).netloc.lower()
        if "arxiv.org" in domain:
            return "arxiv"
        elif "ssrn.com" in domain:
            return "ssrn"
        elif "github.com" in domain:
            return "github"
        elif "reddit.com" in domain:
            return "reddit"
        elif "stackoverflow.com" in domain:
            return "forum"
        elif "medium.com" in domain:
            return "blog"
        elif "quantopian" in domain or "quantconnect" in domain:
            return "platform"
        elif "investopedia" in domain:
            return "educational"
        return "general"

    @staticmethod
    def _should_skip_url(url: str) -> bool:
        """Filter out URLs that won't have useful strategy content."""
        skip_domains = [
            "youtube.com", "youtu.be",
            "twitter.com", "x.com",
            "facebook.com", "instagram.com",
            "linkedin.com",
            "amazon.com", "ebay.com",
            "wikipedia.org",  # Too generic
            "investopedia.com/terms",  # Definitions, not strategies
        ]
        skip_extensions = [".pdf", ".zip", ".tar", ".gz", ".mp4", ".mp3", ".jpg", ".png"]

        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()

        for sd in skip_domains:
            if sd in domain:
                return True

        for ext in skip_extensions:
            if path.endswith(ext):
                return True

        return False

    @staticmethod
    def _extract_text(response: requests.Response) -> str:
        """Extract readable text from an HTTP response."""
        content_type = response.headers.get("Content-Type", "").lower()

        if "text/plain" in content_type:
            return response.text.strip()

        if "text/html" in content_type or "xhtml" in content_type:
            # Use basic HTML stripping - no heavy dependency
            # For better extraction, install beautifulsoup4 + lxml
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove script, style, nav, footer elements
                for tag in soup(["script", "style", "nav", "footer", "header",
                                 "aside", "form", "iframe", "noscript"]):
                    tag.decompose()

                # Get text
                text = soup.get_text(separator="\n", strip=True)

                # Clean up excessive whitespace
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                return "\n".join(lines)

            except ImportError:
                # Fallback: basic regex HTML stripping
                import re
                text = re.sub(r'<script[^>]*>.*?</script>', '', response.text, flags=re.DOTALL)
                text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text

        # JSON responses (e.g., GitHub API)
        if "json" in content_type:
            try:
                data = response.json()
                # Try common fields
                for field in ["body", "content", "text", "description", "readme"]:
                    if field in data and isinstance(data[field], str):
                        return data[field]
                return str(data)
            except Exception:
                return response.text

        return ""

    def _load_custom_queries(self) -> List[Dict[str, str]]:
        """Load custom queries from text file (one query per line)."""
        if not CUSTOM_QUERIES_FILE.exists():
            return []

        queries = []
        with open(CUSTOM_QUERIES_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    queries.append({
                        "query": line,
                        "source_type": "general",
                        "bias": "unknown",
                    })
        return queries

    # ==========================================================================
    # STATS
    # ==========================================================================

    def print_stats(self):
        """Print scraper session statistics."""
        s = self.stats
        print("\n" + "=" * 50)
        print("SEARXNG SCRAPER - SESSION STATS")
        print("=" * 50)
        print(f"Queries run:       {s['queries_run']}")
        print(f"Results found:     {s['results_found']}")
        print(f"Documents fetched: {s['documents_fetched']}")
        print(f"Documents saved:   {s['documents_saved']}")
        print(f"Duplicates skip:   {s['duplicates_skipped']}")
        print(f"Errors:            {s['errors']}")
        print("=" * 50)

    def health_check(self) -> bool:
        """Check if SearXNG is running and responding."""
        try:
            resp = self.session.get(
                f"{self.base_url}/healthz",
                timeout=5,
            )
            if resp.status_code == 200:
                print(f"✓ SearXNG is running at {self.base_url}")
                return True
        except Exception:
            pass

        # Try the main page as fallback
        try:
            resp = self.session.get(self.base_url, timeout=5)
            if resp.status_code == 200:
                print(f"✓ SearXNG is running at {self.base_url}")
                return True
        except Exception:
            pass

        print(f"✗ SearXNG is NOT responding at {self.base_url}")
        print(f"  Start it with: docker run -d --name searxng -p 8080:8080 searxng/searxng:latest")
        return False


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    scraper = SearXNGScraper()

    # Health check
    if not scraper.health_check():
        print("\nStart SearXNG first, then re-run this script.")
        exit(1)

    # Run all queries
    print("\nStarting full search...")
    results = scraper.search_all_queries()

    # Fetch and store
    print(f"\nFetching {len(results)} documents...")
    doc_ids = scraper.fetch_and_store(results)

    # Print stats
    scraper.print_stats()
    scraper.db.print_stats()
