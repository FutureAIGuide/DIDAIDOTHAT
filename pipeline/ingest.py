"""
pipeline/ingest.py — Stage T1: Signal Aggregation

Collects raw signals from RSS feeds, GitHub, YouTube, and arXiv.
Each signal is normalised into a common dict schema:

{
    "id":          str,        # unique identifier
    "title":       str,
    "summary":     str,
    "url":         str,
    "published":   str,        # ISO-8601
    "source_type": str,        # "rss" | "github" | "youtube" | "arxiv"
    "source_name": str,
    "keywords":    list[str],
    "entities":    list[str],
    "engagement":  float,      # normalised 0-1 proxy
}
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import feedparser
import requests

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "it", "its",
        "this", "that", "these", "those", "we", "our", "you", "your", "they",
        "their", "he", "she", "his", "her", "new", "using", "use", "based",
    }
)

_AI_ENTITIES: frozenset[str] = frozenset(
    {
        "openai", "google", "deepmind", "anthropic", "meta", "microsoft",
        "nvidia", "hugging face", "mistral", "stability ai", "midjourney",
        "runway", "sora", "gemini", "gpt", "claude", "llama", "stable diffusion",
        "dall-e", "whisper", "copilot", "chatgpt", "bard",
    }
)


def _make_id(url: str, title: str) -> str:
    return hashlib.sha1(f"{url}|{title}".encode()).hexdigest()[:16]


def _extract_keywords(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    return [w for w in words if w not in _STOP_WORDS][:20]


def _extract_entities(text: str) -> list[str]:
    lower = text.lower()
    return [e for e in _AI_ENTITIES if e in lower]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_date(value: Any) -> str:
    """Convert various date representations to ISO-8601 string."""
    if not value:
        return _now_iso()
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    if hasattr(value, "tm_year"):
        try:
            return datetime(*value[:6], tzinfo=timezone.utc).isoformat()
        except Exception:
            return _now_iso()
    return _now_iso()


# ---------------------------------------------------------------------------
# RSS
# ---------------------------------------------------------------------------

def _fetch_rss(feed_url: str) -> list[dict]:
    items: list[dict] = []
    try:
        parsed = feedparser.parse(feed_url, request_headers={"User-Agent": "DIDaidothat/1.0"})
        source_name = parsed.feed.get("title", urlparse(feed_url).netloc)
        for entry in parsed.entries[:10]:
            title = entry.get("title", "")
            summary = entry.get("summary", entry.get("description", ""))
            # Strip HTML tags
            summary = re.sub(r"<[^>]+>", " ", summary).strip()
            url = entry.get("link", "")
            published = _parse_date(entry.get("published_parsed") or entry.get("updated_parsed"))
            text = f"{title} {summary}"
            items.append(
                {
                    "id": _make_id(url, title),
                    "title": title,
                    "summary": summary[:500],
                    "url": url,
                    "published": published,
                    "source_type": "rss",
                    "source_name": source_name,
                    "keywords": _extract_keywords(text),
                    "entities": _extract_entities(text),
                    "engagement": 0.0,
                }
            )
    except Exception as exc:
        logger.warning("RSS fetch failed for %s: %s", feed_url, exc)
    return items


def fetch_rss_signals(feeds: list[str] | None = None) -> list[dict]:
    if feeds is None:
        feeds = config.RSS_FEEDS
    results: list[dict] = []
    for feed in feeds:
        results.extend(_fetch_rss(feed))
        time.sleep(0.2)
    logger.info("RSS: collected %d items", len(results))
    return results


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------

def _github_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "DIDaidothat/1.0"}
    if config.GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {config.GITHUB_TOKEN}"
    return headers


def _fetch_github_repos(query: str) -> list[dict]:
    items: list[dict] = []
    try:
        resp = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "sort": "updated", "order": "desc", "per_page": 10},
            headers=_github_headers(),
            timeout=10,
        )
        resp.raise_for_status()
        for repo in resp.json().get("items", []):
            title = repo.get("full_name", "")
            summary = repo.get("description") or ""
            url = repo.get("html_url", "")
            stars = repo.get("stargazers_count", 0)
            # normalise engagement: log-scale capped at 1.0
            engagement = min(1.0, (stars ** 0.5) / 100)
            pushed = repo.get("pushed_at", _now_iso())
            text = f"{title} {summary}"
            items.append(
                {
                    "id": _make_id(url, title),
                    "title": title,
                    "summary": summary[:500],
                    "url": url,
                    "published": pushed,
                    "source_type": "github",
                    "source_name": "GitHub",
                    "keywords": _extract_keywords(text),
                    "entities": _extract_entities(text),
                    "engagement": round(engagement, 4),
                }
            )
    except Exception as exc:
        logger.warning("GitHub fetch failed for query '%s': %s", query, exc)
    return items


def fetch_github_signals(queries: list[str] | None = None) -> list[dict]:
    if queries is None:
        queries = config.GITHUB_SEARCH_QUERIES
    results: list[dict] = []
    for query in queries:
        results.extend(_fetch_github_repos(query))
        time.sleep(0.5)
    logger.info("GitHub: collected %d items", len(results))
    return results


# ---------------------------------------------------------------------------
# YouTube
# ---------------------------------------------------------------------------

def _fetch_youtube_channel(channel_id: str) -> list[dict]:
    """Fetch latest videos from a YouTube channel via the Data API v3."""
    items: list[dict] = []
    if not config.YOUTUBE_API_KEY:
        return items
    try:
        resp = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "channelId": channel_id,
                "order": "date",
                "maxResults": 5,
                "type": "video",
                "key": config.YOUTUBE_API_KEY,
            },
            timeout=10,
        )
        resp.raise_for_status()
        for vid in resp.json().get("items", []):
            snippet = vid.get("snippet", {})
            title = snippet.get("title", "")
            summary = snippet.get("description", "")[:500]
            video_id = vid.get("id", {}).get("videoId", "")
            url = f"https://www.youtube.com/watch?v={video_id}"
            published = snippet.get("publishedAt", _now_iso())
            text = f"{title} {summary}"
            items.append(
                {
                    "id": _make_id(url, title),
                    "title": title,
                    "summary": summary,
                    "url": url,
                    "published": published,
                    "source_type": "youtube",
                    "source_name": snippet.get("channelTitle", channel_id),
                    "keywords": _extract_keywords(text),
                    "entities": _extract_entities(text),
                    "engagement": 0.0,  # view count not available in search
                }
            )
    except Exception as exc:
        logger.warning("YouTube fetch failed for channel %s: %s", channel_id, exc)
    return items


def fetch_youtube_signals(channel_ids: list[str] | None = None) -> list[dict]:
    if channel_ids is None:
        channel_ids = config.YOUTUBE_CHANNEL_IDS
    results: list[dict] = []
    for ch in channel_ids:
        results.extend(_fetch_youtube_channel(ch))
        time.sleep(0.3)
    logger.info("YouTube: collected %d items", len(results))
    return results


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------

_ARXIV_API = "https://export.arxiv.org/api/query"


def _fetch_arxiv_category(category: str, max_results: int = 10) -> list[dict]:
    items: list[dict] = []
    try:
        resp = requests.get(
            _ARXIV_API,
            params={
                "search_query": f"cat:{category}",
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "max_results": max_results,
            },
            timeout=15,
        )
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)
        for entry in feed.entries:
            title = entry.get("title", "").replace("\n", " ").strip()
            summary = entry.get("summary", "").replace("\n", " ").strip()[:500]
            url = entry.get("link", "")
            published = _parse_date(entry.get("published_parsed"))
            text = f"{title} {summary}"
            items.append(
                {
                    "id": _make_id(url, title),
                    "title": title,
                    "summary": summary,
                    "url": url,
                    "published": published,
                    "source_type": "arxiv",
                    "source_name": f"arXiv:{category}",
                    "keywords": _extract_keywords(text),
                    "entities": _extract_entities(text),
                    "engagement": 0.0,
                }
            )
    except Exception as exc:
        logger.warning("arXiv fetch failed for %s: %s", category, exc)
    return items


def fetch_arxiv_signals(categories: list[str] | None = None) -> list[dict]:
    if categories is None:
        categories = config.ARXIV_CATEGORIES
    results: list[dict] = []
    for cat in categories:
        results.extend(_fetch_arxiv_category(cat))
        time.sleep(0.5)
    logger.info("arXiv: collected %d items", len(results))
    return results


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

def ingest_all() -> list[dict]:
    """Run all ingestion sources and return a deduplicated list of signals."""
    all_items: list[dict] = []
    all_items.extend(fetch_rss_signals())
    all_items.extend(fetch_github_signals())
    all_items.extend(fetch_youtube_signals())
    all_items.extend(fetch_arxiv_signals())

    # Deduplicate by id
    seen: set[str] = set()
    unique: list[dict] = []
    for item in all_items:
        if item["id"] not in seen:
            seen.add(item["id"])
            unique.append(item)

    logger.info("Ingested %d unique signals total", len(unique))
    return unique
