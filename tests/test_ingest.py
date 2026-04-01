"""
tests/test_ingest.py — Unit tests for pipeline/ingest.py (Stage T1)
"""
import pytest
import responses as responses_lib

from pipeline.ingest import (
    _extract_keywords,
    _extract_entities,
    _make_id,
    _parse_date,
    fetch_rss_signals,
    fetch_github_signals,
    ingest_all,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestExtractKeywords:
    def test_returns_list(self):
        result = _extract_keywords("OpenAI released a new language model today")
        assert isinstance(result, list)

    def test_filters_stop_words(self):
        result = _extract_keywords("the and or but in on at to for")
        assert result == []

    def test_minimum_word_length(self):
        result = _extract_keywords("AI is new and the big idea")
        # "big" and "idea" should pass; "AI", "is" filtered by length or stop words
        assert "idea" in result

    def test_max_results(self):
        long_text = " ".join(f"keyword{i}" for i in range(50))
        result = _extract_keywords(long_text)
        assert len(result) <= 20

    def test_deduplication_not_enforced(self):
        # keywords may repeat — that's fine for frequency counting
        result = _extract_keywords("agent agent agent")
        assert "agent" in result


class TestExtractEntities:
    def test_detects_known_entity(self):
        assert "openai" in _extract_entities("OpenAI released GPT-5 today")

    def test_multiple_entities(self):
        text = "Google DeepMind and Anthropic are competing with OpenAI"
        entities = _extract_entities(text)
        assert "google" in entities
        assert "anthropic" in entities
        assert "openai" in entities

    def test_empty_text(self):
        assert _extract_entities("") == []

    def test_no_match(self):
        assert _extract_entities("weather forecast for tomorrow") == []


class TestMakeId:
    def test_deterministic(self):
        assert _make_id("https://example.com", "Title") == _make_id(
            "https://example.com", "Title"
        )

    def test_different_inputs_differ(self):
        assert _make_id("https://a.com", "A") != _make_id("https://b.com", "B")

    def test_length(self):
        assert len(_make_id("url", "title")) == 16


class TestParseDate:
    def test_iso_string_passthrough(self):
        iso = "2024-01-15T12:00:00+00:00"
        assert _parse_date(iso) == iso

    def test_time_tuple(self):
        import time
        tt = time.strptime("2024-03-01", "%Y-%m-%d")
        result = _parse_date(tt)
        assert "2024" in result

    def test_none_returns_something(self):
        result = _parse_date(None)
        assert isinstance(result, str)
        assert len(result) > 0


class TestFetchRssSignals:
    def test_parses_valid_feed(self, mocker):
        # feedparser uses its own HTTP stack; mock parse() with dict-like entries
        entry1 = {
            "title": "New Language Model Released",
            "link": "https://example.com/article1",
            "summary": "A new language model from OpenAI challenges competitors.",
            "published_parsed": None,
            "updated_parsed": None,
        }
        entry2 = {
            "title": "AI Robotics Breakthrough",
            "link": "https://example.com/article2",
            "summary": "Robotics gets smarter with new AI agent framework.",
            "published_parsed": None,
            "updated_parsed": None,
        }

        import feedparser as fp_module
        import types

        feed_mock = types.SimpleNamespace(
            feed={"title": "Test AI Feed"},
            entries=[entry1, entry2],
        )
        mocker.patch("pipeline.ingest.feedparser.parse", return_value=feed_mock)
        items = fetch_rss_signals(feeds=["https://test-feed.example.com/rss"])
        assert len(items) == 2
        assert items[0]["source_type"] == "rss"
        assert items[0]["title"] == "New Language Model Released"
        assert "openai" in items[0]["entities"]

    def test_handles_feed_error_gracefully(self, mocker):
        mocker.patch(
            "pipeline.ingest.feedparser.parse", side_effect=Exception("Connection refused")
        )
        items = fetch_rss_signals(feeds=["https://bad-feed.example.com/rss"])
        assert items == []

    def test_empty_feeds_list(self):
        items = fetch_rss_signals(feeds=[])
        assert items == []


class TestFetchGithubSignals:
    @responses_lib.activate
    def test_parses_github_response(self):
        payload = {
            "items": [
                {
                    "full_name": "org/ai-agent",
                    "description": "An AI agent framework for production use",
                    "html_url": "https://github.com/org/ai-agent",
                    "stargazers_count": 1000,
                    "pushed_at": "2024-01-15T10:00:00Z",
                }
            ]
        }
        responses_lib.add(
            responses_lib.GET,
            "https://api.github.com/search/repositories",
            json=payload,
        )
        items = fetch_github_signals(queries=["AI agent"])
        assert len(items) == 1
        assert items[0]["source_type"] == "github"
        assert items[0]["engagement"] > 0

    @responses_lib.activate
    def test_handles_github_error_gracefully(self):
        responses_lib.add(
            responses_lib.GET,
            "https://api.github.com/search/repositories",
            status=403,
        )
        items = fetch_github_signals(queries=["AI agent"])
        assert items == []

    def test_empty_queries_list(self):
        items = fetch_github_signals(queries=[])
        assert items == []


class TestIngestAll:
    def test_returns_list(self, mocker):
        mocker.patch("pipeline.ingest.fetch_rss_signals", return_value=[])
        mocker.patch("pipeline.ingest.fetch_github_signals", return_value=[])
        mocker.patch("pipeline.ingest.fetch_youtube_signals", return_value=[])
        mocker.patch("pipeline.ingest.fetch_arxiv_signals", return_value=[])
        result = ingest_all()
        assert isinstance(result, list)

    def test_deduplicates_items(self, mocker):
        item = {
            "id": "abc123",
            "title": "Test",
            "summary": "",
            "url": "https://example.com",
            "published": "2024-01-01T00:00:00+00:00",
            "source_type": "rss",
            "source_name": "Test",
            "keywords": [],
            "entities": [],
            "engagement": 0.0,
        }
        mocker.patch("pipeline.ingest.fetch_rss_signals", return_value=[item, item])
        mocker.patch("pipeline.ingest.fetch_github_signals", return_value=[item])
        mocker.patch("pipeline.ingest.fetch_youtube_signals", return_value=[])
        mocker.patch("pipeline.ingest.fetch_arxiv_signals", return_value=[])
        result = ingest_all()
        # Only one unique item despite duplicates across calls
        assert len(result) == 1
