"""
tests/test_generate.py — Unit tests for pipeline/generate.py and pipeline/atomize.py
"""
import json
import os
import pytest

from pipeline.generate import (
    _slugify,
    _parse_article_response,
    _fallback_body,
    generate_article,
)
from pipeline.atomize import (
    _parse_atoms,
    _fallback_atoms,
    atomize_content,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cluster(topic="AI Agents"):
    return {
        "topic": topic,
        "status": "emerging",
        "momentum_score": 0.72,
        "velocity": 3.0,
        "source_diversity": 2,
        "first_seen": "2024-01-01T00:00:00+00:00",
        "keywords": ["agent", "llm", "model"],
        "entities": ["openai"],
        "items": [
            {
                "id": "a1",
                "title": "AI agents are taking over",
                "url": "https://example.com/1",
                "summary": "Agents are everywhere.",
                "published": "2024-01-15T00:00:00+00:00",
                "source_type": "rss",
                "source_name": "Test",
                "keywords": ["agent"],
                "entities": ["openai"],
                "engagement": 0.5,
            }
        ],
        "narrative": {
            "trend": "AI agents are becoming mainstream",
            "core_question": "Why are agents suddenly everywhere?",
            "why_now": "LLMs are now capable enough for agentic tasks.",
            "contradiction": "More autonomy may mean less control.",
            "prediction": "Agents will handle 30% of software tasks by year end.",
        },
    }


def _make_article(slug="test-article"):
    cluster = _make_cluster()
    return {
        "title": "AI Agents Are Reshaping Development",
        "slug": slug,
        "meta_desc": "A look at the agent revolution.",
        "body": "## AI Agents\n\nThis doesn't add up...\n\nIf you're building with AI, this matters.",
        "cluster": cluster,
        "published": "2024-01-15T12:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# Slugify
# ---------------------------------------------------------------------------

class TestSlugify:
    def test_basic(self):
        assert _slugify("Hello World") == "hello-world"

    def test_special_chars(self):
        assert _slugify("AI: The Future?!") == "ai-the-future"

    def test_max_length(self):
        long_title = "a" * 100
        assert len(_slugify(long_title)) <= 60

    def test_unicode(self):
        result = _slugify("Café AI")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_string(self):
        result = _slugify("")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Article response parsing
# ---------------------------------------------------------------------------

class TestParseArticleResponse:
    def test_valid_json(self):
        data = {
            "title": "Great AI Article",
            "meta_desc": "Short description here.",
            "body": "## Section\n\nContent here.",
        }
        result = _parse_article_response(json.dumps(data))
        assert result["title"] == "Great AI Article"
        assert result["meta_desc"] == "Short description here."

    def test_strips_code_fences(self):
        data = {"title": "T", "meta_desc": "M", "body": "B"}
        raw = f"```json\n{json.dumps(data)}\n```"
        result = _parse_article_response(raw)
        assert result["title"] == "T"

    def test_fallback_on_invalid_json(self):
        result = _parse_article_response("this is not JSON")
        assert "title" in result
        assert "meta_desc" in result
        assert "body" in result

    def test_fallback_on_missing_fields(self):
        result = _parse_article_response('{"title": "Only title"}')
        assert "meta_desc" in result


# ---------------------------------------------------------------------------
# Fallback body
# ---------------------------------------------------------------------------

class TestFallbackBody:
    def test_returns_string(self):
        cluster = _make_cluster()
        result = _fallback_body(cluster)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_hook_phrases(self):
        import config
        cluster = _make_cluster()
        result = _fallback_body(cluster)
        assert config.IDENTITY_HOOK in result
        assert config.PREDICTION_HOOK in result

    def test_includes_item_links(self):
        cluster = _make_cluster()
        result = _fallback_body(cluster)
        assert "https://example.com/1" in result


# ---------------------------------------------------------------------------
# Generate article (without Gemini)
# ---------------------------------------------------------------------------

class TestGenerateArticle:
    def test_generates_article_dict(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.POSTS_DIR", str(tmp_path / "posts"))
        cluster = _make_cluster()
        article = generate_article(cluster, gemini_enabled=False)
        assert isinstance(article, dict)
        assert "title" in article
        assert "slug" in article
        assert "body" in article

    def test_saves_markdown_file(self, tmp_path, monkeypatch):
        posts_dir = str(tmp_path / "posts")
        monkeypatch.setattr("config.POSTS_DIR", posts_dir)
        monkeypatch.setattr("pipeline.generate.config.POSTS_DIR", posts_dir)
        cluster = _make_cluster()
        article = generate_article(cluster, gemini_enabled=False)
        saved_files = os.listdir(posts_dir)
        assert len(saved_files) == 1
        assert saved_files[0].endswith(".md")

    def test_article_has_published_timestamp(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.POSTS_DIR", str(tmp_path / "posts"))
        monkeypatch.setattr("pipeline.generate.config.POSTS_DIR", str(tmp_path / "posts"))
        cluster = _make_cluster()
        article = generate_article(cluster, gemini_enabled=False)
        assert "published" in article
        assert "T" in article["published"]  # ISO-8601 format


# ---------------------------------------------------------------------------
# Atomize content (without Gemini)
# ---------------------------------------------------------------------------

class TestAtomizeContent:
    def test_returns_atoms_dict(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.TRENDS_DIR", str(tmp_path / "trends"))
        monkeypatch.setattr("pipeline.atomize.config.TRENDS_DIR", str(tmp_path / "trends"))
        article = _make_article()
        atoms = atomize_content(article, gemini_enabled=False)
        assert isinstance(atoms, dict)

    def test_fallback_has_required_keys(self):
        article = _make_article()
        atoms = _fallback_atoms(article["title"])
        required = {"twitter_thread", "linkedin_post", "hot_take", "quote_cards", "email_teaser"}
        assert required.issubset(atoms.keys())

    def test_twitter_thread_is_list(self):
        article = _make_article()
        atoms = _fallback_atoms(article["title"])
        assert isinstance(atoms["twitter_thread"], list)
        assert len(atoms["twitter_thread"]) >= 1

    def test_email_teaser_has_subject_and_preview(self):
        article = _make_article()
        atoms = _fallback_atoms(article["title"])
        assert "subject" in atoms["email_teaser"]
        assert "preview" in atoms["email_teaser"]

    def test_saves_atoms_json_file(self, tmp_path, monkeypatch):
        trends_dir = str(tmp_path / "trends")
        monkeypatch.setattr("config.TRENDS_DIR", trends_dir)
        monkeypatch.setattr("pipeline.atomize.config.TRENDS_DIR", trends_dir)
        article = _make_article(slug="test-slug")
        atomize_content(article, gemini_enabled=False)
        expected_file = os.path.join(trends_dir, "test-slug_atoms.json")
        assert os.path.exists(expected_file)

    def test_parse_atoms_valid_json(self):
        data = {
            "twitter_thread": ["tweet1", "tweet2"],
            "linkedin_post": "LinkedIn post content",
            "hot_take": "Spicy take here",
            "quote_cards": ["Quote 1", "Quote 2"],
            "email_teaser": {"subject": "Subject", "preview": "Preview text"},
        }
        result = _parse_atoms(json.dumps(data))
        assert result["hot_take"] == "Spicy take here"

    def test_parse_atoms_fallback_on_invalid(self):
        result = _parse_atoms("not json at all")
        required = {"twitter_thread", "linkedin_post", "hot_take", "quote_cards", "email_teaser"}
        assert required.issubset(result.keys())
