"""
tests/test_feedback.py — Unit tests for pipeline/feedback.py (Stages G4 & G5)
"""
import json
import os
import pytest

from pipeline.feedback import (
    load_feedback,
    save_feedback,
    record_published,
    reinforce_topics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_article(slug="test-article", topic="AI Agents"):
    return {
        "title": "AI Agents Are Reshaping Development",
        "slug": slug,
        "meta_desc": "A look at the agent revolution.",
        "body": "## Content",
        "cluster": {
            "topic": topic,
            "narrative": {
                "trend": "Agents are mainstream",
                "contradiction": "More autonomy = less control",
            },
        },
        "published": "2024-01-15T12:00:00+00:00",
    }


def _make_atoms():
    return {
        "twitter_thread": ["tweet1", "tweet2"],
        "linkedin_post": "LinkedIn content",
        "hot_take": "Hot take here",
        "quote_cards": ["Quote 1", "Quote 2"],
        "email_teaser": {"subject": "Subject", "preview": "Preview"},
    }


# ---------------------------------------------------------------------------
# load_feedback / save_feedback
# ---------------------------------------------------------------------------

class TestLoadSaveFeedback:
    def test_load_returns_empty_list_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.DATA_DIR", str(tmp_path))
        monkeypatch.setattr("config.FEEDBACK_FILE", str(tmp_path / "feedback.json"))
        monkeypatch.setattr("pipeline.feedback.config.DATA_DIR", str(tmp_path))
        monkeypatch.setattr("pipeline.feedback.config.FEEDBACK_FILE", str(tmp_path / "feedback.json"))
        result = load_feedback()
        assert result == []

    def test_round_trip(self, tmp_path, monkeypatch):
        fb_file = str(tmp_path / "feedback.json")
        monkeypatch.setattr("config.DATA_DIR", str(tmp_path))
        monkeypatch.setattr("config.FEEDBACK_FILE", fb_file)
        monkeypatch.setattr("pipeline.feedback.config.DATA_DIR", str(tmp_path))
        monkeypatch.setattr("pipeline.feedback.config.FEEDBACK_FILE", fb_file)
        records = [{"topic": "AI", "published": True}]
        save_feedback(records)
        loaded = load_feedback()
        assert loaded == records

    def test_load_handles_corrupt_file(self, tmp_path, monkeypatch):
        fb_file = str(tmp_path / "feedback.json")
        monkeypatch.setattr("config.DATA_DIR", str(tmp_path))
        monkeypatch.setattr("config.FEEDBACK_FILE", fb_file)
        monkeypatch.setattr("pipeline.feedback.config.DATA_DIR", str(tmp_path))
        monkeypatch.setattr("pipeline.feedback.config.FEEDBACK_FILE", fb_file)
        with open(fb_file, "w") as f:
            f.write("not json")
        result = load_feedback()
        assert result == []

    def test_creates_data_dir_if_missing(self, tmp_path, monkeypatch):
        data_dir = str(tmp_path / "new_data")
        fb_file = str(tmp_path / "new_data" / "feedback.json")
        monkeypatch.setattr("config.DATA_DIR", data_dir)
        monkeypatch.setattr("config.FEEDBACK_FILE", fb_file)
        monkeypatch.setattr("pipeline.feedback.config.DATA_DIR", data_dir)
        monkeypatch.setattr("pipeline.feedback.config.FEEDBACK_FILE", fb_file)
        save_feedback([{"topic": "test"}])
        assert os.path.exists(data_dir)


# ---------------------------------------------------------------------------
# record_published
# ---------------------------------------------------------------------------

class TestRecordPublished:
    def test_appends_record(self, tmp_path, monkeypatch):
        fb_file = str(tmp_path / "feedback.json")
        monkeypatch.setattr("config.DATA_DIR", str(tmp_path))
        monkeypatch.setattr("config.FEEDBACK_FILE", fb_file)
        monkeypatch.setattr("pipeline.feedback.config.DATA_DIR", str(tmp_path))
        monkeypatch.setattr("pipeline.feedback.config.FEEDBACK_FILE", fb_file)
        article = _make_article()
        atoms = _make_atoms()
        records = record_published(article, atoms)
        assert len(records) == 1
        assert records[0]["topic"] == "AI Agents"
        assert records[0]["published"] is True

    def test_increments_appearances(self, tmp_path, monkeypatch):
        fb_file = str(tmp_path / "feedback.json")
        monkeypatch.setattr("config.DATA_DIR", str(tmp_path))
        monkeypatch.setattr("config.FEEDBACK_FILE", fb_file)
        monkeypatch.setattr("pipeline.feedback.config.DATA_DIR", str(tmp_path))
        monkeypatch.setattr("pipeline.feedback.config.FEEDBACK_FILE", fb_file)
        article = _make_article(topic="AI Video")
        atoms = _make_atoms()
        record_published(article, atoms)
        records = record_published(article, atoms)
        appearances = [r["appearances"] for r in records if r["topic"] == "AI Video"]
        assert max(appearances) == 2

    def test_stores_engagement_proxy(self, tmp_path, monkeypatch):
        fb_file = str(tmp_path / "feedback.json")
        monkeypatch.setattr("config.DATA_DIR", str(tmp_path))
        monkeypatch.setattr("config.FEEDBACK_FILE", fb_file)
        monkeypatch.setattr("pipeline.feedback.config.DATA_DIR", str(tmp_path))
        monkeypatch.setattr("pipeline.feedback.config.FEEDBACK_FILE", fb_file)
        article = _make_article()
        records = record_published(article, _make_atoms(), engagement_proxy="high")
        assert records[0]["engagement_proxy"] == "high"

    def test_record_has_required_fields(self, tmp_path, monkeypatch):
        fb_file = str(tmp_path / "feedback.json")
        monkeypatch.setattr("config.DATA_DIR", str(tmp_path))
        monkeypatch.setattr("config.FEEDBACK_FILE", fb_file)
        monkeypatch.setattr("pipeline.feedback.config.DATA_DIR", str(tmp_path))
        monkeypatch.setattr("pipeline.feedback.config.FEEDBACK_FILE", fb_file)
        article = _make_article()
        records = record_published(article, _make_atoms())
        required = {"topic", "published", "published_at", "engagement_proxy",
                    "format", "angle", "slug", "appearances"}
        assert required.issubset(records[0].keys())


# ---------------------------------------------------------------------------
# reinforce_topics
# ---------------------------------------------------------------------------

class TestReinforceTopics:
    def test_returns_topics_above_threshold(self):
        records = [
            {"topic": "AI Agents", "appearances": 3},
            {"topic": "AI Agents", "appearances": 3},
            {"topic": "AI Agents", "appearances": 3},
            {"topic": "AI Video", "appearances": 1},
        ]
        result = reinforce_topics(records, min_appearances=2)
        topics = [r["topic"] for r in result]
        assert "AI Agents" in topics
        assert "AI Video" not in topics

    def test_includes_follow_up_angles(self):
        records = [
            {"topic": "AI Agents"},
            {"topic": "AI Agents"},
        ]
        result = reinforce_topics(records, min_appearances=2)
        assert len(result) == 1
        assert len(result[0]["follow_up_angles"]) >= 2

    def test_sorted_by_appearances_desc(self):
        records = [
            {"topic": "A"}, {"topic": "A"}, {"topic": "A"},
            {"topic": "B"}, {"topic": "B"},
            {"topic": "C"}, {"topic": "C"}, {"topic": "C"}, {"topic": "C"},
        ]
        result = reinforce_topics(records, min_appearances=2)
        appearances = [r["appearances"] for r in result]
        assert appearances == sorted(appearances, reverse=True)

    def test_empty_records(self):
        assert reinforce_topics([]) == []

    def test_no_topics_above_threshold(self):
        records = [{"topic": "AI"}]
        result = reinforce_topics(records, min_appearances=5)
        assert result == []
