"""
tests/test_trends.py — Unit tests for pipeline/trends.py (Stages T4 & T5)
"""
import pytest
from datetime import datetime, timezone

from pipeline.trends import (
    _classify_cluster,
    classify_trends,
    extract_narratives,
    select_emerging_trends,
    cluster_fallback_trend,
    _parse_narrative,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cluster(momentum_score=0.6, velocity=3.0, status=None):
    now = datetime.now(timezone.utc).isoformat()
    c = {
        "topic": "AI Agents",
        "items": [
            {"title": "Agent news", "url": "https://example.com/1", "published": now},
            {"title": "More agent stuff", "url": "https://example.com/2", "published": now},
        ],
        "momentum_score": momentum_score,
        "velocity": velocity,
        "source_diversity": 2,
        "first_seen": now,
        "keywords": ["agent", "llm"],
        "entities": ["openai"],
    }
    if status:
        c["status"] = status
    return c


# ---------------------------------------------------------------------------
# T4 — Classify Cluster
# ---------------------------------------------------------------------------

class TestClassifyCluster:
    def test_emerging_cluster(self):
        cluster = _make_cluster(momentum_score=0.65, velocity=3.0)
        assert _classify_cluster(cluster) == "emerging"

    def test_peaking_cluster_high_momentum(self):
        cluster = _make_cluster(momentum_score=0.9, velocity=5.0)
        assert _classify_cluster(cluster) == "peaking"

    def test_declining_cluster_low_momentum(self):
        cluster = _make_cluster(momentum_score=0.2, velocity=0.5)
        assert _classify_cluster(cluster) == "declining"

    def test_peaking_cluster_medium_momentum_low_velocity(self):
        cluster = _make_cluster(momentum_score=0.7, velocity=0.5)
        # Above momentum threshold but velocity below emerging threshold
        assert _classify_cluster(cluster) == "peaking"

    def test_momentum_threshold_boundary(self):
        import config
        cluster = _make_cluster(
            momentum_score=config.MOMENTUM_THRESHOLD - 0.01, velocity=5.0
        )
        assert _classify_cluster(cluster) == "declining"


class TestClassifyTrends:
    def test_adds_status_field(self):
        clusters = [_make_cluster(0.65, 3.0), _make_cluster(0.1, 0.5)]
        result = classify_trends(clusters)
        assert all("status" in c for c in result)

    def test_correct_statuses(self):
        clusters = [
            _make_cluster(momentum_score=0.65, velocity=3.0),
            _make_cluster(momentum_score=0.1, velocity=0.1),
        ]
        result = classify_trends(clusters)
        statuses = {c["status"] for c in result}
        assert "emerging" in statuses
        assert "declining" in statuses

    def test_returns_same_list(self):
        clusters = [_make_cluster()]
        result = classify_trends(clusters)
        assert result is clusters


# ---------------------------------------------------------------------------
# T5 — Narrative Extraction
# ---------------------------------------------------------------------------

class TestParseNarrative:
    def test_valid_json(self):
        raw = '{"trend": "t", "core_question": "q", "why_now": "w", "contradiction": "c", "prediction": "p"}'
        result = _parse_narrative(raw)
        assert result["trend"] == "t"
        assert result["core_question"] == "q"

    def test_strips_markdown_fences(self):
        raw = '```json\n{"trend": "t", "core_question": "q", "why_now": "w", "contradiction": "c", "prediction": "p"}\n```'
        result = _parse_narrative(raw)
        assert result["trend"] == "t"

    def test_fallback_on_invalid_json(self):
        result = _parse_narrative("not valid json")
        required = {"trend", "core_question", "why_now", "contradiction", "prediction"}
        assert required.issubset(result.keys())

    def test_fallback_on_missing_fields(self):
        raw = '{"trend": "only trend"}'
        result = _parse_narrative(raw)
        required = {"trend", "core_question", "why_now", "contradiction", "prediction"}
        assert required.issubset(result.keys())


class TestExtractNarratives:
    def test_fallback_mode_adds_narrative(self):
        clusters = [_make_cluster()]
        result = extract_narratives(clusters, gemini_enabled=False)
        assert "narrative" in result[0]

    def test_fallback_narrative_has_required_keys(self):
        clusters = [_make_cluster()]
        result = extract_narratives(clusters, gemini_enabled=False)
        required = {"trend", "core_question", "why_now", "contradiction", "prediction"}
        assert required.issubset(result[0]["narrative"].keys())

    def test_gemini_enabled_without_key_falls_back(self, mocker):
        mocker.patch("config.GEMINI_API_KEY", "")
        clusters = [_make_cluster()]
        result = extract_narratives(clusters, gemini_enabled=True)
        assert "narrative" in result[0]

    def test_cluster_fallback_trend(self):
        topic = "AI video generation"
        result = cluster_fallback_trend(topic)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_topic_fallback(self):
        result = cluster_fallback_trend("")
        assert "AI" in result


class TestSelectEmergingTrends:
    def test_filters_to_emerging_only(self):
        clusters = [
            _make_cluster(status="emerging"),
            _make_cluster(status="peaking"),
            _make_cluster(status="declining"),
        ]
        result = select_emerging_trends(clusters)
        assert all(c["status"] == "emerging" for c in result)

    def test_respects_max_trends(self):
        clusters = [_make_cluster(status="emerging") for _ in range(10)]
        result = select_emerging_trends(clusters, max_trends=2)
        assert len(result) <= 2

    def test_sorted_by_momentum(self):
        clusters = [
            _make_cluster(momentum_score=0.6, status="emerging"),
            _make_cluster(momentum_score=0.9, status="emerging"),
        ]
        result = select_emerging_trends(clusters, max_trends=10)
        scores = [c["momentum_score"] for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_clusters(self):
        assert select_emerging_trends([]) == []

    def test_no_emerging_trends(self):
        clusters = [_make_cluster(status="peaking"), _make_cluster(status="declining")]
        assert select_emerging_trends(clusters) == []
