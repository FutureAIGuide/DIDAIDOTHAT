"""
tests/test_cluster.py — Unit tests for pipeline/cluster.py (Stages T2 & T3)
"""
import pytest
from datetime import datetime, timezone, timedelta

from pipeline.cluster import (
    _cosine_similarity,
    _keyword_overlap,
    _entity_overlap,
    _items_are_similar,
    _compute_velocity,
    _compute_source_diversity,
    _compute_novelty,
    _compute_engagement,
    _derive_topic,
    cluster_items,
    score_clusters,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_item(
    keywords=None,
    entities=None,
    published=None,
    source_type="rss",
    engagement=0.0,
):
    if published is None:
        published = datetime.now(timezone.utc).isoformat()
    return {
        "id": "test",
        "title": "Test item",
        "summary": "",
        "url": "https://example.com",
        "published": published,
        "source_type": source_type,
        "source_name": "Test",
        "keywords": keywords or [],
        "entities": entities or [],
        "engagement": engagement,
    }


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = {"ai": 0.5, "model": 0.5}
        assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=0.001)

    def test_orthogonal_vectors(self):
        va = {"ai": 1.0}
        vb = {"robot": 1.0}
        assert _cosine_similarity(va, vb) == pytest.approx(0.0)

    def test_empty_vectors(self):
        assert _cosine_similarity({}, {}) == 0.0

    def test_partial_overlap(self):
        va = {"ai": 0.5, "model": 0.5}
        vb = {"ai": 0.5, "video": 0.5}
        sim = _cosine_similarity(va, vb)
        assert 0.0 < sim < 1.0


class TestKeywordOverlap:
    def test_full_overlap(self):
        a = _make_item(keywords=["ai", "model"])
        b = _make_item(keywords=["ai", "model"])
        assert _keyword_overlap(a, b) == 2

    def test_no_overlap(self):
        a = _make_item(keywords=["robot"])
        b = _make_item(keywords=["video"])
        assert _keyword_overlap(a, b) == 0

    def test_partial_overlap(self):
        a = _make_item(keywords=["ai", "robot"])
        b = _make_item(keywords=["ai", "video"])
        assert _keyword_overlap(a, b) == 1


class TestEntityOverlap:
    def test_shared_entity(self):
        a = _make_item(entities=["openai"])
        b = _make_item(entities=["openai", "google"])
        assert _entity_overlap(a, b) is True

    def test_no_shared_entity(self):
        a = _make_item(entities=["openai"])
        b = _make_item(entities=["google"])
        assert _entity_overlap(a, b) is False

    def test_empty_entities(self):
        a = _make_item(entities=[])
        b = _make_item(entities=[])
        assert _entity_overlap(a, b) is False


class TestItemsAreSimilar:
    def test_similar_via_keywords(self):
        a = _make_item(keywords=["language", "model", "training", "fine"])
        b = _make_item(keywords=["language", "model", "inference", "speed"])
        # 2 shared keywords triggers threshold
        assert _items_are_similar(a, b) is True

    def test_similar_via_entity(self):
        a = _make_item(entities=["openai"])
        b = _make_item(entities=["openai"])
        assert _items_are_similar(a, b) is True

    def test_dissimilar_items(self):
        a = _make_item(keywords=["robot", "sensor"], entities=["boston"])
        b = _make_item(keywords=["finance", "stock"], entities=["bloomberg"])
        assert _items_are_similar(a, b) is False


# ---------------------------------------------------------------------------
# Velocity
# ---------------------------------------------------------------------------

class TestComputeVelocity:
    def test_single_item(self):
        items = [_make_item()]
        assert _compute_velocity(items) == 1.0

    def test_multiple_items_same_day(self):
        now = datetime.now(timezone.utc)
        items = [_make_item(published=now.isoformat()) for _ in range(5)]
        # span < 1 day → velocity capped to len/1
        vel = _compute_velocity(items)
        assert vel == pytest.approx(5.0, abs=0.1)

    def test_items_spread_over_days(self):
        now = datetime.now(timezone.utc)
        items = [
            _make_item(published=(now - timedelta(days=4)).isoformat()),
            _make_item(published=(now - timedelta(days=2)).isoformat()),
            _make_item(published=now.isoformat()),
        ]
        # 3 items over 4 days → velocity ≈ 0.75
        vel = _compute_velocity(items)
        assert vel == pytest.approx(3 / 4, abs=0.1)


# ---------------------------------------------------------------------------
# Source diversity
# ---------------------------------------------------------------------------

class TestComputeSourceDiversity:
    def test_single_source(self):
        items = [_make_item(source_type="rss")] * 3
        assert _compute_source_diversity(items) == 1

    def test_multiple_sources(self):
        items = [
            _make_item(source_type="rss"),
            _make_item(source_type="github"),
            _make_item(source_type="arxiv"),
        ]
        assert _compute_source_diversity(items) == 3


# ---------------------------------------------------------------------------
# Novelty
# ---------------------------------------------------------------------------

class TestComputeNovelty:
    def test_fresh_items_high_novelty(self):
        items = [_make_item(published=datetime.now(timezone.utc).isoformat())]
        novelty = _compute_novelty(items)
        assert novelty > 0.9

    def test_old_items_low_novelty(self):
        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        items = [_make_item(published=old)]
        novelty = _compute_novelty(items)
        assert novelty == 0.0


# ---------------------------------------------------------------------------
# Engagement
# ---------------------------------------------------------------------------

class TestComputeEngagement:
    def test_mean_engagement(self):
        items = [
            _make_item(engagement=0.5),
            _make_item(engagement=1.0),
        ]
        assert _compute_engagement(items) == pytest.approx(0.75)

    def test_zero_engagement(self):
        items = [_make_item(engagement=0.0)] * 3
        assert _compute_engagement(items) == 0.0


# ---------------------------------------------------------------------------
# Cluster items
# ---------------------------------------------------------------------------

class TestClusterItems:
    def test_similar_items_grouped(self):
        items = [
            _make_item(keywords=["language", "model", "openai"], entities=["openai"]),
            _make_item(keywords=["language", "model", "gpt"], entities=["openai"]),
            _make_item(keywords=["language", "model", "finetune"], entities=["openai"]),
        ]
        clusters = cluster_items(items)
        assert len(clusters) >= 1
        # All 3 should be in one cluster
        assert any(len(c) == 3 for c in clusters)

    def test_dissimilar_items_form_different_clusters(self):
        similar_pair = [
            _make_item(keywords=["robot", "motor", "sensor"], entities=["boston"]),
            _make_item(keywords=["robot", "motor", "gripper"], entities=["boston"]),
        ]
        different = _make_item(keywords=["stock", "market", "finance"], entities=[])
        clusters = cluster_items(similar_pair + [different])
        # At least the pair should cluster; different item may be alone (filtered)
        assert any(len(c) == 2 for c in clusters)

    def test_min_items_filter(self):
        # Single-item cluster should be filtered out
        items = [_make_item(keywords=["unique", "topic", "xyz"])]
        clusters = cluster_items(items)
        assert clusters == []

    def test_empty_input(self):
        assert cluster_items([]) == []


# ---------------------------------------------------------------------------
# Score clusters
# ---------------------------------------------------------------------------

class TestScoreClusters:
    def _two_item_cluster(self):
        now = datetime.now(timezone.utc).isoformat()
        return [
            _make_item(keywords=["agent", "model"], entities=["openai"], published=now),
            _make_item(keywords=["agent", "llm"], entities=["openai"], published=now),
        ]

    def test_returns_sorted_by_momentum(self):
        clusters = [self._two_item_cluster(), self._two_item_cluster()]
        scored = score_clusters(clusters)
        scores = [c["momentum_score"] for c in scored]
        assert scores == sorted(scores, reverse=True)

    def test_momentum_score_in_range(self):
        scored = score_clusters([self._two_item_cluster()])
        assert 0.0 <= scored[0]["momentum_score"] <= 1.0

    def test_has_required_keys(self):
        scored = score_clusters([self._two_item_cluster()])
        required = {"topic", "items", "velocity", "source_diversity", "first_seen",
                    "momentum_score", "keywords", "entities"}
        assert required.issubset(scored[0].keys())

    def test_empty_input(self):
        assert score_clusters([]) == []

    def test_derive_topic_returns_string(self):
        items = [
            _make_item(keywords=["agent", "model"], entities=["openai"]),
            _make_item(keywords=["agent", "llm"], entities=["openai"]),
        ]
        topic = _derive_topic(items)
        assert isinstance(topic, str)
        assert len(topic) > 0
