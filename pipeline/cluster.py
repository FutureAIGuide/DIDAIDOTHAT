"""
pipeline/cluster.py — Stages T2 & T3: Clustering + Momentum Scoring

Groups ingested signals into thematic clusters and scores each cluster
for momentum using velocity, source diversity, novelty, and engagement.

Cluster schema:
{
    "topic":            str,
    "items":            list[dict],   # signals belonging to this cluster
    "velocity":         float,        # items-per-day rate
    "source_diversity": int,          # number of distinct source_types
    "first_seen":       str,          # ISO-8601 of oldest item
    "momentum_score":   float,        # weighted composite
    "keywords":         list[str],    # top shared keywords
    "entities":         list[str],    # shared entities
}
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text representation helpers
# ---------------------------------------------------------------------------

def _item_vector(item: dict) -> dict[str, float]:
    """Return a simple TF-inspired keyword frequency dict for an item."""
    counter: Counter[str] = Counter(item.get("keywords", []))
    total = sum(counter.values()) or 1
    return {k: v / total for k, v in counter.items()}


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    shared = set(vec_a) & set(vec_b)
    if not shared:
        return 0.0
    dot = sum(vec_a[k] * vec_b[k] for k in shared)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _keyword_overlap(item_a: dict, item_b: dict) -> int:
    ka = set(item_a.get("keywords", []))
    kb = set(item_b.get("keywords", []))
    return len(ka & kb)


def _entity_overlap(item_a: dict, item_b: dict) -> bool:
    ea = set(item_a.get("entities", []))
    eb = set(item_b.get("entities", []))
    return bool(ea & eb)


def _items_are_similar(item_a: dict, item_b: dict) -> bool:
    """Return True when two items belong to the same cluster."""
    cosine = _cosine_similarity(_item_vector(item_a), _item_vector(item_b))
    if cosine >= config.COSINE_SIMILARITY_THRESHOLD:
        return True
    if _keyword_overlap(item_a, item_b) >= config.KEYWORD_OVERLAP_THRESHOLD:
        return True
    if _entity_overlap(item_a, item_b):
        return True
    return False


# ---------------------------------------------------------------------------
# Clustering (greedy single-linkage)
# ---------------------------------------------------------------------------

def cluster_items(items: list[dict]) -> list[list[dict]]:
    """
    Group items into clusters using greedy single-linkage:
    each new item joins the first cluster it is similar to,
    otherwise starts a new cluster.
    Returns only clusters with >= CLUSTER_MIN_ITEMS items.
    """
    clusters: list[list[dict]] = []

    for item in items:
        placed = False
        for cluster in clusters:
            # Check similarity against the cluster representative (first item)
            if _items_are_similar(item, cluster[0]):
                cluster.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])

    # Filter small clusters
    valid = [c for c in clusters if len(c) >= config.CLUSTER_MIN_ITEMS]
    logger.info("Clustering: %d items → %d valid clusters", len(items), len(valid))
    return valid


# ---------------------------------------------------------------------------
# Momentum Scoring (Stage T3)
# ---------------------------------------------------------------------------

def _parse_dt(iso: str) -> datetime:
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.now(timezone.utc)


def _compute_velocity(items: list[dict]) -> float:
    """Items-per-day, relative to time span from first to last item."""
    if len(items) <= 1:
        return float(len(items))
    dates = sorted(_parse_dt(i["published"]) for i in items)
    span_days = max((dates[-1] - dates[0]).total_seconds() / 86400, 1.0)
    return len(items) / span_days


def _compute_source_diversity(items: list[dict]) -> int:
    return len({i.get("source_type", "") for i in items})


def _compute_novelty(items: list[dict]) -> float:
    """
    Returns 1.0 if the oldest item is within NOVELTY_WINDOW_DAYS,
    sliding down to 0.0 beyond that window.
    """
    now = datetime.now(timezone.utc)
    oldest = min(_parse_dt(i["published"]) for i in items)
    age_days = (now - oldest).total_seconds() / 86400
    return max(0.0, 1.0 - age_days / config.NOVELTY_WINDOW_DAYS)


def _compute_engagement(items: list[dict]) -> float:
    """Mean engagement proxy (already normalised 0-1 in ingest)."""
    values = [i.get("engagement", 0.0) for i in items]
    return sum(values) / len(values) if values else 0.0


def _top_shared_keywords(items: list[dict], n: int = 5) -> list[str]:
    counter: Counter[str] = Counter()
    for item in items:
        counter.update(item.get("keywords", []))
    return [k for k, _ in counter.most_common(n)]


def _shared_entities(items: list[dict]) -> list[str]:
    if not items:
        return []
    sets = [set(i.get("entities", [])) for i in items]
    common = sets[0].copy()
    for s in sets[1:]:
        common &= s
    if not common:
        # fall back to union if nothing shared
        all_entities: Counter[str] = Counter()
        for item in items:
            all_entities.update(item.get("entities", []))
        return [e for e, _ in all_entities.most_common(3)]
    return sorted(common)


def _derive_topic(items: list[dict]) -> str:
    """Derive a human-readable cluster topic label from shared keywords/entities."""
    entities = _shared_entities(items)
    keywords = _top_shared_keywords(items, 3)
    parts = entities[:2] + [k for k in keywords if k not in entities]
    return " + ".join(parts[:3]) if parts else "AI trend"


def score_clusters(raw_clusters: list[list[dict]]) -> list[dict[str, Any]]:
    """
    Convert raw clusters (lists of items) into scored cluster dicts.
    """
    w = config.MOMENTUM_WEIGHTS
    scored: list[dict[str, Any]] = []

    for group in raw_clusters:
        velocity = _compute_velocity(group)
        src_div = _compute_source_diversity(group)
        novelty = _compute_novelty(group)
        engagement = _compute_engagement(group)

        # Normalise source diversity to 0-1
        src_div_norm = min(1.0, src_div / config.MAX_SOURCE_DIVERSITY)

        # Normalise velocity to 0-1 (cap at 10 items/day)
        velocity_norm = min(1.0, velocity / 10.0)

        momentum = (
            w["velocity"] * velocity_norm
            + w["source_diversity"] * src_div_norm
            + w["novelty"] * novelty
            + w["engagement"] * engagement
        )

        oldest = min(i["published"] for i in group)

        scored.append(
            {
                "topic": _derive_topic(group),
                "items": group,
                "velocity": round(velocity, 4),
                "source_diversity": src_div,
                "first_seen": oldest,
                "momentum_score": round(momentum, 4),
                "keywords": _top_shared_keywords(group),
                "entities": _shared_entities(group),
            }
        )

    # Sort by momentum descending
    scored.sort(key=lambda c: c["momentum_score"], reverse=True)
    return scored
