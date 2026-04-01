"""
pipeline/feedback.py — Loop Stages G4 & G5: Feedback Capture + Reinforcement

G4: Persists a structured record every time content is published.
G5: Detects topics with repeated coverage and escalates them.

feedback.json entry schema:
{
    "topic":            str,
    "published":        bool,
    "published_at":     str,   # ISO-8601
    "engagement_proxy": str,   # "high" | "medium" | "low" | "unknown"
    "format":           str,   # "article" | "thread" | ...
    "angle":            str,
    "slug":             str,
    "appearances":      int,   # how many times this topic has appeared
}
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_feedback() -> list[dict[str, Any]]:
    """Load existing feedback records from disk."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    if not os.path.exists(config.FEEDBACK_FILE):
        return []
    try:
        with open(config.FEEDBACK_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load feedback file: %s", exc)
        return []


def save_feedback(records: list[dict[str, Any]]) -> None:
    """Persist feedback records to disk."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(config.FEEDBACK_FILE, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    logger.info("Feedback saved (%d records)", len(records))


# ---------------------------------------------------------------------------
# G4 — Record Published Content
# ---------------------------------------------------------------------------

def record_published(
    article: dict[str, Any],
    atoms: dict[str, Any],
    engagement_proxy: str = "unknown",
) -> list[dict[str, Any]]:
    """
    Append a feedback record for a newly published article.
    Returns the updated full feedback list.
    """
    cluster = article.get("cluster", {})
    topic = cluster.get("topic", article.get("title", "unknown"))
    narrative = cluster.get("narrative", {})
    angle = narrative.get("contradiction", narrative.get("trend", ""))

    records = load_feedback()

    # Count prior appearances of this topic
    prior = sum(1 for r in records if r.get("topic") == topic)

    record: dict[str, Any] = {
        "topic": topic,
        "published": True,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "engagement_proxy": engagement_proxy,
        "format": "article",
        "angle": angle[:200],
        "slug": article.get("slug", ""),
        "appearances": prior + 1,
    }

    records.append(record)
    save_feedback(records)
    logger.info("Feedback recorded for topic: %s (appearance #%d)", topic, record["appearances"])
    return records


# ---------------------------------------------------------------------------
# G5 — Reinforcement: Detect Topics With Traction
# ---------------------------------------------------------------------------

def reinforce_topics(
    records: list[dict[str, Any]],
    min_appearances: int = 2,
) -> list[dict[str, Any]]:
    """
    Return topics that have appeared >= min_appearances times,
    annotated with suggested follow-up angles.
    """
    # Aggregate by topic
    topic_counts: dict[str, int] = {}
    for r in records:
        topic = r.get("topic", "")
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    reinforced: list[dict[str, Any]] = []
    for topic, count in topic_counts.items():
        if count >= min_appearances:
            reinforced.append(
                {
                    "topic": topic,
                    "appearances": count,
                    "follow_up_angles": [
                        f"{topic} — Part {count + 1}: What's changed",
                        f"What we got wrong about {topic}",
                        f"Deep dive: {topic} one month later",
                    ],
                }
            )

    reinforced.sort(key=lambda x: x["appearances"], reverse=True)
    logger.info("Reinforcement: %d topics with traction", len(reinforced))
    return reinforced
