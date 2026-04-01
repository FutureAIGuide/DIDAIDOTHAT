"""
pipeline/trends.py — Stages T4 & T5: Trend Classification + Narrative Extraction

T4: Labels each scored cluster as "emerging", "peaking", or "declining".
T5: Uses Gemini to generate rich narrative metadata for each cluster.

Narrative schema:
{
    "trend":         str,
    "core_question": str,
    "why_now":       str,
    "contradiction": str,
    "prediction":    str,
}
"""
from __future__ import annotations

import json
import logging
from typing import Any

import config
from .gemini import call_gemini

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# T4 — Trend Classification
# ---------------------------------------------------------------------------

def _classify_cluster(cluster: dict[str, Any]) -> str:
    """
    Label a cluster based on momentum_score and velocity.

    Rules:
    - emerging  → momentum_score >= MOMENTUM_THRESHOLD AND velocity is
                  rising (proxy: velocity >= EMERGING_VELOCITY_THRESHOLD
                  but momentum < 0.85 so it hasn't yet peaked)
    - peaking   → very high momentum (>= 0.85) → saturated
    - declining → below MOMENTUM_THRESHOLD
    """
    score = cluster["momentum_score"]
    velocity = cluster["velocity"]

    if score < config.MOMENTUM_THRESHOLD:
        return "declining"
    if score >= 0.85:
        return "peaking"
    if velocity >= config.EMERGING_VELOCITY_THRESHOLD:
        return "emerging"
    return "peaking"


def classify_trends(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach a `status` field to each cluster and return all clusters."""
    for cluster in clusters:
        cluster["status"] = _classify_cluster(cluster)
    emerging = sum(1 for c in clusters if c["status"] == "emerging")
    logger.info(
        "Trend classification: %d emerging / %d peaking / %d declining",
        emerging,
        sum(1 for c in clusters if c["status"] == "peaking"),
        sum(1 for c in clusters if c["status"] == "declining"),
    )
    return clusters


# ---------------------------------------------------------------------------
# T5 — Narrative Extraction
# ---------------------------------------------------------------------------

_NARRATIVE_PROMPT = """\
You are a media strategist specialising in AI trends.

Given this emerging AI trend cluster, extract a compelling narrative.

Cluster topic: {topic}
Keywords: {keywords}
Entities: {entities}
Sample headlines:
{headlines}

Return ONLY valid JSON (no markdown fences) with exactly these fields:
{{
  "trend": "one-sentence trend description",
  "core_question": "the provocative question this trend raises",
  "why_now": "why this is happening right now",
  "contradiction": "the surprising or counter-intuitive angle",
  "prediction": "bold 6-month prediction"
}}
"""


def _build_narrative_prompt(cluster: dict[str, Any]) -> str:
    headlines = "\n".join(
        f"- {item['title']}" for item in cluster["items"][:5]
    )
    return _NARRATIVE_PROMPT.format(
        topic=cluster["topic"],
        keywords=", ".join(cluster.get("keywords", [])[:8]),
        entities=", ".join(cluster.get("entities", [])[:5]),
        headlines=headlines,
    )


def _parse_narrative(raw: str, topic: str = "") -> dict[str, str]:
    """Extract JSON object from Gemini response."""
    raw = raw.strip()
    # Strip optional markdown code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        data = json.loads(raw)
        required = {"trend", "core_question", "why_now", "contradiction", "prediction"}
        if required.issubset(data.keys()):
            return {k: str(data[k]) for k in required}
    except json.JSONDecodeError:
        pass
    # Fallback: build a minimal narrative from cluster metadata
    return {
        "trend": cluster_fallback_trend(topic),
        "core_question": "What does this mean for AI development?",
        "why_now": "Multiple independent signals are converging simultaneously.",
        "contradiction": "The most obvious explanation is probably wrong.",
        "prediction": "This pattern will accelerate significantly in the next 6 months.",
    }


def cluster_fallback_trend(topic: str) -> str:
    return f"A new pattern is emerging around {topic or 'AI'}"


def extract_narratives(
    clusters: list[dict[str, Any]], gemini_enabled: bool = True
) -> list[dict[str, Any]]:
    """
    For each cluster add a `narrative` key with T5 metadata.
    Uses Gemini when available; falls back to heuristic values otherwise.
    """
    for cluster in clusters:
        if gemini_enabled and config.GEMINI_API_KEY:
            prompt = _build_narrative_prompt(cluster)
            raw = call_gemini(prompt)
            cluster["narrative"] = _parse_narrative(raw, topic=cluster.get("topic", ""))
        else:
            # Heuristic fallback
            cluster["narrative"] = {
                "trend": cluster_fallback_trend(cluster.get("topic", "")),
                "core_question": "What does this shift signal for the broader AI landscape?",
                "why_now": "Convergent signals from multiple platforms suggest a tipping point.",
                "contradiction": "The implications may be the opposite of what most assume.",
                "prediction": "This will show up everywhere in 6 months.",
            }
        logger.debug("Narrative extracted for cluster: %s", cluster["topic"])
    return clusters


def select_emerging_trends(
    clusters: list[dict[str, Any]], max_trends: int | None = None
) -> list[dict[str, Any]]:
    """Return only 'emerging' clusters, sorted by momentum, up to max_trends."""
    if max_trends is None:
        max_trends = config.MAX_DAILY_TRENDS
    emerging = [c for c in clusters if c.get("status") == "emerging"]
    emerging.sort(key=lambda c: c["momentum_score"], reverse=True)
    return emerging[:max_trends]
