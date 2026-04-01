"""
main.py — DID AI DO THAT?! Pipeline Orchestrator

Full pipeline:
  Fetch → Normalize → Score → Cluster → Detect Trends
  → Select Emerging Trends → Gemini Filtering
  → Generate Articles → Atomize Content
  → Publish → Capture Feedback → Reinforce Trends

Usage:
    python main.py [--dry-run] [--no-gemini]

Options:
    --dry-run     Run the pipeline without saving files or calling Gemini.
    --no-gemini   Use heuristic fallbacks instead of Gemini API calls.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def run_pipeline(dry_run: bool = False, gemini_enabled: bool = True) -> dict[str, Any]:
    """Execute the full content engine pipeline. Returns a run summary."""
    from pipeline.ingest import ingest_all
    from pipeline.cluster import cluster_items, score_clusters
    from pipeline.trends import classify_trends, extract_narratives, select_emerging_trends
    from pipeline.generate import generate_article
    from pipeline.atomize import atomize_content
    from pipeline.feedback import load_feedback, record_published, reinforce_topics

    summary: dict[str, Any] = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "gemini_enabled": gemini_enabled,
        "signals_collected": 0,
        "clusters_found": 0,
        "emerging_trends": 0,
        "articles_generated": [],
        "reinforcement_topics": [],
    }

    # ------------------------------------------------------------------
    # Stage 1 — Signal Aggregation (T1)
    # ------------------------------------------------------------------
    logger.info("=== Stage T1: Signal Aggregation ===")
    signals = ingest_all()
    summary["signals_collected"] = len(signals)

    if not signals:
        logger.warning("No signals collected — aborting pipeline.")
        return summary

    # ------------------------------------------------------------------
    # Stage 2 — Clustering (T2) + Momentum Scoring (T3)
    # ------------------------------------------------------------------
    logger.info("=== Stages T2-T3: Clustering + Momentum Scoring ===")
    raw_clusters = cluster_items(signals)
    scored_clusters = score_clusters(raw_clusters)
    summary["clusters_found"] = len(scored_clusters)

    if not scored_clusters:
        logger.warning("No clusters formed — try lowering CLUSTER_MIN_ITEMS.")
        return summary

    # Persist clusters
    if not dry_run:
        _save_json(config.CLUSTERS_FILE, scored_clusters_for_json(scored_clusters))

    # ------------------------------------------------------------------
    # Stage 3 — Trend Classification (T4)
    # ------------------------------------------------------------------
    logger.info("=== Stage T4: Trend Classification ===")
    classified = classify_trends(scored_clusters)

    # ------------------------------------------------------------------
    # Stage 4 — Narrative Extraction / Gemini Filtering (T5)
    # ------------------------------------------------------------------
    logger.info("=== Stage T5: Narrative Extraction ===")
    emerging = select_emerging_trends(classified)
    summary["emerging_trends"] = len(emerging)

    if not emerging:
        logger.info("No emerging trends found today — done.")
        return summary

    # Extract narratives for emerging trends only
    emerging = extract_narratives(emerging, gemini_enabled=gemini_enabled and not dry_run)

    # Persist trend history
    if not dry_run:
        _append_trend_history(emerging)

    # ------------------------------------------------------------------
    # Stage 5 — Generate Articles
    # ------------------------------------------------------------------
    logger.info("=== Content Generation ===")
    articles: list[dict[str, Any]] = []
    for cluster in emerging[: config.MAX_DAILY_ARTICLES]:
        logger.info("Generating article for: %s", cluster["topic"])
        if not dry_run:
            article = generate_article(cluster, gemini_enabled=gemini_enabled)
        else:
            article = _dry_run_article(cluster)
        articles.append(article)
        summary["articles_generated"].append(article["title"])

    # ------------------------------------------------------------------
    # Stage 6 — Content Atomization (G1-G3)
    # ------------------------------------------------------------------
    logger.info("=== Content Atomization ===")
    for article in articles:
        if not dry_run:
            atoms = atomize_content(article, gemini_enabled=gemini_enabled)
        else:
            atoms = {}
        logger.info("Atomized: %s", article["title"])

        # ------------------------------------------------------------------
        # Stage 7 — Feedback Capture (G4)
        # ------------------------------------------------------------------
        if not dry_run:
            record_published(article, atoms)

    # ------------------------------------------------------------------
    # Stage 8 — Reinforcement (G5)
    # ------------------------------------------------------------------
    logger.info("=== Reinforcement ===")
    feedback_records = load_feedback()
    reinforced = reinforce_topics(feedback_records)
    summary["reinforcement_topics"] = [r["topic"] for r in reinforced]

    if reinforced:
        logger.info(
            "Topics with traction (escalate coverage): %s",
            [r["topic"] for r in reinforced[:3]],
        )

    logger.info("=== Pipeline complete ===")
    logger.info("Summary: %s", json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scored_clusters_for_json(clusters: list[dict]) -> list[dict]:
    """Strip non-serialisable objects from clusters before JSON dump."""
    safe = []
    for c in clusters:
        sc = {k: v for k, v in c.items() if k != "items"}
        sc["item_count"] = len(c.get("items", []))
        sc["item_ids"] = [i.get("id") for i in c.get("items", [])]
        safe.append(sc)
    return safe


def _save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    logger.info("Saved: %s", path)


def _append_trend_history(emerging: list[dict]) -> None:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    history: list[dict] = []
    if os.path.exists(config.TREND_HISTORY_FILE):
        try:
            with open(config.TREND_HISTORY_FILE, encoding="utf-8") as fh:
                history = json.load(fh)
        except (json.JSONDecodeError, OSError):
            history = []

    for cluster in emerging:
        history.append(
            {
                "topic": cluster["topic"],
                "status": cluster.get("status"),
                "momentum_score": cluster.get("momentum_score"),
                "detected_at": datetime.now(timezone.utc).isoformat(),
                "narrative": cluster.get("narrative", {}),
            }
        )

    with open(config.TREND_HISTORY_FILE, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2, ensure_ascii=False)
    logger.info("Trend history updated (%d entries)", len(history))


def _dry_run_article(cluster: dict) -> dict:
    return {
        "title": f"[DRY RUN] {cluster.get('topic', 'AI Trend')}",
        "slug": "dry-run",
        "meta_desc": "Dry run article",
        "body": "",
        "cluster": cluster,
        "published": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DID AI DO THAT?! Content Engine")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without saving files or calling Gemini",
    )
    parser.add_argument(
        "--no-gemini",
        action="store_true",
        help="Use heuristic fallbacks instead of Gemini API calls",
    )
    args = parser.parse_args()

    gemini_enabled = not args.no_gemini
    if gemini_enabled and not config.GEMINI_API_KEY:
        logger.warning(
            "GEMINI_API_KEY not set — running with heuristic fallbacks (--no-gemini mode)."
        )
        gemini_enabled = False

    run_pipeline(dry_run=args.dry_run, gemini_enabled=gemini_enabled)


if __name__ == "__main__":
    main()
