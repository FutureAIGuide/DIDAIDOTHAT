"""
pipeline/__init__.py — exposes the main pipeline stages.
"""
from .ingest import ingest_all
from .cluster import cluster_items, score_clusters
from .trends import classify_trends, extract_narratives
from .generate import generate_article
from .atomize import atomize_content
from .feedback import load_feedback, save_feedback, record_published, reinforce_topics

__all__ = [
    "ingest_all",
    "cluster_items",
    "score_clusters",
    "classify_trends",
    "extract_narratives",
    "generate_article",
    "atomize_content",
    "load_feedback",
    "save_feedback",
    "record_published",
    "reinforce_topics",
]
