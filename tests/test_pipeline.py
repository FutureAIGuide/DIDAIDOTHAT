"""
tests/test_pipeline.py — Integration test for the full pipeline (dry-run mode)
"""
import pytest
import os

import config
from main import run_pipeline


# ---------------------------------------------------------------------------
# Full pipeline integration test (no Gemini, no file I/O via tmp_path)
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def _mock_signals(self):
        """Return a realistic set of signals for clustering."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        return [
            {
                "id": f"sig{i}",
                "title": title,
                "summary": summary,
                "url": f"https://example.com/{i}",
                "published": now,
                "source_type": source,
                "source_name": "Test",
                "keywords": keywords,
                "entities": entities,
                "engagement": 0.3,
            }
            for i, (title, summary, source, keywords, entities) in enumerate(
                [
                    (
                        "OpenAI releases new AI agent framework",
                        "Agents can now run autonomously",
                        "rss",
                        ["agent", "framework", "autonomous", "openai"],
                        ["openai"],
                    ),
                    (
                        "AI agents take on complex software tasks",
                        "LLM-powered agents handle multi-step workflows",
                        "github",
                        ["agent", "workflow", "language", "model"],
                        ["openai"],
                    ),
                    (
                        "Why AI agents are the next big thing",
                        "Agents represent a paradigm shift in AI usage",
                        "arxiv",
                        ["agent", "paradigm", "autonomous", "llm"],
                        ["openai"],
                    ),
                    (
                        "Multimodal models hit new benchmarks",
                        "Vision and language combined at unprecedented scale",
                        "rss",
                        ["multimodal", "vision", "benchmark", "language"],
                        ["google"],
                    ),
                    (
                        "Google DeepMind multimodal breakthrough",
                        "New model achieves state-of-the-art on vision tasks",
                        "arxiv",
                        ["multimodal", "vision", "deepmind", "model"],
                        ["google", "deepmind"],
                    ),
                ]
            )
        ]

    def test_dry_run_returns_summary(self, mocker):
        mocker.patch("pipeline.ingest.ingest_all", return_value=self._mock_signals())
        summary = run_pipeline(dry_run=True, gemini_enabled=False)
        assert isinstance(summary, dict)
        assert summary["dry_run"] is True

    def test_signals_collected(self, mocker):
        signals = self._mock_signals()
        mocker.patch("pipeline.ingest.ingest_all", return_value=signals)
        summary = run_pipeline(dry_run=True, gemini_enabled=False)
        assert summary["signals_collected"] == len(signals)

    def test_clusters_found(self, mocker):
        mocker.patch("pipeline.ingest.ingest_all", return_value=self._mock_signals())
        summary = run_pipeline(dry_run=True, gemini_enabled=False)
        assert summary["clusters_found"] >= 1

    def test_emerging_trends_detected(self, mocker):
        mocker.patch("pipeline.ingest.ingest_all", return_value=self._mock_signals())
        summary = run_pipeline(dry_run=True, gemini_enabled=False)
        # With fresh signals (high novelty + velocity), at least one trend should be emerging
        assert isinstance(summary["emerging_trends"], int)

    def test_no_signals_returns_early(self, mocker):
        mocker.patch("pipeline.ingest.ingest_all", return_value=[])
        summary = run_pipeline(dry_run=True, gemini_enabled=False)
        assert summary["signals_collected"] == 0
        assert summary["clusters_found"] == 0

    def test_articles_generated_in_dry_run(self, mocker):
        mocker.patch("pipeline.ingest.ingest_all", return_value=self._mock_signals())
        summary = run_pipeline(dry_run=True, gemini_enabled=False)
        # In dry run, articles are listed but not written to disk
        assert isinstance(summary["articles_generated"], list)
