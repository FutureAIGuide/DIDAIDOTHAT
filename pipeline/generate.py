"""
pipeline/generate.py — Content Generation

Uses Gemini to produce a full article from a trend cluster.
Enforces virality psychology: curiosity gap, pattern interrupt,
narrative tension, and share trigger.

Article schema (also written to /content/posts/<slug>.md):
{
    "title":       str,
    "slug":        str,
    "body":        str,   # Markdown
    "meta_desc":   str,
    "cluster":     dict,
    "published":   str,   # ISO-8601
}
"""
from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any

import config
from .gemini import call_gemini

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_ARTICLE_PROMPT = """\
You are a senior tech journalist writing for "DID AI DO THAT?!" — a newsletter
that explains AI trends to builders and founders BEFORE they peak.

Write a high-retention, high-signal article based on the following trend cluster.

## Trend Cluster
Topic: {topic}
Status: {status}
Momentum score: {momentum_score}

## Narrative
Trend: {trend}
Core question: {core_question}
Why now: {why_now}
Contradiction: {contradiction}
Prediction: {prediction}

## Source signals (use these as evidence)
{headlines}

## Mandatory article structure
1. **Hook** — Open with a curiosity gap: "This doesn't add up…" or equivalent.
2. **Pattern interrupt** — Challenge what readers expect.
3. **Evidence** — Synthesise the signals above into a coherent argument.
4. **Narrative tension** — Guide readers: confusion → investigation → insight.
5. **Share trigger** — End with "This changes how we should think about X."

## Tone & format
- Markdown format with ## subheadings
- 600-900 words
- No fluff, no filler
- Include at least one bold "key insight" callout
- Include these three hooks naturally in the text:
  * "{debate_hook}"
  * "{identity_hook}"
  * "{prediction_hook}"

## Output format
Return ONLY valid JSON (no markdown fences) with exactly these fields:
{{
  "title": "compelling article headline",
  "meta_desc": "SEO meta description (max 160 chars)",
  "body": "full article in Markdown"
}}
"""


def _slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-")[:60]


def _build_article_prompt(cluster: dict[str, Any]) -> str:
    narrative = cluster.get("narrative", {})
    headlines = "\n".join(
        f"- [{item['title']}]({item['url']})" for item in cluster["items"][:6]
    )
    return _ARTICLE_PROMPT.format(
        topic=cluster["topic"],
        status=cluster.get("status", "emerging"),
        momentum_score=cluster.get("momentum_score", 0),
        trend=narrative.get("trend", ""),
        core_question=narrative.get("core_question", ""),
        why_now=narrative.get("why_now", ""),
        contradiction=narrative.get("contradiction", ""),
        prediction=narrative.get("prediction", ""),
        headlines=headlines,
        debate_hook=config.DEBATE_HOOK,
        identity_hook=config.IDENTITY_HOOK,
        prediction_hook=config.PREDICTION_HOOK,
    )


def _parse_article_response(raw: str) -> dict[str, str]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        data = json.loads(raw)
        if {"title", "meta_desc", "body"}.issubset(data.keys()):
            return {k: str(data[k]) for k in ("title", "meta_desc", "body")}
    except json.JSONDecodeError:
        pass
    # Fallback: use raw text as body
    return {
        "title": "An Emerging AI Trend You Shouldn't Miss",
        "meta_desc": "Early signals are pointing to a major shift in AI. Here's what's happening.",
        "body": raw,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_article(cluster: dict[str, Any], gemini_enabled: bool = True) -> dict[str, Any]:
    """Generate a full article for a trend cluster."""
    now = datetime.now(timezone.utc).isoformat()

    if gemini_enabled and config.GEMINI_API_KEY:
        prompt = _build_article_prompt(cluster)
        raw = call_gemini(prompt)
        article_data = _parse_article_response(raw)
    else:
        # Heuristic fallback
        narrative = cluster.get("narrative", {})
        article_data = {
            "title": f"Why {cluster.get('topic', 'AI')} Is the Trend You're Not Watching Yet",
            "meta_desc": narrative.get("core_question", "")[:160],
            "body": _fallback_body(cluster),
        }

    slug = _slugify(article_data["title"])
    article = {
        "title": article_data["title"],
        "slug": slug,
        "meta_desc": article_data["meta_desc"],
        "body": article_data["body"],
        "cluster": cluster,
        "published": now,
    }

    _save_article(article)
    return article


def _fallback_body(cluster: dict[str, Any]) -> str:
    narrative = cluster.get("narrative", {})
    items_list = "\n".join(
        f"- [{i['title']}]({i['url']})" for i in cluster["items"][:5]
    )
    return f"""## {cluster.get("topic", "AI Trend")}

{config.DEBATE_HOOK}

{narrative.get("why_now", "Multiple signals are converging right now.")}

**Key insight:** {narrative.get("contradiction", "The implications may surprise you.")}

### What we're seeing

{items_list}

### What this means

{narrative.get("prediction", "This will show up everywhere in 6 months.")}

{config.IDENTITY_HOOK}

{config.PREDICTION_HOOK}

*This changes how we should think about AI development.*
"""


def _save_article(article: dict[str, Any]) -> None:
    os.makedirs(config.POSTS_DIR, exist_ok=True)
    filename = os.path.join(config.POSTS_DIR, f"{article['slug']}.md")
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(f"# {article['title']}\n\n")
        fh.write(f"*Published: {article['published']}*\n\n")
        fh.write(f"*{article['meta_desc']}*\n\n")
        fh.write("---\n\n")
        fh.write(article["body"])
    logger.info("Article saved: %s", filename)
