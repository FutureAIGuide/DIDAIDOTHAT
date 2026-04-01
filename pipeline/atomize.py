"""
pipeline/atomize.py — Loop Stage G1–G3: Content Atomization + Platform Targeting

From each article, generate platform-optimised micro-content:
  • X (Twitter) thread  (5-10 tweets, punchy, controversial)
  • LinkedIn post       (explanatory, professional, implications-focused)
  • Hot take            (1-2 sentences, spicy)
  • Quote cards         (2 text snippets suitable for image overlays)
  • Email teaser        (subject + 2-3 line preview)
  • SEO meta desc       (already in article, surfaced here)

Each piece includes the mandatory distribution hooks (G2).

Output schema saved to /content/trends/<slug>_atoms.json
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import config
from .gemini import call_gemini

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_ATOMIZE_PROMPT = """\
You are a social media strategist for "DID AI DO THAT?!" — an AI trend newsletter.

Based on this article, generate multi-platform micro-content.

## Article
Title: {title}
Meta description: {meta_desc}
Body (excerpt):
{body_excerpt}

## Mandatory hooks to weave in naturally
- Debate hook: "{debate_hook}"
- Identity hook: "{identity_hook}"
- Prediction hook: "{prediction_hook}"

## Platform requirements

### X (Twitter) thread
- 5-8 tweets
- Short, punchy, 1-2 sentences each
- Controversial or surprising angle
- First tweet is the hook (curiosity gap)
- Last tweet is the CTA / share trigger

### LinkedIn post
- 150-200 words
- More explanatory than Twitter
- Slightly professional tone
- Emphasise implications for builders and leaders

### Hot take
- 1-2 sentences maximum
- Spicy, direct, shareable

### Quote cards (2 items)
- Each: a single punchy sentence (max 20 words)
- Suitable for overlaying on an image

### Email teaser
- subject: compelling subject line
- preview: 2-3 sentences to tease the article

Return ONLY valid JSON (no markdown fences) with exactly these keys:
{{
  "twitter_thread": ["tweet1", "tweet2", ...],
  "linkedin_post": "...",
  "hot_take": "...",
  "quote_cards": ["quote1", "quote2"],
  "email_teaser": {{"subject": "...", "preview": "..."}}
}}
"""


def _build_atomize_prompt(article: dict[str, Any]) -> str:
    body_excerpt = article.get("body", "")[:800]
    return _ATOMIZE_PROMPT.format(
        title=article["title"],
        meta_desc=article.get("meta_desc", ""),
        body_excerpt=body_excerpt,
        debate_hook=config.DEBATE_HOOK,
        identity_hook=config.IDENTITY_HOOK,
        prediction_hook=config.PREDICTION_HOOK,
    )


def _parse_atoms(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        data = json.loads(raw)
        required = {"twitter_thread", "linkedin_post", "hot_take", "quote_cards", "email_teaser"}
        if required.issubset(data.keys()):
            return data
    except json.JSONDecodeError:
        pass
    return _fallback_atoms(raw)


def _fallback_atoms(title: str) -> dict[str, Any]:
    return {
        "twitter_thread": [
            f"🧵 {config.DEBATE_HOOK}",
            f"{config.IDENTITY_HOOK}",
            f"{config.PREDICTION_HOOK}",
            "Read the full breakdown →",
        ],
        "linkedin_post": (
            f"{config.DEBATE_HOOK}\n\n"
            f"{config.IDENTITY_HOOK}\n\n"
            f"{config.PREDICTION_HOOK}"
        ),
        "hot_take": config.DEBATE_HOOK,
        "quote_cards": [
            config.IDENTITY_HOOK,
            config.PREDICTION_HOOK,
        ],
        "email_teaser": {
            "subject": "The AI trend most people are sleeping on",
            "preview": (
                f"{config.DEBATE_HOOK} "
                f"{config.IDENTITY_HOOK} "
                "Full breakdown inside."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def atomize_content(article: dict[str, Any], gemini_enabled: bool = True) -> dict[str, Any]:
    """
    Generate platform-specific micro-content from an article.
    Returns the atoms dict and saves it to /content/trends/.
    """
    if gemini_enabled and config.GEMINI_API_KEY:
        prompt = _build_atomize_prompt(article)
        raw = call_gemini(prompt)
        atoms = _parse_atoms(raw)
    else:
        atoms = _fallback_atoms(article.get("title", ""))

    atoms["article_title"] = article["title"]
    atoms["article_slug"] = article.get("slug", "")
    atoms["meta_desc"] = article.get("meta_desc", "")

    _save_atoms(article.get("slug", "atoms"), atoms)
    return atoms


def _save_atoms(slug: str, atoms: dict[str, Any]) -> None:
    os.makedirs(config.TRENDS_DIR, exist_ok=True)
    filename = os.path.join(config.TRENDS_DIR, f"{slug}_atoms.json")
    with open(filename, "w", encoding="utf-8") as fh:
        json.dump(atoms, fh, indent=2, ensure_ascii=False)
    logger.info("Atoms saved: %s", filename)
