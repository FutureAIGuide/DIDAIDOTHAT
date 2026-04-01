# DIDAIDOTHAT
DID AI DO THAT?! - Complete System

```markdown
# DIDAIDOTHAT

**DID AI DO THAT?!** is a Python content engine that detects *emerging* AI trends (before they peak) by aggregating signals from multiple sources, clustering them, scoring momentum, and optionally using **Google Gemini** to extract narratives and generate publish-ready articles plus micro-content.

## What it does

Pipeline (high level):

1. **Ingest signals** from:
   - RSS feeds (AI news / blogs)
   - GitHub repo search results
   - YouTube channels (optional; requires API key)
   - arXiv categories
2. **Normalize** each signal into a common schema (`title`, `summary`, `url`, `published`, `keywords`, `entities`, etc.)
3. **Cluster** similar items into trend groups (greedy single-linkage)
4. **Score momentum** (velocity, novelty, source diversity, engagement proxy)
5. **Classify trends** as `emerging`, `peaking`, or `declining`
6. **(Optional) Gemini narrative extraction** for emerging clusters
7. **(Optional) Gemini article generation** (Markdown article saved to disk)
8. **(Optional) Atomize content** into:
   - X/Twitter thread
   - LinkedIn post
   - Hot take
   - Quote cards
   - Email teaser
9. **Feedback loop**: record what was published and reinforce topics that show repeated traction

## Repository layout

- `main.py` — orchestrates the full pipeline; CLI entry point
- `config.py` — sources, thresholds, output paths, hooks, model selection
- `pipeline/`
  - `ingest.py` — RSS/GitHub/YouTube/arXiv collection + normalization
  - `cluster.py` — clustering + momentum scoring
  - `trends.py` — classification + narrative extraction (Gemini optional)
  - `generate.py` — article generation (Gemini optional) → `content/posts/`
  - `atomize.py` — micro-content generation (Gemini optional) → `content/trends/`
  - `feedback.py` — store published records → `data/feedback.json`
  - `gemini.py` — thin wrapper around the Google Generative AI SDK
- `tests/` — unit tests for key pipeline modules
- `.env.example` — environment variables template
- `requirements.txt` — Python dependencies

## Quickstart

### 1) Create a virtual environment & install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Configure environment variables

Copy the template and fill in values:

```bash
cp .env.example .env
```

Minimum recommended:

- `GEMINI_API_KEY` (enables narrative extraction + content generation)

Optional:

- `GITHUB_TOKEN` (higher GitHub API rate limits)
- `YOUTUBE_API_KEY` (enables YouTube ingestion)

### 3) Run the pipeline

Dry run (no files written, no Gemini calls):

```bash
python main.py --dry-run
```

Run without Gemini (heuristic fallbacks):

```bash
python main.py --no-gemini
```

Full run (writes outputs to disk; uses Gemini if configured):

```bash
python main.py
```

## Outputs

By default, output goes to:

- `data/clusters.json` — scored clusters (without embedding the full item bodies)
- `data/trend_history.json` — historical record of detected emerging trends
- `data/feedback.json` — published-content records used for reinforcement
- `content/posts/<slug>.md` — generated articles
- `content/trends/<slug>_atoms.json` — atomized micro-content per article

(You can change base directories via `CONTENT_DIR` and `DATA_DIR`.)

## Tuning trend detection

Key knobs (see `.env.example` and `config.py`):

- `CLUSTER_MIN_ITEMS` — minimum items needed to form a cluster
- `MOMENTUM_THRESHOLD` — minimum momentum score to be considered “non-declining”
- `EMERGING_VELOCITY_THRESHOLD` — velocity cutoff for “emerging”
- `COSINE_SIMILARITY_THRESHOLD`, `KEYWORD_OVERLAP_THRESHOLD` — cluster similarity rules
- `NOVELTY_WINDOW_DAYS` — how long a signal stays “novel”

## Notes / known limitations

- RSS feeds vary widely in formatting; some summaries may be noisy.
- GitHub “engagement” is a lightweight proxy derived from stars.
- YouTube ingestion requires `YOUTUBE_API_KEY`.
- Gemini calls require `GEMINI_API_KEY`; otherwise the pipeline falls back to heuristic narratives/content.

## Testing

```bash
pytest
```

## License

Add a license if you plan to distribute this publicly (e.g., MIT, Apache-2.0).
```

If you tell me your preferred audience/tone (builders vs. creators vs. “newsletter ops”) and whether you want a “How to deploy” section (cron/GitHub Actions), I can tailor the README further.
