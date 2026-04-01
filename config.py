"""
Configuration for the DID AI DO THAT?! content engine.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
YOUTUBE_API_KEY: str = os.getenv("YOUTUBE_API_KEY", "")

# --- Ingestion Sources ---

# RSS feeds covering AI news
RSS_FEEDS: list[str] = [
    "https://feeds.feedburner.com/blogspot/gJZg",        # Google AI Blog
    "https://openai.com/blog/rss.xml",                   # OpenAI Blog
    "https://huggingface.co/blog/feed.xml",              # HuggingFace Blog
    "https://techcrunch.com/tag/artificial-intelligence/feed/",
    "https://venturebeat.com/category/ai/feed/",
    "https://www.technologyreview.com/topic/artificial-intelligence/feed",
    "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "https://thesequence.substack.com/feed",
]

# GitHub search queries for trending AI repositories
GITHUB_SEARCH_QUERIES: list[str] = [
    "AI agent",
    "LLM fine-tuning",
    "multimodal model",
    "AI video generation",
    "AI robotics",
]

# YouTube channel IDs covering AI topics
YOUTUBE_CHANNEL_IDS: list[str] = [
    "UCbmNph6atAoGfqLoCL_duAg",  # Yannic Kilcher
    "UCWX3yGbODg0JoQqBJhAApUA",  # Two Minute Papers
    "UC0rqucBdTuFTjJiefW5t-IQ",  # AI Explained
]

# arXiv categories for AI research
ARXIV_CATEGORIES: list[str] = [
    "cs.AI",
    "cs.LG",
    "cs.CV",
    "cs.CL",
    "cs.RO",
]

# --- Clustering & Trend Detection ---
CLUSTER_MIN_ITEMS: int = int(os.getenv("CLUSTER_MIN_ITEMS", "2"))
COSINE_SIMILARITY_THRESHOLD: float = 0.30
KEYWORD_OVERLAP_THRESHOLD: int = 2
NOVELTY_WINDOW_DAYS: int = 7       # Items older than this are not "novel"
MOMENTUM_THRESHOLD: float = float(os.getenv("MOMENTUM_THRESHOLD", "0.5"))
EMERGING_VELOCITY_THRESHOLD: float = float(os.getenv("EMERGING_VELOCITY_THRESHOLD", "2.0"))

# Momentum score weights (must sum to 1.0)
MOMENTUM_WEIGHTS: dict[str, float] = {
    "velocity": 0.4,
    "source_diversity": 0.3,
    "novelty": 0.2,
    "engagement": 0.1,
}

# Max source diversity score (number of distinct platform types)
MAX_SOURCE_DIVERSITY: int = 4   # rss, github, youtube, arxiv

# --- File Paths ---
CONTENT_DIR: str = os.getenv("CONTENT_DIR", "content")
DATA_DIR: str = os.getenv("DATA_DIR", "data")
POSTS_DIR: str = f"{CONTENT_DIR}/posts"
TRENDS_DIR: str = f"{CONTENT_DIR}/trends"
CLUSTERS_FILE: str = f"{DATA_DIR}/clusters.json"
TREND_HISTORY_FILE: str = f"{DATA_DIR}/trend_history.json"
FEEDBACK_FILE: str = f"{DATA_DIR}/feedback.json"

# --- Content Generation ---
GEMINI_MODEL: str = "gemini-1.5-flash"
MAX_DAILY_TRENDS: int = 3
MAX_DAILY_ARTICLES: int = 2

# --- Virality Psychology Hooks ---
DEBATE_HOOK: str = "I think people are missing what's actually happening here."
IDENTITY_HOOK: str = "If you're building with AI, this matters."
PREDICTION_HOOK: str = "This is going to show up everywhere in 6 months."
