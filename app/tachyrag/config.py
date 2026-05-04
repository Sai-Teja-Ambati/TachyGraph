import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://tachy_admin:tachy_password_2026@localhost:5432/tachygraph",
)

# LLM Provider (generation)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # ollama, openai, anthropic, gemini

# Embedding Provider (can differ from LLM — e.g. use OpenAI embeddings with Claude generation)
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "")  # empty = same as LLM_PROVIDER. Options: ollama, openai, gemini

# Ollama (default)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-coder:30b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")

# OpenAI (set LLM_PROVIDER=openai)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Anthropic / Claude (set LLM_PROVIDER=anthropic)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# Google Gemini (set LLM_PROVIDER=gemini)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")

# Pool
POOL_MIN = 2
POOL_MAX = 5

# Embedding
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

# MMR
MMR_LAMBDA = 0.7
MMR_CANDIDATES = 50
MMR_EDGE_TOP_K = 5
MMR_ON_INGEST = os.getenv("MMR_ON_INGEST", "false").lower() == "true"

# MMR Cron
MMR_CRON_ENABLED = os.getenv("MMR_CRON_ENABLED", "false").lower() == "true"
MMR_CRON_SCHEDULE = os.getenv("MMR_CRON_SCHEDULE", "0 */6 * * *")  # every 6 hours

# Search
SEARCH_K = 21
MAX_STRANDS = 2

# Search signal weights (must sum to 1.0)
SEARCH_BM25_WEIGHT = float(os.getenv("SEARCH_BM25_WEIGHT", "0.1"))
SEARCH_VECTOR_WEIGHT = float(os.getenv("SEARCH_VECTOR_WEIGHT", "0.6"))
SEARCH_TEMPORAL_WEIGHT = float(os.getenv("SEARCH_TEMPORAL_WEIGHT", "0.3"))

_weight_sum = round(SEARCH_BM25_WEIGHT + SEARCH_VECTOR_WEIGHT + SEARCH_TEMPORAL_WEIGHT, 4)
if _weight_sum != 1.0:
    raise ValueError(
        f"SEARCH_BM25_WEIGHT ({SEARCH_BM25_WEIGHT}) + SEARCH_VECTOR_WEIGHT ({SEARCH_VECTOR_WEIGHT}) "
        f"+ SEARCH_TEMPORAL_WEIGHT ({SEARCH_TEMPORAL_WEIGHT}) = {_weight_sum}, must sum to 1.0"
    )

_valid_providers = {"ollama", "openai", "anthropic", "gemini"}
if LLM_PROVIDER not in _valid_providers:
    raise ValueError(f"LLM_PROVIDER='{LLM_PROVIDER}' invalid. Must be one of: {_valid_providers}")
if EMBED_PROVIDER and EMBED_PROVIDER not in _valid_providers - {"anthropic"}:
    raise ValueError(f"EMBED_PROVIDER='{EMBED_PROVIDER}' invalid. Must be one of: ollama, openai, gemini (anthropic has no embed API)")

# Search features (toggle on/off)
SEARCH_USE_HYDE = os.getenv("SEARCH_USE_HYDE", "false").lower() == "true"
SEARCH_USE_DECOMPOSITION = os.getenv("SEARCH_USE_DECOMPOSITION", "true").lower() == "true"
SEARCH_USE_COMPRESSION = os.getenv("SEARCH_USE_COMPRESSION", "true").lower() == "true"

# FAISS (CPU default, GPU optional)
FAISS_USE_GPU = os.getenv("FAISS_USE_GPU", "false").lower() == "true"
FAISS_GPU_ID = int(os.getenv("FAISS_GPU_ID", "0"))
FAISS_GPU_MEMORY = int(os.getenv("FAISS_GPU_MEMORY", "512"))
FAISS_NLIST = int(os.getenv("FAISS_NLIST", "64"))
FAISS_M = 16
FAISS_NBITS = 8
FAISS_NPROBE = 64
FAISS_SYNC_INTERVAL = 30
FAISS_SYNC_BATCH = 1000
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "/data/faiss")

# Memory
QA_CLUSTER_GLOBAL = os.getenv("QA_CLUSTER_GLOBAL", "true").lower() == "true"
QA_SKIP_LLM_EXTRACT = os.getenv("QA_SKIP_LLM_EXTRACT", "true").lower() == "true"
SIMILARITY_THRESHOLD = 0.95
CONFIDENCE_FLOOR = 0.85
WEAVE_TOP_K = 3

# Temporal
DEFAULT_VALIDITY_DAYS = 5
EXPIRY_WARNING_HOURS = 24

# BM25
BM25_K1 = 1.2
BM25_B = 0.75

# Chunking — large pages stored as content on SUMMARY nodes
# Summary head sentence gets embedded, full content returned for generation
CHUNK_SIZE = 8192
CHUNK_OVERLAP = 200

# Local file ingestion
INGEST_DIR = os.getenv("INGEST_DIR", "/data/ingest")
INGEST_EXTENSIONS = {".txt", ".md", ".rst", ".html", ".json", ".yaml", ".yml", ".pdf", ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}

# Chat
CHAT_CONTEXT_K = int(os.getenv("CHAT_CONTEXT_K", "10"))
CHAT_HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "10"))
CHAT_SYSTEM_PROMPT = """You are a personal knowledge assistant powered by TachyGraph. You have access to a personal knowledge graph containing facts, documents, and Q&A pairs the user has stored.

Rules:
- Answer based on the provided context facts. Cite which facts you used.
- If the context doesn't contain enough information, say so honestly.
- Be concise and direct.
- If the user corrects you, acknowledge it."""

# Cloudflare Browser Rendering (web crawl)
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "")
CF_CRAWL_POLL_INTERVAL = int(os.getenv("CF_CRAWL_POLL_INTERVAL", "5"))
CF_CRAWL_POLL_MAX = int(os.getenv("CF_CRAWL_POLL_MAX", "120"))

# TachyWorker System Prompt
TACHY_EXTRACT_PROMPT = """/no_think
You are a strict JSON extraction agent. Output ONLY a single valid JSON object. No markdown, no explanation, no preamble.

RULES:
- Do NOT use double quotes inside any string value. Use single quotes instead.
- Output MUST be valid JSON and nothing else.

Extract the following from the text below:

1. head: A detailed summary paragraph of 50 to 100 words covering the main topic, key methods, findings, and conclusions in the text. Must be information-dense for search retrieval. No inner double quotes.

2. body: A list of short atomic facts stated in the text. No inner double quotes.

3. keywords: A list of key technical terms, proper nouns, and domain-specific phrases.

4. temporal_context: You MUST output a date in strict YYYY-MM-DD format. Follow these steps IN ORDER:
   Step 1 - Look for explicit publication or submission dates (e.g. 'arXiv ... 7 Apr 2026', 'submitted March 2025', 'published 2024'). If found, use it.
   Step 2 - Look for the most recent calendar year mentioned in a MEANINGFUL context: paper citations like '[1] Author, Title, Journal, 2024', copyright notices, 'funded in 2023', 'proposed in 2020'. A number is a year ONLY when the surrounding text makes it clearly a calendar year. Numbers in equations, figure labels, page numbers, grant IDs, zip codes, phone numbers, dimensions, or parameter values are NEVER years.
   Step 3 - If no date was found in Step 1 or Step 2, output exactly "{system_date}".
   FORMAT: Full date -> YYYY-MM-DD as-is. Month+Year -> YYYY-MM-01. Year only -> YYYY-07-01.

Required JSON format:
{{"head": "...", "body": ["...", "..."], "keywords": ["...", "..."], "temporal_context": "YYYY-MM-DD"}}

Text:
{text}
"""
