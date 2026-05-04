# TachyGraph — Functionalities

## AI Agents
- **Single Agent** — Strands-powered agent with 9 tools; LLM autonomously decides which tools to call, when to iterate, and when to stop.
- **Multi-Agent Orchestrator** — Classifies user intent and delegates to specialist agents (Researcher, Librarian, Assistant) for complex multi-step requests.
- **Research Agent** — Autonomously searches memory, crawls URLs, ingests new content, searches again, synthesizes findings, and stores key points.
- **Memory Manager Agent** — Autonomously reviews graph health, reaffirms important expiring facts, compacts duplicates, and reports maintenance actions.

## Chat
- **RAG Chat** — Searches knowledge graph for context, builds preference-aware prompt with conversation history, generates response via Ollama.
- **Streaming Chat (SSE)** — Token-by-token streaming via Server-Sent Events; self-learning and reaffirmation run after stream completes.
- **Chat Feedback** — User marks responses as correct (confidence→0.95), wrong (expires answer), or provides correction (new answer at 0.98 supersedes old).
- **Persistent Sessions** — Conversation history stored in PostgreSQL; survives container restarts; supports multi-turn context across sessions.
- **Self-Learning Loop** — Every chat response auto-observes the Q&A pair into the graph at confidence 0.85; retrieved facts auto-reaffirmed.

## Search
- **Multi-Signal Search** — Two-stage: HNSW cosine pre-filter → BM25 + vector + temporal decay re-rank with configurable weights.
- **Deep Search** — Ollama disambiguates intent, then runs parallel search strands (exact match, semantic, temporal, context weave).
- **FAISS Fast Search** — Pure FAISS IVF-PQ search (<5ms on GPU, ~20ms on CPU) for high-throughput retrieval.
- **Hybrid Search** — FAISS recall (top 100) → pgvector multi-signal re-rank for precision-critical queries.
- **Fact Chain** — 2-hop edge traversal building provenance-backed fact chains; returns "Data missing" instead of hallucinating.
- **HyDE** — Generates hypothetical answer, embeds that instead of raw query; bridges vocabulary gap between questions and documents.
- **Query Decomposition** — Breaks complex queries ("compare X vs Y") into focused sub-queries, searches each, merges via Reciprocal Rank Fusion.
- **Contextual Compression** — After retrieval, extracts only relevant sentences from 8K pages; reduces context from ~80K to ~5K tokens.
- **Reciprocal Rank Fusion** — Merges ranked lists by position not score; each signal contributes equally regardless of score scale.
- **Adaptive K** — Heuristic estimates how many results a query needs; skips search entirely for greetings, fetches more for complex queries.
- **Smart Reranker** — Diversity + dedup before context injection; caps results per source, skips facts already cited in conversation.
- **Configurable Weights** — `SEARCH_BM25_WEIGHT`, `SEARCH_VECTOR_WEIGHT`, `SEARCH_TEMPORAL_WEIGHT` — set any to 1.0 for pure single-signal search.

## Ingestion
- **Text Ingestion** — Chunks text into 8K pages, LLM extracts summary/keywords/date per page, batch embeds, batch inserts as SUMMARY nodes.
- **Local File Ingestion** — Drop files into `ingest/` folder; supports txt, md, rst, html, json, yaml, pdf, mp3, wav, m4a, ogg, flac, webm.
- **PDF Ingestion** — Extracts text from multi-page PDFs via pymupdf; graceful error for image-only PDFs.
- **Audio Transcription** — Transcribes audio files via faster-whisper (optional), then ingests the text through the standard pipeline.
- **Web Crawl (Cloudflare)** — Async crawl via Cloudflare Browser Rendering API; follows links, returns markdown, respects robots.txt.
- **Web Crawl (Local)** — Fallback scraper using httpx + BeautifulSoup when Cloudflare is not configured; same-domain BFS, respects robots.txt.
- **ChatGPT/Claude/Gemini Import** — Parses conversation exports (JSON) or raw pasted text; auto-detects platform; feeds Q&A pairs into graph.
- **Dedup on Re-ingestion** — Checks for existing SUMMARY nodes with same source_url before inserting; prevents duplicate content.
- **Batch Embeddings** — All summaries embedded in a single Ollama HTTP call instead of N individual calls.
- **Batch Inserts** — Single PostgreSQL transaction per document; one BM25 stats update for the entire batch.

## Memory (Q&A Layer)
- **Fact Observer** — Regex fast-path for Q:/A: formatted input (~2s), LLM fallback for unstructured text; discards below confidence 0.85.
- **Question Clustering** — Global by default (cross-project hubs, cosine > 0.95); set `QA_CLUSTER_GLOBAL=false` to scope per project.
- **10-Slot Eviction** — Each question hub holds up to 10 answers; lowest-confidence answer evicted when full, edges migrated to replacement.
- **Answer Weaving** — New answers linked to top-3 similar answers in other clusters via CONTEXT_OF edges for cross-topic discovery.
- **Access-Count Auto-Reaffirm** — Frequently retrieved facts (access_count ≥ 5) auto-extend validity when expiring within 48 hours.

## Graph Maintenance
- **Temporal Conflict Resolution** — When multiple answers exist for the same question, the newer fact (by extracted date) wins; ties broken by confidence.
- **Auto-Compaction** — Finds SUMMARY nodes with cosine > 0.98 and same source_url; expires duplicate, creates SUPERSEDES edge.
- **Expiry Report** — Dashboard showing total nodes, expiring in 24h/7d, and high-value facts (access_count > 3) about to expire.
- **MMR Edge Linking** — Creates RELEVANT_TO edges between SUMMARY nodes using Maximal Marginal Relevance scoring for diverse connections.
- **Scheduled Maintenance** — Background cron purges expired nodes, compacts all projects, cleans stale sessions and completed tasks.

## User Features
- **User Preferences** — PREFERENCE nodes store response style, expertise level, language, topics of interest; injected into chat prompts.
- **Tasks / Reminders** — Create, list, complete reminders with due dates; tasks can link to related knowledge graph nodes.
- **Export / Import** — Export knowledge graph as JSON (nodes + edges) or markdown (organized by source); import from backup with re-embedding.
- **3D Graph Visualization** — Browser-based 3D force-directed graph with color-coded nodes, similarity edges, threshold slider, hover tooltips.

## Infrastructure
- **MCP Server** — 10 tools for Claude Desktop, VS Code, Cursor; agent-powered chat, search, store, observe, recall, tasks, preferences, feedback.
- **Multi-Provider LLM** — Switch between Ollama (local), OpenAI, Claude, Gemini via `LLM_PROVIDER` env var. `EMBED_PROVIDER` can differ from generation provider. Pure httpx, no SDK dependencies.
- **Multi-Model Support** — Switch Ollama models per request via `model` parameter on /chat and /agent endpoints; GET /models lists available.
- **Webhook Notifications** — HTTP POST to configurable URL for events: facts expiring, tasks due, ingestion complete, maintenance done.
- **Request Logging** — Middleware logs method, path, status code, and latency in milliseconds for every request.
- **FAISS CPU/GPU** — Unified index with auto GPU detection and CPU fallback; IVF-PQ compression for large datasets, automatic flat index fallback when < 1920 vectors; background async sync from pgvector.
- **BM25 Scoring** — Pure Python tokenizer + PL/pgSQL BM25 function with TF saturation (k1=1.2) and length normalization (b=0.75).
