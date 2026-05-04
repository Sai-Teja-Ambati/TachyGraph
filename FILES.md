# TachyGraph — File Descriptions

## Root
- **docker-compose.yaml** — Defines 2 services (tachy_db, tachy_app), volumes, network, and all environment variables.
- **README.md** — Full project documentation: setup, usage, architecture, endpoints, configuration, design decisions.
- **FUNCTIONALITIES.md** — One-line description of every feature in the system.

## SQL
- **sql/init.sql** — Unified PostgreSQL schema: 5 node labels, 6 edge labels, 5 tables, 4 triggers, 4 functions, 7 indexes.

## Config
- **config/postgresql.conf** — PostgreSQL tuning: shared_buffers=128MB, max_connections=10, work_mem=4MB, SSD-optimized.

## UI
- **ui/graph.html** — 3D force-directed knowledge graph visualization using 3d-force-graph; 4 node colors, similarity slider, tooltips.

## App — Root
- **app/Dockerfile** — Python 3.11-slim, gcc/g++, pip install, uvicorn entrypoint on port 8000.
- **app/requirements.txt** — All Python dependencies: psycopg, pgvector, fastapi, httpx, faiss-cpu, strands-agents, mcp, pymupdf, beautifulsoup4.
- **app/tachyrag/config.py** — All constants and environment variable parsing: Ollama, FAISS, search weights, RAG toggles, prompts.
- **app/tachyrag/main.py** — FastAPI app with 41 endpoints, lifespan (DB/FAISS/cron init), CORS, request logging middleware.

## App — agents/
- **agents/tools.py** — 9 Strands `@tool` definitions wrapping existing TachyGraph functions for agent use.
- **agents/core.py** — Core Strands agent with OllamaModel provider, system prompt, all 9 tools, session integration.
- **agents/orchestrator.py** — Multi-agent orchestrator that classifies intent and delegates to researcher, librarian, or assistant.
- **agents/specialists.py** — Three specialist agents (Researcher, Librarian, Assistant) with focused tool subsets and tailored prompts.
- **agents/researcher.py** — Standalone research workflow: searches memory → crawls web → ingests → synthesizes → stores findings.
- **agents/memory_manager.py** — Autonomous graph maintenance agent: checks expiry, reaffirms important facts, compacts duplicates.

## App — chat/
- **chat/chat.py** — RAG chat with HyDE, adaptive K, compression, reranker, streaming SSE, self-learning loop, preference injection.
- **chat/session.py** — Persistent DB-backed sessions: create, get history, add message, list, delete; survives restarts.
- **chat/feedback.py** — Processes user corrections: "correct" reaffirms, "wrong" expires, "correction" inserts new high-confidence answer.

## App — core/
- **core/db.py** — PostgreSQL connection pool (psycopg3), insert_node, insert_edge, bulk_insert_summaries, ensure_project, BM25 DF updates.
- **core/llm_client.py** — Multi-provider LLM client (Ollama, OpenAI, Claude, Gemini): generate, generate_stream, embed_text, embed_batch, list_models, health check. Provider selected via LLM_PROVIDER env var.
- **core/embedder.py** — Embedding wrapper: L2-normalized 1024-dim vectors, @lru_cache(1024), embed_batch for bulk operations.
- **core/tfidf.py** — Pure Python BM25 tokenizer: regex tokenizer, stop words, compute_tf returns {term: count} + doc_length.
- **core/export.py** — Export knowledge graph as JSON (nodes + edges without embeddings) or markdown (grouped by source); import with re-embedding.
- **core/middleware.py** — Starlette middleware logging method, path, status code, and latency in milliseconds for every HTTP request.
- **core/webhooks.py** — Fires HTTP POST to WEBHOOK_URL for events: facts_expiring, task_due, ingestion_complete, maintenance_complete.

## App — ingest/
- **ingest/chunker.py** — Splits documents into 8K-char pages: MarkdownHeaderTextSplitter for markdown, RecursiveCharacterTextSplitter fallback.
- **ingest/summarizer.py** — Parallel LLM extraction (ThreadPoolExecutor(4)): head summary, body facts, keywords, temporal date per chunk.
- **ingest/ingestor.py** — Full ingestion pipeline: chunk → summarize → batch embed → batch insert SUMMARY nodes → PART_OF edges → project summary.
- **ingest/file_ingestor.py** — Local file drop: scans ingest/ folder, handles txt/md/pdf/audio, moves to done/ or failed/ after processing.
- **ingest/web_crawler.py** — Cloudflare /crawl API client (start → poll → collect) with local httpx+BeautifulSoup fallback when CF not configured.
- **ingest/chat_parser.py** — Parses ChatGPT, Claude, Gemini exports (JSON) and raw pasted text into Q&A pairs; auto-detects platform format.
- **ingest/audio_ingestor.py** — Transcribes audio files (mp3, wav, m4a, ogg, flac, webm) via faster-whisper, then feeds text into ingestion pipeline.

## App — memory/
- **memory/observer.py** — Ollama-powered fact extraction: parses interaction text into structured Q&A with subject/predicate/object/confidence.
- **memory/clustering.py** — Question hub management: finds existing hub by cosine > 0.95 or creates new QUESTION node with degree_cap=10.
- **memory/memory.py** — Answer insertion: creates ANSWER node with embedding + BM25 TF, links via ANSWERS edge; DB trigger handles 10-slot eviction.
- **memory/weaver.py** — Cross-cluster weaving: links new answer to top-3 similar answers in OTHER question clusters via CONTEXT_OF edges.

## App — search/
- **search/search.py** — Main search entry: adaptive K, optional HyDE, query decomposition, two-stage retrieval, RRF merge, access-count bump.
- **search/search_hybrid.py** — FAISS-accelerated search: fast (pure FAISS) and hybrid (FAISS recall → pgvector multi-signal re-rank).
- **search/strands.py** — 4 parallel search strategies: exact match (BM25+cosine+decay), context weave (edge hop), temporal (SUPERSEDES chain), semantic.
- **search/disambiguator.py** — Ollama intent extraction: classifies query as debugging/reference/comparison/history with rephrased query and entities.
- **search/responder.py** — Fact-chain builder: 2-hop edge traversal from search results, provenance-backed, returns "Data missing" if empty.
- **search/reranker.py** — Smart context selection: dedup facts already in conversation, diversity cap (max 3 per source), preserves rank order.
- **search/hyde.py** — HyDE: generates hypothetical answer via Ollama, embeds that instead of raw query for better document matching.
- **search/decomposer.py** — Query decomposition: detects complex queries (comparisons, multi-part), breaks into 2-4 focused sub-queries.
- **search/compressor.py** — Contextual compression: extracts only relevant sentences from 8K pages via parallel Ollama calls (ThreadPoolExecutor).
- **search/rrf.py** — Reciprocal Rank Fusion: merges multiple ranked lists by position (1/(k+rank)) instead of raw scores.
- **search/hyde.py** — HyDE: generates hypothetical answer via Ollama, embeds that instead of raw query for better document matching.
- **search/decomposer.py** — Query decomposition: detects complex queries (comparisons, multi-part), breaks into 2-4 focused sub-queries.
- **search/compressor.py** — Contextual compression: extracts only relevant sentences from 8K pages via parallel Ollama calls (ThreadPoolExecutor).
- **search/rrf.py** — Reciprocal Rank Fusion: merges multiple ranked lists by position (1/(k+rank)) instead of raw scores.

## App — graph/
- **graph/temporal.py** — Temporal management: reaffirm facts, find expiring nodes, resolve conflicts by recency (newer valid_from wins), access-count auto-reaffirm.
- **graph/mmr.py** — MMR edge linking: creates RELEVANT_TO edges between SUMMARY nodes using Maximal Marginal Relevance for diverse connections.
- **graph/compaction.py** — Dedup: finds near-duplicate SUMMARY nodes (cosine > 0.98 + same source), expires duplicate; expiry report dashboard.
- **graph/preferences.py** — User preferences: PREFERENCE nodes (never expire), get/set/merge with defaults, builds prompt context fragment.
- **graph/tasks.py** — Task/reminder system: create with due date, list due/all, complete; tasks can link to related graph nodes.
- **graph/maintenance.py** — Scheduled cleanup: purge expired nodes, compact all projects, clean stale sessions (30d), clean completed tasks (7d).

## App — faiss/
- **faiss/vector_index.py** — UnifiedVectorIndex: CPU default, GPU auto-detect with fallback, IVF-PQ, train/add/search/remove/save/load.
- **faiss/sync_faiss.py** — Background async sync: pgvector → FAISS index every 30s or 1000 vectors; queue add/remove, full sync on demand.

## App — mcp/
- **mcp/server.py** — FastMCP server with 10 tools: search, store, observe, recall, chat (agent-powered), tasks, preferences, feedback, projects, status.
