# TachyGraph

A personal knowledge agent with sparse graph memory, multi-signal RAG search, AI agents powered by [Strands SDK](https://github.com/strands-agents/sdk-python), and FAISS-accelerated vector retrieval. Built on PostgreSQL 16 + pgvector, powered by Ollama for local LLM inference.

![Knowledge Graph Overview](pictures/broad_view.png)

---

## Why TachyGraph — The Problems We Solve

Standard RAG systems embed documents into flat vector stores and retrieve by cosine similarity. This works for simple lookups but breaks down in four critical ways.

### 1. The Temporal Conflict Problem

When your knowledge base contains facts from different time periods, flat vector search has no way to know which version is current. A 2020 architecture doc and a 2024 migration guide both match "What database do we use?" — but only one is the truth *right now*.

**Our solution:** Every node carries `valid_from` / `valid_until` timestamps. An LLM extracts dates from content at ingest time, placing nodes on the correct timeline. Search uses exponential temporal decay (`exp(-0.1 × age_days)`) — a 30-day-old fact scores 0.05 while today's fact scores 1.0. Conflicting facts are linked with `SUPERSEDES` edges, and source priority rules (GitHub > GitLab > Bitbucket) resolve ties. Facts auto-expire after their validity window, and frequently accessed facts auto-extend via access-count reaffirmation.

### 2. The Chunking Problem (Contextual Disambiguation)

Naive chunking splits documents into fixed-size blocks, destroying context. The sentence "It uses PostgreSQL" means nothing without knowing what "it" refers to. Chunks lose their heading hierarchy, their position in the document, and their relationship to neighboring content.

**Our solution:** Documents are split into 8K-character pages preserving markdown heading hierarchy. Each page becomes a SUMMARY node that holds the full content (for generation) and a concise head sentence (for embedding). An LLM extracts structured metadata — head summary, atomic body facts, keywords, and temporal context — at ingest time. SUMMARY nodes link to their PROJECT via `PART_OF` edges, and MMR (Maximal Marginal Relevance) computes `RELEVANT_TO` edges between related summaries, rebuilding the context that chunking destroyed.

![Summary Node with Context](pictures/summary.png)

### 3. Knowledge Graphs vs. Flat Vectors

Flat vector stores treat every document as an isolated point in embedding space. There's no structure — no way to traverse from a fact to its source, from a question to its competing answers, or from a current truth to the version it replaced.

**Our solution:** A sparse knowledge graph with typed nodes (PROJECT, SUMMARY, QUESTION, ANSWER, PREFERENCE) and typed edges (PART_OF, ANSWERS, SUPERSEDES, CONTEXT_OF, ELABORATES, RELEVANT_TO). Degree caps enforce sparsity — questions hold max 10 answers (lowest confidence evicted), general nodes cap at degree 10, projects are exempt. This gives O(1) traversal per hop at any scale while maintaining full provenance chains. The fact-chain responder walks 2-hop paths and returns "Data missing" instead of hallucinating when the graph is sparse.

![Project with Summaries](pictures/project_summary.png)

### 4. Intent-Based Deep Search

Single-query vector search assumes every question needs the same retrieval strategy. But "What is X?" (reference lookup), "Why did X break?" (debugging), "How has X changed?" (history), and "Compare X vs Y" (comparison) all need fundamentally different search patterns.

**Our solution:** The deep search pipeline first disambiguates intent via LLM — extracting intent type, rephrased query, entities, and temporal scope. Then it dispatches to parallel search strands: EXACT_MATCH (multi-signal BM25+vector+temporal), CONTEXT_WEAVE (2-hop graph traversal via CONTEXT_OF/ELABORATES edges), TEMPORAL_DEEP (SUPERSEDES chain walking), and SEMANTIC_NEAR (pure cosine). Results merge with deduplication. On top of this, query decomposition breaks complex questions into sub-queries searched independently and merged via Reciprocal Rank Fusion.

---

## The Q&A Memory Pattern

Beyond document ingestion, TachyGraph learns from every interaction through a self-reinforcing Q&A memory layer.

**How it works:**

1. Every chat interaction is auto-observed — the LLM extracts question/answer pairs with confidence scores
2. Questions cluster by semantic similarity (cosine > 0.95) into hub nodes, preventing duplicate question proliferation
3. Each question hub holds up to 10 answer slots — when a new answer arrives with higher confidence, the lowest-confidence answer is evicted
4. Answers weave across clusters via `CONTEXT_OF` edges to the 3 most similar answers in *other* clusters, building a semantic web
5. User feedback (correct/wrong/correction) directly adjusts confidence scores — corrections enter at 0.98 confidence, wrong answers expire immediately

![Question Hub](pictures/question.png)

![Answer Node (High Confidence)](pictures/answer1.png)

![Answer Node (Competing)](pictures/answer2.png)

This creates a living memory that gets smarter with use — frequently confirmed facts rise in confidence and auto-extend their validity, while incorrect or outdated answers naturally decay and get evicted.

---

## Prerequisites

- **Docker** and **Docker Compose**
- **Ollama** running locally on your machine ([install](https://ollama.com/download))
- **Cloudflare account** (optional — only for web crawl ingestion)

## 1. Pull Ollama Models

TachyGraph uses two Ollama models — one for text generation (summarization, fact extraction, disambiguation) and one for 1024-dim embeddings:

```bash
# Generation model
ollama pull qwen3-coder:30b

# Embedding model (1024-dim)
ollama pull mxbai-embed-large
```

Verify they're available:

```bash
ollama list
```

## 2. Start the Stack

```bash
cd tachyon
docker compose up -d --build
```

This starts 2 containers:

| Service | Container | What it does |
|---|---|---|
| `tachy_db` | tachygraph_db | PostgreSQL 16 + pgvector (schema auto-init via `init.sql`) |
| `tachy_app` | tachygraph_app | FastAPI application on port 8000 |

Ollama runs on your **host machine** — the app reaches it at `http://host.docker.internal:11434` from inside Docker.

## 3. Verify Everything is Up

```bash
# Container status
docker compose ps

# Health check (DB + FAISS + Ollama)
curl http://localhost:8000/health
```

Expected:
```json
{"db": "ok", "faiss": "unavailable", "faiss_mode": "none"}
```

FAISS shows "unavailable" until you sync vectors — that's normal.

API docs at: **http://localhost:8000/docs**

---

## AI Agents

TachyGraph includes AI agents powered by [Strands SDK](https://github.com/strands-agents/sdk-python) that can reason about what tools to use, iterate on searches, crawl URLs, and store findings — all autonomously.

### Single Agent (all tools, LLM decides)

```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"message": "What do I know about Docker networking?"}'
```

The agent decides whether to search, which tool to call, and when to stop. It has access to 9 tools: memory_search, deep_search, memory_store, memory_observe, web_ingest, factchain, create_reminder, check_reminders, expiry_report.

> **Note:** Agent endpoints can take 1–10 minutes with large models (e.g. qwen3-coder:30b) due to multi-step LLM reasoning + tool calls. Use a smaller model for faster responses: `"model": "llama3.2:3b"`

### Multi-Agent Orchestrator

```bash
curl -X POST http://localhost:8000/agent/orchestrator \
  -H "Content-Type: application/json" \
  -d '{"message": "Research pgvector HNSW indexes, save key points, and remind me to review tomorrow"}'
```

The orchestrator classifies intent and delegates to specialist agents:
- **Researcher** — finds NEW information (search → crawl → ingest → synthesize)
- **Librarian** — recalls EXISTING knowledge (search → factchain → cite sources)
- **Assistant** — performs ACTIONS (reminders, preferences, memory health)

### Research Agent

```bash
curl -X POST http://localhost:8000/agent/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "FAISS IVF-PQ index internals"}'
```

Autonomously: searches memory → finds gaps → crawls relevant URLs → ingests content → searches again → synthesizes → stores key points.

### Memory Maintenance Agent

```bash
curl -X POST http://localhost:8000/agent/maintain
```

Autonomously reviews graph health, reaffirms important expiring facts, compacts duplicates.

---

## Chat (RAG Pipeline)

Simpler and faster than the agent — always searches, always responds. Good for quick Q&A.

```bash
# Standard chat (blocks until response)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What database does LISA use?"}'

# Streaming chat (SSE, token-by-token)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain Docker networking", "stream": true}'

# With specific model
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Quick question", "model": "llama3.2:3b"}'
```

### Chat Feedback

Correct the agent when it's wrong — this is the most powerful learning signal:

```bash
# Mark last response as correct (confidence → 0.95)
curl -X POST http://localhost:8000/chat/feedback \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<session-id>", "feedback": "correct"}'

# Mark as wrong (expires the auto-observed answer)
curl -X POST http://localhost:8000/chat/feedback \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<session-id>", "feedback": "wrong"}'

# Provide correction (new answer at confidence 0.98)
curl -X POST http://localhost:8000/chat/feedback \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<session-id>", "feedback": "correction", "correction": "The actual answer is..."}'
```

### Sessions

Conversations persist in PostgreSQL across container restarts.

```bash
# List recent sessions
curl http://localhost:8000/sessions

# Delete a session
curl -X DELETE http://localhost:8000/sessions/<session-id>
```

---

## Architecture

```
Document → Chunker (8K chars/page) → N pages
                                       │
                ┌──────────────────────┘
                ▼
      ┌─────────────────┐
      │  SUMMARY nodes  │  ← content: full 8K page text (for generation)
      │  (graph citizen) │     summary: concise head sentence (EMBEDDED)
      │                  │     BM25 TF from full content, keywords, temporal date
      └────────┬─────────┘
               │ PART_OF edge
               ▼
      ┌─────────────────┐
      │  PROJECT node   │  ← Scoping, auto-generated summary
      └─────────────────┘
```

### Node Types

| Label | Role | Expiry |
|---|---|---|
| `PROJECT` | Scoping, auto-created on first ingest | Never |
| `SUMMARY` | Graph citizen — embedded, searchable, holds full page content | 365 days |
| `QUESTION` | Q&A hub, clustered by cosine > 0.95 | 5 days |
| `ANSWER` | Linked to QUESTION, 10-slot eviction | 5 days |
| `PREFERENCE` | User preferences (response style, expertise, etc.) | Never |

### Edge Types

| Label | Connects | Purpose |
|---|---|---|
| `PART_OF` | SUMMARY → PROJECT | Project scoping |
| `ANSWERS` | ANSWER → QUESTION | Q&A linking (max 10 per question) |
| `SUPERSEDES` | New → Old | Temporal versioning |
| `CONTEXT_OF` | ANSWER → ANSWER | Cross-cluster weaving |
| `ELABORATES` | Node → Node | Supporting detail |
| `RELEVANT_TO` | SUMMARY → SUMMARY | MMR-computed edges |

---

## Knowledge Graph Visualization

TachyGraph includes an interactive 3D knowledge graph viewer built with [3d-force-graph](https://github.com/vasturiano/3d-force-graph).

Open in your browser: **http://localhost:8000/ui/graph.html**

Or start from the landing page: **http://localhost:8000/ui/**

Features:
- 3D force-directed layout with color-coded node types (Project, Summary, Question, Answer)
- Similarity edges (cosine threshold slider) and explicit edges (PART_OF, ANSWERS, etc.)
- Click a node to isolate its neighborhood, hover for summary tooltip
- Project selector dropdown to scope the view
- Q&A filter toggle to show only Question/Answer clusters

The graph fetches data from the `GET /graph` API endpoint.

![Knowledge Graph — Full View](pictures/broad_view.png)

---

## Ingesting Data

`project_id` is **optional** on all ingestion endpoints. If omitted, a UUID is auto-generated and returned in the response.

### Option A: Send Text via HTTP

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "# Chapter 1\n\nLISA was created in 2020...",
    "source_url": "https://github.com/my-org/docs",
    "project_name": "my-project"
  }'
```

### Option B: Drop Files Locally

Drop `.txt`, `.md`, `.rst`, `.html`, `.json`, `.yaml`, `.yml`, `.pdf`, `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, or `.webm` files into the `ingest/` folder:

```bash
cp my-document.pdf tachyon/ingest/

# Ingest one file
curl -X POST http://localhost:8000/ingest/local \
  -H "Content-Type: application/json" \
  -d '{"filename": "my-document.pdf", "project_name": "my-project"}'

# Or ingest everything at once
curl -X POST http://localhost:8000/ingest/local/all \
  -H "Content-Type: application/json" \
  -d '{"project_name": "my-project"}'
```

### Option C: Crawl a Website

Works with or without Cloudflare. If `CF_ACCOUNT_ID` is not set, uses a local scraper (httpx + BeautifulSoup).

```bash
curl -X POST http://localhost:8000/ingest/web \
  -H "Content-Type: application/json" \
  -d '{"url": "https://docs.example.com", "limit": 100, "depth": 5}'
```

### Option D: Observe Q&A Interactions (Memory Layer)

```bash
curl -X POST http://localhost:8000/observe \
  -H "Content-Type: application/json" \
  -d '{"interaction_text": "Q: What database does LISA use? A: PostgreSQL 16 with pgvector."}'
```

### Option E: Import AI Conversations (ChatGPT, Claude, Gemini)

```bash
curl -X POST http://localhost:8000/ingest/chat \
  -H "Content-Type: application/json" \
  -d '{"json_data": <conversations.json>, "project_name": "my-chatgpt-knowledge"}'
```

---

## MCP Server (IDE Integration)

TachyGraph exposes a Model Context Protocol server with 10 tools for Claude Desktop, VS Code, Cursor, or any MCP client.

**Run:** `python -m tachyrag.mcp.server`

**Claude Desktop config** (`~/.claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "tachygraph": {
      "command": "python",
      "args": ["-m", "tachyrag.mcp.server"],
      "cwd": "/path/to/tachyon/app"
    }
  }
}
```

**Tools:** memory_search, memory_store, memory_observe, memory_recall, memory_chat (agent-powered), memory_tasks, memory_preferences, memory_feedback, memory_projects, memory_status

---

## Search Tuning

TachyGraph's search combines three signals with configurable weights. You can set any signal to `1.0` for pure single-signal search, or blend them.

### Signal Weights

Set via environment variables (must sum to 1.0):

| Variable | Default | Description |
|---|---|---|
| `SEARCH_BM25_WEIGHT` | `0.1` | Keyword matching (BM25 with TF saturation + length normalization) |
| `SEARCH_VECTOR_WEIGHT` | `0.6` | Semantic similarity (cosine via HNSW) |
| `SEARCH_TEMPORAL_WEIGHT` | `0.3` | Recency bias (exponential decay, 30-day fact = 0.05) |

Example configurations:

```yaml
# Pure vector search (semantic only)
SEARCH_BM25_WEIGHT: "0.0"
SEARCH_VECTOR_WEIGHT: "1.0"
SEARCH_TEMPORAL_WEIGHT: "0.0"

# Pure keyword search (BM25 only)
SEARCH_BM25_WEIGHT: "1.0"
SEARCH_VECTOR_WEIGHT: "0.0"
SEARCH_TEMPORAL_WEIGHT: "0.0"

# Recency-first (prioritize recent facts)
SEARCH_BM25_WEIGHT: "0.1"
SEARCH_VECTOR_WEIGHT: "0.3"
SEARCH_TEMPORAL_WEIGHT: "0.6"

# Balanced (default)
SEARCH_BM25_WEIGHT: "0.1"
SEARCH_VECTOR_WEIGHT: "0.6"
SEARCH_TEMPORAL_WEIGHT: "0.3"
```

### RAG Techniques (toggleable)

| Variable | Default | Description |
|---|---|---|
| `SEARCH_USE_HYDE` | `false` | **HyDE** — Generate a hypothetical answer, embed that instead of the raw query. Improves vocabulary matching but adds ~200ms and can hallucinate wrong direction. Best for vague queries, bad for precise technical terms. |
| `SEARCH_USE_DECOMPOSITION` | `true` | **Query Decomposition** — Break complex queries ("compare X vs Y") into focused sub-queries, search each independently, merge via Reciprocal Rank Fusion. Zero cost for simple queries. |
| `SEARCH_USE_COMPRESSION` | `true` | **Contextual Compression** — After retrieval, extract only the relevant sentences from each 8K page. Reduces context from ~80K tokens to ~5K. Parallel via ThreadPoolExecutor. |

### Full /chat RAG Pipeline

```
User message
  → Adaptive K (skip search for greetings, adjust for complexity)
  → HyDE (optional: hypothetical answer → embed)
  → Query decomposition (optional: complex → sub-queries)
  → Two-stage search per sub-query (HNSW cosine → BM25+vector+temporal re-rank)
  → Reciprocal Rank Fusion (if decomposed)
  → Smart reranker (diversity + dedup, max 3 per source)
  → Contextual compression (optional: 8K → relevant sentences)
  → Preference-aware prompt building
  → Ollama generate (streaming SSE)
  → Self-learning (auto-observe Q&A into graph)
  → Access-count reaffirm (hot facts stay alive)
```

---

## All API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | DB + FAISS + Ollama status |
| GET | `/projects` | List all projects with node counts |
| GET | `/graph` | Graph data for 3D visualization |
| GET | `/models` | List available Ollama models |
| POST | `/agent` | AI agent — LLM decides which tools to call |
| POST | `/agent/orchestrator` | Multi-agent — delegates to specialist agents |
| POST | `/agent/research` | Research agent — autonomous topic research |
| POST | `/agent/maintain` | Memory manager — autonomous graph maintenance |
| POST | `/chat` | RAG chat (simpler, faster, supports streaming) |
| POST | `/chat/feedback` | Correct/confirm last chat response |
| GET | `/sessions` | List recent chat sessions |
| DELETE | `/sessions/{id}` | Delete a session |
| POST | `/ingest` | Ingest text via HTTP |
| POST | `/ingest/web` | Crawl a website and ingest all pages |
| POST | `/ingest/chat` | Import ChatGPT / Claude / Gemini conversations |
| GET | `/ingest/local/scan` | List files pending in `ingest/` folder |
| POST | `/ingest/local` | Ingest one local file by name |
| POST | `/ingest/local/all` | Ingest all pending local files |
| POST | `/ingest/local/auto` | Auto-ingest: scan + ingest all pending files |
| POST | `/observe` | Observe Q&A interaction (memory layer) |
| POST | `/search` | Multi-signal pgvector search (two-stage) |
| POST | `/search/fast` | FAISS-only search |
| POST | `/search/hybrid` | FAISS recall → pgvector re-rank |
| POST | `/search/deep` | Ollama disambiguation → parallel strands |
| POST | `/search/factchain` | Provenance-backed fact chain |
| POST | `/resolve` | Temporal conflict resolution |
| POST | `/temporal/reaffirm` | Extend node validity |
| POST | `/temporal/expiring` | List facts expiring in 24h |
| POST | `/faiss/sync` | Sync pgvector → FAISS index |
| POST | `/mmr/recompute` | Recompute MMR edges (one project or all) |
| POST | `/compact` | Deduplicate near-identical nodes |
| GET | `/expiry/report` | Expiry dashboard (24h, 7d, hot facts) |
| POST | `/export` | Export knowledge graph (JSON or markdown) |
| POST | `/import` | Import from backup |
| POST | `/maintenance` | Run all maintenance tasks |
| GET | `/preferences` | View user preferences |
| POST | `/preferences` | Set user preferences |
| POST | `/tasks` | Create a reminder/task |
| GET | `/tasks` | List all tasks |
| GET | `/tasks/due` | List tasks due within 24h |
| POST | `/tasks/complete` | Mark a task as completed |

---

## Project Structure

```
tachyon/
├── app/
│   ├── tachyrag/
│   │   ├── config.py                # All constants + Ollama + Cloudflare config
│   │   ├── main.py                  # FastAPI app + 41 endpoints
│   │   ├── agents/                  # AI agent layer (Strands SDK)
│   │   │   ├── core.py              # Single agent with all tools
│   │   │   ├── tools.py             # 9 @tool definitions wrapping TachyGraph
│   │   │   ├── orchestrator.py      # Multi-agent orchestrator
│   │   │   ├── specialists.py       # Researcher, Librarian, Assistant agents
│   │   │   ├── researcher.py        # Standalone research workflow
│   │   │   └── memory_manager.py    # Autonomous graph maintenance agent
│   │   ├── chat/                    # Conversational interface
│   │   │   ├── chat.py              # RAG chat + streaming SSE
│   │   │   ├── session.py           # Persistent DB-backed sessions
│   │   │   └── feedback.py          # User corrections + learning
│   │   ├── core/                    # Shared infrastructure
│   │   │   ├── db.py                # PostgreSQL connection pool + CRUD
│   │   │   ├── llm_client.py        # Multi-provider LLM client (Ollama, OpenAI, Claude, Gemini)
│   │   │   ├── embedder.py          # Embedding + LRU cache + batch
│   │   │   ├── tfidf.py             # BM25 tokenizer (pure Python)
│   │   │   ├── export.py            # JSON/markdown export + import
│   │   │   ├── middleware.py        # Request logging
│   │   │   └── webhooks.py          # HTTP webhook notifications
│   │   ├── ingest/                  # All ingestion pipelines
│   │   │   ├── chunker.py           # Markdown + plaintext splitter (8K pages)
│   │   │   ├── summarizer.py        # Parallel LLM extraction
│   │   │   ├── ingestor.py          # Stream A: document pipeline (batch)
│   │   │   ├── file_ingestor.py     # Local file drop (txt, pdf, audio)
│   │   │   ├── web_crawler.py       # Cloudflare + local scraper
│   │   │   ├── chat_parser.py       # ChatGPT / Claude / Gemini parser
│   │   │   └── audio_ingestor.py    # Whisper transcription
│   │   ├── memory/                  # Q&A memory layer (Stream B)
│   │   │   ├── observer.py          # Fact extraction via Ollama
│   │   │   ├── clustering.py        # Question hub clustering (sim > 0.95)
│   │   │   ├── memory.py            # Answer insertion + 10-slot eviction
│   │   │   └── weaver.py            # CONTEXT_OF cross-cluster weaving
│   │   ├── search/                  # All search modes
│   │   │   ├── search.py            # Two-stage multi-signal + deep search
│   │   │   ├── search_hybrid.py     # FAISS fast + hybrid
│   │   │   ├── strands.py           # Parallel search strategies
│   │   │   ├── disambiguator.py     # Ollama intent extraction
│   │   │   ├── responder.py         # Fact-chain builder
│   │   │   ├── reranker.py          # Smart context selection for chat
│   │   │   ├── hyde.py              # HyDE: hypothetical document embeddings
│   │   │   ├── decomposer.py        # Query decomposition into sub-queries
│   │   │   ├── compressor.py        # Contextual compression (8K → relevant sentences)
│   │   │   └── rrf.py               # Reciprocal Rank Fusion for merging ranked lists
│   │   ├── graph/                   # Graph maintenance
│   │   │   ├── temporal.py          # Conflict resolution + auto-reaffirm
│   │   │   ├── mmr.py               # MMR edge linking + recompute
│   │   │   ├── compaction.py        # Dedup + expiry reports
│   │   │   ├── preferences.py       # User preference nodes
│   │   │   ├── tasks.py             # Task/reminder system
│   │   │   └── maintenance.py       # Scheduled cleanup cron
│   │   ├── faiss/                   # FAISS vector index
│   │   │   ├── vector_index.py      # Unified CPU/GPU FAISS
│   │   │   └── sync_faiss.py        # Async pgvector → FAISS sync
│   │   └── mcp/                     # MCP server (10 tools)
│   │       └── server.py            # stdio transport for IDE integration
│   ├── Dockerfile
│   └── requirements.txt
├── ui/
│   └── graph.html                   # 3D knowledge graph visualization
├── config/
│   └── postgresql.conf              # PostgreSQL tuning
├── sql/
│   └── init.sql                     # Unified schema
├── docs/                            # Design documents
├── ingest/                          # Drop files here for local ingestion
├── docker-compose.yaml
└── README.md
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://tachy_admin:tachy_password_2026@localhost:5432/tachygraph` | PostgreSQL connection |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API |
| `OLLAMA_MODEL` | `qwen3-coder:30b` | Generation model |
| `OLLAMA_EMBED_MODEL` | `mxbai-embed-large` | Embedding model (1024-dim) |
| `FAISS_USE_GPU` | `false` | Set `true` for GPU FAISS |
| `FAISS_NLIST` | `64` | FAISS IVF clusters |
| `INGEST_DIR` | `/data/ingest` | Local file drop directory |
| `CF_ACCOUNT_ID` | *(empty)* | Cloudflare account ID (for web crawl) |
| `CF_API_TOKEN` | *(empty)* | Cloudflare API token |
| `LLM_PROVIDER` | `ollama` | LLM for generation: `ollama`, `openai`, `anthropic`, `gemini` |
| `EMBED_PROVIDER` | *(empty)* | Embedding provider (defaults to LLM_PROVIDER): `ollama`, `openai`, `gemini` |
| `OPENAI_API_KEY` | *(empty)* | OpenAI API key (if LLM_PROVIDER=openai) |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI generation model |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `ANTHROPIC_API_KEY` | *(empty)* | Anthropic API key (if LLM_PROVIDER=anthropic) |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Claude model |
| `GEMINI_API_KEY` | *(empty)* | Google Gemini API key (if LLM_PROVIDER=gemini) |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini generation model |
| `SEARCH_BM25_WEIGHT` | `0.1` | BM25 keyword signal weight |
| `SEARCH_VECTOR_WEIGHT` | `0.6` | Cosine similarity signal weight |
| `SEARCH_TEMPORAL_WEIGHT` | `0.3` | Temporal decay signal weight |
| `SEARCH_USE_HYDE` | `false` | HyDE: hypothetical document embeddings |
| `SEARCH_USE_DECOMPOSITION` | `true` | Query decomposition for complex queries |
| `SEARCH_USE_COMPRESSION` | `true` | Contextual compression (8K → relevant sentences) |
| `QA_CLUSTER_GLOBAL` | `true` | Question clustering ignores project boundaries (cross-project Q&A hubs) |
| `QA_SKIP_LLM_EXTRACT` | `true` | Skip LLM for Q:/A: formatted input — regex parse instead (~13x faster) |
| `WEBHOOK_URL` | *(empty)* | HTTP webhook for notifications |
| `MMR_ON_INGEST` | `false` | Auto-link SUMMARY nodes via MMR at ingest time |
| `MMR_CRON_ENABLED` | `false` | Enable background cron (MMR + maintenance) |

---

## Stopping

```bash
docker compose down        # stop containers, keep data
docker compose down -v     # stop containers AND delete all data
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| Strands SDK for agents | Lightweight, native tool-calling, Ollama support. Not LangChain (200MB bloat) |
| Agent + RAG dual endpoints | `/agent` for complex reasoning, `/chat` for fast Q&A. User chooses |
| SUMMARY nodes hold full content | `content` = full 8K page, `summary` = head sentence (embedded). No CHUNK nodes |
| BM25 from full content | Real term frequencies, not LLM keywords. Proper TF saturation + length normalization |
| Two-stage search | Cosine pre-filter via HNSW → BM25 re-rank on candidates only |
| Batch embeddings + inserts | Single Ollama call + single DB transaction per document |
| Persistent sessions in PostgreSQL | Survive container restarts, shareable across instances |
| Self-learning loop | Every chat auto-observes Q&A into graph. Feedback loop for corrections |
| Smart reranker | Diversity + dedup before context injection. No 5 facts from same source |
| Local web scraper fallback | Works without Cloudflare. httpx + BeautifulSoup, respects robots.txt |
| 10-slot Q&A eviction | Each question hub holds up to 10 answers. Lowest confidence evicted |
| Access-count auto-reaffirm | Frequently retrieved facts auto-extend validity. Hot facts don't expire |
| Multi-provider LLM | `LLM_PROVIDER` switches between Ollama (local), OpenAI, Claude, Gemini. `EMBED_PROVIDER` can differ — e.g. Claude for generation + OpenAI for embeddings. Pure httpx, no SDK deps |
| HyDE off by default | Adds ~200ms LLM call per search. Helps with vocabulary mismatch but can hallucinate wrong search direction for precise queries. Enable with `SEARCH_USE_HYDE=true` when queries are vague |
| Configurable signal weights | `SEARCH_BM25_WEIGHT`, `SEARCH_VECTOR_WEIGHT`, `SEARCH_TEMPORAL_WEIGHT` — set any to 1.0 for pure single-signal search. No hardcoded weights anywhere |
| Query decomposition on by default | Zero cost for simple queries (heuristic skips them). Only triggers for comparisons and multi-part questions |
| Contextual compression on by default | Reduces 8K pages to relevant sentences before context injection. Parallel extraction via ThreadPoolExecutor |
| MCP server with 10 tools | IDE integration — Claude Desktop, Cursor, VS Code can use your memory |
