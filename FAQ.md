# TachyGraph FAQ

> A local-first knowledge agent with sparse graph memory, multi-signal retrieval, AI agents, and graph-aware search built on PostgreSQL, pgvector, FAISS, and Ollama.[1][2][3][4]

***

## Table of Contents

1. [What does TachyGraph solve?](#1-what-does-tachygraph-solve)
2. [How does TachyGraph handle temporal conflicts?](#2-how-does-tachygraph-handle-temporal-conflicts)
3. [How does TachyGraph solve the chunking problem?](#3-how-does-tachygraph-solve-the-chunking-problem)
4. [Why not rely only on flat vector search?](#4-why-not-rely-only-on-flat-vector-search)
5. [Why use intent-based search?](#5-why-use-intent-based-search)
6. [Can this become a shared memory graph for a team or enterprise?](#6-can-this-become-a-shared-memory-graph-for-a-team-or-enterprise)
7. [Why is the architecture more complex than a standard RAG app?](#7-why-is-the-architecture-more-complex-than-a-standard-rag-app)
8. [How is the system organized internally?](#8-how-is-the-system-organized-internally)
9. [Why PostgreSQL + pgvector instead of Neo4j?](#9-why-postgresql--pgvector-instead-of-neo4j)
10. [How does TachyGraph compare on cost?](#10-how-does-tachygraph-compare-on-cost)
11. [How accurate is the system?](#11-how-accurate-is-the-system)
12. [What are the main advantages?](#12-what-are-the-main-advantages)
13. [What are the current limitations?](#13-what-are-the-current-limitations)
14. [How does TachyGraph perform at scale?](#14-how-does-tachygraph-perform-at-scale)
15. [How secure is it right now?](#15-how-secure-is-it-right-now)
16. [How does it compare with other tools?](#16-how-does-it-compare-with-other-tools)
17. [Who is it for?](#17-who-is-it-for)
18. [What should be rewritten before a formal submission?](#18-what-should-be-rewritten-before-a-formal-submission)
19. [Quick configuration reference](#19-quick-configuration-reference)

***

## 1. What does TachyGraph solve?

### Q: What does TachyGraph solve?

**A:** TachyGraph is designed to solve four recurring problems in knowledge systems: **temporal conflicts, context loss from chunking, flat-vector retrieval limits, and one-size-fits-all search**.[3][4]

Most retrieval systems treat knowledge as static text broken into chunks and searched with a single retrieval strategy.[3][4] That works for simple lookups, but it struggles when facts change over time, when chunks lose surrounding context, when related ideas need explicit structure, or when different question types need different search behavior.[3][4]

TachyGraph addresses those failures with time-aware nodes, structure-preserving summary pages, a sparse typed graph on top of vector search, and intent-driven search paths that adapt to the kind of question being asked.[2][3][4]

***

## 2. How does TachyGraph handle temporal conflicts?

### Q: How does TachyGraph handle temporal conflicts?

**A:** Temporal conflict happens when multiple versions of a fact are all retrievable, but only one is correct now.[3][4] A flat store cannot easily distinguish whether a matching answer is current, outdated, or superseded.[3][4]

TachyGraph handles this by attaching temporal fields such as `valid_from` and `valid_until` to nodes, extracting dates during ingestion, and applying temporal decay during ranking.[2][3] It also links conflicting facts with `SUPERSEDES` edges and uses recency and confidence to resolve which answer should dominate.[2][3][4]

In practice, this means a current architecture note can outrank a stale migration doc even when both are semantically similar.[3] Frequently used facts can also be reaffirmed so important knowledge does not expire too aggressively.[2][3]

***

## 3. How does TachyGraph solve the chunking problem?

### Q: How does TachyGraph solve the chunking problem?

**A:** Standard chunking often breaks documents into arbitrary slices that lose headings, hierarchy, and local meaning.[3][4] A sentence like “it uses PostgreSQL” becomes much less useful when the surrounding section title and neighboring explanation are gone.[3]

TachyGraph preserves more context by splitting documents into larger 8K-character pages, extracting structured metadata for each page, and storing them as SUMMARY nodes that remain connected to their project and to related summaries.[1][2][3] Each summary node keeps full page content for generation while also storing an embed-friendly summary, keywords, and temporal metadata for retrieval.[3]

That approach keeps the system closer to document structure instead of blind chunk boundaries, so retrieval remains contextual rather than purely fragment-based.[2][3][4]

***

## 4. Why not rely only on flat vector search?

### Q: Why not rely only on flat vector search?

**A:** Vector search is useful for semantic similarity, but by itself it treats each item as an isolated point in embedding space.[3][4] That makes it harder to represent provenance, question-answer competition, source relationships, and temporal versioning.[3]

TachyGraph adds a sparse typed graph with nodes such as PROJECT, SUMMARY, QUESTION, ANSWER, and PREFERENCE, plus typed edges such as PART_OF, ANSWERS, SUPERSEDES, CONTEXT_OF, and RELEVANT_TO.[1][2][3] This lets the system traverse knowledge, not just retrieve the nearest embedding.[3][4]

The graph layer also enables features like fact chains, cross-answer weaving, source scoping, and relationship-aware exploration that pure vector databases do not provide on their own.[2][3]

***

## 5. Why use intent-based search?

### Q: Why use intent-based search?

**A:** Not every question should be answered with the same retrieval strategy.[3][4] A definition, a comparison, a debugging request, and a historical question all need different kinds of evidence.[3]

TachyGraph uses intent disambiguation before deep search, then dispatches to different retrieval strands such as exact match, semantic search, context weaving, and temporal traversal.[2][3] It can also decompose complex questions into sub-queries and merge results with Reciprocal Rank Fusion.[2][3]

This reduces irrelevant results and makes the search behavior better matched to the user’s actual question, not just the text similarity of the query string.[2][3][4]

***

## 6. Can this become a shared memory graph for a team or enterprise?

### Q: Can this become a shared memory graph for a team or enterprise?

**A:** Yes, the architecture can evolve in that direction, but that is **not the current product stage**.[4] Today, TachyGraph is best described as a personal or local-first knowledge system with the foundations for something broader.[3][4]

The current design already includes projects, persistent sessions, tasks, preferences, export/import, web ingestion, and MCP-based IDE access, which are useful building blocks for a shared team memory layer.[1][2][3] A future team-oriented version could add authentication, role-based access control, tenant separation, collaboration workflows, audit trails, and shared governance around what enters or expires from memory.[4]

A good way to present it is this: TachyGraph is currently a personal knowledge graph, but its architecture could be upgraded into a shared memory graph for developer teams or enterprises.[4]

***

## 7. Why is the architecture more complex than a standard RAG app?

### Q: Why is the architecture more complex than a standard RAG app?

**A:** A standard RAG stack can answer many simple questions with chunking, embeddings, reranking, and generation.[4] TachyGraph becomes more complex because it is trying to solve more than retrieval: memory evolution, temporal reasoning, graph structure, interactive learning, and tool-using agents.[2][3][4]

The system separates long-lived document knowledge from short-lived interactive Q&A memory, which allows different retention, clustering, and conflict-handling behavior for each.[3][4] That separation is one of the reasons the codebase is heavier than a “vector DB + prompt” design, but it is also what enables feedback-driven learning and graph-based recall.[2][3]

In short, the complexity is intentional: the design is not just trying to answer a question once, but to build a usable memory system over time.[3][4]

***

## 8. How is the system organized internally?

### Q: How is the system organized internally?

**A:** The codebase follows a modular architecture with separate layers for agents, chat, ingest, memory, search, graph maintenance, FAISS acceleration, and MCP integration.[1][4] The FastAPI app acts as the runtime shell around those modules and exposes the system through a large set of HTTP endpoints.[1][3]

### High-level layout

```text
tachyon/
├── app/tachyrag/
│   ├── main.py
│   ├── config.py
│   ├── agents/
│   ├── chat/
│   ├── core/
│   ├── ingest/
│   ├── memory/
│   ├── search/
│   ├── graph/
│   ├── faiss/
│   └── mcp/
├── sql/init.sql
├── ui/graph.html
└── docker-compose.yaml
```

### Major subsystems

| Subsystem | Purpose |
|---|---|
| Agents | Tool-using AI workflows, including single-agent, orchestrator, researcher, and maintenance flows.[1][2][3] |
| Chat | RAG-style chat, streaming responses, persistent sessions, and user feedback.[1][2][3] |
| Ingest | Text, file, web, chat-export, PDF, and audio ingestion pipelines.[1][2][3] |
| Memory | Q&A observation, clustering, answer insertion, eviction, and cross-cluster weaving.[1][2][3] |
| Search | Multi-signal retrieval, deep search, HyDE, decomposition, reranking, compression, and fact chains.[1][2][3] |
| Graph | Temporal resolution, MMR edges, compaction, preferences, tasks, and maintenance jobs.[1][2][3] |
| FAISS | Fast vector retrieval and sync from pgvector to FAISS indexes.[1][2][3] |
| MCP | IDE-facing tools for Claude Desktop, Cursor, VS Code, and other MCP clients.[1][2][3] |

This structure keeps concerns separated and makes the project easier to evolve than a single-file or tightly coupled implementation.[1][4]

***

## 9. Why PostgreSQL + pgvector instead of Neo4j?

### Q: Why PostgreSQL + pgvector instead of Neo4j?

**A:** PostgreSQL + pgvector was chosen to keep structured metadata, graph relationships, sessions, tasks, text statistics, and vector retrieval in one operational stack.[1][3][4] That avoids maintaining separate graph, vector, and relational systems for a still-evolving project.[4]

TachyGraph also uses a sparse graph with degree caps, which reduces the need for a dedicated graph engine for the current workload.[2][3][4] Because traversal is intentionally shallow and bounded, the design can stay performant while benefiting from the maturity and flexibility of PostgreSQL.[3][4]

Neo4j would still be a reasonable future option for more advanced graph analytics or heavier graph-native workflows.[4] For the current design, though, PostgreSQL + pgvector is a practical unification choice rather than a rejection of graph databases in general.[3][4]

***

## 10. How does TachyGraph compare on cost?

### Q: How does TachyGraph compare on cost?

**A:** Cost efficiency is one of the strongest arguments for this design.[4] When run with Ollama, the system can operate with zero API-token cost because generation and embeddings stay local.[3][4]

It also reduces waste through batch embeddings, batch inserts, contextual compression, adaptive retrieval, and bounded Q&A memory growth.[2][4] That makes it more cost-conscious than systems that repeatedly re-read large histories or send oversized contexts to paid APIs.[4]

### Cost model snapshot

| Approach | Cost profile | Privacy profile | Notes |
|---|---|---|---|
| TachyGraph with Ollama | Near-zero API cost.[4] | Fully local by default.[3][4] | Slower than premium hosted APIs during ingestion.[4] |
| TachyGraph with API providers | Pay-per-token.[2][4] | Depends on provider.[4] | Faster and more scalable for production-like workloads.[4] |
| SaaS knowledge tools | Subscription and/or token spend.[4] | Usually cloud-based.[4] | Easier to start, but less controllable.[4] |

***

## 11. How accurate is the system?

### Q: How accurate is the system?

**A:** Accuracy depends on model quality, retrieval weights, source quality, and how well the graph is maintained.[3][4] TachyGraph tries to improve accuracy by combining vector similarity, BM25 keyword matching, and temporal scoring instead of trusting a single signal.[2][3][4]

It also adds reranking, optional query decomposition, optional HyDE, contextual compression, and conflict-aware graph traversal, all of which help retrieval quality before generation begins.[2][3] That said, it is still a retrieval-and-generation system, so quality depends heavily on ingestion fidelity and model behavior.[4]

### Default search signal weights

| Signal | Default role |
|---|---|
| Vector similarity | Main semantic retrieval signal.[3][4] |
| BM25 | Exact term and keyword precision.[2][3][4] |
| Temporal decay | Recency bias for changing facts.[2][3][4] |

The weights are configurable, which matters because technical lookup, historical recall, and general semantic Q&A often need different blends.[2][3][4]

***

## 12. What are the main advantages?

### Q: What are the main advantages of TachyGraph?

**A:** TachyGraph stands out because it combines privacy, memory, graph structure, and retrieval flexibility in one system.[2][3][4] It is not just a chatbot on top of documents; it is trying to behave like a memory engine.[3][4]

### Key strengths

| Strength | Why it matters |
|---|---|
| Temporal awareness | Helps the system distinguish current facts from stale ones.[2][3][4] |
| Structure-preserving ingestion | Reduces context loss compared with naive chunking.[2][3][4] |
| Sparse graph memory | Adds traversable relationships, provenance, and question-answer competition.[2][3][4] |
| Multi-signal retrieval | Blends keyword, semantic, and temporal search.[2][3][4] |
| Intent-aware deep search | Matches retrieval strategy to question type.[2][3][4] |
| Self-learning feedback loop | Lets the system improve from corrections and confirmations.[2][3][4] |
| Local-first operation | Keeps privacy and cost under user control.[3][4] |
| MCP integration | Makes the graph accessible from IDE workflows.[1][2][3] |

***

## 13. What are the current limitations?

### Q: What are the current limitations?

**A:** The current system is ambitious, but it is still early-stage and has clear trade-offs.[4] It should be described honestly as strong in ideas and architecture, but not yet a finished enterprise product.[4]

### Main limitations

| Limitation | Why it matters | Current mitigation |
|---|---|---|
| Slow ingestion with local models | Summarization and embedding can be slow on Ollama.[4] | Swap to hosted providers for bulk workloads.[2][4] |
| Basic temporal resolution | The current temporal mechanism is practical, but not deeply reasoned.[4] | Temporal decay, date extraction, and `SUPERSEDES` edges.[2][3][4] |
| No built-in auth | Not safe for open internet exposure.[4] | Put behind a reverse proxy and auth layer.[4] |
| No multi-tenant governance | Shared-team usage is not productized yet.[4] | Future roadmap area.[4] |
| Auto-learning can add noise | Bad feedback or weak extraction can pollute memory.[2][4] | Corrections, expiration, confidence handling.[2][3][4] |
| Single-node architecture | Horizontal scale is not a solved story yet.[4] | PostgreSQL + FAISS can still cover meaningful local scale.[2][4] |

Another important limitation is consistency between the document graph and the Q&A graph.[4] They are related subsystems, but they do not fully guarantee automatic reconciliation when the two layers disagree.[4]

***

## 14. How does TachyGraph perform at scale?

### Q: How does TachyGraph perform at scale?

**A:** The project is designed to stay efficient for local and moderately large workloads through sparse edges, two-stage search, HNSW, and optional FAISS acceleration.[2][3][4] It is not yet presented as a distributed large-enterprise platform.[4]

### Reported or documented performance expectations

| Area | Expected behavior |
|---|---|
| FAISS search | Very fast, with CPU and optional GPU modes.[2][4] |
| pgvector HNSW search | Good semantic pre-filtering for the main retrieval pipeline.[2][3][4] |
| Hybrid retrieval | Faster recall via FAISS, then better precision via reranking.[2][3][4] |
| Deep search | Slower than plain retrieval because it includes LLM-based disambiguation and multiple strands.[2][4] |
| Ingestion | Usually the slowest area when local models are used.[3][4] |

The biggest scaling bottlenecks are PostgreSQL as a single write-heavy node, Ollama as a single local model server, and the complexity of keeping vector indexes synced as data grows.[4] Those are reasonable engineering bottlenecks for this stage of the project.[4]

***

## 15. How secure is it right now?

### Q: How secure is it right now?

**A:** TachyGraph is best understood as secure for personal local use, not as production-hardened enterprise software yet.[4] It benefits from local deployment and optional local inference, but it does not yet include the access-control and operational protections expected in a shared enterprise environment.[3][4]

### Current security posture

| Area | Current state |
|---|---|
| Data residency | Strong for local use because data can stay on the same machine.[4] |
| LLM privacy | Strong with Ollama, provider-dependent with external APIs.[2][4] |
| Authentication | Not built in.[4] |
| Encryption in transit | Needs a reverse proxy and TLS for production use.[4] |
| Encryption at rest | Not implemented at the application layer.[4] |
| Input validation | Uses Pydantic validation in the API layer.[4] |

The cleanest way to describe it is: privacy-first by design, but not yet enterprise-secure by default.[4]

***

## 16. How does it compare with other tools?

### Q: How does TachyGraph compare with other knowledge tools?

**A:** TachyGraph is strongest when the user values privacy, local control, custom retrieval behavior, and graph-based memory over convenience or polished collaboration.[4] It is weaker than mainstream SaaS tools when the priority is ease of onboarding, mobile access, and built-in team workflows.[4]

### Positioning snapshot

| Tool type | Strength of TachyGraph | Weakness of TachyGraph |
|---|---|---|
| SaaS note tools | More private and customizable.[4] | Harder setup, less polished collaboration.[4] |
| Obsidian + plugins | More integrated memory logic and graph-aware retrieval.[4] | Less mature plugin ecosystem and UX convenience.[4] |
| Basic LangChain RAG | Richer retrieval and memory model.[4] | More architectural complexity.[4] |
| Enterprise search products | More hackable and local-first.[4] | Lacks enterprise controls, auditability, and admin features today.[4] |

It is best positioned as an experimental but serious knowledge architecture rather than a polished end-user platform.[4]

***

## 17. Who is it for?

### Q: Who is TachyGraph for?

**A:** TachyGraph is well suited to engineers, researchers, and advanced users who want a local-first memory system and are comfortable running Docker, Ollama, and a multi-service stack.[3][4] It is especially appealing to people who care about controllable retrieval, transparent architecture, and owning their knowledge pipeline end to end.[4]

It is less suitable for users who want instant setup, mobile-first use, collaborative editing, or enterprise governance out of the box.[4] In its current form, it is closer to a powerful technical project than a mass-market product.[4]

***

## 18. What should be rewritten before a formal submission?

### Q: What should be rewritten before using this in a formal proposal, portfolio, or submission?

**A:** The documentation should sound more personal, more grounded, and less like a generic AI-generated technical brochure.[4] The current material is strong structurally, but it will be more convincing if it tells the real story of why the project was built and what trade-offs shaped it.[4]

The README and FAQ should emphasize three things more clearly: the original pain points, the architecture decisions made in response, and the honest limitations of the current version.[3][4] A good professional version should sound like a builder explaining choices, not like a tool marketing page.[4]

***

## 19. Quick configuration reference

### Q: What are the most important knobs to tune?

**A:** The project exposes key controls for search behavior, model routing, graph behavior, and acceleration.[2][3][4]

| Variable | What it controls |
|---|---|
| `SEARCH_VECTOR_WEIGHT` | Semantic retrieval strength.[3][4] |
| `SEARCH_BM25_WEIGHT` | Keyword retrieval strength.[2][3][4] |
| `SEARCH_TEMPORAL_WEIGHT` | Recency bias in ranking.[2][3][4] |
| `SEARCH_USE_HYDE` | Whether hypothetical-answer retrieval is enabled.[2][3][4] |
| `SEARCH_USE_DECOMPOSITION` | Whether complex queries are split into sub-queries.[2][3][4] |
| `SEARCH_USE_COMPRESSION` | Whether retrieved pages are compressed to relevant sentences.[2][3][4] |
| `LLM_PROVIDER` | Which generation provider is used.[2][3][4] |
| `EMBED_PROVIDER` | Which embedding provider is used.[2][3][4] |
| `QA_CLUSTER_GLOBAL` | Whether question hubs cluster globally or per project.[2][4] |
| `FAISS_USE_GPU` | Whether FAISS should try GPU acceleration.[2][4] |
| `WEBHOOK_URL` | Where event notifications are sent.[1][2][4] |

***

*Last updated: May 2026.[4]*
