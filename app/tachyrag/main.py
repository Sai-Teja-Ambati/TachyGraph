from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tachyrag.memory.clustering import find_or_create_question
from tachyrag.config import MMR_CRON_ENABLED, MMR_CRON_SCHEDULE
from tachyrag.core.db import pool, get_all_projects
from tachyrag.core.llm_client import check_health as llm_healthy, close as llm_close
from tachyrag.faiss.vector_index import UnifiedVectorIndex, create_vector_index
from tachyrag.ingest.file_ingestor import ingest_all, ingest_auto, ingest_file, scan_pending
from tachyrag.ingest.ingestor import ingest_document
from tachyrag.ingest.web_crawler import crawl_and_collect
from tachyrag.memory.memory import add_answer
from tachyrag.graph.mmr import recompute_all_projects, recompute_project_edges
from tachyrag.memory.observer import observe
from tachyrag.search.responder import build_response, format_response
from tachyrag.search.search import deep_search, search
from tachyrag.search.search_hybrid import search_fast, search_hybrid
from tachyrag.faiss.sync_faiss import FaissSyncService
from tachyrag.graph.temporal import get_expiring_soon, reaffirm_fact, resolve_conflicts
from tachyrag.memory.weaver import weave_answer
from tachyrag.ingest.chat_parser import parse_chat_input
from tachyrag.chat.chat import chat as run_chat, chat_stream as run_chat_stream
from tachyrag.chat.feedback import process_feedback
from tachyrag.chat.session import list_sessions, delete_session
from tachyrag.core.export import export_json, export_markdown, import_json
from tachyrag.core.llm_client import list_models
from tachyrag.core.middleware import RequestLoggingMiddleware
from tachyrag.agents.core import run_agent
from tachyrag.agents.orchestrator import run_orchestrator
from tachyrag.agents.researcher import research as run_research
from tachyrag.agents.memory_manager import run_memory_maintenance
from tachyrag.graph.maintenance import run_maintenance
from tachyrag.graph.compaction import compact_project, get_expiry_report
from tachyrag.graph.preferences import get_preferences, set_preferences
from tachyrag.graph.tasks import create_task, get_due_tasks, complete_task, get_all_tasks

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

faiss_index: UnifiedVectorIndex | None = None
sync_service: FaissSyncService | None = None
mmr_scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global faiss_index, sync_service

    log.info("Checking LLM connection...")
    if llm_healthy():
        log.info("LLM provider is reachable.")
    else:
        log.warning("LLM provider not reachable — generate/embed calls will fail until it's up.")

    log.info("Verifying DB pool...")
    with pool.connection() as conn:
        conn.execute("SELECT 1")

    log.info("Initializing FAISS vector index...")
    try:
        faiss_index = create_vector_index()
        faiss_index.load()
        sync_service = FaissSyncService(faiss_index)
        sync_service.start_background()
        log.info("FAISS ready: %s", faiss_index)
    except Exception as e:
        log.warning("FAISS init failed, running without FAISS search: %s", e)
        faiss_index = None
        sync_service = None

    log.info("TachyGraph v3.1 ready.")

    # MMR cron scheduler
    global mmr_scheduler
    if MMR_CRON_ENABLED:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger
            mmr_scheduler = BackgroundScheduler()
            mmr_scheduler.add_job(recompute_all_projects, CronTrigger.from_crontab(MMR_CRON_SCHEDULE), id="mmr_recompute")
            mmr_scheduler.add_job(run_maintenance, "interval", hours=6, id="maintenance")
            mmr_scheduler.start()
            log.info("MMR cron enabled: %s", MMR_CRON_SCHEDULE)
            log.info("Maintenance cron enabled: every 6 hours")
        except Exception as e:
            log.warning("Cron init failed: %s", e)
            mmr_scheduler = None
    else:
        log.info("Cron jobs disabled.")

    yield

    if mmr_scheduler:
        mmr_scheduler.shutdown(wait=False)
    if sync_service:
        sync_service.stop()
    if faiss_index:
        try:
            faiss_index.save()
        except Exception:
            pass
    pool.close()
    llm_close()


app = FastAPI(title="TachyGraph", version="3.1", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RequestLoggingMiddleware)

import os
_ui_dir = os.environ.get("UI_DIR", "/data/ui")
if os.path.isdir(_ui_dir):
    app.mount("/ui", StaticFiles(directory=_ui_dir, html=True), name="ui")


# --- Request/Response models ---

def _auto_uuid(v: uuid.UUID | None) -> uuid.UUID:
    return v if v is not None else uuid.uuid4()


class IngestRequest(BaseModel):
    text: str
    source_url: str
    project_id: uuid.UUID | None = None
    project_name: str = "default"


class ObserveRequest(BaseModel):
    interaction_text: str
    project_id: uuid.UUID | None = None
    provenance: dict = Field(default_factory=lambda: {"source_url": "user_interaction"})


class SearchRequest(BaseModel):
    query: str
    project_id: uuid.UUID | None = None
    k: int = 21
    bm25_weight: float | None = None
    vector_weight: float | None = None
    temporal_weight: float | None = None


class DeepSearchRequest(BaseModel):
    query: str
    project_id: uuid.UUID | None = None
    k: int = 21
    context: dict | None = None


class FastSearchRequest(BaseModel):
    query: str
    project_id: uuid.UUID | None = None
    k: int = 21
    nprobe: int = 64


class ResolveRequest(BaseModel):
    project_id: uuid.UUID


class ReaffirmRequest(BaseModel):
    node_id: uuid.UUID
    extension_days: int = 5


class FaissSyncRequest(BaseModel):
    project_id: uuid.UUID | None = None


class LocalIngestRequest(BaseModel):
    filename: str
    project_id: uuid.UUID | None = None
    project_name: str = "default"


class LocalIngestAllRequest(BaseModel):
    project_id: uuid.UUID | None = None
    project_name: str = "default"


class MMRRecomputeRequest(BaseModel):
    project_id: uuid.UUID | None = None


class WebCrawlRequest(BaseModel):
    url: str
    project_id: uuid.UUID | None = None
    project_name: str = "default"
    limit: int = 50
    depth: int = 3
    render: bool = False
    include_patterns: list[str] | None = None
    exclude_patterns: list[str] | None = None


class ChatIngestRequest(BaseModel):
    text: str | None = None
    json_data: dict | list | None = None
    project_id: uuid.UUID | None = None
    project_name: str = "default"
    mode: str = "observe"  # "observe" (Q&A memory) or "ingest" (persistent knowledge)


class ChatRequest(BaseModel):
    message: str
    project_id: uuid.UUID | None = None
    session_id: uuid.UUID | None = None
    k: int = 10
    auto_observe: bool = True
    model: str | None = None
    stream: bool = False
    bm25_weight: float | None = None
    vector_weight: float | None = None
    temporal_weight: float | None = None


class ChatFeedbackRequest(BaseModel):
    session_id: uuid.UUID
    feedback: str  # "correct", "wrong", "correction"
    correction: str | None = None


class ExportRequest(BaseModel):
    project_id: uuid.UUID | None = None
    format: str = "json"  # "json" or "markdown"


class ImportRequest(BaseModel):
    data: dict
    project_id: uuid.UUID | None = None


class AgentRequest(BaseModel):
    message: str
    session_id: uuid.UUID | None = None
    model: str | None = None


class ResearchRequest(BaseModel):
    topic: str
    session_id: uuid.UUID | None = None
    model: str | None = None


class CompactRequest(BaseModel):
    project_id: uuid.UUID
    similarity_threshold: float = 0.98


class PreferencesRequest(BaseModel):
    project_id: uuid.UUID | None = None
    preferences: dict


class TaskCreateRequest(BaseModel):
    description: str
    due_days: int = 1
    project_id: uuid.UUID | None = None
    related_node_id: uuid.UUID | None = None


class TaskCompleteRequest(BaseModel):
    task_id: uuid.UUID


# --- Endpoints ---

@app.get("/health")
def health():
    with pool.connection() as conn:
        conn.execute("SELECT 1")
    status = {
        "db": "ok",
        "faiss": "ok" if faiss_index and faiss_index.is_trained else "unavailable",
        "faiss_mode": ("GPU" if faiss_index and faiss_index.use_gpu else "CPU") if faiss_index else "none",
    }
    if faiss_index:
        status["faiss_vectors"] = faiss_index.ntotal
    return status


@app.get("/projects")
def list_projects():
    projects = get_all_projects()
    return {
        "count": len(projects),
        "projects": [
            {
                "id": str(p["id"]),
                "name": p["name"],
                "summary": p.get("summary"),
                "node_count": p.get("node_count", 0),
            }
            for p in projects
        ],
    }


@app.post("/ingest")
def ingest(req: IngestRequest):
    pid = _auto_uuid(req.project_id)
    result = ingest_document(
        text=req.text,
        source_url=req.source_url,
        project_id=pid,
        project_name=req.project_name,
    )
    result["project_id"] = str(pid)
    return result


# --- Agent (Strands-powered) ---

@app.post("/agent")
def agent_endpoint(req: AgentRequest):
    """Single agent with all tools — LLM decides what to do."""
    return run_agent(message=req.message, session_id=req.session_id, model=req.model)


@app.post("/agent/orchestrator")
def orchestrator_endpoint(req: AgentRequest):
    """Multi-agent orchestrator — delegates to specialist agents."""
    return run_orchestrator(message=req.message, session_id=req.session_id, model=req.model)


@app.post("/agent/research")
def research_endpoint(req: ResearchRequest):
    """Research agent — autonomously searches, crawls, ingests, synthesizes."""
    return run_research(topic=req.topic, session_id=req.session_id, model=req.model)


@app.post("/agent/maintain")
def agent_maintain_endpoint():
    """Memory manager agent — autonomously maintains graph health."""
    return run_memory_maintenance()


# --- Chat (RAG pipeline — simpler, faster) ---

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    if req.stream:
        return StreamingResponse(
            run_chat_stream(
                message=req.message, project_id=req.project_id,
                session_id=req.session_id, k=req.k,
                auto_observe=req.auto_observe, model=req.model,
                bm25_weight=req.bm25_weight, vector_weight=req.vector_weight, temporal_weight=req.temporal_weight,
            ),
            media_type="text/event-stream",
        )
    return run_chat(
        message=req.message, project_id=req.project_id,
        session_id=req.session_id, k=req.k,
        auto_observe=req.auto_observe, model=req.model,
        bm25_weight=req.bm25_weight, vector_weight=req.vector_weight, temporal_weight=req.temporal_weight,
    )


@app.post("/chat/feedback")
def chat_feedback_endpoint(req: ChatFeedbackRequest):
    return process_feedback(req.session_id, req.feedback, req.correction)


@app.get("/sessions")
def list_sessions_endpoint(limit: int = 20):
    return {"sessions": list_sessions(limit)}


@app.delete("/sessions/{session_id}")
def delete_session_endpoint(session_id: uuid.UUID):
    if delete_session(session_id):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/models")
def list_models_endpoint():
    models = list_models()
    return {"models": [{"name": m.get("name", ""), "size": m.get("size", 0)} for m in models]}


@app.post("/export")
def export_endpoint(req: ExportRequest):
    if req.format == "markdown":
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(export_markdown(req.project_id), media_type="text/markdown")
    return export_json(req.project_id)


@app.post("/import")
def import_endpoint(req: ImportRequest):
    return import_json(req.data, req.project_id)


@app.post("/maintenance")
def maintenance_endpoint():
    return run_maintenance()


@app.post("/observe")
def observe_interaction(req: ObserveRequest):
    pid = _auto_uuid(req.project_id)
    fact = observe(req.interaction_text)
    if not fact:
        return {"status": "skipped", "reason": "no explicit fact extracted"}

    question_id, is_new = find_or_create_question(fact.question, pid)

    answer_id = add_answer(
        question_id=question_id,
        answer_text=fact.answer,
        confidence=fact.confidence,
        provenance=req.provenance,
        project_id=pid,
    )

    if not answer_id:
        return {"status": "rejected", "reason": "confidence too low for eviction"}

    edges_created = weave_answer(answer_id)

    return {
        "status": "stored",
        "project_id": str(pid),
        "question_id": str(question_id),
        "question_is_new": is_new,
        "answer_id": str(answer_id),
        "weave_edges": edges_created,
    }


@app.post("/search")
def search_endpoint(req: SearchRequest):
    results = search(req.query, req.project_id, k=req.k,
                     bm25_weight=req.bm25_weight, vector_weight=req.vector_weight, temporal_weight=req.temporal_weight)
    if not results:
        raise HTTPException(status_code=404, detail="Data missing")
    return {
        "count": len(results),
        "results": [
            {
                "id": str(r["id"]),
                "label": r["label"],
                "summary": r.get("summary"),
                "content": r["content"][:2000],
                "confidence": r.get("confidence"),
                "rank": r.get("rank"),
                "provenance": r.get("provenance"),
            }
            for r in results
        ],
    }


@app.post("/search/fast")
def fast_search_endpoint(req: FastSearchRequest):
    if not faiss_index or not faiss_index.is_trained:
        raise HTTPException(status_code=503, detail="FAISS index not available")
    results = search_fast(faiss_index, req.query, req.project_id, k=req.k, nprobe=req.nprobe)
    if not results:
        raise HTTPException(status_code=404, detail="Data missing")
    return {
        "count": len(results),
        "search_type": "faiss_gpu" if faiss_index.use_gpu else "faiss_cpu",
        "results": [
            {
                "id": str(r["id"]),
                "label": r["label"],
                "content": r["content"][:500],
                "summary": r.get("summary"),
                "faiss_score": r.get("faiss_score"),
                "provenance": r.get("provenance"),
            }
            for r in results
        ],
    }


@app.post("/search/hybrid")
def hybrid_search_endpoint(req: SearchRequest):
    if not faiss_index or not faiss_index.is_trained:
        raise HTTPException(status_code=503, detail="FAISS index not available")
    results = search_hybrid(faiss_index, req.query, req.project_id, k=req.k)
    if not results:
        raise HTTPException(status_code=404, detail="Data missing")
    return {
        "count": len(results),
        "search_type": "hybrid",
        "results": [
            {
                "id": str(r["id"]),
                "label": r["label"],
                "content": r["content"][:500],
                "summary": r.get("summary"),
                "rank": r.get("rank"),
                "provenance": r.get("provenance"),
            }
            for r in results
        ],
    }


@app.post("/search/deep")
def deep_search_endpoint(req: DeepSearchRequest):
    result = deep_search(req.query, req.project_id, context=req.context, k=req.k)
    if not result["results"]:
        raise HTTPException(status_code=404, detail="Data missing")
    return {
        "intent": result["intent"],
        "count": result["count"],
        "results": [
            {
                "id": str(r["id"]),
                "label": r["label"],
                "content": r["content"][:500],
                "summary": r.get("summary"),
                "strand": r.get("strand"),
                "rank": r.get("rank"),
                "provenance": r.get("provenance"),
            }
            for r in result["results"]
        ],
    }


@app.post("/search/factchain")
def factchain_endpoint(req: SearchRequest):
    chain = build_response(req.query, req.project_id, k=min(req.k, 5))
    return {
        "complete": chain.complete,
        "response": format_response(chain),
        "facts": len(chain.facts),
    }


@app.post("/resolve")
def resolve_endpoint(req: ResolveRequest):
    return {"resolved": resolve_conflicts(req.project_id)}


@app.post("/temporal/reaffirm")
def reaffirm_endpoint(req: ReaffirmRequest):
    reaffirm_fact(req.node_id, req.extension_days)
    return {"status": "extended", "node_id": str(req.node_id), "days": req.extension_days}


@app.post("/temporal/expiring")
def expiring_endpoint(req: ResolveRequest):
    nodes = get_expiring_soon(req.project_id)
    return {
        "count": len(nodes),
        "expiring": [
            {
                "id": str(n["id"]),
                "content": n["content"][:200],
                "summary": n.get("summary"),
                "valid_until": str(n["valid_until"]),
            }
            for n in nodes
        ],
    }


@app.post("/faiss/sync")
def faiss_sync_endpoint(req: FaissSyncRequest):
    if not faiss_index or not sync_service:
        raise HTTPException(status_code=503, detail="FAISS not available")
    count = sync_service.full_sync(req.project_id)
    return {"status": "synced", "vectors": count, "total": faiss_index.ntotal}


@app.post("/mmr/recompute")
def mmr_recompute_endpoint(req: MMRRecomputeRequest):
    if req.project_id:
        result = recompute_project_edges(req.project_id)
        result["project_id"] = str(req.project_id)
        return result
    results = recompute_all_projects()
    return {"projects": len(results), "results": results}


# --- Graph visualization ---

@app.get("/graph")
def get_graph(project_id: uuid.UUID = None, similarity_threshold: float = 0.75):
    """Return nodes and edges for 3D knowledge graph visualization."""
    with pool.connection() as conn, conn.cursor() as cur:
        # Get all nodes
        if project_id:
            cur.execute(
                "SELECT id, label, COALESCE(summary, LEFT(content, 150)) AS summary, created_at FROM nodes WHERE project_id = %s",
                (project_id,),
            )
        else:
            cur.execute("SELECT id, label, COALESCE(summary, LEFT(content, 150)) AS summary, created_at FROM nodes")
        nodes = cur.fetchall()

        # Get all explicit edges
        node_ids = [n["id"] for n in nodes]
        if node_ids:
            cur.execute(
                "SELECT source_id, target_id, label, weight FROM edges WHERE source_id = ANY(%s) OR target_id = ANY(%s)",
                (node_ids, node_ids),
            )
            edges = cur.fetchall()
        else:
            edges = []

        # Similarity edges between SUMMARY nodes
        if project_id:
            cur.execute(
                """
                SELECT s1.id AS source, s2.id AS target,
                       1 - (s1.embedding <=> s2.embedding) AS similarity
                FROM nodes s1, nodes s2
                WHERE s1.label = 'SUMMARY' AND s2.label = 'SUMMARY'
                  AND s1.id < s2.id
                  AND s1.project_id = %s
                  AND 1 - (s1.embedding <=> s2.embedding) > %s
                """,
                (project_id, similarity_threshold),
            )
        else:
            cur.execute(
                """
                SELECT s1.id AS source, s2.id AS target,
                       1 - (s1.embedding <=> s2.embedding) AS similarity
                FROM nodes s1, nodes s2
                WHERE s1.label = 'SUMMARY' AND s2.label = 'SUMMARY'
                  AND s1.id < s2.id
                  AND 1 - (s1.embedding <=> s2.embedding) > %s
                """,
                (similarity_threshold,),
            )
        sim_edges = cur.fetchall()

    return {
        "nodes": [
            {
                "id": str(n["id"]),
                "label": n["label"],
                "summary": n["summary"],
                "created_at": str(n["created_at"]),
            }
            for n in nodes
        ],
        "edges": [
            {
                "source": str(e["source_id"]),
                "target": str(e["target_id"]),
                "label": e["label"],
                "weight": e.get("weight", 1.0),
            }
            for e in edges
        ] + [
            {
                "source": str(e["source"]),
                "target": str(e["target"]),
                "label": "SIMILAR",
                "weight": round(e["similarity"], 3),
            }
            for e in sim_edges
        ],
    }


# --- Local file ingestion ---

@app.get("/ingest/local/scan")
def scan_local_files():
    files = scan_pending()
    return {"count": len(files), "files": files}


@app.post("/ingest/local")
def ingest_local_file(req: LocalIngestRequest):
    pid = _auto_uuid(req.project_id)
    try:
        result = ingest_file(req.filename, pid, req.project_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    result["project_id"] = str(pid)
    return result


@app.post("/ingest/local/all")
def ingest_all_local(req: LocalIngestAllRequest):
    pid = _auto_uuid(req.project_id)
    results = ingest_all(pid, req.project_name)
    return {
        "project_id": str(pid),
        "total": len(results),
        "ingested": sum(1 for r in results if r["status"] == "ingested"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results,
    }


@app.post("/ingest/local/auto")
def ingest_auto_local():
    """Ingest all pending files. Each file becomes its own project (name = filename without extension)."""
    results = ingest_auto()
    return {
        "total": len(results),
        "ingested": sum(1 for r in results if r.get("status") == "ingested"),
        "failed": sum(1 for r in results if r.get("status") == "failed"),
        "results": results,
    }


# --- Web crawl ingestion ---

@app.post("/ingest/web")
def ingest_web(req: WebCrawlRequest):
    pid = _auto_uuid(req.project_id)
    try:
        pages = crawl_and_collect(
            req.url,
            limit=req.limit,
            depth=req.depth,
            render=req.render,
            include_patterns=req.include_patterns,
            exclude_patterns=req.exclude_patterns,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Crawl failed: {e}")

    if not pages:
        raise HTTPException(status_code=404, detail="No content found at URL")

    total_summaries = 0
    page_results = []
    for page in pages:
        result = ingest_document(
            text=page["markdown"],
            source_url=page["url"],
            project_id=pid,
            project_name=req.project_name,
        )
        total_summaries += result.get("summaries", 0)
        page_results.append({"url": page["url"], "title": page["title"], "summaries": result.get("summaries", 0)})

    return {
        "project_id": str(pid),
        "pages_crawled": len(pages),
        "total_summaries": total_summaries,
        "pages": page_results,
    }

# --- Chat conversation ingestion ---

@app.post("/ingest/chat")
def ingest_chat(req: ChatIngestRequest):
    """Ingest ChatGPT, Claude, or Gemini conversations. Auto-detects format."""
    if not req.text and not req.json_data:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'json_data'")

    pairs = parse_chat_input(text=req.text, json_data=req.json_data)
    if not pairs:
        raise HTTPException(status_code=400, detail="No Q&A pairs found. Check format.")

    pid = _auto_uuid(req.project_id)
    from tachyrag.core.db import ensure_project
    ensure_project(pid, req.project_name)

    results = []
    if req.mode == "observe":
        for p in pairs:
            question_id, is_new = find_or_create_question(p.question, pid)
            answer_id = add_answer(
                question_id=question_id,
                answer_text=p.answer,
                confidence=0.90,
                provenance={"source_url": f"{p.source}://{p.conversation_title}"},
                project_id=pid,
            )
            if answer_id:
                weave_answer(answer_id)
                results.append({"question": p.question[:100], "status": "stored", "question_id": str(question_id)})
            else:
                results.append({"question": p.question[:100], "status": "rejected"})
    else:
        for p in pairs:
            text = f"Q: {p.question}\n\nA: {p.answer}"
            r = ingest_document(
                text=text,
                source_url=f"{p.source}://{p.conversation_title}",
                project_id=pid,
                project_name=req.project_name,
            )
            results.append({"question": p.question[:100], "summaries": r.get("summaries", 0)})

    sources = list({p.source for p in pairs})
    conversations = list({p.conversation_title for p in pairs})
    return {
        "project_id": str(pid),
        "source": sources[0] if len(sources) == 1 else sources,
        "conversations": len(conversations),
        "pairs_found": len(pairs),
        "pairs_processed": len(results),
        "mode": req.mode,
        "results": results,
    }


# --- Compaction + Expiry Report ---

@app.post("/compact")
def compact_endpoint(req: CompactRequest):
    return compact_project(req.project_id, req.similarity_threshold)


@app.get("/expiry/report")
def expiry_report(project_id: uuid.UUID = None):
    return get_expiry_report(project_id)


# --- Preferences ---

@app.get("/preferences")
def get_prefs(project_id: uuid.UUID = None):
    return get_preferences(project_id)


@app.post("/preferences")
def set_prefs(req: PreferencesRequest):
    return set_preferences(req.preferences, req.project_id)


# --- Tasks / Reminders ---

@app.post("/tasks")
def create_task_endpoint(req: TaskCreateRequest):
    return create_task(req.description, req.due_days, req.project_id, req.related_node_id)


@app.get("/tasks/due")
def due_tasks(project_id: uuid.UUID = None):
    return {"tasks": get_due_tasks(project_id)}


@app.post("/tasks/complete")
def complete_task_endpoint(req: TaskCompleteRequest):
    if complete_task(req.task_id):
        return {"status": "completed", "task_id": str(req.task_id)}
    raise HTTPException(status_code=404, detail="Task not found or already completed")


@app.get("/tasks")
def list_tasks(project_id: uuid.UUID = None, include_completed: bool = False):
    return {"tasks": get_all_tasks(project_id, include_completed)}
