"""TachyGraph — Streamlit Dashboard. Run: streamlit run ui/dashboard.py"""
import json
import uuid

import requests
import streamlit as st

API = "http://localhost:8000"

st.set_page_config(page_title="TachyGraph", page_icon="⚡", layout="wide")


def api(method: str, path: str, **kwargs):
    try:
        r = getattr(requests, method)(f"{API}{path}", timeout=600, **kwargs)
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        return r.json() if "json" in ct else r.text
    except requests.exceptions.HTTPError:
        try:
            return {"error": r.json()}
        except Exception:
            return {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


# ── Sidebar ──────────────────────────────────────────────
st.sidebar.title("⚡ TachyGraph")

health = api("get", "/health")
if isinstance(health, dict) and "error" not in health:
    c1, c2 = st.sidebar.columns(2)
    c1.metric("DB", health.get("db", "?"))
    c2.metric("FAISS", f"{health.get('faiss','?')} ({health.get('faiss_vectors', 0)})")
else:
    st.sidebar.error("API unreachable")

projects = api("get", "/projects")
project_list = projects.get("projects", []) if isinstance(projects, dict) else []
project_map = {p["name"]: p["id"] for p in project_list}

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "🌐 API Explorer",
    "🔍 Search",
    "💬 Chat",
    "🤖 Agents",
    "📥 Ingest",
    "🧠 Memory",
    "🕸️ Graph",
    "⚙️ Maintenance",
    "👤 Preferences",
    "📋 Tasks",
    "📦 Export / Import",
])


# ── Overview ─────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("TachyGraph Dashboard")

    expiry = api("get", "/expiry/report")
    if isinstance(expiry, dict) and "error" not in expiry:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Projects", len(project_list))
        c2.metric("Total Nodes", expiry.get("total_nodes", 0))
        c3.metric("Expiring 24h", expiry.get("expiring_24h", 0))
        c4.metric("Expiring 7d", expiry.get("expiring_7d", 0))

    if project_list:
        st.subheader("Projects")
        for p in project_list:
            with st.expander(f"**{p['name']}** — {p.get('node_count', 0)} nodes"):
                st.write(p.get("summary", "No summary"))
                st.code(p["id"], language=None)

    models = api("get", "/models")
    if isinstance(models, dict) and "models" in models:
        st.subheader("Ollama Models")
        for m in models["models"]:
            st.write(f"- `{m['name']}` ({m['size'] / 1e9:.1f} GB)")


# ── API Explorer ─────────────────────────────────────────
elif page == "🌐 API Explorer":
    st.title("API Explorer")

    _SECTIONS = [
        ("Health & Status", [
            ("GET", "/health", "DB + FAISS + Ollama status", None),
            ("GET", "/projects", "List all projects with node counts", None),
            ("GET", "/models", "List available Ollama models", None),
            ("GET", "/expiry/report", "Expiry dashboard (24h, 7d, hot facts)", None),
        ]),
        ("Search", [
            ("POST", "/search", "Multi-signal pgvector search", '{"query": "neural network", "k": 5}'),
            ("POST", "/search/deep", "Ollama disambiguation → parallel strands", '{"query": "How do neural networks work?", "k": 5}'),
            ("POST", "/search/factchain", "Provenance-backed fact chain", '{"query": "Fokker-Planck equation", "k": 3}'),
            ("POST", "/search/fast", "FAISS-only search", '{"query": "mobile learning", "k": 5}'),
            ("POST", "/search/hybrid", "FAISS recall → pgvector re-rank", '{"query": "mobile learning English", "k": 5}'),
        ]),
        ("Chat", [
            ("POST", "/chat", "RAG chat (blocking or streaming)", '{"message": "What is the Fokker-Planck equation?"}'),
            ("POST", "/chat/feedback", "Correct/confirm last response", '{"session_id": "<uuid>", "feedback": "correct"}'),
            ("GET", "/sessions", "List recent chat sessions", None),
        ]),
        ("AI Agents", [
            ("POST", "/agent", "Single agent — LLM decides tools", '{"message": "What do I know about neural networks?"}'),
            ("POST", "/agent/orchestrator", "Multi-agent orchestrator", '{"message": "What info do I have about mobile learning?"}'),
            ("POST", "/agent/research", "Research agent — autonomous", '{"topic": "FAISS IVF-PQ internals"}'),
            ("POST", "/agent/maintain", "Memory maintenance agent", None),
        ]),
        ("Ingestion", [
            ("POST", "/ingest", "Ingest text via HTTP", '{"text": "# Test\\nSample.", "source_url": "test://demo", "project_name": "demo"}'),
            ("POST", "/ingest/web", "Crawl a website and ingest", '{"url": "https://docs.example.com", "limit": 10, "depth": 2}'),
            ("POST", "/ingest/chat", "Import ChatGPT/Claude/Gemini", '{"text": "Q: What is X?\\nA: X is Y.", "project_name": "chat-import"}'),
            ("GET", "/ingest/local/scan", "List files pending in ingest/", None),
            ("POST", "/ingest/local", "Ingest one local file", '{"filename": "my-doc.md", "project_name": "docs"}'),
            ("POST", "/ingest/local/all", "Ingest all pending local files", '{"project_name": "docs"}'),
            ("POST", "/ingest/local/auto", "Auto-ingest: scan + ingest all", None),
        ]),
        ("Memory", [
            ("POST", "/observe", "Observe Q&A interaction", '{"interaction_text": "Q: What DB? A: PostgreSQL 16."}'),
        ]),
        ("Graph Maintenance", [
            ("POST", "/compact", "Deduplicate near-identical nodes", '{"project_id": "<uuid>"}'),
            ("POST", "/mmr/recompute", "Recompute MMR edges", '{"project_id": null}'),
            ("POST", "/maintenance", "Run all maintenance tasks", None),
            ("POST", "/resolve", "Temporal conflict resolution", '{"project_id": "<uuid>"}'),
            ("POST", "/temporal/reaffirm", "Extend node validity", '{"node_id": "<uuid>", "extension_days": 10}'),
            ("POST", "/temporal/expiring", "List facts expiring soon", '{"project_id": "<uuid>"}'),
            ("POST", "/faiss/sync", "Sync pgvector → FAISS", '{"project_id": null}'),
        ]),
        ("User Features", [
            ("GET", "/preferences", "View user preferences", None),
            ("POST", "/preferences", "Set user preferences", '{"preferences": {"response_style": "detailed"}}'),
            ("POST", "/tasks", "Create a reminder/task", '{"description": "Review results", "due_days": 1}'),
            ("GET", "/tasks", "List all tasks", None),
            ("GET", "/tasks/due", "List tasks due within 24h", None),
            ("POST", "/tasks/complete", "Mark task completed", '{"task_id": "<uuid>"}'),
        ]),
        ("Export / Import", [
            ("POST", "/export", "Export knowledge graph", '{"format": "json"}'),
            ("POST", "/import", "Import from backup", '{"data": {}}'),
        ]),
    ]

    for sec_name, endpoints in _SECTIONS:
        st.subheader(sec_name)
        for method, path, desc, default_body in endpoints:
            key = f"api_{method}_{path}".replace("/", "_")
            color = {"GET": "🟢", "POST": "🔵", "DELETE": "🔴"}.get(method, "⚪")
            with st.expander(f"{color} **{method}** `{path}` — {desc}"):
                body_val = None
                if method == "POST" and default_body:
                    body_val = st.text_area("Request body (JSON)", value=default_body, height=100, key=f"{key}_body")
                if st.button("Execute", key=f"{key}_btn"):
                    with st.spinner("Calling..."):
                        if method == "GET":
                            result = api("get", path)
                        elif method == "POST":
                            try:
                                payload = json.loads(body_val) if body_val else {}
                            except json.JSONDecodeError as e:
                                st.error(f"Invalid JSON: {e}")
                                payload = None
                            if payload is not None:
                                result = api("post", path, json=payload)
                        elif method == "DELETE":
                            result = api("delete", path)
                        else:
                            result = {"error": f"Unsupported method: {method}"}
                    if isinstance(result, dict):
                        st.json(result)
                    else:
                        st.code(result)

    st.markdown("---")
    st.markdown("Also available: [Swagger UI](/docs) · [ReDoc](/redoc) · [Standalone API Explorer](/ui/api.html)")


# ── Search ───────────────────────────────────────────────
elif page == "🔍 Search":
    st.title("Search")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Multi-Signal", "Deep", "Fact Chain", "FAISS Fast", "Hybrid"])

    with tab1:
        q = st.text_input("Query", key="s1")
        k = st.slider("Results (k)", 1, 50, 10, key="s1k")
        c1, c2, c3 = st.columns(3)
        bw = c1.number_input("BM25 weight", 0.0, 1.0, 0.1, 0.05, key="s1bw")
        vw = c2.number_input("Vector weight", 0.0, 1.0, 0.6, 0.05, key="s1vw")
        tw = c3.number_input("Temporal weight", 0.0, 1.0, 0.3, 0.05, key="s1tw")
        if st.button("Search", key="s1b") and q:
            r = api("post", "/search", json={"query": q, "k": k, "bm25_weight": bw, "vector_weight": vw, "temporal_weight": tw})
            if isinstance(r, dict) and "results" in r:
                st.success(f"{r['count']} results")
                for res in r["results"]:
                    with st.expander(f"[{res.get('label','')}] {res.get('summary','')[:100]}"):
                        st.write(res.get("content", "")[:1000])
                        st.caption(f"ID: {res['id']} | Confidence: {res.get('confidence')}")
            else:
                st.json(r)

    with tab2:
        q = st.text_input("Query", key="s2")
        if st.button("Deep Search", key="s2b") and q:
            st.json(api("post", "/search/deep", json={"query": q, "k": 10}))

    with tab3:
        q = st.text_input("Query", key="s3")
        if st.button("Fact Chain", key="s3b") and q:
            r = api("post", "/search/factchain", json={"query": q, "k": 3})
            if isinstance(r, dict) and "response" in r:
                st.markdown(r["response"])
            else:
                st.json(r)

    with tab4:
        q = st.text_input("Query", key="s4")
        if st.button("FAISS Fast", key="s4b") and q:
            st.json(api("post", "/search/fast", json={"query": q, "k": 10}))

    with tab5:
        q = st.text_input("Query", key="s5")
        if st.button("Hybrid Search", key="s5b") and q:
            st.json(api("post", "/search/hybrid", json={"query": q, "k": 10}))


# ── Chat ─────────────────────────────────────────────────
elif page == "💬 Chat":
    st.title("Chat (RAG Pipeline)")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chat_session_id = None

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask TachyGraph..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                payload = {"message": prompt}
                if st.session_state.chat_session_id:
                    payload["session_id"] = st.session_state.chat_session_id
                r = api("post", "/chat", json=payload)
                if isinstance(r, dict) and "response" in r:
                    st.markdown(r["response"])
                    st.session_state.chat_history.append({"role": "assistant", "content": r["response"]})
                    st.session_state.chat_session_id = r.get("session_id")
                else:
                    st.error(str(r))

    if st.session_state.chat_session_id:
        st.caption(f"Session: {st.session_state.chat_session_id}")
        c1, c2, c3 = st.columns(3)
        if c1.button("👍 Correct"):
            st.json(api("post", "/chat/feedback", json={"session_id": st.session_state.chat_session_id, "feedback": "correct"}))
        if c2.button("👎 Wrong"):
            st.json(api("post", "/chat/feedback", json={"session_id": st.session_state.chat_session_id, "feedback": "wrong"}))
        correction = c3.text_input("Correction", key="corr")
        if c3.button("✏️ Submit Correction") and correction:
            st.json(api("post", "/chat/feedback", json={"session_id": st.session_state.chat_session_id, "feedback": "correction", "correction": correction}))

    st.markdown("---")
    st.subheader("Sessions")
    sessions = api("get", "/sessions")
    if isinstance(sessions, dict) and "sessions" in sessions:
        for s in sessions["sessions"]:
            c1, c2 = st.columns([4, 1])
            c1.write(f"`{s['id']}` — {s.get('last_active', '')}")
            if c2.button("🗑️", key=f"del-{s['id']}"):
                api("delete", f"/sessions/{s['id']}")
                st.rerun()


# ── Agents ───────────────────────────────────────────────
elif page == "🤖 Agents":
    st.title("AI Agents")
    st.warning("Agent calls can take 1–10 minutes with large models.")

    tab1, tab2, tab3, tab4 = st.tabs(["Single Agent", "Orchestrator", "Research", "Maintenance"])

    with tab1:
        msg = st.text_area("Message", key="a1")
        if st.button("Run Agent", key="a1b") and msg:
            with st.spinner("Agent thinking..."):
                r = api("post", "/agent", json={"message": msg})
            if isinstance(r, dict) and "response" in r:
                st.markdown(r["response"])
            else:
                st.json(r)

    with tab2:
        msg = st.text_area("Message", key="a2")
        if st.button("Run Orchestrator", key="a2b") and msg:
            with st.spinner("Orchestrator delegating..."):
                r = api("post", "/agent/orchestrator", json={"message": msg})
            if isinstance(r, dict) and "response" in r:
                st.markdown(r["response"])
            else:
                st.json(r)

    with tab3:
        topic = st.text_input("Research Topic", key="a3")
        if st.button("Start Research", key="a3b") and topic:
            with st.spinner("Researching (this may take several minutes)..."):
                r = api("post", "/agent/research", json={"topic": topic})
            if isinstance(r, dict) and "response" in r:
                st.markdown(r["response"])
            else:
                st.json(r)

    with tab4:
        if st.button("Run Maintenance Agent"):
            with st.spinner("Maintaining graph..."):
                r = api("post", "/agent/maintain")
            if isinstance(r, dict) and "response" in r:
                st.markdown(r["response"])
            else:
                st.json(r)


# ── Ingest ───────────────────────────────────────────────
elif page == "📥 Ingest":
    st.title("Ingestion")
    tab1, tab2, tab3, tab4 = st.tabs(["Text", "Local Files", "Web Crawl", "Chat Import"])

    with tab1:
        text = st.text_area("Text to ingest", height=200)
        source = st.text_input("Source URL", "manual://input")
        pname = st.text_input("Project name", "default", key="ing_pname")
        if st.button("Ingest Text") and text:
            st.json(api("post", "/ingest", json={"text": text, "source_url": source, "project_name": pname}))

    with tab2:
        scan = api("get", "/ingest/local/scan")
        if isinstance(scan, dict) and "files" in scan:
            st.write(f"**{scan['count']} files** pending in ingest/ folder")
            for f in scan["files"]:
                st.write(f"- `{f}`")
        pname = st.text_input("Project name", "default", key="loc_pname")
        c1, c2 = st.columns(2)
        fname = c1.text_input("Filename (for single ingest)")
        if c1.button("Ingest File") and fname:
            st.json(api("post", "/ingest/local", json={"filename": fname, "project_name": pname}))
        if c2.button("Ingest All"):
            st.json(api("post", "/ingest/local/all", json={"project_name": pname}))
        if st.button("Auto-Ingest (each file → own project)"):
            st.json(api("post", "/ingest/local/auto"))

    with tab3:
        url = st.text_input("URL to crawl")
        c1, c2 = st.columns(2)
        limit = c1.number_input("Page limit", 1, 500, 50)
        depth = c2.number_input("Crawl depth", 1, 10, 3)
        pname = st.text_input("Project name", "default", key="web_pname")
        if st.button("Start Crawl") and url:
            with st.spinner("Crawling..."):
                st.json(api("post", "/ingest/web", json={"url": url, "limit": limit, "depth": depth, "project_name": pname}))

    with tab4:
        chat_text = st.text_area("Paste conversation (or JSON)", height=200)
        pname = st.text_input("Project name", "chat-import", key="chat_pname")
        mode = st.radio("Mode", ["observe (Q&A memory)", "ingest (persistent knowledge)"])
        if st.button("Import") and chat_text:
            payload = {"project_name": pname, "mode": mode.split(" ")[0]}
            try:
                payload["json_data"] = json.loads(chat_text)
            except json.JSONDecodeError:
                payload["text"] = chat_text
            st.json(api("post", "/ingest/chat", json=payload))


# ── Memory ───────────────────────────────────────────────
elif page == "🧠 Memory":
    st.title("Memory (Q&A Layer)")
    interaction = st.text_area("Interaction text", placeholder="Q: What is X? A: X is Y.")
    if st.button("Observe") and interaction:
        st.json(api("post", "/observe", json={"interaction_text": interaction}))


# ── Graph ────────────────────────────────────────────────
elif page == "🕸️ Graph":
    st.title("Knowledge Graph")

    c1, c2, c3 = st.columns(3)
    proj_name = c1.selectbox("Project", ["All"] + list(project_map.keys()))
    node_filter = c2.selectbox("Node filter", ["All nodes", "Questions & Answers only", "Summaries only"])
    threshold = c3.slider("Similarity threshold", 0.5, 0.99, 0.75, 0.05)

    params = {"similarity_threshold": threshold}
    if proj_name != "All":
        params["project_id"] = project_map[proj_name]

    graph = api("get", f"/graph?{'&'.join(f'{k}={v}' for k,v in params.items())}")

    if isinstance(graph, dict) and "nodes" in graph:
        nodes = graph["nodes"]
        edges = graph["edges"]

        if node_filter == "Questions & Answers only":
            nodes = [n for n in nodes if n["label"] in ("QUESTION", "ANSWER")]
        elif node_filter == "Summaries only":
            nodes = [n for n in nodes if n["label"] == "SUMMARY"]

        node_ids = {n["id"] for n in nodes}
        edges = [e for e in edges if e["source"] in node_ids and e["target"] in node_ids]

        c1, c2, c3, c4 = st.columns(4)
        label_counts = {}
        for n in nodes:
            label_counts[n["label"]] = label_counts.get(n["label"], 0) + 1
        c1.metric("Nodes", len(nodes))
        c2.metric("Edges", len(edges))
        for i, (label, count) in enumerate(sorted(label_counts.items())):
            [c3, c4][i % 2].metric(label, count)

        edge_types = {}
        for e in edges:
            edge_types[e["label"]] = edge_types.get(e["label"], 0) + 1
        st.write("**Edge breakdown:**", edge_types)

        st.subheader("Nodes")
        for n in nodes[:100]:
            color = {"PROJECT": "🔴", "SUMMARY": "🔵", "QUESTION": "🟡", "ANSWER": "🟢"}.get(n["label"], "⚪")
            with st.expander(f"{color} [{n['label']}] {(n.get('summary') or 'No summary')[:80]}"):
                st.write(f"**ID:** `{n['id']}`")
                st.write(f"**Created:** {n.get('created_at', '')}")
                st.write(n.get("summary", ""))

        st.markdown('<a href="http://localhost:8000/ui/graph.html" target="_blank">Open 3D Graph Visualization →</a>', unsafe_allow_html=True)
    else:
        st.json(graph)


# ── Maintenance ──────────────────────────────────────────
elif page == "⚙️ Maintenance":
    st.title("Graph Maintenance")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Full Maintenance", "Compact", "MMR Recompute", "FAISS Sync", "Temporal"])

    with tab1:
        if st.button("Run Full Maintenance"):
            with st.spinner("Running..."):
                st.json(api("post", "/maintenance"))

    with tab2:
        pid = st.selectbox("Project", list(project_map.keys()), key="compact_proj")
        thresh = st.slider("Similarity threshold", 0.90, 1.0, 0.98, 0.01)
        if st.button("Compact"):
            st.json(api("post", "/compact", json={"project_id": project_map[pid], "similarity_threshold": thresh}))

    with tab3:
        if st.button("Recompute All MMR Edges"):
            with st.spinner("Recomputing..."):
                st.json(api("post", "/mmr/recompute", json={"project_id": None}))

    with tab4:
        if st.button("Sync pgvector → FAISS"):
            with st.spinner("Syncing..."):
                st.json(api("post", "/faiss/sync", json={"project_id": None}))

    with tab5:
        st.subheader("Expiring Facts")
        if project_list:
            pid = st.selectbox("Project", list(project_map.keys()), key="temp_proj")
            if st.button("Check Expiring"):
                st.json(api("post", "/temporal/expiring", json={"project_id": project_map[pid]}))

        st.subheader("Resolve Conflicts")
        if project_list:
            pid2 = st.selectbox("Project", list(project_map.keys()), key="resolve_proj")
            if st.button("Resolve"):
                st.json(api("post", "/resolve", json={"project_id": project_map[pid2]}))

        st.subheader("Reaffirm Node")
        nid = st.text_input("Node ID (UUID)")
        days = st.number_input("Extension days", 1, 365, 5)
        if st.button("Reaffirm") and nid:
            st.json(api("post", "/temporal/reaffirm", json={"node_id": nid, "extension_days": days}))


# ── Preferences ──────────────────────────────────────────
elif page == "👤 Preferences":
    st.title("User Preferences")
    prefs = api("get", "/preferences")
    if isinstance(prefs, dict) and "error" not in prefs:
        style = st.selectbox("Response style", ["concise", "detailed", "technical", "casual"], index=["concise", "detailed", "technical", "casual"].index(prefs.get("response_style", "concise")))
        level = st.selectbox("Expertise level", ["beginner", "intermediate", "advanced", "expert"], index=["beginner", "intermediate", "advanced", "expert"].index(prefs.get("expertise_level", "intermediate")))
        lang = st.text_input("Language", prefs.get("language", "english"))
        topics = st.text_input("Topics of interest (comma-separated)", ", ".join(prefs.get("topics_of_interest", [])))
        max_len = st.number_input("Max response length", 100, 5000, prefs.get("max_response_length", 500))

        if st.button("Save Preferences"):
            new_prefs = {
                "response_style": style,
                "expertise_level": level,
                "language": lang,
                "topics_of_interest": [t.strip() for t in topics.split(",") if t.strip()],
                "max_response_length": max_len,
            }
            st.json(api("post", "/preferences", json={"preferences": new_prefs}))
    else:
        st.json(prefs)


# ── Tasks ────────────────────────────────────────────────
elif page == "📋 Tasks":
    st.title("Tasks & Reminders")

    desc = st.text_input("New task description")
    due = st.number_input("Due in (days)", 1, 365, 1)
    if st.button("Create Task") and desc:
        st.json(api("post", "/tasks", json={"description": desc, "due_days": due}))

    st.subheader("All Tasks")
    tasks = api("get", "/tasks")
    if isinstance(tasks, dict) and "tasks" in tasks:
        for t in tasks["tasks"]:
            status = "✅" if t.get("completed") else "⏳"
            c1, c2 = st.columns([4, 1])
            c1.write(f"{status} **{t['description']}** — due {t.get('due_at', '')[:16]}")
            if not t.get("completed") and c2.button("Complete", key=f"tc-{t['id']}"):
                api("post", "/tasks/complete", json={"task_id": t["id"]})
                st.rerun()

    st.subheader("Due Within 24h")
    due_tasks = api("get", "/tasks/due")
    if isinstance(due_tasks, dict) and "tasks" in due_tasks:
        if due_tasks["tasks"]:
            for t in due_tasks["tasks"]:
                st.write(f"⚠️ **{t['description']}** — due {t.get('due_at', '')[:16]}")
        else:
            st.info("No tasks due within 24h")


# ── Export / Import ──────────────────────────────────────
elif page == "📦 Export / Import":
    st.title("Export / Import")

    tab1, tab2 = st.tabs(["Export", "Import"])

    with tab1:
        fmt = st.radio("Format", ["json", "markdown"])
        if st.button("Export"):
            r = api("post", "/export", json={"format": fmt})
            if fmt == "json" and isinstance(r, dict):
                st.json(r)
                st.download_button("Download JSON", json.dumps(r, indent=2), "tachygraph_export.json", "application/json")
            else:
                st.text(r)
                st.download_button("Download Markdown", str(r), "tachygraph_export.md", "text/markdown")

    with tab2:
        uploaded = st.file_uploader("Upload JSON export", type=["json"])
        if uploaded and st.button("Import"):
            data = json.load(uploaded)
            st.json(api("post", "/import", json={"data": data}))
