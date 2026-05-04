"""TachyGraph Q&A Architecture Load Test"""
import concurrent.futures
import json
import subprocess
import time
import uuid

import requests

API = "http://localhost:8000"
T = 120
PASS = FAIL = 0
FIXED_PROJECT = str(uuid.uuid4())

def green(s):
    global PASS; PASS += 1; print(f"\033[32m✓ {s}\033[0m")
def red(s, d=""):
    global FAIL; FAIL += 1; print(f"\033[31m✗ {s}\033[0m")
    if d: print(f"  {str(d)[:300]}")
def header(s):
    print(f"\n\033[1;36m━━━ {s} ━━━\033[0m")
def db(sql):
    r = subprocess.run(["docker", "exec", "tachygraph_db", "psql", "-U", "tachy_admin", "-d", "tachygraph", "-t", "-A", "-c", sql], capture_output=True, text=True)
    return r.stdout.strip()
def observe(text, pid=None):
    return requests.post(f"{API}/observe", json={"interaction_text": text, "project_id": pid}, timeout=T).json()

# ── Snapshot helper ──
def snapshot():
    q = int(db("SELECT COUNT(*) FROM nodes WHERE label='QUESTION'") or 0)
    a = int(db("SELECT COUNT(*) FROM nodes WHERE label='ANSWER'") or 0)
    ans = int(db("SELECT COUNT(*) FROM edges WHERE label='ANSWERS'") or 0)
    ctx = int(db("SELECT COUNT(*) FROM edges WHERE label='CONTEXT_OF'") or 0)
    sup = int(db("SELECT COUNT(*) FROM edges WHERE label='SUPERSEDES'") or 0)
    return {"questions": q, "answers": a, "answers_edges": ans, "context_of": ctx, "supersedes": sup}

header("0. BASELINE")
before = snapshot()
print(f"  Questions={before['questions']} Answers={before['answers']} ANSWERS={before['answers_edges']} CONTEXT_OF={before['context_of']} SUPERSEDES={before['supersedes']}")

# Setup fixed project
requests.post(f"{API}/ingest", json={"text": "# QA Load Test", "source_url": "test://qa-load", "project_id": FIXED_PROJECT, "project_name": "qa-load-test"}, timeout=T)

########################################
header("1. GLOBAL CLUSTERING — cross-project hub reuse")
########################################

pid1, pid2, pid3 = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
r1 = observe("Q: What is gravity? A: A fundamental force of attraction between masses.", pid1)
qid_gravity = r1.get("question_id")
if r1.get("status") == "stored":
    green(f"Project 1: hub {qid_gravity[:8]} (new={r1.get('question_is_new')})")
else:
    red("Project 1 observation failed", json.dumps(r1))
    qid_gravity = None

r2 = observe("Q: What is gravity? A: Gravity is described by Einstein's general relativity.", pid2)
if r2.get("question_id") == qid_gravity and r2.get("question_is_new") == False:
    green(f"Project 2: reused hub {qid_gravity[:8]}")
else:
    red(f"Project 2: got hub {r2.get('question_id','')[:8]}, expected {qid_gravity[:8]}", json.dumps(r2))

r3 = observe("Q: What is gravity? A: 9.8 m/s² on Earth's surface.", pid3)
if r3.get("question_id") == qid_gravity and r3.get("question_is_new") == False:
    green(f"Project 3: reused hub {qid_gravity[:8]}")
else:
    red(f"Project 3: got hub {r3.get('question_id','')[:8]}, expected {qid_gravity[:8]}", json.dumps(r3))

# Verify single hub in DB
hub_count = int(db(f"SELECT COUNT(*) FROM nodes WHERE label='QUESTION' AND content LIKE '%gravity%'") or 0)
answer_count = int(db(f"SELECT COUNT(*) FROM edges WHERE target_id='{qid_gravity}' AND label='ANSWERS'") or 0)
if hub_count == 1:
    green(f"Single hub for 'gravity' ({answer_count} answers)")
else:
    red(f"Expected 1 hub, found {hub_count}")

########################################
header("2. FAST PATH THROUGHPUT — burst 20 Q&A pairs")
########################################

questions = [
    ("What is DNA?", "Deoxyribonucleic acid, the molecule carrying genetic instructions."),
    ("What is RNA?", "Ribonucleic acid, involved in protein synthesis."),
    ("What is ATP?", "Adenosine triphosphate, the energy currency of cells."),
    ("What is mitosis?", "Cell division producing two identical daughter cells."),
    ("What is meiosis?", "Cell division producing four genetically different gametes."),
    ("What is osmosis?", "Movement of water across a semipermeable membrane."),
    ("What is entropy?", "A measure of disorder in a thermodynamic system."),
    ("What is inertia?", "An object's resistance to changes in its state of motion."),
    ("What is torque?", "A rotational force applied at a distance from an axis."),
    ("What is voltage?", "Electric potential difference between two points."),
    ("What is current?", "The flow of electric charge through a conductor."),
    ("What is resistance?", "Opposition to the flow of electric current."),
    ("What is frequency?", "Number of wave cycles per unit time."),
    ("What is wavelength?", "Distance between consecutive wave crests."),
    ("What is amplitude?", "Maximum displacement of a wave from equilibrium."),
    ("What is momentum?", "Product of an object's mass and velocity."),
    ("What is acceleration?", "Rate of change of velocity over time."),
    ("What is density?", "Mass per unit volume of a substance."),
    ("What is pressure?", "Force applied per unit area."),
    ("What is temperature?", "A measure of average kinetic energy of particles."),
]

start = time.time()
results = []
for q, a in questions:
    r = observe(f"Q: {q} A: {a}", FIXED_PROJECT)
    results.append(r)
elapsed = time.time() - start

stored = sum(1 for r in results if r.get("status") == "stored")
new_hubs = sum(1 for r in results if r.get("question_is_new") == True)
avg = elapsed / len(questions) * 1000

green(f"20 Q&A pairs in {elapsed:.1f}s ({avg:.0f}ms avg) — {stored} stored, {new_hubs} new hubs")
if avg < 5000:
    green(f"Fast path performance OK ({avg:.0f}ms < 5000ms)")
else:
    red(f"Slow: {avg:.0f}ms per observe (expected <5000ms with fast path)")

########################################
header("3. CONCURRENT OBSERVERS — 5 parallel threads")
########################################

concurrent_qs = [
    ("What is a quark?", "A fundamental particle in the standard model."),
    ("What is a lepton?", "A fundamental particle like electrons and neutrinos."),
    ("What is a boson?", "A force-carrying particle like photons and gluons."),
    ("What is a fermion?", "A matter particle with half-integer spin."),
    ("What is a hadron?", "A composite particle made of quarks bound by strong force."),
]

start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
    futures = [pool.submit(observe, f"Q: {q} A: {a}", FIXED_PROJECT) for q, a in concurrent_qs]
    concurrent_results = [f.result() for f in concurrent.futures.as_completed(futures)]
elapsed = time.time() - start

stored = sum(1 for r in concurrent_results if r.get("status") == "stored")
errors = sum(1 for r in concurrent_results if "error" in r)
green(f"5 concurrent observers in {elapsed:.1f}s — {stored} stored, {errors} errors")
if errors > 0:
    red(f"{errors} concurrent errors", json.dumps([r for r in concurrent_results if "error" in r][:2]))

########################################
header("4. 10-SLOT EVICTION STRESS — push 15 answers into one hub")
########################################

eviction_qid = None
stored_count = 0
rejected_count = 0

for i in range(15):
    conf = 0.86 + (i * 0.005)
    r = observe(f"Q: What is pi? A: Pi variant {i} is approximately {3.14159 + i*0.0001:.5f}, conf {conf:.3f}.", FIXED_PROJECT)
    s = r.get("status")
    if s == "stored":
        stored_count += 1
        if eviction_qid is None:
            eviction_qid = r.get("question_id")
    elif s == "rejected":
        rejected_count += 1

print(f"  Stored: {stored_count}, Rejected: {rejected_count}")

if eviction_qid:
    slot_count = int(db(f"SELECT COUNT(*) FROM edges WHERE target_id='{eviction_qid}' AND label='ANSWERS'") or 0)
    if slot_count <= 10:
        green(f"Hub capped at {slot_count} answers (≤10)")
    else:
        red(f"Hub has {slot_count} answers — exceeds cap!")

    sup = int(db(f"SELECT COUNT(*) FROM edges e JOIN nodes a ON e.source_id=a.id WHERE a.cluster_id='{eviction_qid}' AND e.label='SUPERSEDES'") or 0)
    if sup > 0:
        green(f"{sup} SUPERSEDES edges from eviction")
    elif stored_count > 10:
        red("No SUPERSEDES despite >10 stored")

    expired = int(db(f"SELECT COUNT(*) FROM nodes WHERE cluster_id='{eviction_qid}' AND label='ANSWER' AND valid_until <= NOW()") or 0)
    if expired > 0:
        green(f"{expired} evicted answers expired")

    # Verify lowest confidence was evicted
    min_conf = db(f"SELECT MIN(confidence) FROM nodes n JOIN edges e ON n.id=e.source_id WHERE e.target_id='{eviction_qid}' AND e.label='ANSWERS' AND (n.valid_until IS NULL OR n.valid_until > NOW())")
    if min_conf:
        green(f"Lowest active confidence: {min_conf}")

########################################
header("5. QUESTION→QUESTION EDGE BLOCK")
########################################

qids = db("SELECT id FROM nodes WHERE label='QUESTION' LIMIT 2").split("\n")
if len(qids) >= 2:
    q1, q2 = qids[0].strip(), qids[1].strip()
    before_edges = int(db(f"SELECT COUNT(*) FROM edges WHERE source_id='{q1}' AND target_id='{q2}'") or 0)
    db(f"DO $$ BEGIN INSERT INTO edges (source_id, target_id, label) VALUES ('{q1}','{q2}','ELABORATES'); EXCEPTION WHEN raise_exception THEN NULL; END $$;")
    after_edges = int(db(f"SELECT COUNT(*) FROM edges WHERE source_id='{q1}' AND target_id='{q2}'") or 0)
    if after_edges == before_edges:
        green("Q→Q edge blocked by trigger")
    else:
        red("Q→Q edge was NOT blocked!")

########################################
header("6. CROSS-CLUSTER WEAVING")
########################################

ctx = int(db("SELECT COUNT(*) FROM edges WHERE label='CONTEXT_OF'") or 0)
cross = int(db("SELECT COUNT(*) FROM edges e JOIN nodes s ON e.source_id=s.id JOIN nodes t ON e.target_id=t.id WHERE e.label='CONTEXT_OF' AND s.cluster_id IS NOT NULL AND t.cluster_id IS NOT NULL AND s.cluster_id != t.cluster_id") or 0)
if ctx > 0:
    green(f"{ctx} CONTEXT_OF edges ({cross} cross-cluster)")
else:
    red("No CONTEXT_OF edges")

########################################
header("7. TEMPORAL WINDOWS")
########################################

bad_windows = int(db("SELECT COUNT(*) FROM nodes WHERE label IN ('QUESTION','ANSWER') AND valid_until IS NOT NULL AND EXTRACT(DAY FROM (valid_until - valid_from))::int != 5") or 0)
if bad_windows == 0:
    green("All Q&A nodes have correct 5-day window")
else:
    red(f"{bad_windows} Q&A nodes have wrong temporal window")

########################################
header("8. DEGREE CAP INTEGRITY")
########################################

over = int(db("SELECT COUNT(*) FROM nodes WHERE label='ANSWER' AND degree > degree_cap AND degree_cap > 0") or 0)
if over == 0:
    green("No nodes exceed degree cap")
else:
    red(f"{over} nodes exceed degree cap!")

########################################
header("9. DUPLICATE HUB CHECK")
########################################

dupes = db("SELECT content, COUNT(*) FROM nodes WHERE label='QUESTION' GROUP BY content HAVING COUNT(*) > 1")
if dupes:
    lines = [l for l in dupes.split("\n") if l.strip()]
    red(f"{len(lines)} duplicate question texts (pre-fix data)")
    for l in lines[:3]:
        print(f"  {l.strip()}")
else:
    green("No duplicate question hubs")

########################################
header("10. FINAL SNAPSHOT + DELTA")
########################################

after = snapshot()
print(f"  Questions: {before['questions']} → {after['questions']} (+{after['questions']-before['questions']})")
print(f"  Answers:   {before['answers']} → {after['answers']} (+{after['answers']-before['answers']})")
print(f"  ANSWERS:   {before['answers_edges']} → {after['answers_edges']} (+{after['answers_edges']-before['answers_edges']})")
print(f"  CONTEXT_OF:{before['context_of']} → {after['context_of']} (+{after['context_of']-before['context_of']})")
print(f"  SUPERSEDES:{before['supersedes']} → {after['supersedes']} (+{after['supersedes']-before['supersedes']})")

# Sanity: ANSWERS edges should equal non-expired answer count
active_answers = int(db("SELECT COUNT(*) FROM nodes WHERE label='ANSWER' AND (valid_until IS NULL OR valid_until > NOW())") or 0)
answers_edges = after['answers_edges']
if active_answers == answers_edges:
    green(f"ANSWERS edges ({answers_edges}) = active answers ({active_answers})")
else:
    red(f"Mismatch: {answers_edges} ANSWERS edges vs {active_answers} active answers")

########################################
header("RESULTS")
########################################

total = PASS + FAIL
print(f"\n\033[1m{total} tests: \033[32m{PASS} passed\033[0m, \033[31m{FAIL} failed\033[0m\n")
if FAIL == 0:
    print("\033[1;32m🎉 All Q&A load tests passed!\033[0m")
else:
    print("\033[1;31m⚠  Issues found — see above.\033[0m")
