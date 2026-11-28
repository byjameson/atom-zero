"""Local HTTP server for Atom Zero, the tiny transformer daily assistant.

    python -m atomzero.chat_server \
      --model checkpoints/transformer_behavior_50m_scaled \
      --adapter checkpoints/adapters_50m/v0001 \
      --docs data/docs.db

Endpoints:
  GET  /health
  GET  /v1/models
  GET  /v1/documents
  GET  /v1/documents/evidence?source=...&page=...&section=...
  GET  /v1/documents/summary?source=...&source_prefix=...&sentences=8
  GET  /v1/documents/compare?source=...&against=...&max_claims=200
  GET  /v1/documents/timeline?source=...&source_prefix=...&entity=...&limit=100
  GET  /v1/graph?limit_chunks=...&min_edge=...&max_entities=...
  GET  /v1/graph/entity?name=...&limit=...
  GET  /v1/graph/relations?entity=...&relation=...&limit=100
  GET  /v1/graph/relations/path?source=...&target=...&max_depth=3
  GET  /v1/graph/path?source=...&target=...&max_depth=4
  GET  /v1/skills
  GET  /v1/skills/chains
  POST /v1/skills/run
  POST /v1/skills/chain
  POST /v1/skills/chains/run
  POST /v1/documents
  POST /v1/documents/upload?source=...&title=...
  POST /v1/documents/search
  POST /v1/documents/answer
  POST /v1/hungry/run
  POST /v1/briefing
  GET  /v1/briefing?session_id=default&tag=work
  POST /v1/weekly-review
  GET  /v1/weekly-review/export?session_id=...&days=7&tag=work&audit_action=delete_tag
  GET  /v1/tasks/list?session_id=default&status=open&due=week&priority=high&sort=priority
  GET  /v1/tasks/tags?session_id=default
  GET  /v1/tasks/tags/audit?session_id=default&limit=10&tag=work&action=delete_tag
  GET  /v1/tasks/tags/audit/export?session_id=default&format=csv&tag=work
  POST /v1/tasks/tags/rename  (add "confirm": true to apply)
  POST /v1/tasks/tags/delete  (add "confirm": true to apply)
  POST /v1/tasks/tags/undo-delete
  GET  /v1/tasks/export?session_id=default&status=open&due=week&priority=high&sort=priority&format=csv
  GET  /v1/tasks/overdue?session_id=default&sort=priority&tag=work
  GET  /v1/tasks/due-soon?session_id=default&days=7&sort=priority&tag=work
  GET  /v1/tasks/search?session_id=default&q=report&status=open&due=week&priority=high&sort=priority
  POST /v1/tasks/reschedule
  POST /v1/tasks/priority
  POST /v1/tasks/delete
  GET  /v1/weekly-review/reports?limit=10&tag=work
  GET  /v1/weekly-review/reports?id=w0001
  GET  /v1/weekly-review/reports/compare?left=w0001&right=w0002
  GET  /v1/weekly-review/reports/compare/export?left=w0001&right=w0002
  GET  /v1/weekly-review/reports/verify
  POST /v1/weekly-review/reports/repair
  POST /v1/weekly-review/reports/prune
  POST /v1/weekly-review/reports/keep
  POST /v1/weekly-review/reports/delete
  POST /v1/goals/plan
  GET  /v1/teacher/notes
  POST /v1/teacher/notes
  POST /v1/teacher/notes/verify
  POST /v1/teacher/ask
  POST /v1/teach
  POST /v1/research
  GET  /v1/research/queue
  POST /v1/research/queue
  POST /v1/research/queue/approve
  POST /v1/research/queue/reject
  POST /v1/research/queue/run
  POST /v1/learn_url
  POST /v1/agent/chat
  POST /v1/agent/plan
  POST /generate
  POST /v1/chat/completions
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from atomzero import __version__
from atomzero.copy_router import route_copy
from atomzero.numeric_router import route_numeric
from atomzero.recall_router import route_recall
from atomzero.identity_router import route_identity
from atomzero.doc_memory import SourceNotFound, grounded_answer
from atomzero.curiosity import (
    CuriosityError, curiosity_items, propose_from_gaps)
from atomzero.feedback import (
    FeedbackError, FeedbackNotFound, FeedbackStore)
from atomzero.knowledge_gaps import (
    GapNotFound, KnowledgeGapError, KnowledgeGapStore)
from atomzero.neural_embedder import NeuralEmbedderError, NeuralReranker
from atomzero.research_queue import (
    ItemNotFound, ResearchQueue, ResearchQueueError)
from atomzero.teacher_notes import TeacherNoteTooLarge, add_teacher_note
from atomzero.teacher_provider import TeacherProvider, TeacherProviderError
from atomzero.tool_planner import ToolPlanError, execute_plan
from atomzero.plan_audit import audit_plan
from atomzero.transformer import TransformerLM, generate
from atomzero.weekly_reports import (
    DEFAULT_REPORTS_PATH, WeeklyReportArchive, WeeklyReportError)

_lock = threading.Lock()
_state: dict = {}


def _weekly_reports_archive() -> WeeklyReportArchive:
    archive = _state.get("weekly_reports")
    if isinstance(archive, WeeklyReportArchive):
        return archive
    archive = WeeklyReportArchive(
        _state.get("weekly_reports_path", DEFAULT_REPORTS_PATH))
    _state["weekly_reports"] = archive
    return archive

MAX_BODY_BYTES = 2_000_000
MAX_DOC_BYTES = 10_000_000
MAX_SOURCE_CHARS = 512
MAX_TITLE_CHARS = 200


class DocumentTooLarge(ValueError):
    pass


def ingest_document(docs, source, text, title: str = "") -> dict:
    """Validate and index a text document, returning its provenance."""
    if docs is None:
        raise ValueError("no document memory configured")
    source = str(source or "").strip()
    title = str(title or "").strip()
    if not source:
        raise ValueError("source is required")
    if len(source) > MAX_SOURCE_CHARS:
        raise ValueError(f"source is too long (max {MAX_SOURCE_CHARS} chars)")
    if len(title) > MAX_TITLE_CHARS:
        raise ValueError(f"title is too long (max {MAX_TITLE_CHARS} chars)")
    if any(ord(ch) < 32 for ch in source + title):
        raise ValueError("source/title contains control characters")
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    text = text.strip()
    if not text:
        raise ValueError("text is empty")
    raw = text.encode("utf-8")
    if len(raw) > MAX_DOC_BYTES:
        raise DocumentTooLarge(
            f"document is too large (max {MAX_DOC_BYTES} bytes)")
    chunks = docs.add_text(source, text, title=title)
    content_sha = hashlib.sha256(raw).hexdigest()
    return {
        "source": source,
        "title": title or Path(source).stem,
        "chunks": chunks,
        "bytes": len(raw),
        "sha256": content_sha,
        "content_sha256": content_sha,
        "replace": True,
    }


def ingest_upload(docs, source, data, title: str = "") -> dict:
    """Validate and index a text/PDF/DOCX/HTML upload."""
    if docs is None:
        raise ValueError("no document memory configured")
    source = str(source or "").strip()
    title = str(title or "").strip()
    if not source:
        raise ValueError("source is required")
    if len(source) > MAX_SOURCE_CHARS:
        raise ValueError(f"source is too long (max {MAX_SOURCE_CHARS} chars)")
    if len(title) > MAX_TITLE_CHARS:
        raise ValueError(f"title is too long (max {MAX_TITLE_CHARS} chars)")
    if any(ord(ch) < 32 for ch in source + title):
        raise ValueError("source/title contains control characters")
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("document data must be bytes")
    data = bytes(data)
    if len(data) > MAX_DOC_BYTES:
        raise DocumentTooLarge(
            f"document is too large (max {MAX_DOC_BYTES} bytes)")
    from atomzero.doc_ingest import (
        DocumentExtractError, extract_bytes)

    try:
        extracted = extract_bytes(source, data, title=title)
    except DocumentExtractError as exc:
        raise ValueError(str(exc)) from exc
    chunks = docs.add_blocks(
        source, extracted["blocks"], title=extracted["title"])
    pages = [block.get("page") for block in extracted["blocks"]
             if block.get("page") is not None]
    file_sha = hashlib.sha256(data).hexdigest()
    content_text = "\n\n".join(
        str(block.get("text", "")) for block in extracted["blocks"])
    content_sha = hashlib.sha256(content_text.encode("utf-8")).hexdigest()
    return {
        "source": source,
        "title": extracted["title"],
        "chunks": chunks,
        "bytes": len(data),
        "sha256": file_sha,
        "file_sha256": file_sha,
        "content_sha256": content_sha,
        "replace": True,
        "format": extracted["format"],
        "pages": len(set(pages)),
    }


def resolve_adapter(adapter: str,
                    active_file: str = "data/active_adapter.json") -> str:
    """Use the promoted adapter when ``adapter`` is ``auto``."""
    if adapter != "auto":
        return adapter
    path = Path(active_file)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(payload.get("path", ""))


PAGE = """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>Atom Zero - local assistant</title>
<style>
:root{
  --bg:#f4f7fb; --bg-grad-1:#e9f1ff; --bg-grad-2:#f7efff;
  --panel:#ffffff; --panel-soft:#f7f9fc; --text:#142033; --muted:#65748b;
  --border:#d9e2ee; --border-strong:#becbdd;
  --accent:#2563eb; --accent-2:#7c3aed; --accent-soft:#e8efff;
  --danger:#dc2626; --danger-soft:#fee2e2; --ok:#059669;
  --shadow-sm:0 1px 2px rgba(20,32,51,.06);
  --shadow-md:0 10px 28px rgba(20,32,51,.09);
  color-scheme: light;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#0b1220; --bg-grad-1:#0f1b31; --bg-grad-2:#1a1230;
    --panel:#111b2e; --panel-soft:#0e1728; --text:#e6edf7; --muted:#9aa8bd;
    --border:#26344b; --border-strong:#344563;
    --accent:#60a5fa; --accent-2:#a78bfa; --accent-soft:#182b4d;
    --danger:#f87171; --danger-soft:#3a1418; --ok:#34d399;
    --shadow-sm:0 1px 2px rgba(0,0,0,.3);
    --shadow-md:0 12px 32px rgba(0,0,0,.38);
    color-scheme: dark;
  }
}
*{box-sizing:border-box}
body{
  font:16px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  max-width:1120px; margin:0 auto; padding:34px 22px 64px; color:var(--text);
  background:
    radial-gradient(1100px 600px at 8% -12%, var(--bg-grad-1), transparent 60%),
    radial-gradient(900px 520px at 100% 0%, var(--bg-grad-2), transparent 55%),
    var(--bg);
  min-height:100vh; -webkit-font-smoothing:antialiased;
}
h1{font-size:clamp(2.1rem,4vw,3.1rem); line-height:1.05; letter-spacing:-.04em; margin:0 0 6px; font-weight:800}
h1 + .hint{font-size:15px; letter-spacing:.02em}
h3{font-size:1.01rem; letter-spacing:.02em; text-transform:uppercase; margin:30px 0 10px; display:flex; align-items:center; gap:10px}
h3::before{content:""; width:8px; height:1.05em; border-radius:999px; background:linear-gradient(180deg,var(--accent),var(--accent-2)); box-shadow:0 0 0 4px var(--accent-soft)}
.hint{color:var(--muted); font-size:13.5px}
#m{display:inline-flex; align-items:center; gap:8px; padding:7px 12px; border:1px solid var(--border); border-radius:999px; background:var(--panel); box-shadow:var(--shadow-sm)}
#m::before{content:""; width:8px; height:8px; border-radius:50%; background:var(--ok); box-shadow:0 0 0 4px color-mix(in srgb, var(--ok) 18%, transparent)}
#chat{display:flex; flex-direction:column; gap:12px; min-height:240px; max-height:520px; overflow-y:auto; margin:18px 0; padding:16px; border:1px solid var(--border); border-radius:18px; background:color-mix(in srgb, var(--panel) 88%, transparent); box-shadow:var(--shadow-md); scroll-behavior:smooth}
.msg{max-width:min(85%,760px); padding:11px 14px; border-radius:16px; white-space:pre-wrap; border:1px solid transparent; box-shadow:var(--shadow-sm); line-height:1.55}
.user{align-self:flex-end; background:linear-gradient(135deg,var(--accent),color-mix(in srgb,var(--accent) 65%, var(--accent-2))); color:#fff; border-bottom-right-radius:5px}
.bot{align-self:flex-start; background:var(--panel); border-color:var(--border); border-bottom-left-radius:5px}
.meta{font-size:11.5px; color:var(--muted); margin-top:6px}
.user .meta{color:color-mix(in srgb, #fff 76%, transparent)}
.err{background:var(--danger-soft)!important; border-color:color-mix(in srgb,var(--danger) 35%, transparent)!important; color:var(--danger)}
.row{display:flex; flex-wrap:wrap; gap:9px; align-items:center; margin:8px 0}
.row input{flex:1 1 14em; min-width:0}
.row label{display:inline-flex; align-items:center; gap:7px; white-space:nowrap}
input,select,textarea{font:inherit; color:var(--text); background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:9px 11px; transition:border-color .15s, box-shadow .15s, background .15s}
input:hover,select:hover,textarea:hover{border-color:var(--border-strong)}
input:focus-visible,select:focus-visible,textarea:focus-visible,button:focus-visible{outline:none; border-color:var(--accent); box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 22%, transparent)}
textarea{width:100%; min-height:110px; resize:vertical}
select{max-width:200px}
input[type=checkbox]{width:auto; accent-color:var(--accent); padding:0}
input[type=file]{padding:8px}
button{font:inherit; font-weight:600; padding:9px 13px; color:var(--text); background:var(--panel); border:1px solid var(--border); border-radius:10px; cursor:pointer; box-shadow:var(--shadow-sm); transition:transform .12s, box-shadow .15s, border-color .15s, background .15s}
button:hover{transform:translateY(-1px); border-color:var(--border-strong); box-shadow:var(--shadow-md)}
button:active{transform:translateY(0)}
button:disabled{opacity:.55; cursor:not-allowed; transform:none}
#g,#newsession{background:linear-gradient(135deg,var(--accent),var(--accent-2)); border-color:transparent; color:#fff; box-shadow:0 8px 20px color-mix(in srgb,var(--accent) 30%, transparent)}
#forgetdoc,#forgetmem,#deletetagbtn,#deletetaskbtn{color:var(--danger); border-color:color-mix(in srgb,var(--danger) 35%, var(--border))}
#forgetdoc:hover,#forgetmem:hover,#deletetagbtn:hover,#deletetaskbtn:hover{background:var(--danger-soft)}
.chips{display:flex; flex-wrap:wrap; gap:7px; margin:10px 0 18px}
.chips button{font-size:12.5px; font-weight:500; padding:6px 11px; border-radius:999px; background:var(--panel-soft); color:var(--muted); box-shadow:none}
.chips button:hover{color:var(--accent); border-color:color-mix(in srgb,var(--accent) 40%, var(--border)); background:var(--accent-soft); transform:translateY(-1px)}
.qitem{border:1px solid var(--border); border-radius:14px; padding:12px 14px; margin:10px 0; background:var(--panel); box-shadow:var(--shadow-sm); transition:border-color .15s, box-shadow .15s, transform .12s}
.qitem:hover{border-color:var(--border-strong); box-shadow:var(--shadow-md); transform:translateY(-1px)}
.qitem .actions{display:flex; gap:8px; flex-wrap:wrap; margin-top:9px}
.qitem .src{font-size:12px; color:var(--muted); word-break:break-word; margin-top:5px; max-height:140px; overflow:auto}
.msg .actions{display:flex; gap:6px; align-items:center; flex-wrap:wrap; margin-top:8px}
.msg .actions button{font-size:11.5px; padding:4px 8px; box-shadow:none; font-weight:500}
.topbar{display:flex; justify-content:flex-end; align-items:center; gap:10px; margin:14px 0}
.proactive-toggle{display:inline-flex; align-items:center; gap:7px; padding:7px 11px; border:1px solid var(--border); border-radius:10px; background:var(--panel-soft); color:var(--muted); font-size:13px; font-weight:600; cursor:pointer; user-select:none}
.proactive-toggle:hover{border-color:var(--border-strong); color:var(--accent)}
.proactive-toggle input{accent-color:var(--accent); margin:0}
.composer{display:flex; gap:8px; align-items:stretch; padding:8px; border:1px solid var(--border); border-radius:16px; background:var(--panel); box-shadow:var(--shadow-md)}
.composer input{flex:1; min-width:0; border:0; box-shadow:none; background:transparent; padding:10px}
.composer input:focus-visible{box-shadow:none}
.composer #g{flex:0 0 auto; align-self:stretch}
.attach{display:inline-flex; align-items:center; justify-content:center; padding:0 13px; border:1px solid var(--border); border-radius:10px; background:var(--panel-soft); color:var(--muted); cursor:pointer; font-weight:600; font-size:13px; transition:border-color .15s, background .15s, color .15s}
.attach:hover{border-color:var(--border-strong); color:var(--accent); background:var(--accent-soft)}
.attach input{display:none}
.attach-status{margin:8px 2px 0}
.attach-status b{color:var(--text); font-weight:600}
.advanced{margin-top:26px; border:1px solid var(--border); border-radius:16px; background:color-mix(in srgb, var(--panel) 86%, transparent); box-shadow:var(--shadow-sm); overflow:hidden}
.advanced summary{cursor:pointer; padding:14px 16px; font-weight:700; list-style:none; display:flex; align-items:center; gap:10px; user-select:none}
.advanced summary::-webkit-details-marker{display:none}
.advanced summary::before{content:"+"; display:inline-flex; align-items:center; justify-content:center; width:22px; height:22px; border-radius:8px; background:var(--accent-soft); color:var(--accent); font-size:18px; line-height:1}
.advanced[open] summary::before{content:"–"}
.advanced-body{padding:0 18px 18px; border-top:1px solid var(--border)}
.advanced-body > h3:first-of-type{margin-top:22px}
#graphnodes button{background:var(--accent-soft); color:var(--accent); border-color:transparent; font-weight:600}
#graphnodes button:hover{background:var(--accent); color:#fff}
#docresults,#evidence,#cmpresult,#timeline,#graphnodes,#graphedges,#graphevidence,#relations,#docanswer,#tasklistres,#tasktagsres,#tagauditres,#briefres,#overdueres,#dueres,#tasksearchres,#weeklyreports,#weeklyreportspreview,#weeklyreportscheck,#weeklyreportscompareout{display:flex; flex-direction:column; gap:2px; overflow-x:auto}
table{width:100%; border-collapse:separate; border-spacing:0; margin:10px 0; font-size:13.5px; border:1px solid var(--border); border-radius:12px; overflow:hidden}
th,td{padding:9px 11px; text-align:left; border-bottom:1px solid var(--border)}
th{background:var(--panel-soft); color:var(--muted); font-size:11.5px; text-transform:uppercase; letter-spacing:.05em}
tr:last-child td{border-bottom:0}
tbody tr:nth-child(even) td{background:color-mix(in srgb,var(--panel-soft) 55%, transparent)}
code{font-family:ui-monospace, SFMono-Regular, Consolas, monospace; font-size:.9em; padding:2px 6px; border-radius:6px; background:var(--panel-soft); border:1px solid var(--border)}
hr{border:0; height:1px; margin:30px 0; background:linear-gradient(90deg,transparent,var(--border),transparent)}
a{color:var(--accent)}
::selection{background:var(--accent); color:#fff}
::-webkit-scrollbar{width:10px; height:10px}
::-webkit-scrollbar-thumb{background:color-mix(in srgb,var(--muted) 35%, transparent); border-radius:999px; border:2px solid transparent; background-clip:padding-box}
::-webkit-scrollbar-track{background:transparent}
@media (max-width:720px){
  body{padding:20px 14px 48px}
  h1{font-size:2.1rem}
  .row{align-items:stretch}
  .row input,.row select{flex:1 1 100%}
  .row label{white-space:normal}
  .composer{flex-wrap:wrap}
  .composer input{flex:1 1 100%}
  .composer #g,.attach{flex:1 1 auto}
  .msg{max-width:95%}
  #chat{max-height:none; padding:12px}
  select{max-width:none}
}
</style>
<h1>Atom Zero</h1>
<p class="hint">local-first personal assistant</p>
<p id="m" class="hint">loading...</p>
<div class="topbar">
<label class="proactive-toggle" title="Let Atom Zero start a conversation when something needs attention">
<input id="proactivetoggle" type="checkbox">
<span>Proactive</span>
</label>
<button id="newsession">New session</button>
</div>
<div id="chat"></div>
<div class="composer">
<input id="p" placeholder="Ask, command, or attach a PDF/note...">
<label class="attach" title="Attach a PDF, DOCX, HTML, Markdown or text file">
<input id="pfile" type="file"
 accept=".txt,.md,.markdown,.html,.htm,.pdf,.docx,text/plain,text/html,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document">
<span>Attach</span>
</label>
<button id="g">Send</button>
</div>
<p id="attachstatus" class="hint attach-status">Attach a file, then type <b>bunu öğren</b> to learn it or <b>bu bilgileri unut</b> to forget it.</p>
<div class="chips">
<button data-q="what can you do?">what can it do</button>
<button data-q="brief me">brief me</button>
<button data-q="list tasks">tasks</button>
<button data-q="weekly review">weekly review</button>
<button data-q="what do you remember?">memory</button>
</div>
<details id="advanced" class="advanced">
<summary>Advanced tools &amp; data</summary>
<div class="advanced-body">
<hr>
<p class="hint">Quick command demos</p>
<div class="chips">
<button data-q="add task buy milk">add task</button>
<button data-q="list tasks priority">priority list</button>
<button data-q="list tasks done">done list</button>
<button data-q="list tasks due week priority high">filtered list</button>
<button data-q="list tags">tags</button>
<button data-q="my name is Alex">remember my name</button>
<button data-q="what is 12 * 7?">calculate</button>
<button data-q="what is the launch code?">launch code</button>
<button data-q="how long do I have to submit a warranty claim?">warranty</button>
<button data-q="what is the visitor badge code?">visitor badge</button>
<button data-q="what is the population of Paris?">no-source demo</button>
<button data-q="brief me priority">priority brief</button>
<button data-q="weekly review priority">priority review</button>
<button data-q="due soon">due soon</button>
<button data-q="due soon priority">priority due soon</button>
<button data-q="overdue tasks priority">priority overdue</button>
<button data-q="search tasks report">search tasks</button>
<button data-q="search tasks report sort priority">priority search</button>
<button data-q="find report weekly">find report</button>
<button data-q="list reports">list reports</button>
<button data-q="list reports oldest">reports oldest</button>
<button data-q="verify report archive">verify archive</button>
<button data-q="repair report archive">repair metadata</button>
<button data-q="keep reports 2 dry-run">report dry-run</button>
<button data-q="plan goal prepare the warranty page">plan goal</button>
</div>
<h3>Daily brief</h3>
<p class="hint">A local summary of open tasks, pending gaps, index status and
graph entities. No provider is required. A goal plan adds cited local context
and suggested next actions.</p>
<div class="row"><button id="briefrefresh">Get daily brief</button>
<input id="brieftag" placeholder="tag" style="width:7em">
<button id="briefweekly">Weekly review</button>
<input id="weeklytag" placeholder="tag" style="width:7em">
<select id="weeklyauditaction" title="Weekly tag audit action">
<option value="all" selected>All audit actions</option>
<option value="delete_tag">Delete</option>
<option value="undo_tag_delete">Undo</option>
</select>
<button id="overduebtn">Overdue tasks</button>
<input id="duedays" type="number" min="1" max="90" value="7"
       title="Due-soon window in days" style="width:4em">
<button id="duesoonbtn">Due soon</button>
<input id="duetag" placeholder="tag" style="width:7em">
<button id="weeklymd">Download .md</button>
<input id="tasksearch" placeholder="Search tasks..." style="width:12em">
<select id="tasksearchstatus" title="Task status">
<option value="all" selected>Any status</option>
<option value="open">Open</option>
<option value="done">Done</option>
</select>
<select id="tasksearchdue" title="Task due date">
<option value="all" selected>Any due date</option>
<option value="overdue">Overdue</option>
<option value="today">Today</option>
<option value="week">Next 7 days</option>
<option value="none">No due date</option>
</select>
<select id="tasksearchpriority" title="Task priority">
<option value="all" selected>Any priority</option>
<option value="low">Low</option>
<option value="normal">Normal</option>
<option value="high">High</option>
</select>
<input id="tasksearchtag" placeholder="tag" style="width:7em">
<select id="tasksort" title="Task order">
<option value="default" selected>Session order</option>
<option value="priority">Priority first</option>
</select>
<button id="tasksearchbtn">Search tasks</button>
</div>
<div class="row"><span class="hint">Task list:</span>
<select id="taskliststatus" title="Task list status">
<option value="open" selected>Open</option>
<option value="done">Done</option>
<option value="all">All</option>
</select>
<select id="tasklistdue" title="Task list due date">
<option value="all" selected>Any due date</option>
<option value="overdue">Overdue</option>
<option value="today">Today</option>
<option value="week">Next 7 days</option>
<option value="none">No due date</option>
</select>
<select id="tasklistpriority" title="Task list priority">
<option value="all" selected>Any priority</option>
<option value="low">Low</option>
<option value="normal">Normal</option>
<option value="high">High</option>
</select>
<input id="tasklisttag" placeholder="tag" style="width:7em">
<select id="tasklistsort" title="Task list order">
<option value="default" selected>Session order</option>
<option value="priority">Priority first</option>
</select>
<button id="tasklistbtn">Task list</button>
<button id="tasklistcsv">Download CSV</button>
<button id="tasklistmd">Download .md</button>
<button id="tasktagsbtn">Tags</button>
<button id="tagauditbtn">Tag audit</button>
<input id="tagaudittag" placeholder="tag" style="width:7em">
<select id="tagauditaction" title="Audit action">
<option value="all" selected>All actions</option>
<option value="delete_tag">Delete</option>
<option value="undo_tag_delete">Undo</option>
</select>
<button id="tagauditcsv">Download CSV</button>
<button id="tagauditmd">Download .md</button>
<input id="renameoldtag" placeholder="old tag" style="width:7em">
<input id="renamenewtag" placeholder="new tag" style="width:7em">
<label class="hint"><input id="renametagconfirm" type="checkbox">
Confirm tag rename</label>
<button id="renametagbtn">Rename tag</button>
<input id="deletetag" placeholder="tag" style="width:7em">
<label class="hint"><input id="deletetagconfirm" type="checkbox">
Confirm tag delete</label>
<button id="deletetagbtn">Delete tag</button>
<button id="undotagdeletebtn">Undo tag delete</button>
<input id="rescheduletaskid" type="number" min="1" placeholder="task ID"
       style="width:5em">
<input id="rescheduledue" placeholder="new due date" style="width:9em">
<button id="rescheduletaskbtn">Reschedule</button>
<input id="taskpriorityid" type="number" min="1" placeholder="task ID"
       style="width:5em">
<select id="taskprioritylevel" title="New task priority">
<option value="low">Low</option>
<option value="normal" selected>Normal</option>
<option value="high">High</option>
</select>
<button id="taskprioritybtn">Set priority</button>
<input id="deletetaskid" type="number" min="1" placeholder="task ID"
       style="width:5em">
<label class="hint"><input id="deletetaskconfirm" type="checkbox">
Confirm delete</label>
<button id="deletetaskbtn">Delete task</button>
<input id="goalinput" placeholder="Optional goal, e.g. prepare the warranty page">
<button id="goalplan">Plan goal</button></div>
<p id="briefmsg" class="hint"></p>
<div id="briefres"></div>
<div id="overdueres"></div>
<div id="dueres"></div>
<div id="tasksearchres"></div>
<p id="tasksearchmsg" class="hint"></p>
<p id="tasklistmsg" class="hint"></p>
<div id="tasklistres"></div>
<p id="tasktagsmsg" class="hint"></p>
<div id="tasktagsres"></div>
<p id="tagauditmsg" class="hint"></p>
<div id="tagauditres"></div>
<p id="renametagmsg" class="hint"></p>
<p id="deletetagmsg" class="hint"></p>
<p id="reschedulemsg" class="hint"></p>
<p id="taskprioritymsg" class="hint"></p>
<p id="deletetaskmsg" class="hint"></p>
<div class="row"><input id="weeklyreportsearch" placeholder="Search reports...">
<button id="weeklyreportssearch">Search</button>
<input id="weeklyreportstag" placeholder="tag" style="width:7em">
<select id="weeklyreportsort" title="Report order">
<option value="newest" selected>Newest</option>
<option value="oldest">Oldest</option>
<option value="largest">Largest</option>
</select>
<button id="weeklyreportsrefresh">Recent reports</button>
<button id="weeklyreportsprune">Prune missing</button>
<button id="weeklyreportsverify">Verify files</button>
<button id="weeklyreportsrepair">Repair metadata</button>
<input id="weeklykeep" type="number" min="1" max="500" value="5"
       title="Keep newest N reports" style="width:5em">
<label class="hint"><input id="weeklykeepdry" type="checkbox"> Preview only</label>
<button id="weeklyreportskeep">Keep newest</button></div>
<p id="weeklyreportsmsg" class="hint"></p>
<div id="weeklyreports"></div>
<div id="weeklyreportspreview"></div>
<div id="weeklyreportscheck"></div>
<div class="row"><input id="weeklycompareleft" placeholder="left ID, e.g. w0001">
<input id="weeklycompareright" placeholder="right ID, e.g. w0002">
<button id="weeklyreportscompare">Compare reports</button></div>
<div id="weeklyreportscompareout"></div>
<p class="hint">Chat: <code>show report w0001</code> opens an archived report;
<code>compare reports w0001 w0002</code> shows metadata deltas;
<code>repair report w0001</code> repairs its metadata.</p>
<hr>
<h3>Teach a fact</h3>
<p class="hint">Stored in the local document index with a source. Facts do not
change model weights.</p>
<div class="row"><input id="f" placeholder="The project codename is Blue Lantern.">
<button id="t">Teach fact</button></div>
<p id="s" class="hint"></p>
<hr>
<h3>Add a document</h3>
<p class="hint">Text, Markdown, HTML, PDF or DOCX (max 10 MB). PDFs keep page
numbers; DOCX/HTML/Markdown keep section headings. Stored with source, title
and SHA-256; re-adding the same source replaces it. Nothing is written to
weights.</p>
<div class="row"><input id="src" value="my-note.txt" placeholder="Source (required)">
<input id="ttl" placeholder="Title (optional)"></div>
<textarea id="txt" placeholder="Paste text or choose a file..."></textarea>
<div class="row"><input id="file" type="file"
 accept=".txt,.md,.markdown,.html,.htm,.pdf,.docx,text/plain,text/html,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document">
<button id="add">Add document</button></div>
<p id="ds" class="hint"></p>
<h3>Indexed documents</h3>
<div id="docs"></div>
<div class="row"><input id="docq" placeholder="Search indexed documents...">
<select id="docsrc"><option value="">all sources</option></select>
<button id="docsearch">Search</button>
<button id="docanswerbtn">Answer</button></div>
<div class="row"><input id="docprefix" placeholder="Collection prefix, e.g. notes/">
<button id="docsumcollection">Summarize collection</button></div>
<p id="docmsg" class="hint"></p>
<div id="docanswer"></div>
<div id="docresults"></div>
<div id="evidence"></div>
<div class="row"><select id="cmpsource"><option value="">compare source...</option></select>
<select id="cmpagainst"><option value="">against all others</option></select>
<button id="cmpbtn">Compare sources</button></div>
<div id="cmpresult"></div>
<div class="row"><button id="timelinebtn">Show timeline</button>
<input id="timelineentity" placeholder="Entity filter (optional)">
<span class="hint">Uses the selected source and optional entity.</span></div>
<div id="timeline"></div>
<hr>
<h3>Second brain graph</h3>
<p class="hint">Local co-occurrence graph over indexed chunks. It links
capitalized names, code-like identifiers and years that appear together. This
is a statistical navigation aid, not a trained NER or reasoning model. Click an
entity to see evidence.</p>
<div class="row"><button id="graphrefresh">Refresh graph</button>
<input id="gramentity" placeholder="Entity, e.g. Ankara">
<button id="graphentitybtn">Show evidence</button>
<button id="relationsbtn">Show relations</button></div>
<div class="row"><input id="graphsource" placeholder="From entity">
<input id="graphtarget" placeholder="To entity">
<button id="graphpathbtn">Find path</button></div>
<p id="graphmsg" class="hint"></p>
<div id="graphnodes" class="chips"></div>
<div id="graphedges"></div>
<div id="graphevidence"></div>
<div id="relations"></div>
<hr>
<h3>Delete data</h3>
<p class="hint">Delete an indexed source or this browser session's memory. This
cannot be undone.</p>
<div class="row"><input id="forgetsrc" placeholder="Exact source, e.g. my-note.txt">
<button id="forgetdoc">Forget source</button>
<button id="forgetmem">Forget session memory</button></div>
<p id="fmsg" class="hint"></p>
<hr>
<h3>Session backup</h3>
<p class="hint">Export or replace this session's facts, tasks and transcript.
The file contains personal memory; keep it private.</p>
<div class="row"><button id="mexport">Export session</button>
<input id="mimport" type="file" accept="application/json">
<button id="mimportbtn">Import session</button></div>
<p id="bmsg" class="hint"></p>
<hr>
<h3>Add a teacher answer</h3>
<p class="hint">Paste an answer from ChatGPT or another assistant. It is stored
as a derived, unverified note with its source URL. It is not treated as a fact
and does not change model weights.</p>
<div class="row"><input id="tt" value="chatgpt-web" placeholder="Teacher">
<input id="tm" placeholder="Model (optional)"></div>
<div class="row"><input id="tp" placeholder="Topic/question">
<input id="tu" placeholder="Source URL (optional)"></div>
<div class="row"><input id="tq" placeholder="Question for the configured provider">
<button id="task" disabled>Ask provider</button></div>
<p id="tstatus" class="hint"></p>
<p id="tamsg" class="hint"></p>
<textarea id="ta" placeholder="Paste the teacher answer..."></textarea>
<button id="tnote">Save teacher note</button>
<p id="tmsg" class="hint"></p>
<div class="row"><input id="trev" value="local-reviewer" placeholder="Reviewer">
<button id="trefresh">Refresh teacher notes</button></div>
<p id="tvmsg" class="hint"></p>
<div id="tnotes"></div>
<hr>
<h3>Research queue</h3>
<p class="hint">A topic enters as pending and is not fetched. Approve it, then
run approved topics; only approved items touch Wikipedia.</p>
<div class="row">
<input id="rtopic" placeholder="Topic, e.g. Ankara">
<select id="rlang"><option value="en">en</option><option value="tr">tr</option></select>
<select id="rpages"><option value="1">1 page</option><option value="2" selected>2 pages</option><option value="3">3 pages</option></select>
<button id="radd">Add topic</button>
</div>
<div class="row">
<label>run limit <input id="rlimit" type="number" min="1" max="5" value="1"></label>
<button id="rrun">Run approved</button>
</div>
<p id="rmsg" class="hint"></p>
<div id="rq"></div>
<hr>
<h3>Learning queue</h3>
<p class="hint">Rate chat replies with Good / Bad. Approved examples are exported
for a gated LoRA update; nothing changes weights automatically.</p>
<div class="row"><button id="lrefresh">Refresh</button>
<button id="lexport">Export approved</button></div>
<p id="lmsg" class="hint"></p>
<div id="lq"></div>
<hr>
<h3>Adapter versions</h3>
<p class="hint">Accepted <b>vNNNN</b> versions and the active adapter. Rollback
is explicit CLI: <code>python research/learning_loop.py activate v0001</code>.</p>
<div class="row"><button id="arefresh">Refresh versions</button></div>
<p id="amsg" class="hint"></p>
<div id="aq"></div>
<hr>
<h3>Knowledge gaps</h3>
<p class="hint">When no local source answers a question, it is recorded here.
Propose one gap, or use <b>Propose curiosity topics</b> to create pending
research topics from up to three gaps. Approve them in the research queue
before the worker fetches sources or calls the teacher. Nothing changes
weights automatically.</p>
<div class="row"><button id="grefresh">Refresh gaps</button>
<button id="gcuriosity">Propose curiosity topics</button></div>
<p id="gmsg" class="hint"></p>
<div id="gq"></div>
<hr>
<h3>Hungry mode</h3>
<p class="hint">Opt-in. It checks pending gaps and an optional topic, asks the
configured local teacher only for unknown facts, and stores answers as
unverified notes. It never fetches web pages and never changes weights. Each
run is bounded by a question/provider-call budget.</p>
<div class="row">
<label>questions <input id="hmax" type="number" min="1" max="20" value="3"></label>
<label><input id="hdry" type="checkbox"> dry run</label>
<input id="htopic" placeholder="Optional topic, e.g. Ankara">
<button id="hrun">Run hungry mode</button>
</div>
<p id="hmsg" class="hint"></p>
<div id="hres"></div>
<hr>
<h3>Custom skills</h3>
<p class="hint">Drop JSON skill files into <code>data/skills</code>. A skill
matches trigger phrases, can retrieve local context and can call the configured
provider. Skills are declarative: they cannot execute arbitrary code. Restart
Atom Zero after adding files.</p>
<div class="row"><button id="skrefresh">Refresh skills</button></div>
<p id="skmsg" class="hint"></p>
<div id="sklist"></div>
<div class="row"><input id="skillname" placeholder="Skill name, e.g. daily-brief">
<input id="skillquery" placeholder="Query for the skill">
<button id="skillrun">Run skill</button></div>
<div id="skillresult"></div>
<div class="row"><input id="skillchain" placeholder="Chain skills, e.g. study-tutor,brainstorm">
<button id="skillchainrun">Run chain</button></div>
<div class="row"><select id="chainrecipe"><option value="">named recipe</option></select>
<button id="chainreciperun">Run recipe</button></div>
<div id="chainresult"></div>
</div>
</details>
<script>
function getSessionId() {
  try {
    const saved = localStorage.getItem('atomzero_session');
    if (saved) return saved;
  } catch (e) {}
  const id = 'web-' + Math.random().toString(36).slice(2, 10);
  try { localStorage.setItem('atomzero_session', id); } catch (e) {}
  return id;
}
let sessionId = getSessionId();
document.getElementById('newsession').onclick = () => {
  if (!confirm('Start a new session? The old session memory stays on disk until deleted.')) return;
  try { localStorage.removeItem('atomzero_session'); } catch (e) {}
  location.reload();
};
const chat = document.getElementById('chat');
function add(role, text, extra, isError) {
  const box = document.createElement('div');
  box.className = 'msg ' + role + (isError ? ' err' : '');
  box.textContent = text;
  if (extra) {
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = extra;
    box.appendChild(meta);
  }
  chat.appendChild(box);
  box.scrollIntoView({block: 'end'});
  return box;
}
async function restoreChat() {
  try {
    const r = await fetch('/v1/memory/history', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: sessionId, limit: 50})});
    const j = await r.json();
    if (!r.ok) return;
    (j.data || []).forEach(item => {
      const role = item.role === 'assistant' ? 'bot' : 'user';
      add(role, item.text || '');
    });
  } catch (e) {}
}
function extra(result) {
  const parts = [];
  const tools = result.tool_calls || [];
  if (tools.length) parts.push('tools: ' + tools.map(t => t.name).join(', '));
  const confidence = result.confidence || {};
  if (confidence.label && confidence.label !== 'none') {
    parts.push('match: ' + confidence.label + ' (' + confidence.score + ')');
  }
  const sources = result.sources || [];
  if (sources.length) parts.push('sources: ' + sources.map(s => {
    let status = '';
    if (s.kind === 'teacher') {
      status = s.verified
        ? (s.reviewer ? ' [verified: ' + s.reviewer + ']' : ' [verified]')
        : ' [unverified]';
    }
    return s.title + ' (' + (s.source_url || s.source) + ')' + status;
  }).join('; '));
  return parts.join(' | ');
}
function currentAttach() {
  const input = document.getElementById('pfile');
  return input && input.files ? input.files[0] : null;
}
function fileIntent(text) {
  const low = String(text || '').toLowerCase();
  if (/(öğren|ogren|learn|ekle|kaydet|index)/.test(low)) return 'learn';
  if (/(unut|forget|sil|kaldır|kaldir)/.test(low)) return 'forget';
  return '';
}
function setAttachStatus(text, isError) {
  const el = document.getElementById('attachstatus');
  if (!el) return;
  el.textContent = text;
  el.style.color = isError ? 'var(--danger)' : '';
}
async function learnFile(file) {
  setAttachStatus('Öğreniyorum: ' + file.name + '...');
  const source = file.name;
  const query = new URLSearchParams({source, title: file.name});
  try {
    const r = await fetch('/v1/documents/upload?' + query.toString(), {
      method: 'POST',
      headers: {'Content-Type': 'application/octet-stream'},
      body: file,
    });
    const j = await r.json();
    if (!r.ok) {
      add('bot', 'Dosya öğrenilemedi: ' + (j.error || r.status), '', true);
      setAttachStatus('Dosya öğrenilemedi.', true);
      return;
    }
    add('bot',
      'Öğrendim: ' + (j.source || source) + ' — ' + (j.chunks || 0) +
      ' parça yerel indekse eklendi. Artık bu dosya hakkında soru sorabilirsin.',
      'source: ' + (j.source || source));
    setAttachStatus('Öğrenildi: ' + (j.source || source));
    refreshDocs();
    refreshHealth();
  } catch (e) {
    add('bot', 'Dosya öğrenilemedi: ' + e, '', true);
    setAttachStatus('Dosya öğrenilemedi.', true);
  } finally {
    const input = document.getElementById('pfile');
    if (input) input.value = '';
  }
}
async function forgetFile(file) {
  const source = file.name;
  if (!confirm('Delete indexed source "' + source + '"?')) return;
  setAttachStatus('Unutuyorum: ' + source + '...');
  try {
    const r = await fetch('/v1/documents/forget', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({source, confirm: true}),
    });
    const j = await r.json();
    if (!r.ok) {
      add('bot', 'Dosya unutulamadı: ' + (j.error || r.status), '', true);
      setAttachStatus('Dosya unutulamadı.', true);
      return;
    }
    add('bot',
      'Unuttum: ' + source + ' — ' + (j.chunks_deleted || 0) +
      ' parça ve ' + (j.meta_deleted || 0) + ' metadata satırı silindi.',
      'source: ' + source);
    setAttachStatus('Unutuldu: ' + source);
    refreshDocs();
    refreshHealth();
  } catch (e) {
    add('bot', 'Dosya unutulamadı: ' + e, '', true);
    setAttachStatus('Dosya unutulamadı.', true);
  } finally {
    const input = document.getElementById('pfile');
    if (input) input.value = '';
  }
}
async function send(message) {
  const input = document.getElementById('p');
  const text = (message || input.value).trim();
  const file = currentAttach();
  if (file) {
    const intent = fileIntent(text);
    if (!intent) {
      add('user', text || file.name);
      add('bot',
        'Dosya ekli. Öğrenmek için "bunu öğren", unutmak için ' +
        '"bu bilgileri unut" yaz.', '', false);
      return;
    }
    const label = text || (intent === 'learn' ? 'bunu öğren'
                                               : 'bu bilgileri unut');
    add('user', label + ' [' + file.name + ']');
    if (input) input.value = '';
    if (file.size > 10000000) {
      add('bot', 'Dosya çok büyük (en fazla 10 MB).', '', true);
      setAttachStatus('Dosya çok büyük.', true);
      return;
    }
    if (intent === 'learn') await learnFile(file);
    else await forgetFile(file);
    return;
  }
  if (!text) return;
  if (input) input.value = '';
  add('user', text);
  try {
    const r = await fetch('/v1/agent/chat', {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: sessionId, message: text})});
    const j = await r.json();
    if (!r.ok) return add('bot', 'error: ' + (j.error || r.status), '', true);
    const box = add('bot', j.reply, extra(j));
    addFeedback(box, text, j.reply);
    refreshGaps();
  } catch (e) {
    add('bot', 'error: ' + e, '', true);
  }
}
async function proactivePost(path, payload) {
  const r = await fetch(path, {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)});
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.status);
  return j;
}
async function pollProactive() {
  const toggle = document.getElementById('proactivetoggle');
  if (!toggle) return;
  try {
    const j = await proactivePost('/v1/proactive', {
      session_id: sessionId, ack: false});
    if (j.state && typeof j.state.enabled === 'boolean') {
      toggle.checked = j.state.enabled;
    }
    if (!j.due || !j.message) return;
    const box = add('bot', j.message,
      'proactive | ' + (j.reason || 'local signal'));
    const actions = (j.actions || []).slice(0, 3);
    if (actions.length) {
      const row = document.createElement('div');
      row.className = 'actions';
      actions.forEach(action => {
        const button = document.createElement('button');
        button.textContent = action.label;
        button.onclick = () => send(action.command);
        row.appendChild(button);
      });
      box.appendChild(row);
    }
    await proactivePost('/v1/proactive/ack', {
      session_id: sessionId, proactive_id: j.proactive_id || '',
      action: 'shown'});
  } catch (e) {
    // Proactive polling is optional; do not surface transient errors.
  }
}
document.getElementById('proactivetoggle').onchange = async (event) => {
  const enabled = event.target.checked;
  try {
    await proactivePost('/v1/proactive/config', {
      session_id: sessionId, enabled: enabled});
    if (enabled) await pollProactive();
  } catch (e) {
    event.target.checked = !enabled;
  }
};
document.getElementById('pfile').onchange = () => {
  const file = currentAttach();
  if (!file) {
    setAttachStatus('Attach a file, then type "bunu öğren" or "bu bilgileri unut".');
    return;
  }
  if (file.size > 10000000) {
    setAttachStatus('Dosya çok büyük (en fazla 10 MB).', true);
    return;
  }
  setAttachStatus('Seçildi: ' + file.name +
    ' — şimdi "bunu öğren" veya "bu bilgileri unut" yaz.');
};
document.getElementById('p').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    send();
  }
});
function addFeedback(box, user, assistant) {
  const row = document.createElement('div');
  row.className = 'actions';
  const status = document.createElement('span');
  status.className = 'hint';
  const good = document.createElement('button');
  good.textContent = 'Good';
  good.onclick = () => sendFeedback(user, assistant, 1, '', status);
  const bad = document.createElement('button');
  bad.textContent = 'Bad / correct';
  bad.onclick = () => {
    const corrected = prompt('Correct answer (optional):', '');
    sendFeedback(user, assistant, -1, corrected || '', status);
  };
  row.appendChild(good);
  row.appendChild(bad);
  row.appendChild(status);
  box.appendChild(row);
}
async function sendFeedback(user, assistant, rating, corrected, status) {
  status.textContent = 'saving...';
  try {
    const r = await fetch('/v1/learning/feedback', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({user, assistant, rating, corrected,
                            session_id: sessionId})});
    const j = await r.json();
    status.textContent = r.ok ? 'feedback queued'
                              : ('error: ' + (j.error || r.status));
    if (r.ok) refreshLearning();
  } catch (e) {
    status.textContent = 'error: ' + e;
  }
}
async function getBriefing() {
  const msg = document.getElementById('briefmsg');
  const box = document.getElementById('briefres');
  const sort = document.getElementById('tasksort').value;
  const tag = document.getElementById('brieftag').value.trim();
  msg.textContent = 'building local brief...';
  box.textContent = '';
  try {
    const payload = {session_id: sessionId, sort};
    if (tag) payload.tag = tag;
    const r = await fetch('/v1/briefing', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)});
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'brief error: ' + (j.error || r.status);
      return;
    }
    msg.textContent = 'daily brief | ' + (j.open_task_count || 0) +
      ' open task(s) | ' + (j.pending_gap_count || 0) + ' pending gap(s)' +
      ((j.sort || sort) !== 'default' ? ' | sort=' + (j.sort || sort) : '') +
      ((j.tag || 'all') !== 'all' ? ' | tag=' + j.tag : '');
    const card = document.createElement('div');
    card.className = 'qitem';
    card.textContent = j.text || '';
    box.appendChild(card);
  } catch (e) {
    msg.textContent = 'brief error: ' + e;
  }
}
async function getOverdueTasks() {
  const msg = document.getElementById('briefmsg');
  const box = document.getElementById('overdueres');
  const sort = document.getElementById('tasksort').value;
  const tag = document.getElementById('duetag').value.trim();
  msg.textContent = 'loading overdue tasks...';
  box.textContent = '';
  try {
    let url = '/v1/tasks/overdue?session_id=' +
      encodeURIComponent(sessionId) + '&sort=' + encodeURIComponent(sort);
    if (tag) url += '&tag=' + encodeURIComponent(tag);
    const r = await fetch(url);
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'overdue error: ' + (j.error || r.status);
      return;
    }
    msg.textContent = 'overdue tasks | ' +
      (j.overdue_task_count || 0) + ' of ' +
      (j.open_task_count || 0) + ' open' +
      ((j.sort || sort) !== 'default' ? ' | sort=' + (j.sort || sort) : '') +
      ((j.tag || 'all') !== 'all' ? ' | tag=' + j.tag : '');
    const tasks = j.overdue_tasks || [];
    if (!tasks.length) {
      const none = document.createElement('div');
      none.className = 'hint';
      none.textContent = 'No overdue tasks.';
      box.appendChild(none);
      return;
    }
    tasks.forEach(task => {
      const card = document.createElement('div');
      card.className = 'qitem';
      card.textContent = '#' + task.id + ' ' + (task.text || '') +
        ' | due ' + (task.due || '');
      box.appendChild(card);
    });
  } catch (e) {
    msg.textContent = 'overdue error: ' + e;
  }
}
async function getDueSoonTasks() {
  const msg = document.getElementById('briefmsg');
  const box = document.getElementById('dueres');
  const days = parseInt(document.getElementById('duedays').value, 10);
  const sort = document.getElementById('tasksort').value;
  const tag = document.getElementById('duetag').value.trim();
  if (!Number.isInteger(days) || days < 1) {
    msg.textContent = 'enter a due-soon window of at least 1 day';
    return;
  }
  msg.textContent = 'loading due-soon tasks...';
  box.textContent = '';
  try {
    let url = '/v1/tasks/due-soon?session_id=' +
      encodeURIComponent(sessionId) + '&days=' + days +
      '&sort=' + encodeURIComponent(sort);
    if (tag) url += '&tag=' + encodeURIComponent(tag);
    const r = await fetch(url);
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'due-soon error: ' + (j.error || r.status);
      return;
    }
    msg.textContent = 'due soon | ' + (j.due_soon_count || 0) +
      ' of ' + (j.open_task_count || 0) + ' open | next ' +
      (j.days || days) + ' days' +
      ((j.sort || sort) !== 'default' ? ' | sort=' + (j.sort || sort) : '') +
      ((j.tag || 'all') !== 'all' ? ' | tag=' + j.tag : '');
    const tasks = j.due_soon_tasks || [];
    if (!tasks.length) {
      const none = document.createElement('div');
      none.className = 'hint';
      none.textContent = 'No tasks due soon.';
      box.appendChild(none);
      return;
    }
    tasks.forEach(task => {
      const card = document.createElement('div');
      card.className = 'qitem';
      card.textContent = '#' + task.id + ' ' + (task.text || '') +
        ' | due ' + (task.due || '');
      box.appendChild(card);
    });
  } catch (e) {
    msg.textContent = 'due-soon error: ' + e;
  }
}
async function searchTasks() {
  const msg = document.getElementById('tasksearchmsg');
  const box = document.getElementById('tasksearchres');
  const query = document.getElementById('tasksearch').value.trim();
  const status = document.getElementById('tasksearchstatus').value;
  const due = document.getElementById('tasksearchdue').value;
  const priority = document.getElementById('tasksearchpriority').value;
  const tag = document.getElementById('tasksearchtag').value.trim();
  const sort = document.getElementById('tasksort').value;
  if (!query) {
    msg.textContent = 'enter text to search tasks';
    return;
  }
  msg.textContent = 'searching tasks...';
  box.textContent = '';
  try {
    const params = new URLSearchParams({
      session_id: sessionId, q: query, status, due, priority, sort});
    if (tag) params.set('tag', tag);
    const r = await fetch('/v1/tasks/search?' + params.toString());
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'task search error: ' + (j.error || r.status);
      return;
    }
    const filters = [];
    if ((j.status || status) !== 'all') {
      filters.push('status=' + (j.status || status));
    }
    if ((j.due || due) !== 'all') {
      filters.push('due=' + (j.due || due));
    }
    if ((j.priority || priority) !== 'all') {
      filters.push('priority=' + (j.priority || priority));
    }
    if (j.tag && j.tag !== 'all') {
      filters.push('tag=' + j.tag);
    }
    if ((j.sort || sort) !== 'default') {
      filters.push('sort=' + (j.sort || sort));
    }
    msg.textContent = 'task search "' + (j.query || query) + '" | ' +
      (j.count || 0) + ' match(es)' +
      (filters.length ? ' | ' + filters.join(' | ') : '');
    const tasks = j.tasks || [];
    if (!tasks.length) {
      const none = document.createElement('div');
      none.className = 'hint';
      none.textContent = 'No matching tasks.';
      box.appendChild(none);
      return;
    }
    tasks.forEach(task => {
      const card = document.createElement('div');
      card.className = 'qitem';
      let line = '#' + task.id + ' [' + (task.status || '?') + '] [' +
        (task.priority || 'normal') + '] ' + (task.text || '');
      if (task.due) line += ' | due ' + task.due;
      card.textContent = line;
      box.appendChild(card);
    });
  } catch (e) {
    msg.textContent = 'task search error: ' + e;
  }
}
async function getTaskList() {
  const msg = document.getElementById('tasklistmsg');
  const box = document.getElementById('tasklistres');
  const status = document.getElementById('taskliststatus').value;
  const due = document.getElementById('tasklistdue').value;
  const priority = document.getElementById('tasklistpriority').value;
  const tag = document.getElementById('tasklisttag').value.trim();
  const sort = document.getElementById('tasklistsort').value;
  msg.textContent = 'loading task list...';
  box.textContent = '';
  try {
    const params = new URLSearchParams({
      session_id: sessionId, status, due, priority, sort});
    if (tag) params.set('tag', tag);
    const r = await fetch('/v1/tasks/list?' + params.toString());
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'task list error: ' + (j.error || r.status);
      return;
    }
    const filters = [];
    if (j.status !== 'open') filters.push('status=' + j.status);
    if (j.due !== 'all') filters.push('due=' + j.due);
    if (j.priority !== 'all') filters.push('priority=' + j.priority);
    if (j.tag && j.tag !== 'all') filters.push('tag=' + j.tag);
    if (j.sort !== 'default') filters.push('sort=' + j.sort);
    msg.textContent = 'task list | ' + (j.count || 0) + ' task(s)' +
      (filters.length ? ' | ' + filters.join(' | ') : '');
    const tasks = j.tasks || [];
    if (!tasks.length) {
      const none = document.createElement('div');
      none.className = 'hint';
      none.textContent = 'No tasks match.';
      box.appendChild(none);
      return;
    }
    tasks.forEach(task => {
      const card = document.createElement('div');
      card.className = 'qitem';
      let line = '#' + task.id + ' [' + (task.status || '?') + '] [' +
        (task.priority || 'normal') + '] ' + (task.text || '');
      if (task.tags && task.tags.length) {
        line += ' | tags: ' + task.tags.join(',');
      }
      if (task.due) line += ' | due ' + task.due;
      card.textContent = line;
      box.appendChild(card);
    });
  } catch (e) {
    msg.textContent = 'task list error: ' + e;
  }
}
async function getTaskTags() {
  const msg = document.getElementById('tasktagsmsg');
  const box = document.getElementById('tasktagsres');
  msg.textContent = 'loading task tags...';
  box.textContent = '';
  try {
    const params = new URLSearchParams({session_id: sessionId});
    const r = await fetch('/v1/tasks/tags?' + params.toString());
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'task tags error: ' + (j.error || r.status);
      return;
    }
    msg.textContent = 'task tags | ' + (j.tag_count || 0) + ' tag(s) | ' +
      (j.tagged_task_count || 0) + ' tagged task(s)' +
      ((j.untagged_task_count || 0) ? ' | ' + j.untagged_task_count +
        ' untagged' : '');
    const tags = j.tags || [];
    if (!tags.length) {
      const none = document.createElement('div');
      none.className = 'hint';
      none.textContent = 'No task tags yet.';
      box.appendChild(none);
      return;
    }
    tags.forEach(item => {
      const chip = document.createElement('button');
      chip.className = 'qitem';
      chip.style.display = 'inline-block';
      chip.style.margin = '2px';
      chip.title = 'Filter the task list by #' + item.tag;
      chip.textContent = '#' + item.tag + ' (' + item.open + ' open, ' +
        item.done + ' done)';
      chip.onclick = () => {
        document.getElementById('tasklisttag').value = item.tag;
        getTaskList();
      };
      box.appendChild(chip);
    });
  } catch (e) {
    msg.textContent = 'task tags error: ' + e;
  }
}
async function getTagAudit() {
  const msg = document.getElementById('tagauditmsg');
  const box = document.getElementById('tagauditres');
  const tag = document.getElementById('tagaudittag').value.trim();
  const action = document.getElementById('tagauditaction').value;
  msg.textContent = 'loading tag audit...';
  box.textContent = '';
  try {
    const params = new URLSearchParams({session_id: sessionId, limit: '10'});
    if (tag) params.set('tag', tag);
    if (action !== 'all') params.set('action', action);
    const r = await fetch('/v1/tasks/tags/audit?' + params.toString());
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'tag audit error: ' + (j.error || r.status);
      return;
    }
    const filters = [];
    if ((j.tag || 'all') !== 'all') filters.push('tag=' + j.tag);
    if ((j.action || 'all') !== 'all') filters.push('action=' + j.action);
    msg.textContent = 'tag audit | ' + (j.count || 0) + ' of ' +
      (j.total || 0) + ' event(s)' +
      (filters.length ? ' | ' + filters.join(' | ') : '');
    const events = j.events || [];
    if (!events.length) {
      const none = document.createElement('div');
      none.className = 'hint';
      none.textContent = 'No tag audit events yet.';
      box.appendChild(none);
      return;
    }
    events.forEach(event => {
      const card = document.createElement('div');
      card.className = 'qitem';
      const ids = (event.task_ids || []).map(id => '#' + id).join(', ');
      let line = (event.ts || '') + ' | ';
      if (event.action === 'delete_tag') {
        line += "delete tag '" + event.tag + "' | " +
          (event.updated || 0) + ' task(s) (' + (event.open || 0) +
          ' open, ' + (event.done || 0) + ' done)' +
          (ids ? ' [' + ids + ']' : '');
      } else {
        line += "undo delete tag '" + event.tag + "' | restored " +
          (event.restored || 0) + ' task(s)' +
          (ids ? ' [' + ids + ']' : '') +
          (event.already ? ' | ' + event.already + ' already had it' : '') +
          (event.skipped ? ' | ' + event.skipped + ' skipped' : '');
      }
      card.textContent = line;
      box.appendChild(card);
    });
  } catch (e) {
    msg.textContent = 'tag audit error: ' + e;
  }
}
async function downloadTagAudit(format) {
  const msg = document.getElementById('tagauditmsg');
  const tag = document.getElementById('tagaudittag').value.trim();
  const action = document.getElementById('tagauditaction').value;
  msg.textContent = 'exporting tag audit...';
  try {
    const params = new URLSearchParams({
      session_id: sessionId, limit: '10', format});
    if (tag) params.set('tag', tag);
    if (action !== 'all') params.set('action', action);
    const r = await fetch('/v1/tasks/tags/audit/export?' + params.toString());
    if (!r.ok) {
      let detail = r.status;
      try {
        const j = await r.json();
        detail = j.error || detail;
      } catch (e) {}
      msg.textContent = 'tag audit export error: ' + detail;
      return;
    }
    const text = await r.text();
    const disposition = r.headers.get('Content-Disposition') || '';
    const match = disposition.match(/filename="([^"]+)"/);
    const fallback = 'atomzero-tag-audit.' +
      (format === 'markdown' ? 'md' : 'csv');
    const filename = match ? match[1] : fallback;
    const mime = format === 'markdown'
      ? 'text/markdown;charset=utf-8' : 'text/csv;charset=utf-8';
    const blob = new Blob([text], {type: mime});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    msg.textContent = 'tag audit exported | ' + filename;
  } catch (e) {
    msg.textContent = 'tag audit export error: ' + e;
  }
}
async function renameTag() {
  const msg = document.getElementById('renametagmsg');
  const oldTag = document.getElementById('renameoldtag').value.trim();
  const newTag = document.getElementById('renamenewtag').value.trim();
  const confirmed = document.getElementById('renametagconfirm').checked;
  if (!oldTag || !newTag) {
    msg.textContent = 'enter an old and a new tag';
    return;
  }
  if (!confirmed) {
    msg.textContent = 'check Confirm tag rename first';
    return;
  }
  msg.textContent = 'renaming tag...';
  try {
    const r = await fetch('/v1/tasks/tags/rename', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: sessionId, old: oldTag,
                            new: newTag, confirm: true})});
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'tag rename error: ' + (j.error || r.status);
      return;
    }
    msg.textContent = "renamed tag '" + j.old + "' to '" + j.new +
      "' on " + j.updated + ' task(s)';
    const filter = document.getElementById('tasklisttag');
    let refreshList = false;
    if (filter.value.trim().replace(/^#/, '').toLowerCase() === j.old) {
      filter.value = j.new;
      refreshList = true;
    }
    document.getElementById('renameoldtag').value = '';
    document.getElementById('renamenewtag').value = '';
    document.getElementById('renametagconfirm').checked = false;
    getTaskTags();
    if (refreshList) {
      getTaskList();
    }
  } catch (e) {
    msg.textContent = 'tag rename error: ' + e;
  }
}
async function deleteTag() {
  const msg = document.getElementById('deletetagmsg');
  const tag = document.getElementById('deletetag').value.trim();
  const confirmed = document.getElementById('deletetagconfirm').checked;
  if (!tag) {
    msg.textContent = 'enter a tag to delete';
    return;
  }
  if (!confirmed) {
    msg.textContent = 'check Confirm tag delete first';
    return;
  }
  msg.textContent = 'deleting tag...';
  try {
    const r = await fetch('/v1/tasks/tags/delete', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: sessionId, tag: tag,
                            confirm: true})});
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'tag delete error: ' + (j.error || r.status);
      return;
    }
    msg.textContent = "deleted tag '" + j.tag + "' from " +
      j.updated + ' task(s); undo available';
    document.getElementById('deletetag').value = '';
    document.getElementById('deletetagconfirm').checked = false;
    const listFilter = document.getElementById('tasklisttag');
    let refreshList = false;
    if (listFilter.value.trim().replace(/^#/, '').toLowerCase() === j.tag) {
      listFilter.value = '';
      refreshList = true;
    }
    const searchFilter = document.getElementById('tasksearchtag');
    if (searchFilter.value.trim().replace(/^#/, '').toLowerCase() === j.tag) {
      searchFilter.value = '';
    }
    getTaskTags();
    getTagAudit();
    if (refreshList) {
      getTaskList();
    }
  } catch (e) {
    msg.textContent = 'tag delete error: ' + e;
  }
}
async function undoTagDelete() {
  const msg = document.getElementById('deletetagmsg');
  msg.textContent = 'restoring last tag deletion...';
  try {
    const r = await fetch('/v1/tasks/tags/undo-delete', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: sessionId})});
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'tag undo error: ' + (j.error || r.status);
      return;
    }
    msg.textContent = "restored tag '" + j.tag + "' on " +
      j.restored + ' task(s)' +
      (j.already && j.already.length
        ? '; ' + j.already.length + ' already had it' : '') +
      (j.skipped && j.skipped.length
        ? '; ' + j.skipped.length + ' skipped' : '');
    getTaskTags();
    getTaskList();
    getTagAudit();
  } catch (e) {
    msg.textContent = 'tag undo error: ' + e;
  }
}
async function downloadTaskList(format) {
  const msg = document.getElementById('tasklistmsg');
  const status = document.getElementById('taskliststatus').value;
  const due = document.getElementById('tasklistdue').value;
  const priority = document.getElementById('tasklistpriority').value;
  const tag = document.getElementById('tasklisttag').value.trim();
  const sort = document.getElementById('tasklistsort').value;
  msg.textContent = 'exporting task list...';
  try {
    const params = new URLSearchParams({
      session_id: sessionId, status, due, priority, sort, format});
    if (tag) params.set('tag', tag);
    const r = await fetch('/v1/tasks/export?' + params.toString());
    if (!r.ok) {
      let detail = r.status;
      try {
        const j = await r.json();
        detail = j.error || detail;
      } catch (e) {}
      msg.textContent = 'task export error: ' + detail;
      return;
    }
    const text = await r.text();
    const disposition = r.headers.get('Content-Disposition') || '';
    const match = disposition.match(/filename="([^"]+)"/);
    const fallback = 'atomzero-tasks.' +
      (format === 'markdown' ? 'md' : 'csv');
    const filename = match ? match[1] : fallback;
    const mime = format === 'markdown'
      ? 'text/markdown;charset=utf-8' : 'text/csv;charset=utf-8';
    const blob = new Blob([text], {type: mime});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    msg.textContent = 'downloaded ' + filename;
  } catch (e) {
    msg.textContent = 'task export error: ' + e;
  }
}
async function rescheduleTask() {
  const msg = document.getElementById('reschedulemsg');
  const taskId = parseInt(document.getElementById('rescheduletaskid').value, 10);
  const due = document.getElementById('rescheduledue').value.trim();
  if (!Number.isInteger(taskId) || taskId < 1 || !due) {
    msg.textContent = 'enter a task id and a new due date';
    return;
  }
  msg.textContent = 'rescheduling task...';
  try {
    const r = await fetch('/v1/tasks/reschedule', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: sessionId, id: taskId, due})});
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'reschedule error: ' + (j.error || r.status);
      return;
    }
    msg.textContent = 'task ' + j.task.id + ' rescheduled to ' + j.task.due;
  } catch (e) {
    msg.textContent = 'reschedule error: ' + e;
  }
}
async function setTaskPriority() {
  const msg = document.getElementById('taskprioritymsg');
  const taskId = parseInt(document.getElementById('taskpriorityid').value, 10);
  const priority = document.getElementById('taskprioritylevel').value;
  if (!Number.isInteger(taskId) || taskId < 1) {
    msg.textContent = 'enter a task id';
    return;
  }
  msg.textContent = 'updating task priority...';
  try {
    const r = await fetch('/v1/tasks/priority', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: sessionId, id: taskId,
                            priority: priority})});
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'priority error: ' + (j.error || r.status);
      return;
    }
    msg.textContent = 'task ' + j.task.id + ' priority set to ' +
      j.task.priority;
  } catch (e) {
    msg.textContent = 'priority error: ' + e;
  }
}
async function deleteTask() {
  const msg = document.getElementById('deletetaskmsg');
  const taskId = parseInt(document.getElementById('deletetaskid').value, 10);
  const confirmed = document.getElementById('deletetaskconfirm').checked;
  if (!Number.isInteger(taskId) || taskId < 1) {
    msg.textContent = 'enter a task id to delete';
    return;
  }
  if (!confirmed) {
    msg.textContent = 'check Confirm delete first';
    return;
  }
  msg.textContent = 'deleting task...';
  try {
    const r = await fetch('/v1/tasks/delete', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: sessionId, id: taskId,
                            confirm: true})});
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'delete error: ' + (j.error || r.status);
      return;
    }
    msg.textContent = 'deleted task ' + j.deleted.id + ': ' + j.deleted.text;
  } catch (e) {
    msg.textContent = 'delete error: ' + e;
  }
}
async function getWeeklyReview() {
  const msg = document.getElementById('briefmsg');
  const box = document.getElementById('briefres');
  const sort = document.getElementById('tasksort').value;
  const tag = document.getElementById('weeklytag').value.trim();
  const auditAction = document.getElementById('weeklyauditaction').value;
  msg.textContent = 'building weekly review...';
  box.textContent = '';
  try {
    const payload = {session_id: sessionId, days: 7, sort};
    if (tag) payload.tag = tag;
    if (auditAction !== 'all') payload.audit_action = auditAction;
    const r = await fetch('/v1/weekly-review', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)});
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'weekly review error: ' + (j.error || r.status);
      return;
    }
    const counts = j.task_counts || {};
    msg.textContent = 'weekly review | ' + (counts.open || 0) +
      ' open | ' + (counts.completed_recent || 0) + ' completed | ' +
      (j.recent_source_count || 0) + ' new source(s)' +
      ((j.sort || sort) !== 'default' ? ' | sort=' + (j.sort || sort) : '') +
      ((j.tag || 'all') !== 'all' ? ' | tag=' + j.tag : '') +
      ((j.tag_audit_action || 'all') !== 'all'
        ? ' | audit=' + j.tag_audit_action : '');
    const card = document.createElement('div');
    card.className = 'qitem';
    card.textContent = j.text || '';
    box.appendChild(card);
  } catch (e) {
    msg.textContent = 'weekly review error: ' + e;
  }
}
async function downloadWeeklyMarkdown() {
  const msg = document.getElementById('briefmsg');
  msg.textContent = 'exporting weekly review...';
  try {
    const sort = document.getElementById('tasksort').value;
    const tag = document.getElementById('weeklytag').value.trim();
    const auditAction = document.getElementById('weeklyauditaction').value;
    const query = new URLSearchParams({session_id: sessionId, days: 7, sort});
    if (tag) query.set('tag', tag);
    if (auditAction !== 'all') query.set('audit_action', auditAction);
    const r = await fetch('/v1/weekly-review/export?' + query.toString());
    if (!r.ok) {
      let detail = r.status;
      try {
        const j = await r.json();
        detail = j.error || detail;
      } catch (e) {}
      msg.textContent = 'markdown export error: ' + detail;
      return;
    }
    const text = await r.text();
    const disposition = r.headers.get('Content-Disposition') || '';
    const match = disposition.match(/filename="([^"]+)"/);
    const filename = match ? match[1] : 'atomzero-weekly-review.md';
    const blob = new Blob([text], {type: 'text/markdown;charset=utf-8'});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    msg.textContent = 'weekly review exported | ' + filename;
  } catch (e) {
    msg.textContent = 'markdown export error: ' + e;
  }
}
async function refreshWeeklyReports() {
  const msg = document.getElementById('weeklyreportsmsg');
  const box = document.getElementById('weeklyreports');
  const previewBox = document.getElementById('weeklyreportspreview');
  const checkBox = document.getElementById('weeklyreportscheck');
  msg.textContent = 'loading archived reports...';
  box.textContent = '';
  if (previewBox) previewBox.textContent = '';
  if (checkBox) checkBox.textContent = '';
  const query = (document.getElementById('weeklyreportsearch').value || '')
    .trim();
  const tag = (document.getElementById('weeklyreportstag').value || '')
    .trim();
  const sort = document.getElementById('weeklyreportsort').value || 'newest';
  try {
    const params = new URLSearchParams({limit: '10', sort});
    if (query) params.set('query', query);
    if (tag) params.set('tag', tag);
    const r = await fetch('/v1/weekly-review/reports?' + params.toString());
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'report archive error: ' + (j.error || r.status);
      return;
    }
    const items = j.data || [];
    msg.textContent = (query || tag)
      ? ((j.matched || 0) + ' match(es) of ' + (j.total || 0))
      : ((j.total || items.length) + ' archived report(s)');
    if (!items.length) {
      const none = document.createElement('div');
      none.className = 'hint';
      none.textContent = query
        ? ('No archived reports match "' + query + '".')
        : (tag
          ? ("No archived reports tagged '" + tag + "'.")
          : ('No archived reports yet. Use atomzero-brief ' +
             '--weekly --markdown --out PATH.'));
      box.appendChild(none);
      return;
    }
    items.forEach(item => {
      const card = document.createElement('div');
      card.className = 'qitem';
      const head = document.createElement('div');
      head.textContent = item.id + ' | ' + (item.exported_at || '') +
        ' | last ' + (item.period_days || 7) + ' days' +
        (item.tag && item.tag !== 'all' ? ' | tag=' + item.tag : '') +
        (item.exists === false ? ' [missing]' : '');
      card.appendChild(head);
      const meta = document.createElement('div');
      meta.className = 'src';
      meta.textContent = (item.path || '') + ' | ' +
        ((item.task_counts || {}).open || 0) + ' open | ' +
        (item.recent_source_count || 0) + ' source(s)';
      card.appendChild(meta);
      const actions = document.createElement('div');
      actions.className = 'actions';
      if (item.exists !== false) {
        actions.appendChild(qbutton('Download', () =>
          downloadArchivedReport(item.id, item.name)));
      }
      actions.appendChild(qbutton('Delete', () =>
        deleteArchivedReport(item.id)));
      card.appendChild(actions);
      box.appendChild(card);
    });
  } catch (e) {
    msg.textContent = 'report archive error: ' + e;
  }
}
async function keepWeeklyReports() {
  const msg = document.getElementById('weeklyreportsmsg');
  const keep = parseInt(document.getElementById('weeklykeep').value, 10);
  const dryRun = document.getElementById('weeklykeepdry').checked;
  if (!Number.isInteger(keep) || keep < 1) {
    msg.textContent = 'enter a keep count of at least 1';
    return;
  }
  msg.textContent = dryRun ? 'previewing retention policy...'
                           : 'applying retention policy...';
  try {
    const r = await fetch('/v1/weekly-review/reports/keep', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({keep, dry_run: dryRun})});
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'report retention error: ' + (j.error || r.status);
      return;
    }
    const previewBox = document.getElementById('weeklyreportspreview');
    if (j.dry_run) {
      previewBox.textContent = '';
      const head = document.createElement('div');
      head.className = 'hint';
      head.textContent = 'Would remove:';
      previewBox.appendChild(head);
      (j.would_remove || []).forEach(item => {
        const card = document.createElement('div');
        card.className = 'qitem';
        card.textContent = (item.id || '?') + ' | ' +
          (item.exported_at || '') + ' | ' + (item.path || '');
        previewBox.appendChild(card);
      });
      if (!(j.would_remove || []).length) {
        const none = document.createElement('div');
        none.className = 'hint';
        none.textContent = 'Nothing would be removed.';
        previewBox.appendChild(none);
      }
      msg.textContent = 'dry run: would remove ' +
        (j.would_remove_count || 0) + ' older | ' +
        (j.remaining || 0) + ' remain';
      return;
    }
    const trimmed = j.trimmed || 0;
    await refreshWeeklyReports();
    msg.textContent = 'kept ' + (j.remaining || 0) +
      ' newest report(s) | removed ' + trimmed + ' older';
  } catch (e) {
    msg.textContent = 'report retention error: ' + e;
  }
}
async function compareWeeklyReports() {
  const msg = document.getElementById('weeklyreportsmsg');
  const box = document.getElementById('weeklyreportscompareout');
  const left = document.getElementById('weeklycompareleft').value.trim();
  const right = document.getElementById('weeklycompareright').value.trim();
  if (!left || !right) {
    msg.textContent = 'enter two report ids to compare';
    return;
  }
  msg.textContent = 'comparing reports...';
  box.textContent = '';
  try {
    const r = await fetch('/v1/weekly-review/reports/compare?left=' +
      encodeURIComponent(left) + '&right=' + encodeURIComponent(right));
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'report compare error: ' + (j.error || r.status);
      return;
    }
    msg.textContent = 'report comparison | ' + left + ' -> ' + right;
    const head = document.createElement('div');
    head.className = 'hint';
    head.textContent = 'SHA-256: ' +
      (j.hash_equal ? 'equal' : 'different');
    box.appendChild(head);
    (j.metrics || []).forEach(metric => {
      const row = document.createElement('div');
      row.className = 'qitem';
      const delta = metric.delta > 0 ? '+' + metric.delta : metric.delta;
      row.textContent = metric.label + ': ' + metric.left + ' -> ' +
        metric.right + ' (' + delta + ')';
      box.appendChild(row);
    });
    const actions = document.createElement('div');
    actions.className = 'actions';
    actions.appendChild(qbutton('Download .md', () =>
      downloadComparisonMarkdown(left, right)));
    actions.appendChild(qbutton('Copy Markdown', () =>
      copyComparisonMarkdown(left, right)));
    box.appendChild(actions);
  } catch (e) {
    msg.textContent = 'report compare error: ' + e;
  }
}
async function comparisonMarkdown(left, right) {
  const r = await fetch('/v1/weekly-review/reports/compare/export?left=' +
    encodeURIComponent(left) + '&right=' + encodeURIComponent(right));
  if (!r.ok) throw new Error(String(r.status));
  return await r.text();
}
async function downloadComparisonMarkdown(left, right) {
  const msg = document.getElementById('weeklyreportsmsg');
  try {
    const text = await comparisonMarkdown(left, right);
    const blob = new Blob([text], {type: 'text/markdown;charset=utf-8'});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'atomzero-report-comparison-' + left + '-' + right + '.md';
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    msg.textContent = 'comparison markdown downloaded | ' + link.download;
  } catch (e) {
    msg.textContent = 'comparison export error: ' + e;
  }
}
async function copyComparisonMarkdown(left, right) {
  const msg = document.getElementById('weeklyreportsmsg');
  try {
    const text = await comparisonMarkdown(left, right);
    let copied = false;
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
        copied = true;
      }
    } catch (e) {}
    if (!copied) {
      const area = document.createElement('textarea');
      area.value = text;
      area.setAttribute('readonly', '');
      area.style.position = 'fixed';
      area.style.opacity = '0';
      document.body.appendChild(area);
      area.select();
      copied = document.execCommand('copy');
      area.remove();
    }
    if (copied) {
      msg.textContent = 'comparison markdown copied | ' +
        left + ' -> ' + right;
      return;
    }
    const box = document.getElementById('weeklyreportscompareout');
    const area = document.createElement('textarea');
    area.value = text;
    area.className = 'src';
    area.style.width = '100%';
    area.style.minHeight = '8em';
    box.appendChild(area);
    msg.textContent = 'clipboard unavailable; select the text or use Download .md';
  } catch (e) {
    msg.textContent = 'comparison copy error: ' + e;
  }
}
async function verifyWeeklyReports() {
  const msg = document.getElementById('weeklyreportsmsg');
  const box = document.getElementById('weeklyreportscheck');
  msg.textContent = 'verifying archived report files...';
  box.textContent = '';
  try {
    const r = await fetch('/v1/weekly-review/reports/verify');
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'report verify error: ' + (j.error || r.status);
      return;
    }
    const summary = j.summary || {};
    msg.textContent = 'integrity: ' + (summary.ok || 0) + ' ok | ' +
      (summary.failed || 0) + ' failed | ' +
      (summary.missing || 0) + ' missing';
    const failures = (j.entries || []).filter(item => item.status !== 'ok');
    if (!failures.length) {
      const ok = document.createElement('div');
      ok.className = 'hint';
      ok.textContent = 'All reports match stored size and SHA-256.';
      box.appendChild(ok);
      return;
    }
    failures.forEach(item => {
      const card = document.createElement('div');
      card.className = 'qitem';
      card.textContent = (item.id || '?') + ' | ' + (item.status || '?') +
        ' | ' + (item.path || '') + ' | ' +
        (item.problems || []).join(', ');
      box.appendChild(card);
    });
  } catch (e) {
    msg.textContent = 'report verify error: ' + e;
  }
}
async function repairWeeklyReports() {
  const msg = document.getElementById('weeklyreportsmsg');
  msg.textContent = 'repairing report metadata...';
  try {
    const r = await fetch('/v1/weekly-review/reports/repair', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: '{}'});
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'report repair error: ' + (j.error || r.status);
      return;
    }
    await refreshWeeklyReports();
    const summary = j.summary || {};
    msg.textContent = 'repair: ' + (summary.repaired || 0) + ' repaired | ' +
      (summary.ok || 0) + ' unchanged | ' + (summary.missing || 0) +
      ' missing';
    const box = document.getElementById('weeklyreportscheck');
    box.textContent = '';
    (j.entries || []).filter(item => item.status !== 'ok').forEach(item => {
      const card = document.createElement('div');
      card.className = 'qitem';
      card.textContent = (item.id || '?') + ' | ' + (item.status || '?') +
        ' | ' + (item.path || '') +
        (item.status === 'repaired'
          ? ' | ' + item.old_size_bytes + ' -> ' +
            item.actual_size_bytes + ' bytes'
          : '');
      box.appendChild(card);
    });
  } catch (e) {
    msg.textContent = 'report repair error: ' + e;
  }
}
async function pruneWeeklyReports() {
  const msg = document.getElementById('weeklyreportsmsg');
  msg.textContent = 'pruning missing reports...';
  try {
    const r = await fetch('/v1/weekly-review/reports/prune', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: '{}'});
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'report prune error: ' + (j.error || r.status);
      return;
    }
    const removed = j.removed || 0;
    await refreshWeeklyReports();
    msg.textContent = 'pruned ' + removed + ' missing report(s) | ' +
      (j.remaining || 0) + ' remaining';
  } catch (e) {
    msg.textContent = 'report prune error: ' + e;
  }
}
async function downloadArchivedReport(reportId, name) {
  const msg = document.getElementById('weeklyreportsmsg');
  msg.textContent = 'downloading ' + reportId + '...';
  try {
    const r = await fetch('/v1/weekly-review/reports?id=' +
      encodeURIComponent(reportId));
    if (!r.ok) {
      let detail = r.status;
      try {
        const j = await r.json();
        detail = j.error || detail;
      } catch (e) {}
      msg.textContent = 'report download error: ' + detail;
      return;
    }
    const text = await r.text();
    const filename = name || ('weekly-report-' + reportId + '.md');
    const blob = new Blob([text], {type: 'text/markdown;charset=utf-8'});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    msg.textContent = 'downloaded ' + filename;
  } catch (e) {
    msg.textContent = 'report download error: ' + e;
  }
}
async function deleteArchivedReport(reportId) {
  const msg = document.getElementById('weeklyreportsmsg');
  msg.textContent = 'deleting ' + reportId + ' from archive...';
  try {
    const r = await fetch('/v1/weekly-review/reports/delete', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({id: reportId})});
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'report delete error: ' + (j.error || r.status);
      return;
    }
    await refreshWeeklyReports();
    msg.textContent = 'deleted ' + reportId + ' from archive | ' +
      (j.remaining || 0) + ' remaining';
  } catch (e) {
    msg.textContent = 'report delete error: ' + e;
  }
}
async function planGoal(confirmCreate) {
  const goal = document.getElementById('goalinput').value.trim();
  const msg = document.getElementById('briefmsg');
  const box = document.getElementById('briefres');
  if (!goal) { msg.textContent = 'enter a goal first'; return; }
  msg.textContent = confirmCreate ? 'creating suggested tasks...'
                                  : 'planning from local evidence...';
  box.textContent = '';
  try {
    const r = await fetch('/v1/goals/plan', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        session_id: sessionId, goal,
        confirm: confirmCreate === true, max_steps: 3})});
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'goal error: ' + (j.error || r.status);
      return;
    }
    msg.textContent = 'goal plan | confidence: ' +
      ((j.confidence && j.confidence.label) || 'none');
    const card = document.createElement('div');
    card.className = 'qitem';
    card.textContent = j.text || '';
    box.appendChild(card);
    (j.suggested_tasks || []).forEach(task => {
      const item = document.createElement('div');
      item.className = 'qitem';
      item.textContent = 'suggested: ' + task;
      box.appendChild(item);
    });
    (j.context || []).forEach(item => {
      const evidence = document.createElement('div');
      evidence.className = 'qitem';
      const title = document.createElement('div');
      title.textContent = (item.title || item.source) + ' | ' + item.source +
        (item.page ? ' | p. ' + item.page : '') +
        (item.section ? ' | ' + item.section : '');
      evidence.appendChild(title);
      const snippet = document.createElement('div');
      snippet.className = 'src';
      snippet.textContent = item.snippet || '';
      evidence.appendChild(snippet);
      const actions = document.createElement('div');
      actions.className = 'actions';
      actions.appendChild(qbutton('Evidence', () =>
        showEvidence(item.source, item.page, item.section)));
      actions.appendChild(qbutton('Copy citation', () =>
        copyCitation(item.source, item.page, item.section)));
      evidence.appendChild(actions);
      box.appendChild(evidence);
    });
    if ((j.created_tasks || []).length) {
      const created = document.createElement('div');
      created.className = 'hint';
      created.textContent = 'created tasks: ' +
        j.created_tasks.map(task => task.id + ' ' + task.text).join('; ');
      box.appendChild(created);
    } else if (!confirmCreate && (j.suggested_tasks || []).length) {
      const actions = document.createElement('div');
      actions.className = 'actions';
      actions.appendChild(qbutton('Create suggested tasks', () => planGoal(true)));
      box.appendChild(actions);
    }
  } catch (e) {
    msg.textContent = 'goal error: ' + e;
  }
}
document.getElementById('briefrefresh').onclick = () => getBriefing();
document.getElementById('briefweekly').onclick = () => getWeeklyReview();
document.getElementById('overduebtn').onclick = () => getOverdueTasks();
document.getElementById('duesoonbtn').onclick = () => getDueSoonTasks();
document.getElementById('tasksearchbtn').onclick = () => searchTasks();
document.getElementById('tasklistbtn').onclick = () => getTaskList();
document.getElementById('tasktagsbtn').onclick = () => getTaskTags();
document.getElementById('tagauditbtn').onclick = () => getTagAudit();
document.getElementById('tagauditcsv').onclick = () => downloadTagAudit('csv');
document.getElementById('tagauditmd').onclick = () => downloadTagAudit('markdown');
document.getElementById('renametagbtn').onclick = () => renameTag();
document.getElementById('deletetagbtn').onclick = () => deleteTag();
document.getElementById('undotagdeletebtn').onclick = () => undoTagDelete();
document.getElementById('tasklistcsv').onclick = () => downloadTaskList('csv');
document.getElementById('tasklistmd').onclick = () => downloadTaskList('markdown');
document.getElementById('rescheduletaskbtn').onclick = () => rescheduleTask();
document.getElementById('taskprioritybtn').onclick = () => setTaskPriority();
document.getElementById('deletetaskbtn').onclick = () => deleteTask();
document.getElementById('weeklymd').onclick = () => downloadWeeklyMarkdown();
document.getElementById('weeklyreportssearch').onclick =
  () => refreshWeeklyReports();
document.getElementById('weeklyreportsort').onchange =
  () => refreshWeeklyReports();
document.getElementById('weeklyreportsearch').addEventListener(
  'keydown', event => {
    if (event.key === 'Enter') refreshWeeklyReports();
  });
document.getElementById('weeklyreportsrefresh').onclick = () => {
  document.getElementById('weeklyreportsearch').value = '';
  refreshWeeklyReports();
};
document.getElementById('weeklyreportsprune').onclick =
  () => pruneWeeklyReports();
document.getElementById('weeklyreportscompare').onclick =
  () => compareWeeklyReports();
document.getElementById('weeklyreportsverify').onclick =
  () => verifyWeeklyReports();
document.getElementById('weeklyreportsrepair').onclick =
  () => repairWeeklyReports();
document.getElementById('weeklyreportskeep').onclick =
  () => keepWeeklyReports();
document.getElementById('goalplan').onclick = () => planGoal(false);
document.getElementById('goalinput').addEventListener('keydown', event => {
  if (event.key === 'Enter') planGoal(false);
});
document.getElementById('g').onclick = () => send();
document.getElementById('p').addEventListener('keydown', e => {
  if (e.key === 'Enter') send();
});
document.querySelectorAll('.chips button').forEach(b => {
  b.onclick = () => send(b.dataset.q);
});
document.getElementById('t').onclick = async () => {
  const fact = document.getElementById('f').value.trim();
  if (!fact) return;
  const out = document.getElementById('s');
  try {
    const r = await fetch('/v1/teach', {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({fact})});
    const j = await r.json();
    out.textContent = r.ok ? ('Saved ' + j.fact_chunks + ' chunk(s) to local memory.')
                           : ('error: ' + (j.error || r.status));
    if (r.ok) {
      document.getElementById('f').value = '';
      refreshHealth();
    }
  } catch (e) { out.textContent = 'error: ' + e; }
};
let selectedFile = null;
document.getElementById('txt').addEventListener('input', () => {
  selectedFile = null;
});
document.getElementById('file').onchange = () => {
  const file = document.getElementById('file').files[0];
  const out = document.getElementById('ds');
  selectedFile = file || null;
  if (!file) return;
  if (file.size > 10000000) {
    out.textContent = 'file is too large (max 10 MB)';
    selectedFile = null;
    return;
  }
  const src = document.getElementById('src');
  if (!src.value.trim() || src.value === 'my-note.txt') src.value = file.name;
  const title = document.getElementById('ttl');
  if (!title.value.trim()) title.value = file.name;
  const textLike = /\\.(txt|md|markdown|html?)$/i.test(file.name);
  if (textLike) {
    const reader = new FileReader();
    reader.onload = () => {
      document.getElementById('txt').value = reader.result;
      out.textContent = 'Selected ' + file.name + ' - click Add document.';
    };
    reader.readAsText(file);
  } else {
    document.getElementById('txt').value = '';
    out.textContent = 'Selected ' + file.name + ' - click Add document.';
  }
};
document.getElementById('add').onclick = async () => {
  const source = document.getElementById('src').value.trim();
  const text = document.getElementById('txt').value;
  const out = document.getElementById('ds');
  if (!source) {
    out.textContent = 'source is required';
    return;
  }
  if (!selectedFile && !text.trim()) {
    out.textContent = 'source and text (or a file) are required';
    return;
  }
  try {
    let r;
    if (selectedFile) {
      const query = new URLSearchParams({
        source, title: document.getElementById('ttl').value});
      r = await fetch('/v1/documents/upload?' + query.toString(), {
        method: 'POST',
        headers: {'Content-Type': 'application/octet-stream'},
        body: selectedFile});
    } else {
      r = await fetch('/v1/documents', {method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({source,
                              title: document.getElementById('ttl').value,
                              text})});
    }
    const j = await r.json();
    out.textContent = r.ok
      ? ('Indexed ' + j.chunks + ' chunk(s) from ' + j.source +
         (j.format ? ' [' + j.format + ']' : '') +
         ' | sha256 ' + j.sha256.slice(0, 12) + '...')
      : ('error: ' + (j.error || r.status));
    if (r.ok) {
      document.getElementById('txt').value = '';
      document.getElementById('file').value = '';
      selectedFile = null;
      refreshHealth();
      refreshDocs();
    }
  } catch (e) { out.textContent = 'error: ' + e; }
};
async function refreshDocs() {
  const box = document.getElementById('docs');
  if (!box) return;
  try {
    const r = await fetch('/v1/documents');
    const j = await r.json();
    box.textContent = '';
    if (!r.ok) {
      const error = document.createElement('div');
      error.className = 'hint';
      error.textContent = 'document error: ' + (j.error || r.status);
      box.appendChild(error);
      return;
    }
    const data = j.data || [];
    const select = document.getElementById('docsrc');
    if (select) {
      const current = select.value;
      select.textContent = '';
      const all = document.createElement('option');
      all.value = '';
      all.textContent = 'all sources';
      select.appendChild(all);
      data.forEach(item => {
        const option = document.createElement('option');
        option.value = item.source;
        option.textContent = item.title || item.source;
        select.appendChild(option);
      });
      if ([...select.options].some(option => option.value === current)) {
        select.value = current;
      }
    }
    ['cmpsource', 'cmpagainst'].forEach(id => {
      const select = document.getElementById(id);
      if (!select) return;
      const current = select.value;
      select.textContent = '';
      const empty = document.createElement('option');
      empty.value = '';
      empty.textContent = id === 'cmpsource'
        ? 'compare source...' : 'against all others';
      select.appendChild(empty);
      data.forEach(item => {
        const option = document.createElement('option');
        option.value = item.source;
        option.textContent = item.title || item.source;
        select.appendChild(option);
      });
      if ([...select.options].some(option => option.value === current)) {
        select.value = current;
      }
    });
    if (!data.length) {
      const empty = document.createElement('div');
      empty.className = 'hint';
      empty.textContent = 'No indexed documents.';
      box.appendChild(empty);
      return;
    }
    data.forEach(item => {
      const card = document.createElement('div');
      card.className = 'qitem';
      const title = document.createElement('div');
      const pageInfo = item.page_start
        ? ' | pp. ' + item.page_start + '-' + item.page_end : '';
      title.textContent = (item.title || item.source) + ' | ' +
                          (item.chunks || 0) + ' chunk(s)' + pageInfo;
      card.appendChild(title);
      const source = document.createElement('div');
      source.className = 'src';
      source.textContent = item.source + (item.kind && item.kind !== 'document'
        ? ' [' + item.kind + ']' : '');
      card.appendChild(source);
      const actions = document.createElement('div');
      actions.className = 'actions';
      actions.appendChild(qbutton('Ask', () => {
        const select = document.getElementById('docsrc');
        if (select) select.value = item.source;
        document.getElementById('docq').focus();
      }));
      actions.appendChild(qbutton('Summary', () => showSummary(item.source)));
      actions.appendChild(qbutton('Evidence', () => showEvidence(item.source)));
      actions.appendChild(qbutton('Copy citation', () =>
        copyCitation(item.source)));
      actions.appendChild(qbutton('Forget', () => forgetSource(item.source)));
      card.appendChild(actions);
      box.appendChild(card);
    });
  } catch (e) {
    box.textContent = 'document error: ' + e;
  }
}
async function searchDocs() {
  const query = document.getElementById('docq').value.trim();
  const source = document.getElementById('docsrc').value.trim();
  const prefix = document.getElementById('docprefix').value.trim();
  const scope = source ? {source} : (prefix ? {source_prefix: prefix} : {});
  const msg = document.getElementById('docmsg');
  const box = document.getElementById('docresults');
  box.textContent = '';
  document.getElementById('evidence').textContent = '';
  if (!query) { msg.textContent = 'enter a search query'; return; }
  msg.textContent = 'searching...';
  try {
    const r = await fetch('/v1/documents/search', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query, k: 5, diverse: true, ...scope})});
    const j = await r.json();
    if (!r.ok) { msg.textContent = 'error: ' + (j.error || r.status); return; }
    const hits = j.data || [];
    msg.textContent = hits.length + ' hit(s)' +
      (source ? ' | source: ' + source :
        (prefix ? ' | prefix: ' + prefix : ''));
    hits.forEach(hit => {
      const card = document.createElement('div');
      card.className = 'qitem';
      const title = document.createElement('div');
      const where = (hit.page ? ' | p. ' + hit.page : '') +
                    (hit.section ? ' | ' + hit.section : '');
      title.textContent = (hit.title || hit.source) + ' | ' + hit.source +
                          where;
      card.appendChild(title);
      const snippet = document.createElement('div');
      snippet.className = 'src';
      snippet.textContent = (hit.chunk || '').slice(0, 400);
      card.appendChild(snippet);
      const actions = document.createElement('div');
      actions.className = 'actions';
      actions.appendChild(qbutton('Evidence', () =>
        showEvidence(hit.source, hit.page, hit.section)));
      actions.appendChild(qbutton('Copy citation', () =>
        copyCitation(hit.source, hit.page, hit.section)));
      card.appendChild(actions);
      box.appendChild(card);
    });
  } catch (e) { msg.textContent = 'error: ' + e; }
}
async function answerDocs() {
  const query = document.getElementById('docq').value.trim();
  const source = document.getElementById('docsrc').value.trim();
  const prefix = document.getElementById('docprefix').value.trim();
  const scope = source ? {source} : (prefix ? {source_prefix: prefix} : {});
  const msg = document.getElementById('docmsg');
  const box = document.getElementById('docanswer');
  box.textContent = '';
  if (!query) { msg.textContent = 'enter a question'; return; }
  msg.textContent = 'retrieving evidence...';
  try {
    const r = await fetch('/v1/documents/answer', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query, ...scope})});
    const j = await r.json();
    if (!r.ok) { msg.textContent = 'error: ' + (j.error || r.status); return; }
    const conf = j.confidence || {};
    msg.textContent = 'grounded answer | confidence: ' +
      (conf.label || 'none') + ' (' + (conf.score ?? 0) + ')' +
      (source ? ' | source: ' + source :
        (prefix ? ' | prefix: ' + prefix : ''));
    const answer = document.createElement('div');
    answer.className = 'qitem';
    answer.textContent = j.text || '';
    box.appendChild(answer);
    (j.evidence || []).forEach(item => {
      const card = document.createElement('div');
      card.className = 'qitem';
      const title = document.createElement('div');
      title.textContent = (item.title || item.source) + ' | ' + item.source +
        (item.page ? ' | p. ' + item.page : '') +
        (item.section ? ' | ' + item.section : '') +
        ' | score ' + (item.score ?? 0);
      card.appendChild(title);
      const snippet = document.createElement('div');
      snippet.className = 'src';
      snippet.textContent = item.snippet || '';
      card.appendChild(snippet);
      const actions = document.createElement('div');
      actions.className = 'actions';
      actions.appendChild(qbutton('Evidence', () =>
        showEvidence(item.source, item.page, item.section)));
      actions.appendChild(qbutton('Copy citation', () =>
        copyCitation(item.source, item.page, item.section)));
      card.appendChild(actions);
      box.appendChild(card);
    });
  } catch (e) { msg.textContent = 'error: ' + e; }
}
async function compareSources() {
  const source = document.getElementById('cmpsource').value;
  const against = document.getElementById('cmpagainst').value;
  const msg = document.getElementById('docmsg');
  const box = document.getElementById('cmpresult');
  box.textContent = '';
  if (!source) {
    msg.textContent = 'choose a source to compare';
    return;
  }
  msg.textContent = 'comparing sources...';
  try {
    const query = new URLSearchParams({source, against});
    const r = await fetch('/v1/documents/compare?' + query.toString());
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'compare error: ' + (j.error || r.status);
      return;
    }
    msg.textContent = (j.summary?.conflict_count || 0) +
      ' conflict hint(s) | ' + (j.summary?.sources_compared || 0) +
      ' source(s)';
    (j.conflicts || []).forEach(conflict => {
      const card = document.createElement('div');
      card.className = 'qitem';
      const title = document.createElement('div');
      title.textContent = (conflict.reasons || []).join(', ') +
        ' | shared: ' + (conflict.shared_terms || []).join(', ');
      card.appendChild(title);
      [conflict.left, conflict.right].forEach(side => {
        const row = document.createElement('div');
        row.className = 'src';
        row.textContent = (side.title || side.source) + ' | ' +
          side.source + ': ' + (side.text || '');
        card.appendChild(row);
        const actions = document.createElement('div');
        actions.className = 'actions';
        actions.appendChild(qbutton('Evidence', () =>
          showEvidence(side.source, side.page, side.section)));
        actions.appendChild(qbutton('Copy citation', () =>
          copyCitation(side.source, side.page, side.section)));
        card.appendChild(actions);
      });
      box.appendChild(card);
    });
  } catch (e) {
    msg.textContent = 'compare error: ' + e;
  }
}
document.getElementById('cmpbtn').onclick = () => compareSources();
async function showTimeline() {
  const box = document.getElementById('timeline');
  const msg = document.getElementById('docmsg');
  const source = document.getElementById('docsrc').value;
  box.textContent = 'loading timeline...';
  const params = new URLSearchParams();
  if (source) params.set('source', source);
  const entity = document.getElementById('timelineentity').value.trim();
  if (entity) params.set('entity', entity);
  try {
    const r = await fetch('/v1/documents/timeline?' + params.toString());
    const j = await r.json();
    if (!r.ok) {
      box.textContent = 'timeline error: ' + (j.error || r.status);
      return;
    }
    box.textContent = '';
    const head = document.createElement('p');
    head.textContent = 'Timeline (' + j.count + ' event(s))';
    box.appendChild(head);
    (j.events || []).forEach(event => {
      const card = document.createElement('div');
      card.className = 'qitem';
      const title = document.createElement('div');
      title.textContent = event.date + ' (' + event.precision + ') | ' +
        event.source + (event.page ? ' p.' + event.page : '') +
        (event.section ? ' [' + event.section + ']' : '');
      card.appendChild(title);
      const text = document.createElement('div');
      text.className = 'src';
      text.textContent = event.text || '';
      card.appendChild(text);
      const actions = document.createElement('div');
      actions.className = 'actions';
      actions.appendChild(qbutton('Evidence', () =>
        showEvidence(event.source, event.page, event.section)));
      actions.appendChild(qbutton('Copy citation', () =>
        copyCitation(event.source, event.page, event.section)));
      card.appendChild(actions);
      box.appendChild(card);
    });
    if (msg) msg.textContent = j.count + ' timeline event(s)';
  } catch (e) { box.textContent = 'timeline error: ' + e; }
}
document.getElementById('timelinebtn').onclick = () => showTimeline();
function copyCitation(source, page, section) {
  const text = source + (page ? ' (p. ' + page + ')' : '') +
               (section ? ' [' + section + ']' : '');
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).catch(() => {});
  }
}
async function showSummary(source) {
  const box = document.getElementById('docanswer');
  box.textContent = 'summarizing...';
  try {
    const query = new URLSearchParams({source, sentences: 5});
    const r = await fetch('/v1/documents/summary?' + query.toString());
    const j = await r.json();
    box.textContent = '';
    if (!r.ok) {
      box.textContent = 'summary error: ' + (j.error || r.status);
      return;
    }
    const head = document.createElement('div');
    head.className = 'hint';
    head.textContent = 'Extractive summary: ' + source + ' | ' +
      (j.chunks || 0) + ' chunk(s) | ' + (j.candidates || 0) +
      ' candidate sentence(s)';
    box.appendChild(head);
    const text = document.createElement('div');
    text.className = 'qitem';
    text.textContent = j.text || '';
    box.appendChild(text);
    const actions = document.createElement('div');
    actions.className = 'actions';
    const citation = (j.citations || [])[0] || {};
    actions.appendChild(qbutton('Evidence', () => showEvidence(source)));
    actions.appendChild(qbutton('Copy citation', () =>
      copyCitation(source, citation.page, citation.section)));
    box.appendChild(actions);
  } catch (e) { box.textContent = 'summary error: ' + e; }
}
async function summarizeCollection() {
  const prefix = document.getElementById('docprefix').value.trim();
  const msg = document.getElementById('docmsg');
  const box = document.getElementById('docanswer');
  box.textContent = '';
  if (!prefix) {
    msg.textContent = 'enter a collection prefix, e.g. notes/';
    return;
  }
  msg.textContent = 'summarizing collection...';
  try {
    const query = new URLSearchParams({source_prefix: prefix, sentences: 8});
    const r = await fetch('/v1/documents/summary?' + query.toString());
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'collection summary error: ' +
        (j.error || r.status);
      return;
    }
    msg.textContent = 'collection summary | ' + (j.sources || 0) +
      ' source(s) | ' + (j.chunks || 0) + ' chunk(s)';
    const head = document.createElement('div');
    head.className = 'hint';
    head.textContent = 'Extractive collection summary: ' + prefix;
    box.appendChild(head);
    const text = document.createElement('div');
    text.className = 'qitem';
    text.textContent = j.text || '';
    box.appendChild(text);
    (j.citations || []).slice(0, 8).forEach(citation => {
      const card = document.createElement('div');
      card.className = 'qitem';
      const title = document.createElement('div');
      title.textContent = (citation.source || '') +
        (citation.page ? ' | p. ' + citation.page : '') +
        (citation.section ? ' | ' + citation.section : '');
      card.appendChild(title);
      const actions = document.createElement('div');
      actions.className = 'actions';
      actions.appendChild(qbutton('Evidence', () =>
        showEvidence(citation.source, citation.page, citation.section)));
      actions.appendChild(qbutton('Copy citation', () =>
        copyCitation(citation.source, citation.page, citation.section)));
      card.appendChild(actions);
      box.appendChild(card);
    });
  } catch (e) {
    msg.textContent = 'collection summary error: ' + e;
  }
}
async function showEvidence(source, page, section) {
  const box = document.getElementById('evidence');
  box.textContent = 'loading evidence...';
  const query = new URLSearchParams({source});
  if (page) query.set('page', page);
  if (section) query.set('section', section);
  try {
    const r = await fetch('/v1/documents/evidence?' + query.toString());
    const j = await r.json();
    box.textContent = '';
    if (!r.ok) {
      box.textContent = 'evidence error: ' + (j.error || r.status);
      return;
    }
    const head = document.createElement('div');
    head.className = 'hint';
    head.textContent = 'Evidence: ' + source +
      (page ? ' | p. ' + page : '') +
      (section ? ' | ' + section : '') +
      ' | ' + (j.returned || 0) + ' chunk(s)';
    box.appendChild(head);
    (j.data || []).forEach(item => {
      const card = document.createElement('div');
      card.className = 'qitem';
      const title = document.createElement('div');
      title.textContent = (item.page ? 'p. ' + item.page : '') +
        (item.section ? (item.page ? ' | ' : '') + item.section : '');
      if (title.textContent) card.appendChild(title);
      const text = document.createElement('div');
      text.className = 'src';
      text.textContent = item.chunk || '';
      card.appendChild(text);
      box.appendChild(card);
    });
  } catch (e) { box.textContent = 'evidence error: ' + e; }
}
async function refreshGraph() {
  const nodesBox = document.getElementById('graphnodes');
  const edgesBox = document.getElementById('graphedges');
  const msg = document.getElementById('graphmsg');
  nodesBox.textContent = '';
  edgesBox.textContent = '';
  msg.textContent = 'building graph...';
  try {
    const r = await fetch('/v1/graph?max_entities=40&min_edge=1');
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'graph error: ' + (j.error || r.status);
      return;
    }
    const stats = j.stats || {};
    msg.textContent = (stats.nodes || 0) + ' entities, ' +
      (stats.edges || 0) + ' connections over ' +
      (stats.chunks_scanned || 0) + ' chunk(s)';
    (j.nodes || []).forEach(node => {
      const button = document.createElement('button');
      button.textContent = node.name + ' (' + node.mentions + ')';
      button.title = 'Show evidence for ' + node.name;
      button.onclick = () => showEntity(node.name);
      nodesBox.appendChild(button);
    });
    (j.edges || []).slice(0, 30).forEach(edge => {
      const card = document.createElement('div');
      card.className = 'qitem';
      const title = document.createElement('div');
      title.textContent = edge.source + ' ↔ ' + edge.target + ' | ' +
        edge.weight + ' shared chunk(s)';
      card.appendChild(title);
      edgesBox.appendChild(card);
    });
  } catch (e) {
    msg.textContent = 'graph error: ' + e;
  }
}
async function showEntity(name) {
  const box = document.getElementById('graphevidence');
  const msg = document.getElementById('graphmsg');
  if (!name) name = document.getElementById('gramentity').value.trim();
  if (!name) { msg.textContent = 'enter an entity'; return; }
  msg.textContent = 'loading evidence for ' + name + '...';
  box.textContent = '';
  try {
    const query = new URLSearchParams({name, limit: 10});
    const r = await fetch('/v1/graph/entity?' + query.toString());
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'entity error: ' + (j.error || r.status);
      return;
    }
    msg.textContent = j.entity + ' | ' + (j.matched_chunks || 0) +
      ' chunk(s) | ' + (j.connections || []).length + ' connection(s)';
    (j.sources || []).forEach(item => {
      const card = document.createElement('div');
      card.className = 'qitem';
      const title = document.createElement('div');
      title.textContent = (item.title || item.source) + ' | ' + item.source +
        (item.page ? ' | p. ' + item.page : '') +
        (item.section ? ' | ' + item.section : '');
      card.appendChild(title);
      const snippet = document.createElement('div');
      snippet.className = 'src';
      snippet.textContent = item.snippet || '';
      card.appendChild(snippet);
      const actions = document.createElement('div');
      actions.className = 'actions';
      actions.appendChild(qbutton('Evidence', () =>
        showEvidence(item.source, item.page, item.section)));
      actions.appendChild(qbutton('Copy citation', () =>
        copyCitation(item.source, item.page, item.section)));
      card.appendChild(actions);
      box.appendChild(card);
    });
    if ((j.connections || []).length) {
      const connections = document.createElement('div');
      connections.className = 'hint';
      connections.textContent = 'Connections: ' +
        j.connections.map(item => item.source + ' ↔ ' + item.target)
          .join('; ');
      box.appendChild(connections);
    }
  } catch (e) {
    msg.textContent = 'entity error: ' + e;
  }
}
async function showRelations() {
  const entity = document.getElementById('gramentity').value.trim();
  const msg = document.getElementById('graphmsg');
  const box = document.getElementById('relations');
  if (!entity) { msg.textContent = 'enter an entity'; return; }
  msg.textContent = 'loading typed relations for ' + entity + '...';
  box.textContent = '';
  try {
    const query = new URLSearchParams({entity, limit: 20});
    const r = await fetch('/v1/graph/relations?' + query.toString());
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'relations error: ' + (j.error || r.status);
      return;
    }
    msg.textContent = (j.count || 0) + ' typed relation(s) for ' + entity;
    (j.relations || []).forEach(item => {
      const card = document.createElement('div');
      card.className = 'qitem';
      const title = document.createElement('div');
      title.textContent = item.subject + ' --' + item.relation + '--> ' +
        item.object;
      card.appendChild(title);
      const snippet = document.createElement('div');
      snippet.className = 'src';
      snippet.textContent = item.text || '';
      card.appendChild(snippet);
      const actions = document.createElement('div');
      actions.className = 'actions';
      actions.appendChild(qbutton('Evidence', () =>
        showEvidence(item.source, item.page, item.section)));
      actions.appendChild(qbutton('Copy citation', () =>
        copyCitation(item.source, item.page, item.section)));
      card.appendChild(actions);
      box.appendChild(card);
    });
  } catch (e) {
    msg.textContent = 'relations error: ' + e;
  }
}
document.getElementById('relationsbtn').onclick = () => showRelations();
async function refreshSkills() {
  const msg = document.getElementById('skmsg');
  const box = document.getElementById('sklist');
  msg.textContent = 'loading skills...';
  box.textContent = '';
  try {
    const r = await fetch('/v1/skills');
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'skills error: ' + (j.error || r.status);
      return;
    }
    const data = j.data || [];
    msg.textContent = data.length + ' skill(s) from ' + (j.dir || '');
    data.forEach(skill => {
      const card = document.createElement('div');
      card.className = 'qitem';
      const title = document.createElement('div');
      title.textContent = skill.name + (skill.provider ? '' : ' [template]');
      card.appendChild(title);
      const detail = document.createElement('div');
      detail.className = 'src';
      detail.textContent = (skill.description || '') + ' | triggers: ' +
        (skill.triggers || []).join(', ');
      card.appendChild(detail);
      box.appendChild(card);
    });
    try {
      const cr = await fetch('/v1/skills/chains');
      const cj = await cr.json();
      const select = document.getElementById('chainrecipe');
      const current = select.value;
      select.textContent = '';
      const empty = document.createElement('option');
      empty.value = '';
      empty.textContent = 'named recipe';
      select.appendChild(empty);
      (cj.data || []).forEach(recipe => {
        const option = document.createElement('option');
        option.value = recipe.name;
        option.textContent = recipe.name + ' (' + (recipe.steps || []).length +
          ' step)';
        select.appendChild(option);
      });
      if ([...select.options].some(option => option.value === current)) {
        select.value = current;
      }
      if ((cj.data || []).length) {
        msg.textContent += ' | ' + cj.data.length + ' recipe(s)';
      }
    } catch (e) {}
  } catch (e) {
    msg.textContent = 'skills error: ' + e;
  }
}
async function runSkill() {
  const name = document.getElementById('skillname').value.trim();
  const query = document.getElementById('skillquery').value.trim();
  const msg = document.getElementById('skmsg');
  const box = document.getElementById('skillresult');
  box.textContent = '';
  if (!name || !query) {
    msg.textContent = 'enter a skill name and a query';
    return;
  }
  msg.textContent = 'running skill...';
  try {
    const r = await fetch('/v1/skills/run', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name, query})});
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'skill error: ' + (j.error || r.status);
      return;
    }
    msg.textContent = 'skill: ' + j.skill + ' | provider called: ' +
      j.provider_called;
    const card = document.createElement('div');
    card.className = 'qitem';
    card.textContent = j.reply || '';
    box.appendChild(card);
    (j.evidence || []).forEach(item => {
      const evidence = document.createElement('div');
      evidence.className = 'qitem';
      const title = document.createElement('div');
      title.textContent = (item.title || item.source) + ' | ' + item.source +
        (item.page ? ' | p. ' + item.page : '') +
        (item.section ? ' | ' + item.section : '');
      evidence.appendChild(title);
      const snippet = document.createElement('div');
      snippet.className = 'src';
      snippet.textContent = item.snippet || '';
      evidence.appendChild(snippet);
      const actions = document.createElement('div');
      actions.className = 'actions';
      actions.appendChild(qbutton('Evidence', () =>
        showEvidence(item.source, item.page, item.section)));
      actions.appendChild(qbutton('Copy citation', () =>
        copyCitation(item.source, item.page, item.section)));
      evidence.appendChild(actions);
      box.appendChild(evidence);
    });
  } catch (e) {
    msg.textContent = 'skill error: ' + e;
  }
}
async function runSkillChain() {
  const names = document.getElementById('skillchain').value
    .split(',').map(item => item.trim()).filter(Boolean);
  const query = document.getElementById('skillquery').value.trim();
  const msg = document.getElementById('skmsg');
  const box = document.getElementById('chainresult');
  box.textContent = '';
  if (!names.length || !query) {
    msg.textContent = 'enter comma-separated skill names and a query';
    return;
  }
  msg.textContent = 'running skill chain...';
  try {
    const r = await fetch('/v1/skills/chain', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        skills: names, query, carry_context: true})});
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'chain error: ' + (j.error || r.status);
      return;
    }
    msg.textContent = 'chain: ' + (j.chain || []).join(' -> ') +
      ' | provider calls: ' + j.provider_calls;
    const finalCard = document.createElement('div');
    finalCard.className = 'qitem';
    finalCard.textContent = 'Final: ' + (j.reply || '');
    box.appendChild(finalCard);
    (j.steps || []).forEach(step => {
      const card = document.createElement('div');
      card.className = 'qitem';
      card.textContent = step.skill + ': ' + step.reply;
      box.appendChild(card);
    });
    (j.evidence || []).forEach(item => {
      const evidence = document.createElement('div');
      evidence.className = 'qitem';
      const title = document.createElement('div');
      title.textContent = (item.title || item.source) + ' | ' + item.source +
        (item.page ? ' | p. ' + item.page : '') +
        (item.section ? ' | ' + item.section : '');
      evidence.appendChild(title);
      const snippet = document.createElement('div');
      snippet.className = 'src';
      snippet.textContent = item.snippet || '';
      evidence.appendChild(snippet);
      box.appendChild(evidence);
    });
  } catch (e) {
    msg.textContent = 'chain error: ' + e;
  }
}
async function runRecipe() {
  const name = document.getElementById('chainrecipe').value;
  const query = document.getElementById('skillquery').value.trim();
  const msg = document.getElementById('skmsg');
  const box = document.getElementById('chainresult');
  box.textContent = '';
  if (!name || !query) {
    msg.textContent = 'choose a recipe and enter a query';
    return;
  }
  msg.textContent = 'running recipe...';
  try {
    const r = await fetch('/v1/skills/chains/run', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name, query})});
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'recipe error: ' + (j.error || r.status);
      return;
    }
    msg.textContent = 'recipe: ' + j.recipe + ' | provider calls: ' +
      j.provider_calls;
    const finalCard = document.createElement('div');
    finalCard.className = 'qitem';
    finalCard.textContent = 'Final: ' + (j.reply || '');
    box.appendChild(finalCard);
    (j.steps || []).forEach(step => {
      const card = document.createElement('div');
      card.className = 'qitem';
      card.textContent = step.skill + ': ' + step.reply;
      box.appendChild(card);
    });
    (j.evidence || []).forEach(item => {
      const evidence = document.createElement('div');
      evidence.className = 'qitem';
      const title = document.createElement('div');
      title.textContent = (item.title || item.source) + ' | ' + item.source;
      evidence.appendChild(title);
      const snippet = document.createElement('div');
      snippet.className = 'src';
      snippet.textContent = item.snippet || '';
      evidence.appendChild(snippet);
      box.appendChild(evidence);
    });
  } catch (e) {
    msg.textContent = 'recipe error: ' + e;
  }
}
document.getElementById('chainreciperun').onclick = () => runRecipe();
document.getElementById('skillchainrun').onclick = () => runSkillChain();
document.getElementById('skillchain').addEventListener('keydown', event => {
  if (event.key === 'Enter') runSkillChain();
});
document.getElementById('skillrun').onclick = () => runSkill();
document.getElementById('skillquery').addEventListener('keydown', event => {
  if (event.key === 'Enter') runSkill();
});
document.getElementById('skrefresh').onclick = () => refreshSkills();
async function showGraphPath() {
  const source = document.getElementById('graphsource').value.trim();
  const target = document.getElementById('graphtarget').value.trim();
  const msg = document.getElementById('graphmsg');
  const box = document.getElementById('graphevidence');
  box.textContent = '';
  if (!source || !target) {
    msg.textContent = 'enter a source and target entity';
    return;
  }
  msg.textContent = 'finding graph path...';
  try {
    const query = new URLSearchParams({source, target, max_depth: 4});
    const r = await fetch('/v1/graph/path?' + query.toString());
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'path error: ' + (j.error || r.status);
      return;
    }
    if (!j.found) {
      msg.textContent = 'no path: ' + (j.reason || 'not found');
      return;
    }
    msg.textContent = 'path | ' + j.hops + ' hop(s)';
    const head = document.createElement('div');
    head.className = 'qitem';
    head.textContent = 'Path: ' + j.path.join(' → ');
    box.appendChild(head);
    (j.edges || []).forEach(edge => {
      const card = document.createElement('div');
      card.className = 'qitem';
      card.textContent = edge.source + ' ↔ ' + edge.target + ' | ' +
        edge.sources + ' shared chunk(s)';
      box.appendChild(card);
    });
    const actions = document.createElement('div');
    actions.className = 'actions';
    j.path.forEach(name => {
      actions.appendChild(qbutton(name, () => showEntity(name)));
    });
    box.appendChild(actions);
  } catch (e) {
    msg.textContent = 'path error: ' + e;
  }
}
document.getElementById('graphpathbtn').onclick = () => showGraphPath();
document.getElementById('graphrefresh').onclick = () => refreshGraph();
document.getElementById('graphentitybtn').onclick = () => showEntity();
document.getElementById('gramentity').addEventListener('keydown', event => {
  if (event.key === 'Enter') showEntity();
});
document.getElementById('docsearch').onclick = () => searchDocs();
document.getElementById('docanswerbtn').onclick = () => answerDocs();
document.getElementById('docsumcollection').onclick = () => summarizeCollection();
document.getElementById('docsrc').addEventListener('change', () => {
  document.getElementById('docprefix').value = '';
});
document.getElementById('docprefix').addEventListener('input', () => {
  document.getElementById('docsrc').value = '';
});
document.getElementById('docprefix').addEventListener('keydown', event => {
  if (event.key === 'Enter') summarizeCollection();
});
document.getElementById('docq').addEventListener('keydown', event => {
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    answerDocs();
  } else if (event.key === 'Enter') {
    searchDocs();
  }
});
document.getElementById('docsrc').addEventListener('keydown', event => {
  if (event.key === 'Enter') searchDocs();
});
async function forgetSource(source) {
  if (!confirm('Delete indexed source "' + source + '"?')) return;
  try {
    const r = await fetch('/v1/documents/forget', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({source, confirm: true})});
    const j = await r.json();
    setForgetMsg(r.ok
      ? ('deleted ' + j.chunks_deleted + ' chunk(s) from ' + source)
      : ('error: ' + (j.error || r.status)), !r.ok);
    if (r.ok) { refreshHealth(); refreshDocs(); }
  } catch (e) { setForgetMsg('error: ' + e, true); }
}
document.getElementById('tnote').onclick = async () => {
  const answer = document.getElementById('ta').value.trim();
  const out = document.getElementById('tmsg');
  if (!answer) {
    out.textContent = 'answer is required';
    return;
  }
  out.textContent = 'saving...';
  try {
    const r = await fetch('/v1/teacher/notes', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        teacher: document.getElementById('tt').value,
        model: document.getElementById('tm').value,
        topic: document.getElementById('tp').value,
        source_url: document.getElementById('tu').value,
        answer,
      })});
    const j = await r.json();
    out.textContent = r.ok
      ? ('Saved as unverified teacher note: ' + j.source +
         ' (' + j.chunks + ' chunk(s))')
      : ('error: ' + (j.error || r.status));
    if (r.ok) {
      document.getElementById('ta').value = '';
      document.getElementById('tp').value = '';
      document.getElementById('tu').value = '';
      refreshHealth();
      refreshTeacherNotes();
    }
  } catch (e) { out.textContent = 'error: ' + e; }
};
document.getElementById('task').onclick = async () => {
  const prompt = document.getElementById('tq').value.trim();
  const out = document.getElementById('tamsg');
  if (!prompt) {
    out.textContent = 'question is required';
    return;
  }
  const button = document.getElementById('task');
  button.disabled = true;
  out.textContent = 'asking provider...';
  try {
    const r = await fetch('/v1/teacher/ask', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        prompt,
        topic: document.getElementById('tp').value,
        source_url: document.getElementById('tu').value,
        max_tokens: 512,
      })});
    const j = await r.json();
    out.textContent = r.ok
      ? ('Saved as unverified teacher note: ' + j.source +
         ' (' + j.chunks + ' chunk(s))')
      : ('error: ' + (j.error || r.status));
    if (r.ok) {
      document.getElementById('tq').value = '';
      refreshHealth();
      refreshTeacherNotes();
    }
  } catch (e) {
    out.textContent = 'error: ' + e;
  } finally {
    button.disabled = false;
  }
};
async function refreshTeacherNotes() {
  const box = document.getElementById('tnotes');
  try {
    const r = await fetch('/v1/teacher/notes');
    const j = await r.json();
    if (!r.ok) {
      box.textContent = 'teacher notes error: ' + (j.error || r.status);
      return;
    }
    renderTeacherNotes(j.data || [], j.stats || {});
  } catch (e) {
    box.textContent = 'teacher notes error: ' + e;
  }
}
function renderTeacherNotes(data, stats) {
  const box = document.getElementById('tnotes');
  box.textContent = '';
  const summary = document.createElement('div');
  summary.className = 'hint';
  summary.textContent =
    'teacher notes: ' + (stats.total || 0) + ' total, ' +
    (stats.unverified || 0) + ' unverified, ' +
    (stats.verified || 0) + ' verified';
  box.appendChild(summary);
  if (!data.length) {
    const empty = document.createElement('div');
    empty.className = 'hint';
    empty.textContent = 'No teacher notes yet.';
    box.appendChild(empty);
    return;
  }
  data.forEach(note => {
    const card = document.createElement('div');
    card.className = 'qitem';
    const title = document.createElement('div');
    title.textContent = (note.verified ? 'VERIFIED' : 'UNVERIFIED') +
                        ' | ' + (note.title || note.source);
    card.appendChild(title);
    const meta = document.createElement('div');
    meta.className = 'src';
    meta.textContent = 'teacher: ' + (note.teacher || '') +
                       ' / ' + (note.model || '') +
                       ' | ' + (note.source || '');
    card.appendChild(meta);
    if (note.source_url) {
      const url = document.createElement('div');
      url.className = 'src';
      url.textContent = 'source_url: ' + note.source_url;
      card.appendChild(url);
    }
    if (note.verified) {
      const reviewed = document.createElement('div');
      reviewed.className = 'src';
      reviewed.textContent = 'reviewed by ' + (note.reviewer || 'unknown') +
                             ' at ' + (note.verified_at || 'unknown');
      card.appendChild(reviewed);
    } else {
      const actions = document.createElement('div');
      actions.className = 'actions';
      actions.appendChild(qbutton('Verify',
        () => verifyTeacherNote(note.source)));
      card.appendChild(actions);
    }
    box.appendChild(card);
  });
}
async function verifyTeacherNote(source) {
  const reviewer = document.getElementById('trev').value.trim();
  const out = document.getElementById('tvmsg');
  if (!reviewer) {
    out.textContent = 'reviewer is required';
    return;
  }
  out.textContent = 'verifying...';
  try {
    const r = await fetch('/v1/teacher/notes/verify', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({source, reviewer})});
    const j = await r.json();
    out.textContent = r.ok
      ? ('Verified ' + j.source + ' by ' + j.reviewer)
      : ('error: ' + (j.error || r.status));
    if (r.ok) {
      await refreshTeacherNotes();
      refreshHealth();
    }
  } catch (e) {
    out.textContent = 'error: ' + e;
  }
}
document.getElementById('trefresh').onclick = () => refreshTeacherNotes();
function qmsg(text, isError) {
  const el = document.getElementById('rmsg');
  el.textContent = text;
  el.style.color = isError ? '#b00020' : '#666';
}
function qbutton(label, fn) {
  const button = document.createElement('button');
  button.textContent = label;
  button.onclick = fn;
  return button;
}
function renderQueue(data, stats) {
  const box = document.getElementById('rq');
  box.textContent = '';
  const summary = document.createElement('div');
  summary.className = 'hint';
  summary.textContent =
    'total ' + (stats.total || 0) +
    ' | pending ' + (stats.pending || 0) +
    ' | approved ' + (stats.approved || 0) +
    ' | done ' + (stats.done || 0) +
    ' | error ' + (stats.error || 0) +
    ' | rejected ' + (stats.rejected || 0);
  box.appendChild(summary);
  if (!data.length) {
    const empty = document.createElement('div');
    empty.className = 'hint';
    empty.textContent = 'No topics yet.';
    box.appendChild(empty);
    return;
  }
  data.forEach(item => {
    const card = document.createElement('div');
    card.className = 'qitem';
    const title = document.createElement('div');
    title.textContent = item.id + ' | ' + item.status +
      (item.kind === 'curiosity' ? ' | curiosity' : '') + ' | ' + item.topic;
    card.appendChild(title);
    const meta = document.createElement('div');
    meta.className = 'src';
    meta.textContent = (item.lang || 'en') + ' | max pages ' +
                       (item.max_pages || 1);
    card.appendChild(meta);
    if (item.error) {
      const error = document.createElement('div');
      error.className = 'src';
      error.textContent = 'error: ' + item.error;
      card.appendChild(error);
    }
    (item.sources || []).forEach(source => {
      const line = document.createElement('div');
      line.className = 'src';
      line.textContent = 'source: ' + source.title + ' | ' +
                         source.chunks + ' chunks | ' + source.source;
      card.appendChild(line);
    });
    if (item.teacher && (item.teacher.source || item.teacher.error)) {
      const line = document.createElement('div');
      line.className = 'src';
      const teacherStatus = item.teacher.error
        ? item.teacher.error
        : (item.teacher.verified
           ? (item.teacher.reviewer
              ? 'verified by ' + item.teacher.reviewer
              : 'verified')
           : 'unverified');
      line.textContent = 'teacher: ' +
        (item.teacher.source || 'no note') + ' | ' + teacherStatus;
      card.appendChild(line);
      if (item.teacher.answer_preview) {
        const preview = document.createElement('div');
        preview.className = 'src';
        preview.textContent = 'synthesis: ' + item.teacher.answer_preview;
        card.appendChild(preview);
      }
    }
    if (item.status === 'done' && !(item.sources || []).length) {
      const none = document.createElement('div');
      none.className = 'src';
      none.textContent = 'no pages found';
      card.appendChild(none);
    }
    if (item.status === 'pending' || item.status === 'approved') {
      const actions = document.createElement('div');
      actions.className = 'actions';
      if (item.status === 'pending') {
        actions.appendChild(qbutton('Approve',
          () => queueAction(item.id, 'approve')));
      }
      actions.appendChild(qbutton('Reject',
        () => queueAction(item.id, 'reject')));
      card.appendChild(actions);
    }
    box.appendChild(card);
  });
}
async function refreshQueue() {
  try {
    const r = await fetch('/v1/research/queue');
    const j = await r.json();
    if (!r.ok) {
      qmsg('queue error: ' + (j.error || r.status), true);
      return;
    }
    renderQueue(j.data || [], j.stats || {});
  } catch (e) {
    qmsg('queue error: ' + e, true);
  }
}
async function queueAction(id, action) {
  qmsg(action + ' ' + id + '...');
  try {
    const r = await fetch('/v1/research/queue/' + action, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({id})});
    const j = await r.json();
    qmsg(r.ok ? (id + ' -> ' + j.status)
              : ('error: ' + (j.error || r.status)), !r.ok);
    await refreshQueue();
  } catch (e) {
    qmsg('error: ' + e, true);
  }
}
document.getElementById('radd').onclick = async () => {
  const topic = document.getElementById('rtopic').value.trim();
  if (!topic) {
    qmsg('topic is required', true);
    return;
  }
  qmsg('adding ' + topic + '...');
  try {
    const r = await fetch('/v1/research/queue', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        topic,
        lang: document.getElementById('rlang').value,
        pages: parseInt(document.getElementById('rpages').value, 10),
      })});
    const j = await r.json();
    qmsg(r.ok ? ('Added ' + j.id + ' as pending.')
              : ('error: ' + (j.error || r.status)), !r.ok);
    if (r.ok) {
      document.getElementById('rtopic').value = '';
      await refreshQueue();
    }
  } catch (e) {
    qmsg('error: ' + e, true);
  }
};
document.getElementById('rrun').onclick = async () => {
  const button = document.getElementById('rrun');
  const limit = parseInt(document.getElementById('rlimit').value, 10) || 1;
  const label = button.textContent;
  button.disabled = true;
  button.textContent = 'Fetching...';
  qmsg('running approved topics...');
  try {
    const r = await fetch('/v1/research/queue/run', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({limit})});
    const j = await r.json();
    if (!r.ok) {
      qmsg('run error: ' + (j.error || r.status), true);
    } else {
      const ran = j.ran || [];
      const done = ran.filter(item => item.status === 'done').length;
      const errors = ran.filter(item => item.status === 'error').length;
      qmsg('processed ' + ran.length + ' item(s): ' +
           done + ' done, ' + errors + ' error.');
    }
    await refreshQueue();
  } catch (e) {
    qmsg('run error: ' + e, true);
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
};
function setForgetMsg(text, isError) {
  const el = document.getElementById('fmsg');
  el.textContent = text;
  el.style.color = isError ? '#b00020' : '#666';
}
document.getElementById('forgetdoc').onclick = async () => {
  const source = document.getElementById('forgetsrc').value.trim();
  if (!source) return setForgetMsg('Enter the exact source to delete.', true);
  if (!confirm('Delete indexed source "' + source + '"?')) return;
  try {
    const r = await fetch('/v1/documents/forget', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({source, confirm: true})});
    const j = await r.json();
    setForgetMsg(r.ok
      ? ('deleted ' + j.chunks_deleted + ' chunk(s) and ' +
         j.meta_deleted + ' metadata row(s)')
      : ('error: ' + (j.error || r.status)), !r.ok);
    if (r.ok) { refreshHealth(); refreshDocs(); }
  } catch (e) { setForgetMsg('error: ' + e, true); }
};
document.getElementById('forgetmem').onclick = async () => {
  if (!confirm('Delete this session memory (facts, tasks, episodes)?')) return;
  try {
    const r = await fetch('/v1/memory/forget', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: sessionId, all: true,
                            confirm: true})});
    const j = await r.json();
    setForgetMsg(r.ok
      ? ('deleted facts ' + j.facts_deleted + ', episodes ' +
         j.episodes_deleted + ', tasks ' + j.tasks_deleted)
      : ('error: ' + (j.error || r.status)), !r.ok);
  } catch (e) { setForgetMsg('error: ' + e, true); }
};
function setLearningMsg(text, isError) {
  const el = document.getElementById('lmsg');
  el.textContent = text;
  el.style.color = isError ? '#b00020' : '#666';
}
function renderLearning(data, stats) {
  const box = document.getElementById('lq');
  box.textContent = '';
  const summary = document.createElement('div');
  summary.className = 'hint';
  summary.textContent =
    'feedback: ' + (stats.total || 0) + ' total | pending ' +
    (stats.pending || 0) + ' | approved ' + (stats.approved || 0) +
    ' | rejected ' + (stats.rejected || 0);
  box.appendChild(summary);
  if (!data.length) {
    const empty = document.createElement('div');
    empty.className = 'hint';
    empty.textContent = 'No feedback yet. Rate a chat reply above.';
    box.appendChild(empty);
    return;
  }
  data.forEach(item => {
    const card = document.createElement('div');
    card.className = 'qitem';
    const title = document.createElement('div');
    title.textContent = item.id + ' | ' + item.status +
                        ' | rating ' + item.rating;
    card.appendChild(title);
    const user = document.createElement('div');
    user.className = 'src';
    user.textContent = 'user: ' + (item.user || '').slice(0, 240);
    card.appendChild(user);
    const assistant = document.createElement('div');
    assistant.className = 'src';
    assistant.textContent = 'assistant: ' + (item.assistant || '').slice(0, 240);
    card.appendChild(assistant);
    if (item.corrected) {
      const corrected = document.createElement('div');
      corrected.className = 'src';
      corrected.textContent = 'corrected: ' + item.corrected.slice(0, 240);
      card.appendChild(corrected);
    }
    if (item.status === 'pending' || item.status === 'approved') {
      const actions = document.createElement('div');
      actions.className = 'actions';
      if (item.status === 'pending') {
        actions.appendChild(qbutton('Approve',
          () => learningAction(item.id, 'approve')));
      }
      actions.appendChild(qbutton('Reject',
        () => learningAction(item.id, 'reject')));
      card.appendChild(actions);
    }
    box.appendChild(card);
  });
}
async function refreshLearning() {
  try {
    const r = await fetch('/v1/learning/queue');
    const j = await r.json();
    if (!r.ok) {
      setLearningMsg('learning error: ' + (j.error || r.status), true);
      return;
    }
    renderLearning(j.data || [], j.stats || {});
  } catch (e) {
    setLearningMsg('learning error: ' + e, true);
  }
}
async function learningAction(id, action) {
  setLearningMsg(action + ' ' + id + '...');
  try {
    const r = await fetch('/v1/learning/queue/' + action, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({id})});
    const j = await r.json();
    setLearningMsg(r.ok ? (id + ' -> ' + j.status)
                        : ('error: ' + (j.error || r.status)), !r.ok);
    await refreshLearning();
  } catch (e) {
    setLearningMsg('error: ' + e, true);
  }
}
document.getElementById('lrefresh').onclick = () => refreshLearning();
document.getElementById('lexport').onclick = async () => {
  setLearningMsg('exporting...');
  try {
    const r = await fetch('/v1/learning/queue/export', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: '{}'});
    const j = await r.json();
    setLearningMsg(r.ok
      ? ('exported ' + j.examples + ' example(s) to ' + j.path)
      : ('error: ' + (j.error || r.status)), !r.ok);
  } catch (e) {
    setLearningMsg('error: ' + e, true);
  }
};
function setBackupMsg(text, isError) {
  const el = document.getElementById('bmsg');
  el.textContent = text;
  el.style.color = isError ? '#b00020' : '#666';
}
document.getElementById('mexport').onclick = async () => {
  setBackupMsg('exporting...');
  try {
    const r = await fetch('/v1/memory/export', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: sessionId})});
    const j = await r.json();
    if (!r.ok) {
      setBackupMsg('error: ' + (j.error || r.status), true);
      return;
    }
    const blob = new Blob([JSON.stringify(j.data, null, 2)],
                          {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'atomzero-' + sessionId + '.json';
    a.click();
    URL.revokeObjectURL(url);
    setBackupMsg('session exported');
  } catch (e) { setBackupMsg('error: ' + e, true); }
};
document.getElementById('mimportbtn').onclick = () => {
  const file = document.getElementById('mimport').files[0];
  if (!file) { setBackupMsg('choose a JSON backup first', true); return; }
  if (!confirm('Replace this session memory with the imported file?')) return;
  const reader = new FileReader();
  reader.onload = async () => {
    try {
      const data = JSON.parse(reader.result);
      const r = await fetch('/v1/memory/import', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({session_id: sessionId, data})});
      const j = await r.json();
      if (!r.ok) {
        setBackupMsg('error: ' + (j.error || r.status), true);
        return;
      }
      setBackupMsg('imported facts ' + j.facts + ', episodes ' +
                   j.episodes + ', tasks ' + j.tasks);
      setTimeout(() => location.reload(), 500);
    } catch (e) { setBackupMsg('error: ' + e, true); }
  };
  reader.readAsText(file);
};
function setAdapterMsg(text, isError) {
  const el = document.getElementById('amsg');
  el.textContent = text;
  el.style.color = isError ? '#b00020' : '#666';
}
function renderAdapters(data, active) {
  const box = document.getElementById('aq');
  box.textContent = '';
  if (!data.length) {
    const empty = document.createElement('div');
    empty.className = 'hint';
    empty.textContent = 'No accepted adapter versions yet.';
    box.appendChild(empty);
    return;
  }
  data.slice().reverse().forEach(item => {
    const card = document.createElement('div');
    card.className = 'qitem';
    const title = document.createElement('div');
    title.textContent = (item.version || '?') + ' | ' +
      (item.examples || 0) + ' examples | ' +
      (item.path === active ? 'ACTIVE' : 'accepted');
    card.appendChild(title);
    const path = document.createElement('div');
    path.className = 'src';
    path.textContent = item.path || '';
    card.appendChild(path);
    const decision = item.decision || {};
    const metric = document.createElement('div');
    metric.className = 'src';
    metric.textContent =
      'target ' + ((decision.target_gain || 0) * 100).toFixed(2) + '% | ' +
      'bpb ' + ((decision.clean_bpb_reg || 0) * 100).toFixed(2) + '% | ' +
      'assistant ' + ((decision.clean_assistant_reg || 0) * 100).toFixed(2) + '%';
    card.appendChild(metric);
    box.appendChild(card);
  });
}
async function refreshAdapters() {
  try {
    const r = await fetch('/v1/learning/adapters');
    const j = await r.json();
    if (!r.ok) {
      setAdapterMsg('adapter error: ' + (j.error || r.status), true);
      return;
    }
    renderAdapters(j.data || [], j.active || '');
    setAdapterMsg('active: ' + (j.active || 'none'));
  } catch (e) {
    setAdapterMsg('adapter error: ' + e, true);
  }
}
document.getElementById('arefresh').onclick = () => refreshAdapters();
function setGapMsg(text, isError) {
  const el = document.getElementById('gmsg');
  el.textContent = text;
  el.style.color = isError ? '#b00020' : '#666';
}
function renderGaps(data, stats) {
  const box = document.getElementById('gq');
  box.textContent = '';
  const summary = document.createElement('div');
  summary.className = 'hint';
  summary.textContent =
    'gaps: ' + (stats.total || 0) + ' total | pending ' +
    (stats.pending || 0) + ' | proposed ' + (stats.proposed || 0) +
    ' | dismissed ' + (stats.dismissed || 0);
  box.appendChild(summary);
  if (!data.length) {
    const empty = document.createElement('div');
    empty.className = 'hint';
    empty.textContent = 'No knowledge gaps recorded.';
    box.appendChild(empty);
    return;
  }
  data.forEach(item => {
    const card = document.createElement('div');
    card.className = 'qitem';
    const title = document.createElement('div');
    title.textContent = item.id + ' | ' + item.status +
                        ' | seen ' + (item.count || 1);
    card.appendChild(title);
    const query = document.createElement('div');
    query.className = 'src';
    query.textContent = item.query || '';
    card.appendChild(query);
    if (item.research_id) {
      const research = document.createElement('div');
      research.className = 'src';
      research.textContent = 'research: ' + item.research_id;
      card.appendChild(research);
    }
    const actions = document.createElement('div');
    actions.className = 'actions';
    if (item.status === 'pending') {
      actions.appendChild(qbutton('Propose research',
        () => gapAction(item.id, 'propose')));
    }
    if (item.status === 'pending' || item.status === 'proposed') {
      actions.appendChild(qbutton('Dismiss',
        () => gapAction(item.id, 'dismiss')));
    }
    if (actions.childNodes.length) card.appendChild(actions);
    box.appendChild(card);
  });
}
async function refreshGaps() {
  try {
    const r = await fetch('/v1/learning/gaps');
    const j = await r.json();
    if (!r.ok) {
      setGapMsg('gap error: ' + (j.error || r.status), true);
      return;
    }
    renderGaps(j.data || [], j.stats || {});
  } catch (e) {
    setGapMsg('gap error: ' + e, true);
  }
}
async function gapAction(id, action) {
  setGapMsg(action + ' ' + id + '...');
  try {
    const r = await fetch('/v1/learning/gaps/' + action, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({id})});
    const j = await r.json();
    if (!r.ok) {
      setGapMsg('error: ' + (j.error || r.status), true);
      return;
    }
    setGapMsg(j.research
      ? (id + ' -> research ' + j.research.id + ' (pending approval)')
      : (id + ' -> ' + j.status));
    await refreshGaps();
    await refreshQueue();
  } catch (e) {
    setGapMsg('error: ' + e, true);
  }
}
document.getElementById('grefresh').onclick = () => refreshGaps();
async function proposeCuriosity() {
  const button = document.getElementById('gcuriosity');
  const label = button.textContent;
  button.disabled = true;
  button.textContent = 'Proposing...';
  setGapMsg('proposing pending gaps as curiosity topics...');
  try {
    const r = await fetch('/v1/curiosity/propose', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({limit: 3, pages: 1})});
    const j = await r.json();
    if (!r.ok) {
      setGapMsg('error: ' + (j.error || r.status), true);
    } else {
      const count = (j.proposed || []).length;
      setGapMsg(count + ' curiosity topic(s) pending approval' +
        ((j.skipped || []).length ? ' | ' + j.skipped.length + ' skipped' : ''));
      await refreshGaps();
      await refreshQueue();
    }
  } catch (e) {
    setGapMsg('error: ' + e, true);
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}
document.getElementById('gcuriosity').onclick = proposeCuriosity;
async function runHungry() {
  const msg = document.getElementById('hmsg');
  const box = document.getElementById('hres');
  const button = document.getElementById('hrun');
  const parsed = parseInt(document.getElementById('hmax').value, 10);
  const maxQuestions = Number.isFinite(parsed) ? parsed : 3;
  const dryRun = document.getElementById('hdry').checked;
  const topic = document.getElementById('htopic').value.trim();
  button.disabled = true;
  button.textContent = 'Running...';
  msg.textContent = 'running hungry mode...';
  box.textContent = '';
  try {
    const r = await fetch('/v1/hungry/run', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        max_questions: maxQuestions,
        max_provider_calls: Math.max(4, maxQuestions + 3),
        dry_run: dryRun,
        topic,
      })});
    const j = await r.json();
    if (!r.ok) {
      msg.textContent = 'hungry error: ' + (j.error || r.status);
      return;
    }
    msg.textContent = (j.learned || 0) + ' learned, ' +
      (j.known || 0) + ' known, ' + (j.skipped || 0) + ' skipped, ' +
      (j.failed || 0) + ' failed | weights changed: ' + j.weights_changed;
    (j.results || []).forEach(item => {
      const card = document.createElement('div');
      card.className = 'qitem';
      const title = document.createElement('div');
      title.textContent = (item.status || '?') + ' | ' +
                          (item.question || '');
      card.appendChild(title);
      const detail = document.createElement('div');
      detail.className = 'src';
      detail.textContent = (item.source ? 'source: ' + item.source : '') +
        (item.error ? ' | ' + item.error : '');
      card.appendChild(detail);
      box.appendChild(card);
    });
    await refreshGaps();
  } catch (e) {
    msg.textContent = 'hungry error: ' + e;
  } finally {
    button.disabled = false;
    button.textContent = 'Run hungry mode';
  }
}
document.getElementById('hrun').onclick = runHungry;
function refreshHealth() {
  fetch('/health').then(r => r.json()).then(j => {
    const docs = j.documents ? j.documents.chunks : 0;
    document.getElementById('m').textContent =
      'Atom Zero | model: ' + j.model + ' | params: ' + j.params + ' | step: ' + j.step +
      ' | adapter: ' + (j.adapter ? 'yes' : 'none') + ' | doc chunks: ' + docs +
      ' | web research: ' + (j.web_research && j.web_research.url_fetch ? 'URL on' : 'Wikipedia') +
      ' | safety: ' + (j.safety && j.safety.enabled ? 'on' : 'off') +
      ' | reranker: ' + (j.reranker || 'hashed') +
      ' | chat: ' + (j.chat_provider && j.chat_provider.enabled ? j.chat_provider.model : 'tiny-50m') +
      ' | teacher: ' + (j.teacher && j.teacher.enabled ? j.teacher.model : 'off');
    const teacher = j.teacher || {enabled: false};
    document.getElementById('task').disabled = !teacher.enabled;
    document.getElementById('tstatus').textContent = teacher.enabled
      ? ('Provider: ' + (teacher.label || 'openai-compatible') + ' / ' +
         teacher.model + ' @ ' + teacher.host)
      : 'No teacher provider configured; use manual paste.';
  }).catch(() => { document.getElementById('m').textContent = 'server not ready'; });
}
restoreChat();
pollProactive();
setInterval(pollProactive, 60000);
refreshHealth();
refreshQueue();
refreshTeacherNotes();
refreshLearning();
refreshGaps();
refreshDocs();
refreshAdapters();
refreshWeeklyReports();
</script>"""


def doc_answer(text: str, docs, source: str = "",
               source_prefix: str = "", kind: str = "",
               max_context_chars: int = 0) -> dict | None:
    """Return grounded notes with sources, evidence and match confidence."""
    if docs is None:
        return None
    query = str(text or "").strip()
    if not query:
        return None
    hits = docs.search(query, k=6, source=source,
                       source_prefix=source_prefix, kind=kind)
    if not hits:
        return None
    return grounded_answer(query, hits, max_context_chars=max_context_chars)


def copy_answer(prompt: str) -> dict | None:
    """Return a deterministic copy reply, or None to use the normal path.

    Only explicit transcription requests are handled. Everything else keeps
    the existing RAG/model behaviour. The ``source`` field marks provenance for
    API callers; this is a runtime tool, not model generation.
    """
    routed = route_copy(prompt)
    if not routed["handled"]:
        return None
    return {
        "text": routed["answer"],
        "sources": [],
        "source": "copy_tool",
        "format": routed["format"],
        "pairs_count": routed["pairs_count"],
    }


def numeric_answer(messages) -> dict | None:
    """Return a deterministic budget/time reply, or None to use the model.

    Only explicit limit-versus-value questions are handled; all other prompts
    keep the existing RAG/model behaviour. ``source`` marks the runtime tool.
    """
    routed = route_numeric(messages)
    if not routed["handled"]:
        return None
    return {
        "text": routed["answer"],
        "sources": [],
        "source": "numeric_tool",
        "kind": routed["kind"],
    }


def recall_answer(messages) -> dict | None:
    """Return a deterministic recall reply, or None to use the model.

    Only explicit "what is my name/city/..." questions with a fact stated in
    the conversation are handled; everything else keeps the model path.
    """
    routed = route_recall(messages)
    if not routed["handled"]:
        return None
    return {
        "text": routed["answer"],
        "sources": [],
        "source": "recall_tool",
        "kind": routed["kind"],
    }


def identity_answer(messages) -> dict | None:
    """Return a deterministic identity reply, or None to use the model."""
    routed = route_identity(messages)
    if not routed["handled"]:
        return None
    return {
        "text": routed["answer"],
        "sources": [],
        "source": "identity_tool",
        "kind": routed["kind"],
    }


def chat_prompt(messages) -> str:
    """Build the chat format used during training."""
    parts = []
    for message in (messages or [])[-8:]:
        role = message.get("role", "user")
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        tag = "<|assistant|>" if role == "assistant" else "<|user|>"
        parts.append(f"{tag} {content}")
    if not parts or not parts[-1].startswith("<|user|>"):
        parts.append("<|user|> hello")
    # Training conversations have no <|end|> between a user turn and the
    # assistant reply; <|end|> closes the whole conversation.
    return " ".join(parts) + " <|assistant|>"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: bytes, ctype: str = "application/json",
              headers: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict):
        self._send(code, json.dumps(payload).encode("utf-8"))

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {
                "ok": True,
                "model": _state["name"],
                "params": _state["params"],
                "config": _state["config"],
                "step": _state.get("step"),
                "adapter": _state.get("adapter") or None,
                "documents": (_state["docs"].stats()
                              if _state.get("docs") else None),
                "learning_queue": _state.get("queue_path"),
                "web_research": {
                    "wikipedia_lang": _state.get("wiki_lang"),
                    "url_fetch": bool(_state.get("allow_url_fetch")),
                },
                "research_queue": (
                    _state["research_queue"].stats()
                    if _state.get("research_queue") else None),
                "teacher": (
                    {"enabled": True,
                     "model": _state["teacher"].model,
                     "host": _state["teacher"].host,
                     "label": _state["teacher"].label}
                    if _state.get("teacher") else {"enabled": False}),
                "chat_provider": (
                    {"enabled": True,
                     "model": _state["chat_provider"].model,
                     "host": _state["chat_provider"].host,
                     "label": _state["chat_provider"].label,
                     "general_chat": bool(_state.get("allow_general_chat"))}
                    if _state.get("chat_provider")
                    else {"enabled": False}),
                "learning": (
                    _state["feedback"].stats()
                    if _state.get("feedback") else None),
                "knowledge_gaps": (
                    _state["gaps"].stats()
                    if _state.get("gaps") else None),
                "safety": (
                    {"enabled": True,
                     "version": _state["safety"].version,
                     **({"ml": True}
                        if getattr(_state["safety"], "ml_enabled", False)
                        else {})}
                    if _state.get("safety") else {"enabled": False}),
                "reranker": _state.get("reranker", "hashed"),
                "vector_index": (
                    _state["docs"].vector_index_stats()
                    if _state.get("docs") else None),
                "sessions": len(_state.get("sessions", {})),
            })
        elif self.path == "/v1/models":
            self._json(200, {"data": [{
                "id": _state["name"], "object": "model",
                "owned_by": "atomzero",
            }]})
        elif self.path == "/metrics":
            self._json(200, _state["metrics"])
        elif (self.path == "/v1/tasks/export"
              or self.path.startswith("/v1/tasks/export?")):
            params = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            session_id = (params.get("session_id", ["default"])[0]
                          or "default")
            status = str(params.get("status", ["open"])[0] or "open").strip()
            due = str(params.get("due", ["all"])[0] or "all").strip()
            priority = str(
                params.get("priority", ["all"])[0] or "all").strip()
            tag = str(params.get("tag", ["all"])[0] or "all").strip()
            sort = str(params.get("sort", ["default"])[0] or "default").strip()
            export_format = str(
                params.get("format", ["csv"])[0] or "csv").strip().lower()
            if export_format not in ("csv", "markdown"):
                return self._json(400, {
                    "error": "format must be csv or markdown"})
            try:
                session = self._session(session_id)
                from atomzero.briefing import (
                    render_task_list_csv, render_task_list_markdown,
                    task_list_view)

                view = task_list_view(
                    session.human.data, status=status, due=due,
                    priority=priority, tag=tag, sort=sort,
                    now=params.get("today", [None])[0])
                if export_format == "csv":
                    body = render_task_list_csv(view)
                    content_type = "text/csv; charset=utf-8"
                else:
                    body = render_task_list_markdown(view)
                    content_type = "text/markdown; charset=utf-8"
            except Exception as exc:
                return self._json(400, {"error": str(exc)})
            day = str(view.get("generated", ""))[:10] or "latest"
            suffix = "md" if export_format == "markdown" else "csv"
            filename = f"atomzero-tasks-{day}.{suffix}"
            return self._send(
                200, body.encode("utf-8"), content_type,
                {"Content-Disposition": f'attachment; filename="{filename}"'})
        elif (self.path == "/v1/tasks/tags"
              or self.path.startswith("/v1/tasks/tags?")):
            params = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            session_id = (params.get("session_id", ["default"])[0]
                          or "default")
            try:
                session = self._session(session_id)
                from atomzero.briefing import task_tag_view

                result = task_tag_view(
                    session.human.data,
                    now=params.get("today", [None])[0])
            except Exception as exc:
                return self._json(400, {"error": str(exc)})
            return self._json(200, result)
        elif (self.path == "/v1/tasks/tags/audit/export"
              or self.path.startswith("/v1/tasks/tags/audit/export?")):
            params = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            session_id = (params.get("session_id", ["default"])[0]
                          or "default")
            try:
                limit = int(params.get("limit", ["10"])[0])
                if limit < 1:
                    raise ValueError("limit must be at least 1")
                tag = str(params.get("tag", ["all"])[0] or "all").strip()
                action = str(params.get("action", ["all"])[0]
                             or "all").strip()
                export_format = str(
                    params.get("format", ["csv"])[0] or "csv").strip().lower()
                if export_format not in ("csv", "markdown"):
                    raise ValueError("format must be csv or markdown")
                session = self._session(session_id)
                from atomzero.briefing import (
                    render_task_tag_audit_csv,
                    render_task_tag_audit_markdown, task_tag_audit_view)

                view = task_tag_audit_view(
                    session.human.data, limit=limit, tag=tag,
                    action=action,
                    now=params.get("today", [None])[0])
                if export_format == "csv":
                    body = render_task_tag_audit_csv(view)
                    content_type = "text/csv; charset=utf-8"
                else:
                    body = render_task_tag_audit_markdown(view)
                    content_type = "text/markdown; charset=utf-8"
            except (TypeError, ValueError) as exc:
                return self._json(400, {"error": str(exc)})
            except Exception as exc:
                return self._json(400, {"error": str(exc)})
            day = str(view.get("generated", ""))[:10] or "latest"
            suffix = "md" if export_format == "markdown" else "csv"
            filename = f"atomzero-tag-audit-{day}.{suffix}"
            return self._send(
                200, body.encode("utf-8"), content_type,
                {"Content-Disposition": f'attachment; filename="{filename}"'})
        elif (self.path == "/v1/tasks/tags/audit"
              or self.path.startswith("/v1/tasks/tags/audit?")):
            params = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            session_id = (params.get("session_id", ["default"])[0]
                          or "default")
            try:
                limit = int(params.get("limit", ["10"])[0])
                if limit < 1:
                    raise ValueError("limit must be at least 1")
                tag = str(params.get("tag", ["all"])[0] or "all").strip()
                action = str(params.get("action", ["all"])[0]
                             or "all").strip()
                session = self._session(session_id)
                from atomzero.briefing import task_tag_audit_view

                result = task_tag_audit_view(
                    session.human.data, limit=limit, tag=tag,
                    action=action,
                    now=params.get("today", [None])[0])
            except (TypeError, ValueError) as exc:
                return self._json(400, {"error": str(exc)})
            except Exception as exc:
                return self._json(400, {"error": str(exc)})
            return self._json(200, result)
        elif (self.path == "/v1/tasks/list"
              or self.path.startswith("/v1/tasks/list?")):
            params = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            session_id = (params.get("session_id", ["default"])[0]
                          or "default")
            status = str(params.get("status", ["open"])[0] or "open").strip()
            due = str(params.get("due", ["all"])[0] or "all").strip()
            priority = str(
                params.get("priority", ["all"])[0] or "all").strip()
            tag = str(params.get("tag", ["all"])[0] or "all").strip()
            sort = str(params.get("sort", ["default"])[0] or "default").strip()
            try:
                session = self._session(session_id)
                from atomzero.briefing import task_list_view

                result = task_list_view(
                    session.human.data, status=status, due=due,
                    priority=priority, tag=tag, sort=sort,
                    now=params.get("today", [None])[0])
            except Exception as exc:
                return self._json(400, {"error": str(exc)})
            return self._json(200, result)
        elif (self.path == "/v1/tasks/search"
              or self.path.startswith("/v1/tasks/search?")):
            params = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            session_id = (params.get("session_id", ["default"])[0]
                          or "default")
            query = str(params.get("q", [""])[0] or "").strip()
            status = str(params.get("status", ["all"])[0] or "all").strip()
            due = str(params.get("due", ["all"])[0] or "all").strip()
            priority = str(
                params.get("priority", ["all"])[0] or "all").strip()
            tag = str(params.get("tag", ["all"])[0] or "all").strip()
            sort = str(params.get("sort", ["default"])[0] or "default").strip()
            if not query:
                return self._json(400, {"error": "q is required"})
            try:
                session = self._session(session_id)
                from atomzero.briefing import task_search_view

                result = task_search_view(
                    session.human.data, query, status=status, due=due,
                    priority=priority, sort=sort, tag=tag,
                    now=params.get("today", [None])[0])
            except Exception as exc:
                return self._json(400, {"error": str(exc)})
            return self._json(200, result)
        elif (self.path == "/v1/tasks/due-soon"
              or self.path.startswith("/v1/tasks/due-soon?")):
            params = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            session_id = (params.get("session_id", ["default"])[0]
                          or "default")
            try:
                days = int(params.get("days", ["7"])[0])
                if days < 1:
                    raise ValueError("days must be at least 1")
                sort = str(
                    params.get("sort", ["default"])[0] or "default").strip()
                tag = str(
                    params.get("tag", ["all"])[0] or "all").strip()
                session = self._session(session_id)
                from atomzero.briefing import due_soon_view

                result = due_soon_view(
                    session.human.data, days=days, sort=sort, tag=tag,
                    now=params.get("today", [None])[0])
            except (TypeError, ValueError) as exc:
                return self._json(400, {"error": str(exc)})
            except Exception as exc:
                return self._json(400, {"error": str(exc)})
            return self._json(200, result)
        elif (self.path == "/v1/tasks/overdue"
              or self.path.startswith("/v1/tasks/overdue?")):
            params = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            session_id = (params.get("session_id", ["default"])[0]
                          or "default")
            try:
                sort = str(
                    params.get("sort", ["default"])[0] or "default").strip()
                tag = str(
                    params.get("tag", ["all"])[0] or "all").strip()
                session = self._session(session_id)
                from atomzero.briefing import overdue_view

                result = overdue_view(
                    session.human.data, sort=sort, tag=tag,
                    now=params.get("today", [None])[0])
            except Exception as exc:
                return self._json(400, {"error": str(exc)})
            return self._json(200, result)
        elif (self.path == "/v1/weekly-review/reports/compare/export"
              or self.path.startswith(
                  "/v1/weekly-review/reports/compare/export?")):
            params = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            left = str(params.get("left", [""])[0] or "").strip()
            right = str(params.get("right", [""])[0] or "").strip()
            try:
                archive = _weekly_reports_archive()
                comparison = archive.compare(left, right)
                from atomzero.briefing import (
                    render_weekly_report_comparison_markdown)
                markdown = render_weekly_report_comparison_markdown(
                    comparison)
            except WeeklyReportError as exc:
                code = 404 if "not found" in str(exc) else 400
                return self._json(code, {"error": str(exc)})
            except OSError as exc:
                return self._json(400, {"error": str(exc)})
            filename = (f"atomzero-report-comparison-{left}-{right}.md")
            filename = "".join(
                ch for ch in filename
                if ch.isascii() and (ch.isalnum() or ch in "._-"))
            return self._send(
                200, markdown.encode("utf-8"),
                "text/markdown; charset=utf-8",
                {"Content-Disposition":
                 f'attachment; filename="{filename}"'})
        elif (self.path == "/v1/weekly-review/reports/compare"
              or self.path.startswith(
                  "/v1/weekly-review/reports/compare?")):
            params = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            left = str(params.get("left", [""])[0] or "").strip()
            right = str(params.get("right", [""])[0] or "").strip()
            try:
                archive = _weekly_reports_archive()
                result = archive.compare(left, right)
            except WeeklyReportError as exc:
                code = 404 if "not found" in str(exc) else 400
                return self._json(code, {"error": str(exc)})
            except OSError as exc:
                return self._json(400, {"error": str(exc)})
            return self._json(200, result)
        elif self.path == "/v1/weekly-review/reports/verify":
            try:
                archive = _weekly_reports_archive()
                result = archive.verify()
            except (WeeklyReportError, OSError) as exc:
                return self._json(400, {"error": str(exc)})
            return self._json(200, result)
        elif (self.path == "/v1/weekly-review/reports"
              or self.path.startswith("/v1/weekly-review/reports?")):
            params = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            try:
                archive = _weekly_reports_archive()
            except WeeklyReportError as exc:
                return self._json(400, {"error": str(exc)})
            report_id = str(params.get("id", [""])[0] or "").strip()
            if report_id:
                try:
                    entry, markdown = archive.markdown(report_id)
                except WeeklyReportError as exc:
                    code = (404 if "not found" in str(exc) else 400)
                    return self._json(code, {"error": str(exc)})
                filename = (Path(str(entry.get("path", ""))).name
                            or f"weekly-report-{report_id}.md")
                filename = "".join(
                    ch for ch in filename
                    if ch.isascii() and ch not in '"\\\r\n')
                filename = filename or f"weekly-report-{report_id}.md"
                return self._send(
                    200, markdown.encode("utf-8"),
                    "text/markdown; charset=utf-8",
                    {"Content-Disposition":
                     f'attachment; filename="{filename}"'})
            try:
                limit = int(params.get("limit", ["10"])[0])
                query = str(params.get("query", [""])[0] or "").strip()
                tag = str(params.get("tag", ["all"])[0]
                          or "all").strip().lower().lstrip("#") or "all"
                if tag != "all":
                    from atomzero.briefing import normalize_task_tag
                    tag = normalize_task_tag(tag)
                sort = str(params.get("sort", ["newest"])[0]
                           or "newest").strip().lower()
                entries = archive.list(
                    limit, query=query, order=sort, tag=tag)
                total = archive.count()
                matched = archive.count(query=query, tag=tag)
            except (TypeError, ValueError, WeeklyReportError) as exc:
                return self._json(400, {"error": str(exc)})
            payload = {
                "data": entries,
                "count": len(entries),
                "total": total,
                "index": str(archive.path),
                "sort": sort,
            }
            if query:
                payload["query"] = query
            if tag != "all":
                payload["tag"] = tag
            if query or tag != "all":
                payload["matched"] = matched
            return self._json(200, payload)
        elif (self.path == "/v1/briefing"
              or self.path.startswith("/v1/briefing?")):
            params = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            session_id = (params.get("session_id", ["default"])[0]
                          or "default")
            try:
                session = self._session(session_id)
                from atomzero.briefing import build_briefing

                result = build_briefing(
                    session.human.data, _state.get("gaps"),
                    _state.get("docs"),
                    sort=str(params.get("sort", ["default"])[0]
                             or "default").strip(),
                    tag=str(params.get("tag", ["all"])[0]
                            or "all").strip())
            except Exception as exc:
                return self._json(400, {"error": str(exc)})
            return self._json(200, result)
        elif (self.path == "/v1/weekly-review/export"
              or self.path.startswith("/v1/weekly-review/export?")):
            params = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            session_id = (params.get("session_id", ["default"])[0]
                          or "default")
            try:
                days = int(params.get("days", ["7"])[0])
                session = self._session(session_id)
                from atomzero.briefing import (
                    build_weekly_review, render_weekly_markdown)

                review = build_weekly_review(
                    session.human.data, _state.get("gaps"),
                    _state.get("docs"), days=days,
                    sort=str(params.get("sort", ["default"])[0]
                             or "default").strip(),
                    tag=str(params.get("tag", ["all"])[0]
                            or "all").strip(),
                    audit_action=str(
                        params.get("audit_action", ["all"])[0]
                        or "all").strip())
                markdown = render_weekly_markdown(review)
            except Exception as exc:
                return self._json(400, {"error": str(exc)})
            day = str(review.get("generated", ""))[:10] or "latest"
            filename = f"atomzero-weekly-review-{day}.md"
            return self._send(
                200, markdown.encode("utf-8"),
                "text/markdown; charset=utf-8",
                {"Content-Disposition":
                 f'attachment; filename="{filename}"'})
        elif self.path.startswith("/v1/memory/ledger"):
            params = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            try:
                limit = int(params.get("limit", ["50"])[0])
            except (TypeError, ValueError):
                limit = 50
            ledger = self._memory_ledger()
            self._json(200, {
                "data": ledger.read(limit),
                "summary": ledger.summary(),
            })
        elif self.path == "/v1/documents":
            docs = _state.get("docs")
            self._json(200, {"data": docs.sources() if docs else []})
        elif self.path == "/v1/skills":
            registry = _state.get("skills")
            self._json(200, {
                "dir": _state.get("skills_dir", ""),
                "data": registry.list() if registry else [],
            })
        elif self.path == "/v1/skills/chains":
            chains = _state.get("chains")
            self._json(200, {
                "dir": _state.get("chains_dir", ""),
                "data": chains.list() if chains else [],
            })
        elif (self.path == "/v1/graph"
              or self.path.startswith("/v1/graph?")):
            docs = _state.get("docs")
            if docs is None:
                return self._json(400, {
                    "error": "no document memory configured"})
            params = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            from atomzero.knowledge_graph import build_graph

            try:
                result = build_graph(
                    docs,
                    limit_chunks=int(params.get(
                        "limit_chunks", ["500"])[0]),
                    min_edge=int(params.get("min_edge", ["1"])[0]),
                    max_entities=int(params.get(
                        "max_entities", ["80"])[0]))
            except (TypeError, ValueError) as exc:
                return self._json(400, {"error": str(exc)})
            return self._json(200, result)
        elif (self.path == "/v1/graph/path"
              or self.path.startswith("/v1/graph/path?")):
            docs = _state.get("docs")
            if docs is None:
                return self._json(400, {
                    "error": "no document memory configured"})
            params = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            from atomzero.knowledge_graph import graph_path

            try:
                result = graph_path(
                    docs, params.get("source", [""])[0],
                    params.get("target", [""])[0],
                    max_depth=int(params.get("max_depth", ["4"])[0]))
            except (TypeError, ValueError) as exc:
                return self._json(400, {"error": str(exc)})
            return self._json(200, result)
        elif (self.path == "/v1/graph/entity"
              or self.path.startswith("/v1/graph/entity?")):
            docs = _state.get("docs")
            if docs is None:
                return self._json(400, {
                    "error": "no document memory configured"})
            params = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            from atomzero.knowledge_graph import entity_sources

            try:
                result = entity_sources(
                    docs, params.get("name", [""])[0],
                    limit=int(params.get("limit", ["10"])[0]))
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
            return self._json(200, result)
        elif (self.path == "/v1/graph/relations/path"
              or self.path.startswith("/v1/graph/relations/path?")):
            docs = _state.get("docs")
            if docs is None:
                return self._json(400, {
                    "error": "no document memory configured"})
            params = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            from atomzero.relations import relation_path

            try:
                result = relation_path(
                    docs,
                    params.get("source", [""])[0],
                    params.get("target", [""])[0],
                    max_depth=int(params.get("max_depth", ["3"])[0]))
            except (TypeError, ValueError) as exc:
                return self._json(400, {"error": str(exc)})
            return self._json(200, result)
        elif (self.path == "/v1/graph/relations"
              or self.path.startswith("/v1/graph/relations?")):
            docs = _state.get("docs")
            if docs is None:
                return self._json(400, {
                    "error": "no document memory configured"})
            params = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            from atomzero.relations import extract_relations

            try:
                result = extract_relations(
                    docs,
                    entity=params.get("entity", [""])[0],
                    relation=params.get("relation", [""])[0],
                    source=params.get("source", [""])[0],
                    source_prefix=params.get("source_prefix", [""])[0],
                    limit=int(params.get("limit", ["100"])[0]))
            except SourceNotFound as exc:
                return self._json(404, {"error": str(exc)})
            except (TypeError, ValueError) as exc:
                return self._json(400, {"error": str(exc)})
            return self._json(200, result)
        elif (self.path == "/v1/documents/compare"
              or self.path.startswith("/v1/documents/compare?")):
            docs = _state.get("docs")
            if docs is None:
                return self._json(400, {
                    "error": "no document memory configured"})
            params = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            from atomzero.contradictions import compare_sources

            try:
                result = compare_sources(
                    docs, params.get("source", [""])[0],
                    against=params.get("against", [""])[0],
                    max_claims=int(params.get("max_claims", ["200"])[0]))
            except (TypeError, ValueError) as exc:
                return self._json(400, {"error": str(exc)})
            return self._json(200, result)
        elif (self.path == "/v1/documents/timeline"
              or self.path.startswith("/v1/documents/timeline?")):
            docs = _state.get("docs")
            if docs is None:
                return self._json(400, {
                    "error": "no document memory configured"})
            params = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            from atomzero.timeline import extract_timeline

            try:
                result = extract_timeline(
                    docs,
                    source=params.get("source", [""])[0],
                    source_prefix=params.get("source_prefix", [""])[0],
                    entity=params.get("entity", [""])[0],
                    limit=int(params.get("limit", ["100"])[0]))
            except SourceNotFound as exc:
                return self._json(404, {"error": str(exc)})
            except (TypeError, ValueError) as exc:
                return self._json(400, {"error": str(exc)})
            return self._json(200, result)
        elif (self.path == "/v1/documents/evidence"
              or self.path.startswith("/v1/documents/evidence?")):
            docs = _state.get("docs")
            if docs is None:
                return self._json(400, {
                    "error": "no document memory configured"})
            params = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            source = params.get("source", [""])[0]
            section = params.get("section", [""])[0]
            try:
                raw_page = params.get("page", [None])[0]
                page = (int(raw_page)
                        if raw_page not in (None, "") else None)
                result = docs.source_evidence(
                    source, page=page, section=section,
                    limit=int(params.get("limit", ["20"])[0]))
            except SourceNotFound as exc:
                return self._json(404, {"error": str(exc)})
            except (TypeError, ValueError) as exc:
                return self._json(400, {"error": str(exc)})
            return self._json(200, result)
        elif (self.path == "/v1/documents/summary"
              or self.path.startswith("/v1/documents/summary?")):
            docs = _state.get("docs")
            if docs is None:
                return self._json(400, {
                    "error": "no document memory configured"})
            params = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            source = params.get("source", [""])[0].strip()
            prefix = params.get("source_prefix", [""])[0].strip()
            kind = params.get("kind", [""])[0].strip()
            if not source and not prefix and not kind:
                return self._json(400, {
                    "error": "source or source_prefix is required"})
            try:
                if source:
                    result = docs.summarize_source(
                        source,
                        max_sentences=int(
                            params.get("sentences", ["5"])[0]))
                else:
                    result = docs.summarize_collection(
                        source_prefix=prefix, kind=kind,
                        max_sentences=int(
                            params.get("sentences", ["8"])[0]))
            except SourceNotFound as exc:
                return self._json(404, {"error": str(exc)})
            except (TypeError, ValueError) as exc:
                return self._json(400, {"error": str(exc)})
            return self._json(200, result)
        elif self.path == "/v1/teacher/notes":
            docs = _state.get("docs")
            notes = ([source for source in docs.sources()
                      if source.get("kind") == "teacher"]
                     if docs else [])
            self._json(200, {
                "data": notes,
                "stats": {
                    "total": len(notes),
                    "verified": sum(1 for note in notes
                                    if note.get("verified")),
                    "unverified": sum(1 for note in notes
                                      if not note.get("verified")),
                },
            })
        elif self.path == "/v1/learning/queue":
            store = _state.get("feedback")
            if store is None:
                return self._json(400, {
                    "error": "feedback store is not configured"})
            self._json(200, {"data": store.list(), "stats": store.stats()})
        elif self.path == "/v1/learning/gaps":
            store = _state.get("gaps")
            if store is None:
                return self._json(400, {
                    "error": "knowledge-gap store is not configured"})
            self._json(200, {"data": store.list(), "stats": store.stats()})
        elif self.path == "/v1/learning/adapters":
            entries = []
            history = Path(_state.get(
                "adapter_history_path", "data/adapter_history.jsonl"))
            if history.exists():
                for line in history.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            active = ""
            active_path = Path(_state.get(
                "active_adapter_file", "data/active_adapter.json"))
            if active_path.exists():
                try:
                    active = str(json.loads(
                        active_path.read_text(encoding="utf-8")).get("path", ""))
                except (OSError, json.JSONDecodeError):
                    active = ""
            self._json(200, {
                "data": entries, "active": active,
                "history": str(history), "active_file": str(active_path),
            })
        elif self.path == "/v1/curiosity":
            queue = _state.get("research_queue")
            gaps = _state.get("gaps")
            if queue is None or gaps is None:
                return self._json(400, {
                    "error": "research queue or gap store is not configured"})
            self._json(200, {
                "items": curiosity_items(queue),
                "pending_gaps": sum(
                    1 for gap in gaps.list()
                    if gap.get("status") == "pending"),
                "stats": queue.stats(),
            })
        elif self.path == "/v1/research/queue":
            queue = _state.get("research_queue")
            if queue is None:
                return self._json(400, {
                    "error": "research queue is not configured"})
            self._json(200, {"data": self._queue_items(queue),
                             "stats": queue.stats()})
        elif self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        if route not in ("/generate", "/v1/chat/completions",
                         "/v1/documents", "/v1/documents/search",
                         "/v1/documents/answer",
                         "/v1/documents/upload", "/v1/documents/forget",
                         "/v1/memory/forget", "/v1/memory/history",
                         "/v1/memory/ledger/verify",
                         "/v1/memory/export", "/v1/memory/import",
                         "/v1/teacher/notes",
                         "/v1/teacher/notes/verify", "/v1/teacher/ask",
                         "/v1/teach", "/v1/learning/feedback",
                         "/v1/learning/queue/approve",
                         "/v1/learning/queue/reject",
                         "/v1/learning/queue/export",
                         "/v1/learning/gaps/propose",
                         "/v1/learning/gaps/dismiss",
                         "/v1/curiosity/propose", "/v1/hungry/run",
                         "/v1/skills/run", "/v1/skills/chain",
                         "/v1/skills/chains/run",
                         "/v1/briefing", "/v1/weekly-review",
                         "/v1/tasks/reschedule", "/v1/tasks/priority",
                         "/v1/tasks/delete", "/v1/tasks/tags/rename",
                         "/v1/tasks/tags/delete",
                         "/v1/tasks/tags/undo-delete",
                         "/v1/weekly-review/reports/prune",
                         "/v1/weekly-review/reports/keep",
                         "/v1/weekly-review/reports/delete",
                         "/v1/weekly-review/reports/repair",
                         "/v1/goals/plan",
                         "/v1/learn_url", "/v1/research",
                         "/v1/research/queue",
                         "/v1/research/queue/approve",
                         "/v1/research/queue/reject",
                         "/v1/research/queue/run",
                         "/v1/agent/chat", "/v1/agent/plan",
                         "/v1/tools/audit", "/v1/claims/audit",
                         "/v1/consensus", "/v1/freshness",
                         "/v1/temporal/filter",
                         "/v1/improvement/plan", "/v1/improvement/decision",
                         "/v1/scoreboard/record", "/v1/scoreboard/summary",
                         "/v1/scoreboard/verify",
                         "/v1/reliability/summary", "/v1/reliability/verify",
                         "/v1/documents/diff", "/v1/proactive",
                         "/v1/proactive/config", "/v1/proactive/ack"):
            return self._json(404, {"error": "not found"})
        key = _state.get("api_key") or ""
        if key:
            auth = self.headers.get("Authorization", "")
            if auth != f"Bearer {key}":
                return self._json(401, {"error": "unauthorized"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._json(400, {"error": "invalid Content-Length"})
        if n < 0:
            return self._json(400, {"error": "invalid Content-Length"})
        # Raw uploads have their own limit; a 3 MB PDF must not be rejected by
        # the smaller JSON body cap before _upload sees it.
        if route == "/v1/documents/upload":
            return self._upload(parsed.query, n)
        if n > MAX_BODY_BYTES:
            if n <= 1_048_576:
                try:
                    self.rfile.read(n)
                except OSError:
                    pass
            else:
                self.close_connection = True
            return self._json(413, {"error": "request body is too large"})
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._json(400, {"error": "invalid JSON"})
        if not isinstance(req, dict):
            return self._json(400, {"error": "JSON object required"})
        try:
            if route == "/v1/documents":
                return self._ingest(req)
            if route == "/v1/teacher/notes":
                return self._teacher_note(req)
            if route == "/v1/teacher/notes/verify":
                return self._teacher_verify(req)
            if route == "/v1/teacher/ask":
                return self._teacher_ask(req)
            if route == "/v1/learning/feedback":
                return self._feedback_add(req)
            if route == "/v1/learning/queue/approve":
                return self._feedback_update(req, "approve")
            if route == "/v1/learning/queue/reject":
                return self._feedback_update(req, "reject")
            if route == "/v1/learning/queue/export":
                return self._feedback_export(req)
            if route == "/v1/learning/gaps/propose":
                return self._gap_propose(req)
            if route == "/v1/learning/gaps/dismiss":
                return self._gap_dismiss(req)
            if route == "/v1/curiosity/propose":
                return self._curiosity_propose(req)
            if route == "/v1/hungry/run":
                return self._hungry_run(req)
            if route == "/v1/skills/run":
                return self._skills_run(req)
            if route == "/v1/skills/chain":
                return self._skills_chain(req)
            if route == "/v1/skills/chains/run":
                return self._skills_recipe_run(req)
            if route == "/v1/briefing":
                return self._briefing(req)
            if route == "/v1/weekly-review":
                return self._weekly_review(req)
            if route == "/v1/weekly-review/reports/prune":
                return self._weekly_reports_prune()
            if route == "/v1/weekly-review/reports/keep":
                return self._weekly_reports_keep(req)
            if route == "/v1/weekly-review/reports/delete":
                return self._weekly_reports_delete(req)
            if route == "/v1/weekly-review/reports/repair":
                return self._weekly_reports_repair()
            if route == "/v1/tasks/reschedule":
                return self._tasks_reschedule(req)
            if route == "/v1/tasks/priority":
                return self._tasks_priority(req)
            if route == "/v1/tasks/delete":
                return self._tasks_delete(req)
            if route == "/v1/tasks/tags/rename":
                return self._tasks_tags_rename(req)
            if route == "/v1/tasks/tags/delete":
                return self._tasks_tags_delete(req)
            if route == "/v1/tasks/tags/undo-delete":
                return self._tasks_tags_undo_delete(req)
            if route == "/v1/goals/plan":
                return self._goal_plan(req)
            if route == "/v1/research/queue":
                return self._queue_add(req)
            if route == "/v1/research/queue/approve":
                return self._queue_update(req, "approve")
            if route == "/v1/research/queue/reject":
                return self._queue_update(req, "reject")
            if route == "/v1/research/queue/run":
                return self._queue_run(req)
            if route == "/v1/agent/plan":
                return self._agent_plan(req)
            if route == "/v1/tools/audit":
                return self._tools_audit(req)
            if route == "/v1/claims/audit":
                return self._claims_audit(req)
            if route == "/v1/consensus":
                return self._consensus_audit(req)
            if route == "/v1/freshness":
                return self._freshness_audit(req)
            if route == "/v1/temporal/filter":
                return self._temporal_filter(req)
            if route == "/v1/improvement/plan":
                return self._improvement_plan(req)
            if route == "/v1/improvement/decision":
                return self._improvement_decision(req)
            if route == "/v1/scoreboard/record":
                return self._scoreboard_record(req)
            if route == "/v1/scoreboard/summary":
                return self._scoreboard_summary(req)
            if route == "/v1/scoreboard/verify":
                return self._scoreboard_verify(req)
            if route == "/v1/reliability/summary":
                return self._reliability_summary(req)
            if route == "/v1/reliability/verify":
                return self._reliability_verify(req)
            if route == "/v1/documents/diff":
                return self._documents_diff(req)
            if route == "/v1/proactive":
                return self._proactive_next(req)
            if route == "/v1/proactive/config":
                return self._proactive_config(req)
            if route == "/v1/proactive/ack":
                return self._proactive_ack(req)
            if route == "/v1/documents/forget":
                return self._documents_forget(req)
            if route == "/v1/memory/forget":
                return self._memory_forget(req)
            if route == "/v1/memory/ledger/verify":
                return self._json(200, self._memory_ledger().verify())
            if route == "/v1/memory/history":
                return self._memory_history(req)
            if route == "/v1/memory/export":
                return self._memory_export(req)
            if route == "/v1/memory/import":
                return self._memory_import(req)
            if self.path == "/v1/documents/search":
                docs = _state.get("docs")
                query = str(req.get("query", ""))
                try:
                    k = int(req.get("k", 5))
                except (TypeError, ValueError):
                    return self._json(400, {"error": "k must be an integer"})
                source = str(req.get("source", "") or "").strip()
                source_prefix = str(
                    req.get("source_prefix", "") or "").strip()
                kind = str(req.get("kind", "") or "").strip()
                diverse = bool(req.get("diverse", False))
                try:
                    hits = (docs.search(
                        query, k=k, source=source,
                        source_prefix=source_prefix, kind=kind,
                        diverse=diverse)
                        if docs else [])
                except ValueError as exc:
                    return self._json(400, {"error": str(exc)})
                return self._json(200, {"data": hits})
            if self.path == "/v1/documents/answer":
                docs = _state.get("docs")
                if docs is None:
                    return self._json(400, {
                        "error": "no document memory"})
                query = str(req.get("query", "") or "").strip()
                if not query:
                    return self._json(400, {"error": "empty query"})
                source = str(req.get("source", "") or "").strip()
                source_prefix = str(
                    req.get("source_prefix", "") or "").strip()
                kind = str(req.get("kind", "") or "").strip()
                try:
                    max_chars = int(req.get("max_context_chars", 0) or 0)
                except (TypeError, ValueError):
                    return self._json(400, {
                        "error": "max_context_chars must be an integer"})
                if max_chars < 0 or max_chars > 1_000_000:
                    return self._json(400, {
                        "error": "max_context_chars out of range"})
                try:
                    answer = doc_answer(
                        query, docs, source=source,
                        source_prefix=source_prefix, kind=kind,
                        max_context_chars=max_chars)
                except ValueError as exc:
                    return self._json(400, {"error": str(exc)})
                if answer is None:
                    return self._json(200, {
                        "text": "I do not have a source for that yet.",
                        "sources": [],
                        "evidence": [],
                        "confidence": {"score": 0.0, "label": "none",
                                       "coverage": 0.0, "margin": 0.0},
                    })
                return self._json(200, answer)
            if self.path == "/v1/teach":
                return self._teach(req)
            if self.path == "/v1/learn_url":
                if not _state.get("allow_url_fetch"):
                    return self._json(403, {
                        "error": "URL fetch disabled; start with "
                                 "--allow-url-fetch"})
                docs = _state.get("docs")
                if docs is None:
                    return self._json(400, {"error": "no document memory"})
                from atomzero.web_research import index_url
                try:
                    page = index_url(docs, str(req.get("url", "")))
                except Exception as exc:
                    return self._json(400, {"error": str(exc)})
                return self._json(200, page)
            if self.path == "/v1/research":
                docs = _state.get("docs")
                if docs is None:
                    return self._json(400, {"error": "no document memory"})
                from atomzero.web_research import wikipedia_research
                query = str(req.get("query", "")).strip()
                if not query:
                    return self._json(400, {"error": "empty query"})
                try:
                    pages = wikipedia_research(
                        docs, query, _state.get("wiki_lang", "en"),
                        int(req.get("limit", 2)))
                except Exception as exc:
                    return self._json(400, {"error": str(exc)})
                return self._json(200, {"pages": pages})
            session_id = str(req.get("session_id", "")).strip()
            is_agent = self.path == "/v1/agent/chat"
            if is_agent or (self.path == "/v1/chat/completions"
                            and session_id):
                messages = req.get("messages") or []
                message = str(req.get("message", "")).strip()
                if not message:
                    message = next(
                        (str(m.get("content", "")) for m in reversed(messages)
                         if m.get("role") == "user"), "")
                if not message:
                    return self._json(400, {"error": "empty message"})
                try:
                    safety = _state.get("safety")
                    verdict = safety.check(message) if safety else None
                    copy_result = (copy_answer(message)
                                   if verdict is None else None)
                    numeric_result = (numeric_answer(messages or [message])
                                      if verdict is None else None)
                    recall_result = (recall_answer(messages or [message])
                                     if verdict is None else None)
                    identity_result = (identity_answer(messages or [message])
                                       if verdict is None else None)
                    if numeric_result is not None:
                        result = {
                            "reply": numeric_result["text"],
                            "tool_calls": [{
                                "name": "numeric_tool",
                                "kind": numeric_result["kind"],
                            }],
                            "sources": [],
                            "source": "numeric_tool",
                        }
                        result_session_id = session_id or "default"
                    elif recall_result is not None:
                        result = {
                            "reply": recall_result["text"],
                            "tool_calls": [{
                                "name": "recall_tool",
                                "kind": recall_result["kind"],
                            }],
                            "sources": [],
                            "source": "recall_tool",
                        }
                        result_session_id = session_id or "default"
                    elif identity_result is not None:
                        result = {
                            "reply": identity_result["text"],
                            "tool_calls": [{
                                "name": "identity_tool",
                                "kind": identity_result["kind"],
                            }],
                            "sources": [],
                            "source": "identity_tool",
                        }
                        result_session_id = session_id or "default"
                    elif copy_result is not None:
                        result = {
                            "reply": copy_result["text"],
                            "tool_calls": [{
                                "name": "copy_tool",
                                "format": copy_result["format"],
                                "pairs": copy_result["pairs_count"],
                            }],
                            "sources": [],
                            "source": "copy_tool",
                        }
                        result_session_id = session_id or "default"
                    else:
                        session = self._session(session_id or "default")
                        result = session.chat(message)
                        result_session_id = session.id
                except Exception as exc:
                    return self._json(400, {"error": str(exc)})
                if is_agent:
                    return self._json(200, {"session_id": result_session_id,
                                            **result})
                payload = {
                    "id": "tiny-agent-" + str(abs(hash(result["reply"]))
                                              % 10**12),
                    "object": "chat.completion",
                    "model": _state["name"],
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant",
                                    "content": result["reply"]},
                        "finish_reason": "stop",
                    }],
                    "tool_calls": result.get("tool_calls", []),
                    "sources": result.get("sources", []),
                }
                if result.get("source"):
                    payload["source"] = result["source"]
                return self._json(200, payload)
            if self.path == "/v1/chat/completions":
                messages = req.get("messages") or []
                prompt = chat_prompt(messages)
                max_new = int(req.get("max_tokens", 64))
                doc_query = next(
                    (str(m.get("content", "")) for m in reversed(messages)
                     if m.get("role") == "user"), "")
            else:
                doc_query = str(req.get("prompt", "hello"))
                prompt = f"<|user|> {doc_query} <|assistant|>"
                max_new = int(req.get("max_new_tokens", 64))
            max_new = max(1, min(max_new, 256))
            safety = _state.get("safety")
            verdict = safety.check(doc_query) if safety else None
            numeric_turns = (messages if self.path == "/v1/chat/completions"
                             else [doc_query])
            numeric_result = (numeric_answer(numeric_turns)
                              if verdict is None else None)
            if numeric_result is not None:
                _state["metrics"]["requests"] += 1
                if self.path == "/generate":
                    return self._json(200, {
                        "text": numeric_result["text"],
                        "sources": [],
                        "model": _state["name"],
                        "source": "numeric_tool",
                        "kind": numeric_result["kind"],
                    })
                return self._json(200, {
                    "id": "tiny-numeric-" + str(abs(hash(numeric_result["text"]))
                                                % 10**12),
                    "object": "chat.completion",
                    "model": _state["name"],
                    "source": "numeric_tool",
                    "sources": [],
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant",
                                    "content": numeric_result["text"]},
                        "finish_reason": "stop",
                    }],
                })
            recall_result = (recall_answer(numeric_turns)
                             if verdict is None else None)
            if recall_result is not None:
                _state["metrics"]["requests"] += 1
                if self.path == "/generate":
                    return self._json(200, {
                        "text": recall_result["text"],
                        "sources": [],
                        "model": _state["name"],
                        "source": "recall_tool",
                        "kind": recall_result["kind"],
                    })
                return self._json(200, {
                    "id": "tiny-recall-" + str(abs(hash(recall_result["text"]))
                                               % 10**12),
                    "object": "chat.completion",
                    "model": _state["name"],
                    "source": "recall_tool",
                    "sources": [],
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant",
                                    "content": recall_result["text"]},
                        "finish_reason": "stop",
                    }],
                })
            identity_result = (identity_answer(numeric_turns)
                               if verdict is None else None)
            if identity_result is not None:
                _state["metrics"]["requests"] += 1
                if self.path == "/generate":
                    return self._json(200, {
                        "text": identity_result["text"],
                        "sources": [],
                        "model": _state["name"],
                        "source": "identity_tool",
                        "kind": identity_result["kind"],
                    })
                return self._json(200, {
                    "id": "tiny-identity-" + str(abs(hash(identity_result["text"]))
                                                 % 10**12),
                    "object": "chat.completion",
                    "model": _state["name"],
                    "source": "identity_tool",
                    "sources": [],
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant",
                                    "content": identity_result["text"]},
                        "finish_reason": "stop",
                    }],
                })
            copy_result = (copy_answer(doc_query)
                           if verdict is None else None)
            if copy_result is not None:
                _state["metrics"]["requests"] += 1
                if self.path == "/generate":
                    return self._json(200, {
                        "text": copy_result["text"],
                        "sources": [],
                        "model": _state["name"],
                        "source": "copy_tool",
                        "format": copy_result["format"],
                        "pairs_count": copy_result["pairs_count"],
                    })
                return self._json(200, {
                    "id": "tiny-copy-" + str(abs(hash(copy_result["text"]))
                                             % 10**12),
                    "object": "chat.completion",
                    "model": _state["name"],
                    "source": "copy_tool",
                    "sources": [],
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant",
                                    "content": copy_result["text"]},
                        "finish_reason": "stop",
                    }],
                })
            if verdict:
                grounded = {"text": safety.refusal(verdict["category"]),
                            "sources": []}
            else:
                grounded = doc_answer(doc_query, _state.get("docs"))
                if grounded and safety:
                    safe_text, out_verdict = safety.filter_reply(
                        grounded["text"])
                    if out_verdict:
                        grounded = {**grounded, "text": safe_text,
                                    "sources": []}
                        verdict = out_verdict
            if grounded:
                _state["metrics"]["requests"] += 1
                if self.path == "/generate":
                    payload = {
                        "text": grounded["text"],
                        "sources": grounded["sources"],
                        "model": _state["name"],
                    }
                    if verdict:
                        payload["safety"] = verdict
                    return self._json(200, payload)
                payload = {
                    "id": "tiny-doc-" + str(abs(hash(grounded["text"]))
                                            % 10**12),
                    "object": "chat.completion",
                    "model": _state["name"],
                    "sources": grounded["sources"],
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant",
                                    "content": grounded["text"]},
                        "finish_reason": "stop",
                    }],
                }
                if verdict:
                    payload["safety"] = verdict
                return self._json(200, payload)
            temperature = float(req.get("temperature", 0.7))
            top_k = int(req.get("top_k", 20))
            repetition_penalty = float(req.get("repetition_penalty", 1.1))
            with _lock:
                text, stop = generate(
                    _state["model"], _state["tok"], prompt,
                    max_new=max_new, temperature=temperature, top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    banned_ids=(_state["user_id"], _state["assistant_id"]),
                    device=_state["device"])
                _state["metrics"]["requests"] += 1
                _state["metrics"]["generated_tokens"] += max_new
            if safety:
                safe_text, out_verdict = safety.filter_reply(text)
                if out_verdict:
                    text = safe_text
                    verdict = out_verdict
            if self.path == "/generate":
                payload = {"text": text, "stop": stop,
                           "model": _state["name"]}
                if verdict:
                    payload["safety"] = verdict
                return self._json(200, payload)
            payload = {
                "id": "tiny-" + str(abs(hash(text)) % 10**12),
                "object": "chat.completion",
                "model": _state["name"],
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
            }
            if verdict:
                payload["safety"] = verdict
            return self._json(200, payload)
        except Exception as exc:  # pragma: no cover
            return self._json(400, {"error": str(exc)})

    def _feedback_store(self):
        store = _state.get("feedback")
        if store is None:
            raise FeedbackError("feedback store is not configured")
        return store

    def _feedback_add(self, req: dict):
        try:
            item = self._feedback_store().add(
                req.get("user", ""), req.get("assistant", ""),
                req.get("rating", 0), req.get("corrected", ""),
                req.get("session_id", ""))
        except FeedbackError as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, item)

    def _feedback_update(self, req: dict, action: str):
        try:
            store = self._feedback_store()
            item_id = str(req.get("id", ""))
            item = (store.approve(item_id) if action == "approve"
                    else store.reject(item_id))
        except FeedbackNotFound as exc:
            return self._json(404, {"error": str(exc)})
        except FeedbackError as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, item)

    def _feedback_export(self, req: dict):
        try:
            result = self._feedback_store().export_training(
                _state.get("feedback_export_path",
                           "data/feedback_training.jsonl"))
        except FeedbackError as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, result)

    def _gap_store(self):
        store = _state.get("gaps")
        if store is None:
            raise KnowledgeGapError(
                "knowledge-gap store is not configured")
        return store

    def _gap_propose(self, req: dict):
        try:
            gaps = self._gap_store()
            gap_id = str(req.get("id", ""))
            gap = gaps.get(gap_id)
            if gap.get("status") != "pending":
                raise KnowledgeGapError(
                    f"gap is already {gap.get('status')}")
            item = self._queue().add(
                gap["query"], _state.get("wiki_lang", "en"),
                int(req.get("pages", 1)))
            gap = gaps.mark_proposed(gap_id, item["id"])
        except GapNotFound as exc:
            return self._json(404, {"error": str(exc)})
        except (KnowledgeGapError, ResearchQueueError, ValueError) as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, {
            "gap": gap, "research": item,
            "next": (f"approve {item['id']} in the research queue, then run "
                     "the scheduled worker"),
        })

    def _curiosity_propose(self, req: dict):
        try:
            result = propose_from_gaps(
                self._gap_store(), self._queue(),
                limit=req.get("limit", 3),
                lang=_state.get("wiki_lang", "en"),
                pages=int(req.get("pages", 1)),
                min_count=req.get("min_count", 1))
        except (CuriosityError, KnowledgeGapError, ResearchQueueError,
                ValueError) as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, result)

    def _hungry_run(self, req: dict):
        docs = _state.get("docs")
        gaps = _state.get("gaps")
        if docs is None or gaps is None:
            return self._json(400, {
                "error": "document memory and gap store are required"})
        dry_run = bool(req.get("dry_run", False))
        teacher = _state.get("teacher")
        if teacher is None and not dry_run:
            return self._json(400, {
                "error": "teacher provider is not configured; use dry_run "
                         "or start with --provider"})
        topics = req.get("topics") or []
        if isinstance(topics, str):
            topics = [topics]
        if req.get("topic"):
            topics.append(str(req["topic"]))
        try:
            from atomzero.hungry import run_hungry

            result = run_hungry(
                docs, gaps, teacher, topics=topics,
                max_questions=req.get("max_questions", 3),
                max_provider_calls=req.get("max_provider_calls", 6),
                question_limit=req.get("question_limit", 2),
                min_confidence=req.get("min_confidence", 0.55),
                expand=bool(req.get("expand", False)),
                dry_run=dry_run,
                ledger_path=_state.get("hungry_ledger_path"))
        except (ValueError, OSError) as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, result)

    def _skills_run(self, req: dict):
        registry = _state.get("skills")
        if registry is None:
            return self._json(400, {"error": "no skills configured"})
        name = str(req.get("name", "")).strip()
        query = str(req.get("query", "")).strip()
        if not name or not query:
            return self._json(400, {"error": "name and query are required"})
        skill = next(
            (item for item in registry.skills if item["name"] == name), None)
        if skill is None:
            return self._json(404, {"error": "skill not found"})
        safety = _state.get("safety")
        if safety is not None:
            verdict = safety.check(query)
            if verdict is not None:
                return self._json(400, {
                    "error": safety.refusal(verdict["category"]),
                    "safety": verdict,
                })
        try:
            from atomzero.skills import execute_skill

            result = execute_skill(
                skill, query, _state.get("docs"),
                _state.get("generator"))
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        if safety is not None:
            safe, verdict = safety.filter_reply(result.get("reply", ""))
            if verdict is not None:
                result = {**result, "reply": safe, "safety": verdict}
        return self._json(200, result)

    def _skills_chain(self, req: dict):
        registry = _state.get("skills")
        if registry is None:
            return self._json(400, {"error": "no skills configured"})
        steps = req.get("steps") or req.get("skills")
        if isinstance(steps, str):
            steps = [item.strip() for item in steps.split(",")
                     if item.strip()]
        query = str(req.get("query", "")).strip()
        if not steps or not query:
            return self._json(400, {
                "error": "steps/skills and query are required"})
        safety = _state.get("safety")
        if safety is not None:
            verdict = safety.check(query)
            if verdict is not None:
                return self._json(400, {
                    "error": safety.refusal(verdict["category"]),
                    "safety": verdict,
                })
        try:
            from atomzero.skills import execute_chain

            result = execute_chain(
                registry, steps, query, _state.get("docs"),
                _state.get("generator"),
                carry_context=bool(req.get("carry_context", True)))
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        if safety is not None:
            safe, verdict = safety.filter_reply(result.get("reply", ""))
            if verdict is not None:
                result = {**result, "reply": safe, "safety": verdict}
        return self._json(200, result)

    def _skills_recipe_run(self, req: dict):
        chains = _state.get("chains")
        skills = _state.get("skills")
        if chains is None or skills is None:
            return self._json(400, {
                "error": "skills and chain recipes are required"})
        name = str(req.get("name", "")).strip()
        query = str(req.get("query", "")).strip()
        if not name or not query:
            return self._json(400, {"error": "name and query are required"})
        from atomzero.skills import SkillError, execute_chain_recipe

        try:
            recipe = chains.get(name)
        except SkillError as exc:
            return self._json(404, {"error": str(exc)})
        safety = _state.get("safety")
        if safety is not None:
            verdict = safety.check(query)
            if verdict is not None:
                return self._json(400, {
                    "error": safety.refusal(verdict["category"]),
                    "safety": verdict,
                })
        try:
            carry = req.get("carry_context", None)
            if carry is not None:
                carry = bool(carry)
            result = execute_chain_recipe(
                recipe, query, skills, _state.get("docs"),
                _state.get("generator"), carry_context=carry)
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        if safety is not None:
            safe, verdict = safety.filter_reply(result.get("reply", ""))
            if verdict is not None:
                result = {**result, "reply": safe, "safety": verdict}
        return self._json(200, result)

    def _briefing(self, req: dict):
        session_id = str(req.get("session_id", "default")).strip() or "default"
        try:
            session = self._session(session_id)
            from atomzero.briefing import build_briefing

            result = build_briefing(
                session.human.data, _state.get("gaps"), _state.get("docs"),
                sort=str(req.get("sort", "default") or "default").strip(),
                tag=str(req.get("tag", "all") or "all").strip())
        except Exception as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, result)

    def _weekly_reports_prune(self):
        try:
            archive = _weekly_reports_archive()
            removed = archive.prune()
            remaining = archive.count()
        except (WeeklyReportError, OSError) as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, {
            "removed": removed,
            "remaining": remaining,
            "index": str(archive.path),
        })

    def _tasks_delete(self, req: dict):
        session_id = str(req.get("session_id", "default")).strip() or "default"
        try:
            task_id = int(req.get("id"))
            session = self._session(session_id)
            task = next((item for item in session.human.data.get("tasks", [])
                         if int(item.get("id", 0)) == task_id), None)
        except (TypeError, ValueError) as exc:
            return self._json(400, {"error": str(exc)})
        if task is None:
            return self._json(404, {"error": "task not found"})
        if req.get("confirm") is not True:
            return self._json(400, {
                "error": "confirmation required",
                "task": task,
            })
        deleted = session.human.delete_task(task_id)
        return self._json(200, {
            "session_id": session_id,
            "deleted": deleted,
        })

    def _tasks_tags_rename(self, req: dict):
        session_id = str(req.get("session_id", "default")).strip() or "default"
        from atomzero.briefing import normalize_tag_rename, task_tag_view

        try:
            old, new = normalize_tag_rename(req.get("old"), req.get("new"))
            session = self._session(session_id)
            view = task_tag_view(session.human.data)
            entry = next((item for item in view["tags"]
                          if item["tag"] == old), None)
        except (TypeError, ValueError) as exc:
            return self._json(400, {"error": str(exc)})
        except Exception as exc:
            return self._json(400, {"error": str(exc)})
        if entry is None:
            return self._json(404, {"error": f"tag '{old}' is not used"})
        if req.get("confirm") is not True:
            return self._json(400, {
                "error": "confirmation required",
                "old": old,
                "new": new,
                "would_update": entry["total"],
                "open": entry["open"],
                "done": entry["done"],
            })
        try:
            result = session.human.rename_task_tag(old, new)
        except (TypeError, ValueError) as exc:
            return self._json(400, {"error": str(exc)})
        except Exception as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, {
            "session_id": session_id,
            "applied": True,
            **result,
        })

    def _tasks_tags_delete(self, req: dict):
        session_id = str(req.get("session_id", "default")).strip() or "default"
        from atomzero.briefing import normalize_task_tag, task_tag_view

        try:
            tag = normalize_task_tag(req.get("tag"))
            session = self._session(session_id)
            view = task_tag_view(session.human.data)
            entry = next((item for item in view["tags"]
                          if item["tag"] == tag), None)
        except (TypeError, ValueError) as exc:
            return self._json(400, {"error": str(exc)})
        except Exception as exc:
            return self._json(400, {"error": str(exc)})
        if entry is None:
            return self._json(404, {"error": f"tag '{tag}' is not used"})
        if req.get("confirm") is not True:
            return self._json(400, {
                "error": "confirmation required",
                "tag": tag,
                "would_update": entry["total"],
                "open": entry["open"],
                "done": entry["done"],
            })
        try:
            result = session.human.delete_task_tag(tag)
        except (TypeError, ValueError) as exc:
            return self._json(400, {"error": str(exc)})
        except Exception as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, {
            "session_id": session_id,
            "applied": True,
            **result,
        })

    def _tasks_tags_undo_delete(self, req: dict):
        session_id = str(req.get("session_id", "default")).strip() or "default"
        try:
            session = self._session(session_id)
            result = session.human.undo_delete_task_tag()
        except ValueError as exc:
            code = 404 if "no tag delete" in str(exc).lower() else 400
            return self._json(code, {"error": str(exc)})
        except Exception as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, {
            "session_id": session_id,
            **result,
        })

    def _tasks_priority(self, req: dict):
        session_id = str(req.get("session_id", "default")).strip() or "default"
        priority = str(req.get("priority", "") or "").strip().lower()
        if not priority:
            return self._json(400, {"error": "priority is required"})
        try:
            task_id = int(req.get("id"))
            session = self._session(session_id)
            task = session.human.set_task_priority(task_id, priority)
        except (TypeError, ValueError) as exc:
            return self._json(400, {"error": str(exc)})
        except Exception as exc:
            return self._json(400, {"error": str(exc)})
        if task is None:
            return self._json(404, {"error": "task not found"})
        return self._json(200, {
            "session_id": session_id,
            "task": task,
        })

    def _tasks_reschedule(self, req: dict):
        session_id = str(req.get("session_id", "default")).strip() or "default"
        try:
            task_id = int(req.get("id"))
            due = str(req.get("due", "") or "").strip()
            session = self._session(session_id)
            task = session.human.reschedule_task(task_id, due)
        except (TypeError, ValueError) as exc:
            return self._json(400, {"error": str(exc)})
        except Exception as exc:
            return self._json(400, {"error": str(exc)})
        if task is None:
            return self._json(404, {"error": "task not found"})
        return self._json(200, {
            "session_id": session_id,
            "task": task,
        })

    def _weekly_reports_repair(self):
        try:
            archive = _weekly_reports_archive()
            result = archive.repair()
        except (WeeklyReportError, OSError) as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, result)

    def _weekly_reports_delete(self, req: dict):
        try:
            archive = _weekly_reports_archive()
            deleted = archive.delete(str(req.get("id", "") or ""))
            remaining = archive.count()
        except WeeklyReportError as exc:
            code = 404 if "not found" in str(exc) else 400
            return self._json(code, {"error": str(exc)})
        except OSError as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, {
            "deleted": deleted,
            "remaining": remaining,
            "index": str(archive.path),
        })

    def _weekly_reports_keep(self, req: dict):
        try:
            keep = int(req.get("keep"))
            if keep < 1:
                raise ValueError("keep must be at least 1")
            dry_run = req.get("dry_run") is True
            archive = _weekly_reports_archive()
            if dry_run:
                preview = archive.keep_preview(keep)
                remaining = archive.count()
                return self._json(200, {
                    "keep": keep,
                    "dry_run": True,
                    "trimmed": 0,
                    "remaining": remaining,
                    "would_remove": preview,
                    "would_remove_count": len(preview),
                    "index": str(archive.path),
                })
            trimmed = archive.keep(keep)
            remaining = archive.count()
        except (TypeError, ValueError, WeeklyReportError, OSError) as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, {
            "keep": keep,
            "trimmed": trimmed,
            "remaining": remaining,
            "index": str(archive.path),
        })

    def _weekly_review(self, req: dict):
        session_id = str(req.get("session_id", "default")).strip() or "default"
        try:
            session = self._session(session_id)
            from atomzero.briefing import build_weekly_review
            from atomzero.improvement import summarize_ledger

            try:
                _, ledger = self._improvement_ledger()
                improvement = summarize_ledger(ledger)
            except (OSError, ValueError):
                improvement = {}
            try:
                from atomzero.scoreboard import load_jsonl, summarize_runs
                scoreboard = summarize_runs(
                    load_jsonl(self._scoreboard_file()))
            except (OSError, ValueError):
                scoreboard = {}
            result = build_weekly_review(
                session.human.data, _state.get("gaps"), _state.get("docs"),
                days=req.get("days", 7),
                sort=str(req.get("sort", "default") or "default").strip(),
                tag=str(req.get("tag", "all") or "all").strip(),
                audit_action=str(
                    req.get("audit_action", "all") or "all").strip(),
                improvement=improvement, scoreboard=scoreboard)
        except Exception as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, result)

    def _goal_plan(self, req: dict):
        goal = str(req.get("goal", "") or "").strip()
        if not goal:
            return self._json(400, {"error": "goal is required"})
        safety = _state.get("safety")
        if safety is not None:
            verdict = safety.check(goal)
            if verdict is not None:
                return self._json(400, {
                    "error": safety.refusal(verdict["category"]),
                    "safety": verdict,
                })
        session_id = str(req.get("session_id", "default")).strip() or "default"
        try:
            session = self._session(session_id)
            from atomzero.briefing import plan_goal

            plan = plan_goal(
                goal, _state.get("docs"), _state.get("gaps"),
                session.human.data, req.get("max_steps", 3))
            created = []
            if req.get("confirm") is True:
                for text in plan["suggested_tasks"][:3]:
                    task_id = session.human.add_task(text)
                    created.append({"id": task_id, "text": text})
            plan["created_tasks"] = created
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, plan)

    def _gap_dismiss(self, req: dict):
        try:
            item = self._gap_store().dismiss(str(req.get("id", "")))
        except GapNotFound as exc:
            return self._json(404, {"error": str(exc)})
        except KnowledgeGapError as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, item)

    def _queue(self):
        queue = _state.get("research_queue")
        if queue is None:
            raise ResearchQueueError("research queue is not configured")
        return queue

    def _queue_items(self, queue) -> list[dict]:
        items = queue.list()
        docs = _state.get("docs")
        if docs is None:
            return items
        notes = {source["source"]: source for source in docs.sources()
                 if source.get("kind") == "teacher"}
        for item in items:
            teacher = item.get("teacher")
            if not isinstance(teacher, dict):
                continue
            note = notes.get(teacher.get("source", ""))
            if note is not None:
                teacher["verified"] = bool(note.get("verified"))
                teacher["reviewer"] = note.get("reviewer", "")
                teacher["verified_at"] = note.get("verified_at", "")
        return items

    def _queue_add(self, req: dict):
        try:
            item = self._queue().add(
                req.get("topic", ""), req.get("lang", "en"),
                req.get("pages", 2))
        except ResearchQueueError as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, item)

    def _queue_update(self, req: dict, action: str):
        try:
            queue = self._queue()
            item_id = str(req.get("id", ""))
            item = (queue.approve(item_id) if action == "approve"
                    else queue.reject(item_id))
        except ItemNotFound as exc:
            return self._json(404, {"error": str(exc)})
        except ResearchQueueError as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, item)

    def _queue_run(self, req: dict):
        docs = _state.get("docs")
        if docs is None:
            return self._json(400, {"error": "no document memory configured"})
        try:
            queue = self._queue()
            from atomzero.web_research import wikipedia_research
            ran = queue.run_approved(
                docs, req.get("limit", 1), wikipedia_research,
                _state.get("teacher"))
        except ResearchQueueError as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, {"ran": ran, "stats": queue.stats()})

    def _teacher_verify(self, req: dict):
        docs = _state.get("docs")
        if docs is None:
            return self._json(400, {"error": "no document memory configured"})
        try:
            note = docs.verify_source(
                req.get("source", ""), req.get("reviewer", ""))
        except SourceNotFound as exc:
            return self._json(404, {"error": str(exc)})
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, {"ok": True, **note})

    def _teacher_ask(self, req: dict):
        provider = _state.get("teacher")
        if provider is None:
            return self._json(400, {
                "error": "teacher provider is not configured"})
        prompt = str(req.get("prompt") or "").strip()
        topic = str(req.get("topic") or "").strip()
        if not prompt and topic:
            prompt = f"Give a concise, factual summary of: {topic}"
        if not prompt:
            return self._json(400, {"error": "prompt or topic is required"})
        if len(prompt) > 20000:
            return self._json(400, {"error": "prompt is too long"})
        try:
            answer = provider.ask(
                prompt,
                system=("You are a concise research assistant. "
                        "If you are unsure, say so."),
                max_tokens=req.get("max_tokens", 512))
        except TeacherProviderError as exc:
            return self._json(502, {"error": str(exc)})
        try:
            result = add_teacher_note(
                _state.get("docs"), req.get("teacher") or provider.label,
                provider.model, topic or "Provider answer", answer,
                req.get("source_url", ""))
        except TeacherNoteTooLarge as exc:
            return self._json(413, {"error": str(exc)})
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        metrics = _state.get("metrics")
        if isinstance(metrics, dict):
            metrics["teacher_notes"] = metrics.get("teacher_notes", 0) + 1
        return self._json(200, result)

    def _teacher_note(self, req: dict):
        try:
            result = add_teacher_note(
                _state.get("docs"), req.get("teacher", "manual"),
                req.get("model", ""),
                req.get("topic") or req.get("question", ""),
                req.get("answer", ""), req.get("source_url", ""))
        except TeacherNoteTooLarge as exc:
            return self._json(413, {"error": str(exc)})
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        metrics = _state.get("metrics")
        if isinstance(metrics, dict):
            metrics["teacher_notes"] = metrics.get("teacher_notes", 0) + 1
        return self._json(200, result)

    def _ingest(self, req: dict):
        try:
            result = ingest_document(
                _state.get("docs"),
                req.get("source") or req.get("filename"),
                req.get("text") or req.get("content") or "",
                req.get("title", ""))
        except DocumentTooLarge as exc:
            return self._json(413, {"error": str(exc)})
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        metrics = _state.get("metrics")
        if isinstance(metrics, dict):
            metrics["documents"] = metrics.get("documents", 0) + 1
        return self._json(200, result)

    def _upload(self, query: str, n: int):
        params = urllib.parse.parse_qs(query)
        source = (params.get("source", [""])[0]
                  or params.get("filename", [""])[0])
        title = params.get("title", [""])[0]
        if n > MAX_DOC_BYTES:
            # Drain a bounded oversized body before answering. Writing a 413
            # while unread bytes remain in the receive buffer can make Windows
            # abort the client connection (WinError 10053) before it reads the
            # response. Very large bodies still close the connection.
            if n <= 1_048_576:
                try:
                    self.rfile.read(n)
                except OSError:
                    pass
            else:
                self.close_connection = True
            return self._json(413, {
                "error": f"document is too large (max {MAX_DOC_BYTES} bytes)"})
        raw = self.rfile.read(n)
        try:
            result = ingest_upload(
                _state.get("docs"), source, raw, title)
        except DocumentTooLarge as exc:
            return self._json(413, {"error": str(exc)})
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        metrics = _state.get("metrics")
        if isinstance(metrics, dict):
            metrics["documents"] = metrics.get("documents", 0) + 1
        return self._json(200, result)

    def _memory_ledger(self):
        from atomzero.memory_ledger import MemoryLedger
        ledger = _state.get("memory_ledger")
        if ledger is None:
            ledger = MemoryLedger(_state.get(
                "memory_ledger_path", "data/memory_ledger.jsonl"))
            _state["memory_ledger"] = ledger
        return ledger

    def _documents_forget(self, req: dict):
        confirm = req.get("confirm", None)
        if confirm is None:
            return self._json(409, {
                "error": "confirmation required",
                "code": "confirmation_required"})
        if not isinstance(confirm, bool):
            return self._json(400, {"error": "confirm must be a boolean"})
        if confirm is False:
            return self._json(409, {
                "error": "confirmation required",
                "code": "confirmation_required"})
        docs = _state.get("docs")
        if docs is None:
            return self._json(400, {"error": "no document memory configured"})
        try:
            result = docs.forget(str(req.get("source", "")))
        except SourceNotFound as exc:
            return self._json(404, {"error": str(exc)})
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        source = str(req.get("source", ""))
        try:
            self._memory_ledger().append(
                "document_forget", "document", source,
                before={"chunks": result.get("chunks_deleted", 0)},
                after={"chunks": 0})
            result["ledger_recorded"] = True
        except Exception:
            result["ledger_recorded"] = False
        return self._json(200, result)

    def _memory_forget(self, req: dict):
        confirm = req.get("confirm", None)
        if confirm is None:
            return self._json(409, {
                "error": "confirmation required",
                "code": "confirmation_required"})
        if not isinstance(confirm, bool):
            return self._json(400, {"error": "confirm must be a boolean"})
        if confirm is False:
            return self._json(409, {
                "error": "confirmation required",
                "code": "confirmation_required"})
        session_id = str(req.get("session_id", "")).strip()
        if not session_id:
            return self._json(400, {"error": "session_id is required"})
        text = str(req.get("text", "")).strip()
        clear_all = req.get("all", False)
        if not isinstance(clear_all, bool):
            return self._json(400, {"error": "all must be a boolean"})
        if not text and not clear_all:
            return self._json(400, {"error": "text or all is required"})
        try:
            session = self._session(session_id)
            result = session.forget("" if clear_all else text)
        except Exception as exc:
            return self._json(400, {"error": str(exc)})
        try:
            self._memory_ledger().append(
                "session_forget", "session", session_id,
                before={"facts": result.get("facts_deleted", 0),
                        "episodes": result.get("episodes_deleted", 0),
                        "tasks": result.get("tasks_deleted", 0)},
                after={"cleared_all": bool(clear_all)})
            result["ledger_recorded"] = True
        except Exception:
            result["ledger_recorded"] = False
        return self._json(200, {"session_id": session_id, **result})

    def _memory_export(self, req: dict):
        session_id = str(req.get("session_id", "")).strip()
        if not session_id:
            return self._json(400, {"error": "session_id is required"})
        try:
            session = self._session(session_id)
            data = session.export_memory()
        except Exception as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, {"session_id": session_id, "data": data})

    def _memory_import(self, req: dict):
        session_id = str(req.get("session_id", "")).strip()
        data = req.get("data")
        if not session_id:
            return self._json(400, {"error": "session_id is required"})
        if not isinstance(data, dict):
            return self._json(400, {"error": "data must be an object"})
        if len(json.dumps(data)) > 1_000_000:
            return self._json(413, {"error": "memory import is too large"})
        try:
            session = self._session(session_id)
            result = session.import_memory(data)
        except (OSError, ValueError) as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, {"session_id": session_id, **result})

    def _memory_history(self, req: dict):
        session_id = str(req.get("session_id", "")).strip()
        if not session_id:
            return self._json(400, {"error": "session_id is required"})
        try:
            limit = int(req.get("limit", 50))
        except (TypeError, ValueError):
            return self._json(400, {"error": "limit must be an integer"})
        limit = max(1, min(limit, 200))
        try:
            session = self._session(session_id)
            episodes = session.human.data.get("episodes", [])
        except Exception as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, {
            "session_id": session_id,
            "data": episodes[-limit:],
        })

    def _session(self, session_id: str):
        from atomzero.agent_runtime import AgentSession
        sessions = _state.setdefault("sessions", {})
        if session_id not in sessions:
            folder = Path(_state.get("session_dir", "data/sessions"))
            folder.mkdir(parents=True, exist_ok=True)
            sessions[session_id] = AgentSession(
                session_id, folder / f"{session_id}.json",
                model=_state.get("model"), tokenizer=_state.get("tok"),
                device=_state.get("device"), docs=_state.get("docs"),
                wiki_lang=_state.get("wiki_lang", "en"),
                gap_store=_state.get("gaps"),
                safety=_state.get("safety"),
                generator=_state.get("generator"),
                allow_general_chat=_state.get("allow_general_chat", False),
                skills=_state.get("skills"),
                weekly_reports_path=_state.get(
                    "weekly_reports_path", DEFAULT_REPORTS_PATH))
        return sessions[session_id]

    def _agent_plan(self, req: dict):
        session_id = str(req.get("session_id", "")).strip()
        if not session_id:
            return self._json(400, {"error": "session_id is required"})
        confirm = req.get("confirm", False)
        if not isinstance(confirm, bool):
            return self._json(400, {"error": "confirm must be a boolean"})
        try:
            session = self._session(session_id)
            result = execute_plan(
                session, req.get("steps"), confirm=confirm)
        except ToolPlanError as exc:
            return self._json(400, {"error": str(exc)})
        except Exception as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, result)

    def _tools_audit(self, req: dict):
        confirm = req.get("confirm", False)
        if not isinstance(confirm, bool):
            return self._json(400, {"error": "confirm must be a boolean"})
        result = audit_plan(req.get("steps"), confirm=confirm)
        return self._json(200 if result["valid"] else 400, result)

    def _claims_audit(self, req: dict):
        from atomzero.claims import audit_answer
        query = req.get("query")
        hits = req.get("hits")
        if not isinstance(query, str) or not query.strip():
            return self._json(400, {"error": "query is required"})
        if not isinstance(hits, list):
            return self._json(400, {"error": "hits must be a list"})
        if len(hits) > 50:
            return self._json(400, {"error": "at most 50 hits are allowed"})
        clean = []
        for hit in hits:
            if not isinstance(hit, dict):
                return self._json(400, {"error": "each hit must be an object"})
            clean.append({
                key: hit[key] for key in (
                    "source", "title", "section", "page", "chunk", "snippet")
                if key in hit
            })
        result = audit_answer(query, clean)
        session_id = str(req.get("session_id", "")).strip()
        if session_id:
            try:
                from atomzero.proactive import note_claim_audit
                note_claim_audit(self._session(session_id), result)
            except Exception as exc:
                return self._json(400, {"error": str(exc)})
        return self._json(200, result)

    def _consensus_audit(self, req: dict):
        from atomzero.consensus import audit_consensus
        hits = req.get("hits")
        if not isinstance(hits, list):
            return self._json(400, {"error": "hits must be a list"})
        if len(hits) > 50:
            return self._json(400, {"error": "at most 50 hits are allowed"})
        clean = []
        for hit in hits:
            if not isinstance(hit, dict):
                return self._json(400, {"error": "each hit must be an object"})
            clean.append({
                key: hit[key] for key in (
                    "source", "title", "section", "page", "chunk", "snippet")
                if key in hit
            })
        try:
            min_overlap = float(req.get("min_overlap", 0.55))
            min_sources = int(req.get("min_sources", 2))
        except (TypeError, ValueError):
            return self._json(400, {
                "error": "min_overlap and min_sources must be numeric"})
        try:
            result = audit_consensus(
                clean, min_overlap=min_overlap, min_sources=min_sources)
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        session_id = str(req.get("session_id", "")).strip()
        if session_id:
            try:
                from atomzero.proactive import note_claim_audit
                note_claim_audit(self._session(session_id), result)
            except Exception as exc:
                return self._json(400, {"error": str(exc)})
        return self._json(200, result)

    def _freshness_audit(self, req: dict):
        from atomzero.freshness import audit_freshness
        hits = req.get("hits")
        if not isinstance(hits, list):
            return self._json(400, {"error": "hits must be a list"})
        if len(hits) > 50:
            return self._json(400, {"error": "at most 50 hits are allowed"})
        clean = []
        for hit in hits:
            if not isinstance(hit, dict):
                return self._json(400, {"error": "each hit must be an object"})
            clean.append({
                key: hit[key] for key in (
                    "source", "title", "section", "page", "chunk", "snippet")
                if key in hit
            })
        try:
            max_age_days = int(req.get("max_age_days", 365))
            fresh_days = int(req.get("fresh_days", 90))
        except (TypeError, ValueError):
            return self._json(400, {
                "error": "max_age_days and fresh_days must be integers"})
        try:
            result = audit_freshness(
                clean, now=req.get("now"),
                max_age_days=max_age_days, fresh_days=fresh_days)
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, result)

    def _scoreboard_file(self):
        return Path(_state.get("scoreboard_path", "data/scoreboard.jsonl"))

    def _scoreboard_record(self, req: dict):
        from atomzero.scoreboard import (
            load_jsonl, record_run, summarize_runs, write_jsonl)
        try:
            path = self._scoreboard_file()
            runs = load_jsonl(path)
            record_run(
                runs, commit=req.get("commit", ""),
                tests_passed=req.get("tests_passed", 0),
                tests_failed=req.get("tests_failed", 0),
                benchmark_passed=req.get("benchmark_passed", 0),
                benchmark_total=req.get("benchmark_total", 0),
                aisc_passed=req.get("aisc_passed", 0),
                aisc_total=req.get("aisc_total", 0),
                note=req.get("note", ""))
            write_jsonl(path, runs)
            summary = summarize_runs(runs)
        except (OSError, TypeError, ValueError) as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, summary)

    def _scoreboard_summary(self, req: dict):
        from atomzero.scoreboard import load_jsonl, summarize_runs
        try:
            summary = summarize_runs(load_jsonl(self._scoreboard_file()))
        except (OSError, ValueError) as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, summary)

    def _scoreboard_verify(self, req: dict):
        from atomzero.scoreboard import load_jsonl, verify_runs
        try:
            result = verify_runs(load_jsonl(self._scoreboard_file()))
        except (OSError, ValueError) as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, result)

    def _improvement_ledger(self):
        path = Path(_state.get(
            "improvement_ledger_path", "data/improvement_ledger.jsonl"))
        ledger = []
        if path.exists():
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    ledger.append(json.loads(line))
        return path, ledger

    def _improvement_plan(self, req: dict):
        from atomzero.improvement import plan_from_inputs
        inputs = {}
        for key in ("claim_audit", "gaps", "feedback", "freshness",
                    "reliability", "aisc_coverage", "scoreboard", "drift",
                    "runner_audit", "self_code"):
            if key in req:
                inputs[key] = req[key]
        session_id = str(req.get("session_id", "")).strip()
        if session_id:
            try:
                session = self._session(session_id)
            except Exception as exc:
                return self._json(400, {"error": str(exc)})
            data = getattr(getattr(session, "human", None), "data", None)
            if isinstance(data, dict) and isinstance(
                    data.get("claim_audit"), dict):
                inputs["claim_audit"] = data["claim_audit"]
        try:
            limit = int(req.get("limit", 5))
            result = plan_from_inputs(limit=limit, **inputs)
            if session_id:
                from atomzero.proactive import note_improvement_plan
                note_improvement_plan(session, result)
        except (TypeError, ValueError) as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, result)

    def _improvement_decision(self, req: dict):
        from atomzero.improvement import (
            record_decision, record_gated_decision, verify_ledger)
        action = req.get("action")
        decision = req.get("decision")
        note = req.get("note", "")
        gate = req.get("gate")
        if not isinstance(action, dict):
            return self._json(400, {"error": "action must be an object"})
        if not isinstance(note, str):
            return self._json(400, {"error": "note must be a string"})
        if decision == "approve" and not isinstance(gate, dict):
            return self._json(400, {
                "error": "approval requires an evaluation gate report"})
        try:
            path, ledger = self._improvement_ledger()
            before = verify_ledger(ledger)
            if not before["ok"]:
                return self._json(400, {
                    "error": "existing ledger is invalid",
                    "verify": before})
            if isinstance(gate, dict):
                record = record_gated_decision(
                    ledger, action, decision, gate, note=note)
            else:
                record = record_decision(
                    ledger, action, decision, note=note)
            after = verify_ledger(ledger)
            if not after["ok"]:
                return self._json(400, {"error": "ledger verification failed"})
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                "".join(json.dumps(item, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")) + "\n"
                        for item in ledger),
                encoding="utf-8")
            os.replace(tmp, path)
        except (OSError, TypeError, ValueError) as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, {"record": record, "verify": after})

    def _temporal_filter(self, req: dict):
        from atomzero.temporal import filter_as_of
        hits = req.get("hits")
        if not isinstance(hits, list):
            return self._json(400, {"error": "hits must be a list"})
        if len(hits) > 50:
            return self._json(400, {"error": "at most 50 hits are allowed"})
        clean = []
        for hit in hits:
            if not isinstance(hit, dict):
                return self._json(400, {"error": "each hit must be an object"})
            clean.append({
                key: hit[key] for key in (
                    "source", "title", "section", "page", "chunk", "snippet",
                    "valid_from", "valid_until")
                if key in hit
            })
        keep_undated = req.get("keep_undated", True)
        if not isinstance(keep_undated, bool):
            return self._json(400, {"error": "keep_undated must be a boolean"})
        try:
            result = filter_as_of(
                clean, as_of=req.get("as_of"), keep_undated=keep_undated)
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, result)

    def _reliability_entries(self, req: dict):
        entries = req.get("entries")
        if not isinstance(entries, list):
            return None, self._json(400, {"error": "entries must be a list"})
        if len(entries) > 1000:
            return None, self._json(400, {
                "error": "at most 1000 entries are allowed"})
        return entries, None

    def _reliability_summary(self, req: dict):
        from atomzero.reliability import summarize
        entries, error = self._reliability_entries(req)
        if error is not None:
            return error
        try:
            min_observations = int(req.get("min_observations", 1))
            result = summarize(entries, min_observations=min_observations)
        except (TypeError, ValueError) as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, result)

    def _reliability_verify(self, req: dict):
        from atomzero.reliability import verify
        entries, error = self._reliability_entries(req)
        if error is not None:
            return error
        return self._json(200, verify(entries))

    def _documents_diff(self, req: dict):
        from atomzero.version_diff import audit_document_diff
        old_text = req.get("old_text")
        new_text = req.get("new_text")
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            return self._json(400, {
                "error": "old_text and new_text must be strings"})
        if len(old_text) > 200_000 or len(new_text) > 200_000:
            return self._json(400, {"error": "each text is limited to 200000 characters"})
        return self._json(200, audit_document_diff(
            old_text, new_text,
            old_label=str(req.get("old_label") or "old"),
            new_label=str(req.get("new_label") or "new")))

    def _proactive_next(self, req: dict):
        session_id = str(req.get("session_id", "")).strip()
        if not session_id:
            return self._json(400, {"error": "session_id is required"})
        ack = req.get("ack", False)
        if not isinstance(ack, bool):
            return self._json(400, {"error": "ack must be a boolean"})
        try:
            from atomzero.proactive import (
                DEFAULT_CONFIG, next_tick, proactive_state)
            session = self._session(session_id)
            result = next_tick(session, ack=ack)
            result["state"] = {
                **DEFAULT_CONFIG, **proactive_state(session)}
        except Exception as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, {"session_id": session_id, **result})

    def _proactive_config(self, req: dict):
        session_id = str(req.get("session_id", "")).strip()
        if not session_id:
            return self._json(400, {"error": "session_id is required"})
        allowed = ("enabled", "min_interval_minutes", "max_per_day",
                   "quiet_start", "quiet_end")
        updates = {key: req[key] for key in allowed if key in req}
        if not updates:
            return self._json(400, {"error": "no config fields supplied"})
        try:
            from atomzero.proactive import configure
            session = self._session(session_id)
            state = configure(session, **updates)
        except (ValueError, TypeError) as exc:
            return self._json(400, {"error": str(exc)})
        except Exception as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, {"session_id": session_id, "state": state})

    def _proactive_ack(self, req: dict):
        session_id = str(req.get("session_id", "")).strip()
        if not session_id:
            return self._json(400, {"error": "session_id is required"})
        try:
            from atomzero.proactive import ack
            session = self._session(session_id)
            state = ack(
                session, proactive_id=str(req.get("proactive_id", "")),
                action=str(req.get("action", "shown")))
        except Exception as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, {"session_id": session_id, "state": state})

    def _teach(self, req: dict):
        fact = str(req.get("fact", "")).strip()
        user = str(req.get("user", "")).strip()
        assistant = str(req.get("assistant", "")).strip()
        if not fact and not (user and assistant):
            return self._json(400, {"error": "provide fact or user+assistant"})
        if len(fact) > 4000 or len(user) > 2000 or len(assistant) > 2000:
            return self._json(400, {"error": "input too long"})
        result = {"ok": True}
        if fact:
            docs = _state.get("docs")
            if docs is None:
                return self._json(400, {"error": "no document memory configured"})
            result["fact_chunks"] = docs.add_text(
                "user_teach", fact, title="Taught facts", replace=False)
        if user and assistant:
            queue = Path(_state.get("queue_path",
                                    "data/learning_queue.jsonl"))
            queue.parent.mkdir(parents=True, exist_ok=True)
            with queue.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "user": user, "assistant": assistant,
                    "ts": int(time.time()),
                }, ensure_ascii=False) + "\n")
            result["skill_queued"] = True
        return self._json(200, result)

    def log_message(self, fmt, *args):
        return


def _attach_reranker(docs, mode: str = "auto",
                     neural_model: str = "models/all-MiniLM-L6-v2") -> str:
    """Attach the best available reranker; never fail server startup."""
    if docs is None or mode == "hashed":
        return "hashed"
    try:
        neural = NeuralReranker(neural_model)
        neural.embed(["warmup"])
        docs.reranker = neural
        return "neural"
    except (NeuralEmbedderError, OSError, RuntimeError) as exc:
        level = "warning" if mode == "neural" else "note"
        print(f"{level}: neural reranker disabled, using hashed "
              f"embeddings: {exc}", flush=True)
        return "hashed"


def serve(model_path: str, tokenizer_path: str, host: str = "127.0.0.1",
          port: int = 8013, api_key: str = "", adapter: str = "",
          docs_path: str = "",
          queue_path: str = "data/learning_queue.jsonl",
          allow_url_fetch: bool = False, wiki_lang: str = "en",
          session_dir: str = "data/sessions",
          research_queue_path: str = "data/research_queue.json",
          feedback_path: str = "data/feedback_queue.json",
          feedback_export_path: str = "data/feedback_training.jsonl",
          teacher_url: str = "", teacher_model: str = "",
          teacher_timeout: int = 60,
          teacher_label: str = "openai-compatible",
          active_adapter_file: str = "data/active_adapter.json",
          gap_store_path: str = "data/knowledge_gaps.json",
          hungry_ledger_path: str = "data/hungry_ledger.jsonl",
          safety_filter: bool = False,
          reranker: str = "auto",
          neural_model: str = "models/all-MiniLM-L6-v2",
          prewarm_vectors: bool = True,
          vector_index: str = "auto",
          safety_ml: bool = True,
          chat_provider_url: str = "",
          chat_provider_model: str = "",
          chat_provider_api_key: str = "",
          chat_provider_timeout: int = 60,
          chat_provider_label: str = "openai-compatible",
          allow_general_chat: bool = False,
          skills_dir: str = "data/skills",
          weekly_reports_path: str = DEFAULT_REPORTS_PATH) -> int:
    adapter = resolve_adapter(adapter, active_adapter_file)
    root = Path(model_path)
    ck = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = TransformerLM(cfg["vocab"], cfg["dim"], cfg["layers"],
                          cfg["heads"], cfg["ff"])
    model.load_state_dict(ck["state"])
    if adapter:
        from atomzero.lora import apply_adapter
        apply_adapter(model, adapter)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    docs = None
    if docs_path:
        from atomzero.doc_memory import DocMemory
        docs = DocMemory(docs_path)
        docs.vector_index_mode = vector_index
    active_reranker = _attach_reranker(docs, reranker, neural_model)
    safety = None
    if safety_filter:
        from atomzero.ml_safety import build_safety_filter
        encoder = getattr(docs, "reranker", None) if docs else None
        safety = build_safety_filter(prefer_ml=safety_ml, encoder=encoder)
    if docs is not None and prewarm_vectors:
        try:
            stats = docs.prewarm_vectors()
            print(f"vector index: {stats['chunks']} chunks, "
                  f"dim {stats['dim']}, {stats['backend']}, "
                  f"{stats['build_s']:.3f}s", flush=True)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"warning: vector prewarm failed: {exc}", flush=True)
    research_queue = None
    if research_queue_path:
        try:
            research_queue = ResearchQueue(research_queue_path)
        except ResearchQueueError as exc:
            print(f"warning: research queue disabled: {exc}", flush=True)
    feedback = None
    if feedback_path:
        try:
            feedback = FeedbackStore(feedback_path)
        except FeedbackError as exc:
            print(f"warning: feedback store disabled: {exc}", flush=True)
    gaps = None
    if gap_store_path:
        try:
            gaps = KnowledgeGapStore(gap_store_path)
        except KnowledgeGapError as exc:
            print(f"warning: knowledge-gap store disabled: {exc}", flush=True)
    skills = None
    try:
        from atomzero.skills import SkillRegistry

        skills = SkillRegistry.from_dir(skills_dir)
        if skills.skills:
            print(f"skills: {len(skills.skills)} loaded from "
                  f"{skills.directory}", flush=True)
    except Exception as exc:
        print(f"warning: skills disabled: {exc}", flush=True)
    chains = None
    try:
        from atomzero.skills import ChainRecipeRegistry

        chains = ChainRecipeRegistry.from_skills_dir(skills_dir)
        if chains.recipes:
            print(f"chain recipes: {len(chains.recipes)} loaded from "
                  f"{chains.directory}", flush=True)
    except Exception as exc:
        print(f"warning: chain recipes disabled: {exc}", flush=True)
    teacher = None
    if teacher_url or teacher_model:
        try:
            teacher = TeacherProvider(
                teacher_url, teacher_model,
                os.environ.get("TEACHER_API_KEY", ""),
                teacher_timeout, teacher_label)
        except TeacherProviderError as exc:
            print(f"warning: teacher provider disabled: {exc}", flush=True)
    chat_provider = None
    if chat_provider_url or chat_provider_model:
        try:
            chat_provider = TeacherProvider(
                chat_provider_url, chat_provider_model,
                chat_provider_api_key or os.environ.get(
                    "CHAT_PROVIDER_API_KEY", ""),
                chat_provider_timeout, chat_provider_label)
            print(f"chat provider: {chat_provider.label} / "
                  f"{chat_provider.model} @ {chat_provider.host}",
                  flush=True)
        except TeacherProviderError as exc:
            print(f"warning: chat provider disabled: {exc}", flush=True)

    def chat_generate(prompt: str):
        if chat_provider is None:
            return None
        try:
            return chat_provider.ask(
                prompt,
                system=("You are a concise, direct local assistant. Answer "
                        "the user's request without refusing benign topics."),
                max_tokens=512)
        except TeacherProviderError as exc:
            print(f"warning: chat provider request failed: {exc}", flush=True)
            return None

    _state.update({
        "model": model,
        "tok": tok,
        "docs": docs,
        "research_queue": research_queue,
        "research_queue_path": research_queue_path,
        "feedback": feedback,
        "feedback_path": feedback_path,
        "feedback_export_path": feedback_export_path,
        "gaps": gaps,
        "gap_store_path": gap_store_path,
        "hungry_ledger_path": hungry_ledger_path,
        "skills": skills,
        "skills_dir": skills_dir,
        "chains": chains,
        "chains_dir": chains.directory if chains else "",
        "weekly_reports_path": weekly_reports_path,
        "adapter_history_path": "data/adapter_history.jsonl",
        "active_adapter_file": active_adapter_file,
        "safety": safety,
        "reranker": active_reranker,
        "teacher": teacher,
        "chat_provider": chat_provider,
        "generator": chat_generate if chat_provider else None,
        "allow_general_chat": bool(allow_general_chat),
        "adapter": adapter,
        "queue_path": queue_path,
        "allow_url_fetch": allow_url_fetch,
        "wiki_lang": wiki_lang,
        "session_dir": session_dir,
        "sessions": {},
        "device": device,
        "name": root.name,
        "params": sum(p.numel() for p in model.parameters()),
        "config": cfg,
        "step": ck.get("step"),
        "api_key": api_key or os.environ.get("CHAT_API_KEY", ""),
        "user_id": tok.token_to_id("<|user|>"),
        "assistant_id": tok.token_to_id("<|assistant|>"),
        "metrics": {"requests": 0, "generated_tokens": 0, "documents": 0,
                    "teacher_notes": 0},
    })
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Atom Zero serving {model_path} at http://{host}:{port}",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", action="version", version=f"atomzero {__version__}")
    ap.add_argument("--model",
                    default="checkpoints/transformer_behavior_50m_scaled")
    ap.add_argument("--tokenizer", default="data/chat_en_bpe8k.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8013)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--docs", default="")
    ap.add_argument("--queue", default="data/learning_queue.jsonl")
    ap.add_argument("--allow-url-fetch", action="store_true")
    ap.add_argument("--wiki-lang", default="en")
    ap.add_argument("--session-dir", default="data/sessions")
    ap.add_argument("--research-queue", default="data/research_queue.json")
    ap.add_argument("--feedback-store", default="data/feedback_queue.json")
    ap.add_argument("--feedback-export", default="data/feedback_training.jsonl")
    ap.add_argument("--gap-store", default="data/knowledge_gaps.json")
    ap.add_argument("--hungry-ledger", default="data/hungry_ledger.jsonl")
    ap.add_argument("--skills-dir", default="data/skills")
    ap.add_argument("--safety-filter", action="store_true")
    ap.add_argument("--no-safety-ml", dest="safety_ml",
                    action="store_false", default=True,
                    help="use only the exact rule filter, not the local ML head")
    ap.add_argument("--reranker", choices=("auto", "hashed", "neural"),
                    default="auto",
                    help="auto uses the local neural model when present, "
                         "otherwise hashed n-gram embeddings")
    ap.add_argument("--neural-model", default="models/all-MiniLM-L6-v2")
    ap.add_argument("--no-prewarm", dest="prewarm_vectors",
                    action="store_false", default=True,
                    help="skip building the in-memory vector index at startup")
    ap.add_argument("--vector-index", choices=("auto", "ann", "exact"),
                    default="auto",
                    help="auto uses IVF ANN above 256 chunks, exact below; "
                         "exact fallback is always available")
    ap.add_argument("--chat-provider-url", default="",
                    help="optional OpenAI-compatible local chat backend, "
                         "e.g. http://127.0.0.1:11434/v1")
    ap.add_argument("--chat-provider-model", default="",
                    help="model name for --chat-provider-url, e.g. llama3.2:3b")
    ap.add_argument("--chat-provider-api-key", default="")
    ap.add_argument("--chat-provider-timeout", type=int, default=60)
    ap.add_argument("--chat-provider-label", default="local-provider")
    ap.add_argument("--allow-general-chat", action="store_true",
                    help="let the provider answer when no local source matches")
    ap.add_argument("--teacher-url", default="")
    ap.add_argument("--teacher-model", default="")
    ap.add_argument("--teacher-timeout", type=int, default=60)
    ap.add_argument("--teacher-label", default="openai-compatible")
    ap.add_argument("--active-adapter-file", default="data/active_adapter.json")
    ap.add_argument("--reports-index", default=DEFAULT_REPORTS_PATH)
    args = ap.parse_args()
    return serve(args.model, args.tokenizer, args.host, args.port,
                 args.api_key, args.adapter, args.docs, args.queue,
                 args.allow_url_fetch, args.wiki_lang, args.session_dir,
                 args.research_queue, args.feedback_store,
                 args.feedback_export, args.teacher_url, args.teacher_model,
                 args.teacher_timeout, args.teacher_label,
                 args.active_adapter_file, args.gap_store,
                 args.hungry_ledger, args.safety_filter, args.reranker,
                 args.neural_model,
                 args.prewarm_vectors, args.vector_index, args.safety_ml,
                 args.chat_provider_url, args.chat_provider_model,
                 args.chat_provider_api_key, args.chat_provider_timeout,
                 args.chat_provider_label, args.allow_general_chat,
                 skills_dir=args.skills_dir,
                 weekly_reports_path=args.reports_index)


if __name__ == "__main__":
    raise SystemExit(main())
