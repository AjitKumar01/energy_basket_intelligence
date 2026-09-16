#!/usr/bin/env python3
"""MCP server exposing OpenAI Codex CLI session transcripts to Claude Code.

Reads rollout files under ~/.codex/sessions (override with CODEX_HOME) and
serves them over stdio JSON-RPC (MCP). No third-party dependencies.

Tools
  list_codex_sessions   find sessions (filter by cwd / name)
  session_overview      turn-by-turn table of a session
  session_handoff       condensed context for continuing the work
  get_turn              full detail of one turn
  search_session        text / regex search across a session
  file_changes          file edits made during a session (with diffs)

Session references accept: "latest", a full or prefix session id, or a path
to a rollout .jsonl file. "latest" is scoped to the server's working
directory unless the cwd argument says otherwise.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "codex-sessions", "version": "1.0.0"}

CODEX_HOME = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
SESSIONS_DIR = CODEX_HOME / "sessions"
INDEX_FILE = CODEX_HOME / "session_index.jsonl"
DEFAULT_CWD = os.environ.get("CODEX_SESSIONS_CWD") or os.getcwd()

# Per-item text retained in memory; very large command outputs are clipped.
STORE_LIMIT = 60_000


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Item:
    kind: str            # user | agent | command | file_change | web_search | image | compaction | tool
    ts: str
    text: str = ""
    phase: str = ""      # agent: commentary | final_answer
    command: str = ""
    exit_code: Any = None
    output: str = ""
    changes: list = field(default_factory=list)   # [(path, type, diff_or_content)]


@dataclass
class Turn:
    turn_id: str
    started: str = ""
    ended: str = ""
    status: str = "incomplete"
    error: str = ""
    last_agent_message: str = ""
    items: list = field(default_factory=list)

    @property
    def user_text(self) -> str:
        return "\n\n".join(i.text for i in self.items if i.kind == "user")

    @property
    def final_text(self) -> str:
        if self.last_agent_message:
            return self.last_agent_message
        agents = [i for i in self.items if i.kind == "agent"]
        finals = [i for i in agents if i.phase == "final_answer"]
        return (finals or agents)[-1].text if agents else ""

    def count(self, kind: str) -> int:
        return sum(1 for i in self.items if i.kind == kind)


@dataclass
class Session:
    path: Path
    meta: dict
    turns: list
    forked_from: str = ""


def clip(s: str, n: int) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n] + f"\n…[{len(s) - n} more chars truncated]"


def one_line(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


# --------------------------------------------------------------------------
# Session discovery
# --------------------------------------------------------------------------

def all_rollouts() -> list[Path]:
    if not SESSIONS_DIR.exists():
        return []
    return sorted(SESSIONS_DIR.rglob("rollout-*.jsonl"), key=lambda p: p.name)


def session_id_from_path(p: Path) -> str:
    m = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.jsonl$", p.name)
    return m.group(1) if m else p.stem


def thread_names() -> dict[str, str]:
    names: dict[str, str] = {}
    if INDEX_FILE.exists():
        for line in INDEX_FILE.read_text(errors="replace").splitlines():
            try:
                d = json.loads(line)
                names[d["id"]] = d.get("thread_name", "")
            except (ValueError, KeyError):
                continue
    return names


_meta_cache: dict[tuple, dict] = {}


def read_meta(p: Path) -> dict:
    st = p.stat()
    key = (str(p), st.st_mtime, st.st_size)
    if key not in _meta_cache:
        meta: dict = {}
        with p.open(errors="replace") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("type") == "session_meta":
                    meta = d.get("payload", {})
                    break
        _meta_cache[key] = meta
    return _meta_cache[key]


def resolve(ref: str | None, cwd: str | None = None) -> Path:
    ref = (ref or "latest").strip()
    if ref.endswith(".jsonl") and Path(ref).expanduser().exists():
        return Path(ref).expanduser()
    files = all_rollouts()
    if not files:
        raise ValueError(f"No Codex sessions found under {SESSIONS_DIR}")
    if ref == "latest":
        scope = cwd or DEFAULT_CWD
        scoped = [f for f in files if read_meta(f).get("cwd") == scope]
        pool = scoped or files
        # Prefer the most recently modified session that actually has content.
        pool = sorted(pool, key=lambda f: f.stat().st_mtime, reverse=True)
        return pool[0]
    matches = [f for f in files if session_id_from_path(f).startswith(ref)]
    if not matches:
        names = thread_names()
        matches = [f for f in files if ref.lower() in names.get(session_id_from_path(f), "").lower()]
    if not matches:
        raise ValueError(f"No session matches '{ref}'")
    if len(matches) > 1:
        ids = ", ".join(session_id_from_path(m) for m in matches[:10])
        raise ValueError(f"'{ref}' is ambiguous: {ids}")
    return matches[0]


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

_session_cache: dict[tuple, Session] = {}


def load_session(p: Path, _depth: int = 0) -> Session:
    st = p.stat()
    key = (str(p), st.st_mtime, st.st_size)
    if key in _session_cache:
        return _session_cache[key]

    meta: dict = {}
    has_items = False
    lines: list[dict] = []
    with p.open(errors="replace") as fh:
        for raw in fh:
            try:
                d = json.loads(raw)
            except ValueError:
                continue
            t = d.get("type")
            pl = d.get("payload") or {}
            if t == "session_meta" and not meta:
                meta = pl
            if t == "event_msg" and pl.get("type") == "item_completed":
                has_items = True
            if t in ("event_msg", "response_item", "turn_context"):
                # Drop bulky records never used for reconstruction.
                if pl.get("type") in ("token_count", "reasoning", "thread_settings_applied", "agent_reasoning"):
                    continue
                lines.append(d)

    turns = _parse_items(lines) if has_items else _parse_legacy(lines)
    session = Session(path=p, meta=meta, turns=turns)

    parent_id = meta.get("forked_from_id")
    if parent_id and _depth < 5:
        session.forked_from = parent_id
        try:
            parent = load_session(resolve(parent_id), _depth + 1)
            session.turns = parent.turns + session.turns
        except ValueError:
            pass

    _session_cache[key] = session
    return session


def _parse_items(lines: list[dict]) -> list[Turn]:
    turns: dict[str, Turn] = {}

    def turn(tid: str, ts: str) -> Turn:
        if tid not in turns:
            turns[tid] = Turn(turn_id=tid, started=ts)
        return turns[tid]

    for d in lines:
        if d.get("type") != "event_msg":
            continue
        pl = d["payload"]
        ts = d.get("timestamp", "")
        et = pl.get("type")
        tid = pl.get("turn_id")
        if not tid:
            continue
        if et == "task_started":
            turn(tid, ts).started = ts
        elif et == "task_complete":
            t = turn(tid, ts)
            t.ended = ts
            t.last_agent_message = pl.get("last_agent_message") or ""
            err = pl.get("error")
            if err:
                t.status = "error"
                t.error = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            else:
                t.status = "completed"
        elif et == "turn_aborted":
            t = turn(tid, ts)
            t.ended = ts
            t.status = "aborted"
            t.error = pl.get("reason", "")
        elif et == "item_completed":
            it = pl.get("item") or {}
            item = _convert_item(it, ts)
            if item:
                turn(tid, ts).items.append(item)
    return list(turns.values())


def _texts(content: list) -> str:
    out = []
    for c in content or []:
        if isinstance(c, dict) and isinstance(c.get("text"), str):
            out.append(c["text"])
    return "".join(out)


def _convert_item(it: dict, ts: str) -> Item | None:
    kind = it.get("type")
    if kind == "UserMessage":
        return Item("user", ts, text=_texts(it.get("content")))
    if kind == "AgentMessage":
        return Item("agent", ts, text=_texts(it.get("content")), phase=it.get("phase") or "")
    if kind == "CommandExecution":
        cmd = it.get("command")
        if isinstance(cmd, list):
            cmd = cmd[-1] if len(cmd) >= 3 and cmd[1] in ("-lc", "-c") else " ".join(cmd)
        out = it.get("aggregated_output") or it.get("formatted_output") or (it.get("stdout", "") + it.get("stderr", ""))
        return Item("command", ts, command=cmd or "", exit_code=it.get("exit_code"),
                    output=clip(out, STORE_LIMIT))
    if kind == "FileChange":
        changes = []
        for path, ch in (it.get("changes") or {}).items():
            ctype = ch.get("type", "")
            body = ch.get("unified_diff") or ch.get("content") or ""
            if ch.get("move_path"):
                ctype += f" → {ch['move_path']}"
            changes.append((path, ctype, clip(body, STORE_LIMIT)))
        return Item("file_change", ts, changes=changes)
    if kind == "Extension":
        action = it.get("action") or {}
        queries = action.get("queries") or [it.get("query") or action.get("query") or ""]
        urls = [r.get("url", "") for r in (it.get("results") or [])[:5]]
        return Item("web_search", ts, text="; ".join(q for q in queries if q),
                    output="\n".join(u for u in urls if u))
    if kind == "ImageView":
        return Item("image", ts, text=it.get("path", ""))
    if kind == "ContextCompaction":
        return Item("compaction", ts, text="(context compacted)")
    return None


def _parse_legacy(lines: list[dict]) -> list[Turn]:
    """Older rollouts: split on real user messages, collect tool calls."""
    turns: list[Turn] = []
    calls: dict[str, Item] = {}
    current: Turn | None = None
    for d in lines:
        if d.get("type") != "response_item":
            continue
        pl = d["payload"]
        ts = d.get("timestamp", "")
        pt = pl.get("type")
        if pt == "message":
            text = _texts(pl.get("content"))
            role = pl.get("role")
            if role == "user":
                if text.lstrip().startswith(("<", "# AGENTS.md instructions")):
                    continue  # environment / instructions boilerplate
                current = Turn(turn_id=f"legacy-{len(turns)}", started=ts, status="completed")
                turns.append(current)
                current.items.append(Item("user", ts, text=text))
            elif role == "assistant" and current:
                current.items.append(Item("agent", ts, text=text, phase="final_answer"))
                current.ended = ts
        elif pt in ("function_call", "custom_tool_call") and current:
            args = pl.get("arguments") or pl.get("input") or ""
            try:
                parsed = json.loads(args)
                cmd = parsed.get("command") or parsed.get("cmd") or args
                if isinstance(cmd, list):
                    cmd = cmd[-1] if len(cmd) >= 3 and cmd[1] in ("-lc", "-c") else " ".join(cmd)
            except (ValueError, AttributeError):
                cmd = args
            kind = "file_change" if pl.get("name") == "apply_patch" or "*** Begin Patch" in str(cmd) else "command"
            item = Item(kind, ts, command=f"[{pl.get('name')}] {cmd}")
            if kind == "file_change":
                paths = re.findall(r"\*\*\* (?:Add|Update|Delete) File: (.+)", str(cmd))
                item.changes = [(pth, "patch", clip(str(cmd), STORE_LIMIT)) for pth in paths] or [("?", "patch", clip(str(cmd), STORE_LIMIT))]
            calls[pl.get("call_id", "")] = item
            current.items.append(item)
        elif pt in ("function_call_output", "custom_tool_call_output"):
            item = calls.get(pl.get("call_id", ""))
            if item:
                out = pl.get("output")
                if isinstance(out, list):
                    out = _texts(out)
                elif isinstance(out, str):
                    try:
                        o = json.loads(out)
                        out = o.get("output", out) if isinstance(o, dict) else out
                    except ValueError:
                        pass
                item.output = clip(str(out), STORE_LIMIT)
    return turns


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def header(s: Session) -> str:
    m = s.meta
    sid = m.get("id") or session_id_from_path(s.path)
    name = thread_names().get(sid, "")
    lines = [
        f"# Codex session {sid}" + (f" — “{name}”" if name else ""),
        f"- file: {s.path}",
        f"- cwd: {m.get('cwd', '?')}",
        f"- started: {m.get('timestamp', '?')}  |  cli {m.get('cli_version', '?')}",
        f"- turns: {len(s.turns)}",
    ]
    if s.forked_from:
        lines.append(f"- forked from: {s.forked_from} (parent turns included)")
    if s.turns:
        last = s.turns[-1]
        lines.append(f"- last activity: {last.ended or last.started}  |  last turn status: {last.status}"
                     + (f" ({one_line(last.error, 200)})" if last.error else ""))
    return "\n".join(lines)


def render_turn(s: Session, idx: int, max_output: int, include_commands: bool = True,
                include_diffs: bool = True) -> str:
    t = s.turns[idx]
    out = [f"## Turn {idx}  [{t.status}]  {t.started} → {t.ended or '…'}",
           f"turn_id: {t.turn_id}"]
    if t.error:
        out.append(f"error: {t.error}")
    for i in t.items:
        if i.kind == "user":
            out.append(f"\n### USER\n{i.text}")
        elif i.kind == "agent":
            out.append(f"\n### CODEX ({i.phase or 'message'})\n{i.text}")
        elif i.kind == "command" and include_commands:
            out.append(f"\n$ {clip(i.command, 1500)}\n[exit {i.exit_code}]"
                       + (f"\n{clip(i.output, max_output)}" if max_output and i.output else ""))
        elif i.kind == "file_change":
            for path, ctype, body in i.changes:
                out.append(f"\n~ {ctype.upper()} {path}"
                           + (f"\n```diff\n{clip(body, max_output * 3)}\n```" if include_diffs and max_output else ""))
        elif i.kind == "web_search":
            out.append(f"\n🔎 web search: {i.text}" + (f"\n{i.output}" if i.output else ""))
        elif i.kind == "image":
            out.append(f"\n🖼 viewed image: {i.text}")
        elif i.kind == "compaction":
            out.append("\n(context was compacted here)")
    return "\n".join(out)


def turn_index(s: Session, turn: int) -> int:
    n = len(s.turns)
    if not n:
        raise ValueError("Session has no turns")
    idx = turn if turn >= 0 else n + turn
    if not 0 <= idx < n:
        raise ValueError(f"turn {turn} out of range (0..{n - 1}, negatives count from the end)")
    return idx


def aggregate_files(s: Session, turns: range | None = None) -> dict[str, dict]:
    files: dict[str, dict] = {}
    for ti in (turns or range(len(s.turns))):
        for it in s.turns[ti].items:
            if it.kind != "file_change":
                continue
            for path, ctype, _ in it.changes:
                f = files.setdefault(path, {"edits": 0, "types": set(), "first_turn": ti, "last_turn": ti})
                f["edits"] += 1
                f["types"].add(ctype.split(" ")[0])
                f["last_turn"] = ti
    return files


def rel(path: str, cwd: str) -> str:
    return path[len(cwd) + 1:] if cwd and path.startswith(cwd + "/") else path


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

def tool_list_sessions(args: dict) -> str:
    cwd_filter = args.get("cwd_contains") or ""
    name_filter = (args.get("name_contains") or "").lower()
    limit = int(args.get("limit", 20))
    names = thread_names()
    rows = []
    for f in sorted(all_rollouts(), key=lambda p: p.stat().st_mtime, reverse=True):
        m = read_meta(f)
        sid = m.get("id") or session_id_from_path(f)
        name = names.get(sid, "")
        if cwd_filter and cwd_filter not in (m.get("cwd") or ""):
            continue
        if name_filter and name_filter not in name.lower():
            continue
        st = f.stat()
        from datetime import datetime, timezone
        modified = datetime.fromtimestamp(st.st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        fork = f" (fork of {m['forked_from_id'][:13]}…)" if m.get("forked_from_id") else ""
        rows.append(f"- {sid}  “{name}”{fork}\n  cwd={m.get('cwd')}  started={m.get('timestamp', '')[:19]}  "
                    f"modified={modified}  size={st.st_size / 1e6:.1f}MB")
        if len(rows) >= limit:
            break
    return "\n".join(rows) if rows else "No matching sessions."


def tool_overview(args: dict) -> str:
    s = load_session(resolve(args.get("session"), args.get("cwd")))
    width = int(args.get("preview_chars", 220))
    out = [header(s), "", "| # | started | status | cmds | edits | user request | outcome |", "|---|---|---|---|---|---|---|"]
    for i, t in enumerate(s.turns):
        user = one_line(t.user_text, width).replace("|", "\\|") or "—"
        final = one_line(t.error if t.status == "error" else t.final_text, width).replace("|", "\\|") or "—"
        out.append(f"| {i} | {t.started[:16]} | {t.status} | {t.count('command')} | "
                   f"{sum(len(x.changes) for x in t.items if x.kind == 'file_change')} | {user} | {final} |")
    return "\n".join(out)


def tool_get_turn(args: dict) -> str:
    s = load_session(resolve(args.get("session"), args.get("cwd")))
    idx = turn_index(s, int(args.get("turn", -1)))
    text = render_turn(s, idx, int(args.get("max_output_chars", 1500)),
                       include_commands=bool(args.get("include_commands", True)),
                       include_diffs=bool(args.get("include_diffs", True)))
    return clip(text, int(args.get("max_chars", 80_000)))


def tool_handoff(args: dict) -> str:
    s = load_session(resolve(args.get("session"), args.get("cwd")))
    recent = int(args.get("recent_turns", 3))
    max_chars = int(args.get("max_chars", 70_000))
    cwd = s.meta.get("cwd", "")
    n = len(s.turns)
    parts = [header(s), "",
             "This is a reconstructed handoff from an OpenAI Codex CLI session. Codex's own",
             "compaction summaries are encrypted, so this is rebuilt from the raw transcript.",
             "Verify the current repository state (git status, files) before acting on it.", ""]

    parts.append("## Conversation arc (every user request and Codex's final reply, oldest first)")
    early = max(0, n - recent)

    def arc(user_chars: int, codex_chars: int) -> list[str]:
        rows = []
        for i, t in enumerate(s.turns[:early]):
            user = clip(t.user_text.strip(), user_chars)
            if not user and not t.final_text:
                continue
            outcome = t.error if t.status == "error" else t.final_text
            rows.append(f"\n**[{i}] {t.started[:16]} · {t.status}**\nUSER: {user or '(no user text)'}\n"
                        f"CODEX: {clip(outcome.strip(), codex_chars) or '(no final message)'}")
        return rows

    # The arc may use at most ~45% of the budget; shrink snippets until it fits.
    for user_chars, codex_chars in ((1200, 900), (800, 500), (500, 250), (300, 120), (160, 0)):
        rows = arc(user_chars, codex_chars)
        if sum(len(r) for r in rows) <= 0.45 * max_chars:
            break
    parts.extend(rows)

    files = aggregate_files(s)
    if files:
        parts.append("\n## Files created/modified by Codex across the session")
        for path, f in sorted(files.items(), key=lambda kv: kv[1]["last_turn"], reverse=True):
            parts.append(f"- {rel(path, cwd)} — {'/'.join(sorted(f['types']))}, {f['edits']} edit(s), "
                         f"turns {f['first_turn']}–{f['last_turn']}")

    parts.append(f"\n## Most recent {min(recent, n)} turn(s) in detail")
    budget_used = sum(len(p) + 1 for p in parts)
    tail = [render_turn(s, i, int(args.get("max_output_chars", 800)), include_diffs=False)
            for i in range(early, n)]
    remaining = max_chars - budget_used
    tail_text = "\n\n".join(tail)
    if len(tail_text) > remaining:
        # Keep the end of the session, which matters most for resuming.
        tail_text = f"…[{len(tail_text) - remaining} chars of older detail truncated]\n" + tail_text[-remaining:]
    parts.append(tail_text)

    if n and s.turns[-1].status != "completed":
        parts.append(f"\n## ⚠ Session ended mid-task\nLast turn status: {s.turns[-1].status}. "
                     f"{s.turns[-1].error}\nThe final request above was NOT finished by Codex.")
    return clip("\n".join(parts), max_chars + 2000)


def tool_search(args: dict) -> str:
    s = load_session(resolve(args.get("session"), args.get("cwd")))
    query = args.get("query") or ""
    if not query:
        raise ValueError("query is required")
    pattern = re.compile(query if args.get("regex") else re.escape(query), re.IGNORECASE)
    kinds = set(args.get("kinds") or ["user", "agent", "command", "file_change", "web_search"])
    limit = int(args.get("limit", 30))
    ctx = int(args.get("context_chars", 160))
    hits = []
    for ti, t in enumerate(s.turns):
        if t.error and pattern.search(t.error):
            hits.append(f"- turn {ti} · turn_error · {t.ended[:16]}: {one_line(t.error, 2 * ctx)}")
        for it in t.items:
            if it.kind not in kinds:
                continue
            fields = {"user": [it.text], "agent": [it.text], "web_search": [it.text],
                      "command": [it.command, it.output],
                      "file_change": [f"{p}\n{b}" for p, _, b in it.changes]}.get(it.kind, [it.text])
            for fld in fields:
                m = pattern.search(fld or "")
                if m:
                    a, b = max(0, m.start() - ctx), min(len(fld), m.end() + ctx)
                    hits.append(f"- turn {ti} · {it.kind} · {it.ts[:16]}: …{one_line(fld[a:b], 2 * ctx + 200)}…")
                    break
            if len(hits) >= limit:
                return "\n".join(hits) + f"\n(limit {limit} reached)"
    return "\n".join(hits) if hits else "No matches."


def tool_file_changes(args: dict) -> str:
    s = load_session(resolve(args.get("session"), args.get("cwd")))
    cwd = s.meta.get("cwd", "")
    needle = args.get("path_contains") or ""
    turn = args.get("turn")
    include_diffs = bool(args.get("include_diffs", False))
    max_chars = int(args.get("max_chars", 60_000))
    turns = range(len(s.turns)) if turn is None else [turn_index(s, int(turn))]
    out = []
    for ti in turns:
        for it in s.turns[ti].items:
            if it.kind != "file_change":
                continue
            for path, ctype, body in it.changes:
                if needle and needle not in path:
                    continue
                line = f"- turn {ti} · {it.ts[:16]} · {ctype} · {rel(path, cwd)}"
                if include_diffs:
                    line += f"\n```diff\n{clip(body, int(args.get('max_diff_chars', 6000)))}\n```"
                out.append(line)
    return clip("\n".join(out), max_chars) if out else "No file changes matched."


SESSION_PROP = {"type": "string",
                "description": "'latest' (default, scoped to cwd), a session id or id prefix, a thread-name fragment, or a rollout .jsonl path"}
CWD_PROP = {"type": "string", "description": "Project directory used to scope 'latest' (defaults to server cwd)"}

TOOLS = [
    {"name": "list_codex_sessions",
     "description": "List Codex CLI sessions (newest first) with id, thread name, cwd, dates and size.",
     "inputSchema": {"type": "object", "properties": {
         "cwd_contains": {"type": "string", "description": "Only sessions whose cwd contains this"},
         "name_contains": {"type": "string", "description": "Only sessions whose thread name contains this"},
         "limit": {"type": "integer", "default": 20}}},
     "handler": tool_list_sessions},
    {"name": "session_overview",
     "description": "Turn-by-turn table of a Codex session: each user request, status, command/edit counts and outcome.",
     "inputSchema": {"type": "object", "properties": {
         "session": SESSION_PROP, "cwd": CWD_PROP,
         "preview_chars": {"type": "integer", "default": 220}}},
     "handler": tool_overview},
    {"name": "session_handoff",
     "description": "Condensed context for continuing a Codex session's work: full request history with outcomes, all files touched, and the latest turns in detail. Start here when resuming.",
     "inputSchema": {"type": "object", "properties": {
         "session": SESSION_PROP, "cwd": CWD_PROP,
         "recent_turns": {"type": "integer", "default": 3, "description": "How many final turns to show in full detail"},
         "max_output_chars": {"type": "integer", "default": 800, "description": "Per-command output clip in the detailed turns"},
         "max_chars": {"type": "integer", "default": 70000}}},
     "handler": tool_handoff},
    {"name": "get_turn",
     "description": "Full detail of one turn: user message, Codex messages, commands with output, file diffs, web searches.",
     "inputSchema": {"type": "object", "properties": {
         "session": SESSION_PROP, "cwd": CWD_PROP,
         "turn": {"type": "integer", "default": -1, "description": "Turn index; negatives count from the end"},
         "include_commands": {"type": "boolean", "default": True},
         "include_diffs": {"type": "boolean", "default": True},
         "max_output_chars": {"type": "integer", "default": 1500},
         "max_chars": {"type": "integer", "default": 80000}}},
     "handler": tool_get_turn},
    {"name": "search_session",
     "description": "Search a Codex session's messages, commands/outputs, diffs and web searches.",
     "inputSchema": {"type": "object", "required": ["query"], "properties": {
         "session": SESSION_PROP, "cwd": CWD_PROP,
         "query": {"type": "string"},
         "regex": {"type": "boolean", "default": False},
         "kinds": {"type": "array", "items": {"type": "string", "enum": ["user", "agent", "command", "file_change", "web_search"]}},
         "limit": {"type": "integer", "default": 30},
         "context_chars": {"type": "integer", "default": 160}}},
     "handler": tool_search},
    {"name": "file_changes",
     "description": "File edits Codex made in a session, optionally filtered by path or turn, with diffs.",
     "inputSchema": {"type": "object", "properties": {
         "session": SESSION_PROP, "cwd": CWD_PROP,
         "path_contains": {"type": "string"},
         "turn": {"type": "integer"},
         "include_diffs": {"type": "boolean", "default": False},
         "max_diff_chars": {"type": "integer", "default": 6000},
         "max_chars": {"type": "integer", "default": 60000}}},
     "handler": tool_file_changes},
]

PROMPTS = [
    {"name": "resume_codex",
     "description": "Load a Codex session handoff into the conversation and continue its work",
     "arguments": [{"name": "session", "description": "Session id/prefix/name (default: latest for this project)", "required": False},
                   {"name": "recent_turns", "description": "Turns to include in full detail (default 3)", "required": False}]},
]


# --------------------------------------------------------------------------
# JSON-RPC / MCP plumbing
# --------------------------------------------------------------------------

def handle(req: dict) -> dict | None:
    method = req.get("method")
    rid = req.get("id")
    params = req.get("params") or {}

    def ok(result: Any) -> dict:
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    if rid is None:
        return None  # notification
    if method == "initialize":
        return ok({"protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                   "capabilities": {"tools": {}, "prompts": {}},
                   "serverInfo": SERVER_INFO})
    if method == "ping":
        return ok({})
    if method == "tools/list":
        return ok({"tools": [{k: v for k, v in t.items() if k != "handler"} for t in TOOLS]})
    if method == "tools/call":
        tool = next((t for t in TOOLS if t["name"] == params.get("name")), None)
        if not tool:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": f"Unknown tool {params.get('name')}"}}
        try:
            text = tool["handler"](params.get("arguments") or {})
            return ok({"content": [{"type": "text", "text": text}], "isError": False})
        except Exception as e:  # report to the model rather than crash the server
            return ok({"content": [{"type": "text", "text": f"Error: {type(e).__name__}: {e}"}], "isError": True})
    if method == "prompts/list":
        return ok({"prompts": PROMPTS})
    if method == "prompts/get":
        a = params.get("arguments") or {}
        try:
            text = tool_handoff({"session": a.get("session") or "latest",
                                 "recent_turns": int(a.get("recent_turns") or 3)})
        except Exception as e:
            text = f"Could not load Codex session: {e}"
        instruction = ("Below is the handoff from an OpenAI Codex session that ran out of tokens. "
                       "Absorb it, check the current repo state (git status, running processes, logs it mentions), "
                       "then summarize where the work stands and what the next step is. Use the codex-sessions "
                       "tools (get_turn, search_session, file_changes) for any detail you need.\n\n")
        return ok({"description": "Codex session handoff",
                   "messages": [{"role": "user", "content": {"type": "text", "text": instruction + text}}]})
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"Method not found: {method}"}}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        else:
            batch = req if isinstance(req, list) else [req]
            resps = [r for r in (handle(x) for x in batch) if r is not None]
            resp = (resps if isinstance(req, list) else resps[0]) if resps else None
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
