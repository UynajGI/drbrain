# Persistent Reasoning Sessions

SessionAgent provides DB-backed persistent context that survives across CLI invocations.
Create a session, inject build results into it, ask follow-up questions days later — the
agent remembers everything.

## Concept

### Stateless vs Persistent

| | ReasonerAgent (`drbrain reason`) | SessionAgent (`drbrain reason -s`) |
|---|---|---|
| Context | Per-call only | Persisted in DB |
| Multi-turn | No (forgets between calls) | Yes (full message history) |
| Build integration | None | `--session` injects extraction summary |
| Storage | In-memory | `agent_sessions` + `agent_messages` tables |
| Token budget | Unlimited (model window) | Auto-compresses at ~8000 tokens |
| CLI | `drbrain reason "query"` | `drbrain session ask/chat` |

### When to Use Sessions

- **Multi-paper research**: Build several papers into one session, then reason across them
- **Iterative exploration**: Ask → get answer → ask follow-up → refine understanding
- **Long-running analysis**: Build today, reason tomorrow, export next week
- **Team collaboration**: Export session, share with colleagues

## Session Lifecycle

```
create → [build injection] → reason → ask → chat → export → delete
  │              │              │       │      │        │        │
session new  build -s ID   reason -s  session  session  session  session
                            ID        ask ID   chat ID  export   delete ID
```

### 1. Create

```bash
drbrain session new "nlp-research"
# Returns: sess-a1b2c3d4

drbrain session list
# ┌──────────────────┬──────────────────┬────────┐
# │ ID               │ Title            │ Status │
# ├──────────────────┼──────────────────┼────────┤
# │ sess-a1b2c3d4    │ nlp-research     │ active │
# └──────────────────┴──────────────────┴────────┘
```

### 2. Build with Session Context

When you build a paper with `--session ID`, a structured extraction summary is
injected into the session. The SessionAgent now knows:

- Which papers were built
- Concepts by type and their relationships
- Coreference merges
- Refinement corrections

```bash
drbrain build p6a321e --session sess-a1b2c3d4
drbrain build p3b452c --session sess-a1b2c3d4    # inject second paper
```

### 3. Reason with Session

```bash
# Use session context for richer answers
drbrain reason -s sess-a1b2c3d4 "Summarize the key claims across all papers"
drbrain reason -s sess-a1b2c3d4 --workflow compare "Compare approaches to hallucination"
```

### 4. Follow-up Questions

```bash
drbrain session ask sess-a1b2c3d4 "Elaborate on point 3 from your last answer"
drbrain session ask sess-a1b2c3d4 "What methods do these papers have in common?"
```

### 5. Interactive Chat

```bash
drbrain session chat sess-a1b2c3d4
# Entering chat mode. Type /exit to quit.
# > What open problems remain?
# [Agent response with KG-grounded answer]
# > How could I approach the gap in methodology?
# [Agent response referencing earlier context]
# > /exit
```

### 6. Export

```bash
drbrain session export sess-a1b2c3d4 --output nlp-session.json
```

### 7. Delete

```bash
drbrain session delete sess-a1b2c3d4
```

## Context Injection

When `build --session ID` runs, the build pipeline injects a summary like:

```
[build summary: p6a321e]
  title: "Attention Is All You Need"
  concepts: 47 total
    Problem: 5 | Method: 18 | Conclusion: 12 | Gap: 3 | Debate: 2 | Actor: 7
  relations: 89 edges
    addresses: 24 | extends: 15 | challenges: 6 | ...
  coreference: 3 merges
  refinement: 2 corrections
```

This becomes part of the session message history. Subsequent tool calls by
the SessionAgent search these concepts and relations.

### Injection API

```python
from drbrain.extractor.session_agent import SessionAgent

agent = SessionAgent()
agent.load_session(db, "sess-xxx")
agent.inject_context(
    db,
    paper_id="p6a321e",
    concept_types={"Problem": 5, "Method": 18, ...},
    edge_relations={"addresses": 24, "extends": 15, ...},
    coref_merges=3,
    refinement_corrections=2,
)
```

## Session Storage

### DB Schema

Sessions live in two tables in `drbrain.db`:

**`agent_sessions`**:
| Column | Type | Description |
|--------|------|-------------|
| `session_id` | TEXT PK | UUID-based ID (`sess-xxxxxxxx`) |
| `title` | TEXT | Human-readable label |
| `system_prompt` | TEXT | Custom or default system prompt |
| `model_config` | TEXT | JSON array of model configs |
| `created_at` | DATETIME | Session creation time |
| `updated_at` | DATETIME | Last message time |
| `status` | TEXT | `active` or `deleted` |

**`agent_messages`**:
| Column | Type | Description |
|--------|------|-------------|
| `session_id` | TEXT FK | Links to `agent_sessions` |
| `seq` | INTEGER | Message sequence number |
| `role` | TEXT | `system`, `user`, `assistant`, `tool` |
| `content` | TEXT | Message content |
| `created_at` | DATETIME | Message timestamp |

### Token Budget and Compression

SessionAgent tracks an approximate token budget (default 8000 tokens). When
messages exceed the budget, it compresses older messages by summarizing them
and prepending the summary as system context. This keeps the conversation
coherent without exceeding model context windows.

## Best Practices

### Session Naming

Use descriptive names that capture the research theme:

```bash
drbrain session new "hallucination-detection-survey"
drbrain session new "kg-completion-methods-compare"
drbrain session new "ws-nlp-transformers-2024"
```

### Managing Session Growth

Sessions accumulate messages. To keep them focused:

- Create separate sessions for separate research questions
- Export and archive sessions you're done with
- Delete sessions that were exploratory throwaways

### Building into Sessions

- Build papers into the session BEFORE asking questions about them
- Build order matters: earlier papers get referenced more often
- Use `build --skip-refine` for faster injection if you plan to iterate

### Reasoning Workflows with Sessions

Workflows are session-aware. When you use `--workflow` with `--session`:

```bash
drbrain reason -s sess-xxx --workflow gap-analysis "What are the key open problems?"
```

The workflow runs within the session context, so its results are added to
session history for later follow-up.

## Limitations

- **No multi-user isolation**: Sessions are not scoped by user — all sessions
  are visible in the same DB.
- **Compression is lossy**: When the token budget triggers compression, details
  from older messages may be lost.
- **No branching**: Sessions are linear — you can't fork a session into two directions.
- **Graph state is external**: The session stores conversation, not graph state.
  If you re-build a paper, the session doesn't auto-update.
