# Persistent sessions

`SessionAgent` provides a database-backed conversation context for CLI and library callers.

## Lifecycle

```text
create_session → load_session → ask / chat → inject_context → delete_session
```

Creation stores a session ID, title, redacted system prompt, and public model routing metadata. Messages are persisted with sequence order; credentials are never stored. Loading reconstructs the conversation and optionally attaches the graph and model chain.

## Context management

`inject_context()` adds build or retrieval results without an LLM call. When the estimated token budget is exceeded, the agent keeps the system message and recent turns and persists a compact summary for older context.

## CLI

```bash
uv run drbrain session new --title "topic"
uv run drbrain session ask SESSION_ID "question"
uv run drbrain session chat SESSION_ID
uv run drbrain session list
uv run drbrain session export SESSION_ID
uv run drbrain session delete SESSION_ID
```

Session failures are explicit (`not found`, missing models, or provider errors). Message payloads pass through sensitive-value redaction before persistence.
