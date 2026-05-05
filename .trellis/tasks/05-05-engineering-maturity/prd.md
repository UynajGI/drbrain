# Engineering Maturity — Skills Publishing & Multi-Platform Agent Setup

## A. Skills Publishing ✅ DONE

`clawhub.yaml` at repo root, registers 5 DrBrain skills.

## B. Multi-Platform Agent Entry Injection

DrBrain is pip-installed. `drbrain setup` must inject agent entry files into
the user's current project directory. Templates shipped in `src/drbrain/templates/agents/`.

### Injection targets (user's project root)

| Template | Injects to | Platform |
|----------|-----------|----------|
| `CLAUDE.md.j2` | `<project>/CLAUDE.md` | Claude Code |
| `AGENTS.md.j2` | `<project>/AGENTS.md` | Codex, OpenClaw |
| `QWEN.md.j2` | `<project>/QWEN.md` | Qwen |
| `.cursorrules.j2` | `<project>/.cursorrules` | Cursor |
| `.clinerules.j2` | `<project>/.clinerules` | Cline |
| `.windsurfrules.j2` | `<project>/.windsurfrules` | Windsurf |
| `copilot-instructions.md.j2` | `<project>/.github/copilot-instructions.md` | GitHub Copilot |
| `plugin.json.j2` | `<project>/.claude-plugin/plugin.json` | Claude Code plugin |
| `marketplace.json.j2` | `<project>/.claude-plugin/marketplace.json` | Claude Code marketplace |
| `mcp.json.j2` | `<project>/.mcp.json` | MCP hosts |

Each template is under 20 lines. Lightweight wrapper: quick reference + points
to skills via clawhub.yaml registration (Claude Code auto-discovers), or
delegates to AGENTS.md (other platforms).

### `drbrain setup` — platform injection step

Add a new step after config write + dir creation:

1. Auto-detect installed AI platforms:
   - Claude Code: `which claude` or `~/.claude/` exists
   - Codex: `which codex` or `~/.codex/` exists
   - Qwen: check for Qwen-specific markers
   - Cursor/Cline/Windsurf: check for IDE markers
   - Copilot: check for `.github/` dir

2. Interactive prompt (at least one must be selected):
   ```
   Detected AI platforms: Claude Code, Cursor, Cline
   Select platforms to inject agent entries:
     [1] Claude Code
     [2] Cursor
     [3] Cline
     [a] All detected
     [n] None (skip)
   Enter numbers (comma-separated): 
   ```

3. Render template → write to target path. Warn if file already exists, offer skip/overwrite.

4. `.claude-plugin/` and `.mcp.json` are always offered together with Claude Code.

### Template content (brief)

Every template: 1 paragraph project description + key commands + pointer to skills.
Follow scholaraio's `CLAUDE.md` pattern — intentionally light.

### Acceptance
- 10 templates in `src/drbrain/templates/agents/`
- `drbrain setup` detects platforms and injects entries
- User must select at least one platform
- Each injected file is under 20 lines

## C. Community Infrastructure (repo root)

Standard community files for the DrBrain repository:
- `CONTRIBUTING.md` — dev setup, PR flow, commit conventions, test/lint commands
- `SECURITY.md` — supported version, vulnerability reporting
- `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1
- `CITATION.cff` — academic citation metadata
- `.pre-commit-config.yaml` — ruff check + format, trailing-whitespace, end-of-file-fixer, check-yaml, check-json

### Acceptance
- All 5 files exist with correct content
- `.pre-commit-config.yaml` validates
