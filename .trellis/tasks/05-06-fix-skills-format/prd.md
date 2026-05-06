# Fix All 12 Skills to Skill-Creator Format

## What's wrong

Current skill descriptions are passive ("use this skill whenever the user asks to...").
Skill-creator requires "pushy" descriptions that trigger proactively, with ALL trigger
contexts in the description field, not the body.

## Fix (apply to all 12 skills in `skills/` + `clawhub.yaml`)

### Description field (frontmatter)
- Include specific trigger phrases the user might say
- Include contexts where the skill should activate even if the user didn't explicitly ask
- Be "pushy" — "Use this skill whenever..." not "Use this when the user asks..."
- Every "when to use" detail goes in description, NOT in body

### Body (markdown)
- Use imperative form ("Run `drbrain ...`" not "You can run `drbrain ...`")
- Progressive disclosure: start with common case, then edge cases
- Include concrete examples with real-looking paper IDs
- Keep under 100 lines

### Example of wrong vs right

WRONG description:
```
View a paper. Use when the user asks "show me paper X".
```

RIGHT description:
```
View paper contents at any depth — metadata, concepts by type, arguments with evidence, and graph edges. Use this skill whenever the user mentions a specific paper ID, wants to inspect a paper's contents, asks what concepts were extracted from a paper, needs to check if a paper was ingested correctly, or wants to find a paper's DOI, title, or other metadata. Also use before running analysis on a paper to verify its contents.
```

## Files
- `skills/<name>/SKILL.md` × 12 (research-analysis, paper-ingest, paper-query, citation-tracking, workspace-analysis, show, export, audit, translate, graph, import, index)
- Descriptions in `clawhub.yaml` should match SKILL.md descriptions

## Acceptance
- All 12 SKILL.md files have "pushy" descriptions
- All trigger contexts in description frontmatter
- Body uses imperative form
- Each skill has at least 2 concrete examples
