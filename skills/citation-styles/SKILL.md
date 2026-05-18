---
name: citation-styles
description: >
  Manage citation styles for Markdown reference export. Use when the user wants to format
  paper references in APA, Vancouver, Chicago, or MLA style, list available styles, create
  custom citation styles, or export papers with a specific citation format. Trigger when
  user mentions "citation format", "reference style", "bibliography style", "APA format",
  or wants to change how papers are cited in exports.
---

# Citation Styles

Format paper references in 4 built-in styles plus custom user-defined styles.

## Quick Start

```bash
drbrain style --list              # list available styles
drbrain style --show apa           # show APA style source
```

## Export with citation style

```bash
drbrain export --all --format md --style vancouver
drbrain export <paper_id> --format md --style chicago-author-date
```

## Custom styles

Place a Python file at `data/citation_styles/<name>.py` implementing:

```python
def format_ref(meta, idx=None):
    prefix = f"{idx}. " if idx else "- "
    return prefix + f" ... "
```

Then use: `drbrain export --all --style <name>`

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain style --list` | List all available citation styles |
| `drbrain style --show <name>` | Show style source code |
| `drbrain export --style <name>` | Export with specific citation style |

## Built-in Styles

| Name | Description |
|------|-------------|
| `apa` | APA 7th edition (author-year, default) |
| `vancouver` | Vancouver / ICMJE numeric style |
| `chicago-author-date` | Chicago 17th edition author-date |
| `mla` | MLA 9th edition |
