"""Lenient JSON parsing for LLM responses.

LLMs frequently wrap JSON in markdown code fences, prepend/append prose, or
emit output truncated mid-way.  :func:`parse_llm_json` tolerates all three
failure modes before raising the standard ``json.JSONDecodeError``.

Patterns distilled from corpus-scale batch scripts:

- first-brace-block extraction via ``re.search(r'\\{.*\\}', text, re.S)``
  (handles prose around the JSON).
- fence stripping (``re.sub(r"^```[a-z]*\\n?", ...)``) plus truncation
  tolerance (keep the prefix ending at the last parseable ``}``).
"""

from __future__ import annotations

import json
import re
from typing import Any

#: Markdown code fence (```json ... ``` or bare ``` ... ```) around the payload.
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```\s*$", re.S)

#: First brace block: from the first ``{`` to the last ``}`` in the text.
_BLOCK_RE = re.compile(r"\{.*\}", re.S)


def _strip_fence(text: str) -> str:
    """Strip a surrounding ```...``` code fence, if the text is exactly one.

    Returns the input unchanged when no fence is present.
    """
    m = _FENCE_RE.match(text)
    return m.group(1).strip() if m else text


def parse_llm_json(text: str) -> Any:
    r"""Leniently parse JSON from an LLM response.

    Tries, in order:

    1. ``json.loads`` on the raw text.
    2. The same, after stripping a markdown code fence (`` ```json ... ``` ``).
    3. Extraction of the first brace block via ``re.search(r'\{.*\}', text,
       re.S)`` — spans prose written around the JSON object.
    4. Truncation tolerance: walk ``}`` positions right-to-left and return the
       longest prefix that ends at a ``}`` and parses.  Recovers responses
       whose tail is garbage (e.g. stray ``}``s, a second object) or was cut
       off after the closing brace of a leading fragment.

    Returns the parsed value (any valid JSON, not only objects — step 1
    passes through arrays/scalars unchanged).

    Raises:
        json.JSONDecodeError: when no strategy yields valid JSON.
    """
    if not isinstance(text, str) or not text.strip():
        raise json.JSONDecodeError("parse_llm_json: empty or non-string input", text, 0)

    # 1. direct attempt
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. strip ```json fence, then try again
    stripped = _strip_fence(text)
    if stripped is not text:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # 3. extract the first { ... } block (spans surrounding prose)
    m = _BLOCK_RE.search(stripped)
    if m is None:
        raise json.JSONDecodeError("parse_llm_json: no brace block found", text, 0)
    block = m.group(0)
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        pass

    # 4. truncation tolerance: longest prefix ending at a '}' that parses
    #    (handles over-greedy matches, stray trailing '}', truncated tails)
    for pos in range(len(block) - 1, 0, -1):
        if block[pos] == "}":
            try:
                return json.loads(block[: pos + 1])
            except json.JSONDecodeError:
                continue

    raise json.JSONDecodeError("parse_llm_json: no valid JSON object found in response", text, 0)
