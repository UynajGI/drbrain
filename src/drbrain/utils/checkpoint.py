"""Task-level checkpointing and idempotency primitives.

Three small patterns validated by the research-side batch pipelines,
generalized for the main package:

- :class:`Checkpoint` — offset/progress JSON persistence with atomic saves
  and out-of-bounds auto-reset on load.
- :class:`IdempotentFile` — source-file fingerprint marker: skip reprocessing
  while the source file is unchanged since it was last handled.
- :class:`DoneFlag` — completion flag file; ``is_done()`` checks existence
  only, never a volatile offset.

Usage::

    from drbrain.utils.checkpoint import Checkpoint, DoneFlag, IdempotentFile

    ckpt = Checkpoint(Path("data/offset.json"))
    offset = ckpt.load(max_value=len(todo))  # 0 when missing/corrupt/stale
    ...
    ckpt.save(offset)                        # atomic tmp-write + rename

    idem = IdempotentFile(Path("data/refine.jsonl"))
    if not idem.is_done():
        ...
        idem.mark_done()

    flag = DoneFlag(Path("data/concepts_done.flag"))
    if not flag.is_done():
        ...
        flag.done()
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from loguru import logger


class Checkpoint:
    """JSON-backed offset/progress checkpoint with atomic saves.

    Pattern source: ``research/scripts/cg_fulltext_concepts_api.py`` (v2
    stability rework, 2026-08-08/09) — the batch loop dumps ``{"offset": n}``
    every batch, re-reads it on restart to resume exactly, and resets to 0 when
    the saved offset is out of bounds of a regenerated (shorter) todo list.

    This class generalizes that to any JSON-serializable state, with two
    safety properties from the field: :meth:`save` writes a tmp file in the
    same directory and atomically renames it over the checkpoint (a crash never
    leaves half-written JSON), and :meth:`load` accepts a ``max_value`` so a
    stale offset auto-resets to the default instead of silently empty-looping
    to "done".

    Example::

        ckpt = Checkpoint(Path("data/offset.json"))
        offset = ckpt.load(max_value=len(todo))  # 0 if missing/corrupt/stale
        ...
        ckpt.save(offset)
    """

    def __init__(self, path: str | os.PathLike[str], default: Any = 0) -> None:
        """Create a checkpoint stored at *path*.

        Args:
            path: JSON file holding the checkpoint state.
            default: Value returned by :meth:`load` when no valid state exists,
                and the value used for out-of-bounds reset.
        """
        self.path = Path(path)
        self.default = default

    def load(self, max_value: int | None = None) -> Any:
        """Return the saved state, or *default* when it is unavailable.

        Args:
            max_value: When given and the saved state is an ``int`` that is
                ``>= max_value``, the state is stale (e.g. an offset into an
                old, longer todo list) and *default* is returned — mirroring
                the reset-to-0 logic of the research script.  Dict states are
                returned as-is; callers holding progress inside a dict do
                their own bounds check.

        Returns:
            The saved state (an ``int`` offset, a ``dict``, or any other JSON
            value), or ``self.default`` when the file is missing, corrupt, or
            out of bounds.
        """
        if not self.path.exists():
            return self.default
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("[checkpoint] {} unreadable, resetting to {!r}", self.path, self.default)
            return self.default
        if max_value is not None and isinstance(state, int) and state >= max_value:
            logger.warning(
                "[checkpoint] {} state {!r} out of bounds (max {!r}), resetting to {!r}",
                self.path,
                state,
                max_value,
                self.default,
            )
            return self.default
        return state

    def save(self, state: Any) -> None:
        """Persist *state* atomically: write a tmp file, then rename over it.

        The tmp file lives next to the checkpoint so the rename stays on one
        filesystem (atomic on POSIX).  A crash mid-write leaves the previous
        checkpoint intact rather than a truncated JSON blob.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        data = json.dumps(state, ensure_ascii=False) + "\n"
        tmp.write_text(data, encoding="utf-8")
        tmp.replace(self.path)


class IdempotentFile:
    """File-fingerprint idempotency marker: skip work when the source is unchanged.

    Pattern source: ``research/scripts/cg_concept_merge.py`` — it records
    ``md5(f"{st_mtime}:{st_size}")[:16]`` of the source file into
    ``<src>.merged_marker`` and skips the merge when the marker still matches,
    so a rerun over untouched input does nothing.

    Two fingerprint modes:

    - ``use_content_hash=False`` (default): ``md5(mtime:size)`` — cheap stat
      only, byte-for-byte identical to the research script.
    - ``use_content_hash=True``: md5 of the file bytes — immune to mtime
      resolution / same-size rewrites, at the cost of reading the file.

    Example::

        idem = IdempotentFile(Path("data/refine.jsonl"))
        if not idem.is_done():
            ...  # process the file once
            idem.mark_done()
    """

    #: Default marker filename suffix (research convention: ``<src>.merged_marker``).
    _MARKER_SUFFIX = ".idem_marker"

    def __init__(
        self,
        source: str | os.PathLike[str],
        marker: str | os.PathLike[str] | None = None,
        *,
        use_content_hash: bool = False,
    ) -> None:
        """Create an idempotency marker for *source*.

        Args:
            source: Input file whose change should trigger reprocessing.
            marker: Where the fingerprint is stored.  Defaults to
                ``<source>.idem_marker``.
            use_content_hash: Fingerprint the file bytes instead of
                ``(mtime, size)``.  More robust, slower for large files.
        """
        self.source = Path(source)
        self.marker = (
            Path(marker) if marker is not None else Path(str(source) + self._MARKER_SUFFIX)
        )
        self.use_content_hash = use_content_hash

    def fingerprint(self) -> str:
        """Return the current fingerprint of the source file.

        Raises:
            FileNotFoundError: when the source file does not exist.
        """
        if self.use_content_hash:
            digest = hashlib.md5()
            with open(self.source, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    digest.update(chunk)
            return digest.hexdigest()[:16]
        st = self.source.stat()
        digest = hashlib.md5()
        digest.update(f"{st.st_mtime}:{st.st_size}".encode())
        return digest.hexdigest()[:16]

    def is_done(self) -> bool:
        """True when the source is unchanged since :meth:`mark_done`.

        Returns False when no marker exists, the marker is unreadable, or the
        source file is missing — all mean the work still needs to run.
        """
        if not self.source.exists() or not self.marker.exists():
            return False
        try:
            prev = self.marker.read_text(encoding="utf-8").strip()
        except OSError:
            return False
        return prev == self.fingerprint()

    def mark_done(self) -> None:
        """Record the current source fingerprint so later ``is_done()`` is True.

        Raises:
            FileNotFoundError: when the source file does not exist.
        """
        self.marker.parent.mkdir(parents=True, exist_ok=True)
        self.marker.write_text(self.fingerprint() + "\n", encoding="utf-8")


class DoneFlag:
    """Completion marker file; ``is_done()`` checks existence only.

    Pattern source: research batch scripts — e.g.
    ``research/scripts/cg_fulltext_concepts_api.py`` writes
    ``concepts_done.flag`` (a timestamp) beside the offset file, and the
    unattended supervisor keys completion off that flag's presence instead of
    any volatile offset, which can drift as the todo list is regenerated.

    The flag carries no progress data — just presence — so it stays valid
    across restarts and todo-list churn.

    Example::

        flag = DoneFlag(Path("data/concepts_done.flag"))
        if not flag.is_done():
            ...
            flag.done()
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        """Create a done flag stored at *path*."""
        self.path = Path(path)

    def done(self, text: str | None = None) -> None:
        """Write the flag file (completion marker).

        Args:
            text: Optional content.  Defaults to a local timestamp, matching
                the research scripts' ``concepts_done.flag`` convention.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        content = text if text is not None else time.strftime("%Y-%m-%d %H:%M:%S")
        self.path.write_text(content, encoding="utf-8")

    def is_done(self) -> bool:
        """True when the flag file exists."""
        return self.path.exists()
