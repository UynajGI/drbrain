"""``drbrain library`` — bibliographic search over the local library.

The redesign moves the historical ``drbrain search`` (BM25 over paper titles,
concept labels and argument claims) here so ``search`` can mean *evidence*
retrieval, and the two semantics stay distinguishable:

* ``library search`` — bibliographic rows ``{local_id, type, label, score, …}``;
* ``search`` — retrievable evidence rows with text locators and index versions.

The implementation is the untouched legacy ``search_cmd`` (same flags, same
JSON shapes, same exit behavior), only re-registered under the new namespace.
"""

from __future__ import annotations

import typer

from drbrain.cli.query_commands import search_cmd

library_app = typer.Typer(help="Local bibliographic search over papers, concepts and arguments")

library_app.command("search")(search_cmd)

__all__ = ["library_app"]
