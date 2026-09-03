"""DrBrain WebUI — a small local web front-end over the existing CLI capabilities.

``drbrain webui`` serves a single-page workbench (search / ask / autoresearch
loop / compute jobs / assets) backed by the same database, ledger and plugin
registry the CLI uses. No extra dependencies: standard-library HTTP server plus
one static HTML file.
"""

from drbrain.webui.service import (
    RunManager,
    ask,
    assets,
    dashboard,
    experiments,
    plugins,
    run_claims,
    run_events,
    runs,
    search,
)

__all__ = [
    "RunManager",
    "ask",
    "assets",
    "dashboard",
    "experiments",
    "plugins",
    "run_claims",
    "run_events",
    "runs",
    "search",
]
