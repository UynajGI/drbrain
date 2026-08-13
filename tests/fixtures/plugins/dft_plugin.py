"""External plugin fixture: DFT calculation skeleton (``software`` type).

This module lives *outside* ``src/drbrain`` — drbrain never imports it. It
demonstrates the file-based calling pattern for ``software`` plugins:

    1. create a temp working directory,
    2. write the arguments (structure / parameters) to an input file,
    3. run a (fake) software binary against that directory,
    4. parse the software's output file back into structured data.

The command here is a stand-in (``python -c``) — it is not a real VASP/LAMMPS
invocation — but it exercises the exact I/O shape a real software handler
would use, proving the protocol supports software calls without drbrain
knowing anything about the concrete binary.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from typing import Any

from drbrain.rag.plugins import Plugin, run_subprocess


def _run(arguments: dict[str, Any]) -> dict[str, Any]:
    """Write input → run fake software → parse output."""
    workdir = tempfile.mkdtemp(prefix="dft_plugin_")
    try:
        input_path = os.path.join(workdir, "input.json")
        output_path = os.path.join(workdir, "output.json")
        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(arguments, f, ensure_ascii=False)

        # Fake software: read input.json from cwd, write output.json.
        script = (
            "import json;"
            "inp = json.load(open('input.json'));"
            "out = {"
            "'total_energy': -8.123,"
            "'band_gap': 1.34,"
            "'n_sites': len(inp['structure'].get('sites', [])),"
            "'structure': inp['structure'],"
            "'parameters': inp['parameters']"
            "};"
            "json.dump(out, open('output.json', 'w'))"
        )
        proc = run_subprocess([sys.executable, "-c", script], cwd=workdir, timeout=60.0)
        if proc.returncode != 0:
            raise RuntimeError(f"假 DFT 软件退出码 {proc.returncode}: {proc.stderr.strip()[:500]}")

        with open(output_path, encoding="utf-8") as f:
            return json.load(f)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def register(registry: Any) -> None:
    """Register the ``run_dft_calculation`` plugin on ``registry``."""
    plugin = Plugin(
        name="run_dft_calculation",
        description=(
            "对给定晶胞结构与计算参数执行 DFT 计算，返回总能与带隙等 "
            "结构化结果（骨架示范：写输入→跑软件→解析输出）。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "structure": {
                    "type": "object",
                    "description": "晶胞结构（公式/晶格/原子坐标等）",
                },
                "parameters": {
                    "type": "object",
                    "description": "计算参数（截断能/k点/泛函等）",
                },
            },
            "required": ["structure", "parameters"],
        },
        plugin_type="software",
        backend="subprocess",
        version="dft_skeleton_v1",
        summary_fields=("total_energy",),
        metadata={"binary": "fake-vasp", "workflow": "write-run-parse"},
    )
    registry.register(plugin, _run)
