"""Shared agent defaults and canonical tool schemas, with no orchestration state."""

from drbrain.extractor.agent_tools import TOOL_DEFINITIONS

AGENT_TEMPERATURE = 0.3
AGENT_MAX_TOKENS = 1024
MAX_RESULT_SUMMARY_CHARS = 800
SESSION_TOKEN_BUDGET = 8000
SESSION_KEEP_RECENT = 6
CANONICAL_TOOL_SPECS = {
    definition["function"]["name"]: definition for definition in TOOL_DEFINITIONS
}
GRAPH_TOOL_NAMES = list(CANONICAL_TOOL_SPECS)
