"""
Tool execution engine for the knowledge agent.

Implements the function calling protocol:
  1. LLM outputs a JSON with 'tool_calls' list
  2. Executor validates and dispatches each call to the correct tool
  3. Tool runs and returns an 'observation' dict
  4. Observations are injected back into the LLM conversation as the next user message
  5. LLM continues reasoning until it outputs 'final_answer' instead of 'tool_calls'

Supported tools (Layer 2 - live API):
  - query_uniprot   : UniProt REST (protein annotations, gene-protein mapping)
  - query_disgenet  : DisGeNET API (disease-gene associations)
  - query_kegg      : KEGG REST (pathway membership, gene-metabolite links)

Tool schemas are auto-collected from each tool module's TOOL_SCHEMA dict.
"""
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from .tools import uniprot_tool, disgenet_tool, kegg_tool

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

_TOOLS = {
    "query_uniprot": uniprot_tool,
    "query_disgenet": disgenet_tool,
    "query_kegg": kegg_tool,
}

# All tool schemas collected for function calling
ALL_TOOL_SCHEMAS = [
    uniprot_tool.TOOL_SCHEMA,
    disgenet_tool.TOOL_SCHEMA,
    kegg_tool.TOOL_SCHEMA,
]


def _debug_log(msg: str) -> None:
    if str(os.getenv("KNOWLEDGE_AGENT_DEBUG", "0")).strip().lower() in {"1", "true", "yes"}:
        print(f"[tool_executor][debug] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

class ToolExecutionError(Exception):
    pass


def execute_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a single tool call dict:
      {"name": "query_uniprot", "parameters": {...}}

    Returns an observation dict with:
      {"tool": name, "ok": bool, "result": ..., "error": str, "latency_sec": float}
    """
    name = str(tool_call.get("name", "")).strip()
    params = tool_call.get("parameters", tool_call.get("arguments", {}))
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            params = {}

    if name not in _TOOLS:
        return {
            "tool": name,
            "ok": False,
            "result": None,
            "error": f"Unknown tool '{name}'. Available: {list(_TOOLS.keys())}",
            "latency_sec": 0.0,
        }

    _debug_log(f"executing tool={name} params={json.dumps(params, ensure_ascii=False)[:200]}")
    t0 = time.time()
    try:
        result = _TOOLS[name].execute(params)
        ok = isinstance(result, dict) and result.get("ok", True)
        return {
            "tool": name,
            "ok": ok,
            "result": result,
            "error": result.get("error", "") if isinstance(result, dict) else "",
            "latency_sec": round(time.time() - t0, 3),
        }
    except Exception as e:
        _debug_log(f"tool_execution_error tool={name}: {type(e).__name__}: {e}")
        return {
            "tool": name,
            "ok": False,
            "result": None,
            "error": f"{type(e).__name__}: {e}",
            "latency_sec": round(time.time() - t0, 3),
        }


def execute_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Execute a list of tool calls sequentially, return list of observations."""
    observations = []
    for tc in tool_calls:
        obs = execute_tool_call(tc)
        observations.append(obs)
        _debug_log(
            f"tool={obs['tool']} ok={obs['ok']} latency={obs['latency_sec']}s "
            f"error={obs.get('error','')[:100]}"
        )
    return observations


# ---------------------------------------------------------------------------
# Function-calling loop helpers
# ---------------------------------------------------------------------------

def build_tool_schemas_for_llm(provider: str = "openai") -> List[Dict]:
    """
    Return tool schemas in the format expected by the LLM provider.

    OpenAI/DeepSeek/Qwen/GLM use:
      [{"type": "function", "function": {schema}}]

    Anthropic uses:
      [{"name": ..., "description": ..., "input_schema": {parameters}}]

    Gemini uses:
      [{"functionDeclarations": [{...}]}]
    """
    if provider in {"anthropic", "claude"}:
        return [
            {
                "name": s["name"],
                "description": s["description"],
                "input_schema": s["parameters"],
            }
            for s in ALL_TOOL_SCHEMAS
        ]
    elif provider == "gemini":
        return [
            {
                "functionDeclarations": [
                    {
                        "name": s["name"],
                        "description": s["description"],
                        "parameters": s["parameters"],
                    }
                    for s in ALL_TOOL_SCHEMAS
                ]
            }
        ]
    else:
        # OpenAI-compatible (default)
        return [{"type": "function", "function": s} for s in ALL_TOOL_SCHEMAS]


def parse_tool_calls_from_response(response: Any, provider: str = "openai") -> List[Dict]:
    """
    Extract tool call list from an LLM raw response dict.
    Returns a list of {"name": str, "parameters": dict} dicts.
    """
    if not isinstance(response, dict):
        return []

    # OpenAI/DeepSeek/Qwen/GLM: choices[0].message.tool_calls
    choices = response.get("choices", [])
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message", {})
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            result = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args_raw = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args = {}
                result.append({"name": name, "parameters": args, "id": tc.get("id", "")})
            return result

    # Anthropic: content[*].type == "tool_use"
    content = response.get("content", [])
    if isinstance(content, list):
        result = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                result.append({
                    "name": block.get("name", ""),
                    "parameters": block.get("input", {}),
                    "id": block.get("id", ""),
                })
        if result:
            return result

    # Gemini: candidates[0].content.parts[*].functionCall
    candidates = response.get("candidates", [])
    if candidates and isinstance(candidates[0], dict):
        parts = candidates[0].get("content", {}).get("parts", [])
        result = []
        for part in parts:
            if isinstance(part, dict) and "functionCall" in part:
                fc = part["functionCall"]
                result.append({
                    "name": fc.get("name", ""),
                    "parameters": fc.get("args", {}),
                    "id": "",
                })
        if result:
            return result

    # JSON-mode fallback: LLM returned {"tool_calls": [...]} inside JSON
    tool_calls = response.get("tool_calls", [])
    if tool_calls and isinstance(tool_calls, list):
        result = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                result.append({
                    "name": tc.get("name", tc.get("function", {}).get("name", "")),
                    "parameters": tc.get("parameters", tc.get("arguments", tc.get("function", {}).get("arguments", {}))),
                    "id": tc.get("id", ""),
                })
        return result

    return []


def format_observations_as_message(observations: List[Dict]) -> str:
    """
    Format tool observations into a message string to inject back into
    the LLM conversation (used in JSON-mode fallback path).
    """
    parts = []
    for obs in observations:
        tool = obs["tool"]
        ok = obs["ok"]
        latency = obs["latency_sec"]
        if ok:
            result_str = json.dumps(obs.get("result", {}), ensure_ascii=False)[:1500]
            parts.append(f"[TOOL OBSERVATION] {tool} (ok=True, {latency}s):\n{result_str}")
        else:
            parts.append(f"[TOOL OBSERVATION] {tool} (ok=False, {latency}s): ERROR: {obs.get('error','unknown')}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# High-level: run one tool-calling iteration
# ---------------------------------------------------------------------------

def run_tool_calling_round(
    raw_response: Optional[Dict],
    provider: str = "openai",
    max_tools_per_round: int = 3,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Given the raw LLM response dict, parse and execute tool calls.

    Returns:
      (tool_calls_parsed, observations)
      Both are empty lists if no tool calls were found.
    """
    if raw_response is None:
        return [], []

    tool_calls = parse_tool_calls_from_response(raw_response, provider)
    if not tool_calls:
        return [], []

    # Cap to avoid runaway costs
    tool_calls = tool_calls[:max_tools_per_round]
    observations = execute_tool_calls(tool_calls)
    return tool_calls, observations


# ---------------------------------------------------------------------------
# Observation summariser (compress large tool results for LLM context)
# ---------------------------------------------------------------------------

def summarise_observation(obs: Dict, max_chars: int = 800) -> Dict:
    """Return a copy of obs with result truncated/summarised for LLM context."""
    if not obs.get("ok"):
        return obs
    result = obs.get("result", {})
    if not isinstance(result, dict):
        return obs

    # Keep summary + top N results, drop full results list if too large
    summary = result.get("summary", {})
    top_results = result.get("results", [])[:5]
    compact = {
        "ok": result.get("ok", True),
        "query_type": result.get("query_type", ""),
        "summary": summary,
        "top_results": top_results,
    }
    # Add key fields per tool
    for key in ("gene_summary", "top_genes", "shared_pathways", "all_genes",
                "unique_compounds", "gene_pathway_counts"):
        if key in result:
            compact[key] = result[key]

    compact_str = json.dumps(compact, ensure_ascii=False)
    if len(compact_str) > max_chars:
        compact_str = compact_str[:max_chars] + "... [truncated]"
        compact = {"_truncated": True, "raw": compact_str}

    return {**obs, "result": compact}
