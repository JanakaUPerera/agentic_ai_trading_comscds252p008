from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

import requests

from src.agent_tools import get_tool_descriptions, list_tool_names, run_tool


# =========================================================
# Config
# =========================================================

try:
    from src.config import GROQ_API_KEY
except Exception:
    GROQ_API_KEY = ""

try:
    from src.config import GROQ_API_URL
except Exception:
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

try:
    from src.config import GROQ_MODEL
except Exception:
    GROQ_MODEL = "llama-3.3-70b-versatile"

try:
    from src.config import GROQ_TEMPERATURE
except Exception:
    GROQ_TEMPERATURE = 0.2

try:
    from src.config import AGENT_MAX_STEPS
except Exception:
    AGENT_MAX_STEPS = 10


# =========================================================
# Optional memory
# =========================================================

try:
    from src.agent_memory import get_memory_context
except Exception:
    def get_memory_context() -> str:
        return "No previous runs recorded."


# =========================================================
# Agent rules
# =========================================================

DEPENDENCIES: dict[str, list[str]] = {
    "preprocess_market_data": ["fetch_market_data"],
    "run_feature_engineering": ["preprocess_market_data"],
    "run_decisions": ["run_feature_engineering"],
    "apply_risk_controls": ["run_decisions"],
    "run_backtest": ["apply_risk_controls"],
}

OPTIONAL_TOOLS = {"run_eda", "run_clustering", "check_news_sentiment"}

SYSTEM_PROMPT = """
You are an autonomous crypto trading analysis agent.

Your job is to decide which analysis tool to run next, based on the current state
of the workflow, and eventually produce a final trading recommendation.

You may ONLY respond with valid JSON.
Do not include markdown fences.
Do not include explanation outside JSON.

Available tools:
{tool_descriptions}

Hard rules:
1. fetch_market_data must happen before preprocess_market_data.
2. preprocess_market_data must happen before run_feature_engineering.
3. run_feature_engineering must happen before run_decisions.
4. apply_risk_controls must happen after run_decisions.
5. run_backtest must happen after apply_risk_controls.
6. check_news_sentiment is optional but recommended before run_decisions.
7. run_eda and run_clustering are optional and should only be used if they add value.
8. Do not call the same tool twice unless there is a clear retry reason.
9. If a tool fails, decide whether to retry, skip, or finish with a cautious recommendation.
10. When enough information is available, finish.

Valid response formats:

To run a tool:
{{
  "done": false,
  "tool": "tool_name",
  "reason": "brief reason"
}}

To finish:
{{
  "done": true,
  "recommendation": "final recommendation summary",
  "confidence": "High/Medium/Low",
  "rationale": "brief explanation"
}}

Be decisive, structured, and conservative.
"""

CORRECTION_PROMPT = """
Your previous message was invalid.
Reply with valid JSON only.
Choose one of these tool names exactly:
{tool_names}
Or finish with:
{{
  "done": true,
  "recommendation": "...",
  "confidence": "High/Medium/Low",
  "rationale": "..."
}}
"""


# =========================================================
# Helpers
# =========================================================

MOCK_ACTIONS = [
    {"done": False, "tool": "fetch_market_data", "reason": "Start with fresh data."},
    {"done": False, "tool": "preprocess_market_data", "reason": "Clean the fetched data."},
    {"done": False, "tool": "run_feature_engineering", "reason": "Generate indicators."},
    {"done": False, "tool": "check_news_sentiment", "reason": "Get external context."},
    {"done": False, "tool": "run_decisions", "reason": "Create trading decisions."},
    {"done": False, "tool": "apply_risk_controls", "reason": "Adjust for risk."},
    {"done": False, "tool": "run_backtest", "reason": "Evaluate the strategy."},
    {
        "done": True,
        "recommendation": "Dry-run completed successfully.",
        "confidence": "Medium",
        "rationale": "All required workflow stages executed in valid order."
    },
]

def run_agent_loop_mock(verbose: bool = True) -> dict[str, Any]:
    executed_tools = []
    steps = []

    for step_number, action in enumerate(MOCK_ACTIONS, start=1):
        if action["done"]:
            return {
                "status": "success",
                "recommendation": action["recommendation"],
                "confidence": action["confidence"],
                "rationale": action["rationale"],
                "executed_tools": executed_tools,
                "steps": steps,
            }

        tool_name = action["tool"]
        tool_result = run_tool(tool_name)
        executed_tools.append(tool_name)

        steps.append({
            "step": step_number,
            "status": "tool_executed",
            "action": action,
            "tool_result": tool_result,
        })

        if verbose:
            print(f"\n--- Mock step {step_number} ---")
            print(f"Tool: {tool_name}")
            print(tool_result["summary"])

    return {
        "status": "partial",
        "recommendation": "Mock loop ended unexpectedly.",
        "confidence": "Low",
        "rationale": "Mock actions finished without explicit completion.",
        "executed_tools": executed_tools,
        "steps": steps,
    }


def _truncate_text(value: Any, max_len: int = 300) -> str:
    text = str(value)
    return text if len(text) <= max_len else text[:max_len] + "... [truncated]"


def _truncate_details(details: dict[str, Any], max_items: int = 5) -> dict[str, Any]:
    if not isinstance(details, dict):
        return {}

    trimmed = {}
    for i, (key, value) in enumerate(details.items()):
        if i >= max_items:
            trimmed["more"] = "... [truncated]"
            break

        if isinstance(value, list):
            trimmed[key] = value[:3]
        elif isinstance(value, dict):
            small = {}
            for j, (k, v) in enumerate(value.items()):
                if j >= 3:
                    small["more"] = "... [truncated]"
                    break
                small[k] = _truncate_text(v, 150)
            trimmed[key] = small
        else:
            trimmed[key] = _truncate_text(value, 150)

    return trimmed


def call_groq(messages: list[dict[str, str]]) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing in src.config.")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "temperature": GROQ_TEMPERATURE,
        "messages": messages,
    }

    response = requests.post(
        GROQ_API_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )

    if response.status_code >= 400:
        error_text = response.text[:2000]
        raise RuntimeError(
            f"Groq API error {response.status_code}: {error_text}"
        )

    response.raise_for_status()
    
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def extract_json_object(text: str) -> dict[str, Any] | None:
    """
    Try to extract the first JSON object from model output.
    Supports:
    - raw JSON
    - JSON inside ```json fences
    """
    text = text.strip()

    # Case 1: whole response is JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Case 2: fenced JSON
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        candidate = fenced_match.group(1).strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Case 3: first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None

    return None


def validate_action(action: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(action, dict):
        return False, "Action is not a JSON object."

    done = action.get("done")

    if done is True:
        if not isinstance(action.get("recommendation"), str) or not action.get("recommendation").strip():
            return False, "Finished action must include a non-empty recommendation."
        return True, ""

    if done is False:
        tool_name = action.get("tool")
        if tool_name not in list_tool_names():
            return False, f"Invalid tool name: {tool_name}"
        if not isinstance(action.get("reason"), str) or not action.get("reason").strip():
            return False, "Tool action must include a non-empty reason."
        return True, ""

    return False, "Action must include done=true or done=false."


def get_missing_dependencies(tool_name: str, executed_tools: list[str]) -> list[str]:
    required = DEPENDENCIES.get(tool_name, [])
    return [tool for tool in required if tool not in executed_tools]


def format_tool_result_for_agent(result: dict[str, Any]) -> str:
    """
    Keep tool feedback compact but informative for the model.
    """
    compact = {
        "tool": result.get("tool"),
        "status": result.get("status"),
        "summary": result.get("summary"),
        "details": result.get("details", {}),
        "output_files": result.get("output_files", []),
    }
    return json.dumps(compact, indent=2, default=str)


def build_initial_messages() -> list[dict[str, str]]:
    tool_descriptions = get_tool_descriptions()
    memory_context = get_memory_context()

    system_message = SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)

    user_message = (
        f"Today's date is {date.today()}.\n\n"
        f"Memory from previous runs:\n{memory_context}\n\n"
        "Begin the trading workflow.\n"
        "Choose the next best tool, respect dependencies, and finish only when you have enough evidence."
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def build_progress_summary(
    executed_tools: list[str],
    latest_result: dict[str, Any] | None = None,
) -> str:
    lines = [
        f"Executed tools so far: {executed_tools if executed_tools else 'None'}"
    ]

    if latest_result is not None:
        lines.append("Latest tool result:")
        lines.append(format_tool_result_for_agent(latest_result))

    return "\n".join(lines)


# =========================================================
# Main loop
# =========================================================

def run_agent_loop(verbose: bool = True) -> dict[str, Any]:
    executed_tools: list[str] = []
    steps: list[dict[str, Any]] = []
    latest_tool_result: dict[str, Any] | None = None

    base_system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT.format(
            tool_descriptions=get_tool_descriptions()
        ),
    }

    memory_context = get_memory_context()

    if verbose:
        print("Starting agentic trading analysis loop...")
        print("=" * 60)

    for step_number in range(1, AGENT_MAX_STEPS + 1):
        if verbose:
            print(f"\n--- Agent step {step_number} ---")

        state_summary = {
            "today": str(date.today()),
            "executed_tools": executed_tools,
            "latest_tool_result": latest_tool_result,
            "memory_context": memory_context,
            "remaining_tools": [
                tool for tool in list_tool_names()
                if tool not in executed_tools
            ],
        }

        user_message = {
            "role": "user",
            "content": (
                "Current workflow state:\n"
                f"{json.dumps(state_summary, indent=2, default=str)}\n\n"
                "Choose the next best action."
            ),
        }

        messages = [base_system_message, user_message]

        raw_response = call_groq(messages)

        if verbose:
            print("Agent raw response:")
            print(raw_response)

        action = extract_json_object(raw_response)
        if action is None:
            steps.append({
                "step": step_number,
                "status": "invalid_json",
                "raw_response": raw_response,
            })
            latest_tool_result = {
                "tool": "agent_response_parser",
                "status": "error",
                "summary": "Agent returned invalid JSON.",
                "details": {"raw_response": raw_response[:500]},
                "output_files": [],
            }
            continue

        is_valid, error_message = validate_action(action)
        if not is_valid:
            steps.append({
                "step": step_number,
                "status": "invalid_action",
                "action": action,
                "error": error_message,
            })
            latest_tool_result = {
                "tool": "agent_action_validator",
                "status": "error",
                "summary": error_message,
                "details": {"action": action},
                "output_files": [],
            }
            continue

        if action["done"] is True:
            recommendation = action.get("recommendation", "").strip()
            confidence = action.get("confidence", "Medium")
            rationale = action.get("rationale", "").strip()

            if verbose:
                print("Agent finished successfully.")
                print(f"Recommendation: {recommendation}")
                print(f"Confidence: {confidence}")
                print(f"Rationale: {rationale}")

            return {
                "status": "success",
                "recommendation": recommendation,
                "confidence": confidence,
                "rationale": rationale,
                "executed_tools": executed_tools,
                "steps": steps,
            }

        tool_name = action["tool"]
        reason = action.get("reason", "").strip()

        if tool_name in executed_tools:
            latest_tool_result = {
                "tool": tool_name,
                "status": "error",
                "summary": f"Tool '{tool_name}' has already been executed.",
                "details": {"reason": "duplicate_tool_blocked"},
                "output_files": [],
            }
            steps.append({
                "step": step_number,
                "status": "duplicate_tool_blocked",
                "action": action,
            })
            continue

        missing_dependencies = get_missing_dependencies(tool_name, executed_tools)
        if missing_dependencies:
            latest_tool_result = {
                "tool": tool_name,
                "status": "error",
                "summary": f"Missing dependencies: {missing_dependencies}",
                "details": {"reason": "dependency_blocked"},
                "output_files": [],
            }
            steps.append({
                "step": step_number,
                "status": "dependency_blocked",
                "action": action,
                "missing_dependencies": missing_dependencies,
            })
            continue

        if verbose:
            print(f"Running tool: {tool_name}")
            print(f"Reason: {reason}")

        tool_result = run_tool(tool_name)
        executed_tools.append(tool_name)
        latest_tool_result = {
            "tool": tool_result.get("tool"),
            "status": tool_result.get("status"),
            "summary": str(tool_result.get("summary", ""))[:600],
            "details": _truncate_details(tool_result.get("details", {}), max_items=5),
            "output_files": tool_result.get("output_files", [])[:3],
        }

        steps.append({
            "step": step_number,
            "status": "tool_executed",
            "action": action,
            "tool_result": latest_tool_result,
        })

        if verbose:
            print("Tool result:")
            print(json.dumps(latest_tool_result, indent=2, default=str))

    return {
        "status": "partial",
        "recommendation": "Agent reached maximum steps before producing a final recommendation.",
        "confidence": "Low",
        "rationale": "The agent did not finish within the configured step limit.",
        "executed_tools": executed_tools,
        "steps": steps,
    }


# =========================================================
# Optional CLI test
# =========================================================

if __name__ == "__main__":
    result = run_agent_loop(verbose=True)
    print("\nFinal result:")
    print(json.dumps(result, indent=2, default=str))