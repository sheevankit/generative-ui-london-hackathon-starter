"""
LangGraph entry point for the Contract Review Copilot example.

Mirrors agent/main.py's provider seam: Gemini 2.5 Flash via Google's
OpenAI-compatible endpoint. Gemini 3.x is a known trap (thought-signature
replay across tool turns); do NOT change the model line without instruction.
See FROZEN.md and agent/main.py:23 for the full note.

Import note (langgraph dev workaround):
    langgraph CLI loads graphs via `importlib.util.spec_from_file_location`
    when the graph spec is a path (contains "/"), which bypasses Python's
    package machinery and breaks `from .tools import ...`. We sidestep that
    by adding this file's directory to sys.path and using an absolute import
    against the sibling `tools` module. Works both via langgraph dev and via
    `python -c "from agent.graph import graph"`.
"""

import os
import sys
from pathlib import Path

from copilotkit import CopilotKitMiddleware
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from tools import review_contract, apply_redline  # noqa: E402  (sys.path tweak above)


model = ChatOpenAI(
    model=os.getenv("MODEL", "gemini-2.5-flash"),
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url=os.getenv(
        "MODEL_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    model_kwargs={"parallel_tool_calls": False},
)


agent = create_agent(
    model=model,
    tools=[review_contract, apply_redline],
    middleware=[CopilotKitMiddleware()],
    system_prompt="""
        You are a legal-document review assistant. Demo mode only — not legal advice.

        When asked to review a contract, call review_contract with the loaded document.

        ACTION HANDLING: When you receive a log_a2ui_event tool result naming
        "redline_accepted" or "redline_rejected", you MUST call apply_redline with
        the redlineId from the event context and the matching decision. Then briefly
        confirm in chat (1-2 sentences max).

        Keep all chat responses to 1-2 sentences. The UI does the heavy lifting.
    """,
)

graph = agent
