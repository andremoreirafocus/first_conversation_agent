from dotenv import load_dotenv
import os
import math
import json
import numexpr
from typing import List, Dict, Any, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os

load_dotenv()
serp_api_key = os.getenv("SERPAPI_API_KEY")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY not found. You can set it with `%env` in the notebook or enter it below.")
    OPENAI_API_KEY = input("Enter your OPENAI_API_KEY: ").strip()
else:
    print("✅ OPENAI_API_KEY loaded successfully.")

if not serp_api_key:
    print("⚠️ SERPAPI_API_KEY not found. You can set it with `%env` in the notebook or enter it below.")
    serp_api_key = input("Enter your SERPAPI_API_KEY: ").strip()
else:
    print("✅ SERPAPI_API_KEY loaded successfully.")

print("✅ API keys loaded successfully!")

@tool("internet_search")
def internet_search(query: str) -> str:
    """Search Google via SerpAPI for up to date information."""
    serp_api_key = os.environ["SERPAPI_API_KEY"]
    params = {"engine": "google", "gl": "us", "hl": "en"}
    search = SerpAPIWrapper(params=params, serpapi_api_key=serp_api_key)
    return search.run(query)

@tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate a single line mathematical expression with numexpr."""
    local_dict = {"pi": math.pi, "e": math.e}
    out = numexpr.evaluate(
        expression.strip(),
        global_dict={},
        local_dict=local_dict,
    )
    return str(out)

tools = [internet_search, calculator]
tool_map: Dict[str, Any] = {t.name: t for t in tools}
llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=800).bind_tools(tools, tool_choice="auto")

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def run_once(prompt: str, max_steps: int = 4) -> str:
    messages = [HumanMessage(content=prompt)]
    
    for _ in range(max_steps):
        ai: AIMessage = llm.invoke(messages)
        messages.append(ai)

        calls = getattr(ai, "tool_calls", None) or []
        if not calls:
            print("No tool calls found, breaking the loop.")
            break

        for call in calls:
            name = call["name"]
            args = call.get("args", {})
            print(f"Invoking tool: {name} with args: {args}")
            result = tool_map[name].invoke(args)
            print(f"Result from tool {name}: {result}")
            messages.append(ToolMessage(
                content=str(result),
                name=name,
                tool_call_id=call["id"]
            ))
            # print(f"Message from tool {name}: {messages[-1]}")
    return messages[-1].content



def llm_node(state: AgentState) -> AgentState:
    ai = llm.invoke(state["messages"])
    return {"messages": [ai]}

tool_node = ToolNode(tools=tools)

graph = StateGraph(AgentState)
graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "llm")

def route(state: AgentState):
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    return "tools" if calls else END

graph.add_conditional_edges("llm", route, {"tools": "tools", END: END})
graph.add_edge("tools", "llm")
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

def _short(msg: BaseMessage, max_len: int = 140) -> str:
    """Compact one-line view of a message."""
    role = type(msg).__name__.replace("Message", "").lower()
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        # some tool outputs can be list payloads
        try:
            content = json.dumps(content)
        except Exception:
            content = str(content)
    text = str(content).replace("\n", " ").strip()
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    # include tool name or function call info when available
    if hasattr(msg, "tool_calls") and getattr(msg, "tool_calls"):
        tnames = [tc.get("name", "tool") for tc in msg.tool_calls]
        return f"{role}: tool_calls -> {tnames}"
    if isinstance(msg, ToolMessage):
        return f"{role}({msg.name}): {text}"
    return f"{role}: {text}"
def print_state_snapshot(app, config, title: str):
    """Print current graph state and memory for a given thread."""
    snap = app.get_state(config)
    values = snap.values or {}
    msgs: List[BaseMessage] = values.get("messages", [])
    print(f"\n=== {title} | state snapshot ===")
    print(f"messages: {len(msgs)} total")
    for i, m in enumerate(msgs[-5:], start=max(0, len(msgs)-5) + 1):
        print(f"  {i:>3}: {_short(m)}")
    # show routing info and queued tasks if present
    nxt = getattr(snap, "next", None)
    tasks = getattr(snap, "tasks", None)
    if nxt:
        print(f"next nodes: {list(nxt)}")
    if tasks:
        print(f"queued tasks: {tasks}")
    # minimal memory view via checkpointer for this thread
    # MemorySaver keeps one latest checkpoint per thread by default, so show existence
    print("memory: in-memory checkpoint present for this thread")
def run_with_tracing(app, input_state: AgentState, config, title: str):
    """Run the graph while printing per-node updates and final memory."""
    print(f"\n=== {title} | execution trace ===")
    final = None
    # stream_mode="updates" surfaces node-level updates
    for event in app.stream(input_state, config=config, stream_mode="updates"):
        for node, upd in event.items():
            # upd is a dict like {"messages": [<new msg>]} or tool results
            keys = list(upd.keys())
            print(f"[enter {node}] updated: {keys}")
            # if messages updated, print the last one briefly
            msgs = upd.get("messages") or []
            if msgs:
                print(f"  {_short(msgs[-1])}")
            print(f"[leave {node}]")
            final = upd
    # show final assistant message from app.get_state
    print_state_snapshot(app, config, title=f"{title} | after run")
    snap = app.get_state(config)
    msgs = snap.values.get("messages", [])
    return msgs[-1].content if msgs else ""

def main():
    # configs
    cfg = {"configurable": {"thread_id": "nyc-weather-session"}}
    print("Starting interactive conversation agent...")
    print("Type 'exit' to quit the conversation.")
    while True:
        question = input("Your question: ")
        if question.lower() == "exit":
            break
        answer = run_with_tracing(
            app,
            {"messages": [HumanMessage(content=question)]},
            config={**cfg, "recursion_limit": 20, "max_concurrency": 1},
            title="Question",
        )
        print("\nTURN 1 (final assistant):\n", answer)
        #print_state_snapshot(app, cfg, title="MAIN THREAD memory view")

if __name__ == "__main__":
    main()
