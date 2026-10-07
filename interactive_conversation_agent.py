from dotenv import load_dotenv
import os
import math
import json
import numexpr
from typing import List, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os
from getpass import getpass
import traceback

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

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def load_api_keys():
    serp_api_key = os.getenv("SERPAPI_API_KEY")
    open_api_key = os.getenv('OPENAI_API_KEY')
    if not open_api_key:
        print("⚠️ OPENAI_API_KEY not found. You can set it with `%env` in the notebook or enter it below.")
        open_api_key = getpass("Enter your OPENAI_API_KEY: ").strip()
        os.environ["OPENAI_API_KEY"] = open_api_key
    else:
        print("✅ OPENAI_API_KEY loaded successfully.")
    if not serp_api_key:
        print("⚠️ SERPAPI_API_KEY not found. You can set it with `%env` in the notebook or enter it below.")
        serp_api_key = getpass("Enter your SERPAPI_API_KEY: ").strip()
        os.environ["SERPAPI_API_KEY"] = serp_api_key
    else:
        print("✅ SERPAPI_API_KEY loaded successfully.")
    print("✅ API keys loaded successfully!")
    return open_api_key, serp_api_key


class ConversationAgent:
    """One stateful, tool-enabled conversation session."""

    def __init__(
        self,
        model_name: str,
        thread_id: str = "interactive-session",
        debug: bool = False,
    ):
        self.debug = debug
        self.config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 20,
            "max_concurrency": 1,
        }
        self.app = self._build_app(model_name)

    @staticmethod
    def _route(state: AgentState):
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        return "tools" if calls else END

    @staticmethod
    def _short(message: BaseMessage, max_len: int = 140) -> str:
        """Return a compact one-line representation of a graph message."""
        role = type(message).__name__.replace("Message", "").lower()
        content = getattr(message, "content", "")
        if isinstance(content, list):
            try:
                content = json.dumps(content)
            except Exception:
                content = str(content)

        text = str(content).replace("\n", " ").strip()
        if len(text) > max_len:
            text = text[: max_len - 3] + "..."
        if getattr(message, "tool_calls", None):
            names = [call.get("name", "tool") for call in message.tool_calls]
            return f"{role}: tool_calls -> {names}"
        if isinstance(message, ToolMessage):
            return f"{role}({message.name}): {text}"
        return f"{role}: {text}"

    def _print_state_snapshot(self, title: str) -> None:
        """Print the current LangGraph state for this conversation session."""
        snapshot = self.app.get_state(self.config)
        values = snapshot.values or {}
        messages: List[BaseMessage] = values.get("messages", [])
        print(f"\n=== {title} | state snapshot ===")
        print(f"messages: {len(messages)} total")
        for index, message in enumerate(
            messages[-5:], start=max(0, len(messages) - 5) + 1
        ):
            print(f"  {index:>3}: {self._short(message)}")
        if snapshot.next:
            print(f"next nodes: {list(snapshot.next)}")
        if snapshot.tasks:
            print(f"queued tasks: {snapshot.tasks}")
        print("memory: in-memory checkpoint present for this thread")

    def ask(self, question: str) -> str:
        """Run one turn and return the final assistant response."""
        final = self._run_with_tracing(
            {"messages": [HumanMessage(content=question)]}
        )
        messages = final.get("messages") if final else None
        if not messages:
            raise RuntimeError("Agent completed without an assistant response.")
        return messages[-1].content

    def _build_app(self, model_name: str):
        tools = [internet_search, calculator]
        llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            max_tokens=800,
        ).bind_tools(tools, tool_choice="auto")

        def llm_node(state: AgentState) -> AgentState:
            ai_message = llm.invoke(state["messages"])
            return {"messages": [ai_message]}

        graph = StateGraph(AgentState)
        graph.add_node("llm", llm_node)
        graph.add_node("tools", ToolNode(tools=tools, handle_tool_errors=True))
        graph.add_edge(START, "llm")
        graph.add_conditional_edges(
            "llm",
            self._route,
            {"tools": "tools", END: END},
        )
        graph.add_edge("tools", "llm")
        return graph.compile(checkpointer=MemorySaver())

    def _run_with_tracing(self, input_state: AgentState):
        if self.debug:
            print("\n=== Question | execution trace ===")

        final = None
        for event in self.app.stream(
            input_state,
            config=self.config,
            stream_mode="updates",
        ):
            for node, update in event.items():
                if self.debug:
                    print(f"[enter {node}] updated: {list(update.keys())}")
                    messages = update.get("messages") or []
                    if messages:
                        print(f"  {self._short(messages[-1])}")
                    print(f"[leave {node}]")
                final = update

        return final


def main():
    load_dotenv()
    try:
        load_api_keys()
    except (EOFError, KeyboardInterrupt):
        print("\nSetup cancelled.")
        return

    model_name = os.getenv("AI_MODEL")
    if not model_name:
        print("AI_MODEL is required. Set it in .env.")
        return

    debug = os.getenv("DEBUG", "false").lower() in ("true", "1", "t")
    agent = ConversationAgent(model_name, debug=debug)
    print("Starting interactive conversation agent...")
    print("Type 'exit' or 'quit' to terminate the conversation.")
    while True:
        try:
            question = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nConversation ended due to user interruption.")
            break
        if not question.strip():
            continue
        if question.lower().strip() in ("exit", "quit"):
            break
        try:
            response = agent.ask(question)
            print("\nResponse:\n", response)
        except KeyboardInterrupt:
            print("\nRequest cancelled.")
        except Exception:
            print("\nUnable to complete that request. Please try again.")
            if debug:
                traceback.print_exc()

if __name__ == "__main__":
    main()
