import json
from typing import Annotated, List, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


class ConversationAgent:
    """One stateful, tool-enabled conversation session."""

    def __init__(
        self,
        model_name: str,
        tools: Sequence[BaseTool],
        thread_id: str = "interactive-session",
        debug: bool = False,
    ):
        if not tools:
            raise ValueError("ConversationAgent requires at least one tool.")
        self.tools = tuple(tools)
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
        llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            max_tokens=800,
        ).bind_tools(self.tools, tool_choice="auto")

        def llm_node(state: AgentState) -> AgentState:
            ai_message = llm.invoke(state["messages"])
            return {"messages": [ai_message]}

        graph = StateGraph(AgentState)
        graph.add_node("llm", llm_node)
        graph.add_node(
            "tools",
            ToolNode(tools=self.tools, handle_tool_errors=True),
        )
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
