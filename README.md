# First Conversation Agent

An educational Python script that builds an OpenAI-powered agent in stages. It starts with a single model call, then shows how to let a model call tools manually, and finally implements the same idea as a stateful LangGraph workflow.

The example task is deliberately small: find New York City's current temperature with Google/SerpAPI, then use a calculator to perform a follow-up computation. The final section preserves conversation context, so a later prompt such as “square that temperature” can refer to a value found in an earlier turn.

## What is in the repository

| File | Purpose |
| --- | --- |
| `first_agent.py` | All examples, tools, LangGraph workflow, tracing helpers, and demo runs. |
| `requirements.txt` | Python dependencies used by the script. |
| `.gitignore` | Excludes credentials in `.env`, the virtual environment, and editor/cache files. |

There are no tests, web UI, server, database, or persistent storage. `MemorySaver` retains checkpoints only in the running Python process.

## Prerequisites

- Python 3.10 or later (the type hints use Python 3.10 union syntax).
- An OpenAI API key with access to the configured models (`gpt-5-mini` and `gpt-4o`).
- A [SerpAPI](https://serpapi.com/) API key, used by the Google search tool.

The sample makes real API requests. Running the complete script incurs OpenAI and SerpAPI usage and can produce changing answers because the question asks for the current weather.

## Setup and run

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```dotenv
OPENAI_API_KEY=your_openai_api_key
SERPAPI_API_KEY=your_serpapi_api_key
```

Run the demonstration:

```bash
python first_agent.py
```

`first_agent.py` runs every example from top to bottom. Expect console output from the initial model call, two one-shot tool-agent runs, per-node LangGraph traces, final answers, and memory snapshots.

## How the script works

### 1. Configuration and a model smoke test

`load_dotenv()` loads `.env`. The script then creates `ChatOpenAI(model="gpt-5-mini")`, asks “What are AI agents?”, and prints both the answer and its usage metadata. This verifies the OpenAI connection before the agent examples run.

### 2. Tools

The script declares two LangChain tools:

- `internet_search(query)` sends a Google search through `SerpAPIWrapper`, with US/English result settings. It returns SerpAPI's text result.
- `calculator(expression)` evaluates a one-line numerical expression through `numexpr`; it exposes `pi` and `e` as constants.

Both tools are passed to the model through `bind_tools()`. The model chooses a tool by emitting a tool call; application code then invokes the matching tool and appends a `ToolMessage` containing the result.

### 3. Manual agent loop — first version

The first `run_once()` implementation uses `gpt-4o`, permits at most four model iterations, and is bound with `tool_choice="any"`. It prints each requested tool and its result. This is a minimal illustration of the tool-calling protocol, but forcing a tool call makes it awkward for a model to finish with a normal answer.

### 4. Manual agent loop — improved version

The later `run_once()` definition replaces the first one. It uses `tool_choice="auto"`, allows up to eight iterations, and returns as soon as the model emits an ordinary response with no tool calls. If it reaches the iteration limit, it adds a final “Finish now” message and asks the model to produce the requested answer format.

The two definitions have the same name, but this is intentional in the teaching sequence: the second one overwrites the first only after the first demo has already executed.

### 5. Stateful LangGraph agent

The last section promotes the tool loop to a graph:

```text
START → llm ── tool call? ──yes──→ tools ──→ llm
              │
              no
              ↓
             END
```

Key pieces:

- `AgentState` has one `messages` field. `add_messages` appends each graph update instead of replacing prior conversation history.
- `llm_node` invokes the tool-enabled `gpt-4o` model with the accumulated messages.
- `ToolNode` runs any requested tools and adds their results to state.
- `route` inspects the final message: tool calls go to `tools`; an ordinary assistant reply ends the graph.
- `MemorySaver` checkpoints state by `thread_id`.

`run_with_tracing()` streams node-level updates and then reads the saved state. The two prompts that use `nyc-weather-session` are a single conversation: turn two receives turn one's messages and tool result. `nyc-weather-session-branch` is a separate, empty thread despite its label, so it does **not** inherit the main thread's weather result.

## Important behavior and limitations

- **Provide keys through `.env` or environment variables.** The interactive fallback only saves typed keys in Python variables. `ChatOpenAI` and `internet_search` read from `os.environ`, so the fallback does not reliably configure them. A `.env` file is the supported path for this script.
- **The script is a tutorial, not a reusable module.** Imports, tool definitions, and `run_once()` are repeated as each learning stage is introduced. Importing the module also runs all API-backed demos because no code is protected by an `if __name__ == "__main__":` block.
- **No durable memory.** Restarting Python clears `MemorySaver`; use a durable checkpointer for memory across processes.
- **Results may be incomplete or ambiguous.** The search wrapper returns formatted search results, not a structured weather API response. The model must interpret the temperature itself.
- **A tool-loop limit is not a correctness guarantee.** `max_steps`/`recursion_limit` prevent unbounded tool use, but the model can still select an unsuitable tool or fail to extract a usable value.
- **Treat tool inputs as untrusted in a real application.** Add validation, timeouts, observability, rate limits, error handling, and a narrowly scoped search/calculation policy before exposing this pattern to users.

## Useful customizations

- Change the configured OpenAI model in the three `ChatOpenAI(...)` declarations.
- Replace SerpAPI with a structured weather provider if exact temperature data matters.
- Add a system prompt to set the agent's role, tool-use policy, and output format.
- Wrap the final graph invocation in a CLI, API, or UI rather than executing fixed demo prompts.
- Replace `MemorySaver` with a persistent LangGraph checkpointer for long-lived conversations.

## Verification performed

The script compiles successfully with the repository's virtual environment (Python 3.10.12). Its live examples were not executed during documentation because they require credentials and make billable external API calls.
