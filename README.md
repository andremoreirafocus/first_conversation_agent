# First Conversation Agent

A small LangGraph learning project with an interactive terminal agent and its original tutorial script.

Use `interactive_conversation_agent.py` for normal use. `first_agent.py` is retained as a **legacy** walkthrough that led to the CLI version.

## Project files

| File | Status | Purpose |
| --- | --- | --- |
| `interactive_conversation_agent.py` | Active | The one-user terminal CLI: environment setup, credential prompts, and terminal input/output. |
| `conversation_agent.py` | Active | Reusable `ConversationAgent` implementation: LangGraph workflow, memory, and tracing. It receives its tools through its constructor. |
| `agent_tools.py` | Active | Tool definitions and `DEFAULT_TOOLS`, currently containing web search and calculator capabilities. |
| `first_agent.py` | Legacy | A sequential tutorial: basic model call, manual tool loops, then a fixed-prompt LangGraph example. It runs API calls immediately. |
| `requirements.txt` | Active | Python dependencies. |
| `.gitignore` | Active | Excludes `.env`, virtual environments, and common editor/cache files. |

## Prerequisites

- Python 3.10 or later.
- An OpenAI API key with access to the model selected by `AI_MODEL`.
- A [SerpAPI](https://serpapi.com/) API key for Google search.

Both scripts make real external requests. OpenAI and SerpAPI usage may incur charges.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```dotenv
OPENAI_API_KEY=your_openai_api_key
SERPAPI_API_KEY=your_serpapi_api_key
AI_MODEL=gpt-4o
# Optional: print LangGraph node and tool traces.
DEBUG=true
```

`.env` is ignored by Git. `AI_MODEL` is required by the interactive CLI and selects the OpenAI model used for each turn. Choose a model your account can access that supports tool calling; `gpt-4o` is the current example. If either API key is absent, the interactive script asks for it without echoing the value in a terminal.

## Interactive CLI

Run the active CLI:

```bash
python3 interactive_conversation_agent.py
```

Ask questions at the `Your question:` prompt. The same conversation remains available while the program is running, so follow-up questions can refer to earlier answers.

Type `exit` or `quit` to end the session. Empty input is ignored. Pressing Ctrl-C or sending EOF exits or cancels safely.

With `DEBUG=true`, the CLI prints each graph node update. This is useful for seeing whether the model answered directly or invoked a tool.

### Agent flow

```text
user input → llm ── tool call? ──yes──→ tools ──→ llm
                    │
                    no
                    ↓
                   answer
```

The graph uses a single `messages` state field. LangGraph's `add_messages` reducer appends user messages, model responses, and tool outputs. `MemorySaver` keeps that state under one thread ID for the life of the current CLI process.

Tool calls execute with `max_concurrency=1`. This is intentional because the installed SerpAPI wrapper temporarily redirects process stdout; serial execution avoids output-stream conflicts when the model requests more than one search.

### Tools

- `internet_search(query)` searches Google through SerpAPI using US/English settings and returns parsed result text.
- `calculator(expression)` evaluates a numeric expression with `numexpr`; `pi` and `e` are available as constants.

The model selected by `AI_MODEL` chooses tools automatically. Tool and API errors are reported without ending the normal conversation loop; enable debug mode to print their traceback.

## Legacy tutorial: `first_agent.py`

`first_agent.py` is preserved for reference, not as the normal entry point. It runs every example at module load and does not provide a user conversation prompt.

It demonstrates these stages:

1. A direct `gpt-5-mini` question and response-usage printout.
2. A manual tool-calling loop that forces tool use with `tool_choice="any"`.
3. A revised manual loop with `tool_choice="auto"` and a maximum-step fallback.
4. A LangGraph workflow with tracing and scripted New York weather follow-up prompts.

Because it repeats imports, tool definitions, and `run_once()` while teaching each stage, it is intentionally more verbose and less suitable for extension than the interactive CLI. Its model names are hard-coded and do not use `AI_MODEL`. It also makes API calls as soon as it is executed or imported.

Run it only when you want to study that progression:

```bash
python3 first_agent.py
```

## Current limitations

- Conversation memory is in memory only; restarting the CLI clears it.
- Search responses are text snippets, not structured and cited factual data.
- The project has no automated tests.
- Depending on the installed LangChain/LangGraph versions, startup can show a dependency deprecation warning. Updating the compatible LangGraph and LangChain packages is the appropriate long-term remedy.

## Verification

Both Python scripts compile with the repository virtual environment. The documented live flows were not run during documentation updates because they require credentials and make external, billable requests.
