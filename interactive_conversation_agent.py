import os
import traceback
from getpass import getpass
from dotenv import load_dotenv
from agent_tools import internet_search, calculator
from conversation_agent import ConversationAgent


def load_api_keys(debug=False):
    serp_api_key = os.getenv("SERPAPI_API_KEY")
    open_api_key = os.getenv("OPENAI_API_KEY")
    if not open_api_key:
        print("⚠️ OPENAI_API_KEY not found. Enter it below.")
        open_api_key = getpass("Enter your OPENAI_API_KEY: ").strip()
        os.environ["OPENAI_API_KEY"] = open_api_key
    else:
        if debug:
            print("✅ OPENAI_API_KEY loaded successfully.")
    if not serp_api_key:
        print("⚠️ SERPAPI_API_KEY not found. Enter it below.")
        serp_api_key = getpass("Enter your SERPAPI_API_KEY: ").strip()
        os.environ["SERPAPI_API_KEY"] = serp_api_key
    else:
        if debug:
            print("✅ SERPAPI_API_KEY loaded successfully.")
    if debug:
        print("✅ API keys loaded successfully!")


def main():
    load_dotenv()
    debug = os.getenv("DEBUG", "false").lower() in ("true", "1", "t")
    try:
        load_api_keys(debug=debug)
    except (EOFError, KeyboardInterrupt):
        print("\nSetup cancelled.")
        return
    model_name = os.getenv("AI_MODEL")
    if not model_name:
        print("AI_MODEL is required. Set it in .env.")
        return
    tools = (internet_search, calculator)
    agent = ConversationAgent(model_name, tools=tools, debug=debug)
    print("Starting interactive conversation agent...")
    print("Type 'exit' or 'quit' to terminate the conversation.")
    while True:
        try:
            question = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nConversation ended due to user interruption.")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
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
