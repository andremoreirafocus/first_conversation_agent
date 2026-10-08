import os

import streamlit as st
from dotenv import load_dotenv

from agent_tools import calculator, internet_search
from conversation_agent import ConversationAgent


def is_enabled(value: str | None) -> bool:
    return (value or "").lower() in ("true", "1", "t")


def get_agent(model_name: str, debug: bool) -> ConversationAgent:
    if "conversation_agent" not in st.session_state:
        tools = (internet_search, calculator)
        st.session_state.conversation_agent = ConversationAgent(
            model_name,
            tools=tools,
            debug=debug,
        )
    return st.session_state.conversation_agent


def start_new_conversation() -> None:
    st.session_state.pop("conversation_agent", None)
    st.session_state.pop("chat_history", None)


def main() -> None:
    st.set_page_config(page_title="Conversation Agent", page_icon="💬")
    load_dotenv()

    model_name = os.getenv("AI_MODEL")
    missing_variables = [
        name
        for name in ("OPENAI_API_KEY", "SERPAPI_API_KEY", "AI_MODEL")
        if not os.getenv(name)
    ]
    if missing_variables:
        st.error(
            "Missing configuration: " + ", ".join(missing_variables) + ". "
            "Set these values in .env or your deployment environment."
        )
        st.stop()

    debug = is_enabled(os.getenv("DEBUG"))
    with st.sidebar:
        st.caption(f"Model: {model_name}")
        st.caption("Tools: internet search, calculator")
        if st.button("New conversation"):
            start_new_conversation()
            st.rerun()

    st.title("Conversation Agent")
    st.caption("Ask a question. The agent can search the web and calculate.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = get_agent(model_name, debug).ask(prompt)
                except Exception as error:
                    st.error("Unable to complete that request. Please try again.")
                    if debug:
                        st.exception(error)
                else:
                    st.markdown(response)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )


if __name__ == "__main__":
    main()
