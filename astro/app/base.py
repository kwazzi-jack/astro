import asyncio
import sys
from typing import Literal

import click
import streamlit as st
from pydantic_ai import (
    AgentRunResultEvent,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from streamlit import runtime
from streamlit.web import cli as stcli

from astro.agents.base import create_astro_chat
from astro.app.config import DisplayTheme, StreamlitConfig
from astro.loggings.base import LogLevel, get_loggy
from astro.paths import get_module_dir

# Load logger
_logger = get_loggy(__file__)


def run_streamlit_app():
    # Setup configuration for page
    st.set_page_config(
        page_title="Astro",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for styling
    try:
        css_path = get_module_dir(__file__) / "streamlit.css"
        with open(css_path) as file:
            css = file.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("streamlit.css not found. App will use default styling.")

    st.title("Astro")

    # Initialize chat in session state to maintain conversation
    if "chat_stream" not in st.session_state or "messages" not in st.session_state:
        chat_stream, messages = create_astro_chat()
        st.session_state.chat_stream = chat_stream
        st.session_state.messages = messages

    chat_stream = st.session_state.chat_stream
    messages = st.session_state.messages

    # Display all messages from history
    for message in messages:
        if isinstance(message, ModelResponse) and len(message.parts) > 0:
            with st.chat_message("assistant", avatar="🌌"):
                for part in message.parts:
                    if isinstance(part, TextPart):
                        st.markdown(part.content)

    # Chat input
    if prompt := st.chat_input("Enter message..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response with streaming
        with st.chat_message("Astro", avatar="🌌"):
            response_placeholder = st.empty()
            full_response = ""

            # Run async chat_stream in sync context
            async def process_stream():
                nonlocal full_response
                async for event in chat_stream(prompt):
                    # Handle text delta events for streaming display
                    if isinstance(event, PartDeltaEvent):
                        if isinstance(event.delta, TextPartDelta):
                            full_response += event.delta.content_delta
                            response_placeholder.markdown(full_response)

                    if isinstance(event, PartStartEvent):
                        if isinstance(event.part, TextPart) and len(event.part.content):
                            full_response += event.part.content
                            response_placeholder.markdown(full_response)

                    # Handle final result
                    elif isinstance(event, AgentRunResultEvent):
                        if isinstance(event.result.response.parts[0], TextPart):
                            full_response = event.result.response.parts[0].content
                            response_placeholder.markdown(full_response)

            # Execute async function in sync context
            asyncio.run(process_stream())


@click.command()
@click.option(
    "--port",
    default=8501,
    show_default=True,
    type=int,
    help="Port to run the Streamlit server on (1024-65535)",
)
@click.option(
    "--host",
    default="localhost",
    show_default=True,
    help="Host address for the Streamlit server (use '0.0.0.0' for all interfaces)",
)
@click.option(
    "--no-browser/--browser",
    default=False,
    show_default=True,
    help="If set, do not open the browser automatically on startup",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    help="Logging level for the Streamlit app",
)
@click.option(
    "--debug-mode/--no-debug-mode",
    default=False,
    show_default=True,
    help="Enable Streamlit debug mode (shows extra logs and tracebacks)",
)
@click.option(
    "--theme",
    default="system",
    show_default=True,
    type=click.Choice(["light", "dark", "system"], case_sensitive=False),
    help="Streamlit theme: 'light', 'dark' or 'system'",
)
@click.option(
    "--run-on-save/--no-run-on-save",
    default=True,
    show_default=True,
    help="Automatically rerun the app when source code is saved",
)
def astro_start(
    port: int,
    host: str,
    no_browser: bool,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    debug_mode: bool,
    theme: str,
    run_on_save: bool,
):
    """Run the Astro Streamlit application."""

    _logger.info("Starting Astro Streamlit application")
    if runtime.exists():
        _logger.debug("Running Streamlit function")
        run_streamlit_app()
    else:
        # Path to streamlit app
        APP_PATH = get_module_dir(__file__) / "base.py"

        # Load configuration
        try:
            _logger.debug("Parsing command options")
            config = StreamlitConfig(
                port=port,
                host=host,
                no_browser=no_browser,
                log_level=LogLevel.from_str(log_level),
                debug_mode=debug_mode,
                theme=DisplayTheme.from_str(theme),
                run_on_save=run_on_save,
            )
        except ValueError as error:
            _logger.error(f"Configuration validation error: {error}")
            raise

        sys.argv = [
            "streamlit",
            "run",
            __file__,
            "--server.port",
            str(port),
            "--server.address",
            host,
            "--server.headless",
            "true" if no_browser else "false",
            "--server.runOnSave",
            str(run_on_save).lower(),
            "--logger.level",
            log_level.upper(),
            "--theme.base",
            theme,
        ]
        if no_browser:
            sys.argv += ["--browser.gatherUsageStats", "false"]
        if debug_mode:
            sys.argv += ["--global.developmentMode", "true"]
        sys.exit(stcli.main())


if __name__ == "__main__":
    astro_start()

# @st.dialog(title="Error", width="small")
# def error_popup(message: str):
#     """Error popup dialog for Streamlit"""
#     st.error(message)


# def main():
#     st.set_page_config(page_title="Astropath", page_icon="🌌", layout="wide")
#     st.title("Astropath 🌌")

#     # Custom CSS for styling
#     st.markdown(
#         """
#         <style>
#         .stChatMessage {
#             border-radius: 15px;
#             padding: 1.5rem;
#             margin: 1rem 0;
#             box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#         }
#         [data-testid="stChatMessage"] > div:first-child {
#             margin-right: 1rem;
#         }
#         .sidebar .conversation-item {
#             padding: 0.5rem;
#             margin: 0.25rem 0;
#             border-radius: 8px;
#             cursor: pointer;
#             transition: all 0.2s;
#         }
#         .sidebar .conversation-item:hover {
#             background: #f0f2f6;
#         }
#         .sidebar .active-chat {
#             background: #e3f2fd !important;
#             font-weight: 500;
#         }
#         </style>
#     """,
#         unsafe_allow_html=True,
#     )

#     conv_manager = ConversationManager()

#     with st.sidebar:
#         st.header("Conversations")

#         # Conversations list
#         for conv in conv_manager.conversations:
#             is_active = conv["id"] == st.session_state.current_conversation_id
#             col1, col2 = st.columns([6, 1])

#             with col1:
#                 display_name = f"{conv['name']}"
#                 if st.button(
#                     display_name,
#                     key=f"conv_{conv['id']}",
#                     use_container_width=True,
#                     type="primary" if is_active else "secondary",
#                 ):
#                     if not is_active:
#                         st.session_state.current_conversation_id = conv["id"]
#                         st.rerun()

#             with col2:
#                 if st.button(
#                     ":material/delete:",
#                     key=f"delete_{conv['id']}",
#                     help="Delete conversation",
#                 ):
#                     if len(conv_manager.conversations) > 1:
#                         conv_manager.delete_conversation(conv["id"])
#                         if is_active:
#                             st.session_state.current_conversation_id = (
#                                 conv_manager.conversations[0]["id"]
#                             )
#                         st.rerun()
#                     else:
#                         error_popup("Cannot delete the only conversation")

#         # New Chat button
#         if st.button(":material/add:", use_container_width=True):
#             new_id = conv_manager.create_new_conversation()
#             st.session_state.current_conversation_id = new_id
#             st.rerun()

#     # Main chat interface
#     current_conv_id = st.session_state.current_conversation_id

#     # Load messages for current conversation
#     messages = conv_manager.load_history(current_conv_id)
#     if created_date := conv_manager.get_conversation_date(current_conv_id):
#         st.caption(f"Created at: {created_date}")

#     # Display messages
#     for msg in messages:
#         avatar = "🌌" if msg["role"] == "assistant" else None
#         with st.chat_message(msg["role"], avatar=avatar):
#             st.write(msg["content"])

#     # Chat input
#     if prompt := st.chat_input("Ask Astropath..."):
#         # Add user message
#         processed_prompt = fix_llm_output(prompt)
#         messages.append({"role": "user", "content": processed_prompt})

#         # Display user message
#         with st.chat_message("user"):
#             st.write(processed_prompt)

#         # Generate bot response
#         with st.chat_message("assistant", avatar="🌌"):
#             full_response = []
#             placeholder = st.empty()

#             with placeholder.container():
#                 st.spinner(text=random.choice(SPINNER_MESSAGES), show_time=True)

#             for chunk in conv_manager.stream_response(messages):
#                 full_response.append(chunk.content)
#                 placeholder.markdown(fix_llm_output("".join(full_response)))
#             placeholder.markdown(fix_llm_output("".join(full_response)))

#         # Save messages
#         messages.append(
#             {"role": "assistant", "content": fix_llm_output("".join(full_response))}
#         )
#         conv_manager.save_history(current_conv_id, messages)
#         st.rerun()
