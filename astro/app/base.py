import sys
from typing import Literal

import click
import streamlit as st
from pydantic import ValidationError
from streamlit import runtime
from streamlit.web import cli as stcli

from astro.agents.chat import AstroChatAgent
from astro.app.config import DisplayTheme, StreamlitConfig
from astro.llms.base import ModelName, ModelProvider
from astro.loggings.base import LogLevel, get_logger
from astro.paths import get_module_dir

# Load logger
_logger = get_logger(__file__)


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

    # Initialize session state for agent and manager
    if "chat_agent" not in st.session_state:
        st.session_state.chat_agent = None
    if "conversation_manager" not in st.session_state:
        st.session_state.conversation_manager = ConversationManager()

    manager = st.session_state.conversation_manager

    # Sidebar for model and conversation configuration
    with st.sidebar:
        st.header("Configuration")

        # Provider selection
        provider_options = [provider.value for provider in ModelProvider]
        selected_provider = st.selectbox(
            "Select Provider:",
            provider_options,
            index=provider_options.index(st.session_state.current_provider),  # pyright: ignore[reportArgumentType]
        )
        model_provider = ModelProvider(selected_provider)
        model_options = model_provider.models

        selected_model = st.selectbox("Select Model:", model_options)

        # Initialize or update chat agent if provider changed
        if (
            st.session_state.current_provider != selected_provider
            or st.session_state.chat_agent is None
        ):
            try:
                st.session_state.chat_agent = AstroChatAgent(
                    identifier=selected_model, provider=selected_provider
                )
                st.session_state.current_provider = selected_provider
                st.success(f"Initialized {selected_provider} with {selected_model}")
            except Exception as e:
                st.error(f"Failed to initialize model: {e}")
                st.session_state.chat_agent = None

        st.header("Conversations")

        # New Chat button
        if st.button("New Chat", use_container_width=True, type="primary"):
            st.session_state.conversation_counter += 1
            new_id = str(st.session_state.conversation_counter)
            st.session_state.conversations.append(
                {"id": new_id, "name": f"Chat {new_id}", "messages": []}
            )
            st.session_state.current_conversation_id = new_id
            st.rerun()

        # Conversations list
        for conv in st.session_state.conversations:
            is_active = conv["id"] == st.session_state.current_conversation_id
            col1, col2 = st.columns([5, 1])

            with col1:
                display_name = f"{conv['name']}"
                if st.button(
                    display_name,
                    key=f"conv_{conv['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    if not is_active:
                        st.session_state.current_conversation_id = conv["id"]
                        st.rerun()

            with col2:
                if st.button(
                    "🗑️",
                    key=f"delete_{conv['id']}",
                    help="Delete conversation",
                    use_container_width=True,
                ):
                    if len(st.session_state.conversations) > 1:
                        st.session_state.conversations = [
                            c
                            for c in st.session_state.conversations
                            if c["id"] != conv["id"]
                        ]
                        if is_active:
                            st.session_state.current_conversation_id = (
                                st.session_state.conversations[0]["id"]
                            )
                        st.rerun()
                    else:
                        st.warning("Cannot delete the only conversation.")

        # Clear chat button
        if st.button("Clear Current Chat", use_container_width=True):
            messages = get_current_conversation_messages()
            messages.clear()
            st.rerun()

    # Main chat interface
    if st.session_state.chat_agent is None:
        st.warning("Please configure a model in the sidebar to start chatting.")
        return

    messages = get_current_conversation_messages()

    # Display chat messages
    for message in messages:
        avatar = "🌌" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask Astro anything..."):
        # Add user message to chat history
        messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)

        # Generate assistant response
        with st.chat_message("assistant", avatar="🌌"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chat_agent.act(prompt)
                    st.write(response)

                    # Add assistant response to chat history
                    messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    _logger.error(f"Chat agent error: {e}", exc_info=True)
        st.rerun()


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
