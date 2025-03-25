import json
import random
from typing import Iterable, Optional
import uuid
from pathlib import Path
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

from astropath.bot import Bot

load_dotenv()

SPINNER_MESSAGES = [
    "Checking the ancient texts...",
    "Consulting the Emperor's Tarot...",
    "Piercing the Immaterium...",
    "Channeling the Astronomican...",
    "Seeking guidance from the Warp...",
    "Decrypting sacred data-scrolls...",
    "Navigating the tides of fate...",
    "Interpreting the echoes of the void...",
    "Whispering with the spirits of the Astronomicon...",
    "Scrying the Emperor's divine will...",
    "Attuning to the psychic resonance...",
    "Braving the warp's unpredictable currents...",
    "Filtering out the whispers of the void...",
    "Reaching beyond the veil of reality...",
    "Aligning with the great celestial patterns...",
    "Traversing the empyrean corridors...",
    "Calling upon the echoes of lost souls...",
    "Binding the roiling visions into clarity...",
    "Shielding against the horrors of the Warp...",
    "Consulting the Librarium of the Adeptus Astra Telepathica...",
]


def fix_llm_output(text: str) -> str:
    pairs = [
        ("\\(", "$"),
        ("\\)", "$"),
        ("\\[", "$$"),
        ("\\]", "$$"),
    ]
    for old, new in pairs:
        text = text.replace(old, new)
    return text


class ConversationManager:
    def __init__(self, history_dir: str | Path = "chat_histories"):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(exist_ok=True)
        self.index_file = self.history_dir / "index.json"
        self.bot = Bot()
        self.conversations = self.load_conversations()
        self.initialize_session_state()

    def initialize_session_state(self):
        if "current_conversation_id" not in st.session_state:
            if self.conversations:
                st.session_state.current_conversation_id = self.conversations[0]["id"]
            else:
                new_id = self.create_new_conversation("New Chat")
                st.session_state.current_conversation_id = new_id

        if "messages" not in st.session_state:
            st.session_state.messages = self.load_history(
                st.session_state.current_conversation_id
            )

    def load_conversations(self) -> list[dict]:
        if self.index_file.exists():
            with open(self.index_file, "r") as file:
                return json.load(file)
        return []

    def save_conversations(self):
        with open(self.index_file, "w") as file:
            json.dump(self.conversations, file)

    def create_new_conversation(self) -> str:
        new_id = str(uuid.uuid4())
        self.conversations.append(
            {
                "id": new_id,
                "name": "New Chat",
                "created_at": datetime.now().isoformat(),
            }
        )
        self.save_conversations()
        (self.history_dir / f"{new_id}.jsonl").touch()
        return new_id

    def load_history(self, conversation_id: str) -> list[dict]:
        history_file = self.history_dir / f"{conversation_id}.jsonl"
        if history_file.exists():
            with open(history_file, "r") as file:
                return [json.loads(line) for line in file]
        return []

    def generate_new_name(self, conversation_id: str, messages: list[dict]) -> str:
        # Get the current conversation
        current_conv = next(
            (conv for conv in self.conversations if conv["id"] == conversation_id), None
        )

        # Generate title if it's the first interaction
        if current_conv and len(messages) == 2:
            try:
                # Create a title generation prompt based on the first user message
                first_user_message = next(
                    msg["content"] for msg in messages if msg["role"] == "user"
                )
                title_prompt = [
                    {
                        "role": "system",
                        "content": "Generate a very short title (max 10 words) for this conversation based on the following query. Respond only with the title.",
                    },
                    {"role": "user", "content": first_user_message},
                ]

                # Get title from the bot
                new_name = self.bot.chat(title_prompt).strip()

                # Clean up any quotes or special characters
                new_name = new_name.strip('"').strip("'").split("\n")[0]

                self.rename_conversation(conversation_id, new_name)
            except Exception:
                # Fallback to default name
                pass

    def save_history(self, conversation_id: str, messages: list[dict]):
        history_file = self.history_dir / f"{conversation_id}.jsonl"
        if len(messages) == 2:
            self.generate_new_name(conversation_id, messages)

        with open(history_file, "w") as file:
            for msg in messages:
                file.write(json.dumps(msg) + "\n")

    def delete_conversation(self, conversation_id: str):
        self.conversations = [
            conv for conv in self.conversations if conv["id"] != conversation_id
        ]
        self.save_conversations()
        (self.history_dir / f"{conversation_id}.jsonl").unlink(missing_ok=True)

    def rename_conversation(self, conversation_id: str, new_name: str):
        for conv in self.conversations:
            if conv["id"] == conversation_id:
                conv["name"] = new_name
                break
        self.save_conversations()

    def get_conversation_date(self, conversation_id: str) -> Optional[str]:
        for conv in self.conversations:
            if conv["id"] == conversation_id:
                return datetime.fromisoformat(conv["created_at"]).strftime(
                    "%H:%M, %d %B, %Y"
                )
        return None

    def stream_response(self, messages: list[dict]) -> Iterable[str]:
        """Stream response from the bot"""
        return self.bot.chat_stream(messages)


@st.dialog(title="Error", width="small")
def error_popup(message: str) -> None:
    st.error(message)


def main():
    st.set_page_config(page_title="Astropath", page_icon="🌌", layout="wide")
    st.title("Astropath 🌌")

    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .stChatMessage {
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        [data-testid="stChatMessage"] > div:first-child {
            margin-right: 1rem;
        }
        .sidebar .conversation-item {
            padding: 0.5rem;
            margin: 0.25rem 0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .sidebar .conversation-item:hover {
            background: #f0f2f6;
        }
        .sidebar .active-chat {
            background: #e3f2fd !important;
            font-weight: 500;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    conv_manager = ConversationManager()

    with st.sidebar:
        st.header("Conversations")

        # Conversations list
        for conv in conv_manager.conversations:
            is_active = conv["id"] == st.session_state.current_conversation_id
            col1, col2 = st.columns([6, 1])

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
                    ":material/delete:",
                    key=f"delete_{conv['id']}",
                    help="Delete conversation",
                ):
                    if len(conv_manager.conversations) > 1:
                        conv_manager.delete_conversation(conv["id"])
                        if is_active:
                            st.session_state.current_conversation_id = (
                                conv_manager.conversations[0]["id"]
                            )
                        st.rerun()
                    else:
                        error_popup("Cannot delete the only conversation")

        # New Chat button
        if st.button(":material/add:", use_container_width=True):
            new_id = conv_manager.create_new_conversation()
            st.session_state.current_conversation_id = new_id
            st.rerun()

    # Main chat interface
    current_conv_id = st.session_state.current_conversation_id

    # Load messages for current conversation
    messages = conv_manager.load_history(current_conv_id)
    if created_date := conv_manager.get_conversation_date(current_conv_id):
        st.caption(f"Created at: {created_date}")

    # Display messages
    for msg in messages:
        avatar = "🌌" if msg["role"] == "assistant" else None
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask Astropath..."):
        # Add user message
        processed_prompt = fix_llm_output(prompt)
        messages.append({"role": "user", "content": processed_prompt})

        # Display user message
        with st.chat_message("user"):
            st.write(processed_prompt)

        # Generate bot response
        with st.chat_message("assistant", avatar="🌌"):
            full_response = []
            placeholder = st.empty()

            with placeholder.container():
                st.spinner(text=random.choice(SPINNER_MESSAGES), show_time=True)

            for chunk in conv_manager.stream_response(messages):
                full_response.append(chunk.content)
                placeholder.markdown(fix_llm_output("".join(full_response)))
            placeholder.markdown(fix_llm_output("".join(full_response)))

        # Save messages
        messages.append(
            {"role": "assistant", "content": fix_llm_output("".join(full_response))}
        )
        conv_manager.save_history(current_conv_id, messages)
        st.rerun()


if __name__ == "__main__":
    main()
