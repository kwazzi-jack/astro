from typing import List


from pydantic import BaseModel, Field

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver


class AstroState(BaseModel):
    messages: list[AnyMessage] = Field(default_factory=list)

# Initialize the LangChain chat model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# Define the function that calls the LLM
def call_model(state: AstroState):
    """Invokes the LLM with the current state's messages and returns the updated message list."""
    print(f"--- Calling LLM with {len(state.messages)} messages ---")
    response = llm.invoke(state.messages)
    print(f"--- LLM Response received ---")
    # Return a dictionary mapping the state field to its new, complete value
    # LangGraph will merge this back into the Pydantic state object.
    return {"messages": state.messages + [response]}


# --- Build the Graph ---
# Instantiate StateGraph with the Pydantic model type
graph_builder = StateGraph(AgentState)

# Add the node that calls the model
graph_builder.add_node("llm", call_model)

# Set the entry point and finish point
graph_builder.set_entry_point("llm")
graph_builder.set_finish_point("llm")

# Define the checkpointer for persistence
# SqliteSaver works fine with Pydantic models (as they are serializable)
memory = SqliteSaver.("conversations.sqlite")

# Compile the graph with the checkpointer
app = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["llm"],  # Optional
    interrupt_after=["llm"],  # Optional
)


# --- Helper function for streaming ---
# This remains largely the same, but ensures it works with the updated graph structure
async def stream_response(messages: List[AnyMessage], config: dict):
    """Streams the final response from the LangGraph app."""
    # The input to app.astream_events is still a dictionary matching the state structure
    async for event in app.astream_events(
        {"messages": messages}, config=config, version="v2"
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                yield chunk.content
        # Example of accessing state *after* the node runs, if needed:
        # elif kind == "on_chain_end":
        #     if event["name"] == "llm": # Check if it's the end of our specific node
        #         # The output of the node is in event["data"]["output"]
        #         # The full state *after* the update is implicitly handled by the checkpointer
        #         # If you needed the full state explicitly here, you might call app.get_state(config)
        #         pass
