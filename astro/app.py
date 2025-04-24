import os

from dotenv import load_dotenv

import instructor
import openai
from pydantic import Field
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from atomic_agents.lib.components.agent_memory import AgentMemory
from atomic_agents.agents.base_agent import (
    BaseAgent,
    BaseAgentConfig,
    BaseAgentOutputSchema,
)

load_dotenv()
API_KEY = os.getenv("OPEN_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

# Initialize a Rich Console for pretty console outputs
console = Console()

# Memory setup
memory = AgentMemory()

# Initialize memory with an initial message from the assistant
initial_message = BaseAgentOutputSchema(chat_message="Hey, how can I help you today?")
memory.add_message("assistant", initial_message)

# OpenAI client setup using the Instructor library
# Note, you can also set up a client using any other LLM provider, such as Anthropic, Cohere, etc.
# See the Instructor library for more information: https://github.com/instructor-ai/instructor
client = instructor.from_openai(openai.OpenAI(api_key=API_KEY))

# Instead of the default system prompt, we can set a custom system prompt
system_prompt_generator = SystemPromptGenerator(
    background=[
        "You are Astro.",
        "You are a radio interferometry and astronomy software and data science assistant.",
        "You are South African."
        "You are a member of the centre for *Radio Astronomy Techniques and Technologies* (RATT) group."
        "They are based in the Physics & Electronics Department at Rhodes University, Makhanda, South Africa.",
        "Your primary language will be english, but if the user responds in another, do so accordingly.",
        "You will only use metric standard units.",
        "You will be helping with software-related problems and running software for users.",
        "You have access to the `SimmsAgent` which allows for creating empty CASA measurement sets using the `simms` tool.",
    ],
    steps=[
        "Understand the user's input and provide a relevant response.",
        "Respond to the user.",
        "If the user wants a new measurement set, request to use the `SimmsAgent`.",
        "Once requested, you will be provided the input schema to fill out to run it.",
        "It will require a description of the dataset based on the `simms` help text context."
        "The agent will run and provide the result of the action.",
        "If successful, then return the path to user where the measurement set is.",
        "If not successful, notify the user of this and ask for a way forward.",
    ],
    output_instructions=[
        "Provide helpful and relevant information to assist the user.",
        "Be friendly and respectful in all interactions.",
        "Summarize the response instead of returning a lot of overwhelming information.",
    ],
)
console.print(
    Panel(
        system_prompt_generator.generate_prompt(),
        width=console.width,
        style="bold cyan",
    ),
    style="bold cyan",
)


class AstroOutputSchema(BaseAgentOutputSchema):
    """Output schema for the Astro chat-bot. Contains the message to display
    and whether to do a tool call or not."""

    call_tool: bool = Field(False, description="Request to run the Simms Agent")


# Agent setup with specified configuration
agent = BaseAgent(
    config=BaseAgentConfig(
        client=client,
        model="gpt-4o-mini",
        system_prompt_generator=system_prompt_generator,
        memory=memory,
        output_schema=AstroOutputSchema,
    )
)

from astro.simms import SimmsAgent, SimmsHelpText, InitialInputSchema, OutputSchema

agent.register_context_provider("simms help text", SimmsHelpText("Simms Help Text"))

# Display the initial message from the assistant
console.print("[bold green]Astro:[/bold green]", end=" ")
console.print(Text(initial_message.chat_message))

# Start an infinite loop to handle user inputs and agent responses
while True:
    # Prompt the user for input with a styled prompt
    user_input = console.input("[bold blue]You:[/bold blue] ")
    # Check if the user wants to exit the chat
    if user_input.lower() in ["/exit", "/quit"]:
        console.print("Exiting chat...")
        break

    # Process the user's input through the agent and get the response and display it
    response = agent.run(agent.input_schema(chat_message=user_input))

    if isinstance(response, AstroOutputSchema) and response.call_tool:

        console.print("[bold green]Astro:[/bold green] Calling `simms`")
        agent.output_schema = InitialInputSchema
        response = agent.run()
        simms = SimmsAgent()
        console.print(
            f"[bold green]Astro:[/bold green] Running `simms` as:\n\n{response}"
        )
        response = simms.run(response)
        agent.input_schema = OutputSchema
        agent.output_schema = AstroOutputSchema
        console.print(
            f"[bold green]System:[/bold green] Run successful: {response.success}"
        )
        agent.memory.add_message("system", response)
        response = agent.run()

    agent_message = Text(response.chat_message)
    console.print("[bold green]Astro:[/bold green]", end=" ")
    console.print(agent_message)
