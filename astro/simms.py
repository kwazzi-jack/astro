from enum import Enum, auto
from pathlib import Path
from typing import Any, Literal

from atomic_agents.lib.components.agent_memory import AgentMemory
from atomic_agents.agents.base_agent import (
    BaseAgent,
    BaseAgentConfig,
    BaseIOSchema,
)
from atomic_agents.lib.components.system_prompt_generator import (
    SystemPromptGenerator,
    SystemPromptContextProviderBase,
)
from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig

import instructor
import openai
from pydantic import BaseModel, Field

from astro.system import BashOutputSchema, command
from astro.utilities.functions import format_function


class ParserAgent:
    class InputSchema(BaseIOSchema):
        """The input schema for the Parser agent that contains the description
        of the arguments wanted for function and the arguments names and default
        values to reference this against."""

        function_signature: str = Field(
            ...,
            description=(
                "The function signature of the function we are trying to parse too."
            ),
        )
        description: str = Field(
            ...,
            description=(
                "The detailed description of the arguments wanted "
                "for a given function to extract from."
            ),
        )

    class OutputSchema(BaseIOSchema):
        """The out schema for the Parser agent that contains the extracted
        arguments cross-referenced against their default values for the provided
        function."""

        arguments: dict[str, Any] = Field(
            None,
            description=(
                "Dictionary of arguments with their names as keys "
                "and the corresponding extracted value from the "
                "description."
            ),
        )
        error: str = Field(
            None, description=("Error raised by the parser agent during parsing.")
        )

    def __init__(self, function: callable):
        # Store
        self.function = function
        self.signature = format_function(function)

        # Setup agent
        self.system_prompt_generator = SystemPromptGenerator(
            background=[
                "You are a function argument parser agent.",
                "Your task is to extract function arguments from natural language descriptions.",
                "You will analyze a function signature with typed arguments and their default values.",
                "Then extract appropriate values for each argument from a user's description.",
            ],
            steps=[
                f"1. Analyze the function signature:\n\n{format_function(function)}",
                "2. Extract values for each argument from the user's description that follows this prompt.",
                "3. Apply these parsing rules for each argument:",
                "   - For boolean arguments: Extract true/false values from explicit mentions or implied intent",
                "   - For list arguments: Extract all relevant items as a properly formatted list",
                "   - For numeric arguments: Extract numbers including units if relevant",
                "   - For string arguments: Extract relevant text segments",
                "4. Handle missing values:",
                "   - If an argument has a default value, use it when the description doesn't specify a value",
                "   - If a required argument (no default) is missing, raise an error",
                "5. Handle type validation:",
                "   - If a provided value doesn't match the expected type, raise an error",
                "   - For union types (e.g., int | None), ensure the value matches at least one acceptable type",
            ],
            output_instructions=[
                "Format values according to their Python type (booleans as true/false, lists as arrays, etc.)",
                "If you raise an error, place it in the output error argument as a string with a description of the issue",
                "Also, if you raise an error, set the output arguments argument to null",
                "If the arguments are valid, put them in the output arguments argument",
                "Also, if the arguments are valid, then make the output error argument null",
                "If you say 'Invalid output format. Please ensure the output is structured correctly.' and agree, please explain in the error in-depth.",
            ],
        )
        self.memory = AgentMemory()
        self.config = BaseAgentConfig(
            client=instructor.from_openai(openai.OpenAI()),
            model="gpt-4o-mini",
            system_prompt_generator=self.system_prompt_generator,
            input_schema=self.InputSchema,
            output_schema=self.OutputSchema,
            memory=self.memory,
            temperature=0.0,
            max_tokens=1024,
        )
        self.agent = BaseAgent(self.config)
        self.last_input: ParserAgent.InputSchema | None = None
        self.last_output: ParserAgent.OutputSchema | None = None

    def run(self, description: str) -> "ParserAgent.OutputSchema":
        self.last_input = self.InputSchema(
            function_signature=self.signature, description=description
        )
        self.last_output = self.agent.run(self.last_input)
        self.agent.get_response
        return self.last_output


def simms_command(*args: str) -> BashOutputSchema:
    return command("simms", *args)


class SimmsHelpText(SystemPromptContextProviderBase):
    def __init__(self, title):
        super().__init__(title)

    def get_info(self) -> str:
        # FIX: simms cli makes life hard for error handling

        output = simms_command("--help")

        if output.stderr or output.return_code:
            return (
                "Error occured running `simms`"
                + (
                    f" (return-code={output.return_code}):\n\n"
                    if output.return_code
                    else ":\n\n"
                )
                + f'"{output.stderr}"'
            )
        else:
            return f'CLI help text from `simms`:\n\n"{output.stdout}"'


class SimmsToolInputSchema(BaseIOSchema):
    """Tool for running `simms` command to create empty CASA measurement sets. Acts
    as an interface with the `simms` CLI command on the system. Based on the provided
    arguments, a wide variety of measurement sets can be constructed."""

    tel: str | None = Field("meerkat", alias="-T", description="Telescope name")

    pos: str | None = Field(None, description="Antenna positions file")

    use_known_config: bool | None = Field(
        None,
        alias="-ukc",
        description="Use known antenna configuration. For some reason sm.setknownconfig() is not working.",
    )

    type: Literal["casa", "ascii"] | None = Field(
        None, alias="-t", description="Position list type"
    )

    coord_sys: Literal["itrf", "enu", "wgs84"] | None = Field(
        None,
        alias="-cs",
        description="Coordinate system of antenna positions (only relevant when type=ascii)",
    )

    lon_lat_elv: str | None = Field(
        None,
        alias="-lle",
        description="Reference position of telescope as comma-separated longitude,latitude,elevation [deg,deg,m]",
    )

    noup: bool | None = Field(
        None,
        alias="-nu",
        description="Indicate that ENU file does not have an 'up' dimension",
    )

    name: str | None = Field(
        None, alias="-n", description="MS name. Auto-generated if not specified"
    )

    outdir: str | None = Field(
        None,
        alias="-od",
        description="Directory to save the MS (default is working directory)",
    )

    label: str | None = Field(
        None, alias="-l", description="Label to add to auto-generated MS name"
    )

    direction: list[str] | None = Field(
        None,
        alias="-dir",
        description="Pointing direction (can be specified multiple times)",
    )

    ra: str | None = Field(
        None, alias="-ra", description="Right Ascension in hms or val[unit]"
    )

    dec: str | None = Field(
        None, alias="-dec", description="Declination in dms or val[unit]"
    )

    synthesis_time: float | None = Field(
        None, alias="-st", description="Synthesis time in hours"
    )

    scan_length: float | None = Field(
        None,
        alias="-sl",
        description="Scan length in hours (defaults to synthesis time)",
    )

    dtime: float | None = Field(
        None, alias="-dt", description="Integration time in seconds"
    )

    init_ha: float | None = Field(
        None, alias="-ih", description="Initial hour angle for observation (DEPRECATED)"
    )

    nchan: list[int] | None = Field(
        None,
        alias="-nc",
        description="Number of frequency channels (comma-separated for multiple subbands)",
    )

    freq0: list[str] | None = Field(
        None,
        alias="-f0",
        description="Start frequency as val[unit] (comma-separated for multiple subbands)",
    )

    dfreq: list[str] | None = Field(
        None,
        alias="-df",
        description="Channel width as val[unit] (comma-separated for multiple subbands)",
    )

    nband: int | None = Field(None, alias="-nb", description="Number of subbands")

    pol: list[str] | None = Field(None, alias="-pl", description="Polarization")

    feed: list[str] | None = Field(
        None, alias="-feed", description="Feed specification"
    )

    date: str | None = Field(
        None,
        alias="-date",
        description="Date of observation in format 'EPOCH,yyyy/mm/dd[/h:m:s]' (default is today)",
    )

    optimise_start: bool | None = Field(
        None,
        alias="-os",
        description="Modify observation start time to maximise source visibility",
    )

    scan_lag: float | None = Field(
        None, alias="-slg", description="Lag time between scans in hours (DEPRECATED)"
    )

    set_limits: bool | None = Field(
        None,
        alias="-stl",
        description="Set telescope limits (elevation and shadow limits)",
    )

    elevation_limit: float | None = Field(
        None,
        alias="-el",
        description="Dish elevation limit (only if set_limits is enabled)",
    )

    shadow_limit: float | None = Field(
        None, alias="-shl", description="Shadow limit (only if set_limits is enabled)"
    )

    auto_correlations: bool | None = Field(
        None, alias="-ac", description="Enable auto-correlations"
    )

    nolog: bool | None = Field(None, alias="-ng", description="Don't keep log file")

    json_config: str | None = Field(None, alias="-jc", description="JSON config file")

    def to_args(self, no_alias: bool = False) -> list[str]:
        """Convert the model instance to a list of CLI arguments for simms"""
        args = []

        # Handle positional argument (antenna positions)
        if self.pos is not None:
            args.append(self.pos)

        # Helper function to handle field conversion
        def add_arg(field_name: str, value: Any, alias: str | None = None):
            if value is not None:
                if isinstance(value, bool) and value:
                    args.append(alias or f"--{field_name.replace('_', '-')}")
                elif not isinstance(value, bool):
                    if alias:
                        arg_name = alias
                    else:
                        arg_name = f"--{field_name.replace('_', '-')}"

                    if isinstance(value, list):
                        if field_name in ["pol", "feed"]:
                            # These are space-separated lists
                            args.extend([arg_name, *value])
                        else:
                            # Most others are comma-separated
                            args.extend([arg_name, ",".join(map(str, value))])
                    else:
                        args.extend([arg_name, str(value)])

        # Add all optional arguments
        for field_name, field_info in self.model_fields.items():
            value = getattr(self, field_name)

            if value is None:
                continue

            if field_name == "pos":
                continue

            if no_alias:
                add_arg(field_name, value)
            else:
                alias = field_info.alias
                add_arg(field_name, value, alias)

        return args


class SimmsToolConfig(BaseToolConfig):
    """Configuration for the SimmsTool"""


class SimmsTool(BaseTool):
    """Tool for generating empty CASA measurement sets using the `simms` CLI command.

    Attributes:
        input_schema (SimmsToolInputSchema): The schema for the input data.
        output_schema (BashOutputSchema): The schema for the output data.
    """

    input_schema = SimmsToolInputSchema
    output_schema = BashOutputSchema

    def __init__(self, config: SimmsToolConfig = SimmsToolConfig()):
        super().__init__(config)

    def run(self, inputs: SimmsToolInputSchema) -> BashOutputSchema:
        """
        Executes the `simms` command with the given parameters.

        Args:
            inputs (SimmsToolInputSchema): The input parameters for the tool.

        Returns:
            BashOutputSchema: The result of the command execution.
        """
        return simms_command(*inputs.to_args(no_alias=True))


class InitialInputSchema(BaseIOSchema):
    """The initial input schema for the Simms Agent that contains the description
    of the empty CASA measurement set to generate using the `simms` CLI."""

    description: str = Field(
        ...,
        description="The detailed description of the type CASA measurement set the user wants created with `simms`.",
    )
    path: Path = Field(
        ...,
        description="The path object pointing to where the measurement set should go.",
    )


class RepeatedInputSchema(InitialInputSchema):
    """The repeated input schema for the Simms Agent that contains the description
    of the empty CASA measurement set to generate using the `simms` CLI. Contains
    the bash output from a previous Simms tool run."""

    ms_created: bool = Field(
        ..., description="Whether the measurement set exists on the given path or not."
    )


class OutputSchema(BaseIOSchema):
    """Final output schema of the Simms Agent providing information from the creation
    of the measurement set using the Simms Tool."""

    success: bool = Field(
        ...,
        description="Whether the creation of the measurement set was a success or not.",
    )


class SimmsAgent(BaseAgent):
    def __init__(self):
        config = BaseAgentConfig(
            client=instructor.from_openai(openai.OpenAI()),
            model="gpt-4o-mini",
            system_prompt_generator=SystemPromptGenerator(
                background=[
                    "You are an AI assistant that translates user requests into command-line arguments for the 'simms' tool.",
                    "The 'simms' tool creates empty CASA measurement sets based on specified parameters. You are provided with the tool's help text.",
                    "Your goal is to determine the exact arguments needed to fulfill the user's request and then structure the output according to the specified schema.",
                ],
                steps=[
                    "1. Parse the user's 'description' to identify all specified observational parameters (e.g., telescope, pointing, frequency, time, configuration, polarization, antenna positions).",
                    "3. Extract the target directory from the input 'path' and determine the argument for the '-od' flag.",
                    "4. Extract the desired measurement set filename from the input 'path' and determine the argument for the '-n' flag.",
                    "5. If antenna positions are provided as a file path (either in the description or potentially as the 'pos' argument context if available), determine the 'pos' positional argument.",
                    "6. Internally assemble the complete list of flags and arguments for the 'simms' command based on your analysis. This list will be used by the system to execute the command.",
                    "7. Execute the `simms` command using the `simms tool` based on these parsed arguments and flags.",
                    "8. Once complete, examine the bash output of the `simms tool` and decide what do to do from there.",
                ],
                output_instructions=[
                    "For more information, use the 'simms help text' tool to provide the CLI help text of the `simms` command.",
                    "If the creation of the measurement set encounters and error, then output `success` as False",
                    "Otherwise, if it was a success, populate the `success` with True",
                ],
            ),
            input_schema=InitialInputSchema,
            output_schema=SimmsToolInputSchema,
            max_tokens=1024,
            temperature=0.0,
        )
        super().__init__(config)

        self.tool = SimmsTool()
        self.register_context_provider(
            "simms help text", SimmsHelpText("Simms Help Text")
        )

    def add_tool_message(self, output: BashOutputSchema):
        self.memory.add_message("system", output)

    def add_assistant_message(self, output: OutputSchema):
        self.memory.add_message("assistant", output)

    def run(self, user_input: InitialInputSchema) -> OutputSchema:
        path = user_input.path
        max_tries = 3
        while max_tries:
            output = super().run(user_input)
            try:
                # WARNING: Manually setting paths because AI is messing it up
                output.name = path.name
                output.outdir = str(path.parent)
                bash_output = self.tool.run(output)
            except Exception as e:
                bash_output = BashOutputSchema(
                    stdout=None, stderr=str(e), return_code=None
                )

            # Success, stop
            if bash_output.is_success() and path.exists():
                output = OutputSchema(success=True)
                self.add_assistant_message(output)
                return output

            # Failure, first retry
            if isinstance(self.input_schema, InitialInputSchema):
                self.input_schema = RepeatedInputSchema

            # Repeat with additional information
            user_input = RepeatedInputSchema(
                description=user_input.description,
                path=path,
                ms_created=path.exists(),
            )
            self.add_tool_message(bash_output)
            max_tries -= 1

        output = OutputSchema(success=False)
        self.add_assistant_message(output)
        return output


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # desc = "Please create a simple example measurement set using defaults."
    # path = Path("tester.ms")
    # if path.exists():
    #     path.rmdir()
    # agent = SimmsAgent()
    # print(agent.run(InitialInputSchema(description=desc, path=path)))

    # with open("output-simple.jsonl", "w") as file:
    #     for message in agent.memory.history:
    #         content = message.content
    #         file.write(content.model_dump_json(indent=2) + "\n")

    desc = """Create a empty VLA-B measurement set with:
    - X-band (8-12 GHz) setup
    - ~1-hour total observation
    - Short (~5-10 sec) integrations
    - A generic mid-declination pointing
    - All antennas included
    """

    path = Path("data/vla_xband.MS")

    agent = SimmsAgent()
    print(agent.run(InitialInputSchema(description=desc, path=path)))

    with open("output-complex.jsonl", "w") as file:
        for message in agent.memory.history:
            content = message.content
            file.write(content.model_dump_json(indent=2) + "\n")
