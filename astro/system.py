import subprocess

from pydantic import Field

from atomic_agents.agents.base_agent import BaseIOSchema


class BashOutputSchema(BaseIOSchema):
    """Output schema for a bash command. Contains the standard
    output and error plus the return code from the command."""

    stdout: str | None = Field(
        None, description="Captured standard output from bash command"
    )
    stderr: str | None = Field(
        None, description="Captured standard error from bash command"
    )
    return_code: int | None = Field(None, description="Return code from bash command")

    def is_success(self) -> bool:
        return self.return_code == 0


def command(*args: str) -> BashOutputSchema:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
        return BashOutputSchema(
            stdout=result.stdout.strip() if result.stdout else "",
            stderr=result.stderr.strip() if result.stderr else "",
            return_code=result.returncode,
        )
    except FileNotFoundError as e:
        # Specific handling for command not found
        return BashOutputSchema(
            stdout=None, stderr=str(e), return_code=None
        )  # return_code is None here
    except Exception as e:
        # Handle other potential errors during subprocess start
        return BashOutputSchema(stdout=None, stderr=str(e), return_code=None)
