from dotenv import load_dotenv

from astro import run_astro_with
from astro.agents.base import create_agent
from examples.galaxy_classification.classify_image import ClassifyResult

load_dotenv()


def create_classify_agent():
    from classify_image import ClassifyResult, classify

    agent = create_agent(
        "openai:gpt-4o-mini",
        tools=classify,
        output_type=ClassifyResult,
        model_settings={
            "temperature": 0.0,
            "max_tokens": 1024,
        },
        instructions=(
            "You are a helpful assistant that will classify galaxy images. "
            "When given an image path, use the classify tool to classify the galaxy and return the result."
        ),
    )
    return agent


# Tool function that uses the classification agent
def classify_galaxy(image_path: str) -> ClassifyResult:
    """Classify a galaxy image.

    Args:
        image_path: Path to the galaxy image.

    Returns:
        ClassifyResult: Classification result.
    """
    agent = create_classify_agent()
    result = agent.run_sync(f"Classify the galaxy image at {image_path}")
    return result.data


# Tools for the chat agent
tools = (classify_galaxy,)

# Instructions for the chat agent
instructions = (
    "You are a helpful assistant specializing in astronomy and galaxy classification. "
    "When a user asks to classify a galaxy image, use the classify_galaxy tool with the provided image path. "
    "Provide the classification results in a clear and informative way."
)


if __name__ == "__main__":
    # Run the chat agent with the classification tool
    run_astro_with(items=tools, instructions=instructions)
