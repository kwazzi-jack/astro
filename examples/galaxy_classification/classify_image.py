import urllib.request
from dataclasses import dataclass
from pathlib import Path

import timm  # type: ignore
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_URL = "https://huggingface.co/kwazzi-jack/astro-galaxy-classify-demo/resolve/main/galaxy_classify_mixed_inception_v3.pth"
MODEL_PATH = Path("galaxy_classify_mixed_inception_v3.pth")


@dataclass
class ClassifyResult:
    FRI_pred: float
    FRII_pred: float
    pred: float


@dataclass
class ClassifyInput:
    image_path: str | Path


def create_model(
    model_name: str,
    model_weights_path: str,
    num_classes: int = 2,
    in_chans: int = 1,
    pretrained: bool = True,
):
    """
    Create a model with conditional handling for specific architectures.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_chans,
    )
    input_size = (in_chans, 299, 299)

    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()

    return model, input_size


def set_transform(dataset, transform):
    """Safely apply transform to any dataset type recursively."""
    if hasattr(dataset, "transform"):
        dataset.transform = transform
    if hasattr(dataset, "dataset"):  # Subset
        set_transform(dataset.dataset, transform)
    if hasattr(dataset, "datasets"):  # ConcatDataset
        for ds in dataset.datasets:
            set_transform(ds, transform)


def download_model():
    if not MODEL_PATH.exists():
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def classify(
    image_path: str,
) -> ClassifyResult:
    """Classifies a galaxy image into FRI or FRII categories using a pre-trained model.

    This function loads a specified model, preprocesses the input image, performs
    inference to predict the galaxy class, and returns the probabilities for each
    class along with the predicted class index.

    Args:
        image_path: The file path to the input galaxy image.

    Returns:
        ClassifyResult: A dataclass containing the probabilities for each class (FRI and FRII) as floats,
        and the predicted class index (as an int, but typed as float in the return annotation for consistency).

    Raises:
        IOError: If an error occurs while loading or saving the model, or processing the image.

    Examples:
        >>> result = classify("/path/to/galaxy_image.jpg")
        >>> print(result)
        ClassifyResult(FRI_pred=0.3, FRII_pred=0.7, pred=1.0)
        >>> result = classify("/path/to/another_image.png")
        >>> print(result.pred)
        1.0
    """

    # Download model
    download_model()

    # Creates model and loads model weights
    model, input_size = create_model(
        model_name="inception_v3",
        model_weights_path=str(MODEL_PATH),
    )

    _, H, W = input_size
    transform = transforms.Compose(
        [
            transforms.Resize((H, W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # --- Function to predict class ---
    image = Image.open(image_path).convert("L")  # Ensure grayscale
    image = transform(image).unsqueeze(0)  # pyright: ignore[reportAttributeAccessIssue] # Add batch dimension
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()  # evaluates the predicted class
        probs = F.softmax(output, dim=1).squeeze(
            0
        )  # evaluates probabilities for each class

    result = ClassifyResult(
        FRI_pred=float(probs[0]),
        FRII_pred=float(probs[1]),
        pred=pred,
    )
    return result


def main():
    print(classify("test-images/F_100_11.png"))


tools = (classify,)

if __name__ == "__main__":
    main()
