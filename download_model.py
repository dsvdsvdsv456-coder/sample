from pathlib import Path

import requests


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path("models") / "hand_landmarker.task"


def download_model() -> None:
    """Download the official MediaPipe Hand Landmarker task model."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading hand landmarker model to {MODEL_PATH}...")

    response = requests.get(MODEL_URL, stream=True, timeout=30)
    response.raise_for_status()

    with MODEL_PATH.open("wb") as model_file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                model_file.write(chunk)

    print("Download complete.")


if __name__ == "__main__":
    try:
        download_model()
    except requests.RequestException as error:
        print("Could not download the model.")
        print(f"Reason: {error}")
