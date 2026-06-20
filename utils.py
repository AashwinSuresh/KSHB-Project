import os


def check_model_exists():

    model_path = "models/dla-model.pt"

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )


def create_temp_folder():

    os.makedirs(
        "temp",
        exist_ok=True
    )