import os
import shutil


def save_to_folders(original_image_path, designation_text):
    """
    Save the original letter into all matching folders.

    Parameters
    ----------
    original_image_path : str
        Path to the uploaded letter.

    designation_text : str
        Output from Gemini.

    Returns
    -------
    list
        List of folders where the file was saved.
    """

    saved_folders = []

    designation_text = designation_text.lower()

    # Detect categories
    if "chief engineer" in designation_text:
        saved_folders.append("chief engineer")

    if "chairman" in designation_text:
        saved_folders.append("chairman")

    if "secretary" in designation_text:
        saved_folders.append("secretary")

    # If no category matched
    if len(saved_folders) == 0:
        saved_folders.append("unknown")

    # Create parent folder
    parent_folder = "classified"
    os.makedirs(parent_folder, exist_ok=True)

    filename = os.path.basename(original_image_path)

    # Copy image into each category folder
    for folder in saved_folders:

        folder_path = os.path.join(parent_folder, folder)

        # Create category folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

        destination = os.path.join(folder_path, filename)

        shutil.copy(original_image_path, destination)

    return [os.path.join(parent_folder, folder) for folder in saved_folders]