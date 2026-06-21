import gradio as gr

from cropper import process_image
from gemini_extractor import extract_designation
from classifier import save_to_folders


def analyze_document(image_path):
    """
    Complete pipeline:

    Upload Image
            ↓
    Layout Detection
            ↓
    Crop Region
            ↓
    Gemini Vision
            ↓
    Extract Receiver Category
            ↓
    Save Letter Into Corresponding Folder(s)
    """

    try:

        # Perform layout analysis and cropping
        original_image, boxed_image, cropped_image = process_image(
            image_path
        )

        # Extract designation(s) from cropped image
        designation = extract_designation(
            "temp/cropped_region.jpg"
        )

        # Save the original letter into all matching folders
        folders = save_to_folders(
            image_path,
            designation
        )

        result_text = (
            f"Designation(s):\n"
            f"{designation}\n\n"
            f"Saved to folder(s):\n"
            f"{', '.join(folders)}"
        )

        return (
            original_image,
            boxed_image,
            cropped_image,
            result_text
        )

    except Exception as e:

        return (
            None,
            None,
            None,
            f"Error: {str(e)}"
        )


iface = gr.Interface(
    fn=analyze_document,

    inputs=gr.Image(
        type="filepath",
        label="Upload Malayalam Letter"
    ),

    outputs=[
        gr.Image(label="Original Image"),
        gr.Image(label="Layout Detection"),
        gr.Image(label="Cropped Region"),
        gr.Textbox(label="Classification Result")
    ],

    title="KSHB Malayalam Letter Classification",

    description="""
1. Upload a Malayalam letter.
2. Perform document layout analysis using YOLO.
3. Crop the important region.
4. Send only the cropped image to Gemini Vision.
5. Identify whether the receiver(s) are:
   - Chief Engineer
   - Chairman
   - Secretary
6. Automatically create folders if needed.
7. Save the original letter into all matching folders.
"""
)


if __name__ == "__main__":
    iface.launch(share=True)