import gradio as gr
from cropper import process_image
from gemini_extractor import extract_designation


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
    Extract Designation
    """

    try:

        # Run layout analysis and cropping
        original_image, boxed_image, cropped_image = process_image(
            image_path
        )

        # Extract designation from cropped image
        designation = extract_designation(
            "temp/cropped_region.jpg"
        )

        return (
            original_image,
            boxed_image,
            cropped_image,
            designation
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
        gr.Textbox(label="Designation")
    ],

    title="KSHB Malayalam Letter Classification",

    description="""
1. Upload a Malayalam letter.
2. Perform document layout analysis using YOLO.
3. Crop the important region.
4. Send only the cropped image to Gemini Vision.
5. Extract the receiver's designation.
"""
)

if __name__ == "__main__":
    iface.launch(share=True)