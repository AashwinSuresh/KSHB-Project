from google import genai
from PIL import Image
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Gemini API key
API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini client
client = genai.Client(
    api_key=API_KEY
)


def extract_designation(image_path):
    """
    Extract the designation of the receiver from a Malayalam letter.

    Parameters
    ----------
    image_path : str
        Path to cropped image.

    Returns
    -------
    str
        Designation in English.
    """

    img = Image.open(image_path)

    prompt = """
This image contains a Malayalam official letter.

Identify the designation of the person to whom the letter is addressed.

Translate the designation into English.

Rules:
- Return ONLY the designation.
- Do NOT include explanations.
- Do NOT include bullet points.
- Do NOT include labels.
- Do NOT include extra text.

Examples:

Class Teacher

Executive Engineer

Secretary

Assistant Professor

Junior Superintendent

Chairman

Village Officer

If the designation cannot be identified, return:

Unknown
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[img, prompt]
        )

        designation = response.text.strip()

        # Remove accidental markdown if Gemini adds it
        designation = designation.replace("```", "")
        designation = designation.strip()

        return designation

    except Exception as e:
        return f"Error: {str(e)}"