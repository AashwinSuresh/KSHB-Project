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

Identify the receiver(s) of the letter.

Classify each receiver into one or more of the following categories:

Chief Engineer
Chairman
Secretary

Rules:
- Return only the category names.
- If there are multiple receivers, return each category on a separate line.
- Do not include explanations.
- Do not include bullet points.
- Do not include numbering.
- Do not include labels.
- Do not use markdown.

Examples:

Chief Engineer

Chairman

Secretary

Chief Engineer
Chairman

Chairman
Secretary

If none of the receivers belong to these categories, return:

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