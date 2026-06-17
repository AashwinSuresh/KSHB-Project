from google import genai
from PIL import Image
from dotenv import load_dotenv
import os


load_dotenv()

# 1. Initialize the client with your key directly
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

try:
    # 2. Load the Malayalam letter image using Pillow
    # Replace "letter.jpg" with your image's actual filename or path
    img = Image.open(r".\test_images\mal_sample.png")
    
    # 3. Create a clear prompt telling Gemini exactly what to do
    prompt = """
    This is an image of a formal letter written in Malayalam. 
    Please identify who the letter is addressed to (the recipient / 'To' address section).
    Translate their name, designation, and address into English and print it clearly.
    Only give me the name of the sender in english (no other things needed)
    """
    
    print("Analyzing the image... Please wait.")
    
    # 4. Pass the image and the prompt to the model
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[img, prompt]
    )
    
    print("\n--- Recipient Details (Translated to English) ---")
    print(response.text)

except FileNotFoundError:
    print("Error: Could not find the image file. Make sure the filename is correct.")
except Exception as e:
    print("An error occurred:")
    print(e)