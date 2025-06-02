import cv2
import os
import base64
import requests
from PIL import Image 
from ultralytics import YOLO
# Config
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    from dotenv import load_dotenv
    load_dotenv(dotenv_path)


VIDEO_PATH = "videoplayback (1).mp4"
CHUNK_DURATION = 10 
FRAME_SAMPLE_RATE = 2  
FPS = 20

GEMINI_MODEL = "google/gemini-2.0-flash-001"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HARMFUL_KEYWORDS = ["fight", "violence", "attack", "slap", "punch", "abuse", "harm", "injury", "danger", "threat", "assault", "aggression", "hostile", "disturbance", "altercation"]

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/jpeg;base64,{img_b64}"

# Example usage
img1 = encode_image_to_base64("Sample Images\sample6.jpg")
img2 = encode_image_to_base64("Sample Images\sample2.jpg")

def relativeDescription(b64_frames, faces, frame_size, general_summary=""):
    width, height = frame_size
    person_descriptions = []
    face_names = []
    bboxes = []

    for face in faces:
        name = face.get("name", "Unknown")
        bbox = face.get("bbox", [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            position = (
                "left" if center_x < width / 2 else
                # "center" if center_x < 2 * width / 3 else
                "right"
            )
            person_descriptions.append(f"{name} is on the {position} side")
            face_names.append(name)
            bboxes.append(bbox)

    # Few-shot examples
    few_shot_prompt = (
        "**Few-shot Examples**:\n"
        "Example 1:\n"
        f"Images: {img2}\n"
        "Names: [Aman, Bassi]\n"
        "Bounding Boxes: [[100, 120, 180, 220], [300, 130, 380, 210]]\n"
        "Event: '2 people are seen fighting'\n"
        "Expected Response:\n"
        "Event Summary: Fight between Aman and Bassi\n"
        "Details: Aman is beating Bassi \n\n"

        "Example 2:\n"
        f"Images: {img1}\n"
        "Names: [Alice, Sanya, Amanda, Elie]\n"
        "Bounding Boxes: [[350, 100, 450, 220], [300, 150, 400, 320],[300, 50,  400, 230], [209, 120, 250, 20]]\n"
        "Event: 'A group is beating a lady'\n"
        "Expected Response:\n"
        "Event Summary: Three girls are beating one lady\n"
        "Details: Alice, Amanda and Elie are beating Sanya \n\n"
    )

    base_prompt = (
        f"{few_shot_prompt}\n"
        "Now analyze the following:\n"
    )

    if general_summary:
        base_prompt += f"General Event Summary: {general_summary}\n\n"

    if face_names:
        base_prompt += f"Names: {face_names}\n"

    if bboxes:
        base_prompt += f"Bounding Boxes: {bboxes}\n"

    if person_descriptions:
        base_prompt += "Detected People and Relative Positions:\n" + "\n".join(person_descriptions) + "\n\n"

    base_prompt += (
        "You are a visual reasoning expert.\n"
        "You will be shown a sequence of images from a video.\n"
        "Your task is to describe the short event (6–7 words max) and give a relative positioning + interaction explanation of people involved.\n"
        "Mention names if available; otherwise use relative references to identify the faces in the image (left/ right person).\n"
        "Use the image + text context to generate an accurate response.\n\n"
        "Respond in this format:\n"
        "Event Summary: <short summary>\n"
        "Details: <who is where and what they're doing>\n"
    )

    system_message = {
        "role": "system",
        "content": "You are an expert in spatial visual reasoning and human interaction analysis from images."
    }

    user_message = {
        "role": "user",
        "content": [{"type": "text", "text": base_prompt}]
    }

    for img_b64 in b64_frames:
        user_message["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })

    messages = [system_message, user_message]

    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "HTTP-Referer": "https://your-app.com",
        "X-Title": "Relative Person Positioning"
    }

    payload = {
        "model": GEMINI_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "top_p": 1,
        "n": 1
    }

    # Send request
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        json=payload,
        headers=headers
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
