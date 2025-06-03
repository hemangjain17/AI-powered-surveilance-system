import re
import os 
import cv2
import base64
import requests
from faceDetectionSeq import processFaces
from eventDescription import relativeDescription

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    from dotenv import load_dotenv
    load_dotenv(dotenv_path)

VIDEO_PATH = "testingVideo/videoplayback.mp4"
CHUNK_DURATION = 8 
FRAME_SAMPLE_RATE = 3  
FPS = 20

GEMINI_MODEL = "google/gemini-2.0-flash-001"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HARMFUL_KEYWORDS = ["fight","walk", "walking", "violence", "attack", "slap", "punch", "abuse", "harm", "injury", "danger", "threat", "assault", "aggression", "hostile", "disturbance", "altercation"]

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/jpeg;base64,{img_b64}"


img1 = encode_image_to_base64("Sample Images/sample1.jpg")
img2 = encode_image_to_base64("Sample Images/sample2.jpg")

def get_gemini_response(image_b64_list):
    """
    Uses Gemini-2.0 to analyze a sequence of images and return a short description,
    with enhanced prompting and reasoning capability.
    """

    one_shot_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "You are an expert visual event analyst specializing in detecting and describing potentially harmful scenarios. "
                    "Carefully observe the sequence of images and describe the event unfolding, focusing on interactions between individuals. "
                    "Your response should be a concise 6-7 word summary of the event or scenario depicted in the image sequence."
                )},
                {"type": "image_url", "image_url": {"url": img1}},
                {"type": "image_url", "image_url": {"url": img2}}
            ]
        },
        {
            "role": "assistant",
            "content": "Two people are fighting"
        }
    ]

    user_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": (
                "You are an expert visual event analyst specializing in detecting and describing potentially harmful scenarios. "
                "Carefully observe the sequence of images and describe the event unfolding, focusing on interactions between individuals. "
                "Your response should be a concise 6-7 word summary of the event or scenario depicted in the image sequence."
            )}
        ]
    }

    # Attach user-provided images
    for img_b64 in image_b64_list:
        user_message["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })

    # Combine one-shot with main input
    messages = one_shot_messages + [user_message]

    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "HTTP-Referer": "https://your-app.com",
        "X-Title": "Video Event Analysis"
    }

    payload = {
        "model": GEMINI_MODEL,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "n": 1,
    }


    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        json=payload,
        headers=headers
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def encode_image(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def detect_harmful(description):
    return any(keyword in description.lower() for keyword in HARMFUL_KEYWORDS)

def trigger_action(chunk_id, frames, short_description):
    print(f"Harmful event detected in chunk {chunk_id}: {short_description}")
    for i, frame in enumerate(frames):
        cv2.imwrite(f"harmful_chunk_{chunk_id}_frame_{i}.jpg", frame)

def overlay_text(frame, text):
    cv2.putText(frame, text, (25, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 1, cv2.LINE_AA)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    chunk_frames = int(CHUNK_DURATION * fps)
    print(f"Chunk size: {chunk_frames} frames ({CHUNK_DURATION} seconds)")
    sample_step = int(FRAME_SAMPLE_RATE * fps)
    
    chunk_id = 0
    while cap.isOpened():
        frames = []
        for i in range(chunk_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i % sample_step == 0:
                frames.append(frame)

        if not frames:
            break

        b64_frames = [encode_image(f) for f in frames]
        try:
            response = get_gemini_response(b64_frames)
            print(f"Chunk {chunk_id} description: {response}")

            harmful = detect_harmful(response)
            display_text = response  

            if harmful:
                trigger_action(chunk_id, frames, response)

                faceDescription = processFaces(b64_frames)
                print(f"Face analysis for chunk {chunk_id}: {faceDescription} \n")

                valid_faces = [
                    {
                        'frame_index': frame_data['frame_index'],
                        'name': face['name'],
                        'bbox': face['bbox']
                    }
                    for frame_data in faceDescription
                    for face in frame_data['faces']
                ]

                eventDescription = relativeDescription(
                    b64_frames=b64_frames,
                    faces=valid_faces,
                    frame_size=frame_size,
                    general_summary=response
                )
                print(f"Event description for chunk {chunk_id}: {eventDescription} \n")

                match = re.search(r"Details:\s*(.+)", eventDescription, re.DOTALL)
                if match:
                    details_text = match.group(1).strip()
                    display_text = details_text 
                    print(f"Details for chunk {chunk_id}: {details_text} \n")
                else:
                    print(f"No 'Details:' section found in chunk {chunk_id} response. \n")

            for frame in frames:
                display_frame = frame.copy()
                if harmful:
                    overlay_text(display_frame, display_text) 
                cv2.imshow("Video Stream", display_frame)

                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
        except Exception as e:
            print(f"Error in processing chunk {chunk_id}: {e} \n")

        chunk_id += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Video processing complete.")

if __name__ == "__main__":
    process_video(VIDEO_PATH)
