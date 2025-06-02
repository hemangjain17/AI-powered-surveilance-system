import cv2
import base64
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
import string

def decode_base64_to_image(b64_string):
    img_data = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 3) * max(0, yB - yA + 3)
    if interArea == 0:
        return 0
    boxAArea = (boxA[2] - boxA[0] + 3) * (boxA[3] - boxA[1] + 3)
    boxBArea = (boxB[2] - boxB[0] + 3) * (boxB[3] - boxB[1] + 3)
    return interArea / float(boxAArea + boxBArea - interArea)

def processFaces(b64_images):
    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    model = YOLO(model_path)

    saved_faces = []
    alphabet = list(string.ascii_uppercase)
    face_id_map = {}

    results_summary = []

    for img_index, b64_img in enumerate(b64_images):
        frame = decode_base64_to_image(b64_img)
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output = model(pil_img)
        results = Detections.from_ultralytics(output[0])

        image_results = []

        for i in range(len(results)):
            x1, y1, x2, y2 = map(int, results.xyxy[i])
            conf = float(results.confidence[i])
            new_face = frame[y1:y2, x1:x2]
            is_new = True
            face_index = -1

            for idx, (bbox, stored_conf, _) in enumerate(saved_faces):
                iou = compute_iou([x1, y1, x2, y2], bbox)
                if iou > 0.5:
                    face_index = idx
                    if conf > stored_conf:
                        saved_faces[idx] = ([x1, y1, x2, y2], conf, new_face.copy())
                    is_new = False
                    break

            if is_new:
                saved_faces.append(([x1, y1, x2, y2], conf, new_face.copy()))
                face_index = len(saved_faces) - 1

            name = alphabet[face_index] if face_index < len(alphabet) else f"Face{face_index}"
            face_id_map[face_index] = name

            image_results.append({
                "name": name,
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 2),
                "image_index": img_index
            })

        results_summary.append({
            "frame_index": img_index,
            "faces": image_results
        })

    return results_summary
