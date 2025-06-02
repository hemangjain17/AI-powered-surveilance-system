import cv2
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
import numpy as np
import os
import string

# Utility: IoU between two boxes
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

def processFaces(video_path: str, output_path: str, face_dir: str):
    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    saved_faces = []
    alphabet = list(string.ascii_uppercase)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output = model(pil_img)
        results = Detections.from_ultralytics(output[0])

        annotated_frame = frame.copy()
        face_ids = []

        for i in range(len(results)):
            x1, y1, x2, y2 = map(int, results.xyxy[i])
            conf = results.confidence[i]
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
                face_ids.append(alphabet[len(face_ids)])
                face_index = len(face_ids) - 1

            if 0 <= face_index < len(face_ids):
                name = face_ids[face_index]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                print(f"Detected face '{name}' - BBox: [{x1}, {y1}, {x2}, {y2}] - Confidence: {conf:.2f}")

        combined = np.hstack((frame, annotated_frame))
        out.write(combined)  
        cv2.imshow("Original vs Face Detection", combined)  # Removed

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    assert len(saved_faces) <= len(alphabet), "Too many faces to assign unique letters."
    for idx, (_, conf, face_img) in enumerate(saved_faces):
        face_name = alphabet[idx]
        filename = os.path.join(face_dir, f"{face_name}.jpg")
        cv2.imwrite(filename, face_img)
        print(f"Saved face {face_name} with confidence {conf:.2f} to {filename}")

    print(f"Saved {len(saved_faces)} uniquely labeled faces to '{face_dir}'")

if __name__ == "__main__":
    processFaces(
        video_path="videoplayback (1).mp4",
        output_path="output_face_detection.mp4",
        face_dir="detected_faces"
    )
