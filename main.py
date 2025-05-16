import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image
import os
import numpy as np
import datetime
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def load_cnn_model():
    model = efficientnet_b0(weights=None)  # Tidak menggunakan pretrained weights
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
    return model.to(device).eval()


# ==== Preprocessing Functions ====
def remove_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# ==== Configuration ====
YOLO_MODEL_PATH = "bestYolo11.pt"
CNN_MODEL_PATH = "best_model_efficientnet.pth"
VIDEO_PATH = "C:\\TA-Lalu-Lintas\\dataset-mentah\\vidios\\WIN_20250507_15_05_58_Pro.mp4"
SAVE_DIR = "pelanggar"
MIN_BOX_AREA = 90000
CNN_CONFIDENCE_THRESHOLD = 0.8
HEAD_CROP_RATIO = 0.4  # Bagian atas bounding box untuk crop kepala
gst_str = (
    "libcamerasrc ! "
    "video/x-raw, width=640, height=480, framerate=30/1 ! "
    "videoconvert ! "
    "appsink"
)   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load YOLO Model ====
yolo_model = YOLO(YOLO_MODEL_PATH).to(device)

# ==== Load EfficientNetB0 Model ====
def load_cnn_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
    return model.to(device).eval()

cnn_model = load_cnn_model()

cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

os.makedirs(SAVE_DIR, exist_ok=True)

# ==== Tracker ====
tracker = DeepSort(max_age=15, n_init=3, max_cosine_distance=0.4, nn_budget=100)

# ==== Pelacakan Pelanggar ====
processed_ids = set()
HELMET_LABELS = ["Helm", "Head"]  # 0 = pakai, 1 = tidak pakai

# ==== Video Processing ====
cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
# cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.flip(frame, 0)  # Flip frame horizontally
    # frame = cv2.flip(frame, 1)  # Flip frame horizontally

    results = yolo_model(frame, verbose=False)[0]
    detections = []

    # Ambil deteksi pengendara motor (kelas 0)
    for box in results.boxes:
        cls = int(box.cls[0].item())
        if cls == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, "Pengendara"))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        conf = track.det_conf if hasattr(track, 'det_conf') else 0.0

        # Tampilkan bounding box untuk semua deteksi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label_text = f"ID {track_id} | YOLO:{conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 6, y1), (0, 255, 255), -1)
        cv2.putText(frame, label_text, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        box_area = (x2 - x1) * (y2 - y1)
        if box_area < MIN_BOX_AREA or track_id in processed_ids:
            continue

        # Crop full body
        cropped_body = frame[y1:y2, x1:x2]
        if cropped_body.size == 0:
            continue

        # Crop kepala dari bagian atas bounding box
        head_h = int((y2 - y1) * HEAD_CROP_RATIO)
        head_crop = frame[y1:y1 + head_h, x1:x2]
        if head_crop.size == 0:
            continue

        try:
            img = cv2.resize(head_crop, (224, 224))
            img = remove_noise(img)
            img = sharpen_image(img)
            img = enhance_contrast(img)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tensor = cnn_transform(img_pil).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Preprocessing error: {e}")
            continue

        with torch.no_grad():
            outputs = cnn_model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

        label_idx = preds.item()
        cnn_conf = probs[0][label_idx].item()
        processed_ids.add(track_id)

        # Simpan pelanggaran jika tidak pakai helm
        if label_idx == 1 and cnn_conf > CNN_CONFIDENCE_THRESHOLD:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
            save_path = os.path.join(SAVE_DIR, f"violation_ID{track_id}_{timestamp}.jpg")
            cv2.imwrite(save_path, cropped_body)
            print(f"[!] Pelanggaran disimpan: ID {track_id} | CNN Conf: {cnn_conf:.2f}")

    # Tampilkan frame
    cv2.imshow("E-Tilang Helmet Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()