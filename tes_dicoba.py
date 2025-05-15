import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv11 model
model_path = "C:\\TA-Lalu-Lintas\\bestYolo11.pt"
model = YOLO(model_path)

# Inisialisasi Deep SORT
tracker = DeepSort(
    max_age=15,
    n_init=3,
    max_cosine_distance=0.4,
    nn_budget=100
)

# Buka video input
video_path = "C:\\TA-Lalu-Lintas\\dataset-mentah\\vidios\\WIN_20250507_15_05_58_Pro.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []

    # Ambil deteksi dari YOLO
    for box in results.boxes:
        cls = int(box.cls[0].item())

        if cls == 0:  # Hanya deteksi "Pengendara"
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            w, h = x2 - x1, y2 - y1
            bbox = [x1, y1, w, h]
            detections.append((bbox, conf, "Pengendara"))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        conf = track.det_conf if hasattr(track, 'det_conf') else 0.0

        # Gambar bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Teks label
        text = f"ID {track_id} | Pengendara | {conf:.2f}"

        # Ukuran dan posisi label
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        label_bg_topleft = (x1, y1 - text_height - 10)
        label_bg_bottomright = (x1 + text_width + 6, y1)

        # Gambar kotak background untuk teks
        cv2.rectangle(frame, label_bg_topleft, label_bg_bottomright, (0, 255, 0), -1)

        # Tampilkan teks di atas box (dengan outline biar makin jelas)
        cv2.putText(frame, text, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    # Tampilkan frame
    cv2.imshow("Tracking Pengendara", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32:
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
