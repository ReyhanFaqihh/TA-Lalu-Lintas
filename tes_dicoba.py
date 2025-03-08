import cv2
from ultralytics import YOLO

# Load model YOLOv11 yang telah dilatih
model_path = "D:\\Tim Lalu Lintas\\TA-Lalu-Lintas\\runs\\detect\\train_fold4\\weights\\best.pt"
model = YOLO(model_path)

# Buka video input
video_path = "D:\\Tim Lalu Lintas\\TA-Lalu-Lintas\\vidio\\TimeVideo_20250123_092456.mp4"
cap = cv2.VideoCapture(video_path)

# Threshold confidence untuk tiap kelas
conf_threshold = {
    0: 0.8,  # Threshold untuk "Pengendara"
    1: 0.7   # Threshold untuk "Helmet"
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Lakukan deteksi menggunakan YOLO
    results = model(frame)

    # Gambar bounding box pada frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinat bounding box
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Kelas deteksi
            
            # Tentukan label berdasarkan kelas
            label = "Pengendara" if cls == 0 else "Helmet"
            color = (0, 255, 0) if cls == 0 else (255, 0, 0)  # Merah untuk "No Helmet", biru untuk "Helmet"

            # Gunakan threshold berbeda untuk tiap kelas
            if conf >= conf_threshold[cls]:  
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Tampilkan hasil
    cv2.imshow("YOLOv11 Helmet Detection", frame)

    # Tambahkan opsi pause dengan tombol "Space"
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Tekan "q" untuk keluar
        break
    elif key == 32:  # Tekan "Space" untuk pause
        cv2.waitKey(0)  

# Tutup semua proses
cap.release()
cv2.destroyAllWindows()
