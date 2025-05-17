import cv2
import os

os.system("libcamera-still --shutter 3000 --gain 8 --flicker 50hz -t 1000 --nopreview -0 /dev/null ")

gst_str = (
     "libcamerasrc ! "
     "video/x-raw, width=640, height=480, framerate=30/1,format=NV12 ! "
     "videoconvert ! "
     "video/x-raw, format=BGR ! "
     "appsink"
)

cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Gagal membuka Pipeline Gstreamer")
    exit()

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
         print("Gagal membaca frame dari kamera")
         continue

    cv2.imshow("Kamera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break

cap.release()
cv2.destroyAllWindows()
