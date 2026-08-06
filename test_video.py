import cv2
import time

path = r"D:\work\detectors_validator\web_app\temp\364f5d3f-7601-476d-9630-f0ea84a5d161_signal-2026-07-02-15-19-19-043.mp4"
cap = cv2.VideoCapture(path)

print(f"Is opened: {cap.isOpened()}")
if cap.isOpened():
    ret, frame = cap.read()
    print(f"Read first frame: {ret}")
    if ret:
        print(f"Frame shape: {frame.shape}")
        
cap.release()
