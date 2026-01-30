import sys
import os
import torch
from ultralytics import YOLO

from PyQt6.QtWidgets import QApplication
from src.gui import TrafficApp


MODEL_PATH = "runs/detect/models/traffic_run/weights/best.pt" 
VIDEO_SOURCE = "data/Inference -1.mp4" 

def main():
    if not os.path.exists(MODEL_PATH):
        print(f" Error: Model not found at {MODEL_PATH}")

        return

    if not os.path.exists(VIDEO_SOURCE):
        print(f" Error: Video not found at {VIDEO_SOURCE}")
        return


    app = QApplication(sys.argv)
    

    try:
        window = TrafficApp(MODEL_PATH, VIDEO_SOURCE)
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Runtime Error: {e}")

if __name__ == "__main__":
    main()