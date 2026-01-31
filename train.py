import os
from ultralytics import YOLO
import torch

def train_traffic_model():

    # 1. Setup Device
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f" Training on device: {device}")

    model = YOLO('yolo11s.pt') 

    print("starting training...")
    model.train(
        
        data='data/data.yaml',  
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        project='models',      
        name='traffic_run',
        exist_ok=True,
        plots=True,
        verbose=True
    )

    metrics = model.val()
    print(f"raining Complete. mAP50-95: {metrics.box.map}")

    model.export(format='onnx', dynamic=True)
    print("Export Complete: models/traffic_run/weights/best.onnx")

if __name__ == '__main__':
    train_traffic_model()