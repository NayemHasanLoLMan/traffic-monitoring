import time
import numpy as np
from ultralytics import YOLO
import supervision as sv
from src.tracker import get_tracker_config
import torch

class TrafficDetector:
    def __init__(self, model_path):
        print(f"Loading model from: {model_path}")
        # Determine the device to use
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Running detection on device: {device}")
        self.model = YOLO(model_path)
        self.model.to(device)
        self.tracker_config = get_tracker_config()

        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30)

    def process_frame(self, frame):
        start_time = time.time()

        results = self.model.track(
            frame, 
            persist=True, 
            tracker=self.tracker_config, 
            verbose=False
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        if detections.tracker_id is None:
            detections.tracker_id = np.array([], dtype=int)
            detections.class_id = np.array([], dtype=int)
            detections.confidence = np.array([], dtype=float)

        labels = []
        counts = {}

        for tracker_id, class_id, conf in zip(detections.tracker_id, detections.class_id, detections.confidence):
            class_name = self.model.names[class_id]
            labels.append(f"#{tracker_id} {class_name} {conf:.2f}")
            counts[class_name] = counts.get(class_name, 0) + 1

        annotated_frame = self.box_annotator.annotate(frame.copy(), detections)
        annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)
        
        fps = 1.0 / (time.time() - start_time)
        
        metrics = {
            "fps": fps,
            "total": len(detections),
            "breakdown": counts
        }

        return annotated_frame, metrics