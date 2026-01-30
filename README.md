# AI Traffic Monitoring System

**Regnum Resource Ltd. - AI Engineer Technical Assessment**

**Candidate:** Hasan Mahmood
**Date:** January 31, 2026

---

## Project Overview

This repository contains the complete solution for the **Regnum Resource Ltd. AI Engineer Technical Assessment**. The system is a real-time computer vision pipeline designed to detect, track, and count vehicles in a video feed using **YOLOv11** and **ByteTrack**. It includes a **PyQt6** graphical user interface (GUI) for live analytics.

### Key Features

* **Detection Model:** Custom trained **YOLOv11s** on the provided dataset (11 classes)
* **Optimization:** Model exported to **ONNX** format for high-throughput inference (meeting Requirement 3.A)
* **Tracking:** Implemented **ByteTrack** for robust ID retention during occlusions (meeting Requirement 3.B)
* **GUI:** Responsive dashboard built with **PyQt6**, displaying real-time FPS, total vehicle counts, and class breakdowns (meeting Requirement 3.C)

---

## Project Structure

```
Regnum_Assessment/
│
├── data/                          # Dataset and Source Video
│   ├── train/                     # Training images/labels
│   ├── valid/                     # Validation images/labels
│   ├── data.yaml                  # YOLO dataset configuration
│   └── source_video.mp4           # The provided input video for testing
│
├── models/                        # Saved model weights
│   └── traffic_run/               # (Generated after training)
│       └── weights/
│           ├── best.pt            # PyTorch weights
│           └── best.onnx          # Optimized ONNX weights
│
├── src/                           # Source Code
│   ├── detector.py                # Inference engine & Annotation logic
│   ├── gui.py                     # PyQt6 Application & Dashboard
│   └── tracker.py                 # ByteTrack configuration
│
├── main.py                        # Application Entry Point
├── train.py                       # Model Training & Export Script
├── requirements.txt               # Project Dependencies
└── README.md                      # Documentation
```

---

## Setup & Installation

### 1. Prerequisites

- Python 3.10 or 3.11
- NVIDIA GPU (Recommended for training) with CUDA 11.8 or 12.x

### 2. Installation

Clone the repository and install the required dependencies:

```bash
python -m venv .venv

.\.venv\Scripts\activate

# Note: Installs PyTorch, Ultralytics, PyQt6, Supervision, and ONNX Runtime
pip install -r requirements.txt
```

---

## Usage

### 1. Training the Model

To reproduce the training process and generate new weights:

```bash
python train.py
```

**Output:**
- This will train YOLOv11s for 50 epochs and save the best model to `models/traffic_run/weights/best.pt`
- The script automatically exports the model to `best.onnx` at the end of training

### 2. Running the Application

To launch the Traffic Monitoring Dashboard:

```bash
python main.py
```

**Usage:**
- Click the **START** button to begin the feed
- **Note:** Ensure `data/Inference -1.mp4` exists before running

---

## Technical Details

### Model Selection

**Architecture:** YOLOv11 Small (yolo11s.pt) was selected as the optimal balance between inference speed (necessary for real-time processing) and detection accuracy (mAP).

**Classes:** Trained on 11 distinct classes including specific local vehicles like "CNG / Tempo" and "Auto Rickshaw".

### Optimization

**ONNX Runtime:** The pipeline is designed to leverage `onnxruntime-gpu` for inference. This bypasses Python overhead during the forward pass, significantly increasing FPS compared to standard PyTorch inference.

### Tracking Logic

**ByteTrack:** Unlike DeepSORT, ByteTrack utilizes low-confidence detection boxes to track objects that are partially occluded. This is crucial for dense traffic scenarios where vehicles often block each other.

---

## Deliverables Checklist

- [x] Source Code: Modular Python scripts (`src/`, `main.py`, `train.py`)
- [x] Optimized Model: ONNX weights included in submission folder
- [x] GUI: Functional PyQt6 application with analytics
- [x] Documentation: This README.md and inline code comments

---

## Contact

**Hasan Mahmood**  
Email: [hasanmahmudnayeem3027@gmail.com]

---

**Submitted for Regnum Resource Ltd. AI Engineer Position**  
**Date:** January 31, 2026