import sys
import cv2
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QVBoxLayout, QWidget, QHBoxLayout, QGridLayout, QFrame)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import QTimer, Qt
from src.detector import TrafficDetector

class TrafficApp(QMainWindow):
    def __init__(self, model_path, video_source):
        super().__init__()
        self.setWindowTitle("Regnum AI Traffic Monitor")
        self.setGeometry(100, 100, 1280, 800)
        self.video_source = video_source
        self.is_running = False
        self.cap = None

        # Initialize Detector
        self.detector = TrafficDetector(model_path)
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.video_label = QLabel("Feed Stopped")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background: black; color: white;")
        self.video_label.setMinimumSize(960, 540)
        main_layout.addWidget(self.video_label, stretch=3)

        dash_layout = QVBoxLayout()
        
        title = QLabel("Traffic Analytics")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        dash_layout.addWidget(title)

        self.fps_lbl = QLabel("FPS: 0.0")
        self.fps_lbl.setStyleSheet("font-size: 14px; font-weight: bold;")
        dash_layout.addWidget(self.fps_lbl)

        self.count_lbl = QLabel("Vehicles: 0")
        self.count_lbl.setStyleSheet("font-size: 14px; font-weight: bold;")
        dash_layout.addWidget(self.count_lbl)

        self.breakdown_lbl = QLabel("Waiting...")
        dash_layout.addWidget(self.breakdown_lbl)

        self.btn = QPushButton("START")
        self.btn.setStyleSheet("background: green; color: white; padding: 10px;")
        self.btn.clicked.connect(self.toggle)
        dash_layout.addWidget(self.btn)

        main_layout.addLayout(dash_layout, stretch=1)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def toggle(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(self.video_source)
            self.timer.start(30)
            self.btn.setText("STOP")
            self.btn.setStyleSheet("background: red; color: white; padding: 10px;")
            self.is_running = True
        else:
            self.timer.stop()
            if self.cap: self.cap.release()
            self.btn.setText("START")
            self.btn.setStyleSheet("background: green; color: white; padding: 10px;")
            self.is_running = False

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            processed_frame, metrics = self.detector.process_frame(frame)
            
            self.display_image(processed_frame)
            
            self.fps_lbl.setText(f"FPS: {metrics['fps']:.1f}")
            self.count_lbl.setText(f"Vehicles: {metrics['total']}")
            
            txt = "\n".join([f"{k}: {v}" for k,v in metrics['breakdown'].items()])
            self.breakdown_lbl.setText(txt)
        else:
            self.toggle()

    def display_image(self, img):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch*w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio
        ))