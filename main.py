import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import joblib

# Constants
IMAGE_SIZE = 64
ROI_COORDS = (100, 100, 400, 400)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load label encoder
le = joblib.load("model/label_encoder.pkl")

# Improved CNN model (must match training script)
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load trained model
num_classes = len(le.classes_)
model = ImprovedCNN(num_classes)
model.load_state_dict(torch.load("model/cnn_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Recognition (CNN)")
        self.root.geometry("700x550")
        self.root.configure(bg="#f0f0f0")

        self.label = Label(root, text="Detected Gesture:", font=("Arial", 20), bg="#f0f0f0")
        self.label.pack(pady=10)

        self.result_label = Label(root, text="Waiting...", font=("Arial", 30), fg="green", bg="#f0f0f0")
        self.result_label.pack()

        self.canvas = Label(root)
        self.canvas.pack()

        self.btn_start = Button(root, text="Start Camera", font=("Arial", 14), command=self.start_camera)
        self.btn_start.pack(pady=10)

        self.btn_stop = Button(root, text="Stop Camera", font=("Arial", 14), command=self.stop_camera)
        self.btn_stop.pack(pady=5)

        self.cap = None
        self.running = False

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.canvas.config(image="")
        self.result_label.config(text="Stopped")

    def preprocess(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5  # match training normalization
        tensor = torch.tensor(normalized).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, 64, 64]
        tensor = tensor.to(DEVICE)
        return tensor

    def predict_gesture(self, frame):
        x1, y1, x2, y2 = ROI_COORDS
        roi = frame[y1:y2, x1:x2]
        input_tensor = self.preprocess(roi)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            gesture = le.inverse_transform(predicted.cpu().numpy())[0]
        return gesture

    def update_frame(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                gesture = self.predict_gesture(frame)
                self.result_label.config(text=gesture)

                x1, y1, x2, y2 = ROI_COORDS
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.imgtk = imgtk
                self.canvas.config(image=imgtk)

            self.root.after(10, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()