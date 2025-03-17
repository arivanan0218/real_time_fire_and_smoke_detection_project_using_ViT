import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import torch
from model import FireSmokeDetector
import torchvision.transforms as transforms
import os

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FireSmokeDetector(num_classes=2).to(device)
model.load_state_dict(torch.load(
    'fire_smoke_detector.pth', map_location=device))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to predict


def predict(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item()

# Function to open camera


def open_camera():
    cap = cv2.VideoCapture(0)
    output_dir = "detection_results"
    os.makedirs(output_dir, exist_ok=True)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        prediction, confidence = predict(pil_image)
        label = f"{'No Fire/Smoke' if prediction == 1 else 'Fire/Smoke Detected'} (Confidence: {confidence:.2f})"
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Real-Time Fire/Smoke Detection', frame)

        # Save frames with fire/smoke detection
        if prediction == 1:
            cv2.imwrite(os.path.join(
                output_dir, f"frame_{frame_count}.jpg"), frame)
            frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Function to upload image


def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        prediction, confidence = predict(image)
        result_label.config(
            text=f"Prediction: {'No Fire/Smoke' if prediction == 1 else 'Fire/Smoke Detected'} (Confidence: {confidence:.2f})")
        image = image.resize((300, 300))
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo


# Create the main window
root = tk.Tk()
root.title("Fire and Smoke Detection")

# Create buttons
camera_button = tk.Button(root, text="Open Camera", command=open_camera)
camera_button.pack(pady=20)

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=20)

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

# Run the application
root.mainloop()
