from flask import Flask, render_template, request
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import FireSmokeDetector
import os

app = Flask(__name__)

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


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", result="No file uploaded")
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", result="No file selected")
        if file:
            image = Image.open(file.stream)
            prediction, confidence = predict(image)
            result = f"{'No Fire/Smoke' if prediction == 1 else 'Fire/Smoke Detected'} (Confidence: {confidence:.2f})"
            return render_template("index.html", result=result)
    return render_template("index.html", result=None)


if __name__ == "__main__":
    app.run(debug=True)
