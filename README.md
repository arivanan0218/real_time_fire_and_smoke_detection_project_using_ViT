# Real-Time Fire and Smoke Detection using Vision Transformers (ViT)

This project implements a **real-time fire and smoke detection system** using **Vision Transformers (ViT)**. The system is designed to detect fire and smoke in real-time video streams or images, making it suitable for applications like surveillance, disaster management, and safety monitoring.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Installation](#installation)
5. [Dataset Preparation](#dataset-preparation)
6. [Training the Model](#training-the-model)
7. [Running the GUI](#running-the-gui)
8. [Deployment with Flask](#deployment-with-flask)
9. [Folder Structure](#folder-structure)
10. [Contributing](#contributing)

---

## Project Overview

The goal of this project is to build a **real-time fire and smoke detection system** using **Vision Transformers (ViT)**. The system can:

- Detect fire and smoke in real-time video streams.
- Classify images as containing fire/smoke or not.
- Save frames with detected fire/smoke for further analysis.
- Provide a user-friendly GUI for interaction.

---

## Features

- **Real-Time Detection**: Detects fire and smoke in real-time video streams.
- **Image Classification**: Classifies uploaded images as containing fire/smoke or not.
- **Confidence Scores**: Displays confidence scores for predictions.
- **Save Results**: Saves frames with detected fire/smoke to a folder.
- **GUI Interface**: Provides a user-friendly interface for real-time detection and image upload.
- **Web Deployment**: Can be deployed as a web application using Flask.

---

## Technologies Used

- **Python**: Primary programming language.
- **PyTorch**: Deep learning framework for building and training the model.
- **Transformers**: Library for using pre-trained Vision Transformers (ViT).
- **OpenCV**: For real-time video processing.
- **Tkinter**: For building the GUI.
- **Flask**: For deploying the model as a web application.
- **Pillow**: For image processing.

---

## Installation

Follow these steps to set up the project on your local machine.

### Prerequisites

- Python 3.8+
- GPU (recommended for faster training and inference)

### Steps

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/arivanan0218/real_time_fire_and_smoke_detection_project_using_ViT.git
   cd real-time-fire-smoke-detection
   ```

2. **Create a Virtual Environment**:

   ```bash
   python -m venv fire_smoke_env
   fire_smoke_env\Scripts\activate
   ```

3. **Install Required Libraries**:
   ```bash
   pip install torch torchvision transformers opencv-python pillow tkinter flask
   ```

---

## Dataset Preparation

1. **Download the Dataset**:

   - Use a dataset like [Fire Detection Dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset) from Kaggle.

2. **Organize the Dataset**:

   - Create the following folder structure:
     ```
     dataset/
       train/
         fire/
         non_fire/
       val/
         fire/
         non_fire/
     ```

3. **Preprocess the Dataset**:
   - Resize images to 224x224 (ViT input size).
   - Normalize images using mean and standard deviation for RGB channels.

---

## Training the Model

1. **Train the Model**:

   - Run the `train.py` script to train the ViT model:
     ```bash
     python train.py
     ```

2. **Model Output**:
   - The trained model weights will be saved as `fire_smoke_detector.pth`.

---

## Running the GUI

1. **Run the GUI**:

   - Use the `gui.py` script to launch the real-time fire and smoke detection GUI:
     ```bash
     python gui.py
     ```

2. **Features**:
   - **Open Camera**: Start real-time fire/smoke detection using your webcam.
   - **Upload Image**: Upload an image for fire/smoke classification.
   - **Save Results**: Frames with detected fire/smoke are saved in the `detection_results` folder.

---

## Deployment with Flask

1. **Run the Flask App**:

   - Use the `app.py` script to deploy the model as a web application:
     ```bash
     python app.py
     ```

2. **Access the Web App**:
   - Open your browser and navigate to `http://127.0.0.1:5000/`.
   - Upload an image to classify it as containing fire/smoke or not.

---

## Folder Structure

```
real_time_fire_and_smoke_detection_project_using_ViT/
├── dataset/                  # Dataset for training and validation
│   ├── train/
│   │   ├── fire/
│   │   └── non_fire/
│   └── val/
│       ├── fire/
│       └── non_fire/
├── detection_results/        # Saved frames with detected fire/smoke
├── model.py                 # ViT model definition
├── train.py                 # Script to train the model
├── gui.py                   # GUI for real-time detection
├── app.py                   # Flask app for web deployment
├── templates/               # HTML templates for Flask app
│   └── index.html
├── fire_smoke_detector.pth  # Trained model weights
└── README.md                # Project documentation
```

---

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Submit a pull request.

---

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the pre-trained ViT model.
- [Kaggle](https://www.kaggle.com/) for the fire detection dataset.

---

## Contact

For any questions or feedback, feel free to reach out:

- **Arivanan V.**: [Your Email](mailto:vamathevanarivanan@gmail.com)
- **GitHub**: [Your GitHub Profile](https://github.com/arivanan0218)
"# real_time_fire_and_smoke_detection_project_using_ViT" 
