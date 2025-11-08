🧼 Clean Vision
Deep Learning-Based Waste Classification for Smart Waste Management

Clean Vision is a deep learning–powered waste classification system designed to enhance smart waste management through automated image-based sorting.
The model classifies waste into Organic and Recyclable categories using modern CNN architectures trained on a large public dataset.

📊 Dataset

Source: Waste Classification Data (Kaggle)

Total Images: 22,000+
Categories: Organic, Recyclable

Preprocessing Steps

Image resizing to 224×224

Normalization

Data augmentation (rotation, flip, zoom, shift)

🧠 Models Trained
Model	Accuracy	Epochs	Notes
EfficientNet-B0	97.2%	30–50	Best performing model
MobileNetV2	95.6%	30–50	Lightweight, fast on edge devices
ResNet50	92.4%	30–50	Stable baseline model

All models were trained using NVIDIA GeForce RTX 3050 GPU
Optimization: Adam optimizer, Early stopping, and Learning rate scheduling
Achieved high accuracy with limited overfitting

⚙️ Features

✅ High-accuracy waste classification

🧩 Multiple deep learning models implemented

💻 Easy deployment on edge devices (ESP32-CAM, Raspberry Pi, Jetson Nano)

🧱 Well-structured training and evaluation pipeline

⚡ Supports real-time prediction

🧠 Modular codebase (data pipeline, training, evaluation, inference)

🚀 Installation
git clone https://github.com/skyaseen005/Clean-Vision.git
cd Clean-Vision
pip install -r requirements.txt

📈 Results
EfficientNet-B0 – Best Model (97.2%)

Excellent generalization capability

Works efficiently for deployment

Balanced accuracy and speed

Confusion Matrices, Accuracy & Loss Curves:

<p align="center"> <img width="505" height="470" alt="res-1" src="https://github.com/user-attachments/assets/e09d4638-9b08-4c25-ae38-223f3cffb0cb" /> <img width="505" height="470" alt="mbnet-1" src="https://github.com/user-attachments/assets/e8d0ee4f-764e-4dac-97b1-be6c2bb51126" /> <img width="505" height="470" alt="efficient-1" src="https://github.com/user-attachments/assets/ef876a99-0b05-46ed-b76f-09accc7c05f4" /> </p> </br> <p align="center"> <img width="336" height="252" alt="accuracy-graph" src="https://github.com/user-attachments/assets/f7ffbbb5-d23a-411e-9574-6a04f8164117" /> </p>

Final Accuracy Graph

🧰 Technologies Used

Python

TensorFlow / Keras

NumPy, Pandas

Matplotlib, Seaborn

Scikit-Learn

CUDA + cuDNN

🏗️ Future Improvements

🌐 Integration with IoT for smart dustbins

⚙️ Deployment on embedded hardware

♻️ Expansion to multi-class waste types (metal, plastic, glass, etc.)

🔍 Real-time object detection using YOLO / EfficientDet

⭐ If you like this project, don’t forget to star the repository!

Developed with 💚 for a cleaner, smarter planet.
