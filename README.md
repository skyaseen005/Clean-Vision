# 🌍 Clean Vision

<div align="center">

![Clean Vision Banner](https://img.shields.io/badge/Clean_Vision-Waste_Classification-green?style=for-the-badge)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg?style=flat&logo=kaggle)](https://www.kaggle.com/)

**Deep Learning-Powered Waste Classification for Smart Waste Management**

[Features](#-features) • [Installation](#-installation) • [Dataset](#-dataset) • [Models](#-models) • [Results](#-results) • [Future Work](#-future-improvements)

</div>

---

## 📋 Overview

**Clean Vision** is an advanced deep learning–powered waste classification system designed to revolutionize smart waste management through automated image-based sorting. Using state-of-the-art CNN architectures, the system accurately classifies waste into **Organic** and **Recyclable** categories, enabling efficient waste segregation and promoting environmental sustainability.

## ✨ Features

- 🎯 **High-Accuracy Classification** – Achieves up to 97.2% accuracy
- 🧠 **Multiple Deep Learning Models** – EfficientNet-B0, MobileNetV2, ResNet50
- 🚀 **Edge Device Ready** – Optimized for ESP32-CAM, Raspberry Pi, Jetson Nano
- ⚡ **Real-Time Prediction** – Fast inference for live applications
- 🔧 **Modular Architecture** – Clean separation of data pipeline, training, evaluation, and inference
- 📊 **Comprehensive Evaluation** – Detailed accuracy/loss curves and confusion matrices
- 🔄 **Data Augmentation** – Robust preprocessing for improved generalization

## 📊 Dataset

- **Source**: [Waste Classification Data (Kaggle)](https://www.kaggle.com/)
- **Total Images**: 22,000+
- **Categories**: 
  - 🍃 Organic Waste
  - ♻️ Recyclable Waste

### Preprocessing Pipeline

```
✓ Image resizing to 224×224 pixels
✓ Normalization (pixel value scaling)
✓ Data augmentation (rotation, flip, zoom, shift)
✓ Train/validation/test split
```

## 🤖 Models

### Model Performance Comparison

| Model | Accuracy | Epochs | Key Characteristics |
|-------|----------|--------|---------------------|
| **EfficientNet-B0** | **97.2%** | 30–50 | 🏆 Best performing model |
| **MobileNetV2** | **95.6%** | 30–50 | ⚡ Lightweight, fast on edge devices |
| **ResNet50** | **92.4%** | 30–50 | 🎯 Stable baseline model |

### Training Configuration

- **Hardware**: NVIDIA GeForce RTX 3050 GPU
- **Optimizer**: Adam
- **Techniques**: 
  - Early stopping
  - Learning rate scheduling
  - Regularization to prevent overfitting

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/skyaseen005/Clean-Vision.git

# Navigate to project directory
cd Clean-Vision

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```python
# Example usage
from clean_vision import WasteClassifier

# Initialize classifier
classifier = WasteClassifier(model='efficientnet')

# Predict waste type
result = classifier.predict('path/to/image.jpg')
print(f"Waste Type: {result['category']} (Confidence: {result['confidence']:.2%})")
```

## 📈 Results

### EfficientNet-B0 – Best Model (97.2%)

✅ Excellent generalization capability  
✅ Efficient for deployment scenarios  
✅ Balanced accuracy and inference speed  

### Performance Visualizations

<div align="center">

#### Confusion Matrices

| ResNet50 | MobileNetV2 | EfficientNet-B0 |
|----------|-------------|-----------------|
| ![res-1](res-1) | ![mbnet-1](mbnet-1) | ![efficient-1](efficient-1) |

#### Training History

![Accuracy Graph](accuracy-graph)

*Accuracy and Loss curves showing model convergence and minimal overfitting*

</div>

## 🔧 Technologies Used

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

</div>

## 🚀 Future Improvements

- [ ] 🌐 **IoT Integration** – Connect with smart dustbins for automated sorting
- [ ] ⚙️ **Embedded Deployment** – Optimize for ESP32-CAM, Raspberry Pi, Jetson Nano
- [ ] ♻️ **Multi-Class Expansion** – Add categories: metal, plastic, glass, paper, e-waste
- [ ] 🔍 **Object Detection** – Implement YOLO/EfficientDet for real-time detection
- [ ] 📱 **Mobile Application** – Develop Android/iOS app for on-the-go classification
- [ ] 🌍 **Cloud API** – Deploy as RESTful API for scalable access
- [ ] 📊 **Analytics Dashboard** – Track waste trends and environmental impact

## 📁 Project Structure

```
Clean-Vision/
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
├── models/
│   ├── efficientnet_model.h5
│   ├── mobilenet_model.h5
│   └── resnet_model.h5
├── src/
│   ├── data_pipeline.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── notebooks/
│   └── exploration.ipynb
├── requirements.txt
├── README.md
└── LICENSE
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Sky Aseen**

- GitHub: [@skyaseen005](https://github.com/skyaseen005)
- Project Link: [https://github.com/skyaseen005/Clean-Vision](https://github.com/skyaseen005/Clean-Vision)

## 🙏 Acknowledgments

- Kaggle for providing the waste classification dataset
- TensorFlow and Keras teams for excellent deep learning frameworks
- The open-source community for inspiration and support

---

<div align="center">

### ⭐ If you find this project useful, please consider giving it a star!

**Developed with 💚 for a cleaner, smarter planet**

![Footer](https://img.shields.io/badge/Made%20with-Love%20%26%20Code-red?style=for-the-badge)

</div>
