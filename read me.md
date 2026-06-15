Markdown# Face Ancestry Recognition System 🧬🤖

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.0-red.svg)](https://keras.io/)
[![GPU Training](https://img.shields.io/badge/Computed%20On-NVIDIA%20T4%20GPU-green.svg)](https://cloud.google.com/)

An end-to-end Computer Vision and Deep Learning framework designed to analyze facial geometry and classify ancestral backgrounds into 7 distinct global regions. This system utilizes a lightweight, high-efficiency Convolutional Neural Network (CNN) optimized for deployment on both cloud infrastructure and local desktop systems.

---

## 🚀 Key Features

- **Large-Scale Data Pipeline:** Capable of processing a comprehensive dataset consisting of **86,744 validated facial images**.
- **State-of-the-Art Architecture:** Implements **MobileNetV2** via Transfer Learning, capitalizing on pre-trained ImageNet weights for optimized feature extraction.
- **Hardware-Accelerated Training:** Conducted via Google Colab utilizing a cloud-hosted **NVIDIA Tesla T4 GPU** featuring XLA (Accelerated Linear Algebra) compilation.
- **Robust Data Augmentation:** Real-time data pipeline generation using `ImageDataGenerator` with scaling, rotation, and horizontal flips to reduce overfitting.
- **Interactive Local Desktop GUI:** Features a custom UI automation layer using `Tkinter` to facilitate native OS file dialog popups for effortless image testing.

---

## 📊 Dataset & Target Classes

The model acts as a multi-class classification network mapping inputs across **7 distinct ancestral lineages**:

| Class ID | Target Ancestry Label |
| :---: | :--- |
| `0` | Black |
| `1` | East Asian |
| `2` | Indian |
| `3` | Latino / Hispanic |
| `4` | Middle Eastern |
| `5` | Southeast Asian |
| `6` | White |

---

## 🛠️ Tech Stack & Frameworks

- **Core Language:** Python 3.11
- **Deep Learning Ecosystem:** TensorFlow 2.x, Keras 3
- **Data Engineering & Math:** NumPy, Pandas, SciPy
- **Image Processing Engine:** Pillow (PIL), OpenCV-Python
- **UI Engine:** Native Tkinter (OS-level Dialog Modules)

---

## 🗂️ Project Directory Structure

```text
Face_Ancestry_Project/
│
├── data/
│   ├── train/                       # Directory containing raw image extractions
│   └── train_labels.csv             # Structured mapping file metadata
│
├── models/
│   └── face_ancestry_model.keras    # Fully compiled serialization file (Keras Artifact)
│
├── src/
│   ├── data_loader.py               # Pre-processing & Augmentation Engine
│   ├── model.py                     # Neural Network Backbone Custom Defs
│   ├── train.py                     # Training Routine and Loss Minimization Pipeline
│   ├── predict.py                   # Standard local static prediction script
│   └── predict_select.py            # Interactive OS File Dialog Testing Demo
│
└── README.md                        # Documentation Blueprint
📈 Model Performance & Training InsightsOptimization Strategy: Categorical Cross-Entropy loss minimization executed via the Adam optimizer initialized at a learning rate of $\alpha = 0.001$.Dimensions: Input spatial matrices downscaled dynamically to 224x224x3 channels to balance computational load and geometric feature fidelity.Evaluation Summary: The system displayed steady descent in convergence loss values, registering a structural training baseline accuracy of ~40.5% over highly diverse global facial topologies.⚙️ Local Setup and Execution Guide1. Clone the Repository & Configure DependenciesEnsure you are running Python 3.11 environments. Open your terminal inside the project directory and execute:Bashpip install tensorflow scipy numpy pandas pillow
2. Verify Model Artifact PlacementEnsure your downloaded weight matrix file face_ancestry_model.keras is placed perfectly inside the models/ subdirectory directory path.3. Launch Interactive Native Inference TestingTo run live evaluations on any customized portrait image stored on your local computer machine without writing code, execute:Bashpython src/predict_select.py
A native Windows/OS explorer window will prompt dynamically. Choose any image file (.jpg, .png, .jpeg) to trigger the feature scanning mapping matrix instantly.🤝 ContributingContributions, structural optimization issues, or deep learning architectural pull requests aimed at boosting evaluation parameters are welcome. Feel free to open an issue or fork the development framework.Developed with Precision by Mohsin Atta Department of Data Science, The Islamia University of Bahawalpur ```
