# 🧠 Brain Tumor Analysis

A multimodal AI application for brain tumor detection and segmentation using deep learning models.

## 🎯 Features

- **Brain Tumor Classification**: ResNet50-based model for tumor detection (Tumor vs No Tumor)
- **Brain Tumor Segmentation**: U-Net model for precise tumor area segmentation
- **Streamlit Web Interface**: User-friendly web application
- **Real-time Analysis**: Upload TIF images and get instant results
- **Download Results**: Save segmented images for further analysis

## 🏗️ Architecture

### Classification Model
- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Input Size**: 224x224x3
- **Output**: Binary classification (Tumor/No Tumor)
- **Framework**: TensorFlow 2.12

### Segmentation Model
- **Architecture**: U-Net
- **Input Size**: 128x128x3
- **Output**: Binary mask (tumor segmentation)
- **Framework**: TensorFlow 2.12

## 📁 Project Structure

```
Brain Tumor Classification & Segmentation/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore rules
├── models/                         # Trained models
│   ├── classification/
│   │   └── brain_Tumor_model_v3.h5
│   └── segmentation/
│       └── unet_128_mri_seg.hdf5
├── notebooks/                      # Jupyter notebooks
│   ├── classification/
│   │   └── brain_tumor_classification.ipynb
│   └── segmentation/
│       └── brain-mri-tumor-segmentation-keras-unet.ipynb
└── brain_tumor_env/                # Virtual environment
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- TensorFlow 2.12.0
- Streamlit 1.28.1

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Brain Tumor Classification & Segmentation"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv brain_tumor_env
   ```

3. **Activate virtual environment**
   ```bash
   # Windows
   brain_tumor_env\Scripts\activate
   
   # Linux/Mac
   source brain_tumor_env/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open in browser**
   - Navigate to `http://localhost:8501`
   - Upload a TIF brain MRI image
   - Click "Analyze Image" to get results

## 📊 Usage

1. **Upload Image**: Select a TIF format brain MRI image
2. **Analyze**: Click the "🔍 Analyze Image" button
3. **View Results**: 
   - Classification result (Tumor/No Tumor with confidence)
   - Segmentation overlay (red highlighting)
   - Tumor area percentage and pixel count
4. **Download**: Save the segmented image for further analysis

## 🔧 Technical Details

### Model Specifications

#### Classification Model
- **Architecture**: ResNet50 + Custom layers
- **Input**: 224x224x3 RGB images
- **Preprocessing**: ResNet50 preprocess_input
- **Output**: 2 classes (Tumor, No Tumor)

#### Segmentation Model
- **Architecture**: U-Net
- **Input**: 128x128x3 RGB images
- **Preprocessing**: Normalize to [0,1] range
- **Output**: Binary mask (sigmoid activation)

### Dependencies

```
streamlit==1.28.1
tensorflow==2.12.0
keras==2.12.0
numpy==1.23.5
opencv-python==4.8.1.78
Pillow==10.1.0
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
h5py==3.9.0
protobuf==3.20.3
```

## 📈 Results

The application provides:
- **Classification accuracy**: Tumor detection with confidence scores
- **Segmentation precision**: Detailed tumor area mapping
- **Visual analysis**: Side-by-side comparison of original and segmented images
- **Quantitative metrics**: Tumor area percentage and pixel count


## 🙏 Acknowledgments

- TensorFlow and Keras for deep learning framework
- Streamlit for web application framework
- Medical imaging community for datasets and research

---

**Note**: This application is for research and educational purposes. Always consult medical professionals for clinical decisions. 
