import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras import regularizers

# --- Model Loading ---
@st.cache_resource
def load_models():
    # Load Classification Model
    def load_classification_model():
        try:
            # First try to load the model directly
            classification_model = tf.keras.models.load_model(
                'models/classification/brain_Tumor_model_v3.h5',
                compile=False
            )
            return classification_model, "success"
        except Exception as e:
            try:
                # Recreate the exact model architecture from the notebook
                base = ResNet50(include_top=False, weights="imagenet", input_shape=(224,224,3), pooling="max")
                
                # Freeze the base model layers
                for layer in base.layers:
                    layer.trainable = False
                
                # Create the exact model architecture
                model = Sequential([
                    base,
                    Flatten(),
                    Dense(224, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
                    Dropout(0.3),
                    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
                    Dropout(0.2),
                    Dense(2, activation="softmax")
                ])
                
                # Try to load weights
                try:
                    model.load_weights('models/classification/brain_Tumor_model_v3.h5')
                    return model, "success"
                except Exception as weight_error:
                    return model, "no_weights"
                    
            except Exception as recreate_error:
                return None, "failed"
    
    # Load Segmentation Model
    def load_segmentation_model():
        def dice_coef(y_true, y_pred):
            smooth = 100
            y_true = tf.keras.backend.flatten(y_true)
            y_pred = tf.keras.backend.flatten(y_pred)
            intersec = tf.keras.backend.sum(y_true * y_pred)
            mod_sum = tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred)
            return (2 * intersec + smooth) / (mod_sum + smooth)
        def dice_coef_loss(y_true, y_pred):
            return -dice_coef(y_true, y_pred)
        def iou(y_true, y_pred):
            smooth = 100
            intersec = tf.keras.backend.sum(y_true * y_pred)
            comb_area = tf.keras.backend.sum(y_true + y_pred) - intersec
            return (intersec + smooth) / (comb_area + smooth)
        def iou_loss(y_true, y_pred):
            return -iou(y_true, y_pred)
        
        try:
            segmentation_model = tf.keras.models.load_model(
                'models/segmentation/unet_128_mri_seg.hdf5',
                custom_objects={
                    'dice_coef': dice_coef,
                    'dice_coef_loss': dice_coef_loss,
                    'iou': iou,
                    'iou_loss': iou_loss
                },
                compile=False
            )
            return segmentation_model, "success"
        except Exception as e:
            return None, "failed"
    
    return load_classification_model(), load_segmentation_model()

# --- UI ---
st.set_page_config(page_title="Brain Tumor Analysis", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    /* Remove white bars and improve overall design */
    .block-container {
        padding-top: 10rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }
    
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Enhanced title */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5em;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtext {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2em;
        font-weight: 400;
    }
    
    /* Enhanced button */
    .stButton > button {
        font-size: 1.2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 25px;
        padding: 0.8rem 3rem;
        color: white;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
        max-width: 300px;
        margin: 0 auto;
        display: block;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Enhanced result boxes */
    .result-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    .result-box h3 {
        color: #333;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .result-box p {
        font-size: 1.1rem;
        color: #555;
        margin: 0.5rem 0;
    }
    
    /* Enhanced image containers */
    .image-container {
        background: white;
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    
    .image-container h4 {
        color: #333;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Prediction highlight */
    .prediction-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .prediction-highlight h2 {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }
    
    .prediction-highlight p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Enhanced sidebar */
    .css-1d391kg {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Remove extra spacing */
    .stMarkdown {
        margin-bottom: 0;
    }
    
    /* Enhanced file uploader */
    .stFileUploader {
        background: #123;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Enhanced download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 15px;
        padding: 0.8rem 2rem;
        color: white;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧠 Brain Tumor Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Advanced AI-powered brain MRI analysis for tumor detection and segmentation</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white;">
        <h3 style="color: white; margin-bottom: 1rem;">📋 Instructions</h3>
        <ol style="color: white; font-size: 0.9rem;">
            <li>Upload a brain MRI image (TIF format)</li>
            <li>Click <strong>Analyze Image</strong></li>
            <li>View results side by side</li>
            <li>Download the segmented image</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 15px; margin-top: 1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h4 style="color: #333; margin-bottom: 1rem;">🔬 Models</h4>
        <ul style="color: #555; font-size: 0.9rem;">
            <li><strong>Classification:</strong> ResNet50 (224x224)</li>
            <li><strong>Segmentation:</strong> U-Net (128x128)</li>
            <li><strong>Framework:</strong> TensorFlow 2.12</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- Image Upload ---
uploaded_file = st.file_uploader("📁 Upload MRI Image (TIF only)", type=["tif", "tiff"]) 

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # --- Preprocess for both models ---
    img_arr = np.array(image)
    
    # For Classification (224x224, ResNet50 preprocessing)
    img_clf = cv2.resize(img_arr, (224, 224))
    img_clf = preprocess_input(img_clf)
    img_clf = np.expand_dims(img_clf, axis=0)
    
    # For Segmentation (128x128, normalize to [0,1])
    img_seg = cv2.resize(img_arr, (128, 128))
    img_seg_norm = img_seg.astype(np.float32) / 255.0
    img_seg_norm = np.expand_dims(img_seg_norm, axis=0)
    
    # --- Analyze Button ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("Running classification and segmentation..."):
                # Load models
                (classification_model, clf_status), (segmentation_model, seg_status) = load_models()
                
                if segmentation_model is None:
                    st.error("Failed to load segmentation model. Please check the model files.")
                    st.stop()
                
                # --- Run Classification (if available) ---
                classification_result = None
                if clf_status in ["success", "no_weights"] and classification_model is not None:
                    try:
                        pred_clf = classification_model.predict(img_clf)
                        label_map = {0: "Tumor", 1: "No Tumor"}
                        predicted_label = np.argmax(pred_clf)
                        confidence = np.max(pred_clf) * 100
                        classification_result = {
                            "label": label_map[predicted_label],
                            "confidence": confidence,
                            "status": clf_status
                        }
                    except Exception as e:
                        classification_result = None
                
                # --- Run Segmentation ---
                pred_mask = segmentation_model.predict(img_seg_norm)[0]
                pred_mask = (pred_mask > 0.5).astype(np.uint8)
                
                # --- Create Red Overlay ---
                red_mask = np.zeros_like(img_seg)
                red_mask[:, :, 2] = pred_mask.squeeze() * 255  # Red channel only
                
                # --- Blend Original with Red Overlay ---
                overlay = cv2.addWeighted(img_seg, 1.0, red_mask, 0.5, 0)
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                overlay_img = Image.fromarray(overlay_rgb)
                
                # --- Tumor Area Calculation ---
                tumor_pixels = np.sum(pred_mask > 0)
                total_pixels = pred_mask.size
                tumor_percent = (tumor_pixels / total_pixels) * 100
                
                # --- Display Results ---
                st.markdown("### 📊 Prediction Results")
                
                # Enhanced Prediction Display
                if classification_result:
                    if classification_result["status"] == "no_weights":
                        st.markdown(f"""
                        <div class="prediction-highlight">
                            <h2>⚠️ {classification_result['label']}</h2>
                            <p>Confidence: {classification_result['confidence']:.2f}% (Random Prediction)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-highlight">
                            <h2>🎯 {classification_result['label']}</h2>
                            <p>Confidence: {classification_result['confidence']:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="prediction-highlight" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);">
                        <h2>❌ Model Unavailable</h2>
                        <p>Classification model failed to load</p>
                    </div>
                    """, unsafe_allow_html=True)
                

                
                # Images side by side (smaller size)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📸 Original Image**")
                    st.image(image, width=300, use_container_width=True)
                
                with col2:
                    st.markdown("**🎯 Segmented Image (Red Overlay)**")
                    st.image(overlay_img, width=300, use_container_width=True)
                
                # --- Download Button ---
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.download_button(
                        "💾 Download Segmented Image", 
                        data=overlay_img.tobytes(), 
                        file_name="brain_tumor_analysis.tif", 
                        mime="image/tiff",
                        use_container_width=True
                    )
else:
    st.info("📁 Please upload a brain MRI image in TIF format to begin.") 