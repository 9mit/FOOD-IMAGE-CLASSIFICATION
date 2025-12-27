"""
Food Image Classification - Streamlit Web Application

A production-ready web interface for food classification.
Model: ResNet-18 (87.26% accuracy on 20 food classes)

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import os
from PIL import Image
import pandas as pd
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference.predict import FoodClassifier, FOOD_CLASSES

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="🍛 Food Image Classifier",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main { padding: 2rem; }
    
    .title-container {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .title-container h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
    .title-container p { font-size: 1.1rem; opacity: 0.9; }
    
    .prediction-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-class {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .confidence-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .high-confidence { background-color: #27ae60; color: white; }
    .medium-confidence { background-color: #f39c12; color: white; }
    .low-confidence { background-color: #e74c3c; color: white; }
    
    .stats-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .stats-value { font-size: 1.8rem; font-weight: bold; color: #667eea; }
    .stats-label { font-size: 0.9rem; color: #7f8c8d; }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #7f8c8d;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.markdown("""
<div class="title-container">
    <h1>🍛 Food Image Classifier</h1>
    <p>AI-powered food recognition for 20 Indian & international cuisines</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# MODEL STATS
# ============================================
col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1:
    st.markdown("""
    <div class="stats-card">
        <div class="stats-value">87.3%</div>
        <div class="stats-label">Test Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col_s2:
    st.markdown("""
    <div class="stats-card">
        <div class="stats-value">20</div>
        <div class="stats-label">Food Classes</div>
    </div>
    """, unsafe_allow_html=True)

with col_s3:
    st.markdown("""
    <div class="stats-card">
        <div class="stats-value">6,269</div>
        <div class="stats-label">Training Images</div>
    </div>
    """, unsafe_allow_html=True)

with col_s4:
    st.markdown("""
    <div class="stats-card">
        <div class="stats-value">ResNet-18</div>
        <div class="stats-label">Model Architecture</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.header("⚙️ Settings")
    
    top_k = st.slider("Top predictions to show", 3, 10, 5)
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
    
    st.divider()
    
    st.header("📊 Model Performance")
    st.write("**Per-Class F1 Scores (Top 5):**")
    top_classes = [
        ("Burger", "98%"),
        ("Jalebi", "97%"),
        ("Fried Rice", "94%"),
        ("Dhokla", "92%"),
        ("Chole Bhature", "91%")
    ]
    for food, score in top_classes:
        st.write(f"• {food}: **{score}**")
    
    st.divider()
    
    st.header("🍽️ Supported Foods")
    
    categories = {
        "🍞 Breads": ["Butter Naan", "Chapati", "Masala Dosa"],
        "🥟 Snacks": ["Samosa", "Pakode", "Momos", "Dhokla"],
        "🍛 Curries": ["Dal Makhani", "Kadai Paneer", "Chole Bhature"],
        "🍨 Sweets": ["Jalebi", "Kulfi"],
        "🌏 Others": ["Pizza", "Burger", "Fried Rice", "Chai", "Idli"]
    }
    
    for cat, foods in categories.items():
        with st.expander(cat):
            for f in foods:
                st.write(f"• {f}")
    
    st.divider()
    
    st.header("ℹ️ About")
    st.write("""
    **Model:** ResNet-18 with Transfer Learning
    
    **Training:** Kaggle (Tesla T4 GPU)
    
    **Dataset:** 6,269 food images
    
    **Accuracy:** 87.26% test, 89.80% validation
    """)

# ============================================
# MAIN CONTENT
# ============================================

@st.cache_resource
def load_classifier():
    try:
        return FoodClassifier()
    except FileNotFoundError:
        return None

classifier = load_classifier()

if classifier is None:
    st.error("⚠️ Model not found. Please ensure model weights exist in `ml_models/` folder.")
    st.info("Download trained weights from Kaggle or train using: `python models/resnet18.py`")
    st.stop()

# File uploader
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose a food image...",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Upload a clear image of food for classification"
    )
    
    # Sample images option
    st.write("**Or try a sample:**")
    
    sample_foods = ['pizza', 'samosa', 'chai', 'idli', 'burger']
    sample_cols = st.columns(len(sample_foods))
    
    selected_sample = None
    for i, food in enumerate(sample_foods):
        with sample_cols[i]:
            if st.button(food.title(), key=f"sample_{food}"):
                sample_dir = f"data/test/{food}"
                if os.path.exists(sample_dir):
                    images = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                    if images:
                        selected_sample = os.path.join(sample_dir, images[0])

# Determine which image to process
image_to_process = None

if uploaded_file is not None:
    image_to_process = Image.open(uploaded_file).convert('RGB')
elif selected_sample is not None:
    image_to_process = Image.open(selected_sample).convert('RGB')

# Process and display results
if image_to_process is not None:
    with col1:
        st.image(image_to_process, caption="Input Image", use_container_width=True)
    
    with st.spinner("🔍 Analyzing image..."):
        result = classifier.predict(image_to_process, top_k=top_k)
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        predicted_class = result['predicted_class'].replace('_', ' ').title()
        confidence = result['confidence']
        
        if confidence >= 0.8:
            conf_class = "high-confidence"
            conf_emoji = "✅"
        elif confidence >= 0.5:
            conf_class = "medium-confidence"
            conf_emoji = "⚠️"
        else:
            conf_class = "low-confidence"
            conf_emoji = "❓"
        
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-class">{predicted_class}</div>
            <span class="confidence-badge {conf_class}">
                {conf_emoji} {confidence:.1%} Confidence
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        if confidence < confidence_threshold:
            st.warning(f"⚠️ Low confidence ({confidence:.1%}). Try a clearer image.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Top predictions
        st.subheader("📊 Top Predictions")
        
        for i, pred in enumerate(result['top_predictions'], 1):
            class_name = pred['class'].replace('_', ' ').title()
            prob = pred['confidence']
            
            col_rank, col_name, col_bar = st.columns([0.5, 2, 3])
            
            with col_rank:
                st.write(f"**#{i}**")
            with col_name:
                st.write(class_name)
            with col_bar:
                st.progress(prob, text=f"{prob:.1%}")

else:
    with col2:
        st.info("👆 Upload an image or click a sample button to classify food!")
        
        st.subheader("📈 Model Performance")
        
        # Show accuracy comparison
        perf_data = pd.DataFrame({
            'Model': ['Baseline CNN', 'ResNet-18'],
            'Accuracy': [29.26, 87.26]
        })
        
        st.bar_chart(perf_data.set_index('Model'), height=200)
        
        st.success("✅ ResNet-18 achieves **87.26%** accuracy with transfer learning!")
        
        st.subheader("🏆 Best Performing Classes")
        
        best_classes = {
            'Burger': 98, 'Jalebi': 97, 'Fried Rice': 94,
            'Dhokla': 92, 'Pakode': 91, 'Idli': 90
        }
        
        st.bar_chart(pd.Series(best_classes), height=200)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🧠 ResNet-18 Transfer Learning | 📊 87.26% Accuracy | 🍽️ 20 Food Classes</p>
    <p>Trained on Kaggle (Tesla T4 GPU) | Made with ❤️ using PyTorch & Streamlit</p>
</div>
""", unsafe_allow_html=True)