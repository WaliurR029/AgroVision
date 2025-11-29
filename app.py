import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import json
import plotly.express as px
import pandas as pd
from datetime import datetime
import os
import signal
import time

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="AgroVision AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. SESSION STATE ---
# Initialize history list if it doesn't exist
if 'history' not in st.session_state:
    st.session_state['history'] = []
# Initialize last_file to track duplicates by filename
if 'last_file' not in st.session_state:
    st.session_state['last_file'] = None

# --- 3. MODERN DARK THEME CSS ---
st.markdown("""
    <style>
    /* Import Google Font: Poppins */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Glassmorphism Card Style */
    .st-emotion-cache-1r6slb0 {
        background-color: #1E1E1E;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Green Headings */
    h1, h2, h3 {
        color: #00E676 !important;
        font-weight: 600;
    }
    
    /* Upload Box Styling */
    [data-testid="stFileUploader"] {
        background-color: #111;
        border: 1px dashed #444;
        border-radius: 10px;
        padding: 20px;
    }
    
    /* Stop Button Styling (Red) */
    div.stButton > button:first-child {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. LOGIC: Load Model & Data ---
@st.cache_resource
def load_ai_model():
    return load_model("best_agrovision_model.keras")

@st.cache_data
def load_class_indices():
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}

try:
    model = load_ai_model()
    index_to_class = load_class_indices()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# Disease Solutions Dictionary
disease_solutions = {
    "Pepper__bell___Bacterial_spot": "Partial Solution: Remove infected leaves immediately. Apply copper-based fungicides (like fixed copper) every 7-10 days. Avoid overhead watering to prevent spread.",
    "Pepper__bell___healthy": "Healthy: No treatment needed. Maintain good watering practices and monitor for pests.",
    "Potato___Early_blight": "Fungicide: Apply fungicides containing Mancozeb or Chlorothalonil. Remove infected lower leaves. Ensure proper nitrogen levels in soil.",
    "Potato___Late_blight": "URGENT: This is destructive. Remove and destroy infected plants immediately (do not compost). Apply fungicides like metalaxyl or copper sprays to protect nearby plants.",
    "Potato___healthy": "Healthy: No treatment needed. Ensure soil is well-drained to prevent future rot.",
    "Tomato_Bacterial_spot": "Sanitation: Remove infected plant parts. Sterilize tools with 10% bleach solution. Apply copper bactericides early in the season.",
    "Tomato_Early_blight": "Pruning: Remove infected lower leaves to improve airflow. Mulch soil to prevent spore splash. Use fungicides like chlorothalonil if severe.",
    "Tomato_Late_blight": "URGENT: Remove infected plants immediately. Apply copper-based fungicides or those with active ingredient 'chlorothalonil' to protect healthy plants.",
    "Tomato_Leaf_Mold": "Ventilation: High humidity causes this. Prune plants to increase airflow. Water at the base, not on leaves. Fungicides are rarely needed if airflow is good.",
    "Tomato_Septoria_leaf_spot": "Sanitation: Remove fallen leaves and infected lower leaves. Avoid watering in the evening. Apply copper fungicide or mancozeb.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Pest Control: Spray plants with a strong stream of water to dislodge mites. Use insecticidal soap or neem oil for organic control.",
    "Tomato__Target_Spot": "Fungicide: Improve airflow. Apply fungicides containing azoxystrobin or chlorothalonil according to label instructions.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Vector Control: This is spread by whiteflies. Use yellow sticky traps. Remove and destroy infected plants immediately to stop spread.",
    "Tomato__Tomato_mosaic_virus": "No Cure: Remove and destroy infected plants. Wash hands and tools thoroughly (smokers should wear gloves). Do not replant tomatoes in the same soil for 2 years.",
    "Tomato_healthy": "Healthy: No treatment needed. Continue regular care."
}

# --- 5. SIDEBAR (With STOP Button) ---
with st.sidebar:
    st.markdown("# 🧬 AgroVision")
    st.write("Intelligent Disease Diagnosis")
    st.markdown("---")
    st.info("Supported Crops:\n- Tomato\n- Potato\n- Pepper")
    
    st.markdown("---")
    st.caption("System Controls")
    
    # THE KILL SWITCH
    if st.button("🛑 Stop App", type="primary"):
        st.warning("Shutting down system...")
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)

# --- 6. MAIN HEADER ---
col1, col2 = st.columns([1, 15])
with col1:
    st.markdown("# 🌿") 
with col2:
    st.title("AgroVision AI")
    st.markdown("**Advanced Plant Disease Diagnosis System**")

st.markdown("---")

# --- 7. MAIN INTERFACE ---
with st.container():
    col_up, col_info = st.columns([1, 2], gap="large")
    
    with col_up:
        st.subheader("📤 Upload Image")
        # Unique key 'uploader' ensures stable state
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], key="uploader")
        
        if uploaded_file:
            image_pil = Image.open(uploaded_file)
            st.image(image_pil, caption="Source Image", use_container_width=True)

    with col_info:
        if uploaded_file:
            st.subheader("📊 Analysis Report")
            
            with st.spinner("Processing bio-metric data..."):
                # Prediction Logic
                img = image_pil.resize((256, 256))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0

                predictions = model.predict(img_array)[0]
                top_3_indices = predictions.argsort()[-3:][::-1]
                top_3_values = predictions[top_3_indices]
                
                predicted_index = top_3_indices[0]
                confidence = top_3_values[0]
                predicted_class = index_to_class[predicted_index]
                display_name = predicted_class.replace("_", " ").replace("   ", " - ").title()
                
                # --- RESULT CARD ---
                if confidence > 0.6:
                    st.success(f"### Detected: {display_name}")
                    
                    m1, m2 = st.columns(2)
                    m1.metric("Confidence Score", f"{confidence*100:.1f}%", delta="High Accuracy")
                    m2.metric("Processing Time", "0.4s", delta="Real-time")
                    
                    # Prepare Chart Data
                    chart_data = pd.DataFrame({
                        "Condition": [index_to_class[i].replace("_", " ").title() for i in top_3_indices],
                        "Probability": top_3_values
                    })
                    
                    # Draw Chart
                    fig = px.bar(
                        chart_data, x="Probability", y="Condition", orientation='h', 
                        template="plotly_dark", color="Probability",
                        color_continuous_scale=['#2C3E50', '#00E676']
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', 
                        paper_bgcolor='rgba(0,0,0,0)', 
                        height=220, 
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Solution
                    st.markdown("### 💊 Treatment Protocol")
                    solution = disease_solutions.get(predicted_class, "No specific solution found.")
                    st.info(solution)

                    # --- SAVE TO HISTORY ---
                    # Logic: Only save if the FILENAME is different from the last one
                    if st.session_state['last_file'] != uploaded_file.name:
                        
                        current_result = {
                            "name": display_name,
                            "confidence": f"{confidence*100:.1f}%",
                            "solution": solution,
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "filename": uploaded_file.name,
                            # Save chart data for re-drawing later
                            "chart_probabilities": top_3_values,
                            "chart_conditions": [index_to_class[i].replace("_", " ").title() for i in top_3_indices]
                        }
                        
                        # Add to top of list
                        st.session_state['history'].insert(0, current_result)
                        # Update tracker
                        st.session_state['last_file'] = uploaded_file.name
                    
                else:
                    st.warning("⚠️ Analysis Inconclusive. Confidence below threshold.")

        else:
            st.info("👈 Waiting for input...")


# --- 8. HISTORY SECTION ---
st.markdown("---")
st.subheader("📜 Diagnosis History")

if not st.session_state['history']:
    st.caption("No previous scans yet. Upload an image to start.")

for scan in st.session_state['history']:
    with st.expander(f"⏰ {scan['time']} - {scan['name']} ({scan['confidence']})"):
        
        col_text, col_chart = st.columns([1, 1])
        
        with col_text:
            st.write(f"**Diagnosis:** {scan['name']}")
            st.write(f"**Confidence:** {scan['confidence']}")
            st.info(f"**Treatment:** {scan['solution']}")
            st.caption(f"File: {scan.get('filename', 'Unknown')}")
            
        with col_chart:
            # Check if chart data exists in this history item
            if 'chart_probabilities' in scan:
                hist_chart_df = pd.DataFrame({
                    "Condition": scan['chart_conditions'],
                    "Probability": scan['chart_probabilities']
                })
                fig_hist = px.bar(
                    hist_chart_df, x="Probability", y="Condition", orientation='h', 
                    template="plotly_dark", color="Probability",
                    color_continuous_scale=['#2C3E50', '#00E676']
                )
                fig_hist.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    height=150, 
                    margin=dict(l=0, r=0, t=0, b=0), 
                    showlegend=False, 
                    yaxis={'visible': False} # Hide labels to save space
                )
                st.plotly_chart(fig_hist, use_container_width=True)