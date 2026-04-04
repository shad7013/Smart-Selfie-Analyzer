import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from PIL import Image
import tempfile

from src.inference import load_model, predict


# 🔹 Page config
st.set_page_config(page_title="Smart Selfie Analyzer", layout="centered")

st.title("📸 Smart Selfie Analyzer")
st.write("Upload a selfie to predict **Age, Gender, and Emotion**")


# 🔹 Load model
@st.cache_resource
def get_model():
    return load_model()

model = get_model()


# 🔹 Session state init
if "result" not in st.session_state:
    st.session_state["result"] = None

if "show_gradcam" not in st.session_state:
    st.session_state["show_gradcam"] = False

if "selected_task" not in st.session_state:
    st.session_state["selected_task"] = "emotion"


# 🔹 Input method
option = st.radio("Choose input method:", ["Upload Image", "Use Camera"])
image_file = None

if option == "Upload Image":
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

elif option == "Use Camera":
    image_file = st.camera_input("Take a selfie")


# 🔹 Reset state when a new image is uploaded
if "last_file_name" not in st.session_state:
    st.session_state["last_file_name"] = None

if image_file is not None:
    current_name = getattr(image_file, "name", "camera_input")
    
    st.image(image_file, caption="Uploaded Image", width=300)

    if current_name != st.session_state["last_file_name"]:
        st.session_state["last_file_name"] = current_name
        st.session_state["result"] = None
        st.session_state["show_gradcam"] = False

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(image_file.read())
        temp_path = tmp_file.name


    # 🔹 Analyze button
    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            st.session_state["result"] = predict(temp_path, model)
            st.session_state["show_gradcam"] = False  # reset heatmap

    # 🔹 Get stored result
    result = st.session_state.get("result")

    # 🔹 Show predictions
    if result is not None:

        st.subheader("🧠 Prediction Results")

        if not result["face_detected"]:
            st.warning("⚠️ No face detected. Results may be inaccurate.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Age Group", result["age"])
            col2.metric("Gender", result["gender"])
            col3.metric("Emotion", result["emotion"])

            # 🔹 Grad-CAM task selector
            task_option = st.radio(
                "Select Grad-CAM For:",
                ["emotion", "age", "gender"],
                index=["emotion", "age", "gender"].index(st.session_state["selected_task"])
            )

            # Save selected task
            st.session_state["selected_task"] = task_option

            # 🔹 Grad-CAM button
            if st.button("🔥 Show Grad-CAM Heatmap"):
                st.session_state["show_gradcam"] = True

            # 🔹 Show Grad-CAM
            if st.session_state.get("show_gradcam"):

                selected_task = st.session_state["selected_task"]

                # Generate Grad-CAM for selected task
                result_cam = predict(temp_path, model, task=selected_task)

                
                st.image(result_cam["heatmap"], caption=selected_task.capitalize(), width="content")

    # Cleanup temp file
    os.remove(temp_path)
