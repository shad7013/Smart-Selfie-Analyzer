# 📸 Smart Selfie Analyzer

A deep learning-based web application that analyzes a selfie to predict **Age Group, Gender, and Emotion**, along with **visual explanations using Grad-CAM**.

---

## 🚀 Features

* 🔍 Multi-task prediction:

  * Age Group classification
  * Gender classification
  * Emotion detection
* 📷 Supports:

  * Image upload
  * Real-time camera input
* 🧠 Explainable AI:

  * Grad-CAM heatmaps to visualize model attention
* 🎯 Clean and interactive UI built with Streamlit

---

## 🧠 Model Architecture

* Backbone: **ResNet18 (pretrained)**
* Multi-task learning with **shared feature extractor**
* Three output heads:

  * Age classification
  * Gender classification
  * Emotion classification

---

## ⚙️ How it Works

1. User uploads or captures a selfie
2. Face is detected using OpenCV
3. Image is preprocessed and passed to the model
4. Model predicts:

   * Age group
   * Gender
   * Emotion
5. Grad-CAM generates a heatmap showing where the model focused

---

## 🔥 Grad-CAM Explainability

Grad-CAM is used to visualize which regions of the face influenced the model's predictions.

**Note:**
Due to the shared backbone architecture, heatmaps across tasks may appear similar, as the model relies on common facial features (e.g., eyes, mouth).

---

## 🛠️ Tech Stack

* Python
* PyTorch
* OpenCV
* Streamlit
* NumPy, PIL

---

## ▶️ Run Locally

```bash
git clone <your-repo-link>
cd smart-selfie-analyzer

pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## 📊 Example Output

* Age Group: Young Adult
* Gender: Female
* Emotion: Happy
* Grad-CAM highlights mouth and facial expressions

---

## ⚠️ Limitations

* Performance may vary with lighting conditions
* Model may rely on shared features across tasks
* Accuracy depends on dataset quality (UTKFace + FER)

---

## 💡 Future Improvements

* Improve task-specific feature separation
* Use advanced face detection (MTCNN / MediaPipe)
* Add confidence scores
* Deploy as web app

---

## 👤 Author

**Name:** Mohammed Shad T  
**GitHub:** [https://github.com/shad7013](https://github.com/shad7013)  
**LinkedIn:** [www.linkedin.com/in/mohammed-shad-t-4b6866269](https://www.linkedin.com/in/mohammed-shad-t-4b6866269)

