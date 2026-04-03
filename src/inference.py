import torch
import torchvision.transforms as transforms
from PIL import Image

from src.model import MultiTaskModel
from src.preprocess import detect_and_crop_face
from src.gradcam import GradCAM, overlay_heatmap

# Device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Labels for mapping 
AGE_CLASSES = ["Child", "Teen", "Young Adult", "Adult", "Senior"]
GENDER_CLASSES = ["Male", "Female"]
EMOTION_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model
def load_model(model_path="models/best_model.pth"):
    model = MultiTaskModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Preprocess the image with face detection and cropping
def preprocess_image(image_path):
    image, face_detected = detect_and_crop_face(image_path)
    image = transform(image).unsqueeze(0)
    return image.to(device), face_detected

# Predict function
def predict(image_path, model, task="emotion"):
    image, face_detected = preprocess_image(image_path)

    with torch.no_grad():
        outputs = model(image)

    age_out = outputs["age"]
    gender_out = outputs["gender"]
    emotion_out = outputs["emotion"]

    age_pred = torch.argmax(age_out, dim=1).item()
    gender_pred = torch.argmax(gender_out, dim=1).item()
    emotion_pred = torch.argmax(emotion_out, dim=1).item()

    # Get the last convolutional layer for Grad-CAM
    target_layer = model.backbone.layer4[-1]

    # generate Grad-CAM heatmap for emotion prediction
    gradcam = GradCAM(model, target_layer)

    # dynamically generate CAM for the predicted emotion class
    if task == "emotion":
        target_class = emotion_pred 
    elif task == "age":
        target_class = age_pred
    else:
        target_class = gender_pred

    cam = gradcam.generate(image, target_class, task=task)

    # get the original face image for overlay
    face_img, _ = detect_and_crop_face(image_path)
    face_img = face_img.resize((224, 224))

    # overlay heatmap on face image
    heatmap = overlay_heatmap(face_img, cam)

    return {
        "age": AGE_CLASSES[age_pred],
        "gender": GENDER_CLASSES[gender_pred],
        "emotion": EMOTION_CLASSES[emotion_pred],
        "face_detected": face_detected,
        "heatmap": heatmap
    }

    

