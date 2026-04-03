import cv2
import numpy as np 
from PIL import Image 

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
def detect_and_crop_face(image_path):
    """
    Detects face in an image and returns cropped face as PIL image.
    and boolean value variable indicating if a face was detected.
    """
    
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Invalid image path or unable to read image. {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
        )

    if len(faces) == 0:
        face = image
        face_detected = False
    else:
        # Assuming the first detected face is the one we want
        (x, y, w, h) = faces[0]
        face = image[y:y+h, x:x+w]
        face_detected = True

    # Convert the cropped face to PIL image (RGB format)
    # Safety check
    if face is None or face.size == 0:
        raise ValueError("Face crop failed or empty image.")

    # Convert BGR → RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Convert safely to PIL RGB
    face = Image.fromarray(face).convert("RGB")

    return face, face_detected