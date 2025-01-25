import torch
import cv2
import numpy as np
from PIL import Image
from transformers import SwinForImageClassification, AutoImageProcessor
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the Swin Transformer model for feature extraction
model_name = "microsoft/swin-base-patch4-window7-224"
model = SwinForImageClassification.from_pretrained(model_name, num_labels=0)  # Feature extraction
model.load_state_dict(torch.load("./models/swin_model_feature_extractor.pth"))  # Load trained feature extractor weights
model.eval()

# Load the SVM classifier and label encoder
classifier = joblib.load("./models/svm_classifier.pkl")  # Load trained SVM classifier
label_encoder = np.load("./models/label_encoder.npy", allow_pickle=True)  # Load label encoder classes

# Load the image processor
image_processor = AutoImageProcessor.from_pretrained(model_name)

# Load OpenCV's face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess an image for Swin Transformer
def preprocess_image(image):
    """
    Preprocess an image for Swin Transformer.
    """
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert to PIL image
    inputs = image_processor(images=image, return_tensors="pt")  # Preprocess for Swin
    return inputs

# Function to recognize faces in a given frame
def recognize_faces(frame, model, classifier, label_encoder, threshold=60):
    """
    Recognizes faces in a given frame using the Swin Transformer model and classifier.
    """
    # Convert the frame into preprocessed input
    inputs = preprocess_image(frame)

    # Extract features using the Swin Transformer
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.logits.cpu().numpy()  # Get feature vector from logits

    # Predict the label using the SVM classifier
    predictions = classifier.predict(features)
    probabilities = classifier.predict_proba(features)

    # Decode the label and confidence
    max_confidence = max(probabilities[0]) * 100  # Confidence in percentage
    label = label_encoder[predictions[0]] if max_confidence >= threshold else "Unknown"

    return label, max_confidence

# Load video file or webcam
path = 'C:/chess/Face_Recognition_System_with-_Swin_Transformer_and_SVM_Classifier/855564-hd_1920_1080_24fps.mp4'
cap = cv2.VideoCapture(path)

print("Starting face recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting.")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Loop over the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the face region from the frame
        face = frame[y:y + h, x:x + w]

        # Recognize the face
        label, confidence = recognize_faces(face, model, classifier, label_encoder)

        # Display the label and confidence on the frame
        text = f"{label} ({confidence:.2f}%)"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame with the recognized faces
    cv2.imshow("Face Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
