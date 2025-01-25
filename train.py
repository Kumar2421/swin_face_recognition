import os
import torch
from transformers import SwinForImageClassification, AutoImageProcessor
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from PIL import Image
import joblib

# Paths
data_dir = "./dataset"  # Directory containing face images organized by folders (one folder per person)
models_dir = "./models"  # Directory for saving model files
swin_model_name = "microsoft/swin-base-patch4-window7-224"

# Load Swin Transformer model and processor
model = SwinForImageClassification.from_pretrained(swin_model_name, num_labels=0)  # Feature extraction only
model.eval()
image_processor = AutoImageProcessor.from_pretrained(swin_model_name)

# Ensure the models directory exists
os.makedirs(models_dir, exist_ok=True)

# Prepare dataset
def load_dataset(data_dir):
    """
    Loads images and labels from the dataset directory.
    Each subdirectory in `data_dir` represents a class.
    """
    images = []
    labels = []

    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                try:
                    image = Image.open(img_path).convert("RGB")  # Ensure all images are RGB
                    images.append(image)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    return images, labels

# Extract features using the Swin Transformer
def extract_features(images, model, image_processor):
    """
    Extract features from a list of images using the Swin Transformer model.
    """
    features = []

    with torch.no_grad():
        for image in images:
            inputs = image_processor(images=image, return_tensors="pt")  # Preprocess image
            outputs = model(**inputs)  # Forward pass
            features.append(outputs.logits.cpu().numpy().flatten())  # Extract logits as features

    return np.array(features)

# Load images and labels
print("Loading dataset...")
images, labels = load_dataset(data_dir)

# Encode labels
print("Encoding labels...")
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)

# Extract features
print("Extracting features from training set...")
train_features = extract_features(X_train, model, image_processor)

print("Extracting features from testing set...")
test_features = extract_features(X_test, model, image_processor)

# Train SVM classifier
print("Training SVM classifier...")
svm_classifier = SVC(probability=True, kernel="linear")
svm_classifier.fit(train_features, y_train)

# Evaluate the classifier
print("Evaluating the classifier...")
y_pred = svm_classifier.predict(test_features)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save the trained model and label encoder
print("Saving model and label encoder...")
joblib.dump(svm_classifier, os.path.join(models_dir, "svm_classifier.pkl"))
np.save(os.path.join(models_dir, "label_encoder.npy"), label_encoder.classes_)
torch.save(model.state_dict(), os.path.join(models_dir, "swin_model_feature_extractor.pth"))

print("Training complete!")
