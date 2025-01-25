import cv2
import os
import time

# Initialize OpenCV to capture video from webcam
cap = cv2.VideoCapture(0)

# Define the face detection model (Haarcascades)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Get person's name
name = input("Enter your name: ")

# Root directory for saving captured images
root_dir = 'dataset'

# Create dataset directory if it doesn't exist
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

# Create a folder for the person in dataset folder
person_dir = os.path.join(root_dir, name)
if not os.path.exists(person_dir):
    os.makedirs(person_dir)

# Create a subfolder for images
images_dir = os.path.join(person_dir)
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# Initialize counter for image saving
count = 0
print("Starting image capture...")

# Record the starting time for controlling the lag
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Crop the face from the frame
        face = frame[y:y+h, x:x+w]

        # Save the face image with the label (name) in the images subfolder
        cv2.imwrite(f'{images_dir}/image_{count}.jpg', face)
        count += 1

        # Display face and label
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the frame with faces
    cv2.imshow('Capture Faces', frame)

    # Wait for 0.1 second and check for 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
        break

    # Limit the loop to 10 FPS (0.1 second per frame)
    while time.time() - start_time < 0.1:
        # Wait until 0.1 seconds have passed before processing the next frame
        time.sleep(1)

    start_time = time.time()

cap.release()
cv2.destroyAllWindows()
