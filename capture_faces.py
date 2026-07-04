import cv2
import os

# -------------------------------
# Configuration
# -------------------------------
PERSON_NAME = "Sneha"
MAX_IMAGES = 200
IMAGE_SIZE = (200, 200)

# -------------------------------
# Load Haar Cascade
# -------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Error: Could not load Haar Cascade.")
    exit()

# -------------------------------
# Create Dataset Folder
# -------------------------------
dataset_path = os.path.join("dataset", PERSON_NAME)
os.makedirs(dataset_path, exist_ok=True)

# -------------------------------
# Open Webcam
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("=" * 50)
print("AI FACE DATASET CAPTURE")
print(f"Person : {PERSON_NAME}")
print(f"Images to Capture : {MAX_IMAGES}")
print("Press Q to Quit")
print("=" * 50)

count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improve lighting
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        # Resize every face to same size
        face = cv2.resize(face, IMAGE_SIZE)

        count += 1

        filename = os.path.join(dataset_path, f"{count}.jpg")
        cv2.imwrite(filename, face)

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # Save only one face per frame
        break

    # Display information
    cv2.putText(
        frame,
        "AI Face Dataset Capture",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Person : {PERSON_NAME}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Captured : {count}/{MAX_IMAGES}",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Capture Face Dataset", frame)

    key = cv2.waitKey(80) & 0xFF

    if key == ord('q'):
        break

    if count >= MAX_IMAGES:
        print("Dataset Capture Completed.")
        break

cap.release()
cv2.destroyAllWindows()

print("=" * 50)
print("Dataset Saved Successfully!")
print("Location :", dataset_path)
print("Total Images :", count)
print("=" * 50)