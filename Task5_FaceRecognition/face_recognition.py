import cv2
import os

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Check if cascade loaded correctly
if face_cascade.empty():
    print("Error: Could not load Haar Cascade.")
    exit()

# Create recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load trained model
recognizer.read("trainer/trainer.yml")

# Load labels
labels = {}

with open("trainer/labels.txt", "r") as f:
    for line in f.readlines():
        id, name = line.strip().split(",")
        labels[int(id)] = name

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("=" * 50)
print("AI FACE RECOGNITION SYSTEM")
print("Press 'Q' to Quit")
print("=" * 50)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        id, confidence = recognizer.predict(face)

        if confidence < 70:
            name = labels.get(id, "Unknown")
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.putText(
            frame,
            name,
            (x, y-35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {100-confidence:.1f}%",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.putText(
        frame,
        "AI Face Recognition",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Program Closed.")