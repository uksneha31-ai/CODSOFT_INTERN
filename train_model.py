import cv2
import os
import numpy as np
from PIL import Image

# Create LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

dataset_path = "dataset"
trainer_path = "trainer"

faces = []
ids = []
label_map = {}
current_id = 0

print("Reading dataset...")

# Read all person folders
for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    label_map[current_id] = person_name

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        img = Image.open(image_path).convert('L')
        img_numpy = np.array(img, 'uint8')

        faces.append(img_numpy)
        ids.append(current_id)

    current_id += 1

if len(faces) == 0:
    print("No training images found!")
    exit()

print("Training model...")

recognizer.train(faces, np.array(ids))

os.makedirs(trainer_path, exist_ok=True)

recognizer.save(os.path.join(trainer_path, "trainer.yml"))

# Save labels
with open(os.path.join(trainer_path, "labels.txt"), "w") as f:
    for key, value in label_map.items():
        f.write(f"{key},{value}\n")

print("=" * 40)
print("Training Completed Successfully!")
print(f"Total Persons : {len(label_map)}")
print(f"Total Images  : {len(faces)}")
print("Model saved in trainer/trainer.yml")
print("=" * 40)