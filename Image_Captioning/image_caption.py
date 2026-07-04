import os
import warnings
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers.utils import logging

# Hide warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

print("Loading model... Please wait.")

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Get image name from user
image_path = input("Enter image name (e.g., image.jpg): ").strip()

# Check if image exists
if not os.path.exists(image_path):
    print(f"Error: '{image_path}' not found!")
    print("Place the image in the same folder as this program.")
    exit()

# Open image
image = Image.open(image_path).convert("RGB")

# Process image
inputs = processor(images=image, return_tensors="pt")

# Generate caption
output = model.generate(**inputs, max_new_tokens=20)

# Decode caption
caption = processor.decode(output[0], skip_special_tokens=True)

print("\nGenerated Caption:")
print(caption)