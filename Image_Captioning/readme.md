# Image Captioning AI

## Overview

This project is an **AI-based Image Captioning System** that automatically generates a textual description for an input image. It combines **Computer Vision** and **Natural Language Processing (NLP)** using a pre-trained BLIP (Bootstrapping Language-Image Pre-training) model from Hugging Face.

The system analyzes the contents of an image and generates a meaningful caption in natural language.

## Features

* Automatically generates captions for images.
* Uses a pre-trained BLIP image captioning model.
* Accepts any JPG, JPEG, or PNG image.
* Simple command-line interface.
* No model training required.

## Technologies Used

* Python
* Transformers (Hugging Face)
* PyTorch
* Pillow (PIL)

## Project Structure

```text
Image_Captioning/
│── image_caption.py
│── image.jpg
│── requirements.txt
└── README.md
```

## Installation

Install the required libraries:

```bash
pip install transformers torch pillow
```

## How to Run

1. Place an image in the project folder.
2. Open the terminal in the project directory.
3. Run the program:

```bash
python image_caption.py
```

4. Enter the image file name when prompted.

Example:

```text
Enter image name (e.g., image.jpg): image.jpg
```

## Sample Output

```text
Loading model... Please wait.

Enter image name (e.g., image.jpg): image.jpg

Generated Caption:
the mercedes sports car driving on a road.
```

## How It Works

1. The program loads a pre-trained BLIP image captioning model.
2. The input image is converted into visual features.
3. The transformer-based model understands the image content.
4. A natural language caption is generated and displayed.

## Applications

* Automatic image description
* Image search and indexing
* Social media caption generation
* Digital asset management
* Assistive technology for visually impaired users

## Future Enhancements

* Support multiple images at once.
* Build a graphical user interface (GUI).
* Deploy as a web application using Flask or Streamlit.
* Generate captions in multiple languages.
* Improve caption quality using larger vision-language models.

## Requirements

```text
transformers
torch
Pillow
```

## Author

Developed as part of an AI Internship Task on **Image Captioning using Computer Vision and Natural Language Processing (NLP)**.
Sneha UK