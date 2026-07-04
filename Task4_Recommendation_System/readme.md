# Movie Recommendation System

## Overview

This project is a simple **Content-Based Recommendation System** developed using Python. It recommends movies to users based on the genre of a movie they like. The system compares movie genres and suggests similar movies using **Cosine Similarity**.

## Features

* Recommends similar movies based on genres.
* Uses Content-Based Filtering.
* Simple command-line interface.
* Easy to understand and modify.

## Technologies Used

* Python
* Pandas
* Scikit-learn

## Dataset

The project uses a small custom dataset (`movies.csv`) containing:

* Movie Title
* Genre

## How It Works

1. Load the movie dataset.
2. Convert movie genres into numerical vectors using `CountVectorizer`.
3. Calculate similarity between movies using Cosine Similarity.
4. Ask the user to enter a movie name.
5. Display the top 5 most similar movie recommendations.

## Installation

Install the required libraries:

```bash
pip install pandas scikit-learn
```

## Run the Project

```bash
python recommendation.py
```

## Example

**Input:**

```
Enter a movie name:
Iron Man
```

**Output:**

```
Recommendations for 'Iron Man':

Avengers
Spider-Man
Doctor Strange
Inception
Interstellar
```

## Project Structure

```
Movie_Recommendation_System/
│── recommendation.py
│── movies.csv
└── README.md
```

## Recommendation Technique

**Content-Based Filtering**

The recommendation system analyzes movie genres and suggests movies that have similar genre characteristics using **CountVectorizer** and **Cosine Similarity**.

## Future Improvements

* Use a larger movie dataset.
* Add movie posters.
* Build a graphical user interface (GUI).
* Develop a web application using Flask or Streamlit.
* Include movie ratings and user reviews for better recommendations.

## Author

Developed as part of an AI Internship Task on Recommendation Systems.
Sneha UK