import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")

# Convert genres into vectors
cv = CountVectorizer()
count_matrix = cv.fit_transform(movies["genre"])

# Calculate similarity
similarity = cosine_similarity(count_matrix)

def recommend(movie_name):
    movie_name = movie_name.lower()

    # Find movie
    index = None
    for i, title in enumerate(movies["title"]):
        if title.lower() == movie_name:
            index = i
            break

    if index is None:
        print("Movie not found!")
        return

    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print(f"\nRecommendations for '{movies.iloc[index]['title']}':\n")

    count = 0
    for movie in scores[1:]:
        print(movies.iloc[movie[0]]["title"])
        count += 1
        if count == 5:
            break

movie = input("Enter a movie name: ")
recommend(movie)