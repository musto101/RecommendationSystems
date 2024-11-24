import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movies data
movies = pd.read_csv("data/ml-latest-small/movies.csv")

movies.columns

# Create a TF-IDF matrix for genres
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"].fillna(""))

# Compute cosine similarity between movies
movie_similarity = cosine_similarity(tfidf_matrix)

# Recommend movies similar to a given movie
movie_index = 1  # Example: Toy Story (1995)
similar_movies = list(enumerate(movie_similarity[movie_index]))
similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:6]

print("Movies similar to Toy Story (1995):")
for i, score in similar_movies:
    print(movies.iloc[i]["title"])
