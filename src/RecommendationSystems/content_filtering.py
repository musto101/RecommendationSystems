# create a class for content-based filtering
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedFiltering:
    def __init__(self, movies):
        self.movies = movies
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(movies["genres"].fillna(""))
        self.movie_similarity = cosine_similarity(self.tfidf_matrix)

    def recommend_movies(self, movie_index, top_n=5):
        similar_movies = list(enumerate(self.movie_similarity[movie_index]))
        similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:top_n+1]

        recommended_movies = []
        for i, score in similar_movies:
            recommended_movies.append(self.movies.iloc[i]["title"])
        return recommended_movies


