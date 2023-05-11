from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer

from temptor.models import TemporalAffinityModel

# This gotta be your corpus, like all texts y'know
corpus: list[str] = []

# This is the corpus divided into different periods
periods: list[list[str]] = [[]]

# Can be TfidfVectorizer too
# HAS TO BE FITTED BEFORE GIVING IT TO TemporalAffinityModel
vectorizer = CountVectorizer(stop_words="english").fit(corpus)

# Can be any kind of dimentionality reduction method
nmf = NMF(n_components=50)

# Defining the temporal model
temporal_model = TemporalAffinityModel(vectorizer=vectorizer, dim_red=nmf)

# Getting embeddings
# it's gonna be an array of shape (n_periods, n_vocab, 50)
embeddings = temporal_model.fit_transform(periods)
