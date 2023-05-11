import numpy as np
import pandas as pd
from scipy import sparse as spr
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_json("News_Category_Dataset_v3.json", orient="records", lines=True)

data = data.sort_values("date")

corpus = data.headline
vectorizer = CountVectorizer(stop_words="english", max_features=40000).fit(corpus)

windows = pd.Series(corpus).rolling(window=40000, step=40000)

n_windows = len(windows.count())
n_features = vectorizer.get_feature_names_out().shape[0]

full_embeddings = []
for window in windows:
    dtm = vectorizer.transform(window)
    similarities = cosine_similarity(dtm.T, dense_output=False)
    full_embeddings.append(similarities)

cooccurrances = spr.vstack(full_embeddings)

model = NMF(n_components=20)
embeddings = model.fit_transform(cooccurrances)

model.n_components

embeddings = embeddings.reshape((n_windows, n_features, 20))

top_n = 5
components = model.components_
vocab = vectorizer.get_feature_names_out()
highest = np.argpartition(-components, top_n)[:, :top_n]
top_words = vocab[highest]
topic_names = []
for i_topic, words in enumerate(top_words):
    name = "_".join(words)
    topic_names.append(f"{i_topic}_{name}")

topic_names
