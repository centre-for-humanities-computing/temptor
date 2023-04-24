import pandas as pd

from temptor.models import iterative_word2vec, train_umap
from temptor.plots import (
    relative_position_over_time,
    similarity_heatmap_over_time,
    similarity_timeline,
)

data = pd.read_csv("dat/abcnews-date-text.csv")
data = data.sort_values("publish_date")
sentences = data.headline_text.map(lambda s: s.split())

rolling_windows = sentences.rolling(window=50_000, step=25_000)
total = len(rolling_windows.count())
print(total)

embeddings, vocab = iterative_word2vec(periods=rolling_windows)

umap = train_umap(embeddings=embeddings)

interesting_words = [
    "women",
    "kitchen",
    "garden",
    "programming",
    "men",
    "industry",
    "secretary",
]

# Relative positions
fig = relative_position_over_time(
    words=interesting_words,
    embeddings=embeddings,
    vocab=vocab,
    dim_red=umap,
)
fig.write_html("relative_pos.html")
fig.show()

# Similarity heatmap
fig = similarity_heatmap_over_time(
    words=interesting_words,
    embeddings=embeddings,
    vocab=vocab,
)
fig.write_html("similarity_heatmap.html")
fig.show()

# Similarity timelines
fig = similarity_timeline(
    words=interesting_words,
    embeddings=embeddings,
    vocab=vocab,
)
fig.write_html("similarity_timeline.html")
fig.show()
