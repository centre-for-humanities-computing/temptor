import numpy as np
import pandas as pd
import plotly.express as px
import umap
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

from temptor.models import TemporalAffinityModel, train_umap
from temptor.plots import (
    relative_position_over_time,
    similarity_heatmap_over_time,
    similarity_timeline,
)

data = pd.read_json("dat/News_Category_Dataset_v3.json", orient="records", lines=True)
data = data.sort_values("date")

corpus = data.headline
vectorizer = CountVectorizer(stop_words="english", max_features=10000).fit(corpus)
dim_red = NMF(n_components=10)

windows = pd.Series(corpus).rolling(window=40000, step=40000)

model = TemporalAffinityModel(vectorizer, dim_red=dim_red)
embeddings = model.fit_transform(windows)

vocab = model.vocab

n_periods, n_vocab, n_dimensions = embeddings.shape

embeddings.shape

# # Topic naming
# top_n = 5
# components = model.dim_red.components_
# vocab = vectorizer.get_feature_names_out()
# highest = np.argpartition(-components, top_n)[:, :top_n]
# top_words = vocab[highest]
# topic_names = []
# for i_topic, words in enumerate(top_words):
#     name = "_".join(words)
#     topic_names.append(f"{i_topic}_{name}")
# topic_names = np.array(topic_names)

# Visualization
vis_red = umap.UMAP()
pos = vis_red.fit_transform(embeddings.reshape((n_periods * n_vocab, n_dimensions)))
pos = pos.reshape((n_periods, n_vocab, 2))

vis_red.components_

interesting_words = [
    "police",
    "brutality",
    "black",
    "protest",
    "killed",
]
words = []
for i_period in range(n_periods):
    topic = topic_names[np.argmax(embeddings[i_period], axis=1)]
    period_df = pd.DataFrame(
        dict(
            x=pos[i_period, :, 0],
            y=pos[i_period, :, 1],
            topic=topic,
            word=vocab,
            period=i_period,
            text=np.where(pd.Series(vocab).isin(interesting_words), vocab, ""),
        )
    )
    words.append(period_df)
words = pd.concat(words)
px.scatter(
    words,
    x="x",
    y="y",
    color="topic",
    text="text",
    hover_data=["word"],
    facet_col="period",
    facet_col_wrap=3,
)

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
