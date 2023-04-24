import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import pairwise_distances


def add_slider(fig: go.Figure) -> go.Figure:
    """Adds a slider to a plotly figure, so that you can slide through
    the different traces in the figure.

    Parameters
    ----------
    fig: Figure
        Figure to add the slider to.

    Returns
    -------
    Figure
        Same figure with slider.
    """
    for i in range(len(fig.data)):
        fig.data[i].visible = False
    fig.data[0].visible = True
    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
                {"title": "Slider switched to step: " + str(i)},
            ],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)
    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "Window: "},
            pad={"t": 50},
            steps=steps,
        )
    ]
    fig = fig.update_layout(sliders=sliders)
    return fig


def similarity_heatmap_over_time(
    words: list[str], embeddings: np.ndarray, vocab: np.ndarray
) -> go.Figure:
    """Plots the similarities of the given words on a heatmap with
    a slider, so you can scroll through the different time windows.

    Parameters
    ----------
    words: list of str
        List of words you want compared.
    embeddings: array of shape (n_windows, n_vocab, n_features)
        Embeddings from the different training windows.
    vocab: array of shape (n_vocab)
        Array of all terms in the vocabulary.

    Returns
    -------
    Figure
        Heatmap with slider.
    """
    vocab_lookup = {word: i_word for i_word, word in enumerate(vocab)}
    interesting_indices = np.array([vocab_lookup[word] for word in words])
    interesting_embeddings = embeddings[:, interesting_indices, :]
    fig = go.Figure()
    for i_window in range(interesting_embeddings.shape[0]):
        window_embeddings = interesting_embeddings[i_window]
        distance = pairwise_distances(window_embeddings, metric="cosine")
        fig = fig.add_trace(
            go.Heatmap(
                z=1 - distance,
                x=words,
                y=words,
                name=f"Window {i_window}",
            )
        )
    fig = add_slider(fig)
    return fig


def relative_position_over_time(
    words: list[str], embeddings: np.ndarray, vocab: np.ndarray, dim_red
) -> go.Figure:
    """Plots the words on a scatter plot with dimensionality reduction.
    A slider is added, so you can scroll over the different time periods.

    Parameters
    ----------
    words: list of str
        List of words you want compared.
    embeddings: array of shape (n_windows, n_vocab, n_features)
        Embeddings from the different training windows.
    vocab: array of shape (n_vocab)
        Array of all terms in the vocabulary.
    dim_red: sklearn.base.TransformerMixin
        Dimentionality reduction method for plotting.
        Can be UMAP, TSNE or whatever. The model has to be
        trained.

    Returns
    -------
    Figure
        Scatterplot with slider.
    """
    n_words = len(words)
    n_periods = embeddings.shape[0]
    n_features = embeddings.shape[-1]
    vocab_lookup = {word: i_word for i_word, word in enumerate(vocab)}
    interesting_indices = np.array([vocab_lookup[word] for word in words])
    reshaped = embeddings[:, interesting_indices, :].reshape(
        (n_words * n_periods, n_features)
    )
    x, y = dim_red.transform(reshaped).T
    iteration = np.floor_divide(
        np.arange(n_periods * interesting_indices.shape[0]),
        interesting_indices.shape[0],
    )
    word = np.array(words * n_periods)
    data = pd.DataFrame(dict(x=x, y=y, iteration=iteration, word=word))
    fig = go.Figure()
    for i_window in range(embeddings.shape[0]):
        window_data = data[data.iteration == i_window]
        fig.add_trace(
            go.Scatter(
                visible=False,
                name=f"Window {i_window}",
                x=window_data.x,
                y=window_data.y,
                text=window_data.word,
                mode="markers+text",
            )
        )
    fig = fig.update_xaxes(range=[data.x.min() - 0.5, data.x.max() + 0.5])
    fig = fig.update_yaxes(range=[data.y.min() - 0.5, data.y.max() + 0.5])
    fig = add_slider(fig)
    return fig


def similarity_timeline(
    words: list[str], embeddings: np.ndarray, vocab: np.ndarray
) -> go.Figure:
    """Plots pairwise similarities of the given words on a timeline.

    Parameters
    ----------
    words: list of str
        List of words you want compared.
    embeddings: array of shape (n_windows, n_vocab, n_features)
        Embeddings from the different training windows.
    vocab: array of shape (n_vocab)
        Array of all terms in the vocabulary.

    Returns
    -------
    Figure
        Similarity timeline line plot.
    """
    vocab_lookup = {word: i_word for i_word, word in enumerate(vocab)}
    n_words = len(words)
    interesting_indices = np.array([vocab_lookup[word] for word in words])
    interesting_embeddings = embeddings[:, interesting_indices, :]
    records = []
    for i_window in range(interesting_embeddings.shape[0]):
        window_embeddings = interesting_embeddings[i_window]
        distance = pairwise_distances(window_embeddings, metric="cosine")
        affinity = 1 - distance
        for i_word in range(n_words):
            for j_word in range(n_words):
                if j_word < i_word:
                    records.append(
                        dict(
                            word1=words[i_word],
                            word2=words[j_word],
                            similarity=affinity[i_word, j_word],
                            window=i_window,
                        )
                    )
    data = pd.DataFrame.from_records(records)
    data["word_pair"] = data["word1"] + ", " + data["word2"]
    return px.line(data, color="word_pair", x="window", y="similarity")
