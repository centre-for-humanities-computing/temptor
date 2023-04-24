from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap
from embedding_explorer.model import Model
from gensim.models import Word2Vec
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


def iterative_word2vec(
    periods: Iterable[list[list[str]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively trains a Word2Vec model on the given chunks of text.

    Parameters
    ----------
    periods: iterable of list of list of str
        Sequence of chunks in the form of lists of tokenized sentences.
        The iterable has to be repeatable, so iterator classes should
        be preferred over generator functions.

    Returns
    -------
    embeddings: array of shape (n_windows, n_vocab, n_features)
        Embeddings from the different training windows.
    vocab: array of shape (n_vocab)
        Array of all terms in the vocabulary.
    """

    # Initializing model with vocabulary
    model = Word2Vec()
    model.build_vocab([sentence for chunk in periods for sentence in chunk])
    # Getting vocab from model
    vocab = np.copy(model.wv.index_to_key)  # type: ignore

    # If the length of the periods can be determined we use tqdm
    try:
        n_periods = len(periods)  # type: ignore
        periods = tqdm(periods, total=n_periods)
    except TypeError:
        print(
            "Number of iterations cannot be predetermined, "
            "progress will not be tracked."
        )
    # Iteratively training and saving the embeddings
    # from different stages of the model.
    embeddings = []
    for chunk in periods:
        model.train(chunk, epochs=10, total_examples=len(chunk))
        current_embeddings = np.copy(model.wv.vectors)
        embeddings.append(current_embeddings)

    # Stacking 'em up so it's one 3-rank tensor
    embeddings = np.stack(embeddings)
    return embeddings, vocab


def train_umap(embeddings: np.ndarray) -> umap.UMAP:
    """Trains UMAP on the mean embeddings of the different word2vec models,
    so embeddings can be plotted in 2D space later.

    Parameters
    ----------
    embeddings: array of shape (n_windows, n_vocab, n_features)
        Embeddings from the different training windows.

    Returns
    -------
    UMAP
        Trained UMAP dimentionality reduction model.
    """
    mean_embeddings = np.mean(embeddings, axis=0)
    return umap.UMAP().fit(mean_embeddings)
