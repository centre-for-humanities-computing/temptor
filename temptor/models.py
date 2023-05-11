from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.sparse as spr
import umap
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
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


class TemporalAffinityModel(BaseEstimator):
    """Temporal word embedding model based on cosine affinities of words in
    document embedding space and dimensionality reduction.

    Parameters
    ----------
    vectorizer
        Vectorizer model that produces embeddings of documents.
        The vectorizer HAS TO BE fitted before passed to this model.
    dim_red
        Dimentionality reduction model that will be used for
        reducing the dimentionality of the embeddings from the
        similarity matrices.
        The model SHOULD NOT BE fitted before passed to this model.

    Attributes
    ----------
    vocab: array of shape (n_vocab)
        Array of output features of the vectorizer, aka.
        the the order in which words will be arranged in the produced
        embeddings.

    components_: array of shape (n_dimensions, n_vocab)
        Linearly estimated term importances for each term, or
        components of the dimentionality reduction method.
    """

    def __init__(self, vectorizer, dim_red):
        super().__init__()
        self.vectorizer = vectorizer
        self.dim_red = dim_red
        self.vocab = None
        self.components_ = None

    def _get_affinities(self, periods: Iterable[Iterable[str]]):
        affinities = []
        for period in periods:
            dtm = self.vectorizer.transform(period)
            affinity = cosine_similarity(dtm.T, dense_output=False)
            affinities.append(affinity)
        return affinities

    def fit(self, periods: Iterable[Iterable[str]]):
        """Fits the model on the given periods.

        Parameters
        ----------
        periods: iterable of iterable of str
            All documents for each period.

        Returns
        -------
        TemporalAffinityModel
            Fitted model.
        """
        self.vocab = self.vectorizer.get_feature_names_out()
        affinities = self._get_affinities(periods)
        affinities = spr.vstack(affinities)
        y = self.dim_red.fit_transform(affinities)
        try:
            self.components_ = self.dim_red.components_
        except AttributeError:
            ridge = Ridge(fit_intercept=False).fit(affinities, y)
            self.components_ = ridge.coef_
        return self

    def transform(self, periods: Iterable[Iterable[str]]) -> np.ndarray:
        """Transforms the periods into temporal word embeddings for each
        period.

        Parameters
        ----------
        periods: iterable of iterable of str
            All documents for each period.

        Returns
        -------
        array of shape (n_periods, n_vocab, n_dimentions)
            Embeddings of all words for each period.
        """
        if self.vocab is None:
            raise ValueError(
                "Model has not yet been fitted, plese fit before transforming."
            )
        else:
            n_vocab = len(self.vocab)
        n_dimensions = self.dim_red.n_components
        affinities = self._get_affinities(periods)
        n_periods = len(affinities)
        affinities = spr.vstack(affinities)
        embeddings = self.dim_red.transform(affinities)
        embeddings = embeddings.reshape((n_periods, n_vocab, n_dimensions))
        return embeddings

    def fit_transform(self, periods: Iterable[Iterable[str]]) -> np.ndarray:
        """Fits the model, then transforms the periods
        into temporal word embeddings for each period.

        Parameters
        ----------
        periods: iterable of iterable of str
            All documents for each period.

        Returns
        -------
        array of shape (n_periods, n_vocab, n_dimentions)
            Embeddings of all words for each period.
        """
        self.fit(periods)
        return self.transform(periods)
