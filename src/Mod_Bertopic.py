"""Topic modelling module for the NLP survey corpus.

Phase 4 of the NLP pipeline. Consumes the synonym-normalised text
produced upstream (``limpieza.py`` -> ``synonyms.py``) and discovers
latent topics with BERTopic.

Pipeline of the underlying model:
    1. Embeddings   -- multilingual Sentence-Transformer (Spanish-ready).
    2. Reduction    -- UMAP to a low-dimensional space.
    3. Clustering   -- HDBSCAN (auto-selects the number of topics).
    4. Topic terms  -- c-TF-IDF for representative words per topic.

The heavy dependencies (``bertopic``, ``sentence-transformers``,
``umap-learn``, ``hdbscan``) are imported lazily inside the methods that
need them, so importing this module stays cheap and the rest of the
pipeline does not pay for them unless topic modelling is actually run.

Typical usage::

    from Mod_Bertopic import TopicModeler

    modeler = TopicModeler(random_state=42)
    modeler.fit(corpus_series)          # corpus = _no_stopwords_synonyms
    print(modeler.topic_overview())
    modeler.save("models/bertopic_eval_docente")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from bertopic import BERTopic
    from plotly.graph_objects import Figure as PlotlyFigure


# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------
#: Multilingual model that performs well on Spanish while staying small.
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

#: Minimum documents a topic must contain (HDBSCAN ``min_cluster_size``).
DEFAULT_MIN_TOPIC_SIZE = 15


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _prepare_documents(corpus: pd.Series) -> list[str]:
    """Convert a Series into a clean list of non-empty documents.

    Args:
        corpus: Text Series (typically the ``_no_stopwords_synonyms``
            column).

    Returns:
        List of non-empty, stripped string documents.

    Raises:
        ValueError: If no usable documents remain.
    """
    documents = [text.strip() for text in corpus.fillna("").astype(str) if text.strip()]
    if not documents:
        raise ValueError("The corpus contains no non-empty documents to model.")
    return documents


# ===================================================================
# TOPIC MODELER
# ===================================================================
class TopicModeler:
    """Discover and inspect topics in survey responses with BERTopic.

    The class wraps a configured :class:`bertopic.BERTopic` instance and
    exposes a small, task-focused API: :meth:`fit`, inspection methods,
    visualisations and persistence. It mirrors the design of the other
    pipeline modules (type hints, Google docstrings, single-responsibility
    methods) so it plugs in cleanly.

    Args:
        embedding_model: Name of the Sentence-Transformer model to use.
        min_topic_size: Minimum number of documents per topic. Larger
            values yield fewer, broader topics.
        language: Language passed to BERTopic for its default vectoriser.
        random_state: Seed for UMAP, ensuring reproducible topics.
        calculate_probabilities: Whether to compute per-document topic
            probabilities (slower, but enables soft assignments).
    """

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        min_topic_size: int = DEFAULT_MIN_TOPIC_SIZE,
        language: str = "multilingual",
        random_state: int = 42,
        *,
        calculate_probabilities: bool = True,
    ) -> None:
        self.embedding_model = embedding_model
        self.min_topic_size = min_topic_size
        self.language = language
        self.random_state = random_state
        self.calculate_probabilities = calculate_probabilities

        self._model: BERTopic | None = None
        self._documents: list[str] | None = None
        self._topics: list[int] | None = None

    # -- Model construction -------------------------------------------------

    def _build_model(self) -> BERTopic:
        """Construct a configured BERTopic instance.

        A fixed ``random_state`` is injected into UMAP for
        reproducibility, since BERTopic's default UMAP is stochastic.

        Returns:
            An unfitted :class:`bertopic.BERTopic` instance.

        Raises:
            ImportError: If BERTopic or its dependencies are missing.
        """
        try:
            from bertopic import BERTopic
            from umap import UMAP
        except ImportError as exc:
            raise ImportError(
                "Topic modelling requires 'bertopic' and 'umap-learn'. "
                "Install them with: poetry add bertopic umap-learn"
            ) from exc

        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=self.random_state,
        )

        return BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            min_topic_size=self.min_topic_size,
            language=self.language,
            calculate_probabilities=self.calculate_probabilities,
            verbose=True,
        )

    # -- Fitting ------------------------------------------------------------

    def fit(self, corpus: pd.Series) -> TopicModeler:
        """Fit the topic model on a corpus of survey responses.

        Args:
            corpus: Text Series to model (``_no_stopwords_synonyms``).

        Returns:
            The fitted modeler (to allow chaining).
        """
        self._documents = _prepare_documents(corpus)
        self._model = self._build_model()
        topics, _ = self._model.fit_transform(self._documents)
        self._topics = list(topics)
        return self

    # -- State guard --------------------------------------------------------

    @property
    def model(self) -> BERTopic:
        """The fitted BERTopic model.

        Raises:
            RuntimeError: If accessed before :meth:`fit`.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before using the model.")
        return self._model

    # -- Inspection ---------------------------------------------------------

    def topic_overview(self) -> pd.DataFrame:
        """Return the table of discovered topics.

        Returns:
            DataFrame with one row per topic (``Topic``, ``Count``,
            ``Name`` and representative terms). Topic ``-1`` collects
            outlier documents that were not assigned to any cluster.
        """
        return self.model.get_topic_info()

    def topic_terms(self, topic_id: int, top_k: int = 10) -> pd.DataFrame:
        """Return the representative terms for a single topic.

        Args:
            topic_id: The topic identifier (``-1`` is the outlier topic).
            top_k: Number of terms to return.

        Returns:
            DataFrame with ``Term`` and ``Weight`` (c-TF-IDF score).

        Raises:
            ValueError: If the topic id is unknown.
        """
        terms = self.model.get_topic(topic_id)
        if terms is False or terms is None:
            raise ValueError(f"Unknown topic id: {topic_id}")
        rows = [
            {"Term": term, "Weight": round(float(weight), 6)}
            for term, weight in terms[:top_k]
        ]
        return pd.DataFrame(rows)

    def representative_docs(self, topic_id: int, n_docs: int = 3) -> list[str]:
        """Return example documents most representative of a topic.

        Args:
            topic_id: The topic identifier.
            n_docs: Maximum number of documents to return.

        Returns:
            List of representative document strings.
        """
        docs = self.model.get_representative_docs(topic_id) or []
        return list(docs[:n_docs])

    def assignments(self) -> pd.DataFrame:
        """Return the topic assigned to each fitted document.

        Returns:
            DataFrame with ``document`` and ``topic`` columns.

        Raises:
            RuntimeError: If accessed before :meth:`fit`.
        """
        if self._documents is None or self._topics is None:
            raise RuntimeError("Call fit() before requesting assignments.")
        return pd.DataFrame({"document": self._documents, "topic": self._topics})

    def topic_count(self) -> int:
        """Return the number of topics found, excluding outliers.

        Returns:
            Count of topics with id >= 0.
        """
        return sum(1 for t in self.model.get_topics() if t != -1)

    # -- Post-processing ----------------------------------------------------

    def reduce_topics(self, target: int) -> TopicModeler:
        """Merge topics down to a target number.

        Useful when HDBSCAN produces more granularity than needed for
        reporting.

        Args:
            target: Desired number of topics after reduction.

        Returns:
            The modeler, for chaining.
        """
        if self._documents is None:
            raise RuntimeError("Call fit() before reducing topics.")
        self.model.reduce_topics(self._documents, nr_topics=target)
        self._topics = list(self.model.topics_)
        return self

    # -- Visualisation ------------------------------------------------------

    def plot_topics(self) -> PlotlyFigure:
        """Interactive intertopic distance map.

        Returns:
            A Plotly figure (call ``.show()`` in a notebook).
        """
        return self.model.visualize_topics()

    def plot_barchart(self, top_k_topics: int = 8) -> PlotlyFigure:
        """Bar charts of the top terms for the leading topics.

        Args:
            top_k_topics: Number of topics to display.

        Returns:
            A Plotly figure.
        """
        return self.model.visualize_barchart(top_n_topics=top_k_topics)

    def plot_hierarchy(self) -> PlotlyFigure:
        """Hierarchical clustering dendrogram of the topics.

        Returns:
            A Plotly figure.
        """
        return self.model.visualize_hierarchy()

    # -- Persistence --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the fitted model to disk.

        Args:
            path: Destination directory or file for the model.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(destination))
        print(f"Topic model saved to '{destination}'.")

    @classmethod
    def load(cls, path: str | Path) -> TopicModeler:
        """Load a previously saved model into a new modeler.

        Args:
            path: Path passed to :meth:`bertopic.BERTopic.load`.

        Returns:
            A modeler wrapping the loaded model. Note that the original
            documents are not restored, so document-level methods
            (:meth:`assignments`, :meth:`reduce_topics`) are unavailable
            until a new :meth:`fit`.

        Raises:
            ImportError: If BERTopic is not installed.
        """
        try:
            from bertopic import BERTopic
        except ImportError as exc:
            raise ImportError(
                "Loading a topic model requires 'bertopic'. "
                "Install it with: poetry add bertopic"
            ) from exc

        instance = cls()
        instance._model = BERTopic.load(str(path))
        return instance
