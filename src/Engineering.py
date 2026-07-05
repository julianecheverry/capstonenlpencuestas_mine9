"""Feature Engineering module for the NLP survey corpus.

Phase 3 of the NLP pipeline. Transforms clean text (output of
``limpieza.py``) into numeric feature matrices ready for modelling:

* Bag of N-Grams via :class:`sklearn.feature_extraction.text.CountVectorizer`
* TF-IDF via :class:`sklearn.feature_extraction.text.TfidfVectorizer`
* Vocabulary and weight inspection utilities

Typical usage::

    from Engineering import BagOfNgrams, TfIdfTransformer

    bow = BagOfNgrams(ngram_range=(1, 2), min_df=3)
    bow.fit_transform(corpus_series)
    bow.info()
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

if TYPE_CHECKING:
    from scipy.sparse import spmatrix


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _prepare_corpus(corpus: pd.Series) -> list[str]:
    """Convert a pandas Series to a clean list of strings.

    Args:
        corpus: Text Series (may contain NaN or non-string values).

    Returns:
        List of non-null string documents.

    Raises:
        ValueError: If the corpus is empty after cleaning.
    """
    texts = corpus.fillna("").astype(str).tolist()
    if not any(t.strip() for t in texts):
        raise ValueError("The corpus is empty after cleaning.")
    return texts


def _ensure_fitted(matrix: spmatrix | None) -> spmatrix:
    """Guard that raises if the vectoriser has not been fitted.

    Args:
        matrix: Internal sparse matrix (``None`` before fitting).

    Returns:
        The validated sparse matrix.

    Raises:
        RuntimeError: When called before ``fit_transform()``.
    """
    if matrix is None:
        raise RuntimeError("Call fit_transform() first.")
    return matrix


# ===================================================================
# 1. BAG OF N-GRAMS
# ===================================================================
class BagOfNgrams:
    """Wrapper around :class:`CountVectorizer`.

    Generates a count matrix (Bag of N-Grams), an interpretable
    DataFrame and exposes the resulting vocabulary.

    Args:
        ngram_range: Range of n-gram sizes, e.g. ``(1, 2)``.
        min_df: Minimum document frequency (int=absolute, float=ratio).
        max_df: Maximum document frequency (ratio).
        max_features: Cap on retained features.
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int | float = 1,
        max_df: float = 1.0,
        max_features: int | None = None,
    ) -> None:
        self._vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
        )
        self._matrix: spmatrix | None = None
        self._feature_names: list[str] | None = None

    # -- Core API -----------------------------------------------------------

    def fit_transform(self, corpus: pd.Series) -> spmatrix:
        """Fit the vectoriser and return the sparse count matrix.

        Args:
            corpus: Series of clean text documents.

        Returns:
            Sparse matrix of shape ``(n_docs, n_terms)``.
        """
        texts = _prepare_corpus(corpus)
        self._matrix = self._vectorizer.fit_transform(texts)
        self._feature_names = list(
            self._vectorizer.get_feature_names_out()
        )
        return self._matrix

    # -- Properties (guard with _ensure_fitted) -----------------------------

    @property
    def matrix(self) -> spmatrix:
        """Fitted sparse count matrix."""
        return _ensure_fitted(self._matrix)

    @property
    def feature_names(self) -> list[str]:
        """Ordered list of vocabulary terms."""
        _ensure_fitted(self._matrix)
        assert self._feature_names is not None  # noqa: S101
        return self._feature_names

    # -- Inspection ---------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the sparse matrix to a dense DataFrame.

        Returns:
            DataFrame with terms as column headers.

        Note:
            Use only for inspection on small corpora.
        """
        return pd.DataFrame(
            self.matrix.toarray(),
            columns=self.feature_names,
        )

    def vocabulary(self) -> dict[str, int]:
        """Return the vocabulary mapping ``{term: column_index}``.

        Returns:
            Vocabulary dictionary.
        """
        return dict(self._vectorizer.vocabulary_)

    def info(self) -> dict[str, int | float]:
        """Summary of matrix dimensions and sparsity.

        Returns:
            Dict with ``n_documents``, ``n_terms``,
            ``non_zero_elements`` and ``density_pct``.
        """
        mat = self.matrix
        total = mat.shape[0] * mat.shape[1]
        return {
            "n_documents": mat.shape[0],
            "n_terms": mat.shape[1],
            "non_zero_elements": mat.nnz,
            "density_pct": round(mat.nnz / total * 100, 4) if total else 0.0,
        }


# ===================================================================
# 2. TF-IDF
# ===================================================================
class TfIdfTransformer:
    """Wrapper around :class:`TfidfVectorizer`.

    Generates the TF-IDF matrix and provides inspection methods
    for global and per-document term weights.

    Args:
        ngram_range: Range of n-gram sizes.
        min_df: Minimum document frequency.
        max_df: Maximum document frequency.
        max_features: Cap on retained features.
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int | float = 1,
        max_df: float = 1.0,
        max_features: int | None = None,
    ) -> None:
        self._vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
        )
        self._matrix: spmatrix | None = None
        self._feature_names: list[str] | None = None

    # -- Core API -----------------------------------------------------------

    def fit_transform(self, corpus: pd.Series) -> spmatrix:
        """Fit the vectoriser and return the sparse TF-IDF matrix.

        Args:
            corpus: Series of clean text documents.

        Returns:
            Sparse TF-IDF matrix of shape ``(n_docs, n_terms)``.
        """
        texts = _prepare_corpus(corpus)
        self._matrix = self._vectorizer.fit_transform(texts)
        self._feature_names = list(
            self._vectorizer.get_feature_names_out()
        )
        return self._matrix

    # -- Properties ---------------------------------------------------------

    @property
    def matrix(self) -> spmatrix:
        """Fitted sparse TF-IDF matrix."""
        return _ensure_fitted(self._matrix)

    @property
    def feature_names(self) -> list[str]:
        """Ordered list of vocabulary terms."""
        _ensure_fitted(self._matrix)
        assert self._feature_names is not None  # noqa: S101
        return self._feature_names

    # -- Basic inspection ---------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the sparse matrix to a dense DataFrame.

        Returns:
            DataFrame with terms as column headers.
        """
        return pd.DataFrame(
            self.matrix.toarray(),
            columns=self.feature_names,
        )

    def vocabulary(self) -> dict[str, int]:
        """Return the vocabulary mapping ``{term: column_index}``.

        Returns:
            Vocabulary dictionary.
        """
        return dict(self._vectorizer.vocabulary_)

    def info(self) -> dict[str, int | float]:
        """Summary of matrix dimensions and sparsity.

        Returns:
            Dict with ``n_documents``, ``n_terms``,
            ``non_zero_elements`` and ``density_pct``.
        """
        mat = self.matrix
        total = mat.shape[0] * mat.shape[1]
        return {
            "n_documents": mat.shape[0],
            "n_terms": mat.shape[1],
            "non_zero_elements": mat.nnz,
            "density_pct": round(mat.nnz / total * 100, 4) if total else 0.0,
        }

    # -- Advanced inspection ------------------------------------------------

    def global_term_ranking(self, top_k: int = 30) -> pd.DataFrame:
        """Rank terms by mean TF-IDF weight across all documents.

        Args:
            top_k: Number of terms to return.

        Returns:
            DataFrame with ``Term``, ``Mean_TFIDF`` and ``Max_TFIDF``.
        """
        mean_w = np.asarray(self.matrix.mean(axis=0)).ravel()
        max_w = np.asarray(self.matrix.max(axis=0).toarray()).ravel()
        idx_sorted = mean_w.argsort()[::-1][:top_k]

        rows = [
            {
                "Term": self.feature_names[i],
                "Mean_TFIDF": round(float(mean_w[i]), 6),
                "Max_TFIDF": round(float(max_w[i]), 6),
            }
            for i in idx_sorted
        ]
        return pd.DataFrame(rows)

    def top_terms_per_document(
        self,
        doc_index: int,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Return the highest-weight terms for a single document.

        Args:
            doc_index: Row index of the document.
            top_k: Number of terms to return.

        Returns:
            DataFrame with ``Term`` and ``TFIDF_Weight``.

        Raises:
            IndexError: If *doc_index* exceeds the matrix row count.
        """
        if doc_index >= self.matrix.shape[0]:
            raise IndexError(
                f"doc_index {doc_index} out of range "
                f"(max {self.matrix.shape[0] - 1})."
            )
        row = self.matrix[doc_index].toarray().ravel()
        idx_sorted = row.argsort()[::-1][:top_k]
        rows = [
            {
                "Term": self.feature_names[i],
                "TFIDF_Weight": round(float(row[i]), 6),
            }
            for i in idx_sorted
            if row[i] > 0
        ]
        return pd.DataFrame(rows)

    def search_term(self, term: str) -> dict[str, int | float] | None:
        """Look up a specific term in the vocabulary.

        Args:
            term: The term to search for.

        Returns:
            Dict with ``term``, ``index``, ``mean_tfidf``,
            ``max_tfidf`` and ``document_count``;
            ``None`` if the term is absent.
        """
        vocab = self.vocabulary()
        if term not in vocab:
            return None
        idx = vocab[term]
        col = self.matrix[:, idx].toarray().ravel()
        return {
            "term": term,
            "index": idx,
            "mean_tfidf": round(float(col.mean()), 6),
            "max_tfidf": round(float(col.max()), 6),
            "document_count": int((col > 0).sum()),
        }


# ===================================================================
# 3. COMBINED UTILITIES
# ===================================================================
def compare_vocabularies(
    bow: BagOfNgrams,
    tfidf: TfIdfTransformer,
) -> pd.DataFrame:
    """Compare vocabularies produced by BoW and TF-IDF.

    Args:
        bow: Fitted :class:`BagOfNgrams` instance.
        tfidf: Fitted :class:`TfIdfTransformer` instance.

    Returns:
        DataFrame summarising overlap and differences.
    """
    bow_v = set(bow.feature_names)
    tfidf_v = set(tfidf.feature_names)
    return pd.DataFrame(
        {
            "Metric": [
                "BoW terms",
                "TF-IDF terms",
                "Intersection",
                "Only in BoW",
                "Only in TF-IDF",
            ],
            "Value": [
                len(bow_v),
                len(tfidf_v),
                len(bow_v & tfidf_v),
                len(bow_v - tfidf_v),
                len(tfidf_v - bow_v),
            ],
        }
    )


def build_feature_matrices(
    corpus: pd.Series,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    max_features: int | None = 5000,
) -> tuple[BagOfNgrams, TfIdfTransformer]:
    """Convenience builder: create BoW and TF-IDF with shared params.

    Args:
        corpus: Series of clean text.
        ngram_range: N-gram range.
        min_df: Minimum document frequency.
        max_df: Maximum document frequency.
        max_features: Feature cap.

    Returns:
        Tuple ``(BagOfNgrams, TfIdfTransformer)`` already fitted.
    """
    shared_kwargs = {
        "ngram_range": ngram_range,
        "min_df": min_df,
        "max_df": max_df,
        "max_features": max_features,
    }
    bow = BagOfNgrams(**shared_kwargs)
    tfidf = TfIdfTransformer(**shared_kwargs)
    bow.fit_transform(corpus)
    tfidf.fit_transform(corpus)

    print("\u2500\u2500 Feature matrices built \u2500\u2500")
    print(f"  Bag of N-Grams : {bow.info()}")
    print(f"  TF-IDF         : {tfidf.info()}")
    return bow, tfidf
