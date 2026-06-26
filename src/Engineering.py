# =============================================================================
# Engineering.py — Módulo de Ingeniería de Características para PLN
# =============================================================================
# Fase 3 del pipeline NLP.
# Recibe textos limpios (salida de limpieza.py → EDA.py) y genera:
#   • Bolsa de N-Gramas (CountVectorizer)
#   • TF-IDF (TfidfVectorizer)
#   • Funciones de inspección de vocabulario y pesos
# =============================================================================

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# ── 1. BOLSA DE N-GRAMAS ─────────────────────────────────────────────────────

class BagOfNgrams:
    """
    Wrapper sobre CountVectorizer de scikit-learn.
    Genera la matriz de conteo (Bag of N-Grams), un DataFrame interpretable
    y expone el vocabulario resultante.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 1),
        min_df: int | float = 1,
        max_df: float = 1.0,
        max_features: Optional[int] = None,
    ):
        """
        Args:
            ngram_range: Rango de n-gramas, e.g. (1,2) para uni+bigramas.
            min_df: Frecuencia mínima de documento (absoluta si int, relativa si float).
            max_df: Frecuencia máxima de documento (relativa).
            max_features: Número máximo de características a retener.
        """
        self.vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
        )
        self._matrix: Optional[spmatrix] = None
        self._feature_names: Optional[List[str]] = None

    def fit_transform(self, corpus: pd.Series) -> spmatrix:
        """
        Ajusta el vectorizador y transforma el corpus.

        Args:
            corpus: Serie de textos limpios.

        Returns:
            Matriz dispersa de conteo (documentos × términos).
        """
        texts = corpus.fillna("").astype(str).tolist()
        self._matrix = self.vectorizer.fit_transform(texts)
        self._feature_names = list(self.vectorizer.get_feature_names_out())
        return self._matrix

    @property
    def matrix(self) -> spmatrix:
        if self._matrix is None:
            raise RuntimeError("Primero ejecute fit_transform().")
        return self._matrix

    @property
    def feature_names(self) -> List[str]:
        if self._feature_names is None:
            raise RuntimeError("Primero ejecute fit_transform().")
        return self._feature_names

    def to_dataframe(self) -> pd.DataFrame:
        """Convierte la matriz dispersa en un DataFrame denso (útil para inspección, no para datasets grandes)."""
        return pd.DataFrame(
            self.matrix.toarray(),
            columns=self.feature_names,
        )

    def vocabulary(self) -> Dict[str, int]:
        """Retorna el vocabulario {término: índice}."""
        return dict(self.vectorizer.vocabulary_)

    def info(self) -> dict:
        """Resumen de dimensiones y vocabulario."""
        m = self.matrix
        return {
            "n_documentos": m.shape[0],
            "n_terminos": m.shape[1],
            "elementos_no_cero": m.nnz,
            "densidad_%": round(m.nnz / (m.shape[0] * m.shape[1]) * 100, 4),
        }


# ── 2. TF-IDF ────────────────────────────────────────────────────────────────

class TfIdfTransformer:
    """
    Wrapper sobre TfidfVectorizer de scikit-learn.
    Genera la matriz TF-IDF y funciones de inspección de pesos.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 1),
        min_df: int | float = 1,
        max_df: float = 1.0,
        max_features: Optional[int] = None,
    ):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
        )
        self._matrix: Optional[spmatrix] = None
        self._feature_names: Optional[List[str]] = None

    def fit_transform(self, corpus: pd.Series) -> spmatrix:
        """Ajusta y transforma el corpus. Retorna la matriz TF-IDF dispersa."""
        texts = corpus.fillna("").astype(str).tolist()
        self._matrix = self.vectorizer.fit_transform(texts)
        self._feature_names = list(self.vectorizer.get_feature_names_out())
        return self._matrix

    @property
    def matrix(self) -> spmatrix:
        if self._matrix is None:
            raise RuntimeError("Primero ejecute fit_transform().")
        return self._matrix

    @property
    def feature_names(self) -> List[str]:
        if self._feature_names is None:
            raise RuntimeError("Primero ejecute fit_transform().")
        return self._feature_names

    def to_dataframe(self) -> pd.DataFrame:
        """Convierte la matriz TF-IDF en DataFrame denso."""
        return pd.DataFrame(
            self.matrix.toarray(),
            columns=self.feature_names,
        )

    def vocabulary(self) -> Dict[str, int]:
        return dict(self.vectorizer.vocabulary_)

    def info(self) -> dict:
        m = self.matrix
        return {
            "n_documentos": m.shape[0],
            "n_terminos": m.shape[1],
            "elementos_no_cero": m.nnz,
            "densidad_%": round(m.nnz / (m.shape[0] * m.shape[1]) * 100, 4),
        }

    # ---- Inspección de pesos ----

    def global_term_ranking(self, top_k: int = 30) -> pd.DataFrame:
        """
        Ranking global de términos por peso TF-IDF promedio.

        Args:
            top_k: Número de términos a retornar.

        Returns:
            DataFrame con columnas: Término, TF-IDF_Promedio, TF-IDF_Max.
        """
        mean_tfidf = np.asarray(self.matrix.mean(axis=0)).ravel()
        max_tfidf = np.asarray(self.matrix.max(axis=0).toarray()).ravel()
        idx_sorted = mean_tfidf.argsort()[::-1][:top_k]

        rows = []
        for i in idx_sorted:
            rows.append({
                "Término": self.feature_names[i],
                "TF-IDF_Promedio": round(mean_tfidf[i], 6),
                "TF-IDF_Max": round(max_tfidf[i], 6),
            })
        return pd.DataFrame(rows)

    def top_terms_per_document(self, doc_index: int, top_k: int = 10) -> pd.DataFrame:
        """
        Retorna los términos con mayor peso TF-IDF para un documento dado.

        Args:
            doc_index: Índice del documento (fila).
            top_k: Número de términos a retornar.
        """
        row = self.matrix[doc_index].toarray().ravel()
        idx_sorted = row.argsort()[::-1][:top_k]

        rows = []
        for i in idx_sorted:
            if row[i] > 0:
                rows.append({
                    "Término": self.feature_names[i],
                    "Peso_TF-IDF": round(row[i], 6),
                })
        return pd.DataFrame(rows)

    def search_term(self, term: str) -> Optional[dict]:
        """
        Consulta un término específico: devuelve su índice, peso promedio y peso máximo.
        Retorna None si el término no existe en el vocabulario.
        """
        vocab = self.vocabulary()
        if term not in vocab:
            return None
        idx = vocab[term]
        col = self.matrix[:, idx].toarray().ravel()
        return {
            "término": term,
            "índice": idx,
            "tf_idf_promedio": round(col.mean(), 6),
            "tf_idf_max": round(col.max(), 6),
            "documentos_con_termino": int((col > 0).sum()),
        }


# ── 3. FUNCIONES DE INSPECCIÓN COMBINADAS ────────────────────────────────────

def compare_vocabularies(
    bow: BagOfNgrams,
    tfidf: TfIdfTransformer,
) -> pd.DataFrame:
    """Compara los vocabularios generados por BoW y TF-IDF."""
    bow_v = set(bow.feature_names)
    tfidf_v = set(tfidf.feature_names)
    return pd.DataFrame({
        "Métrica": [
            "Términos BoW",
            "Términos TF-IDF",
            "Intersección",
            "Solo en BoW",
            "Solo en TF-IDF",
        ],
        "Valor": [
            len(bow_v),
            len(tfidf_v),
            len(bow_v & tfidf_v),
            len(bow_v - tfidf_v),
            len(tfidf_v - bow_v),
        ],
    })


def build_feature_matrices(
    corpus: pd.Series,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    max_features: Optional[int] = 5000,
) -> Tuple[BagOfNgrams, TfIdfTransformer]:
    """
    Función de conveniencia: construye BoW y TF-IDF con los mismos parámetros.

    Returns:
        Tupla (BagOfNgrams, TfIdfTransformer) ya ajustados.
    """
    bow = BagOfNgrams(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
    )
    tfidf = TfIdfTransformer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
    )
    bow.fit_transform(corpus)
    tfidf.fit_transform(corpus)

    print("── Matrices generadas ──")
    print(f"  Bag of N-Grams : {bow.info()}")
    print(f"  TF-IDF         : {tfidf.info()}")
    return bow, tfidf
