"""
mod_nmf.py

Pipeline de producción para extracción de tópicos mediante NMF
(Non-Negative Matrix Factorization) sobre observaciones de encuestas.

Ubicación esperada: <project_root>/src/mod_nmf.py

Este script asume que los parámetros de modelado (n_features, k_final,
variante ganadora, umbral de asignación) ya fueron decididos previamente
mediante el proceso exploratorio documentado en el notebook
`notebooks/modelo_nmf-prueba.ipynb` (método del codo + coherencia semántica
con gensim, comparación de las 4 variantes de NMF, refinamiento de k).

Este script NO vuelve a explorar esos parámetros — los aplica directamente.
"""

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


# ----------------------------------------------------------------------------
# Configuración / parámetros finales
# (decididos previamente en el notebook exploratorio; ver docstring del módulo)
# ----------------------------------------------------------------------------

TEXT_COLUMN = "obs20_no_stopwords_no_adverbs"
SHEET_NAME = "obs20"

N_FEATURES = 3000          # cobertura ~91.7% de ocurrencias del corpus (diagnóstico)
N_TOP_WORDS = 20            # palabras mostradas por tópico en el gráfico
K_FINAL = 14                # número de tópicos, tras método del codo + coherencia (c_v)
INIT = "nndsvda"

# Variante ganadora: NMF - Frobenius norm
NMF_PARAMS = dict(
    n_components=K_FINAL,
    random_state=1,
    init=INIT,
    beta_loss="frobenius",
    alpha_W=0.00005,
    alpha_H=0.00005,
    l1_ratio=1,
)

TOPIC_THRESHOLD = 0.10       # peso mínimo normalizado para que un tópico se asigne a una fila
MAX_TOPICS_PER_ROW = 5
NO_TOPIC_LABEL = "SIN_TOPICO"
MAX_WORDS_FOR_TOPIC_NAME = 4  # tope al desambiguar nombres de tópicos duplicados

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------

def find_project_root(marker: str = "pyproject.toml") -> Path:
    """Busca la raíz del proyecto subiendo desde la ubicación de ESTE archivo
    (no desde el directorio de trabajo, para que el script sea independiente
    de dónde se invoque)."""
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(
        f"No se encontró '{marker}' en ningún directorio padre de {here}.\n"
        "Verifica que pyproject.toml exista en la raíz del proyecto."
    )


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
EXCEL_PATH = DATA_DIR / "processed" / "obs20_nostopwords_noadverbs.xlsx"
FIGURE_PATH = PROJECT_ROOT / "figures" / "nmf"
OUTPUT_PATH = DATA_DIR / "processed" / f"obs20_con_topicos_nmf_frobenius_k{K_FINAL}.xlsx"


# ----------------------------------------------------------------------------
# Carga de datos
# ----------------------------------------------------------------------------

def load_data(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    logger.info("Cargando datos desde %s", excel_path)
    data = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    logger.info("Datos cargados: %d filas", len(data))
    return data


# ----------------------------------------------------------------------------
# TF-IDF
# ----------------------------------------------------------------------------

def build_tfidf(data: pd.DataFrame, text_column: str, n_features: int):
    logger.info("Construyendo matriz TF-IDF (n_features=%d)...", n_features)
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features)
    tfidf = vectorizer.fit_transform(data[text_column])
    logger.info("TF-IDF construido: shape=%s", tfidf.shape)
    return tfidf, vectorizer


# ----------------------------------------------------------------------------
# Entrenamiento del modelo final
# ----------------------------------------------------------------------------

def train_final_model(tfidf, nmf_params: dict) -> NMF:
    logger.info("Entrenando modelo NMF final (Frobenius, k=%d)...", nmf_params["n_components"])
    model = NMF(**nmf_params).fit(tfidf)
    logger.info("Modelo entrenado.")
    return model


# ----------------------------------------------------------------------------
# Visualización de tópicos (grilla dinámica según n_components)
# ----------------------------------------------------------------------------

def plot_top_words(model, feature_names, n_top_words, title,
                    save_dir: Path = FIGURE_PATH, filename: str = None):
    n_topics = model.components_.shape[0]

    n_cols = math.ceil(n_topics / 2)
    n_rows = 2 if n_topics > n_cols else 1
    while n_rows * n_cols < n_topics:
        n_cols += 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 7.5 * n_rows), sharex=True)
    axes = np.array(axes).flatten()

    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 30})
        ax.tick_params(axis="both", which="major", labelsize=20)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        fig.suptitle(title, fontsize=40)

    for empty_idx in range(n_topics, len(axes)):
        axes[empty_idx].axis("off")

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura de tópicos guardada en %s", save_path)


# ----------------------------------------------------------------------------
# Nombres automáticos de tópicos (con validación de duplicados)
# ----------------------------------------------------------------------------

def name_topics(model, feature_names, n_words_for_name: int = 1) -> dict:
    topic_names = {}
    for topic_idx, topic in enumerate(model.components_):
        top_idx = topic.argsort()[-n_words_for_name:][::-1]
        top_words = [feature_names[i] for i in top_idx]
        topic_names[topic_idx] = "_".join(top_words)
    return topic_names


def check_duplicate_names(topic_names: dict) -> dict:
    name_to_topics = {}
    for topic_idx, name in topic_names.items():
        name_to_topics.setdefault(name, []).append(topic_idx)
    return {name: idxs for name, idxs in name_to_topics.items() if len(idxs) > 1}


def resolve_topic_names(model, feature_names,
                         max_words_for_name: int = MAX_WORDS_FOR_TOPIC_NAME) -> dict:
    n_words_for_name = 1
    topic_names = name_topics(model, feature_names, n_words_for_name)
    duplicates = check_duplicate_names(topic_names)

    while duplicates and n_words_for_name < max_words_for_name:
        n_words_for_name += 1
        topic_names = name_topics(model, feature_names, n_words_for_name)
        duplicates = check_duplicate_names(topic_names)

    if duplicates:
        logger.warning(
            "Persisten nombres de tópicos duplicados incluso con n_words_for_name=%d: %s",
            n_words_for_name, duplicates,
        )
    else:
        logger.info("Nombres de tópicos únicos logrados con n_words_for_name=%d", n_words_for_name)

    return topic_names


# ----------------------------------------------------------------------------
# Asignación de tópicos a nivel fila
# ----------------------------------------------------------------------------

def assign_row_topics(W_normalized: np.ndarray, topic_names: dict,
                       threshold: float = TOPIC_THRESHOLD,
                       max_topics: int = MAX_TOPICS_PER_ROW) -> list:
    row_topics = []
    for row_weights in W_normalized:
        sorted_idx = np.argsort(row_weights)[::-1]
        selected = []
        for idx in sorted_idx:
            if row_weights[idx] >= threshold and len(selected) < max_topics:
                selected.append(topic_names[idx])
            else:
                break
        row_topics.append(", ".join(selected))
    return row_topics


def normalize_weights(W: np.ndarray) -> np.ndarray:
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # evita división por cero en filas sin señal
    return W / row_sums


# ----------------------------------------------------------------------------
# Flujo principal
# ----------------------------------------------------------------------------

def main():
    data = load_data(EXCEL_PATH, SHEET_NAME)

    tfidf, tfidf_vectorizer = build_tfidf(data, TEXT_COLUMN, N_FEATURES)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    final_model = train_final_model(tfidf, NMF_PARAMS)

    plot_top_words(
        final_model,
        feature_names,
        N_TOP_WORDS,
        f"Topics in final model (nmf_frobenius, k={K_FINAL})",
        filename=f"topics_final_nmf_frobenius_k{K_FINAL}.png",
    )

    topic_names = resolve_topic_names(final_model, feature_names)
    for idx, name in topic_names.items():
        logger.info("Topic %d: %s", idx + 1, name)

    W = final_model.transform(tfidf)
    W_normalized = normalize_weights(W)

    data["topicos_nmf"] = assign_row_topics(
        W_normalized, topic_names, TOPIC_THRESHOLD, MAX_TOPICS_PER_ROW
    )
    data["topicos_nmf"] = data["topicos_nmf"].replace("", NO_TOPIC_LABEL)

    n_sin_topico = (data["topicos_nmf"] == NO_TOPIC_LABEL).sum()
    logger.info(
        "Filas sin tópico asignado (umbral=%.0f%%): %d de %d (%.1f%%)",
        TOPIC_THRESHOLD * 100, n_sin_topico, len(data), n_sin_topico / len(data) * 100,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_excel(OUTPUT_PATH, index=False)
    logger.info("Archivo final guardado en %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
