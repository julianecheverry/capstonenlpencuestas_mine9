# =============================================================================
# EDA.py — Módulo de Análisis Exploratorio de Datos para PLN
# =============================================================================
# Fase 2 del pipeline NLP.
# Recibe DataFrames ya procesados por limpieza.py y ejecuta:
#   • Distribución de longitud de textos
#   • Nubes de palabras
#   • Análisis de N-Gramas (uni / bi / tri)
#   • Visualizaciones reutilizables
# =============================================================================

from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from wordcloud import WordCloud  # type: ignore[import]
except ImportError:
    WordCloud = None

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)


# ── 1. UTILIDADES ────────────────────────────────────────────────────────────

def _get_stopwords(extra: Optional[List[str]] = None) -> set:
    """Devuelve el conjunto de stopwords en español (NLTK) + términos adicionales."""
    from nltk.corpus import stopwords as _sw

    sw = set(_sw.words("spanish"))
    domain_terms = {
        "académico", "academia", "universidad", "si", "no", "ns",
        "na", "nan", "ninguno", "ninguna", "nada", "ns/nc",
    }
    sw |= domain_terms
    if extra:
        sw |= set(w.lower() for w in extra)
    return sw


def _tokenize(text: str) -> List[str]:
    """Tokenización simple basada en split (ya se asume texto limpio)."""
    if not isinstance(text, str) or text.strip() == "":
        return []
    return text.split()


# ── 2. DISTRIBUCIÓN DE LONGITUD DE TEXTOS ────────────────────────────────────

class TextLengthAnalyzer:
    """Calcula estadísticas de longitud (caracteres y palabras) sobre una columna de texto."""

    def __init__(self, series: pd.Series, column_name: str = "texto"):
        self.series = series.dropna().astype(str)
        self.column_name = column_name
        self._char_lengths = self.series.str.len()
        self._word_lengths = self.series.apply(lambda t: len(t.split()))

    # ---- Estadísticos descriptivos ----

    def summary(self) -> pd.DataFrame:
        """Retorna un DataFrame con estadísticos descriptivos de longitud."""
        metrics: Dict[str, list] = {
            "Métrica": [],
            "Caracteres": [],
            "Palabras": [],
        }

        def _add(name: str, char_val, word_val):
            metrics["Métrica"].append(name)
            metrics["Caracteres"].append(round(char_val, 2))
            metrics["Palabras"].append(round(word_val, 2))

        _add("Conteo", len(self._char_lengths), len(self._word_lengths))
        _add("Promedio", self._char_lengths.mean(), self._word_lengths.mean())
        _add("Mediana", self._char_lengths.median(), self._word_lengths.median())
        _add("Mínimo", self._char_lengths.min(), self._word_lengths.min())
        _add("Máximo", self._char_lengths.max(), self._word_lengths.max())
        _add("Desv. Estándar", self._char_lengths.std(), self._word_lengths.std())
        for p in [10, 25, 75, 90]:
            _add(
                f"Percentil {p}",
                np.percentile(self._char_lengths, p),
                np.percentile(self._word_lengths, p),
            )
        return pd.DataFrame(metrics)

    # ---- Visualizaciones ----

    def plot_histograms(self, bins: int = 40, figsize: Tuple[int, int] = (14, 5)):
        """Histogramas de longitud por caracteres y palabras."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].hist(self._char_lengths, bins=bins, color="#4C72B0", edgecolor="white", alpha=0.85)
        axes[0].set_title(f"Distribución de caracteres — {self.column_name}", fontsize=11)
        axes[0].set_xlabel("Nº de caracteres")
        axes[0].set_ylabel("Frecuencia")

        axes[1].hist(self._word_lengths, bins=bins, color="#55A868", edgecolor="white", alpha=0.85)
        axes[1].set_title(f"Distribución de palabras — {self.column_name}", fontsize=11)
        axes[1].set_xlabel("Nº de palabras")
        axes[1].set_ylabel("Frecuencia")

        plt.tight_layout()
        return fig

    def plot_boxplots(self, figsize: Tuple[int, int] = (14, 5)):
        """Boxplots de longitud por caracteres y palabras."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        sns.boxplot(x=self._char_lengths, ax=axes[0], color="#4C72B0")
        axes[0].set_title(f"Boxplot de caracteres — {self.column_name}", fontsize=11)

        sns.boxplot(x=self._word_lengths, ax=axes[1], color="#55A868")
        axes[1].set_title(f"Boxplot de palabras — {self.column_name}", fontsize=11)

        plt.tight_layout()
        return fig

    def plot_comparative(self, figsize: Tuple[int, int] = (14, 5)):
        """Distribución superpuesta (KDE) de caracteres y palabras normalizadas."""
        fig, ax = plt.subplots(figsize=figsize)
        sns.kdeplot(self._char_lengths / self._char_lengths.max(), label="Caracteres (norm.)", ax=ax, fill=True, alpha=0.3)
        sns.kdeplot(self._word_lengths / self._word_lengths.max(), label="Palabras (norm.)", ax=ax, fill=True, alpha=0.3)
        ax.set_title(f"Distribución comparativa normalizada — {self.column_name}", fontsize=11)
        ax.set_xlabel("Valor normalizado [0,1]")
        ax.legend()
        plt.tight_layout()
        return fig


# ── 3. NUBES DE PALABRAS ─────────────────────────────────────────────────────

class WordCloudGenerator:
    """Genera nubes de palabras (globales, por columna o por categoría)."""

    DEFAULT_WC_PARAMS = dict(
        width=900,
        height=450,
        background_color="white",
        colormap="viridis",
        max_words=200,
        collocations=False,
    )

    def __init__(self, extra_stopwords: Optional[List[str]] = None):
        self.stopwords = _get_stopwords(extra_stopwords)

    def _build_wc(self, text: str, **kwargs) -> WordCloud:
        params = {**self.DEFAULT_WC_PARAMS, **kwargs, "stopwords": self.stopwords}
        return WordCloud(**params).generate(text)

    def generate(
        self,
        series: pd.Series,
        title: str = "Nube de Palabras",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None,
        **wc_kwargs,
    ):
        """Genera y muestra una nube de palabras para una Serie de texto."""
        corpus = " ".join(series.dropna().astype(str))
        wc = self._build_wc(corpus, **wc_kwargs)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(title, fontsize=13)
        ax.axis("off")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def generate_by_category(
        self,
        df: pd.DataFrame,
        text_col: str,
        cat_col: str,
        figsize_per_plot: Tuple[int, int] = (10, 5),
        save_dir: Optional[str] = None,
        **wc_kwargs,
    ) -> List:
        """Genera una nube de palabras por cada categoría única en *cat_col*."""
        figs = []
        categories = df[cat_col].dropna().unique()
        for cat in sorted(categories):
            subset = df.loc[df[cat_col] == cat, text_col]
            if subset.dropna().empty:
                continue
            title = f"WordCloud — {text_col[:50]} | {cat_col}={cat}"
            sp = None
            if save_dir:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                sp = str(Path(save_dir) / f"wc_{cat_col}_{cat}.png")
            fig = self.generate(subset, title=title, figsize=figsize_per_plot, save_path=sp, **wc_kwargs)
            figs.append(fig)
        return figs


# ── 4. ANÁLISIS DE N-GRAMAS ──────────────────────────────────────────────────

class NgramAnalyzer:
    """Construye y visualiza unigramas, bigramas y trigramas."""

    def __init__(self, series: pd.Series, extra_stopwords: Optional[List[str]] = None):
        self.series = series.dropna().astype(str)
        self.stopwords = _get_stopwords(extra_stopwords)
        self._tokens_cache: Optional[List[List[str]]] = None

    @property
    def tokens(self) -> List[List[str]]:
        if self._tokens_cache is None:
            self._tokens_cache = [
                [w for w in _tokenize(t) if w.lower() not in self.stopwords]
                for t in self.series
            ]
        return self._tokens_cache

    # ---- Cálculo de n-gramas ----

    def _build_ngrams(self, n: int) -> List[Tuple[str, ...]]:
        ngrams = []
        for tok_list in self.tokens:
            ngrams.extend(zip(*(tok_list[i:] for i in range(n))))
        return ngrams

    def frequency_table(self, n: int = 1) -> pd.DataFrame:
        """
        Retorna un DataFrame con frecuencia absoluta y relativa para n-gramas.

        Args:
            n: Tamaño del n-grama (1=unigrama, 2=bigrama, 3=trigrama).
        """
        ngrams = self._build_ngrams(n)
        counts = Counter(ngrams)
        total = sum(counts.values()) or 1

        rows = []
        for gram, freq in counts.most_common():
            label = " ".join(gram)
            rows.append({"Ngrama": label, "Frecuencia": freq, "Frecuencia_Relativa": round(freq / total, 6)})
        return pd.DataFrame(rows)

    # ---- Visualización ----

    def plot_top_ngrams(
        self,
        n: int = 1,
        top_k: int = 20,
        figsize: Tuple[int, int] = (12, 7),
        title: Optional[str] = None,
        color: str = "#4C72B0",
    ):
        """Gráfico de barras horizontales con los *top_k* n-gramas más frecuentes."""
        df = self.frequency_table(n).head(top_k).iloc[::-1]
        label_map = {1: "Unigramas", 2: "Bigramas", 3: "Trigramas"}
        n_label = label_map.get(n, f"{n}-gramas")

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(df["Ngrama"], df["Frecuencia"], color=color, edgecolor="white")
        ax.set_xlabel("Frecuencia absoluta")
        ax.set_title(title or f"Top {top_k} {n_label}", fontsize=12)
        plt.tight_layout()
        return fig

    def plot_comparative_ngrams(
        self,
        n: int = 1,
        tops: Tuple[int, int] = (20, 30),
        figsize: Tuple[int, int] = (16, 7),
    ):
        """Genera dos subplots lado a lado para dos valores de top_k."""
        label_map = {1: "Unigramas", 2: "Bigramas", 3: "Trigramas"}
        n_label = label_map.get(n, f"{n}-gramas")
        colors = ["#4C72B0", "#DD8452"]

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for idx, top_k in enumerate(tops):
            df = self.frequency_table(n).head(top_k).iloc[::-1]
            axes[idx].barh(df["Ngrama"], df["Frecuencia"], color=colors[idx], edgecolor="white")
            axes[idx].set_title(f"Top {top_k} {n_label}", fontsize=11)
            axes[idx].set_xlabel("Frecuencia absoluta")

        plt.tight_layout()
        return fig


# ── 5. FUNCIONES DE CONVENIENCIA ─────────────────────────────────────────────

def run_full_eda(
    df: pd.DataFrame,
    text_columns: List[str],
    category_column: Optional[str] = None,
    extra_stopwords: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    show_plots: bool = True,
) -> Dict[str, dict]:
    """
    Ejecuta el EDA completo sobre una lista de columnas de texto.

    Retorna un diccionario con los resultados por columna:
      {column: {"summary": DataFrame, "ngram_tables": {...}, "figures": [...]}}
    """
    results: Dict[str, dict] = {}
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for col in text_columns:
        if col not in df.columns:
            print(f"⚠ Columna '{col}' no encontrada en el DataFrame. Se omite.")
            continue

        print(f"\n{'='*70}")
        print(f"  EDA — {col[:70]}")
        print(f"{'='*70}")

        col_result: dict = {"figures": []}

        # 2.1 Distribución de longitud
        tla = TextLengthAnalyzer(df[col], column_name=col[:50])
        summary = tla.summary()
        col_result["summary"] = summary
        print("\n▸ Estadísticos descriptivos de longitud:")
        print(summary.to_string(index=False))

        for fig_fn in [tla.plot_histograms, tla.plot_boxplots, tla.plot_comparative]:
            fig = fig_fn()
            col_result["figures"].append(fig)
            if show_plots:
                plt.show()
            else:
                plt.close(fig)

        # 2.2 Nube de palabras
        wc_gen = WordCloudGenerator(extra_stopwords=extra_stopwords)
        fig_wc = wc_gen.generate(df[col], title=f"WordCloud — {col[:60]}", save_path=(
            str(Path(save_dir) / f"wc_{col[:30]}.png") if save_dir else None
        ))
        col_result["figures"].append(fig_wc)
        if show_plots:
            plt.show()
        else:
            plt.close(fig_wc)

        if category_column and category_column in df.columns:
            cat_figs = wc_gen.generate_by_category(df, col, category_column, save_dir=save_dir)
            col_result["figures"].extend(cat_figs)
            if show_plots:
                for f in cat_figs:
                    plt.show()
            else:
                for f in cat_figs:
                    plt.close(f)

        # 2.3 N-gramas
        nga = NgramAnalyzer(df[col], extra_stopwords=extra_stopwords)
        ngram_tables: Dict[str, pd.DataFrame] = {}
        for n, name in [(1, "unigramas"), (2, "bigramas"), (3, "trigramas")]:
            tbl = nga.frequency_table(n)
            ngram_tables[name] = tbl
            print(f"\n▸ Top 10 {name}:")
            print(tbl.head(10).to_string(index=False))

            fig_ng = nga.plot_comparative_ngrams(n=n, tops=(20, 30))
            col_result["figures"].append(fig_ng)
            if show_plots:
                plt.show()
            else:
                plt.close(fig_ng)

        col_result["ngram_tables"] = ngram_tables
        results[col] = col_result

    return results
