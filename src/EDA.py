"""Exploratory Data Analysis module for the NLP survey corpus.

Phase 2 of the NLP pipeline. Consumes DataFrames already processed by
``limpieza.py`` (cleaning + stopword removal) and, from beta_2 onward,
by ``synonyms.py`` (canonical-term normalisation). It provides:

* Text-length distribution analysis
* Word cloud generation (global, per-column, per-category, per-survey)
* N-gram frequency analysis (uni / bi / tri)
* Reusable, exportable visualisations for reporting (PPT)

Column contract:
    Stopword removal is owned by ``limpieza.Cleaner`` and synonym
    normalisation by ``synonyms.SynonymReplacer``. This module never
    loads its own stopword lists or synonym dictionaries. It expects
    to receive text columns that are already normalised, preferably
    the ``_no_stopwords_synonyms`` column (stopword-free **and**
    synonym-collapsed). The optional ``extra_filter_words`` parameter
    is reserved for ad-hoc, per-plot visual exclusions only.

Typical usage::

    from EDA import SurveyWordClouds, NgramAnalyzer

    clouds = SurveyWordClouds(corpus_by_survey, text_suffix="_synonyms")
    clouds.generate_all(save_dir="figures/wordclouds")
"""

from __future__ import annotations

import functools
import warnings
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

if TYPE_CHECKING:

    from matplotlib.figure import Figure

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_NGRAM_LABELS: dict[int, str] = {
    1: "Unigrams",
    2: "Bigrams",
    3: "Trigrams",
}

_DEFAULT_WC_PARAMS: dict[str, Any] = {
    "width": 1000,
    "height": 500,
    "background_color": "white",
    "colormap": "viridis",
    "max_words": 200,
    "collocations": False,
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    """Split already-clean text on whitespace.

    Args:
        text: Pre-processed text string.

    Returns:
        List of token strings (empty list for blank input).
    """
    if not isinstance(text, str) or not text.strip():
        return []
    return text.split()


def _validate_series(series: pd.Series) -> pd.Series:
    """Drop NaN/blank values and cast to ``str``.

    Args:
        series: Raw pandas Series.

    Returns:
        Cleaned Series with no null or empty values.

    Raises:
        ValueError: If the resulting Series is completely empty.
    """
    clean = series.dropna().astype(str)
    clean = clean[clean.str.strip() != ""]
    if clean.empty:
        raise ValueError("The provided Series is empty after cleaning.")
    return clean


def _show_or_close(fig: Figure, show: bool) -> None:
    """Display or silently close a figure.

    Args:
        fig: Matplotlib Figure.
        show: Call ``plt.show()`` when ``True``, else close.
    """
    if show:
        plt.show()
    else:
        plt.close(fig)


def _ensure_dir(path: str | None) -> Path | None:
    """Create *path* as a directory if given.

    Args:
        path: Directory path or ``None``.

    Returns:
        Resolved ``Path`` or ``None``.
    """
    if path is None:
        return None
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _slugify(text: str, max_len: int = 40) -> str:
    """Produce a filesystem-safe slug from arbitrary text.

    Args:
        text: Source text.
        max_len: Maximum slug length.

    Returns:
        Lower-case slug with non-alphanumerics replaced by underscores.
    """
    cleaned = "".join(c if c.isalnum() else "_" for c in text.lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")[:max_len]


# ===================================================================
# 1. TEXT LENGTH ANALYSIS
# ===================================================================
class TextLengthAnalyzer:
    """Compute descriptive statistics and plots for text length.

    Args:
        series: Pandas Series containing text data.
        column_name: Human-readable label used in plot titles.
    """

    def __init__(self, series: pd.Series, column_name: str = "text") -> None:
        self._series: pd.Series = _validate_series(series)
        self._column_name: str = column_name

    @functools.cached_property
    def char_lengths(self) -> pd.Series:
        """Character count per document."""
        return self._series.str.len()

    @functools.cached_property
    def word_lengths(self) -> pd.Series:
        """Word count per document."""
        return self._series.apply(lambda t: len(t.split()))

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame of descriptive statistics.

        Includes count, mean, median, min, max, std and percentiles
        (10, 25, 75, 90) for characters and words.

        Returns:
            Summary DataFrame.
        """
        rows: list[dict[str, str | float]] = []

        def _add(name: str, char_v: float, word_v: float) -> None:
            rows.append(
                {
                    "Metric": name,
                    "Characters": round(char_v, 2),
                    "Words": round(word_v, 2),
                }
            )

        _add("Count", len(self.char_lengths), len(self.word_lengths))
        _add("Mean", self.char_lengths.mean(), self.word_lengths.mean())
        _add("Median", self.char_lengths.median(), self.word_lengths.median())
        _add("Min", self.char_lengths.min(), self.word_lengths.min())
        _add("Max", self.char_lengths.max(), self.word_lengths.max())
        _add("Std", self.char_lengths.std(), self.word_lengths.std())
        for pct in (10, 25, 75, 90):
            _add(
                f"P{pct}",
                float(np.percentile(self.char_lengths, pct)),
                float(np.percentile(self.word_lengths, pct)),
            )
        return pd.DataFrame(rows)

    def plot_histograms(
        self, bins: int = 40, figsize: tuple[int, int] = (14, 5)
    ) -> Figure:
        """Side-by-side histograms for character and word counts.

        Args:
            bins: Number of histogram bins.
            figsize: Figure dimensions ``(width, height)``.

        Returns:
            Matplotlib Figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        self._draw_histogram(
            axes[0], self.char_lengths, "# Characters", "#4C72B0", bins
        )
        self._draw_histogram(axes[1], self.word_lengths, "# Words", "#55A868", bins)
        plt.tight_layout()
        return fig

    def plot_boxplots(self, figsize: tuple[int, int] = (14, 5)) -> Figure:
        """Side-by-side box plots for character and word counts.

        Args:
            figsize: Figure dimensions.

        Returns:
            Matplotlib Figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        sns.boxplot(x=self.char_lengths, ax=axes[0], color="#4C72B0")
        axes[0].set_title(f"Boxplot chars \u2014 {self._column_name}", fontsize=11)
        sns.boxplot(x=self.word_lengths, ax=axes[1], color="#55A868")
        axes[1].set_title(f"Boxplot words \u2014 {self._column_name}", fontsize=11)
        plt.tight_layout()
        return fig

    def plot_comparative(self, figsize: tuple[int, int] = (14, 5)) -> Figure:
        """Overlaid KDE of normalised character and word lengths.

        Args:
            figsize: Figure dimensions.

        Returns:
            Matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=figsize)
        char_max = self.char_lengths.max() or 1
        word_max = self.word_lengths.max() or 1
        sns.kdeplot(
            self.char_lengths / char_max,
            label="Characters (norm.)",
            ax=ax,
            fill=True,
            alpha=0.3,
        )
        sns.kdeplot(
            self.word_lengths / word_max,
            label="Words (norm.)",
            ax=ax,
            fill=True,
            alpha=0.3,
        )
        ax.set_title(
            f"Comparative distribution \u2014 {self._column_name}", fontsize=11
        )
        ax.set_xlabel("Normalised value [0, 1]")
        ax.legend()
        plt.tight_layout()
        return fig

    def _draw_histogram(
        self, ax: plt.Axes, data: pd.Series, xlabel: str, color: str, bins: int = 40
    ) -> None:
        """Render a single histogram on the given Axes.

        Args:
            ax: Target Axes object.
            data: Numeric Series.
            xlabel: X-axis label.
            color: Bar fill colour.
            bins: Number of bins.
        """
        ax.hist(data, bins=bins, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(f"{xlabel} distribution \u2014 {self._column_name}", fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")


# ===================================================================
# 2. WORD CLOUD GENERATION
# ===================================================================
class WordCloudGenerator:
    """Generate word clouds from normalised text data.

    Note:
        Expects stopword-free, synonym-normalised text. The optional
        *extra_filter_words* set is for ad-hoc per-plot exclusions
        only, never for stopword management.

    Args:
        extra_filter_words: Optional set of words to exclude from the
            rendering of a specific cloud.
    """

    def __init__(self, extra_filter_words: set[str] | None = None) -> None:
        self._extra_filter: set[str] = extra_filter_words or set()

    def generate(
        self,
        series: pd.Series,
        title: str = "Word Cloud",
        figsize: tuple[int, int] = (12, 6),
        save_path: str | None = None,
        **wc_kwargs: Any,
    ) -> Figure:
        """Create and (optionally) save a word cloud for a Series.

        Args:
            series: Text data (ideally the ``_synonyms`` column).
            title: Plot title.
            figsize: Figure dimensions.
            save_path: If given, the figure is saved as PNG.
            **wc_kwargs: Forwarded to :class:`wordcloud.WordCloud`.

        Returns:
            Matplotlib Figure.
        """
        corpus = " ".join(_validate_series(series))
        cloud = self._build_wordcloud(corpus, **wc_kwargs)
        fig = self._render_wordcloud(cloud, title, figsize)
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig

    def generate_by_category(
        self,
        df: pd.DataFrame,
        text_col: str,
        cat_col: str,
        figsize_per_plot: tuple[int, int] = (10, 5),
        save_dir: str | None = None,
        **wc_kwargs: Any,
    ) -> list[Figure]:
        """Create one word cloud per unique category value.

        Args:
            df: Source DataFrame.
            text_col: Text column name.
            cat_col: Categorical column name.
            figsize_per_plot: Per-figure dimensions.
            save_dir: Directory for exported PNGs.
            **wc_kwargs: Forwarded to :class:`wordcloud.WordCloud`.

        Returns:
            List of Matplotlib Figures.
        """
        figures: list[Figure] = []
        directory = _ensure_dir(save_dir)
        for cat in sorted(df[cat_col].dropna().unique()):
            subset = df.loc[df[cat_col] == cat, text_col].dropna()
            if subset.empty:
                continue
            title = f"WordCloud \u2014 {text_col[:45]} | {cat_col}={cat}"
            sp = (
                None
                if directory is None
                else str(directory / f"wc_{_slugify(str(cat))}.png")
            )
            figures.append(
                self.generate(
                    subset,
                    title=title,
                    figsize=figsize_per_plot,
                    save_path=sp,
                    **wc_kwargs,
                )
            )
        return figures

    def _build_wordcloud(self, text: str, **kwargs: Any) -> WordCloud:
        """Instantiate and fit a WordCloud.

        Args:
            text: Concatenated corpus.
            **kwargs: Parameter overrides.

        Returns:
            Fitted :class:`wordcloud.WordCloud`.
        """
        params: dict[str, Any] = {**_DEFAULT_WC_PARAMS, **kwargs}
        if self._extra_filter:
            params["stopwords"] = self._extra_filter
        return WordCloud(**params).generate(text)

    @staticmethod
    def _render_wordcloud(
        cloud: WordCloud, title: str, figsize: tuple[int, int]
    ) -> Figure:
        """Draw a WordCloud on a new Figure.

        Args:
            cloud: Fitted WordCloud.
            title: Plot title.
            figsize: Figure dimensions.

        Returns:
            Matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(cloud, interpolation="bilinear")
        ax.set_title(title, fontsize=13)
        ax.axis("off")
        plt.tight_layout()
        return fig


# ===================================================================
# 3. N-GRAM ANALYSIS
# ===================================================================
class NgramAnalyzer:
    """Build and visualise n-gram frequency tables.

    Note:
        Expects stopword-free, synonym-normalised text. The optional
        *extra_filter_words* set is for ad-hoc exclusions only.

    Args:
        series: Pandas Series of normalised text.
        extra_filter_words: Optional set of words to exclude during
            tokenisation.
    """

    def __init__(
        self, series: pd.Series, extra_filter_words: set[str] | None = None
    ) -> None:
        self._series: pd.Series = _validate_series(series)
        self._extra_filter: set[str] = extra_filter_words or set()

    @functools.cached_property
    def tokens(self) -> list[list[str]]:
        """Tokenised, optionally extra-filtered document list."""
        if self._extra_filter:
            return [
                [w for w in _tokenize(text) if w not in self._extra_filter]
                for text in self._series
            ]
        return [_tokenize(text) for text in self._series]

    def frequency_table(self, n: int = 1) -> pd.DataFrame:
        """Compute absolute and relative n-gram frequencies.

        Args:
            n: N-gram size (1, 2 or 3).

        Returns:
            DataFrame with ``Ngram``, ``Frequency``, ``Relative_Frequency``.
        """
        ngrams = self._extract_ngrams(n)
        counts = Counter(ngrams)
        total = sum(counts.values()) or 1
        rows = [
            {
                "Ngram": " ".join(gram),
                "Frequency": freq,
                "Relative_Frequency": round(freq / total, 6),
            }
            for gram, freq in counts.most_common()
        ]
        return pd.DataFrame(rows)

    def plot_top_ngrams(
        self,
        n: int = 1,
        top_k: int = 20,
        figsize: tuple[int, int] = (12, 7),
        title: str | None = None,
        color: str = "#4C72B0",
        save_path: str | None = None,
    ) -> Figure:
        """Horizontal bar chart of the most frequent n-grams.

        Args:
            n: N-gram size.
            top_k: Number of top entries.
            figsize: Figure dimensions.
            title: Custom title (auto-generated when ``None``).
            color: Bar colour.
            save_path: If given, the figure is saved as PNG.

        Returns:
            Matplotlib Figure.
        """
        df_top = self.frequency_table(n).head(top_k).iloc[::-1]
        label = _NGRAM_LABELS.get(n, f"{n}-grams")
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(df_top["Ngram"], df_top["Frequency"], color=color, edgecolor="white")
        ax.set_xlabel("Absolute frequency")
        ax.set_title(title or f"Top {top_k} {label}")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig

    def plot_comparative_ngrams(
        self,
        n: int = 1,
        tops: tuple[int, int] = (20, 30),
        figsize: tuple[int, int] = (16, 7),
    ) -> Figure:
        """Two bar charts side-by-side for two *top_k* values.

        Args:
            n: N-gram size.
            tops: Pair of top_k values for left/right panels.
            figsize: Figure dimensions.

        Returns:
            Matplotlib Figure.
        """
        label = _NGRAM_LABELS.get(n, f"{n}-grams")
        colors = ("#4C72B0", "#DD8452")
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for idx, top_k in enumerate(tops):
            df_top = self.frequency_table(n).head(top_k).iloc[::-1]
            axes[idx].barh(
                df_top["Ngram"],
                df_top["Frequency"],
                color=colors[idx],
                edgecolor="white",
            )
            axes[idx].set_title(f"Top {top_k} {label}")
            axes[idx].set_xlabel("Absolute frequency")
        plt.tight_layout()
        return fig

    def _extract_ngrams(self, n: int) -> list[tuple[str, ...]]:
        """Build raw n-gram tuples from cached tokens.

        Args:
            n: N-gram size.

        Returns:
            Flat list of n-gram tuples.
        """
        ngrams: list[tuple[str, ...]] = []
        for tok_list in self.tokens:
            ngrams.extend(zip(*(tok_list[i:] for i in range(n))))
        return ngrams


# ===================================================================
# 4. PER-SURVEY WORD CLOUDS (beta_2)
# ===================================================================
class SurveyWordClouds:
    """Generate and export one word cloud per survey for reporting.

    Designed to feed the final PowerPoint: each survey (Excel sheet)
    yields a labelled PNG built from its synonym-normalised text.

    Args:
        corpus_by_survey: Mapping of ``survey_name -> DataFrame``.
        column_resolver: Callable mapping a survey name to the text
            column to use, or a fixed suffix string (e.g. ``"_synonyms"``)
            resolved against the first matching column.
        extra_filter_words: Optional per-plot exclusions.
        colormap: Matplotlib colormap for all clouds.
    """

    def __init__(
        self,
        corpus_by_survey: dict[str, pd.DataFrame],
        column_resolver: str,
        extra_filter_words: set[str] | None = None,
        colormap: str = "viridis",
    ) -> None:
        self._corpus = corpus_by_survey
        self._suffix = column_resolver
        self._generator = WordCloudGenerator(extra_filter_words)
        self._colormap = colormap

    def _resolve_column(self, df: pd.DataFrame) -> str | None:
        """Find the first column ending with the configured suffix.

        Args:
            df: Survey DataFrame.

        Returns:
            Matching column name, or ``None`` if absent.
        """
        matches = [c for c in df.columns if c.endswith(self._suffix)]
        return matches[0] if matches else None

    def generate_all(
        self,
        save_dir: str | None = None,
        figsize: tuple[int, int] = (12, 6),
        *,
        show_plots: bool = True,
    ) -> dict[str, Figure]:
        """Generate a labelled word cloud per survey.

        Args:
            save_dir: Directory for exported PNGs (created if needed).
            figsize: Per-figure dimensions.
            show_plots: Display figures interactively when ``True``.

        Returns:
            Mapping ``survey_name -> Figure``.
        """
        directory = _ensure_dir(save_dir)
        figures: dict[str, Figure] = {}

        for survey_name, df in self._corpus.items():
            column = self._resolve_column(df)
            if column is None:
                print(
                    f"\u26a0 No '{self._suffix}' column in '{survey_name}'. Skipping."
                )
                continue

            title = f"Nube de Palabras \u2014 {survey_name}"
            save_path = (
                None
                if directory is None
                else str(directory / f"wordcloud_{_slugify(survey_name)}.png")
            )
            fig = self._generator.generate(
                df[column],
                title=title,
                figsize=figsize,
                save_path=save_path,
                colormap=self._colormap,
            )
            figures[survey_name] = fig
            _show_or_close(fig, show_plots)
            if save_path:
                print(f"\u2713 Saved: {save_path}")

        return figures


# ===================================================================
# 5. ORCHESTRATOR
# ===================================================================
def run_full_eda(
    df: pd.DataFrame,
    text_columns: list[str],
    category_column: str | None = None,
    extra_filter_words: set[str] | None = None,
    save_dir: str | None = None,
    *,
    show_plots: bool = True,
) -> dict[str, dict[str, Any]]:
    """Execute the complete EDA pipeline on selected text columns.

    Note:
        *text_columns* should reference the ``_synonyms`` (or
        ``_no_stopwords``) columns produced upstream.

    Args:
        df: Source DataFrame.
        text_columns: Column names to analyse.
        category_column: Optional categorical column for grouped clouds.
        extra_filter_words: Ad-hoc words to exclude from visualisations.
        save_dir: Directory for exported images.
        show_plots: Display figures interactively when ``True``.

    Returns:
        Nested dict keyed by column with ``summary``, ``ngram_tables``
        and ``figures``.
    """
    results: dict[str, dict[str, Any]] = {}
    _ensure_dir(save_dir)

    for col in text_columns:
        if col not in df.columns:
            print(f"\u26a0 Column '{col}' not found. Skipping.")
            continue

        print(f"\n{'=' * 70}\n  EDA \u2014 {col[:70]}\n{'=' * 70}")

        col_result: dict[str, Any] = {"figures": []}
        col_result.update(_run_length_analysis(df[col], col, show_plots))
        col_result["figures"].extend(
            _run_wordclouds(
                df, col, extra_filter_words, category_column, save_dir, show_plots
            )
        )
        ngram_tables, ngram_figs = _run_ngram_analysis(
            df[col], extra_filter_words, show_plots
        )
        col_result["ngram_tables"] = ngram_tables
        col_result["figures"].extend(ngram_figs)
        results[col] = col_result

    return results


def _run_length_analysis(
    series: pd.Series, col_name: str, show_plots: bool
) -> dict[str, Any]:
    """Run text-length statistics and plots.

    Args:
        series: Text column.
        col_name: Label for plot titles.
        show_plots: Display figures interactively.

    Returns:
        Dict with ``summary`` and ``figures`` keys.
    """
    tla = TextLengthAnalyzer(series, column_name=col_name[:50])
    summary = tla.summary()
    print("\n\u25b8 Descriptive length statistics:")
    print(summary.to_string(index=False))

    figures: list[Figure] = []
    for plot_fn in (tla.plot_histograms, tla.plot_boxplots, tla.plot_comparative):
        fig = plot_fn()
        figures.append(fig)
        _show_or_close(fig, show_plots)
    return {"summary": summary, "figures": figures}


def _run_wordclouds(
    df: pd.DataFrame,
    col: str,
    extra_filter_words: set[str] | None,
    category_column: str | None,
    save_dir: str | None,
    show_plots: bool,
) -> list[Figure]:
    """Generate global and per-category word clouds.

    Args:
        df: Source DataFrame.
        col: Text column name.
        extra_filter_words: Ad-hoc exclusions for visualisation.
        category_column: Optional grouping column.
        save_dir: Export directory.
        show_plots: Display figures interactively.

    Returns:
        List of Figures.
    """
    wc_gen = WordCloudGenerator(extra_filter_words=extra_filter_words)
    directory = _ensure_dir(save_dir)
    sp = None if directory is None else str(directory / f"wc_{_slugify(col)}.png")
    fig = wc_gen.generate(df[col], title=f"WordCloud \u2014 {col[:60]}", save_path=sp)
    figures: list[Figure] = [fig]
    _show_or_close(fig, show_plots)

    if category_column and category_column in df.columns:
        cat_figs = wc_gen.generate_by_category(
            df, col, category_column, save_dir=save_dir
        )
        for cat_fig in cat_figs:
            _show_or_close(cat_fig, show_plots)
        figures.extend(cat_figs)
    return figures


def _run_ngram_analysis(
    series: pd.Series, extra_filter_words: set[str] | None, show_plots: bool
) -> tuple[dict[str, pd.DataFrame], list[Figure]]:
    """Compute and visualise n-gram frequencies.

    Args:
        series: Text column (normalised).
        extra_filter_words: Ad-hoc exclusions.
        show_plots: Display figures interactively.

    Returns:
        Tuple of (tables dict, Figures list).
    """
    nga = NgramAnalyzer(series, extra_filter_words=extra_filter_words)
    tables: dict[str, pd.DataFrame] = {}
    figures: list[Figure] = []
    for n, name in ((1, "unigrams"), (2, "bigrams"), (3, "trigrams")):
        tbl = nga.frequency_table(n)
        tables[name] = tbl
        print(f"\n\u25b8 Top 10 {name}:")
        print(tbl.head(10).to_string(index=False))
        fig = nga.plot_comparative_ngrams(n=n, tops=(20, 30))
        figures.append(fig)
        _show_or_close(fig, show_plots)
    return tables, figures
