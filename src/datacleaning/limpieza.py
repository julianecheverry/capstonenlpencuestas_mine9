"""Module for cleaning the key text column of survey data.

Phase 1 of the NLP pipeline. The :class:`Cleaner` class loads a survey
sheet, normalises the target text column, and removes stopwords.

Stopword policy (single source of truth):
    Stopwords come from two places, combined at construction time:

    1. The NLTK Spanish corpus (base list).
    2. An external ``stopwords.xlsx`` workbook with two sheets:
         * ``globales``     — columns ``encuesta`` | ``palabra``; every
           row applies to all surveys.
         * ``particulares`` — columns ``encuesta`` | ``palabra``; only
           rows whose ``encuesta`` matches this cleaner's *survey_name*
           are applied.

    The accent-normalisation map is imported from ``settings`` so that
    configuration stays centralised.
"""

from __future__ import annotations

import re
from pathlib import Path

import nltk
import pandas as pd
from pandas.api.types import is_string_dtype

from settings import accents


class Cleaner:
    """Load, clean and remove stopwords from a survey's key text column.

    Args:
        file_path: Path to the corpus file (.csv/.xlsx/.xls/.xlsm).
        survey_name: Survey identifier; also used to select
            survey-specific rows from the ``particulares`` sheet of the
            stopwords workbook.
        key_column: Name of the open-text column to process.
        separator: Field separator for CSV inputs.
        sheet_name: Sheet to read from an Excel corpus.
        stopwords_path: Optional path to ``stopwords.xlsx``. When given,
            its terms extend the NLTK base list.
    """

    SUPPORTED_FORMATS = {".csv", ".xlsx", ".xls", ".xlsm"}

    #: Column and sheet names expected inside the stopwords workbook.
    _STOPWORD_SURVEY_COL = "encuesta"
    _STOPWORD_TERM_COL = "palabra"
    _STOPWORD_GLOBAL_SHEET = "globales"
    _STOPWORD_PARTICULAR_SHEET = "particulares"

    def __init__(
        self,
        file_path: str | Path,
        survey_name: str,
        key_column: str,
        separator: str = ",",
        sheet_name: str | int = 0,
        stopwords_path: str | Path | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.survey_name = survey_name
        self.separator = separator
        self.key_column = key_column
        self.sheet_name = sheet_name
        self.stopwords_path = (
            Path(stopwords_path) if stopwords_path is not None else None
        )
        self.cleaned_data: pd.DataFrame | None = None

        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("stopwords", quiet=True)

        # Build the effective stopword set once, at construction time.
        self.stop_words: set[str] = self._build_stopwords()

    # -- Stopword construction ---------------------------------------------

    def _build_stopwords(self) -> set[str]:
        """Assemble the effective stopword set for this survey.

        Combines the NLTK Spanish base list with the global and
        survey-specific terms from the stopwords workbook (if provided).

        Returns:
            The complete set of lower-cased stopwords.
        """
        stop_words: set[str] = set(nltk.corpus.stopwords.words("spanish"))

        if self.stopwords_path is not None:
            stop_words |= self._load_external_stopwords()

        return {w.lower() for w in stop_words}

    def _load_external_stopwords(self) -> set[str]:
        """Read global and survey-specific stopwords from the workbook.

        Returns:
            Set of stopwords drawn from the ``globales`` sheet (all rows)
            and the ``particulares`` sheet (rows matching *survey_name*).

        Raises:
            FileNotFoundError: If the stopwords workbook does not exist.
            ValueError: If a required column is missing.
        """
        path = self.stopwords_path
        if path is None or not path.exists():
            raise FileNotFoundError(f"Stopwords workbook not found: {path}")

        terms: set[str] = set()
        terms |= self._read_stopword_sheet(
            path, self._STOPWORD_GLOBAL_SHEET, survey_filter=None
        )
        terms |= self._read_stopword_sheet(
            path,
            self._STOPWORD_PARTICULAR_SHEET,
            survey_filter=self.survey_name,
        )
        return terms

    def _read_stopword_sheet(
        self,
        path: Path,
        sheet: str,
        survey_filter: str | None,
    ) -> set[str]:
        """Read one stopword sheet, optionally filtered by survey.

        Args:
            path: Path to the stopwords workbook.
            sheet: Sheet name to read.
            survey_filter: If given, keep only rows whose ``encuesta``
                column equals this value; otherwise keep every row.

        Returns:
            Set of stopword terms from the sheet (empty if the sheet is
            absent, which is tolerated for optional sheets).

        Raises:
            ValueError: If the expected term column is missing.
        """
        try:
            frame = pd.read_excel(path, sheet_name=sheet)
        except ValueError:
            # Sheet not present: tolerated (e.g. no 'particulares' tab).
            return set()

        if self._STOPWORD_TERM_COL not in frame.columns:
            raise ValueError(
                f"Sheet '{sheet}' must contain a "
                f"'{self._STOPWORD_TERM_COL}' column."
            )

        if survey_filter is not None:
            if self._STOPWORD_SURVEY_COL not in frame.columns:
                raise ValueError(
                    f"Sheet '{sheet}' must contain an "
                    f"'{self._STOPWORD_SURVEY_COL}' column to filter by "
                    f"survey."
                )
            frame = frame[frame[self._STOPWORD_SURVEY_COL] == survey_filter]

        terms = frame[self._STOPWORD_TERM_COL].dropna().astype(str)
        return {t.strip().lower() for t in terms if t.strip()}

    # -- Data loading -------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        """Load and validate the corpus file for text processing.

        Reads a CSV or Excel file, verifying existence, supported
        format, presence of the key column, and its text dtype.

        Returns:
            The loaded DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the format is unsupported, the key column is
                missing, it is not textual, or the file cannot be read.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found in the file path: {self.file_path}"
            )

        extension = self.file_path.suffix.lower()
        if extension not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Format '{extension}' is not supported. Supported "
                f"formats: {self.SUPPORTED_FORMATS}"
            )

        try:
            if extension == ".csv":
                data = pd.read_csv(self.file_path, sep=self.separator)
            else:
                data = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ValueError(f"Error reading the file format: {exc}") from exc

        if self.key_column not in data.columns:
            raise ValueError(f"Column '{self.key_column}' not found in the file.")

        # Replace NaN with empty strings so the column stays textual.
        data[self.key_column] = data[self.key_column].fillna("")

        if not is_string_dtype(data[self.key_column]):
            raise ValueError(f"Column '{self.key_column}' is not of text type.")

        return data

    # -- Text cleaning ------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalise a text string.

        Steps: strip, lowercase, remove accents (via the ``settings``
        map), and drop characters other than alphanumerics, spaces and
        the letter 'ñ'.

        Args:
            text: The raw text string.

        Returns:
            The normalised text (empty string for non-text input).
        """
        if not isinstance(text, str):
            return ""

        cleaned_text = text.strip().lower()

        for accent, letter in accents.items():
            cleaned_text = cleaned_text.replace(accent, letter)

        cleaned_text = re.sub(r"[^a-zA-Z0-9\sñÑ]", "", cleaned_text)
        return cleaned_text

    def clean_key_column(self) -> pd.DataFrame:
        """Apply text cleaning to the key column.

        Loads the data and generates a ``{key_column}_clean`` column.

        Returns:
            The DataFrame stored in ``self.cleaned_data`` with the new
            cleaned column.
        """
        data = self.load_data().copy()
        data[f"{self.key_column}_clean"] = data[self.key_column].apply(self._clean_text)
        self.cleaned_data = data
        return self.cleaned_data

    # -- Stopword removal ---------------------------------------------------

    def _clean_stopwords(self, text: str) -> str:
        """Remove stopwords from a single text string.

        Args:
            text: Text to filter (already cleaned).

        Returns:
            The text without stopwords.
        """
        words = nltk.word_tokenize(text)
        filtered_words = [w for w in words if w.lower() not in self.stop_words]
        return " ".join(filtered_words)

    def eliminate_stopwords(self) -> pd.DataFrame:
        """Generate a stopword-free version of the cleaned column.

        Ensures the ``_clean`` column exists, then produces a
        ``{key_column}_no_stopwords`` column by removing every term in
        ``self.stop_words`` (NLTK base + workbook terms).

        Returns:
            The DataFrame stored in ``self.cleaned_data`` with the new
            ``_no_stopwords`` column.
        """
        if self.cleaned_data is None:
            self.cleaned_data = self.clean_key_column()

        goal_column = f"{self.key_column}_clean"
        self.cleaned_data[f"{self.key_column}_no_stopwords"] = self.cleaned_data[
            goal_column
        ].apply(self._clean_stopwords)
        return self.cleaned_data

    # -- Persistence --------------------------------------------------------

    def save_cleaned_data(self, out_path: str) -> None:
        """Export the processed DataFrame to a CSV file.

        Args:
            out_path: Destination path (including filename and .csv).

        Raises:
            ValueError: If no processed data exists, or writing fails.
        """
        if self.cleaned_data is None:
            raise ValueError(
                "No processed data to save. Run 'clean_key_column' "
                "and/or 'eliminate_stopwords' first."
            )
        try:
            self.cleaned_data.to_csv(out_path, index=False)
        except OSError as exc:
            raise ValueError(f"An error occurred while saving the file: {exc}") from exc
        print(f"Clean data saved in '{out_path}'.")
