"""Module for cleaning the key text column of survey data.

Phase 1 of the NLP pipeline. The :class:`Cleaner` class loads a survey
sheet, normalises the target text column, and removes stopwords.

Stopword policy (single source of truth):
    Stopwords now come from ``settings.py``:

    1. The NLTK Spanish corpus (base list), combined at import time with
       ``settings.stopwords_global`` (module-level ``stop_words``).
    2. ``settings.stopwords_dict``, filtered by this Cleaner's
       *survey_name* (the "particulares" terms).
    3. ``settings.adverbios_dict``, used separately to produce an
       adverb-free variant of the cleaned text.

    The accent-normalisation map (``settings.accents``) stays centralised
    there as well.

------------------------------------------------------------------------
RESUMEN DE CORRECCIONES APLICADAS EN ESTE ARCHIVO (limpieza_corregido.py)
Cada una está señalada en el código con un comentario "CORRECCIÓN N".
------------------------------------------------------------------------


  4. Se restaura la validación de columna de texto en `load_data()`.
     La versión anterior hacía `.astype(str)` sin condición, lo cual
     resolvía el problema de celdas numéricas sueltas (ej. un 0 en vez
     de texto) pero eliminaba por completo la protección contra
     seleccionar por error una columna que NO es de texto abierto.
     Ahora se avisa cuántas celdas no eran texto y solo se convierte
     cuando la columna es claramente de texto en su mayoría.

  


  
  
------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path

import nltk
import pandas as pd
from pandas.api.types import is_string_dtype

# CORRECCIÓN 7: se quita `import openpyxl` (no se usaba explícitamente
# en este archivo). Sigue siendo una dependencia requerida por pandas
# para leer .xlsx -- debe permanecer en pyproject.toml/poetry.lock,
# solo no hace falta importarla aquí.
from settings import (
    accents,
    stopwords_global,
    stopwords_dict,
    adverbios_dict,
)

# NLTK elements import (se descarga una sola vez, al importar el módulo)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(nltk.corpus.stopwords.words("spanish"))
stop_words.update(stopwords_global.get('stopwords_global', []))


class Cleaner:
    """Load, clean and remove stopwords from a survey's key text column.

    Args:
        file_path: Path to the corpus file (.csv/.xlsx/.xls/.xlsm).
        survey_name: Survey identifier; selects this survey's rows from
            ``settings.stopwords_dict`` (the "particulares" stopwords).
        key_column: Name of the open-text column to process.
        separator: Field separator for CSV inputs.
        sheet_name: Sheet to read from an Excel corpus.
    """
    
    SUPPORTED_FORMATS = {'.csv', '.xlsx', '.xls', '.xlsm'}

    def __init__(self, file_path: str | Path,
                 survey_name: str,
                 key_column: str,
                 separator: str = ',',
                 sheet_name: str | int = 0) -> None:
        """Initializes the Cleaner class with the specified parameters."""
        
        self.file_path = Path(file_path)

        
        self.survey_name = survey_name

        self.separator = separator
        self.key_column = key_column
        self.sheet_name = sheet_name

        
        self.stop_words = {str(w).strip().lower() for w in stop_words}
        self.new_stopwords = {
            str(w).strip().lower()
            for w in stopwords_dict.get(survey_name, [])
        }
        self.stop_words.update(self.new_stopwords)
        self.stopwords_adv = {
            str(w).strip().lower()
            for w in adverbios_dict.get('adverbios', [])
        }

        self.cleaned_data = None

    def load_data(self) -> pd.DataFrame:
        """
        Loads and validate the data file for text processing.

        This method attempts to read a CSV (.csv) or Excel (.xlsx, .xls) file
        in the specified path, first verifying the existence of the file,
        the supported format, the presence of the key column, and the validity
        of the data type of that column.

        Returns:
            pd.DataFrame: A pandas DataFrame with the loaded data if the
                operation was successful.

        Raises:
            FileNotFoundError: If the file does not exist in the specified
            path.
            ValueError: If the file format is not supported, if the key column
            does not exist, if it is not of text type, or if any unexpected
            error occurs during reading.
        """

        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset not found in the file path: "
                                    f"{self.file_path}")
        # Validate extension with the suffix attribute
        extension = self.file_path.suffix.lower()
        if extension not in self.SUPPORTED_FORMATS:
            raise ValueError(
            f"Format '{extension}' is not supported. Supported formats: "
            f"{self.SUPPORTED_FORMATS}"
        )

        try:
            if extension == '.csv':
                data = pd.read_csv(self.file_path, sep=self.separator)

            elif extension in {'.xlsx', '.xls', '.xlsm'}:
                data = pd.read_excel(self.file_path,
                                        sheet_name=self.sheet_name)

        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ValueError(
                f"Error reading the file format: {exc}"
            ) from exc

        
        # Validations to ensure the key column exists and is of text type
        if self.key_column not in data.columns:
            raise ValueError(f"Column '{self.key_column}' "
                                f"not found in the file.")

        # Avoids pandas representing NaN values as 'float' in the key column
        data[self.key_column] = data[self.key_column].fillna('')

        is_text = data[self.key_column].apply(lambda v: isinstance(v, str))
        text_pct = is_text.mean() if len(data) else 0.0

        if text_pct < 0.5:
            raise ValueError(f"Column '{self.key_column}' "
                                f"is not of text type.")

        non_text_count = int((~is_text).sum())
        if non_text_count:
            print(
                f"Notice: {non_text_count} cell(s) in '{self.key_column}' were "
                f"not text (e.g. numeric) and were converted to text. "
                f"Check whether they represent a 'no response' code."
            )
        data[self.key_column] = data[self.key_column].astype(str)

        if not is_string_dtype(data[self.key_column]):
            raise ValueError(f"Column '{self.key_column}' "
                                f"is not of text type.")

        return data

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Cleans a text string by removing noise and normalizing characters.
        The function performs the following operations in order:
        1. Removes leading and trailing whitespace.
        2. Converts the text to lowercase.
        3. Normalizes accented characters (tildes) to their plaintext
        counterparts.
        4. Removes special characters, keeping only alphanumeric characters,
        spaces, and the letter 'ñ'.

        Args:
            text (str): The raw text string to process.

        Returns:
            str: The processed and normalized text.
        """
        # Returns an empty string if the value is not text
        # to avoid pandas representing it as 'float'
        if not isinstance(text, str):
            return ''

        # Remove leading and trailing whitespace
        cleaned_text = text.strip()

        # Convert to lowercase
        cleaned_text = cleaned_text.lower()

        # Remove accented letters
        for accent, letter in accents.items():
            cleaned_text = cleaned_text.replace(accent, letter)

        # Eliminate other special characters using a regular expression
        cleaned_text = re.sub(r'[^a-zA-Z0-9\sñÑ]', '', cleaned_text)

        return cleaned_text

    def clean_key_column(self) -> pd.DataFrame:
        """
        Performs the cleaning of the text in the key column of the loaded
        DataFrame.

        This method delegates the data loading to 'load_data', applies a
        cleaning transformation (defined in 'clean_text') to the specified
        column in 'self.key_column' and generates a new resulting column
        in the DataFrame.

        Returns:
            Pandas DataFrame: The DataFrame 'self.cleaned_data'
            with the new column '{self.key_column}_clean'
            containing the cleaned text.

        Raises:
            ValueError: If the method 'load_data' returns None, indicating
            that it was not possible to obtain the original data source.
        """
        # Make a copy to avoid modifying the original DataFrame
        data = self.load_data().copy()
        if data is not None:
            # Apply the cleaning function to each value in the key column
            # without overwriting the original column.
            data[f"{self.key_column}_clean"] = (
                data[self.key_column].apply(self._clean_text)
                )

            self.cleaned_data = data

            return self.cleaned_data

        else:
            raise ValueError("The data could not be loaded for cleaning.")

    def _clean_stopwords(self, text: str) -> str:
        """Remove stopwords (NLTK base + settings.py terms) from a string.

        Args:
            text: Already-cleaned text (output of `_clean_text`).

        Returns:
            The text without any term present in `self.stop_words`.
        """
        
        words = nltk.word_tokenize(text)
        filtered_words = [w for w in words if w.lower() not in self.stop_words]
        return ' '.join(filtered_words)

    def _clean_adverbs(self, text: str) -> str:
        words = nltk.word_tokenize(text)
        filtered_words = [
            w for w in words
            if w.lower() not in self.stopwords_adv]
        return ' '.join(filtered_words)

    def eliminate_stopwords(self) -> pd.DataFrame:
        """
        Eliminates stopwords from a specific column in the DataFrame.

        The function uses the NLTK corpus for Spanish, extended with terms
        specific to the academic domain. It filters the words from the column
        defined in 'self.key_column' and generates two new columns with
        the clean text: one keeping adverbs (for sentiment analysis) and
        one without them (for wordclouds and topic modeling).

        Raises:
            ImportError: If the 'nltk' library is not installed in the
            environment.

        Returns:
            pd.DataFrame: The DataFrame 'self.cleaned_data' with the new
            columns '{self.key_column}_no_stopwords' and
            '{self.key_column}_no_stopwords_no_adverbs'.
        """
        if not hasattr(self, 'cleaned_data') or self.cleaned_data is None:
             self.cleaned_data = self.clean_key_column()

        goal_column = f"{self.key_column}_clean"

        # Apply the function to the entire column at once (vectorized)
        # preserving the adverbes in the text.
        # use: sentiment analysis

        self.cleaned_data[f"{self.key_column}_no_stopwords"] = (
            self.cleaned_data[goal_column].apply(
                lambda x:self._clean_stopwords(x))
            )

        # Apply the function to the entire column at once (vectorized)
        # eliminating the adverbes in the text.
        # use: wordclouds and topic modeling

        self.cleaned_data[f"{self.key_column}_no_stopwords_no_adverbs"] = (
            self.cleaned_data[goal_column]
                .apply(lambda x:self._clean_stopwords(x))
                .apply(lambda x: self._clean_adverbs(x))
            )

        return self.cleaned_data

    @staticmethod
    def _is_purely_numeric(text: str) -> bool:
        """Return True if `text`, once stripped, contains only digits
        and whitespace (e.g. "0", "5", "5 5"). A response like this
        carries no topical content -- it should not be treated as a
        real opinion for modeling, even though it is not an empty
        string. Numbers that appear WITHIN a longer response (e.g.
        "reprobe 0 materias este semestre") are left untouched, since
        only the response as a whole is evaluated here, not individual
        tokens."""
        stripped = text.strip()
        return bool(stripped) and bool(re.fullmatch(r"[\d\s]+", stripped))

    def remove_empty_records(self, column: str | None = None) -> pd.DataFrame:
        """Drop rows whose generated text column is empty, or contains
        only digits, after cleaning.

        Defaults to checking `{key_column}_no_stopwords_no_adverbs` --
        the column meant for topic modeling (see `eliminate_stopwords`).
        This does NOT catch explicit non-answers like "N/A"/"NR" when
        they are mixed with a stopword removed some other way -- those
        cases are already handled upstream if "na"/"nr" are registered
        as stopwords (see settings.py). This method only removes
        records with no content (empty) or with purely numeric content
        (e.g. a lone "0"), regardless of how they got that way.

        Returns:
            The DataFrame stored in `self.cleaned_data` with empty or
            purely-numeric records removed.
        """
        if self.cleaned_data is None:
            self.cleaned_data = self.eliminate_stopwords()

        target = column or f"{self.key_column}_no_stopwords_no_adverbs"
        if target not in self.cleaned_data.columns:
            target = f"{self.key_column}_clean"
        if target not in self.cleaned_data.columns:
            raise ValueError(
                "No generated text column found. Run 'clean_key_column' "
                "and/or 'eliminate_stopwords' first."
            )

        before = len(self.cleaned_data)
        text_values = self.cleaned_data[target].astype(str)
        is_empty = text_values.str.strip() == ""
        is_purely_numeric = text_values.apply(self._is_purely_numeric)
        self.cleaned_data = self.cleaned_data.loc[~(is_empty | is_purely_numeric)].reset_index(drop=True)

        removed = before - len(self.cleaned_data)
        pct = (100 * removed / before) if before else 0.0
        print(
            f"Removed {removed} empty or purely-numeric record(s) "
            f"({pct:.1f}%) based on column '{target}'."
        )
        return self.cleaned_data

    def remove_duplicate_records(self, subset: list[str] | None = None) -> pd.DataFrame:
        """Drop exact duplicate rows (e.g. a response exported twice).

        Defaults to comparing FULL ROWS (`subset=None`). Two different
        respondents giving the same short open-text answer are NOT
        duplicates -- pass `subset` (e.g. the survey's key columns)
        only when you want a more targeted definition of "duplicate",
        understanding that it changes response counts.

        Returns:
            The DataFrame stored in `self.cleaned_data` with duplicate
            records removed.
        """
        if self.cleaned_data is None:
            self.cleaned_data = self.eliminate_stopwords()

        before = len(self.cleaned_data)
        self.cleaned_data = self.cleaned_data.drop_duplicates(subset=subset).reset_index(drop=True)

        removed = before - len(self.cleaned_data)
        pct = (100 * removed / before) if before else 0.0
        criteria = subset if subset else "all columns (full row)"
        print(f"Removed {removed} duplicate record(s) ({pct:.1f}%) using subset={criteria}.")
        return self.cleaned_data

    def export_independent_corpus(
        self,
        output_dir: str | Path,
        id_columns: list[str],
        question_label: str | None = None,
        output_format: str = "csv",
    ) -> Path:
        """Export a reduced, independent corpus file for this key_column.

        Beyond the columns generated in place on `self.cleaned_data`
        (unchanged), this writes a SEPARATE file limited to `id_columns`
        plus this Cleaner's `key_column` and its generated `_clean` /
        `_no_stopwords` / `_no_stopwords_no_adverbs` columns. This is how
        a corpus that mixes several surveys in one sheet gets split into
        one independent, minimal file per open question.

        `id_columns` is REQUIRED (not hardcoded here) because different
        surveys have different identifier columns -- e.g. "Código Cruce
        Autoevaluación" for EvaDoc, "llave" for Autoevaluación de
        Acreditación. See `preguntas.list_key_columns`.

        Empty rows and duplicate records are evaluated ONLY on the
        selected columns (`id_columns` + `_no_stopwords_no_adverbs`),
        never on the full original row. A response that, after cleaning,
        contains only digits (e.g. a lone "0") is treated the same as
        an empty one -- it carries no topical content -- without
        affecting numbers that appear inside a longer response.

        Returns:
            Path to the exported file.
        """
        if self.cleaned_data is None:
            self.cleaned_data = self.eliminate_stopwords()

        clean_col = f"{self.key_column}_clean"
        no_stop_col = f"{self.key_column}_no_stopwords"
        no_stop_no_adv_col = f"{self.key_column}_no_stopwords_no_adverbs"

        required_columns = [*id_columns, self.key_column, clean_col, no_stop_col, no_stop_no_adv_col]
        missing = [c for c in required_columns if c not in self.cleaned_data.columns]
        if missing:
            raise ValueError(
                f"Missing columns required to export the independent corpus: {missing}"
            )

        subset = self.cleaned_data[required_columns].copy()

        before = len(subset)
        text_values = subset[no_stop_no_adv_col].astype(str)
        is_empty = text_values.str.strip() == ""
        is_purely_numeric = text_values.apply(self._is_purely_numeric)
        subset = subset.loc[~(is_empty | is_purely_numeric)]
        empty_removed = before - len(subset)

        before_dup = len(subset)
        subset = subset.drop_duplicates(subset=[*id_columns, no_stop_no_adv_col])
        duplicates_removed = before_dup - len(subset)

        subset = subset.reset_index(drop=True)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        label = question_label or re.sub(r"\s+", "_", self.key_column[:40].strip())
        file_name = re.sub(r"[^\w\-]+", "_", f"{self.survey_name}__{label}").strip("_")
        ext = ".csv" if output_format == "csv" else ".xlsx"
        output_path = output_dir / f"{file_name}{ext}"

        if output_format == "csv":
            subset.to_csv(output_path, index=False)
        else:
            subset.to_excel(output_path, index=False)

        print(
            f"Independent corpus exported: {output_path}\n"
            f"  original rows: {before} | empty removed: {empty_removed} "
            f"| duplicates removed: {duplicates_removed} | final rows: {len(subset)}"
        )
        return output_path

    def save_cleaned_data(self, out_path: str) -> None:
        """
        Runs the cleaning process and export the result to a CSV file.

        This method internally invokes the data cleaning workflow, and if the
        process is successful, it saves the resulting DataFrame to the
        specified path. It also updates the internal state of the object with
        the processed data.

        Args:
            out_path (str): The file system path (including the file name and
                .csv extension) where the cleaned data will be saved.

        Raises:
            ValueError: If the cleaning process fails or if an error occurs
            during writing the file to disk (e.g., permissions, invalid path).

        Returns:
            None: The function does not return a value, but prints a
            confirmation message in the console upon successful completion.
        """
        if self.cleaned_data is not None:
            try:
                self.cleaned_data.to_csv(out_path, index=False)
                print(f"Clean data saved in '{out_path}'.")
            
            except OSError as exc:
                raise ValueError(
                    f"An error occurred while saving the file: {exc}"
                ) from exc
        else:
            raise ValueError("The data could not be cleared for saving.")
