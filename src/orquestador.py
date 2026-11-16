"""
orquestador.py

Processes ALL active open questions of a survey in a single call
(`process_survey`), without the code knowing in advance how many
questions there are or what their columns are called -- that
information lives in data/utilities/preguntas_abiertas.xlsx (see
`preguntas.py`).

For each active question:
  1. Resolves the file, sheet and column (tolerant to renames/rewording).
  2. Cleans it (Cleaner: clean_key_column + eliminate_stopwords).
  3. Exports an independent corpus (one row per valid response, with
     that survey's key columns) to data/processed/corpus_independientes/.

At the end, it also exports ONE file with the survey's full corpus
(all original columns + the generated ones for EVERY question
processed) to data/processed/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "datacleaning"))
from limpieza import Cleaner  # noqa: E402
from preguntas import (  # noqa: E402
    list_open_questions,
    list_key_columns,
    resolve_file,
    resolve_column_and_sheet,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
DATA_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEPENDENT_DIR = PROCESSED_DIR / "corpus_independientes"
QUESTIONS_CONFIG = DATA_DIR / "utilities" / "preguntas_abiertas.xlsx"


def process_survey(
    survey: str,
    config_path: str | Path = QUESTIONS_CONFIG,
) -> pd.DataFrame:
    """Process every active open question of a survey.

    Args:
        survey: logical name registered in preguntas_abiertas.xlsx
            (e.g. "evadoc_evaluacion", "autoevaluacion_acreditacion").
        config_path: path to the configuration file.

    Returns:
        The survey's full DataFrame (all original columns + one
        `_clean` / `_no_stopwords` / `_no_stopwords_no_adverbs` per
        question processed).
    """
    questions = list_open_questions(config_path, survey)
    key_columns = list_key_columns(config_path, survey)

    print(f"Survey '{survey}': {len(questions)} active question(s) to process.")
    print(f"Key columns: {key_columns}\n")

    full_df = None

    for question in questions:
        logical_key = question["logical_key"]

        file_path = resolve_file(config_path, survey, DATA_RAW_DIR)
        sheet, real_column = resolve_column_and_sheet(
            file_path, question["sheet_name"], question["aliases"]
        )

        print(f"── {logical_key} ──")
        print(f"  file: '{file_path.name}' | sheet: '{sheet}' | column: '{real_column}'")

        cleaner = Cleaner(
            file_path=file_path,
            survey_name=survey,
            key_column=real_column,
            sheet_name=sheet,
        )
        cleaner.clean_key_column()
        cleaner.eliminate_stopwords()

        INDEPENDENT_DIR.mkdir(parents=True, exist_ok=True)
        cleaner.export_independent_corpus(
            output_dir=INDEPENDENT_DIR,
            id_columns=key_columns,
            question_label=logical_key,
        )
        print()

        if full_df is None:
            full_df = cleaner.cleaned_data
        else:
            new_columns = [c for c in cleaner.cleaned_data.columns if c not in full_df.columns]
            full_df = full_df.join(cleaner.cleaned_data[new_columns])

    # Deduplicate the full file by key columns (a genuine re-export of
    # the same response), WITHOUT dropping empty rows: "empty" is
    # specific to each question, not to the record as a whole.
    before = len(full_df)
    full_df = full_df.drop_duplicates(subset=key_columns).reset_index(drop=True)
    print(f"Deduplicated the full file by key columns {key_columns}: "
          f"{before - len(full_df)} row(s) removed.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    full_path = PROCESSED_DIR / f"{survey}_procesado.csv"
    full_df.to_csv(full_path, index=False)
    print(f"\nFull file exported: {full_path}")

    return full_df
