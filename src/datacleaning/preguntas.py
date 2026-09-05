"""
preguntas.py

Lookup and resolution of open-question columns from an EXTERNAL
configuration file (data/utilities/preguntas_abiertas.xlsx), maintained
by the Quality Assurance team -- not by developers. No question or
column name lives in this code.

NOTE ON LANGUAGE: the configuration file itself (preguntas_abiertas.xlsx)
keeps its sheet names and column headers in Spanish (`preguntas`,
`llaves`, `archivos`, `encuesta`, `clave_logica`, etc.), since it is a
business artifact edited directly by Quality staff, not a code file.
Everything in this module (function/variable names, docstrings,
messages) is in English, per the project's language convention.

Expected schema of preguntas_abiertas.xlsx:

  Sheet 'preguntas': encuesta | clave_logica | alias_redaccion
                     | hoja_excel | archivo | activa
    One row per KNOWN WORDING of a question. If a question's wording
    changes over time, a NEW row is added with the same
    (encuesta, clave_logica) and the new wording -- the previous row is
    never edited or deleted, to keep the history.

  Sheet 'llaves': encuesta | columna_llave
    One row per identifier column that must accompany the export of
    every question for that survey (e.g. Periodo, ID de respuesta, and
    that survey's specific cross-reference column).

  Sheet 'archivos': encuesta | alias_archivo
    One row per KNOWN file name for a survey. If the source file gets
    renamed, a new row is added with the current name -- the previous
    one is kept for traceability.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from limpieza import Cleaner


def normalize(text: str) -> str:
    """Normalize text for column-name matching.

    Reuses `Cleaner._clean_text` (lowercase, accent removal via
    `settings.accents`, punctuation stripping) so that column-name
    matching stays in sync with the same normalization rules used
    everywhere else in the pipeline, instead of maintaining a second,
    independent implementation.
    """
    cleaned = Cleaner._clean_text(str(text))
    return re.sub(r"\s+", " ", cleaned).strip()


def _read_sheet(config_path: str | Path, sheet_name: str, expected_columns: set[str]) -> pd.DataFrame:
    df = pd.read_excel(config_path, sheet_name=sheet_name)
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in sheet '{sheet_name}' of {config_path}.")
    return df


def list_open_questions(config_path: str | Path, survey: str) -> list[dict]:
    """Return the ACTIVE open questions for a survey, exactly as
    registered in the configuration file -- without the code knowing in
    advance how many there are or what they are called.

    Returns:
        A list of dicts with: logical_key, aliases (list of known
        wordings), sheet_name, file_name.

    Raises:
        ValueError: If the survey has no active questions registered.
    """
    df = _read_sheet(
        config_path, "preguntas",
        {"encuesta", "clave_logica", "alias_redaccion", "hoja_excel", "archivo", "activa"},
    )
    df = df[
        (df["encuesta"] == survey)
        & (df["activa"].astype(str).str.strip().str.lower() == "si")
    ]

    if df.empty:
        raise ValueError(
            f"No active questions registered for survey '{survey}' in "
            f"{config_path}. Check the 'preguntas' sheet."
        )

    questions = []
    for logical_key, group in df.groupby("clave_logica", sort=False):
        questions.append({
            "logical_key": logical_key,
            "aliases": group["alias_redaccion"].dropna().astype(str).tolist(),
            "sheet_name": group["hoja_excel"].iloc[0],
            "file_name": group["archivo"].iloc[0],
        })
    return questions


def list_key_columns(config_path: str | Path, survey: str) -> list[str]:
    """Return the identifier (key) columns for a survey.

    Raises:
        ValueError: If the survey has no key columns registered.
    """
    df = _read_sheet(config_path, "llaves", {"encuesta", "columna_llave"})
    keys = df.loc[df["encuesta"] == survey, "columna_llave"].dropna().astype(str).tolist()
    if not keys:
        raise ValueError(
            f"No key columns registered for survey '{survey}' in "
            f"{config_path}. Check the 'llaves' sheet."
        )
    return keys


def resolve_column(df: pd.DataFrame, aliases: list[str]) -> str:
    """Find, among the real columns of df, which one matches one of the
    known wordings (aliases). Fails explicitly if none match -- never
    guesses the closest one.

    Raises:
        ValueError: If no known wording matches.
    """
    normalized_columns = {normalize(c): c for c in df.columns}
    for alias in aliases:
        key = normalize(alias)
        if key in normalized_columns:
            return normalized_columns[key]

    raise ValueError(
        f"None of the known wordings match the file's columns.\n"
        f"Tried wordings: {aliases}\n"
        f"Available columns: {list(df.columns)}\n"
        f"If the question was reworded, add a new row in "
        f"preguntas_abiertas.xlsx with the current wording (do not delete the old one)."
    )


def resolve_file(config_path: str | Path, survey: str, data_dir: str | Path) -> Path:
    """Find, among the known file names for a survey (sheet 'archivos'),
    which one actually exists in `data_dir`.

    If the file was renamed, add a new row in the 'archivos' sheet with
    the current name -- do not delete the previous one, so the history
    is kept and the pipeline keeps working even if an old copy is
    received by mistake.

    Raises:
        ValueError: If the survey has no file names registered.
        FileNotFoundError: If none of the known names exist.
    """
    df = _read_sheet(config_path, "archivos", {"encuesta", "alias_archivo"})
    candidates = df.loc[df["encuesta"] == survey, "alias_archivo"].dropna().astype(str).tolist()

    if not candidates:
        raise ValueError(
            f"No file names registered for survey '{survey}' in "
            f"{config_path}. Check the 'archivos' sheet."
        )

    for name in candidates:
        path = Path(data_dir) / name
        if path.exists():
            return path

    raise FileNotFoundError(
        f"None of the known file names for '{survey}' exist in {data_dir}.\n"
        f"Tried: {candidates}\n"
        f"If the file was renamed, add a new row in the 'archivos' sheet "
        f"of preguntas_abiertas.xlsx with the current name."
    )


def resolve_column_and_sheet(
    file_path: str | Path, preferred_sheet: str, aliases: list[str]
) -> tuple[str, str]:
    """Resolve the column by trying `preferred_sheet` first; if that
    sheet doesn't exist or doesn't contain any known wording, scan
    EVERY sheet in the file looking for one that does.

    Unlike the file name, this does NOT require anyone to register the
    new sheet name in advance -- it is discovered automatically. It
    only warns (does not fail) when the column is found in a sheet
    other than the registered one, so the configuration file can be
    updated when convenient.

    Returns:
        (actual_sheet_name, actual_column_name)

    Raises:
        ValueError: If no sheet in the file contains the column.
    """
    workbook = pd.ExcelFile(file_path)
    sheets_to_try = [preferred_sheet] + [s for s in workbook.sheet_names if s != preferred_sheet]

    for sheet in sheets_to_try:
        if sheet not in workbook.sheet_names:
            continue
        headers = pd.read_excel(workbook, sheet_name=sheet, nrows=0)
        try:
            column = resolve_column(headers, aliases)
        except ValueError:
            continue

        if sheet != preferred_sheet:
            print(
                f"Notice: the question is no longer in sheet '{preferred_sheet}' "
                f"(as registered in preguntas_abiertas.xlsx) -- found in "
                f"'{sheet}' instead. Update 'hoja_excel' in the configuration "
                f"file when you get a chance."
            )
        return sheet, column

    raise ValueError(
        f"No sheet in '{file_path}' contains a column matching the known "
        f"wordings: {aliases}"
    )
