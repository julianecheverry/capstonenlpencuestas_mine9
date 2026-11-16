# test_load_data.py
"""
Tests for Cleaner.load_data().

UPDATED for the v3 project layout:
- The Cleaner constructor now requires a `survey_name` positional
  parameter (used to look up survey-specific stopwords in
  settings.stopwords_dict). These tests use a survey name that does not
  need to exist in the real stopwords Excel: when the name is unknown,
  stopwords_dict.get(survey_name, []) simply returns an empty list,
  which is a guaranteed behavior regardless of the Excel's content.
- All test data texts are in Spanish, per the team rule.
"""

import pandas as pd
import pytest

from limpieza import Cleaner

# A survey name deliberately NOT expected to exist in stopwords.xlsx:
# guarantees new_stopwords = [] regardless of the real file's content.
TEST_SURVEY = "encuesta_de_prueba_inexistente"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_csv(tmp_path):
    content = pd.DataFrame({
        "id": [1, 2, 3],
        "comentario": ["Muy bueno", "  Regular  ", "Excelente atención"],
    })
    path = tmp_path / "encuesta.csv"
    content.to_csv(path, index=False)
    return path


@pytest.fixture
def valid_xlsx(tmp_path):
    content = pd.DataFrame({
        "id": [1, 2, 3],
        "comentario": ["Muy bueno", "  Regular  ", "Excelente atención"],
    })
    path = tmp_path / "encuesta.xlsx"
    content.to_excel(path, index=False)
    return path


@pytest.fixture
def csv_with_numeric_column(tmp_path):
    content = pd.DataFrame({
        "id": [1, 2, 3],
        "comentario": [5, 4, 3],
    })
    path = tmp_path / "encuesta_numerica.csv"
    content.to_csv(path, index=False)
    return path


@pytest.fixture
def csv_missing_key_column(tmp_path):
    content = pd.DataFrame({
        "id": [1, 2, 3],
        "otra_columna": ["a", "b", "c"],
    })
    path = tmp_path / "encuesta_incompleta.csv"
    content.to_csv(path, index=False)
    return path


@pytest.fixture
def csv_with_nan_in_key_column(tmp_path):
    content = pd.DataFrame({
        "id": [1, 2, 3],
        "comentario": ["bueno", None, "malo"],
    })
    path = tmp_path / "encuesta_con_vacios.csv"
    content.to_csv(path, index=False)
    return path


@pytest.fixture
def empty_csv(tmp_path):
    path = tmp_path / "vacio.csv"
    path.write_text("")
    return path


# ---------------------------------------------------------------------------
# Happy path: CSV
# ---------------------------------------------------------------------------

def test_loading_a_valid_csv_returns_a_dataframe(valid_csv):
    instance = Cleaner(file_path=valid_csv, survey_name=TEST_SURVEY,
                       key_column="comentario")
    result = instance.load_data()

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "comentario" in result.columns


def test_loading_a_valid_csv_preserves_original_values(valid_csv):
    # load_data must NOT clean anything yet -- that is the
    # responsibility of _clean_text / clean_key_column.
    instance = Cleaner(file_path=valid_csv, survey_name=TEST_SURVEY,
                       key_column="comentario")
    result = instance.load_data()

    assert result["comentario"].iloc[1] == "  Regular  "


# ---------------------------------------------------------------------------
# Happy path: Excel
# ---------------------------------------------------------------------------

def test_loading_a_valid_xlsx_returns_a_dataframe(valid_xlsx):
    instance = Cleaner(file_path=valid_xlsx, survey_name=TEST_SURVEY,
                       key_column="comentario")
    result = instance.load_data()

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# str vs Path: the constructor converts with Path(file_path)
# ---------------------------------------------------------------------------

def test_accepts_file_path_as_string(valid_csv):
    instance = Cleaner(file_path=str(valid_csv), survey_name=TEST_SURVEY,
                       key_column="comentario")
    result = instance.load_data()

    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# fillna('') on the key column before the dtype check
# ---------------------------------------------------------------------------

def test_nan_values_in_key_column_are_replaced_with_empty_string(csv_with_nan_in_key_column):
    instance = Cleaner(file_path=csv_with_nan_in_key_column,
                       survey_name=TEST_SURVEY, key_column="comentario")
    result = instance.load_data()

    assert result["comentario"].tolist() == ["bueno", "", "malo"]


def test_column_that_is_entirely_nan_is_accepted_after_fillna(tmp_path):
    content = pd.DataFrame({"comentario": [None, None, None]})
    path = tmp_path / "todo_vacio.csv"
    content.to_csv(path, index=False)

    instance = Cleaner(file_path=path, survey_name=TEST_SURVEY,
                       key_column="comentario")
    result = instance.load_data()

    assert result["comentario"].tolist() == ["", "", ""]


# ---------------------------------------------------------------------------
# Unhappy paths
# ---------------------------------------------------------------------------

def test_nonexistent_file_raises_file_not_found_error(tmp_path):
    missing_path = tmp_path / "no_existe.csv"
    instance = Cleaner(file_path=missing_path, survey_name=TEST_SURVEY,
                       key_column="comentario")

    with pytest.raises(FileNotFoundError):
        instance.load_data()


def test_unsupported_format_raises_value_error(tmp_path):
    txt_path = tmp_path / "encuesta.txt"
    txt_path.write_text("esto no es un csv ni un excel")

    instance = Cleaner(file_path=txt_path, survey_name=TEST_SURVEY,
                       key_column="comentario")

    with pytest.raises(ValueError, match="not supported"):
        instance.load_data()


def test_missing_key_column_raises_value_error(csv_missing_key_column):
    instance = Cleaner(file_path=csv_missing_key_column,
                       survey_name=TEST_SURVEY, key_column="comentario")

    with pytest.raises(ValueError, match="not found"):
        instance.load_data()


def test_numeric_key_column_raises_value_error(csv_with_numeric_column):
    instance = Cleaner(file_path=csv_with_numeric_column,
                       survey_name=TEST_SURVEY, key_column="comentario")

    with pytest.raises(ValueError, match="not of text type"):
        instance.load_data()


def test_empty_file_raises_value_error(empty_csv):
    instance = Cleaner(file_path=empty_csv, survey_name=TEST_SURVEY,
                       key_column="comentario")

    with pytest.raises(ValueError, match="Error reading the file"):
        instance.load_data()
