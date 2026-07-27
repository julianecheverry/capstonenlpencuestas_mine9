# test_load_data.py
"""
Tests for Cleaner.load_data().

UPDATED for the new version of limpieza.py. Summary of what changed
since the previous version (see HALLAZGOS_Y_TESTING.md for full detail):

- Finding 3 (str vs Path) is RESOLVED: __init__ now does
  self.file_path = Path(file_path), so passing a str works correctly.
  The corresponding xfail test was removed.
- Finding 4 (dtype != 'object' failing on pandas >= 3.0) is RESOLVED:
  load_data now uses pandas.api.types.is_string_dtype(), which is
  version-agnostic. The corresponding xfail test was removed.
- The generic `except Exception: raise ValueError(...)` wrapper is GONE.
  FileNotFoundError now propagates directly instead of being wrapped
  into ValueError -- tests must be updated accordingly.
- A new line, data[key_column] = data[key_column].fillna(''), runs
  before the dtype check, so NaN values in the key column no longer
  break the text-type validation.
- pd.errors.EmptyDataError is now explicitly caught and re-raised as
  ValueError, covering empty files (this branch did not exist before).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from limpieza import Cleaner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_csv(tmp_path):
    content = pd.DataFrame({
        "id": [1, 2, 3],
        "comment": ["Very good", "  Average  ", "Excellent service"],
    })
    path = tmp_path / "survey.csv"
    content.to_csv(path, index=False)
    return path


@pytest.fixture
def valid_xlsx(tmp_path):
    content = pd.DataFrame({
        "id": [1, 2, 3],
        "comment": ["Very good", "  Average  ", "Excellent service"],
    })
    path = tmp_path / "survey.xlsx"
    content.to_excel(path, index=False)
    return path


@pytest.fixture
def csv_with_numeric_column(tmp_path):
    content = pd.DataFrame({
        "id": [1, 2, 3],
        "comment": [5, 4, 3],
    })
    path = tmp_path / "numeric_survey.csv"
    content.to_csv(path, index=False)
    return path


@pytest.fixture
def csv_missing_key_column(tmp_path):
    content = pd.DataFrame({
        "id": [1, 2, 3],
        "other_column": ["a", "b", "c"],
    })
    path = tmp_path / "incomplete_survey.csv"
    content.to_csv(path, index=False)
    return path


@pytest.fixture
def csv_with_nan_in_key_column(tmp_path):
    content = pd.DataFrame({
        "id": [1, 2, 3],
        "comment": ["bueno", None, "malo"],
    })
    path = tmp_path / "survey_with_nan.csv"
    content.to_csv(path, index=False)
    return path


@pytest.fixture
def empty_csv(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    return path


# ---------------------------------------------------------------------------
# Happy path: CSV -- previously xfail due to Finding 4, now resolved
# ---------------------------------------------------------------------------

def test_loading_a_valid_csv_returns_a_dataframe(valid_csv):
    instance = Cleaner(file_path=valid_csv, key_column="comment")
    result = instance.load_data()

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "comment" in result.columns


def test_loading_a_valid_csv_preserves_original_values(valid_csv):
    instance = Cleaner(file_path=valid_csv, key_column="comment")
    result = instance.load_data()

    assert result["comment"].iloc[1] == "  Average  "


# ---------------------------------------------------------------------------
# Happy path: Excel -- previously untested entirely (Coverage Category B)
# ---------------------------------------------------------------------------

def test_loading_a_valid_xlsx_returns_a_dataframe(valid_xlsx):
    instance = Cleaner(file_path=valid_xlsx, key_column="comment")
    result = instance.load_data()

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "comment" in result.columns


# ---------------------------------------------------------------------------
# str vs Path -- previously xfail due to Finding 3, now resolved
# ---------------------------------------------------------------------------

def test_accepts_file_path_as_string(valid_csv):
    instance = Cleaner(file_path=str(valid_csv), key_column="comment")
    result = instance.load_data()

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# NEW: fillna('') on the key column before the dtype check
# ---------------------------------------------------------------------------

def test_nan_values_in_key_column_are_replaced_with_empty_string(csv_with_nan_in_key_column):
    instance = Cleaner(file_path=csv_with_nan_in_key_column, key_column="comment")
    result = instance.load_data()

    assert result["comment"].tolist() == ["bueno", "", "malo"]


def test_column_that_is_entirely_nan_is_accepted_after_fillna(tmp_path):
    """
    Before fillna('') was added, a column that is entirely NaN would be
    inferred as float64 and would fail the text-type validation. Now it
    is accepted, since fillna('') runs first.
    """
    content = pd.DataFrame({"comment": [None, None, None]})
    path = tmp_path / "all_nan.csv"
    content.to_csv(path, index=False)

    instance = Cleaner(file_path=path, key_column="comment")
    result = instance.load_data()

    assert result["comment"].tolist() == ["", "", ""]


# ---------------------------------------------------------------------------
# Unhappy path: file does not exist
#
# CHANGED BEHAVIOR: the generic except-Exception wrapper that used to
# turn every error into ValueError is gone. FileNotFoundError now
# propagates directly.
# ---------------------------------------------------------------------------

def test_nonexistent_file_raises_file_not_found_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    instance = Cleaner(file_path=missing_path, key_column="comment")

    with pytest.raises(FileNotFoundError):
        instance.load_data()


# ---------------------------------------------------------------------------
# Unhappy path: unsupported file format
# ---------------------------------------------------------------------------

def test_unsupported_format_raises_value_error(tmp_path):
    txt_path = tmp_path / "survey.txt"
    txt_path.write_text("this is neither a csv nor an excel file")

    instance = Cleaner(file_path=txt_path, key_column="comment")

    with pytest.raises(ValueError, match="not supported"):
        instance.load_data()


# ---------------------------------------------------------------------------
# Unhappy path: key column does not exist in the file
# ---------------------------------------------------------------------------

def test_missing_key_column_raises_value_error(csv_missing_key_column):
    instance = Cleaner(file_path=csv_missing_key_column, key_column="comment")

    with pytest.raises(ValueError, match="not found"):
        instance.load_data()


# ---------------------------------------------------------------------------
# Unhappy path: key column exists but is not text type
# ---------------------------------------------------------------------------

def test_numeric_key_column_raises_value_error(csv_with_numeric_column):
    instance = Cleaner(file_path=csv_with_numeric_column, key_column="comment")

    with pytest.raises(ValueError, match="not of text type"):
        instance.load_data()


# ---------------------------------------------------------------------------
# NEW: empty file -- pd.errors.EmptyDataError did not have an explicit
# branch in the previous version
# ---------------------------------------------------------------------------

def test_empty_file_raises_value_error(empty_csv):
    instance = Cleaner(file_path=empty_csv, key_column="comment")

    with pytest.raises(ValueError, match="Error reading the file"):
        instance.load_data()
