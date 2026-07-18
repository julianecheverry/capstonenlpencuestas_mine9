# test_clean_key_column.py
"""
Tests for Cleaner.clean_key_column().

UPDATED for the new version of limpieza.py:

- Since Findings 3 and 4 in load_data are resolved, these tests no
  longer need to mock load_data()'s return value to work around a
  dtype/path bug -- a real CSV fixture is now enough for the happy path.
  Mocking is still used in a couple of tests where the point is to
  isolate clean_key_column's own logic regardless of what load_data
  returns (e.g. confirming the .copy() behavior).
- New: clean_key_column() now also sets self.cleaned_data as a side
  effect (it used to only be set inside save_cleaned_data() in the
  previous version). This is the new origin point for Finding 2.
- Documentation finding: the docstring for clean_key_column says
  "Returns: None", but the method actually returns self.cleaned_data
  (a DataFrame). A test below confirms the real return value to make
  this inconsistency explicit and traceable.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from limpieza import Cleaner


@pytest.fixture
def real_csv(tmp_path):
    content = pd.DataFrame({
        "id": [1, 2, 3],
        "comment": ["Very GOOD!", "  average  ", "Terrible service"],
    })
    path = tmp_path / "survey.csv"
    content.to_csv(path, index=False)
    return path


def test_adds_clean_column_without_modifying_the_original(real_csv):
    instance = Cleaner(file_path=real_csv, key_column="comment")
    result = instance.clean_key_column()

    assert result["comment"].iloc[0] == "Very GOOD!"
    assert "comment_clean" in result.columns
    assert result["comment_clean"].iloc[0] == "very good"


def test_new_column_name_uses_key_column_dynamically(tmp_path):
    content = pd.DataFrame({"open_response": ["Buen servicio"]})
    path = tmp_path / "survey.csv"
    content.to_csv(path, index=False)

    instance = Cleaner(file_path=path, key_column="open_response")
    result = instance.clean_key_column()

    assert "open_response_clean" in result.columns


def test_does_not_mutate_the_original_dataframe_in_place():
    """
    Mocking is used here specifically to get a reference to the
    "original" DataFrame object as returned by load_data, so we can
    confirm clean_key_column's .copy() does not add the new column
    to it.
    """
    instance = Cleaner(file_path="not_relevant.csv", key_column="comment")
    original_data = pd.DataFrame({"comment": ["Buen servicio"]})

    with patch.object(Cleaner, "load_data", return_value=original_data):
        instance.clean_key_column()

    assert "comment_clean" not in original_data.columns


def test_propagates_the_error_if_load_data_fails(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    instance = Cleaner(file_path=missing_path, key_column="comment")

    with pytest.raises(FileNotFoundError):
        instance.clean_key_column()


# ---------------------------------------------------------------------------
# self.cleaned_data is now set here, not only in save_cleaned_data()
# (the new origin point of Finding 2 -- see test_stopwords.py and
# test_save_cleaned_data.py for the consequences of this change)
# ---------------------------------------------------------------------------

def test_sets_self_cleaned_data_as_a_side_effect(real_csv):
    instance = Cleaner(file_path=real_csv, key_column="comment")

    assert not hasattr(instance, "cleaned_data")  # not set before calling
    instance.clean_key_column()
    assert hasattr(instance, "cleaned_data")
    assert "comment_clean" in instance.cleaned_data.columns


# ---------------------------------------------------------------------------
# Documentation finding: the docstring says "Returns: None", but the
# method actually returns a DataFrame
# ---------------------------------------------------------------------------

def test_return_value_is_a_dataframe_despite_docstring_saying_none(real_csv):
    """
    FINDING: clean_key_column's docstring states 'Returns: None', but
    the method ends with 'return self.cleaned_data', which is a
    pd.DataFrame. This test documents the real, observable behavior
    (a DataFrame is returned), which contradicts the written docstring.
    """
    instance = Cleaner(file_path=real_csv, key_column="comment")
    result = instance.clean_key_column()

    assert isinstance(result, pd.DataFrame)
    assert result is instance.cleaned_data
