# test_clean_key_column.py
"""
Tests for Cleaner.clean_key_column().

UPDATED for the v3 project layout:
- Constructor requires survey_name.
- The docstring-mismatch finding from the previous review is RESOLVED:
  the docstring now correctly states that a DataFrame is returned. The
  test that documented that finding was updated into a plain
  return-contract test.
- self.cleaned_data is now initialized to None in __init__ (previous
  Finding 2 infrastructure), and set by this method on success.
- All test data texts are in Spanish, per the team rule.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from limpieza import Cleaner

TEST_SURVEY = "encuesta_de_prueba_inexistente"


@pytest.fixture
def real_csv(tmp_path):
    content = pd.DataFrame({
        "id": [1, 2, 3],
        "comentario": ["Muy BUENO!", "  regular  ", "Pésimo servicio"],
    })
    path = tmp_path / "encuesta.csv"
    content.to_csv(path, index=False)
    return path


def test_adds_clean_column_without_modifying_the_original(real_csv):
    instance = Cleaner(file_path=real_csv, survey_name=TEST_SURVEY,
                       key_column="comentario")
    result = instance.clean_key_column()

    assert result["comentario"].iloc[0] == "Muy BUENO!"
    assert "comentario_clean" in result.columns
    assert result["comentario_clean"].iloc[0] == "muy bueno"
    assert result["comentario_clean"].iloc[2] == "pesimo servicio"


def test_new_column_name_uses_key_column_dynamically(tmp_path):
    content = pd.DataFrame({"respuesta_abierta": ["Buen servicio"]})
    path = tmp_path / "encuesta.csv"
    content.to_csv(path, index=False)

    instance = Cleaner(file_path=path, survey_name=TEST_SURVEY,
                       key_column="respuesta_abierta")
    result = instance.clean_key_column()

    assert "respuesta_abierta_clean" in result.columns


def test_does_not_mutate_the_original_dataframe_in_place():
    """
    Mocking is used here specifically to hold a reference to the exact
    DataFrame object load_data returns, so we can confirm the .copy()
    inside clean_key_column protects it from getting the new column.
    """
    instance = Cleaner(file_path="no_relevante.csv", survey_name=TEST_SURVEY,
                       key_column="comentario")
    original_data = pd.DataFrame({"comentario": ["Buen servicio"]})

    with patch.object(Cleaner, "load_data", return_value=original_data):
        instance.clean_key_column()

    assert "comentario_clean" not in original_data.columns


def test_propagates_the_error_if_load_data_fails(tmp_path):
    missing_path = tmp_path / "no_existe.csv"
    instance = Cleaner(file_path=missing_path, survey_name=TEST_SURVEY,
                       key_column="comentario")

    with pytest.raises(FileNotFoundError):
        instance.clean_key_column()


def test_sets_self_cleaned_data_on_success(real_csv):
    # cleaned_data starts as None (initialized in __init__, an
    # improvement over previous versions where the attribute did not
    # exist at all until some method created it).
    instance = Cleaner(file_path=real_csv, survey_name=TEST_SURVEY,
                       key_column="comentario")

    assert instance.cleaned_data is None
    instance.clean_key_column()
    assert instance.cleaned_data is not None
    assert "comentario_clean" in instance.cleaned_data.columns


def test_returns_the_same_dataframe_stored_in_cleaned_data(real_csv):
    # Return contract (the docstring now correctly documents this):
    # the method returns self.cleaned_data itself, a pd.DataFrame.
    instance = Cleaner(file_path=real_csv, survey_name=TEST_SURVEY,
                       key_column="comentario")
    result = instance.clean_key_column()

    assert isinstance(result, pd.DataFrame)
    assert result is instance.cleaned_data
