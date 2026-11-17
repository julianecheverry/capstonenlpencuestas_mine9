# test_save_cleaned_data.py
"""
Tests for Cleaner.save_cleaned_data().

UPDATED for the v3 project layout. Status of the previous finding:

- RESOLVED (previous Finding A, the call-order regression), via what
  the previous report called "Option C": the method still requires
  cleaned_data to exist (it does not run the pipeline itself), but a
  premature call no longer crashes with a cryptic AttributeError --
  it now raises ValueError with an instructive message telling the
  user exactly which methods to run first. The dead `else` branch
  from earlier versions is also gone.

Test data texts are in Spanish, per the team rule.
"""

import pandas as pd
import pytest

from limpieza import Cleaner

TEST_SURVEY = "encuesta_de_prueba_inexistente"


@pytest.fixture
def real_csv(tmp_path):
    content = pd.DataFrame({
        "id": [1, 2],
        "comentario": ["Muy BUENO!", "  regular  "],
    })
    path = tmp_path / "encuesta.csv"
    content.to_csv(path, index=False)
    return path


def test_saves_the_csv_file_with_the_clean_column(real_csv, tmp_path):
    output_path = tmp_path / "salida.csv"
    instance = Cleaner(file_path=real_csv, survey_name=TEST_SURVEY,
                       key_column="comentario")

    instance.clean_key_column()
    instance.save_cleaned_data(str(output_path))

    assert output_path.exists()
    content = pd.read_csv(output_path)
    assert "comentario_clean" in content.columns
    assert content["comentario_clean"].iloc[0] == "muy bueno"


def test_saves_the_full_pipeline_output_including_stopwords_column(real_csv, tmp_path):
    output_path = tmp_path / "salida_completa.csv"
    instance = Cleaner(file_path=real_csv, survey_name=TEST_SURVEY,
                       key_column="comentario")

    instance.eliminate_stopwords()   # self-healing runs the whole chain
    instance.save_cleaned_data(str(output_path))

    content = pd.read_csv(output_path)
    assert "comentario_no_stopwords" in content.columns


def test_premature_call_raises_value_error_with_instructive_message(real_csv, tmp_path):
    """
    RESOLVED (previous Finding A): calling save_cleaned_data() on a
    fresh instance no longer produces a cryptic AttributeError. It now
    raises ValueError whose message names the methods that must run
    first ('clean_key_column' / 'eliminate_stopwords').
    """
    output_path = tmp_path / "no_deberia_existir.csv"
    instance = Cleaner(file_path=real_csv, survey_name=TEST_SURVEY,
                       key_column="comentario")

    with pytest.raises(ValueError, match="clean_key_column"):
        instance.save_cleaned_data(str(output_path))

    assert not output_path.exists()


def test_invalid_output_path_raises_value_error(real_csv):
    instance = Cleaner(file_path=real_csv, survey_name=TEST_SURVEY,
                       key_column="comentario")
    instance.clean_key_column()

    invalid_path = "/ruta/que/no/existe/salida.csv"
    with pytest.raises(ValueError, match="An error occurred while saving"):
        instance.save_cleaned_data(invalid_path)
