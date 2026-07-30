# test_integration.py
"""
Integration tests: the real, chained workflow across both classes.

UPDATED for the v3 project layout. The chain under test is now longer:

    Cleaner.load_data
      -> Cleaner.clean_key_column
      -> Cleaner.eliminate_stopwords     (self-healing, order-free)
      -> Cleaner.save_cleaned_data
      -> SynonymReplacer.synonyms_to_dataframe   (on the saved output)

Everything runs against real files on disk with NO mocks, except the
Word2Vec model inside SynonymReplacer (a ~1 GB download -- mocked for
the same reasons explained in test_synonyms.py; the mock only affects
which synonyms exist, not the mechanics of the chain).

Test data texts are in Spanish, per the team rule.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from limpieza import Cleaner
from synonyms import SynonymReplacer

TEST_SURVEY = "encuesta_de_prueba_inexistente"


@pytest.fixture
def real_survey_csv(tmp_path):
    content = pd.DataFrame({
        "id_encuestado": [1, 2, 3, 4],
        "comentario": [
            "Excelente atención, muy rápido!",
            "  regular, nada especial  ",
            "PÉSIMO. no vuelvo.",
            "Buena relación precio/calidad",
        ],
    })
    path = tmp_path / "encuesta_real.csv"
    content.to_csv(path, index=False)
    return path


def make_synonym_replacer():
    """Replacer with a minimal mocked model (see test_synonyms.py)."""
    replacer = SynonymReplacer()
    mock_model = MagicMock()
    mock_model.key_to_index = {"bueno": 0}
    mock_model.most_similar.side_effect = (
        lambda seed, topn: [("buena", 0.9)] if seed == "bueno" else []
    )
    replacer.model = mock_model
    replacer.build_synonyms_dict()
    return replacer


# ---------------------------------------------------------------------------
# Full chain, end to end
# ---------------------------------------------------------------------------

def test_full_chain_from_raw_file_to_synonyms_column(real_survey_csv, tmp_path):
    output_path = tmp_path / "salida_limpia.csv"

    # --- Cleaner: load, clean, remove stopwords, save ---
    instance = Cleaner(file_path=real_survey_csv, survey_name=TEST_SURVEY,
                       key_column="comentario")
    result = instance.eliminate_stopwords()      # self-healing chain
    instance.save_cleaned_data(str(output_path))

    assert output_path.exists()
    assert result["comentario_clean"].iloc[0] == "excelente atencion muy rapido"

    # --- SynonymReplacer: applied over the saved, reloaded output ---
    saved = pd.read_csv(output_path)
    # save/reload round-trip turns empty strings into NaN; normalize
    # like a real downstream consumer would:
    saved["comentario_no_stopwords"] = (
        saved["comentario_no_stopwords"].fillna('')
    )

    replacer = make_synonym_replacer()
    final = replacer.synonyms_to_dataframe(saved, "comentario_no_stopwords")

    assert "comentario_no_stopwords_synonyms" in final.columns
    # 'buena' (row 4) was mapped to its canonical 'bueno':
    assert "bueno" in final["comentario_no_stopwords_synonyms"].iloc[3]


def test_full_chain_stops_cleanly_on_missing_key_column(tmp_path):
    content = pd.DataFrame({"id": [1], "otra": ["texto"]})
    path = tmp_path / "columna_incorrecta.csv"
    content.to_csv(path, index=False)

    instance = Cleaner(file_path=path, survey_name=TEST_SURVEY,
                       key_column="comentario")

    with pytest.raises(ValueError, match="not found"):
        instance.load_data()
    with pytest.raises(ValueError, match="not found"):
        instance.clean_key_column()
    with pytest.raises(ValueError, match="not found"):
        instance.eliminate_stopwords()


def test_full_chain_stops_cleanly_on_nonexistent_file(tmp_path):
    missing_path = tmp_path / "no_existe.csv"
    instance = Cleaner(file_path=missing_path, survey_name=TEST_SURVEY,
                       key_column="comentario")

    with pytest.raises(FileNotFoundError):
        instance.load_data()
    with pytest.raises(FileNotFoundError):
        instance.eliminate_stopwords()


def test_two_surveys_get_independent_stopword_sets(real_survey_csv):
    """
    Two Cleaner instances with different survey names must not share
    stopword state: each __init__ copies the module-level set before
    adding its survey-specific words. Adding a word manually to one
    instance's set must not affect the other.
    """
    a = Cleaner(file_path=real_survey_csv, survey_name="encuesta_a",
                key_column="comentario")
    b = Cleaner(file_path=real_survey_csv, survey_name="encuesta_b",
                key_column="comentario")

    a.stop_words.add("palabra_exclusiva_de_a")

    assert "palabra_exclusiva_de_a" in a.stop_words
    assert "palabra_exclusiva_de_a" not in b.stop_words
