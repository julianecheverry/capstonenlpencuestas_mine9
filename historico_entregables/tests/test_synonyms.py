# test_synonyms.py
"""
Tests for the SynonymReplacer class (synonyms.py) -- NEW in v3.

The real Word2Vec model (SBW-vectors-300-min5.bin) weighs ~1 GB and
would be downloaded from the internet on first use. Downloading or
loading it inside a test suite is not viable, so every test here that
needs a model uses a MagicMock configured per-seed:

- mock.key_to_index defines which seed words "exist" in the vocabulary.
- mock.most_similar uses side_effect with a per-seed dictionary, so
  each seed returns its own realistic candidates. (A single fixed
  return_value would wrongly map every candidate to whichever canonical
  category was processed last -- confirmed while designing this suite.)

What is NOT covered here, deliberately: _download_model and load_model
against the real file/URL. Those are thin wrappers over urllib/gensim
and would require network access and gigabytes of data; they remain
exercised only in real usage.

Test data texts are in Spanish, per the team rule.

NEW FINDING documented below: multi-word colloquialisms defined in
settings.colombian_colloquialisms ("una chimba", "una nota", "un paseo")
can never match, because replace_synonyms tokenizes word by word --
a dictionary key containing a space will never equal a single token.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from synonyms import SynonymReplacer
from settings import colombian_colloquialisms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_replacer_with_mock_model():
    """
    Builds a SynonymReplacer with a mocked Word2Vec model configured
    per-seed: each seed word returns its own candidate list, mimicking
    how the real model behaves (different neighbors per word).
    """
    replacer = SynonymReplacer()

    per_seed_candidates = {
        "bueno":     [("buenisimo", 0.90), ("agradable", 0.70),
                      ("regular", 0.40)],
        "excelente": [("optimo", 0.80)],
        "malo":      [("malisimo", 0.88), ("aceptable", 0.30)],
        "rapido":    [("veloz", 0.75)],
    }

    mock_model = MagicMock()
    mock_model.key_to_index = {
        seed: idx for idx, seed in enumerate(per_seed_candidates)
    }
    mock_model.most_similar.side_effect = (
        lambda seed, topn: per_seed_candidates[seed]
    )

    replacer.model = mock_model
    return replacer


# ---------------------------------------------------------------------------
# Unhappy paths: methods called out of order
# ---------------------------------------------------------------------------

def test_build_synonyms_dict_without_model_raises_value_error():
    replacer = SynonymReplacer()

    with pytest.raises(ValueError, match="load_model"):
        replacer.build_synonyms_dict()


def test_replace_synonyms_without_dict_raises_value_error():
    replacer = SynonymReplacer()

    with pytest.raises(ValueError, match="build_synonyms_dict"):
        replacer.replace_synonyms("la clase fue buena")


def test_synonyms_to_dataframe_without_dict_raises_value_error():
    replacer = SynonymReplacer()
    data = pd.DataFrame({"comentario": ["todo bien"]})

    with pytest.raises(ValueError, match="build_synonyms_dict"):
        replacer.synonyms_to_dataframe(data, "comentario")


def test_synonyms_to_dataframe_with_missing_column_raises_value_error():
    replacer = SynonymReplacer()
    replacer.synonyms_dict = {}   # built, but the column does not exist
    data = pd.DataFrame({"comentario": ["todo bien"]})

    with pytest.raises(ValueError, match="not found"):
        replacer.synonyms_to_dataframe(data, "columna_inexistente")


# ---------------------------------------------------------------------------
# _expand_seed: threshold filtering and vocabulary check
# ---------------------------------------------------------------------------

def test_expand_seed_keeps_only_candidates_above_threshold():
    replacer = make_replacer_with_mock_model()

    expanded = replacer._expand_seed("bueno", "bueno")

    # 0.90 and 0.70 pass the 0.65 threshold; 0.40 does not.
    assert expanded == {"buenisimo": "bueno", "agradable": "bueno"}


def test_expand_seed_returns_empty_dict_for_unknown_seed(capsys):
    replacer = make_replacer_with_mock_model()

    expanded = replacer._expand_seed("bueno", "palabra_fuera_del_vocabulario")

    assert expanded == {}
    # The method warns (via print) instead of raising -- documented as-is.
    assert "not found in the" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# build_synonyms_dict: composition of the three sources
# ---------------------------------------------------------------------------

def test_build_synonyms_dict_combines_seeds_expansion_and_colloquialisms():
    replacer = make_replacer_with_mock_model()

    result = replacer.build_synonyms_dict()

    # 1. Identity mapping for every canonical term:
    assert result["bueno"] == "bueno"
    assert result["malo"] == "malo"

    # 2. Auto-expanded words above threshold, mapped per-seed:
    assert result["buenisimo"] == "bueno"
    assert result["optimo"] == "bueno"      # seed 'excelente' -> canonical 'bueno'
    assert result["malisimo"] == "malo"
    assert result["veloz"] == "rapido"

    # 3. Below-threshold candidates excluded:
    assert "regular" not in result
    assert "aceptable" not in result

    # 4. Manual colloquialisms present (they are applied last):
    assert result["chevere"] == "bueno"

    # 5. The dict is stored on the instance as well:
    assert replacer.synonyms_dict is result


# ---------------------------------------------------------------------------
# replace_synonyms: word-level replacement
# ---------------------------------------------------------------------------

def test_replace_synonyms_replaces_known_words_and_keeps_the_rest():
    replacer = make_replacer_with_mock_model()
    replacer.build_synonyms_dict()

    result = replacer.replace_synonyms("el curso fue buenisimo y chevere")

    assert result == "el curso fue bueno y bueno"


def test_replace_synonyms_is_case_insensitive_on_lookup():
    replacer = make_replacer_with_mock_model()
    replacer.build_synonyms_dict()

    result = replacer.replace_synonyms("BUENISIMO el servicio")

    assert result.split()[0] == "bueno"


def test_replace_synonyms_returns_empty_string_for_non_text():
    replacer = make_replacer_with_mock_model()
    replacer.build_synonyms_dict()

    assert replacer.replace_synonyms(None) == ''
    assert replacer.replace_synonyms(5) == ''
    assert replacer.replace_synonyms(float('nan')) == ''


# ---------------------------------------------------------------------------
# NEW FINDING: multi-word colloquialisms can never match
# ---------------------------------------------------------------------------

def test_multi_word_colloquialisms_are_unreachable():
    """
    FINDING (confirmed at runtime): settings.colombian_colloquialisms
    contains multi-word keys ("una chimba", "una nota", "un paseo"),
    but replace_synonyms tokenizes the text word by word with
    nltk.word_tokenize and looks each token up individually. A key
    containing a space can never equal a single token, so those
    entries are dead configuration: "una chimba de curso" becomes
    "una bueno de curso" (only the single word 'chimba' matches),
    never "bueno de curso" as a phrase replacement.
    """
    multi_word_keys = [k for k in colombian_colloquialisms if " " in k]
    if not multi_word_keys:
        pytest.skip("No multi-word colloquialisms defined in settings")

    replacer = make_replacer_with_mock_model()
    replacer.build_synonyms_dict()

    result = replacer.replace_synonyms("el curso fue una chimba")

    # The phrase was NOT replaced as a whole ("una" survives)...
    assert "una" in result.split()
    # ...only the individual word 'chimba' matched its single-word entry.
    assert "chimba" not in result.split()


# ---------------------------------------------------------------------------
# synonyms_to_dataframe: column-level application
# ---------------------------------------------------------------------------

def test_synonyms_to_dataframe_creates_the_synonyms_column():
    replacer = make_replacer_with_mock_model()
    replacer.build_synonyms_dict()

    data = pd.DataFrame({
        "comentario_clean": ["curso buenisimo", "profesor malisimo", ""],
    })
    result = replacer.synonyms_to_dataframe(data, "comentario_clean")

    assert "comentario_clean_synonyms" in result.columns
    assert result["comentario_clean_synonyms"].iloc[0] == "curso bueno"
    assert result["comentario_clean_synonyms"].iloc[1] == "profesor malo"
    assert result["comentario_clean_synonyms"].iloc[2] == ""


def test_synonyms_to_dataframe_column_name_follows_source_column():
    replacer = make_replacer_with_mock_model()
    replacer.build_synonyms_dict()

    data = pd.DataFrame({"otra_columna": ["texto bueno"]})
    result = replacer.synonyms_to_dataframe(data, "otra_columna")

    assert "otra_columna_synonyms" in result.columns
