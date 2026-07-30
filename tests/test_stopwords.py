# test_stopwords.py
"""
Tests for the stopwords functionality: Cleaner._clean_stopwords,
Cleaner.eliminate_stopwords, and the module-level stopwords assembly.

UPDATED for the v3 project layout. Status of previous findings:

- RESOLVED: the call-order dependency (previous Finding 2). The
  self-healing block in eliminate_stopwords is now active: calling it
  as the very first method on a fresh instance works, because it runs
  clean_key_column() itself when cleaned_data is still None.
- NEW MECHANISM: stopwords now come from three sources -- the NLTK
  Spanish corpus, a "global" sheet in data/utilities/stopwords.xlsx,
  and a per-survey sheet keyed by the survey_name constructor argument.

- NEW FINDING (documented below, confirmed at runtime): the global
  stopwords never actually reach the stopword set. limpieza.py does
  `stop_words.update(stopwords_global)` where stopwords_global is a
  DICT of {survey: [words]}; set.update() over a dict adds its KEYS
  (the survey names), not the word lists. The words in the 'globales'
  sheet are silently ignored.

Test data texts are in Spanish, per the team rule.
"""

import pandas as pd
import pytest

import limpieza
from limpieza import Cleaner
from settings import stopwords_dict, stopwords_global

TEST_SURVEY = "encuesta_de_prueba_inexistente"


@pytest.fixture
def real_csv(tmp_path):
    content = pd.DataFrame({
        "comentario": ["El profesor explica de una manera excelente"],
    })
    path = tmp_path / "encuesta.csv"
    content.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# _clean_stopwords: direct behavior
# ---------------------------------------------------------------------------

def test_clean_stopwords_removes_spanish_nltk_stopwords(real_csv):
    instance = Cleaner(file_path=real_csv, survey_name=TEST_SURVEY,
                       key_column="comentario")
    result = instance._clean_stopwords("el gato come pescado")
    assert result == "gato come pescado"


def test_clean_stopwords_respects_instance_specific_words(real_csv):
    """
    The filtering uses self.stop_words (an instance-level copy), so a
    word added to one instance's set is filtered by that instance only.
    This validates the per-survey mechanism without depending on the
    real content of stopwords.xlsx.
    """
    instance = Cleaner(file_path=real_csv, survey_name=TEST_SURVEY,
                       key_column="comentario")
    instance.stop_words.add("pescado")

    assert instance._clean_stopwords("el gato come pescado") == "gato come"


def test_unknown_survey_name_yields_no_extra_stopwords(real_csv):
    # Guaranteed behavior regardless of the Excel content:
    # stopwords_dict.get(unknown_name, []) returns an empty list.
    instance = Cleaner(file_path=real_csv,
                       survey_name="nombre_que_no_existe_en_el_excel",
                       key_column="comentario")
    assert instance.new_stopwords == []


def test_known_survey_name_loads_its_specific_stopwords(real_csv):
    """
    For every survey actually present in the real stopwords.xlsx, its
    words must end up in the instance's stop_words set. Parametrizing
    over the real stopwords_dict keeps this test valid for any content
    the team maintains in the Excel file. Skipped if the sheet is empty.
    """
    if not stopwords_dict:
        pytest.skip("No per-survey stopwords defined in stopwords.xlsx")

    for survey_name, words in stopwords_dict.items():
        instance = Cleaner(file_path=real_csv, survey_name=survey_name,
                           key_column="comentario")
        assert instance.new_stopwords == words
        assert set(words).issubset(instance.stop_words)


# ---------------------------------------------------------------------------
# RESOLVED: stop_words.update() over the raw dict is fixed
# ---------------------------------------------------------------------------

def test_set_update_over_a_dict_used_to_add_keys_not_word_lists():
    """
    HISTORICAL MECHANISM (no longer present in limpieza.py): calling
    set.update() with a raw dict adds its KEYS, not its values. Kept
    as a plain Python fact-check, decoupled from limpieza.py, both to
    document why the original bug happened and to guard against a
    future regression back to `stop_words.update(stopwords_global)`
    (passing the whole dict) instead of the current, fixed line:
    `stop_words.update(stopwords_global.get("stopwords_global", []))`.
    """
    demo = set()
    demo.update({"stopwords_global": ["nr", "academico"]})

    assert demo == {"stopwords_global"}   # what the OLD bug produced
    assert "nr" not in demo


def test_global_stopword_words_now_reach_the_module_stop_words():
    """
    RESOLVED: limpieza.py now extracts the word list before calling
    update() -- `stopwords_global.get("stopwords_global", [])` --
    instead of passing the whole dict. Every real word from the
    'globales' sheet of stopwords.xlsx is confirmed present in the
    module-level stop_words set, and the old leaked dict key is
    confirmed gone.
    """
    if not stopwords_global:
        pytest.skip("No global stopwords defined in stopwords.xlsx")

    for key, words in stopwords_global.items():
        for word in words:
            assert word in limpieza.stop_words
        # The dict key itself must NOT leak in anymore:
        assert key not in limpieza.stop_words


# ---------------------------------------------------------------------------
# FINDING: the Excel loader swallows errors silently
# ---------------------------------------------------------------------------

def test_stopwords_loader_swallows_errors_and_returns_empty_dict(capsys):
    """
    FINDING (testability/robustness): settings.cargar_stopwords_desde_excel
    catches every exception, prints the error, and returns {}. If the
    stopwords Excel were missing or corrupted, the pipeline would keep
    running with ZERO survey-specific stopwords and no exception --
    only a console print would hint at the problem. This test documents
    that behavior as-is. (The function name, docstring and comments are
    also in Spanish, breaking the code-in-English team rule -- reported
    in HALLAZGOS_Y_TESTING.md as a language-consistency finding.)
    """
    from settings import cargar_stopwords_desde_excel

    result = cargar_stopwords_desde_excel(
        "ruta/que/no/existe.xlsx", sheet_name="particulares"
    )

    assert result == {}
    assert "Error cargando stopwords" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# NEW FINDING (minor): multi-word entries in the global stopwords sheet
# are dead configuration, mirroring the multi-word colloquialisms
# finding in test_synonyms.py
# ---------------------------------------------------------------------------

def test_multi_word_global_stopwords_never_match_as_phrases(real_csv):
    """
    FINDING (minor): _clean_stopwords tokenizes word by word
    (nltk.word_tokenize + per-token membership check), so any
    multi-word entry in the 'globales' sheet (e.g. "sin embargo",
    "no tengo") can never match as a whole phrase -- only its
    individual words can, and only when those words are ALSO covered
    by another source (the standard NLTK Spanish corpus, or a
    separate single-word entry in the sheet). In the real
    stopwords.xlsx, this happens to produce the expected end result
    for the two multi-word entries present today ("sin embargo",
    "no tengo") purely by coincidence -- every one of their individual
    words is independently a stopword through another path. The
    multi-word entry itself never participates in the filtering.
    Skipped if no multi-word entry exists.
    """
    multi_word_entries = [
        word
        for words in stopwords_global.values()
        for word in words
        if " " in word
    ]
    if not multi_word_entries:
        pytest.skip("No multi-word global stopwords defined in stopwords.xlsx")

    instance = Cleaner(file_path=real_csv, survey_name="ninguna",
                       key_column="comentario")

    for phrase in multi_word_entries:
        # The phrase is stored as a literal multi-word string inside
        # the stop_words set, but nltk.word_tokenize never produces a
        # single token equal to a full phrase -- so this entry can
        # never be matched by the per-token check in _clean_stopwords.
        assert phrase in instance.stop_words          # present as configured...
        assert len(phrase.split()) > 1                # ...but it's a phrase...
        result = instance._clean_stopwords(f"esto es una prueba {phrase} de verdad")
        # Whether the phrase's individual words end up removed or not
        # depends entirely on OTHER stopword sources, never on this
        # multi-word entry matching as a unit:
        for token in phrase.split():
            covered_elsewhere = token in (instance.stop_words - {phrase})
            if not covered_elsewhere:
                assert token in result.split()


# ---------------------------------------------------------------------------
# eliminate_stopwords: previous Finding 2 is RESOLVED
# ---------------------------------------------------------------------------

def test_eliminate_stopwords_works_as_the_first_method_called(real_csv):
    """
    RESOLVED (previous Finding 2): the self-healing block is now
    active. Calling eliminate_stopwords() on a fresh instance, with no
    prior clean_key_column() call, chains the whole pipeline itself.
    """
    instance = Cleaner(file_path=real_csv, survey_name=TEST_SURVEY,
                       key_column="comentario")
    result = instance.eliminate_stopwords()

    assert "comentario_no_stopwords" in result.columns
    tokens = result["comentario_no_stopwords"].iloc[0].split()
    assert "el" not in tokens         # NLTK stopword removed
    assert "profesor" in tokens       # content word kept


def test_eliminate_stopwords_also_works_after_clean_key_column(real_csv):
    instance = Cleaner(file_path=real_csv, survey_name=TEST_SURVEY,
                       key_column="comentario")
    instance.clean_key_column()
    result = instance.eliminate_stopwords()

    assert "comentario_no_stopwords" in result.columns
