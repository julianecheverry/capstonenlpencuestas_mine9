# test_clean_text.py
"""
Tests for Cleaner._clean_text -- the text-cleaning static method.

UPDATED for the v3 project layout:
- The accents mapping is no longer hardcoded anywhere: settings.py now
  loads it from data/utilities/accents.xlsx at import time, and
  limpieza.py imports it from settings (the previous code-duplication
  finding is resolved). The parametrized test below runs against
  whatever the real Excel file declares, so it automatically covers any
  accent pair the team adds or removes there in the future.
- All test input texts are in Spanish, per the team rule: code in
  English, test data in Spanish (the programs clean Spanish survey
  text, so the inputs must be representative).

_clean_text is still a @staticmethod: it can be called directly on the
class, with no instance, no file, and no survey_name required.
"""

import pytest

from limpieza import Cleaner
from settings import accents


# ---------------------------------------------------------------------------
# Rule 1: strips leading and trailing whitespace
# ---------------------------------------------------------------------------

def test_strips_leading_and_trailing_whitespace():
    result = Cleaner._clean_text("   buen servicio   ")
    assert result == "buen servicio"


def test_does_not_strip_internal_whitespace():
    result = Cleaner._clean_text("  buen   servicio  ")
    assert result == "buen   servicio"


# ---------------------------------------------------------------------------
# Rule 2: converts to lowercase
# ---------------------------------------------------------------------------

def test_converts_uppercase_to_lowercase():
    result = Cleaner._clean_text("EXCELENTE Servicio")
    assert result == "excelente servicio"


# ---------------------------------------------------------------------------
# Rule 3: normalizes accented characters -- parametrized directly from
# settings.accents, which is loaded from data/utilities/accents.xlsx.
# This keeps the test in sync with the real mapping: if the team edits
# the Excel file, the parametrized cases update automatically.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("accented_char,plain_char", list(accents.items()))
def test_each_accent_in_settings_is_normalized(accented_char, plain_char):
    input_text = f"prueba{accented_char}palabra"
    expected = f"prueba{str(plain_char).lower()}palabra"
    result = Cleaner._clean_text(input_text)
    assert result == expected


@pytest.mark.parametrize("input_text,expected", [
    ("café", "cafe"),
    ("rápido", "rapido"),
    ("atención", "atencion"),
    ("número", "numero"),
    ("público", "publico"),
])
def test_removes_acute_accents_in_real_words(input_text, expected):
    assert Cleaner._clean_text(input_text) == expected


@pytest.mark.parametrize("input_text,expected", [
    ("pingüino", "pinguino"),
    ("vergüenza", "verguenza"),
])
def test_removes_diaeresis_in_real_words(input_text, expected):
    assert Cleaner._clean_text(input_text) == expected


def test_removes_uppercase_accents_after_lowercasing():
    result = Cleaner._clean_text("CAFÉ RÁPIDO")
    assert result == "cafe rapido"


# ---------------------------------------------------------------------------
# Rule 4: preserves the letter "ñ"
# ---------------------------------------------------------------------------

def test_preserves_the_letter_ene():
    result = Cleaner._clean_text("muy buena señal, niño contento")
    assert result == "muy buena señal niño contento"


# ---------------------------------------------------------------------------
# Rule 5: removes special characters/punctuation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("input_text,expected", [
    ("¡muy bueno!", "muy bueno"),
    ("¿qué tal?", "que tal"),
    ("excelente...", "excelente"),
    ("100% satisfecho", "100 satisfecho"),
    ("bueno, pero caro", "bueno pero caro"),
    ("precio/calidad", "preciocalidad"),
])
def test_removes_punctuation(input_text, expected):
    assert Cleaner._clean_text(input_text) == expected


def test_preserves_numbers():
    result = Cleaner._clean_text("calificación 8 de 10")
    assert result == "calificacion 8 de 10"


def test_strips_html_pasted_by_copy_paste_mistake():
    # Typical online-survey artifact: respondent pastes formatted text.
    # The <, >, / characters are removed but the letters of the tags
    # remain -- this documents the behavior as-is.
    result = Cleaner._clean_text("<p>buena atención</p>")
    assert result == "pbuena atencionp"


# ---------------------------------------------------------------------------
# Rule 6 (unhappy path): values that are NOT text
# ---------------------------------------------------------------------------

def test_none_value_returns_empty_string():
    assert Cleaner._clean_text(None) == ''


def test_numeric_value_returns_empty_string():
    assert Cleaner._clean_text(5) == ''


def test_nan_value_returns_empty_string():
    nan = float('nan')
    assert Cleaner._clean_text(nan) == ''


def test_list_value_returns_empty_string():
    assert Cleaner._clean_text(['no', 'es', 'texto']) == ''


# ---------------------------------------------------------------------------
# Rule 7: empty string and whitespace-only string
# ---------------------------------------------------------------------------

def test_empty_string_stays_empty():
    assert Cleaner._clean_text("") == ''


def test_whitespace_only_string_becomes_empty():
    assert Cleaner._clean_text("     ") == ''


def test_not_applicable_response_is_not_auto_normalized():
    # Documents a NON-rule: _clean_text does not interpret "N/A" as an
    # empty answer -- it only strips the "/" and returns "na". Treating
    # "n/a"-style responses as empty is a separate, higher-level
    # decision, not something this method resolves.
    result = Cleaner._clean_text("N/A")
    assert result == "na"
