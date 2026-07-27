# test_clean_text.py
"""
Tests for Cleaner._clean_text -- the text-cleaning function.

UPDATED for the new version of limpieza.py:
- The class was renamed from `cleaner` to `Cleaner`.
- The accent-removal dictionary was extended with umlaut/diaeresis
  vowels (ä, ë, ï, ö, ü and their uppercase forms), in addition to the
  Spanish acute accents that existed before.

_clean_text is still a @staticmethod, so it can be called directly on the
class without instantiating `Cleaner(...)` or needing a real file.

Accent cases are parametrized directly from settings.accents (the
project's single source of truth for this mapping) instead of a
hardcoded list. This has a side benefit: if settings.py and the
hardcoded dictionary inside limpieza.py's _clean_text ever drift apart
(see the code-duplication finding in HALLAZGOS_Y_TESTING.md), this test
will catch the mismatch automatically, since it always checks against
whatever settings.py currently declares.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from limpieza import Cleaner
from settings import accents


# ---------------------------------------------------------------------------
# Rule 1: strips leading and trailing whitespace
# ---------------------------------------------------------------------------

def test_strips_leading_and_trailing_whitespace():
    result = Cleaner._clean_text("   good service   ")
    assert result == "good service"


def test_does_not_strip_internal_whitespace():
    result = Cleaner._clean_text("  good   service  ")
    assert result == "good   service"


# ---------------------------------------------------------------------------
# Rule 2: converts to lowercase
# ---------------------------------------------------------------------------

def test_converts_uppercase_to_lowercase():
    result = Cleaner._clean_text("EXCELLENT Service")
    assert result == "excellent service"


# ---------------------------------------------------------------------------
# Rule 3: normalizes accented vowels -- parametrized from settings.accents,
# covering both acute accents (café, rápido) and umlauts (pingüino, Müller)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("accented_char,plain_char", list(accents.items()))
def test_each_accent_in_settings_is_normalized(accented_char, plain_char):
    """
    For every single accented character declared in settings.accents,
    confirm that _clean_text normalizes it correctly when surrounded by
    plain text. This directly exercises the mapping that is duplicated
    between settings.py and limpieza.py's internal dictionary.
    """
    input_text = f"test{accented_char}word"
    expected = f"test{plain_char.lower()}word"
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
    ("Müller", "muller"),
    ("Übermensch", "ubermensch"),
])
def test_removes_umlauts_in_real_words(input_text, expected):
    """
    NEW in this version: umlaut/diaeresis vowels (ä, ë, ï, ö, ü) were
    added to the accent-removal mapping. These were not present in the
    previous version of limpieza.py.
    """
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
    ("¿que tal?", "que tal"),
    ("excelente...", "excelente"),
    ("100% satisfecho", "100 satisfecho"),
    ("bueno, pero caro", "bueno pero caro"),
    ("precio/calidad", "preciocalidad"),
])
def test_removes_punctuation(input_text, expected):
    assert Cleaner._clean_text(input_text) == expected


def test_preserves_numbers():
    result = Cleaner._clean_text("calificacion 8 de 10")
    assert result == "calificacion 8 de 10"


def test_strips_html_pasted_by_copy_paste_mistake():
    result = Cleaner._clean_text("<p>buena atencion</p>")
    assert result == "pbuena atencionp"


# ---------------------------------------------------------------------------
# Rule 6 (unhappy path): values that are NOT text -- unchanged from
# the previous version
# ---------------------------------------------------------------------------

def test_none_value_returns_empty_string():
    assert Cleaner._clean_text(None) == ''


def test_numeric_value_returns_empty_string():
    assert Cleaner._clean_text(5) == ''


def test_nan_value_returns_empty_string():
    nan = float('nan')
    assert Cleaner._clean_text(nan) == ''


def test_list_value_returns_empty_string():
    assert Cleaner._clean_text(['not', 'text']) == ''


# ---------------------------------------------------------------------------
# Rule 7: empty string and whitespace-only string
# ---------------------------------------------------------------------------

def test_empty_string_stays_empty():
    assert Cleaner._clean_text("") == ''


def test_whitespace_only_string_becomes_empty():
    assert Cleaner._clean_text("     ") == ''


def test_not_applicable_response_is_not_auto_normalized():
    result = Cleaner._clean_text("N/A")
    assert result == "na"
