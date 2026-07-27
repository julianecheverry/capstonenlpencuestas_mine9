# test_stopwords.py
"""
Tests for Cleaner._clean_stopwords and Cleaner.eliminate_stopwords.

UPDATED for the new version of limpieza.py:

- Finding 1 is RESOLVED: _clean_stopwords is no longer a @staticmethod,
  it correctly receives self, and `nltk` is now imported at module
  level. The function works correctly when called directly. The
  corresponding xfail test was removed.

- Finding 2 PERSISTS, in a new form: self.cleaned_data is still never
  initialized in __init__. It is now set as a side effect of
  clean_key_column() (previously it was only set in save_cleaned_data).
  Calling eliminate_stopwords() without having called clean_key_column()
  first still raises AttributeError. There is commented-out code in
  eliminate_stopwords() that looks like an unfinished attempt to fix
  this (a self-healing check that would call clean_key_column() if
  needed), but it is currently disabled.

- NEW environment requirement found while confirming these tests:
  __init__ calls nltk.download('punkt', quiet=True), but the NLTK
  tokenizer used by _clean_stopwords (nltk.word_tokenize) requires the
  'punkt_tab' resource specifically in the NLTK version used here. The
  'punkt' resource alone is not enough; without 'punkt_tab' downloaded,
  _clean_stopwords raises LookupError instead of running correctly.
  This is documented in README.md's setup instructions.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from limpieza import Cleaner


@pytest.fixture
def real_csv(tmp_path):
    content = pd.DataFrame({
        "comment": ["El gato come pescado todos los dias"],
    })
    path = tmp_path / "survey.csv"
    content.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Finding 1 RESOLVED: _clean_stopwords now works correctly when called
# directly, as long as the punkt_tab resource is downloaded (see module
# docstring above and README.md).
# ---------------------------------------------------------------------------

def test_clean_stopwords_removes_spanish_stopwords():
    instance = Cleaner(file_path="not_relevant.csv", key_column="comment")
    result = instance._clean_stopwords("el gato come pescado")
    assert result == "gato come pescado"


def test_clean_stopwords_removes_extended_academic_terms():
    """
    __init__ extends the standard Spanish stopword list with academic
    terms: 'académico', 'academia', 'universidad'. Confirms those are
    also filtered out, not just the standard NLTK list.
    """
    instance = Cleaner(file_path="not_relevant.csv", key_column="comment")
    result = instance._clean_stopwords("la universidad academica es buena")
    assert "universidad" not in result.split()
    assert "academica" in result  # only the exact extended terms are filtered


# ---------------------------------------------------------------------------
# Finding 2 PERSISTS (new form): eliminate_stopwords still depends on
# self.cleaned_data, which is no longer initialized in __init__ and is
# only set as a side effect of clean_key_column().
# ---------------------------------------------------------------------------

def test_eliminate_stopwords_fails_without_prior_clean_key_column():
    """
    FINDING 2 (persists, new origin point): self.cleaned_data is set
    inside clean_key_column() in this version (it used to be set only
    inside save_cleaned_data() in the previous version). Calling
    eliminate_stopwords() without having called clean_key_column()
    first still raises AttributeError, confirming the call-order
    dependency was not removed, only relocated.
    """
    instance = Cleaner(file_path="not_relevant.csv", key_column="comment")

    with pytest.raises(AttributeError, match="cleaned_data"):
        instance.eliminate_stopwords()


def test_eliminate_stopwords_works_after_clean_key_column(real_csv):
    """
    Happy path: following the correct (still undocumented) call order
    -- clean_key_column() before eliminate_stopwords() -- works
    correctly end to end.
    """
    instance = Cleaner(file_path=real_csv, key_column="comment")
    instance.clean_key_column()
    result = instance.eliminate_stopwords()

    assert "comment_no_stopwords" in result.columns
    assert "gato" in result["comment_no_stopwords"].iloc[0]
    assert "el" not in result["comment_no_stopwords"].iloc[0].split()


def test_eliminate_stopwords_unfinished_self_healing_code_is_disabled():
    """
    NOTE: eliminate_stopwords() contains commented-out code that looks
    like an attempt to make self.cleaned_data self-heal:

        # if not hasattr(self, 'cleaned_data') or self.cleaned_data is None:
        #      self.cleaned_data = self.clean_key_column()

    This test simply confirms that, as currently written (commented
    out), this fix is NOT active -- the AttributeError above still
    happens. If a teammate uncomments and finishes this code, this
    test (and test_eliminate_stopwords_fails_without_prior_clean_key_column
    above) should be revisited, since the call-order dependency would
    then be resolved.
    """
    instance = Cleaner(file_path="not_relevant.csv", key_column="comment")
    with pytest.raises(AttributeError):
        instance.eliminate_stopwords()
