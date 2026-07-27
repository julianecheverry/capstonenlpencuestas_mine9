# test_save_cleaned_data.py
"""
Tests for Cleaner.save_cleaned_data().

UPDATED for the new version of limpieza.py -- IMPORTANT BEHAVIOR CHANGE:

In the previous version, save_cleaned_data() called
self.clean_key_column() internally (`cleaned_data = self.clean_key_column()`),
so it worked correctly even when called as the very first method on a
fresh instance.

In this version, that line is commented out:
    # cleaned_data = self.clean_key_column()

This means save_cleaned_data() now ALSO depends on self.cleaned_data
having been set beforehand by a prior call to clean_key_column() (or
indirectly via eliminate_stopwords, which also requires it). Calling
save_cleaned_data() on a fresh instance, with no prior calls, now
raises AttributeError -- a regression compared to the previous version,
and the same category of problem as Finding 2 (eliminate_stopwords'
dependency on self.cleaned_data).

This file documents BOTH the correct-order happy path and this new
call-order dependency as a confirmed, non-xfail finding (it's not
something we expect to be fixed by a specific known patch the way
Findings 3/4 were -- it's a new regression to flag).
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
        "id": [1, 2],
        "comment": ["Very GOOD!", "  average  "],
    })
    path = tmp_path / "survey.csv"
    content.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Happy path -- requires clean_key_column() to have been called first
# ---------------------------------------------------------------------------

def test_saves_the_csv_file_with_the_clean_column(real_csv, tmp_path):
    output_path = tmp_path / "output.csv"
    instance = Cleaner(file_path=real_csv, key_column="comment")

    instance.clean_key_column()  # required first, in this version
    instance.save_cleaned_data(str(output_path))

    assert output_path.exists()
    content = pd.read_csv(output_path)
    assert "comment_clean" in content.columns


def test_fails_with_value_error_if_the_output_path_does_not_exist(real_csv):
    instance = Cleaner(file_path=real_csv, key_column="comment")
    instance.clean_key_column()

    invalid_path = "/path/that/does/not/exist/output.csv"
    with pytest.raises(ValueError, match="An error occurred while saving"):
        instance.save_cleaned_data(invalid_path)


# ---------------------------------------------------------------------------
# NEW FINDING (regression): save_cleaned_data no longer self-sufficient.
# In the previous version, this exact test (calling save_cleaned_data
# directly on a fresh instance) PASSED, because save_cleaned_data called
# clean_key_column() internally. It no longer does.
# ---------------------------------------------------------------------------

def test_calling_save_cleaned_data_without_prior_clean_key_column_now_fails(real_csv, tmp_path):
    """
    REGRESSION FOUND: in the previous version of limpieza.py,
    save_cleaned_data() called self.clean_key_column() internally, so
    it worked correctly even as the first call on a fresh instance.
    That internal call is now commented out in the source code:

        # cleaned_data = self.clean_key_column()

    As a result, calling save_cleaned_data() directly -- without having
    called clean_key_column() (or eliminate_stopwords, which itself
    requires clean_key_column to have run) first -- now raises
    AttributeError, because self.cleaned_data was never set.

    This is not marked xfail because it is not a bug being tracked for
    a specific fix the way Findings 3/4 were; it's a new, confirmed
    finding to flag for the team.
    """
    output_path = tmp_path / "output.csv"
    instance = Cleaner(file_path=real_csv, key_column="comment")

    with pytest.raises(AttributeError, match="cleaned_data"):
        instance.save_cleaned_data(str(output_path))

    assert not output_path.exists()
