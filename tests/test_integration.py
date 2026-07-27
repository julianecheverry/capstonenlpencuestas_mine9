# test_integration.py
"""
Integration tests for the `Cleaner` class.

UPDATED for the new version of limpieza.py -- MAJOR CHANGE:

In the previous version, the integration test suite's main finding was
that Finding 4 (the dtype check) blocked EVERY real CSV/Excel file at
the very first step of the workflow, in any environment running
pandas >= 3.0. No mock-free integration test of the full chain was
possible.

In this version, Findings 3 and 4 are both resolved, and the full real
workflow -- load_data -> clean_key_column -> save_cleaned_data ->
eliminate_stopwords -- now runs successfully end to end, with a real
file on disk and NO mocking anywhere in this file. This is a complete
reversal of the previous integration-level finding.

The remaining integration-level finding is the call-order dependency
(Finding 2, in its new form, plus the new save_cleaned_data regression):
the real chain only works if methods are called in the correct order
(clean_key_column before save_cleaned_data or eliminate_stopwords).
This file confirms both the full happy path and what happens when that
order is violated, using only real method calls -- no mocks.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from limpieza import Cleaner


@pytest.fixture
def real_survey_csv(tmp_path):
    content = pd.DataFrame({
        "respondent_id": [1, 2, 3, 4],
        "comment": [
            "Excelente atención, muy rápido!",
            "  regular, nada especial  ",
            "PÉSIMO. no vuelvo.",
            "Buena relación precio/calidad",
        ],
    })
    path = tmp_path / "real_survey.csv"
    content.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# REVERSED FINDING: the full real workflow now works end to end, with
# no mocks at all -- this was impossible in the previous version due
# to Finding 4.
# ---------------------------------------------------------------------------

def test_full_workflow_load_clean_save_and_remove_stopwords(real_survey_csv, tmp_path):
    """
    Exercises the entire real chain -- load_data, clean_key_column,
    save_cleaned_data, eliminate_stopwords -- with a real file on disk
    and zero mocking. This is the test that was NOT possible in the
    previous version of limpieza.py, where Finding 4 blocked load_data
    for every real file in this environment.
    """
    output_path = tmp_path / "cleaned_output.csv"
    instance = Cleaner(file_path=real_survey_csv, key_column="comment")

    # Step 1: load -- real file, real pandas read, no mocks
    loaded = instance.load_data()
    assert len(loaded) == 4
    assert "comment" in loaded.columns

    # Step 2: clean -- real call
    cleaned = instance.clean_key_column()
    assert "comment_clean" in cleaned.columns
    assert cleaned["comment_clean"].iloc[0] == "excelente atencion muy rapido"
    assert cleaned["comment_clean"].iloc[1] == "regular nada especial"

    # Step 3: save -- real file write
    instance.save_cleaned_data(str(output_path))
    assert output_path.exists()

    saved = pd.read_csv(output_path)
    assert "comment_clean" in saved.columns

    # Step 4: remove stopwords -- real call, real nltk corpus
    final = instance.eliminate_stopwords()
    assert "comment_no_stopwords" in final.columns
    assert "no" not in final["comment_no_stopwords"].iloc[2].split()  # "PESIMO. no vuelvo." -> "no" is a stopword


def test_full_workflow_stops_cleanly_on_missing_key_column(tmp_path):
    content = pd.DataFrame({"respondent_id": [1, 2], "feedback": ["good", "bad"]})
    path = tmp_path / "wrong_column.csv"
    content.to_csv(path, index=False)

    instance = Cleaner(file_path=path, key_column="comment")

    with pytest.raises(ValueError, match="not found"):
        instance.load_data()
    with pytest.raises(ValueError, match="not found"):
        instance.clean_key_column()


def test_full_workflow_stops_cleanly_on_nonexistent_file(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    instance = Cleaner(file_path=missing_path, key_column="comment")

    with pytest.raises(FileNotFoundError):
        instance.load_data()
    with pytest.raises(FileNotFoundError):
        instance.clean_key_column()


# ---------------------------------------------------------------------------
# Confirms the call-order dependency (Finding 2's new form, plus the
# new save_cleaned_data regression) in a real, unmocked scenario.
# ---------------------------------------------------------------------------

def test_calling_methods_out_of_order_fails_even_with_real_data(real_survey_csv, tmp_path):
    """
    Confirms, with a real file and no mocks, that calling
    save_cleaned_data() or eliminate_stopwords() BEFORE clean_key_column()
    fails with AttributeError -- the call-order dependency identified
    in test_save_cleaned_data.py and test_stopwords.py also holds in
    this fully real, integration-level scenario.
    """
    output_path = tmp_path / "should_not_be_created.csv"
    instance = Cleaner(file_path=real_survey_csv, key_column="comment")

    with pytest.raises(AttributeError, match="cleaned_data"):
        instance.save_cleaned_data(str(output_path))
    assert not output_path.exists()

    with pytest.raises(AttributeError, match="cleaned_data"):
        instance.eliminate_stopwords()


def test_correct_call_order_recovers_from_the_above(real_survey_csv, tmp_path):
    """
    Confirms that simply calling clean_key_column() first (even after
    the failed attempts above on the same instance) unblocks the rest
    of the chain -- the failures are about call order, not about a
    permanently broken instance.
    """
    output_path = tmp_path / "output.csv"
    instance = Cleaner(file_path=real_survey_csv, key_column="comment")

    with pytest.raises(AttributeError):
        instance.eliminate_stopwords()

    instance.clean_key_column()
    instance.save_cleaned_data(str(output_path))
    result = instance.eliminate_stopwords()

    assert output_path.exists()
    assert "comment_no_stopwords" in result.columns
