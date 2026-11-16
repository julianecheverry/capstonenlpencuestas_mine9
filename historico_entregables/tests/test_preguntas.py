"""
test_preguntas.py

Tests for `preguntas.py` against a test configuration file (not the
real production one), following the project's pytest convention.
"""
import pandas as pd
import pytest

from preguntas import list_open_questions, list_key_columns, resolve_column, normalize


@pytest.fixture
def test_config(tmp_path):
    questions = pd.DataFrame([
        {"encuesta": "enc_test", "clave_logica": "pregunta_1",
         "alias_redaccion": "¿Qué opina del servicio?", "hoja_excel": "Hoja1",
         "archivo": "test.xlsx", "activa": "Si"},
        {"encuesta": "enc_test", "clave_logica": "pregunta_1",
         "alias_redaccion": "Redacción anterior de la pregunta 1", "hoja_excel": "Hoja1",
         "archivo": "test.xlsx", "activa": "Si"},
        {"encuesta": "enc_test", "clave_logica": "pregunta_2",
         "alias_redaccion": "Comentarios adicionales", "hoja_excel": "Hoja1",
         "archivo": "test.xlsx", "activa": "No"},  # inactive on purpose
    ])
    keys = pd.DataFrame([
        {"encuesta": "enc_test", "columna_llave": "ID"},
        {"encuesta": "enc_test", "columna_llave": "Periodo"},
    ])
    path = tmp_path / "preguntas_abiertas_test.xlsx"
    with pd.ExcelWriter(path) as writer:
        questions.to_excel(writer, sheet_name="preguntas", index=False)
        keys.to_excel(writer, sheet_name="llaves", index=False)
    return path


def test_normalize_removes_accents_and_punctuation():
    assert normalize("¿Qué opina?") == "que opina"


def test_lists_only_active_questions(test_config):
    questions = list_open_questions(test_config, "enc_test")
    keys = [q["logical_key"] for q in questions]
    assert "pregunta_1" in keys
    assert "pregunta_2" not in keys  # was marked activa="No"


def test_one_question_groups_multiple_aliases(test_config):
    questions = list_open_questions(test_config, "enc_test")
    q1 = next(q for q in questions if q["logical_key"] == "pregunta_1")
    assert len(q1["aliases"]) == 2


def test_survey_without_active_questions_fails(test_config):
    with pytest.raises(ValueError):
        list_open_questions(test_config, "nonexistent_survey")


def test_list_key_columns(test_config):
    keys = list_key_columns(test_config, "enc_test")
    assert keys == ["ID", "Periodo"]


def test_list_key_columns_without_registry_fails(test_config):
    with pytest.raises(ValueError):
        list_key_columns(test_config, "nonexistent_survey")


def test_resolve_column_with_current_wording():
    df = pd.DataFrame({"¿Qué opina del servicio?": ["texto"]})
    aliases = ["¿Qué opina del servicio?", "Redacción anterior"]
    assert resolve_column(df, aliases) == "¿Qué opina del servicio?"


def test_resolve_column_with_previous_wording():
    # The real file brings the OLD wording -- resolution must still work.
    df = pd.DataFrame({"Redacción anterior de la pregunta 1": ["texto"]})
    aliases = ["¿Qué opina del servicio?", "Redacción anterior de la pregunta 1"]
    assert resolve_column(df, aliases) == "Redacción anterior de la pregunta 1"


def test_resolve_column_fails_when_no_match():
    df = pd.DataFrame({"unrelated_column": ["x"]})
    with pytest.raises(ValueError):
        resolve_column(df, ["¿Qué opina del servicio?"])
