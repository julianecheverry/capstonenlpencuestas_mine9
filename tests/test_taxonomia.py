"""Pruebas del validador de la taxonomía (utilidad del NLP, no de la Etapa 1)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nlp_utilidades.taxonomia import comparar_con_clases, leer_taxonomia, version_taxonomia  # noqa: E402

PREGUNTAS = {"obs19", "obs20", "obs24"}


def _escribir(tmp_path: Path, filas: list[tuple[str, str, str]]) -> Path:
    ruta = tmp_path / "taxonomia.xlsx"
    pd.DataFrame([{"pregunta_codigo": p, "aspecto_id": a, "aspecto_nombre": n, "descripcion": ""}
                  for p, a, n in filas],
                 columns=["pregunta_codigo", "aspecto_id", "aspecto_nombre", "descripcion"]
                 ).to_excel(ruta, sheet_name="aspectos", index=False)
    return ruta


def test_taxonomia_valida(tmp_path):
    ruta = _escribir(tmp_path, [("obs19", "CLARIDAD", "Claridad"), ("obs20", "CLARIDAD", "Claridad"),
                                ("obs24", "ESTRATEGIAS_DIDACTICAS", "Estrategias didácticas")])
    tax = leer_taxonomia(ruta, PREGUNTAS)
    assert len(tax) == 3 and len(version_taxonomia(ruta)) == 10


@pytest.mark.parametrize("filas, mensaje", [
    ([("obs99", "CLARIDAD", "Claridad")], "no están en la hoja"),
    ([("obs19", "Claridad", "Claridad")], "formato inválido"),
    ([("obs19", "SIN_ASPECTO", "Sin aspecto")], "reservados"),
    ([("obs19", "CLARIDAD", "Claridad"), ("obs19", "CLARIDAD", "Claridad")], "repetido"),
    ([("obs19", "CLARIDAD", "Claridad"), ("obs19", "CLARIDAD_2", "claridad")], "mismo nombre"),
    ([("obs19", "TECNOLOGIA", "Tecnología"), ("obs24", "TECNOLOGIA", "Tecnología docente")],
     "nombre o descripción distintos"),
])
def test_taxonomia_invalida(tmp_path, filas, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        leer_taxonomia(_escribir(tmp_path, filas), PREGUNTAS)


def test_comparar_con_clases_del_modelo(tmp_path):
    tax = leer_taxonomia(_escribir(tmp_path, [("obs19", "CLARIDAD", "Claridad"),
                                              ("obs19", "RELACION_ESTUDIANTES", "Relación estudiantes")]),
                         PREGUNTAS)
    assert comparar_con_clases(tax, "obs19", ["claridad", "relacion_estudiantes"]) == {
        "solo_en_taxonomia": [], "solo_en_modelo": []}
    assert comparar_con_clases(tax, "obs19", ["claridad", "puntualidad"]) == {
        "solo_en_taxonomia": ["Relación estudiantes"], "solo_en_modelo": ["puntualidad"]}
