"""Punto 3 (NLP etapa 2): los archivos exportados no traen datos de no-respuestas ni de respuestas con alerta.

Corpus sintético con las combinaciones de es_no_respuesta/tiene_alerta; se corre la parte pura de la exportación
(documentos, cláusulas, Excel unificado de la celda del cuaderno y las tablas del contrato) y se comprueba, por
conjuntos de respuesta_id, que ninguna respuesta excluida aparece. No imprime textos de clientes.
"""
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "tests"))

ft = pytest.importorskip("nlp_utilidades.modulos_nlp")
from nlp_utilidades import salidas_contrato as sc  # noqa: E402
from test_nlp_ingesta import _escribir, _filas_cuatro_combinaciones  # noqa: E402
import test_nlp_salidas as ts  # noqa: E402

PROCESABLES = {"evadoc_evaluacion|2025-1|001-1|obs19_aspectos_resaltar", "evadoc_evaluacion|2025-1|002-1|obs19_aspectos_resaltar",
               "evadoc_evaluacion|2024-2|007-1|obs20_oportunidades_mejora"}


@pytest.fixture()
def corrida(tmp_path):
    """(respuestas.csv de la Etapa 1, carpeta de modelado con los archivos que escribe el cuaderno)."""
    carpeta_e1 = _escribir(tmp_path / "etapa1", _filas_cuatro_combinaciones())
    respuestas = pd.read_csv(carpeta_e1 / "respuestas.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    encuestas = ft.cargar_encuestas_etapa1(tmp_path / "etapa1")
    recursos = ft.RecursosLexicos.cargar(RAIZ / "data" / "utilities")
    cfg = ft.Config()
    modelado = tmp_path / "nlp" / "modelado" / "enc" / "run1"
    modelado.mkdir(parents=True)
    docs_todos, clau_todas = [], []
    for k, (_, enc) in enumerate(sorted(encuestas.items())):
        d, c = ft.construir_unidades(enc, cfg, recursos, segmentador=ft.Segmentador(cfg, recursos))
        docs_todos.append(d.assign(doc_id=lambda x, k=k: x["doc_id"] + 1000 * k))
        clau_todas.append(c.assign(doc_id=lambda x, k=k: x["doc_id"] + 1000 * k))
    docs, clau = pd.concat(docs_todos, ignore_index=True), pd.concat(clau_todas, ignore_index=True)
    docs.to_csv(modelado / "documentos.csv", index=False)
    clau.to_csv(modelado / "clausulas.csv", index=False, encoding="utf-8")
    # el Excel unificado (misma función que usa el cuaderno: left, conserva documentos sin cláusulas, + banderas)
    sc.escribir_unificado(sc.construir_unificado(docs, clau, respuestas), modelado)
    pd.DataFrame({"pregunta_codigo": ["a"], "total": [1], "no_respuestas": [0], "con_alerta": [0]}).to_csv(
        modelado.parent / "auditoria_ingesta_etapa1.csv", index=False, encoding="utf-8-sig")
    return respuestas, modelado


def test_excluidas_son_no_respuestas_y_alertas(corrida):
    respuestas, _ = corrida
    assert sc.ids_respuestas_excluidas(respuestas) == {
        "evadoc_evaluacion|2025-1|003-1|obs19_aspectos_resaltar", "evadoc_evaluacion|2025-1|004-1|obs19_aspectos_resaltar",
        "evadoc_evaluacion|2025-1|005-1|obs19_aspectos_resaltar", "evadoc_evaluacion|2024-2|008-1|obs20_oportunidades_mejora"}
    assert sc.ids_respuestas_excluidas(respuestas, ["obs20_oportunidades_mejora"]) == {"evadoc_evaluacion|2024-2|008-1|obs20_oportunidades_mejora"}


def test_documentos_solo_procesables_y_ningun_archivo_trae_excluidas(corrida):
    respuestas, modelado = corrida
    docs = pd.read_csv(modelado / "documentos.csv", dtype=str, keep_default_na=False)
    assert set(docs["respuesta_id"]) == PROCESABLES
    informe = sc.verificar_archivos_sin_excluidas(modelado.parent, respuestas)
    assert (informe["excluidas"] == 0).all()
    por_archivo = informe.set_index("archivo")
    assert por_archivo.loc["run1/documentos.csv", "filas_revisadas"] == 3
    assert por_archivo.loc["run1/clausulas_documentos_unificado.xlsx", "columnas_id"] == 1
    assert por_archivo.loc["run1/clausulas.csv", "columnas_id"] == 0          # solo doc_id: no revisable por id (se declara)
    assert por_archivo.loc["auditoria_ingesta_etapa1.csv", "columnas_id"] == 0  # solo conteos


def test_unificado_con_left_join_no_reintroduce_excluidas(corrida):
    respuestas, modelado = corrida
    unif = pd.read_excel(modelado / "clausulas_documentos_unificado.xlsx", dtype=str, keep_default_na=False)
    assert set(unif["respuesta_id"]) <= PROCESABLES
    assert not (set(unif["respuesta_id"]) & sc.ids_respuestas_excluidas(respuestas))


def test_detecta_una_excluida_en_csv(corrida):
    respuestas, modelado = corrida
    ruta = modelado / "documentos.csv"
    docs = pd.read_csv(ruta, dtype=str, keep_default_na=False)
    pd.concat([docs, docs.iloc[[0]].assign(respuesta_id="evadoc_evaluacion|2025-1|004-1|obs19_aspectos_resaltar")]).to_csv(ruta, index=False)
    with pytest.raises(sc.ErrorSalidasNLP, match=r"documentos\.csv: 1 filas de respuestas excluidas"):
        sc.verificar_archivos_sin_excluidas(modelado.parent, respuestas)


def test_detecta_una_excluida_en_xlsx_de_varias_hojas(corrida):
    respuestas, modelado = corrida
    with pd.ExcelWriter(modelado / "auditoria.xlsx") as libro:
        pd.DataFrame({"x": [1]}).to_excel(libro, sheet_name="corrida", index=False)
        pd.DataFrame({"respuesta_id": ["evadoc_evaluacion|2025-1|003-1|obs19_aspectos_resaltar"]}).to_excel(libro, sheet_name="muestra", index=False)
    with pytest.raises(sc.ErrorSalidasNLP, match=r"auditoria\.xlsx/muestra: 1 filas"):
        sc.verificar_archivos_sin_excluidas(modelado.parent, respuestas)


def test_el_mensaje_solo_trae_conteos_no_textos(corrida):
    respuestas, modelado = corrida
    pd.DataFrame({"respuesta_id": ["evadoc_evaluacion|2025-1|004-1|obs19_aspectos_resaltar"], "texto": ["Sufrí acoso"]}).to_csv(
        modelado / "extra.csv", index=False)
    with pytest.raises(sc.ErrorSalidasNLP) as e:
        sc.verificar_archivos_sin_excluidas(modelado.parent, respuestas)
    assert "Sufrí" not in str(e.value) and re.search(r"extra\.csv: 1 filas", str(e.value))


def test_tablas_del_contrato_sin_excluidas_y_con_ellas_falla(tmp_path):
    s = ts.construir()
    carpeta = tmp_path / "contrato"
    sc.escribir_salidas_contrato(s, carpeta)
    r = ts.respuestas_csv()    # incluye 2 no-respuestas y 2 con alerta que no están en las tablas
    informe = sc.verificar_archivos_sin_excluidas(carpeta, r)
    assert (informe["excluidas"] == 0).all() and set(informe["archivo"]) == set(sc.ARCHIVOS.values())
    assert informe.set_index("archivo").loc["auditoria_clasificacion.csv", "columnas_id"] == 1   # clausula_id = respuesta_id|orden
    cl = pd.read_csv(carpeta / "clausulas.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    falsa = cl.iloc[[0]].assign(respuesta_id=ts.rid("al0"), clausula_id=ts.rid("al0") + "|1")
    pd.concat([cl, falsa]).to_csv(carpeta / "clausulas.csv", index=False, encoding="utf-8-sig")
    with pytest.raises(sc.ErrorSalidasNLP, match=r"clausulas\.csv: 2 filas"):   # respuesta_id y clausula_id
        sc.verificar_archivos_sin_excluidas(carpeta, r)
