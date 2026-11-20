"""Punto 6 (NLP etapa 2, ajustado): el Excel que consolida cláusulas y respuestas va junto a clausulas.csv y metadatos_nlp.csv.

``respuestas.csv`` ya existe (Etapa 1), así que el NLP no lo escribe. Datos sintéticos; no imprime textos de clientes.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "tests"))

from nlp_utilidades import salidas_contrato as sc  # noqa: E402
import test_nlp_salidas as ts  # noqa: E402


def internos():
    """documentos y cláusulas internas de la entrada sintética (con un documento sin cláusulas)."""
    e = ts.entrada()
    docs = pd.concat([e.documentos, pd.DataFrame({"doc_id": [100], "respuesta_id": [ts.rid("v01")], "respuesta_completa": ["..."]})], ignore_index=True)
    docs["estado"] = ["CON_OPINION"] * (len(docs) - 1) + ["VACIA"]
    return docs, e.clausulas


def respuestas():
    r = ts.respuestas_csv()
    return pd.concat([r, pd.DataFrame([{"respuesta_id": ts.rid("v01"), "pregunta_codigo": ts.PREG, "es_no_respuesta": "False", "tiene_alerta": "False"}])],
                     ignore_index=True)


def test_union_conserva_documentos_sin_clausulas_y_agrega_las_banderas():
    docs, cl = internos()
    u = sc.construir_unificado(docs, cl, respuestas())
    assert len(u) == len(cl) + 1                                              # una fila por cláusula + el documento sin cláusulas
    assert list(u.columns[-2:]) == ["es_no_respuesta", "tiene_alerta"]
    assert set(u["es_no_respuesta"]) == {"False"} and set(u["tiene_alerta"]) == {"False"}
    sin = u[u["respuesta_id"] == ts.rid("v01")]
    assert len(sin) == 1 and sin["clausula"].isna().all()                     # columnas de cláusula vacías
    assert set(docs.columns) <= set(u.columns) and "p_u4_claridad" in u.columns and "asignacion" in u.columns   # mismas columnas de hoy


def test_falla_si_llega_una_no_respuesta_o_una_alerta():
    docs, cl = internos()
    for col, mensaje in (("es_no_respuesta", "no-respuestas o respuestas con alerta"), ("tiene_alerta", "no-respuestas o respuestas con alerta")):
        r = respuestas()
        r.loc[r["respuesta_id"] == ts.rid("001"), col] = "True"
        with pytest.raises(sc.ErrorSalidasNLP, match=mensaje):
            sc.construir_unificado(docs, cl, r)


def test_falla_si_falta_el_respuesta_id_o_las_columnas():
    docs, cl = internos()
    with pytest.raises(sc.ErrorSalidasNLP, match="no está en respuestas.csv"):
        sc.construir_unificado(docs, cl, respuestas()[lambda d: d["respuesta_id"] != ts.rid("001")])
    with pytest.raises(sc.ErrorSalidasNLP, match="falta la columna tiene_alerta"):
        sc.construir_unificado(docs, cl, respuestas().drop(columns=["tiene_alerta"]))
    with pytest.raises(sc.ErrorSalidasNLP, match="documentos: falta la columna respuesta_id"):
        sc.construir_unificado(docs.drop(columns=["respuesta_id"]), cl, respuestas())


def test_caracteres_de_control_se_eliminan_para_excel(tmp_path):
    docs, cl = internos()
    cl = cl.copy()
    cl.loc[0, "clausula"] = "texto" + chr(0) + "con" + chr(11) + "control\ty salto\n"
    u = sc.construir_unificado(docs, cl, respuestas())
    assert u["clausula"].iloc[0] == "textoconcontrol\ty salto\n"
    sc.escribir_unificado(u, tmp_path)                                         # no lanza IllegalCharacterError


def test_se_escribe_junto_a_clausulas_y_metadatos(tmp_path):
    s = ts.construir()
    carpeta = tmp_path / s.id_corrida_nlp
    sc.escribir_salidas_contrato(s, carpeta)
    docs, cl = internos()
    ruta = sc.escribir_unificado(sc.construir_unificado(docs, cl, respuestas()), carpeta)
    assert ruta == carpeta / "clausulas_documentos_unificado.xlsx" and ruta.parent == (carpeta / "clausulas.csv").parent == (carpeta / "metadatos_nlp.csv").parent
    leido = pd.read_excel(ruta, sheet_name="unificado", dtype=str, keep_default_na=False)
    assert {"doc_id", "respuesta_id", "clausula", "es_no_respuesta", "tiene_alerta"} <= set(leido.columns)
    # el Excel no es una de las tablas del contrato ni entra en el hash (no es determinista byte a byte)
    assert sc.ARCHIVO_UNIFICADO not in sc.ARCHIVOS.values()
    antes = sc.huella_archivos(carpeta)
    sc.escribir_unificado(sc.construir_unificado(docs, cl, respuestas()), carpeta)
    assert sc.huella_archivos(carpeta) == antes
    # la verificación del Punto 3 sobre la carpeta de la corrida cubre también el Excel (por respuesta_id)
    informe = sc.verificar_archivos_sin_excluidas(carpeta, ts.respuestas_csv())
    fila = informe[informe["archivo"] == sc.ARCHIVO_UNIFICADO].iloc[0]
    assert fila["columnas_id"] == 1 and fila["excluidas"] == 0


def test_la_verificacion_detecta_una_excluida_en_el_excel(tmp_path):
    docs, cl = internos()
    u = sc.construir_unificado(docs, cl, respuestas())
    u.loc[0, "respuesta_id"] = ts.rid("al0")                                   # una respuesta con alerta que se coló
    sc.escribir_unificado(u, tmp_path)
    with pytest.raises(sc.ErrorSalidasNLP, match=r"clausulas_documentos_unificado\.xlsx/unificado: 1 filas"):
        sc.verificar_archivos_sin_excluidas(tmp_path, ts.respuestas_csv())


def test_compatible_con_el_lector_por_nombre_de_la_etapa_3(tmp_path):
    """El adaptador de la Etapa 3 lee columnas por nombre (usecols): agregar columnas no lo rompe."""
    docs, cl = internos()
    ruta = sc.escribir_unificado(sc.construir_unificado(docs, cl, respuestas()), tmp_path)
    leido = pd.read_excel(ruta, sheet_name="unificado", usecols=["doc_id", "estado", "clausula_id", "clausula"], dtype=str)
    assert list(leido.columns) == ["doc_id", "estado", "clausula_id", "clausula"] and len(leido) == len(cl) + 1
