"""Punto 5 (NLP etapa 2, ajustado): clausulas.csv conserva las cláusulas con y sin opinión.

El filtro es_opinion == True se descartó. Lo que sí cambia: las respuestas procesables que quedan vacías o de ruido no
generan cláusulas y el validador las declara (``sin_clausulas``). Datos sintéticos; no imprime textos de clientes.
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "tests"))

from nlp_utilidades import salidas_contrato as sc  # noqa: E402
import test_nlp_salidas as ts  # noqa: E402

PREPARACION = "20261117-211304_3d69efc5d6"


def entrada_con_vacias():
    """Entrada con las 8 respuestas de siempre más una VACIA y una RUIDO (sin cláusulas) y estados en documentos."""
    e = ts.entrada()
    por_doc = e.clausulas.groupby("doc_id")["asignacion"].agg(lambda s: "NO_OPINION" if set(s) <= {"no_opinion", "ruido"} else "CON_OPINION")
    e.documentos = e.documentos.assign(estado=e.documentos["doc_id"].map(por_doc).fillna("VACIA"))
    extra = pd.DataFrame({"doc_id": [100, 101], "respuesta_id": [ts.rid("v01"), ts.rid("r01")],
                          "respuesta_completa": ["...", "-"], "estado": ["VACIA", "RUIDO"]})
    e.documentos = pd.concat([e.documentos, extra], ignore_index=True)
    return e


def respuestas_completas():
    r = ts.respuestas_csv()
    nuevas = pd.DataFrame([{"respuesta_id": ts.rid(o), "pregunta_codigo": ts.PREG, "es_no_respuesta": "False", "tiene_alerta": "False"}
                           for o in ("v01", "r01")])
    return pd.concat([r, nuevas], ignore_index=True)


def construir(e=None):
    return sc.construir_salidas_contrato([e or entrada_con_vacias()], taxonomia=ts.TAXONOMIA, id_corrida_preparacion=PREPARACION,
                                         version_taxonomia="9517d60bf5", fecha=ts.FECHA, pos_fn=ts.pos_falso)


def releer(s):
    d = Path(tempfile.mkdtemp())
    sc.escribir_salidas_contrato(s, d)
    return sc.leer_salidas_contrato(d)


def validar(tablas, sin_clausulas=(), respuestas=None):
    sc.validar_salidas_nlp(tablas, respuestas_completas() if respuestas is None else respuestas, id_corrida_preparacion=PREPARACION,
                           preguntas_procesadas=[ts.PREG], clases_por_pregunta={ts.PREG: ts.CLASES}, taxonomia=ts.TAXONOMIA,
                           sin_clausulas=sin_clausulas)


def test_clausulas_conserva_con_y_sin_opinion():
    s = construir()
    c = s.clausulas
    assert set(c["es_opinion"]) == {"True", "False"}
    sin = c[c["es_opinion"] == "False"]
    assert len(sin) >= 1 and set(sin["aspecto_id"]) == {"SIN_ASPECTO"} and set(sin["aspecto_origen"]) == {"hdbscan"}   # regla del Punto 4
    assert set(sin["sentimiento"]) == {""} and sin["confianza_sentimiento"].isna().all()
    assert len(c) == len(ts.entrada().clausulas)                                                                        # nada se descarta


def test_orden_sin_saltos_y_auditoria_uno_a_uno():
    t = releer(construir())
    g = t["clausulas"].assign(o=t["clausulas"]["orden"].astype(int)).groupby("respuesta_id")["o"]
    assert (g.min() == 1).all() and (g.max() == g.count()).all()
    assert set(t["auditoria"]["clausula_id"]) == set(t["clausulas"]["clausula_id"]) and len(t["auditoria"]) == len(t["clausulas"])


def test_vacias_y_ruido_se_declaran_y_no_tienen_clausulas():
    s = construir()
    assert s.sin_clausulas == frozenset({ts.rid("v01"), ts.rid("r01")})
    assert not (set(s.clausulas["respuesta_id"]) & s.sin_clausulas)
    assert s.metadatos.iloc[0]["n_respuestas"] == 10 and s.metadatos.iloc[0]["n_clausulas"] == len(s.clausulas)          # n_respuestas = procesadas
    validar(releer(s), s.sin_clausulas)


def test_sin_estado_en_documentos_no_declara_nada():
    s = sc.construir_salidas_contrato([ts.entrada()], taxonomia=ts.TAXONOMIA, id_corrida_preparacion=PREPARACION,
                                      version_taxonomia="x", fecha=ts.FECHA, pos_fn=ts.pos_falso)
    assert s.sin_clausulas == frozenset()


def test_falla_si_no_se_declaran_las_vacias():
    s = construir()
    with pytest.raises(sc.ErrorSalidasNLP, match="2 respuestas procesables sin ninguna cláusula"):
        validar(releer(s))


def test_falla_si_se_declara_una_respuesta_que_si_tiene_clausulas():
    s = construir()
    with pytest.raises(sc.ErrorSalidasNLP, match="1 respuestas declaradas sin cláusulas.*sí tienen cláusulas"):
        validar(releer(s), set(s.sin_clausulas) | {ts.rid("001")})


def test_falla_si_se_declara_una_no_respuesta():
    s = construir()
    with pytest.raises(sc.ErrorSalidasNLP, match="1 respuestas declaradas sin cláusulas no son procesables"):
        validar(releer(s), set(s.sin_clausulas) | {ts.rid("nr0")})


def test_declarar_no_oculta_una_respuesta_perdida():
    """Si falta una respuesta con cláusulas (no declarada), la cobertura sigue fallando."""
    s = construir()
    t = releer(s)
    t["clausulas"] = t["clausulas"][t["clausulas"]["respuesta_id"] != ts.rid("007")]
    t["auditoria"] = t["auditoria"][t["auditoria"]["clausula_id"].isin(t["clausulas"]["clausula_id"])]
    with pytest.raises(sc.ErrorSalidasNLP, match="1 respuestas procesables sin ninguna cláusula"):
        validar(t, s.sin_clausulas)


def test_falla_si_n_respuestas_no_cuadra(tmp_path):
    t = releer(construir())
    t["metadatos"].loc[0, "n_respuestas"] = "8"   # no cuenta las vacías/ruido declaradas
    with pytest.raises(sc.ErrorSalidasNLP, match="conteos no coinciden"):
        validar(t, {ts.rid("v01"), ts.rid("r01")})


def test_determinismo(tmp_path):
    sc.escribir_salidas_contrato(construir(), tmp_path / "a")
    sc.escribir_salidas_contrato(construir(), tmp_path / "b")
    assert sc.huella_archivos(tmp_path / "a") == sc.huella_archivos(tmp_path / "b")
