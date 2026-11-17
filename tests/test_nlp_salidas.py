"""Fase 4 (NLP etapa 2): las cinco salidas del contrato v0.2 y su validación.

Mini corpus sintético con probabilidades simuladas (no necesita torch, spaCy ni datos del cliente).
"""
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from nlp_utilidades import salidas_contrato as sc  # noqa: E402

PREG = "obs19_aspectos_resaltar"
ENC = "evadoc_evaluacion"
OTRA = "obs20_oportunidades_mejora"
FECHA = datetime(2026, 11, 17, 15, 30, 0, tzinfo=timezone.utc)
CLASES = ["Claridad", "Puntualidad", "Opinión general"]
COLS_P = ["p_u4_claridad", "p_u4_puntualidad", "p_u4_opinion_general"]

TAXONOMIA = pd.DataFrame({
    "pregunta_codigo": [PREG] * 3 + [OTRA] * 2,
    "aspecto_id": ["CLARIDAD", "PUNTUALIDAD", "OPINION_GENERAL", "CLARIDAD", "OPINION_GENERAL"],
    "aspecto_nombre": ["Claridad", "Puntualidad", "Opinión general", "Claridad", "Opinión general"],
    "descripcion": ["", "", "", "", ""],
})
ALERTAS = pd.DataFrame({"termino": ["acoso", "acoso laboral", "irrespetuoso"], "categoria": ["acoso", "acoso", "discriminación"]})
NEGADORES = ["no", "nunca", "sin"]


def rid(origen, periodo="2025-1", pregunta=PREG):
    return f"{ENC}|{periodo}|{origen}|{pregunta}"


TEXTOS = {  # origen -> texto original
    "001": "Explica muy bien, pero llega tarde.",
    "002": "Sin comentarios",
    "003": "Hubo Acoso Laboral en el curso",
    "004": "Nunca ha sido irrespetuoso con nadie",
    "005": "Bien",
    "006": "Excelente profesor",
    "007": "Explica claro. Es puntual. Buen docente en general",
    "008": "Las dinámicas que propone son diferentes a todo lo visto antes",
    "009": "Cosas varias sin relación con el curso en absoluto",
    "010": "NA",
}


def _clausula(doc, n, texto, asig, aspecto=None, probs=None, sent=None, pm=(0.7, 0.2, 0.1), visto=False, topico=pd.NA, etiqueta="", p_res=np.nan):
    fila = {"doc_id": doc, "clausula_id": n, "clausula": texto, "asignacion": asig, "aspecto": aspecto or "",
            "p_aspecto_u4": max(probs) if probs else np.nan, "sentimiento": sent, "confianza": max(pm) if sent else np.nan,
            "p_pos": pm[0] if sent else np.nan, "p_neu": pm[1] if sent else np.nan, "p_neg": pm[2] if sent else np.nan,
            "u4_visto_en_entrenamiento": visto, "topico_residual": topico, "etiqueta_residual": etiqueta, "p_residual": p_res}
    for col, v in zip(COLS_P, probs or (np.nan,) * 3):
        fila[col] = v
    return fila


def entrada(pregunta=PREG, claves_alerta=("003", "004")):
    origenes = list(TEXTOS)
    docs = pd.DataFrame({"doc_id": range(len(origenes)), "respuesta_id": [rid(o, pregunta=pregunta) for o in origenes],
                         "respuesta_completa": [TEXTOS[o] for o in origenes]})
    d = {o: i for i, o in enumerate(origenes)}
    cl = [
        _clausula(d["001"], 7, "Explica muy bien,", "u4", "Claridad", (0.8, 0.1, 0.1), "POS", (0.9, 0.05, 0.05), visto=True),  # id interno 7: se renumera
        _clausula(d["001"], 9, "pero llega tarde.", "u4", "Puntualidad", (0.1, 0.7, 0.2), "NEG", (0.1, 0.2, 0.7)),
        _clausula(d["002"], 0, "Sin comentarios", "no_opinion"),
        _clausula(d["003"], 0, TEXTOS["003"], "alerta", sent="NEG", pm=(0.1, 0.1, 0.8)),  # hoy fuerza NEG: el contrato lo deja vacío
        _clausula(d["004"], 0, TEXTOS["004"], "alerta", sent="NEG"),
        _clausula(d["005"], 0, "Bien", "regla_comentario_general", "Comentario General", (0.3, 0.3, 0.4), "POS", (0.6, 0.3, 0.1)),
        _clausula(d["006"], 0, "Excelente profesor", "valoracion_global", "VALORACIÓN GLOBAL (sin aspecto)", sent="POS", pm=(0.95, 0.03, 0.02)),
        _clausula(d["007"], 0, "Explica claro.", "u4", "Claridad", (0.9, 0.05, 0.05), "POS", (0.8, 0.1, 0.1)),
        _clausula(d["007"], 1, "Es puntual.", "u4", "Puntualidad", (0.05, 0.9, 0.05), "POS", (0.7, 0.2, 0.1)),
        _clausula(d["007"], 2, "Buen docente en general", "u4", "Opinión general", (0.1, 0.1, 0.8), "POS", (0.85, 0.1, 0.05)),
        _clausula(d["008"], 0, TEXTOS["008"], "residual_hdbscan", "Por descubrir", (0.2, 0.2, 0.1), "NEU", (0.2, 0.6, 0.2), topico=0, etiqueta="dinámicas", p_res=0.77),
        _clausula(d["009"], 0, TEXTOS["009"], "residual_hdbscan", "Por descubrir", (0.1, 0.1, 0.1), "NEU", (0.3, 0.4, 0.3), topico=-1),
        _clausula(d["010"], 0, "NA", "u4", "Opinión general", (0.1, 0.1, 0.8), "NEU", (0.2, 0.6, 0.2)),
    ]
    return sc.EntradaPregunta(
        pregunta_codigo=pregunta, documentos=docs, clausulas=pd.DataFrame(cl), clases=CLASES, columnas_p=COLS_P,
        modelo_u4_id="u4_test_1", modelo_sentimiento="pysentimiento/robertuito-sentiment-analysis", accuracy_aspecto=0.72,
        metricas_sentimiento={"f1_macro_cv": 0.61},
        topicos_residuales=pd.DataFrame({"topico": [0], "etiqueta": ["dinámicas"], "terminos": ["dinámicas, propone, visto"]}),
    )


def respuestas_csv(extra_no_resp=2, preguntas=(PREG,)):
    filas = []
    for p in preguntas:
        for o in TEXTOS:
            filas.append({"respuesta_id": rid(o, pregunta=p), "pregunta_codigo": p, "es_no_respuesta": "False"})
        for k in range(extra_no_resp):
            filas.append({"respuesta_id": rid(f"nr{k}", pregunta=p), "pregunta_codigo": p, "es_no_respuesta": "True"})
    return pd.DataFrame(filas)


def construir(parametros=None, entradas=None, fecha=FECHA):
    return sc.construir_salidas_contrato(
        entradas or [entrada()], taxonomia=TAXONOMIA, alertas=ALERTAS, negadores=NEGADORES,
        id_corrida_preparacion="20261117-211304_3d69efc5d6", version_taxonomia="9517d60bf5", huella_alertas="abc123def4",
        parametros=parametros, fecha=fecha)


def validar(tablas, **kw):
    base = dict(id_corrida_preparacion="20261117-211304_3d69efc5d6", preguntas_procesadas=[PREG],
                clases_por_pregunta={PREG: CLASES}, taxonomia=TAXONOMIA)
    base.update(kw)
    sc.validar_salidas_nlp(tablas, respuestas_csv(), **base)


@pytest.fixture()
def salidas(tmp_path):
    s = construir()
    sc.escribir_salidas_contrato(s, tmp_path)
    return s, sc.leer_salidas_contrato(tmp_path), tmp_path


# ── construcción y contenido ───────────────────────────────────────────────


def test_las_cinco_tablas_pasan_la_validacion(salidas):
    s, tablas, _ = salidas
    validar(tablas)
    assert {k: list(v.columns) for k, v in tablas.items()} == sc.COLUMNAS
    assert re.fullmatch(r"20261117-153000_[0-9a-f]{10}", s.id_corrida_nlp)


def test_clausulas_ids_orden_y_reglas(salidas):
    _, t, _ = salidas
    c = t["clausulas"].set_index("clausula_id")
    assert c.loc[rid("001") + "|1", "texto_clausula"] == "Explica muy bien,"        # id interno 7 -> orden 1
    assert c.loc[rid("001") + "|2", "texto_clausula"] == "pero llega tarde."
    assert (t["clausulas"]["orden"].astype(int) >= 1).all()
    assert len(c.loc[[i for i in c.index if i.startswith(rid("007") + "|")]]) == 3
    # sin opinión y alertas: SIN_ASPECTO, sistema, sin sentimiento
    for o in ("002", "003", "004"):
        f = c.loc[rid(o) + "|1"]
        assert (f["es_opinion"], f["aspecto_id"], f["aspecto_origen"], f["sentimiento"], f["confianza_sentimiento"]) == ("False", "SIN_ASPECTO", "sistema", "", "")
    # reglas -> OPINION_GENERAL con origen regla
    for o in ("005", "006"):
        f = c.loc[rid(o) + "|1"]
        assert (f["aspecto_id"], f["aspecto_origen"], f["es_opinion"]) == ("OPINION_GENERAL", "regla", "True") and f["sentimiento"] != ""
    assert c.loc[rid("001") + "|1", "aspecto_id"] == "CLARIDAD" and c.loc[rid("001") + "|1", "aspecto_origen"] == "clasificador"
    assert float(c.loc[rid("001") + "|1", "confianza_aspecto"]) == pytest.approx(0.8)
    # HDBSCAN: id emergente con el periodo; sin grupo -> SIN_ASPECTO/sistema
    assert (c.loc[rid("008") + "|1", "aspecto_id"], c.loc[rid("008") + "|1", "aspecto_origen"]) == ("EMERGENTE_2025-1_0", "hdbscan")
    assert float(c.loc[rid("008") + "|1", "confianza_aspecto"]) == pytest.approx(0.77)
    assert (c.loc[rid("009") + "|1", "aspecto_id"], c.loc[rid("009") + "|1", "aspecto_origen"]) == ("SIN_ASPECTO", "sistema")
    assert c.loc[rid("010") + "|1", "texto_clausula"] == "NA"                       # «NA» como comentario real se conserva
    assert set(c["version_modelo"]) == {"u4_test_1|pysentimiento/robertuito-sentiment-analysis"}


def test_auditoria_top3_y_vacios(salidas):
    _, t, _ = salidas
    a = t["auditoria"].set_index("clausula_id")
    f = a.loc[rid("007") + "|3"]  # Opinión general ganadora
    assert (f["aspecto_1_id"], f["aspecto_2_id"], f["aspecto_3_id"]) == ("OPINION_GENERAL", "CLARIDAD", "PUNTUALIDAD") or f["aspecto_1_id"] == "OPINION_GENERAL"
    assert float(f["confianza_aspecto_1"]) >= float(f["confianza_aspecto_2"]) >= float(f["confianza_aspecto_3"])
    assert (a.loc[rid("001") + "|1", "visto_en_entrenamiento"], a.loc[rid("001") + "|2", "visto_en_entrenamiento"]) == ("True", "False")
    for o in ("002", "003", "004", "006"):  # no pasan por el clasificador: aspectos vacíos
        g = a.loc[rid(o) + "|1"]
        assert (g["aspecto_1_id"], g["aspecto_2_id"], g["aspecto_3_id"], g["confianza_aspecto_1"]) == ("", "", "", "")
    assert a.loc[rid("002") + "|1", "confianza_pos"] == ""                               # sin opinión: sin probabilidades de sentimiento
    assert len(a) == len(t["clausulas"])


def test_alertas_una_fila_por_termino_con_fragmento_literal(salidas):
    _, t, _ = salidas
    al = t["alertas"]
    assert set(al["respuesta_id"]) == {rid("003"), rid("004")}
    f3 = al[al["respuesta_id"] == rid("003")]
    assert list(f3["termino"]) == ["acoso laboral"] and list(f3["texto_coincidente"]) == ["Acoso Laboral"]  # «acoso» queda dentro de la frase más larga
    assert list(f3["alerta_id"]) == [rid("003") + "|1"] and list(f3["clausula_id"]) == [rid("003") + "|1"]
    assert list(f3["categoria_alerta"]) == ["acoso"]
    assert set(al["negada"]) == {"False"}                                                 # alerta_negacion apagada (valor de hoy)


def test_alerta_negacion_encendida():
    s = construir(sc.ParametrosSalidas(alerta_negacion=True))
    f = s.alertas.set_index("respuesta_id")
    assert f.loc[rid("004"), "negada"] == "True" and f.loc[rid("003"), "negada"] == "False"


def test_detectar_alertas_sin_tildes_y_varios_terminos():
    al = pd.DataFrame({"termino": ["acoso", "acoso laboral", "irrespetuoso", "discriminación"], "categoria": ["a", "a", "b", "c"]})
    r = sc.detectar_alertas("Sufrí DISCRIMINACION y fue irrespetuoso, además de acoso", al)
    assert [x["termino"] for x in r] == ["discriminación", "irrespetuoso", "acoso"]
    assert r[0]["texto_coincidente"] == "DISCRIMINACION"
    assert sc.detectar_alertas("el acosador", al) == []                                   # límites de palabra
    assert sc.detectar_alertas("texto limpio", al) == []


def test_catalogo(salidas):
    _, t, _ = salidas
    cat = t["catalogo"]
    ids = list(cat["aspecto_id"])
    assert ids == ["CLARIDAD", "PUNTUALIDAD", "OPINION_GENERAL", "EMERGENTE_2025-1_0", "SIN_ASPECTO"]
    assert list(cat["orden"]) == ["1", "2", "3", "4", "5"]
    assert list(cat["grupo_visual"]) == ["Aspectos definidos"] * 3 + ["Temas emergentes", "Aspectos definidos"]
    emg = cat[cat["aspecto_id"] == "EMERGENTE_2025-1_0"].iloc[0]
    assert (emg["aspecto_origen"], emg["aspecto_nombre"], emg["palabras_clave"]) == ("hdbscan", "dinámicas", "dinámicas, propone, visto")
    assert set(cat["encuesta"]) == {ENC}


def test_metadatos(salidas):
    s, t, _ = salidas
    m = t["metadatos"].iloc[0]
    assert (m["id_corrida_nlp"], m["id_corrida_preparacion"], m["encuesta"], m["pregunta_codigo"]) == (s.id_corrida_nlp, "20261117-211304_3d69efc5d6", ENC, PREG)
    assert (m["n_respuestas"], m["n_clausulas"], m["n_alertas"]) == ("10", str(len(t["clausulas"])), "2")
    assert (m["version_taxonomia"], m["huella_alertas"], m["accuracy_aspecto"]) == ("9517d60bf5", "abc123def4", "0.72")
    assert m["fecha_corrida"] == "2026-11-17T15:30:00Z" and "f1_macro_cv" in m["metricas_sentimiento"]
    pct = sum(float(m[k]) for k in ("pct_aspecto_clasificador", "pct_aspecto_hdbscan", "pct_aspecto_regla", "pct_sin_aspecto"))
    assert pct == pytest.approx(100.0, abs=0.05)


def test_varias_preguntas_catalogo_sin_duplicados_y_emergentes_unicos():
    """OTRA solo tiene CLARIDAD y OPINION_GENERAL en su taxonomía: sus cláusulas de Puntualidad pasan a Claridad."""
    e2 = entrada(OTRA)
    e2.clases, e2.columnas_p = ["Claridad", "Opinión general"], ["p_u4_claridad", "p_u4_opinion_general"]
    e2.clausulas = e2.clausulas.drop(columns=["p_u4_puntualidad"]).assign(
        aspecto=lambda d: d["aspecto"].replace({"Puntualidad": "Claridad"}))
    s = construir(entradas=[entrada(), e2])
    assert not s.catalogo.duplicated(["aspecto_id", "encuesta"]).any()                   # CLARIDAD y OPINION_GENERAL se comparten
    emerg = s.catalogo[s.catalogo["grupo_visual"] == "Temas emergentes"]["aspecto_id"].tolist()
    assert emerg == ["EMERGENTE_2025-1_0", "EMERGENTE_2025-1_1"]                          # únicos entre preguntas de la misma encuesta
    assert set(s.metadatos["pregunta_codigo"]) == {PREG, OTRA}


# ── formato de archivo ─────────────────────────────────────────────────────


def test_formato_de_archivos(salidas):
    _, t, ruta = salidas
    for f in sc.ARCHIVOS.values():
        crudo = (ruta / f).read_bytes()
        assert crudo.startswith(b"\xef\xbb\xbf")                                          # utf-8-sig
    assert "2025-1" in t["clausulas"]["respuesta_id"].iloc[0]                             # el periodo sigue siendo texto
    assert not t["metadatos"].astype(str).isin(["NA", "nan", "None"]).to_numpy().any()
    assert (t["clausulas"]["confianza_aspecto"].replace("", "0").map(lambda x: re.fullmatch(r"\d(\.\d+)?", x) is not None)).all()  # punto decimal
    assert set(t["clausulas"]["es_opinion"]) <= {"True", "False"}


def test_determinismo(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    sc.escribir_salidas_contrato(construir(), a)
    sc.escribir_salidas_contrato(construir(), b)
    assert sc.huella_archivos(a) == sc.huella_archivos(b)
    sc.escribir_salidas_contrato(construir(fecha=datetime(2026, 11, 18, 0, 0, 0, tzinfo=timezone.utc)), tmp_path / "c")
    assert sc.huella_archivos(a) != sc.huella_archivos(tmp_path / "c")                    # cambia el id/fecha de la corrida


# ── pruebas negativas: la validación debe FALLAR con el mensaje esperado ───


def _falla(tablas, mensaje):
    with pytest.raises(sc.ErrorSalidasNLP, match=mensaje):
        validar(tablas)


def test_falla_ids_duplicados(salidas):
    _, t, _ = salidas
    t = {k: v.copy() for k, v in t.items()}
    t["clausulas"] = pd.concat([t["clausulas"], t["clausulas"].iloc[[0]]], ignore_index=True)
    _falla(t, "clausula_id repetido")
    t = {k: v.copy() for k, v in salidas[1].items()}
    t["alertas"] = pd.concat([t["alertas"], t["alertas"].iloc[[0]]], ignore_index=True)
    _falla(t, "alerta_id repetido")


def test_falla_orden_con_salto(salidas):
    t = {k: v.copy() for k, v in salidas[1].items()}
    t["clausulas"].loc[t["clausulas"]["clausula_id"] == rid("001") + "|2", ["orden", "clausula_id"]] = ["3", rid("001") + "|3"]
    t["auditoria"].loc[t["auditoria"]["clausula_id"] == rid("001") + "|2", "clausula_id"] = rid("001") + "|3"
    _falla(t, "saltos")


def test_falla_aspecto_fuera_del_catalogo(salidas):
    t = {k: v.copy() for k, v in salidas[1].items()}
    t["clausulas"].loc[0, "aspecto_id"] = "INVENTADO"
    _falla(t, "no están en catalogo_aspectos")


def test_falla_confianzas_que_no_suman_uno(salidas):
    t = {k: v.copy() for k, v in salidas[1].items()}
    t["auditoria"].loc[t["auditoria"]["confianza_pos"] != "", "confianza_pos"] = "0.2"
    _falla(t, "distinta de 1")
    t = {k: v.copy() for k, v in salidas[1].items()}
    t["clausulas"].loc[0, "confianza_aspecto"] = "1.5"
    _falla(t, "fuera de \\[0, 1\\]")


def test_falla_cobertura_y_no_respuestas(salidas):
    _, t, _ = salidas
    t2 = {k: v.copy() for k, v in t.items()}
    t2["clausulas"] = t2["clausulas"][~t2["clausulas"]["respuesta_id"].eq(rid("010"))]
    t2["auditoria"] = t2["auditoria"][t2["auditoria"]["clausula_id"].isin(t2["clausulas"]["clausula_id"])]
    _falla(t2, "sin ninguna cláusula")
    r = respuestas_csv()
    r.loc[r["respuesta_id"] == rid("010"), "es_no_respuesta"] = "True"
    with pytest.raises(sc.ErrorSalidasNLP, match="no-respuestas tienen cláusulas"):
        sc.validar_salidas_nlp(t, r, id_corrida_preparacion="20261117-211304_3d69efc5d6", preguntas_procesadas=[PREG])


def test_falla_sentimiento_y_dominios(salidas):
    t = {k: v.copy() for k, v in salidas[1].items()}
    t["clausulas"].loc[t["clausulas"]["es_opinion"] == "False", "sentimiento"] = "NEG"
    _falla(t, "sentimiento vacío no coincide")
    t = {k: v.copy() for k, v in salidas[1].items()}
    t["clausulas"].loc[0, "aspecto_origen"] = "magia"
    _falla(t, "aspecto_origen fuera de")


def test_falla_corrida_de_preparacion_y_taxonomia(salidas):
    _, t, _ = salidas
    with pytest.raises(sc.ErrorSalidasNLP, match="id_corrida_preparacion"):
        validar(t, id_corrida_preparacion="otra")
    with pytest.raises(sc.ErrorSalidasNLP, match="no coincide con las clases"):
        validar(t, clases_por_pregunta={PREG: CLASES[:2]})


def test_falla_auditoria_desordenada(salidas):
    t = {k: v.copy() for k, v in salidas[1].items()}
    f = t["auditoria"]["clausula_id"] == rid("007") + "|1"
    t["auditoria"].loc[f, ["confianza_aspecto_1", "confianza_aspecto_2"]] = ["0.1", "0.5"]
    _falla(t, "no ordenados")


def test_falla_conteos_de_metadatos(salidas):
    t = {k: v.copy() for k, v in salidas[1].items()}
    t["metadatos"].loc[0, "n_clausulas"] = "999"
    _falla(t, "conteos no coinciden")


# ── parámetros, insumos y errores de construcción ──────────────────────────


@pytest.mark.parametrize("params", [sc.ParametrosSalidas(regla_opinion_general="contenido"),
                                    sc.ParametrosSalidas(alerta_reemplaza_aspecto=False)])
def test_alternativas_no_implementadas_fallan_claro(params):
    with pytest.raises(NotImplementedError):
        construir(params)


def test_mapeo_configurable_a_otro_aspecto():
    mapeo = dict(sc.MAPEO_ASIGNACION_DEFECTO, regla_comentario_general=("CLARIDAD", "regla"))
    s = construir(sc.ParametrosSalidas(mapeo_asignacion=mapeo))
    f = s.clausulas.set_index("clausula_id").loc[rid("005") + "|1"]
    assert (f["aspecto_id"], f["aspecto_origen"]) == ("CLARIDAD", "regla")


def test_clase_sin_aspecto_id_falla():
    e = entrada()
    e.clases = ["Claridad", "Puntualidad", "Otra cosa"]
    with pytest.raises(KeyError, match="sin aspecto_id"):
        construir(entradas=[e])


def test_modo_bertopic_sin_asignacion_mapeada_falla():
    e = entrada()
    e.clausulas.loc[0, "asignacion"] = "directa"
    with pytest.raises(ValueError, match="sin mapeo"):
        construir(entradas=[e])


def test_leer_alertas_con_y_sin_categoria(tmp_path, caplog):
    sin = tmp_path / "alertas.xlsx"
    pd.DataFrame({"TERMINO": ["acoso", "abuso"]}).to_excel(sin, index=False)
    with caplog.at_level(logging.WARNING, logger="salidas_contrato"):
        t, huella, hay = sc.leer_alertas(sin)
    assert not hay and list(t["categoria"]) == ["", ""] and len(huella) == 10 and "CATEGORIA" in caplog.text
    con = tmp_path / "alertas2.xlsx"
    pd.DataFrame({"TERMINO": ["acoso"], "CATEGORIA": ["acoso laboral"]}).to_excel(con, index=False)
    t2, _, hay2 = sc.leer_alertas(con)
    assert hay2 and list(t2["categoria"]) == ["acoso laboral"]


def test_insumos_reales_se_leen():
    t, huella, _ = sc.leer_alertas(RAIZ / "data" / "utilities" / "alertas.xlsx")
    assert len(t) >= 100 and len(huella) == 10
    neg = sc.leer_negadores(RAIZ / "data" / "utilities" / "negaciones.xlsx")
    assert "no" in neg and "nunca" in neg


def test_describir_regla_ajuste():
    assert "margen_neu=0.25" in sc.describir_regla_ajuste_sentimiento(0.25, False)
    assert "calibrador" in sc.describir_regla_ajuste_sentimiento(0.25, True, 0.6)
