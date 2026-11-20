"""Punto 7 (NLP etapa 2): VALORACIÓN GLOBAL, COMENTARIO GENERAL y OPINIÓN GENERAL son una sola categoría («Opinión general»).

Etapa A: los códigos internos -3/-5 se conservan; la columna interna ``motivo_opinion_general`` distingue el origen. El modelo U4 se
simula (sin torch): se parchea ``etiquetar_clausulas``. Datos sintéticos; no imprime textos de clientes.
"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "tests"))

ft = pytest.importorskip("nlp_utilidades.modulos_nlp")
from nlp_utilidades import salidas_contrato as sc  # noqa: E402
import test_nlp_salidas as ts  # noqa: E402

CLASES = ["Claridad", "Puntualidad", "Opinión general"]

# texto de la cláusula -> probabilidades simuladas de U4 (Claridad, Puntualidad, Opinión general)
PROBS = {
    "excelente profesor": (0.10, 0.10, 0.80),        # sin contenido (GLOBAL) de respuesta corta: si pasara por U4 iría a «Opinión general»
    "clases claras": (0.30, 0.30, 0.40),             # respuesta corta con probabilidad baja -> Comentario General
    "explica los temas": (0.05, 0.05, 0.90),         # U4 predice «Opinión general» con confianza
    "explica claro": (0.90, 0.05, 0.05),             # respuesta corta con probabilidad alta -> Claridad
    "es puntual siempre": (0.05, 0.90, 0.05),
}
TEXTOS = ["Excelente profesor", "Clases claras", "El profesor explica los temas y además es puntual siempre", "Explica claro", "NR"]


def _clausula(doc, n, texto, motivo, aspecto=""):
    return {"doc_id": doc, "clausula_id": n, "clausula": texto, "clausula_clean": texto.lower(), "texto_aspecto": aspecto,
            "n_contenido": 1 if aspecto else 0, "motivo": motivo}


def corpus():
    marco = pd.DataFrame({"respuesta_id": [ts.rid(f"{i:03d}") for i in range(len(TEXTOS))], "texto_original": TEXTOS})
    enc = ft.Encuesta(nombre=f"evadoc_evaluacion :: {ts.PREG}", hoja="evadoc_evaluacion", columna="texto_original", marco=marco,
                      texto_original=TEXTOS, columnas_llave=["respuesta_id"])
    docs = pd.DataFrame({"doc_id": range(len(TEXTOS)), "estado": ["CON_OPINION"] * 4 + ["RUIDO"], "respuesta_id": marco["respuesta_id"]})
    cl = pd.DataFrame([
        _clausula(0, 0, "excelente profesor", ft.MOTIVO_GLOBAL),
        _clausula(1, 0, "clases claras", ft.MOTIVO_ASPECTO, "clases claras"),
        _clausula(2, 0, "explica los temas", ft.MOTIVO_ASPECTO, "explica temas"),
        _clausula(2, 1, "es puntual siempre", ft.MOTIVO_ASPECTO, "puntual"),
        _clausula(3, 0, "explica claro", ft.MOTIVO_ASPECTO, "explica"),
    ])
    return enc, docs, cl


@pytest.fixture()
def modelo(monkeypatch):
    def fake(_modelo, textos):
        return np.array([PROBS[t] for t in textos], dtype=float)

    monkeypatch.setattr(ft, "etiquetar_clausulas", fake)
    return SimpleNamespace(clases=CLASES, run_id="u4_falso", textos_vistos=set(), holdout=None, config={}, carpeta=None,
                           clases_especiales=["Opinión general"], metricas={})


def etiquetar(modelo, **cfg_kw):
    enc, docs, cl = corpus()
    cfg = ft.Config(modo_aspectos="u4", umbral_comentario_general=0.8, residual_hdbscan_activo=False, **cfg_kw)
    return ft.etiquetar_corpus_u4(modelo, enc, docs, cl, cfg, escribir_diagnosticos=False)


def fila(etapas, texto):
    c = etapas.clausulas
    return c[c["clausula"] == texto].iloc[0]


# ── nombre unificado y columna interna ─────────────────────────────────────


def test_las_tres_categorias_se_llaman_opinion_general():
    assert ft.ETIQUETAS_ESPECIALES[ft.VALORACION_GLOBAL] == ft.ETIQUETAS_ESPECIALES[ft.COMENTARIO_GENERAL] == "Opinión general"
    assert ft.NOMBRE_OPINION_GENERAL == "Opinión general" and "VALORACIÓN GLOBAL" not in ft.ETIQUETAS_ESPECIALES.values()


def test_sin_contenido_en_cortas_activo_por_defecto(modelo):
    assert ft.Config().sin_contenido_en_cortas is True
    e = etiquetar(modelo)
    g = fila(e, "excelente profesor")                                          # respuesta corta, sin contenido
    assert (g["topico"], g["asignacion"], g["aspecto"], g["motivo_opinion_general"]) == (ft.VALORACION_GLOBAL, "valoracion_global", "Opinión general", "sin_contenido")
    assert pd.isna(g["p_aspecto_u4"]) and pd.isna(g["aspecto_u4"])             # no pasó por U4


def test_comentario_corto_y_clasificador_y_aspecto_normal(modelo):
    e = etiquetar(modelo)
    cg = fila(e, "clases claras")
    assert (cg["topico"], cg["asignacion"], cg["aspecto"], cg["motivo_opinion_general"]) == (ft.COMENTARIO_GENERAL, "regla_comentario_general", "Opinión general", "comentario_corto")
    og = fila(e, "explica los temas")                                          # U4 predice la clase «Opinión general»
    assert (og["asignacion"], og["aspecto"], og["motivo_opinion_general"]) == ("u4", "Opinión general", "clasificador") and og["topico"] == 2
    normal = fila(e, "explica claro")
    assert (normal["aspecto"], normal["motivo_opinion_general"]) == ("Claridad", "")
    assert fila(e, "es puntual siempre")["motivo_opinion_general"] == ""
    assert set(e.clausulas["motivo_opinion_general"]) <= set(ft.MOTIVOS_OPINION_GENERAL) | {""}


def test_con_el_parametro_apagado_como_antes(modelo):
    e = etiquetar(modelo, sin_contenido_en_cortas=False)
    g = fila(e, "excelente profesor")                                          # ahora pasa por U4: probabilidad 0.80 de «Opinión general»
    assert g["asignacion"] == "u4" and g["aspecto"] == "Opinión general" and g["motivo_opinion_general"] == "clasificador"
    assert g["p_aspecto_u4"] == pytest.approx(0.80)


def test_el_parametro_cambia_la_huella_solo_en_modo_u4():
    assert ft.Config(modo_aspectos="u4").huella() != ft.Config(modo_aspectos="u4", sin_contenido_en_cortas=False).huella()
    assert ft.Config().huella() == ft.Config(sin_contenido_en_cortas=False).huella()   # bertopic: los campos U4 no cuentan


@pytest.mark.parametrize("activo", [True, False])
def test_conciliacion_cuadra_con_la_regla_prendida_o_apagada(modelo, activo):
    e = etiquetar(modelo, sin_contenido_en_cortas=activo)
    c = e.clausulas.assign(sentimiento=np.where(e.clausulas["motivo"].isin([ft.MOTIVO_ASPECTO, ft.MOTIVO_GLOBAL]), "POS", None))
    concil = ft.conciliar(e.documentos, c, pd.DataFrame())
    malos = concil[(concil["nivel"] == "chequeo") & (concil["valor"] != 1)]
    assert malos.empty, malos.to_dict("records")


def test_ruido_no_es_opinion_general(modelo):
    e = etiquetar(modelo)
    assert set(e.clausulas.loc[e.clausulas["motivo"] == ft.MOTIVO_GLOBAL, "motivo_opinion_general"]) == {"sin_contenido"}


# ── salidas del contrato: una sola categoría OPINION_GENERAL ───────────────


def _entrada_desde_etapas(etapas, clases=CLASES):
    cl = etapas.clausulas.copy()
    cols_p = ft.nombres_columnas_p(clases)
    cl["sentimiento"] = "POS"
    cl["confianza"], cl["p_pos"], cl["p_neu"], cl["p_neg"] = 0.9, 0.9, 0.05, 0.05
    return sc.EntradaPregunta(ts.PREG, etapas.documentos.assign(respuesta_completa=TEXTOS), cl, clases, cols_p, "u4_falso", "robertuito")


def test_las_tres_categorias_salen_como_opinion_general(modelo):
    e = etiquetar(modelo)
    s = sc.construir_salidas_contrato([_entrada_desde_etapas(e)], taxonomia=ts.TAXONOMIA, id_corrida_preparacion="x",
                                      version_taxonomia="v", fecha=ts.FECHA, pos_fn=ts.pos_falso)
    c = s.clausulas.set_index("texto_clausula")
    og = c.loc[["excelente profesor", "clases claras", "explica los temas"]]
    assert set(og["aspecto_id"]) == {"OPINION_GENERAL"}
    assert dict(og["aspecto_origen"]) == {"excelente profesor": "regla", "clases claras": "regla", "explica los temas": "clasificador"}
    assert c.loc["explica claro", "aspecto_id"] == "CLARIDAD"
    assert (s.catalogo["aspecto_id"] == "OPINION_GENERAL").sum() == 1 and (s.catalogo["aspecto_nombre"] == "Opinión general").any()
    assert {k: list(v.columns) for k, v in s.tablas().items()} == sc.COLUMNAS
    # la columna del contrato dice cuál de las tres vías fue (y coincide con la columna interna del pipeline)
    assert dict(og["motivo_opinion_general"]) == {"excelente profesor": "sin_contenido", "clases claras": "comentario_corto", "explica los temas": "clasificador"}
    assert c.loc["explica claro", "motivo_opinion_general"] == ""
    interna = e.clausulas.set_index("clausula")["motivo_opinion_general"]
    assert all(interna[t] == c.loc[t, "motivo_opinion_general"] for t in c.index if t in interna.index)


# ── 7.3: «Sin aspecto» y «Ninguno de la lista» ─────────────────────────────


def test_clases_sin_aspecto_y_motivos_coinciden_con_el_modulo():
    assert tuple(sc.CLASES_SIN_ASPECTO) == tuple(ft.CLASES_SIN_ASPECTO_U4)
    assert tuple(sc.MOTIVOS_OPINION_GENERAL) == tuple(ft.MOTIVOS_OPINION_GENERAL)


def test_si_u4_predijera_sin_aspecto_se_publica_como_opinion_general():
    clases = ["Claridad", "Puntualidad", "Opinión general", "Sin aspecto", "Ninguno de la lista"]
    e = ts.entrada()
    e.clases, e.columnas_p = clases, [f"p_u4_{i}" for i in range(5)]
    for i, col in enumerate(e.columnas_p):
        e.clausulas[col] = np.where(e.clausulas["asignacion"] == "u4", [0.1, 0.1, 0.1, 0.5, 0.2][i], np.nan)
    e.clausulas.loc[e.clausulas["asignacion"] == "u4", "aspecto"] = "Sin aspecto"
    s = sc.construir_salidas_contrato([e], taxonomia=ts.TAXONOMIA, id_corrida_preparacion="x", version_taxonomia="v", fecha=ts.FECHA,
                                      pos_fn=ts.pos_falso)
    u4 = s.clausulas[s.clausulas["aspecto_origen"] == "clasificador"]
    assert len(u4) > 0 and set(u4["aspecto_id"]) == {"OPINION_GENERAL"} and set(u4["motivo_opinion_general"]) == {"clasificador"}
    a = s.auditoria
    for _, f in a.iterrows():                                                  # un aspecto no se repite en el top 3
        ids = [x for x in (f["aspecto_1_id"], f["aspecto_2_id"], f["aspecto_3_id"]) if x]
        assert len(ids) == len(set(ids))
    # la suma de las probabilidades de las clases que comparten id sale en el primer lugar (0.1 + 0.5 + 0.2 = 0.8)
    primera = a[a["aspecto_1_id"] == "OPINION_GENERAL"]
    assert len(primera) > 0 and float(primera["confianza_aspecto_1"].iloc[0]) == pytest.approx(0.8)
    # el validador no exige que esas clases estén en la taxonomía
    d = Path(tempfile.mkdtemp())
    sc.escribir_salidas_contrato(s, d)
    sc.validar_salidas_nlp(sc.leer_salidas_contrato(d), ts.respuestas_csv(), id_corrida_preparacion="x", preguntas_procesadas=[ts.PREG],
                           clases_por_pregunta={ts.PREG: clases}, taxonomia=ts.TAXONOMIA)


def test_clase_desconocida_sigue_fallando():
    e = ts.entrada()
    e.clases = ["Claridad", "Puntualidad", "Otra cosa"]
    with pytest.raises(KeyError, match="sin aspecto_id"):
        sc.construir_salidas_contrato([e], taxonomia=ts.TAXONOMIA, id_corrida_preparacion="x", version_taxonomia="v", fecha=ts.FECHA, pos_fn=ts.pos_falso)


# ── columna motivo_opinion_general en clausulas.csv ────────────────────────


def test_columna_en_el_contrato_vacia_fuera_de_opinion_general():
    s = ts.construir()
    c = s.clausulas
    assert "motivo_opinion_general" in sc.COLUMNAS["clausulas"] and list(c.columns) == sc.COLUMNAS["clausulas"]
    og = c["aspecto_id"] == "OPINION_GENERAL"
    assert og.any() and set(c.loc[og, "motivo_opinion_general"]) <= set(sc.MOTIVOS_OPINION_GENERAL) and (c.loc[og, "motivo_opinion_general"] != "").all()
    assert (c.loc[~og, "motivo_opinion_general"] == "").all()
    assert set(c.loc[c["aspecto_origen"] == "regla", "motivo_opinion_general"]) <= {"sin_contenido", "comentario_corto"}


def _releer(s):
    d = Path(tempfile.mkdtemp())
    sc.escribir_salidas_contrato(s, d)
    return sc.leer_salidas_contrato(d)


def _falla(t, mensaje):
    with pytest.raises(sc.ErrorSalidasNLP, match=mensaje):
        ts.validar(t)


def test_validador_pasa_y_relee_la_columna_como_texto():
    t = _releer(ts.construir())
    ts.validar(t)
    assert set(t["clausulas"]["motivo_opinion_general"]) <= {*sc.MOTIVOS_OPINION_GENERAL, ""} and "NA" not in set(t["clausulas"]["motivo_opinion_general"])


def test_falla_motivo_fuera_de_dominio_o_en_cualquier_otro_aspecto():
    t = {k: v.copy() for k, v in _releer(ts.construir()).items()}
    t["clausulas"].loc[t["clausulas"]["aspecto_id"] == "OPINION_GENERAL", "motivo_opinion_general"] = "magia"
    _falla(t, "motivo_opinion_general fuera de")
    t = {k: v.copy() for k, v in _releer(ts.construir()).items()}
    t["clausulas"].loc[t["clausulas"]["aspecto_id"] == "CLARIDAD", "motivo_opinion_general"] = "clasificador"
    _falla(t, "con motivo_opinion_general que no son OPINION_GENERAL")


def test_falla_opinion_general_sin_motivo_o_con_motivo_incoherente_con_el_origen():
    t = {k: v.copy() for k, v in _releer(ts.construir()).items()}
    t["clausulas"].loc[t["clausulas"]["aspecto_id"] == "OPINION_GENERAL", "motivo_opinion_general"] = ""
    _falla(t, "OPINION_GENERAL sin motivo_opinion_general")
    t = {k: v.copy() for k, v in _releer(ts.construir()).items()}
    c = t["clausulas"]
    c.loc[(c["aspecto_id"] == "OPINION_GENERAL") & (c["aspecto_origen"] == "regla"), "motivo_opinion_general"] = "clasificador"
    _falla(t, "de una regla con un motivo distinto")
    t = {k: v.copy() for k, v in _releer(ts.construir()).items()}
    c = t["clausulas"]
    c.loc[(c["aspecto_id"] == "OPINION_GENERAL") & (c["aspecto_origen"] == "clasificador"), "motivo_opinion_general"] = "sin_contenido"
    _falla(t, "del clasificador con un motivo distinto")


def test_falla_si_falta_la_columna():
    t = {k: v.copy() for k, v in _releer(ts.construir()).items()}
    t["clausulas"] = t["clausulas"].drop(columns=["motivo_opinion_general"])
    _falla(t, "faltan columnas")


def test_mapeo_configurable_a_otro_aspecto_deja_el_motivo_vacio():
    mapeo = dict(sc.MAPEO_ASIGNACION_DEFECTO, regla_comentario_general=("CLARIDAD", "regla"), valoracion_global=("CLARIDAD", "regla"))
    c = ts.construir(sc.ParametrosSalidas(mapeo_asignacion=mapeo)).clausulas
    assert (c.loc[c["aspecto_origen"] == "regla", "motivo_opinion_general"] == "").all()
