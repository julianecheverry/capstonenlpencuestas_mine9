"""Punto 4 (NLP etapa 2): SIN_ASPECTO y temas de HDBSCAN con aspecto_id = primer sustantivo de palabras_clave.

El etiquetador POS se inyecta (``pos_fn``): no necesita spaCy. Con ``es_core_news_lg`` instalado hay además una prueba
con el modelo real. No imprime textos de clientes.
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

PROCESO = dict(id_corrida_preparacion="20261117-211304_3d69efc5d6")


def con_grupos(e, grupos: dict[str, tuple[int, str]]):
    """Convierte las cláusulas de los documentos indicados (origen) en residuales de HDBSCAN.

    ``grupos``: origen del documento -> (grupo, palabras_clave como «a | b | c»). Reemplaza ``topicos_residuales``.
    """
    ids = list(e.documentos["respuesta_id"])
    terminos = {}
    for origen, (g, term) in grupos.items():
        d = ids.index(ts.rid(origen, pregunta=e.pregunta_codigo))
        m = e.clausulas["doc_id"] == d
        e.clausulas.loc[m, "asignacion"] = "residual_hdbscan"
        e.clausulas.loc[m, "topico_residual"] = g
        e.clausulas.loc[m, "etiqueta_residual"] = f"R{g}"
        e.clausulas.loc[m, "p_residual"] = 0.5
        terminos[g] = term
    e.topicos_residuales = pd.DataFrame({"topico": list(terminos), "etiqueta": [f"R{g}" for g in terminos], "terminos": list(terminos.values())})
    return e


def sin_grupos(e):
    """Sin ningún residual con grupo: no hace falta etiquetador."""
    m = e.clausulas["asignacion"] == "residual_hdbscan"
    e.clausulas.loc[m, "asignacion"] = "u4"
    e.clausulas.loc[m, "aspecto"] = "Claridad"
    e.topicos_residuales = pd.DataFrame(columns=["topico", "etiqueta", "terminos"])
    return e


def construir(entradas, **kw):
    return sc.construir_salidas_contrato(entradas, taxonomia=ts.TAXONOMIA, version_taxonomia="9517d60bf5", fecha=ts.FECHA,
                                         pos_fn=kw.pop("pos_fn", ts.pos_falso), **PROCESO, **kw)


# ── utilidades de ID ───────────────────────────────────────────────────────


@pytest.mark.parametrize("texto, esperado", [
    ("dinámicas", "DINAMICAS"), ("Niño", "NINO"), ("evaluación académica", "EVALUACION_ACADEMICA"),
    ("a-b", "A_B"), ("  clases  ", "CLASES"), ("año2", "ANO2"), ("--", ""), ("", "")])
def test_normalizar_id(texto, esperado):
    assert sc.normalizar_id(texto) == esperado
    assert not esperado or sc.PATRON_ID.match(esperado)


def test_parsear_palabras_clave():
    assert sc.parsear_palabras_clave("dinámicas | propone | visto") == ["dinámicas", "propone", "visto"]
    assert sc.parsear_palabras_clave("a, b ,, c") == ["a", "b", "c"]
    assert sc.parsear_palabras_clave(["a ", "", "b"]) == ["a", "b"] and sc.parsear_palabras_clave(None) == []


def test_primer_sustantivo_en_contexto_y_reglas():
    llamadas = []

    def pos(palabras):
        llamadas.append(list(palabras))
        return [{"muy": "ADV", "propone": "VERB", "Pérez": "PROPN", "buen": "ADJ"}.get(w, "NOUN") for w in palabras]

    assert sc.primer_sustantivo(["muy", "propone", "Pérez", "clases", "docente"], pos) == ("clases", True)   # PROPN no cuenta
    assert llamadas == [["muy", "propone", "Pérez", "clases", "docente"]]                                    # una sola frase de contexto
    assert sc.primer_sustantivo(["buen docente", "clases"], pos) == ("docente", True)                         # término de varias palabras
    assert sc.primer_sustantivo(["muy", "propone"], pos) == ("muy", False)                                    # sin sustantivo: primer término
    assert sc.primer_sustantivo([], pos) == (None, False) and sc.primer_sustantivo(["--"], pos) == (None, False)
    with pytest.raises(ValueError, match="etiquetas"):
        sc.primer_sustantivo(["a", "b"], lambda w: ["NOUN"])


def test_id_de_tema_choques():
    k = dict(fijos={"CLARIDAD"}, n_respaldo=3)
    assert sc.id_de_tema("clases", usados_pregunta=set(), **k) == "CLASES"
    assert sc.id_de_tema("clases", usados_pregunta={"CLASES"}, **k) == "CLASES_2"
    assert sc.id_de_tema("clases", usados_pregunta={"CLASES", "CLASES_2"}, **k) == "CLASES_3"
    assert sc.id_de_tema("Claridad", usados_pregunta=set(), **k) == "EMERGENTE_CLARIDAD"                       # choque con la taxonomía
    assert sc.id_de_tema("claridad", usados_pregunta={"EMERGENTE_CLARIDAD"}, **k) == "EMERGENTE_CLARIDAD_2"
    assert sc.id_de_tema("sin aspecto", usados_pregunta=set(), **k) == "EMERGENTE_SIN_ASPECTO"                 # SIN_ASPECTO está reservado
    assert sc.id_de_tema(None, usados_pregunta=set(), **k) == "TEMA_3"


# ── SIN_ASPECTO y temas en las tablas ──────────────────────────────────────


def test_sin_aspecto_siempre_hdbscan_y_sin_grupo_visual():
    s = construir([ts.entrada()])
    c = s.clausulas
    sin = c[c["aspecto_id"] == "SIN_ASPECTO"]
    assert len(sin) >= 2 and set(sin["aspecto_origen"]) == {"hdbscan"}                                         # residual sin grupo y sin opinión
    assert "sistema" not in set(c["aspecto_origen"])
    fila = s.catalogo[s.catalogo["aspecto_id"] == "SIN_ASPECTO"]
    assert len(fila) == 1 and (fila.iloc[0]["aspecto_origen"], fila.iloc[0]["grupo_visual"]) == ("hdbscan", "Sin grupo visual")


def test_temas_con_id_del_sustantivo_sin_emergente_periodo_n():
    e = con_grupos(ts.entrada(), {"008": (0, "propone | clases | muy"), "005": (1, "visto | docente"), "006": (2, "muy | plataforma")})
    s = construir([e])
    cat, c = s.catalogo, s.clausulas
    emg = cat[cat["grupo_visual"] == "Temas emergentes"]
    assert set(emg["aspecto_id"]) == {"CLASES", "DOCENTE", "PLATAFORMA"}
    assert not cat["aspecto_id"].str.startswith("EMERGENTE_").any() and not c["aspecto_id"].str.contains(r"\d{4}-\d").any()
    assert set(emg["aspecto_origen"]) == {"hdbscan"} and set(emg["aspecto_nombre"]) == {"clases", "docente", "plataforma"}   # nombre = sustantivo
    assert dict(zip(emg["aspecto_id"], emg["palabras_clave"]))["CLASES"] == "propone, clases, muy"                  # coma, como pide el contrato
    assert c.set_index("clausula_id").loc[ts.rid("008") + "|1", "aspecto_id"] == "CLASES"
    assert all(sc.PATRON_ID.match(i) for i in set(c["aspecto_id"]) | set(cat["aspecto_id"]))
    assert set(c["aspecto_id"]) <= set(cat["aspecto_id"]) and not cat.duplicated(["aspecto_id", "encuesta"]).any()
    ts.validar({k: v for k, v in _releer(s).items()})


def _releer(s):
    d = Path(tempfile.mkdtemp())
    sc.escribir_salidas_contrato(s, d)
    return sc.leer_salidas_contrato(d)


def test_choque_con_la_taxonomia_lleva_prefijo():
    e = con_grupos(ts.entrada(), {"008": (0, "claridad | propone"), "005": (1, "puntualidad | visto")})
    s = construir([e])
    cat = s.catalogo
    assert {"CLARIDAD", "PUNTUALIDAD", "EMERGENTE_CLARIDAD", "EMERGENTE_PUNTUALIDAD"} <= set(cat["aspecto_id"])
    grupos = dict(zip(cat["aspecto_id"], cat["grupo_visual"]))
    assert grupos["CLARIDAD"] == "Aspectos definidos" and grupos["EMERGENTE_CLARIDAD"] == "Temas emergentes"
    assert not cat.duplicated(["aspecto_id", "encuesta"]).any()
    ts.validar(_releer(s))


def test_choque_con_taxonomia_de_otra_pregunta_de_la_encuesta():
    """PUNTUALIDAD es fijo en obs19 (no en obs20): un tema «puntualidad» de obs20 lleva prefijo igual (misma encuesta)."""
    e1 = ts.entrada()
    e2 = ts.entrada(ts.OTRA)
    e2.clases, e2.columnas_p = ["Claridad", "Opinión general"], ["p_u4_claridad", "p_u4_opinion_general"]
    e2.clausulas = e2.clausulas.drop(columns=["p_u4_puntualidad"]).assign(aspecto=lambda d: d["aspecto"].replace({"Puntualidad": "Claridad"}))
    e2 = con_grupos(e2, {"008": (0, "puntualidad | propone")})
    s = construir([e1, e2])
    assert "EMERGENTE_PUNTUALIDAD" in set(s.catalogo["aspecto_id"])


def test_choque_entre_temas_de_la_misma_pregunta_sufijo():
    e = con_grupos(ts.entrada(), {"008": (0, "clases | propone"), "005": (1, "clases | visto"), "006": (2, "clases | muy")})
    s = construir([e])
    emg = s.catalogo[s.catalogo["grupo_visual"] == "Temas emergentes"]
    assert set(emg["aspecto_id"]) == {"CLASES", "CLASES_2", "CLASES_3"}
    c = s.clausulas.set_index("clausula_id")
    assert [c.loc[ts.rid(o) + "|1", "aspecto_id"] for o in ("008", "005", "006")] == ["CLASES", "CLASES_2", "CLASES_3"]   # por orden de grupo


def test_fallback_primer_termino_con_aviso():
    e = con_grupos(ts.entrada(), {"008": (0, "propone | visto"), "005": (1, "clases | docente")})
    s = construir([e])
    assert {"PROPONE", "CLASES"} <= set(s.catalogo["aspecto_id"])
    assert any("1 temas sin ningún sustantivo" in a for a in s.avisos)
    e2 = con_grupos(ts.entrada(), {"008": (7, "|| ,")})   # sin términos utilizables
    assert "TEMA_7" in set(construir([e2]).catalogo["aspecto_id"])


def test_diez_grupos_sinteticos_todos_validos():
    docs = ["001", "005", "006", "007", "008", "009", "010"]   # documentos con opinión
    nombres = ["clases", "docente", "tareas", "material", "tiempo", "ejemplos", "plataforma"]
    e = con_grupos(ts.entrada(), {o: (g, f"{nombres[g]} | propone") for g, o in enumerate(docs)})
    s = construir([e])
    emg = s.catalogo[s.catalogo["grupo_visual"] == "Temas emergentes"]
    assert len(emg) == 7 and all(sc.PATRON_ID.match(i) for i in emg["aspecto_id"])
    assert not s.catalogo["aspecto_id"].str.match(r"^EMERGENTE_\d").any()


def test_sin_grupos_no_necesita_etiquetador(monkeypatch):
    monkeypatch.setattr(sc, "crear_pos_spacy", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no debía pedirse spaCy")))
    s = sc.construir_salidas_contrato([sin_grupos(ts.entrada())], taxonomia=ts.TAXONOMIA, version_taxonomia="x", fecha=ts.FECHA, **PROCESO)
    assert not (s.catalogo["grupo_visual"] == "Temas emergentes").any()


def test_sin_spacy_falla_con_mensaje_claro():
    try:
        import spacy  # noqa: F401
        pytest.skip("spaCy está instalado")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="es_core_news_lg"):
        sc.construir_salidas_contrato([con_grupos(ts.entrada(), {"008": (0, "clases | propone")})], taxonomia=ts.TAXONOMIA,
                                      version_taxonomia="x", fecha=ts.FECHA, **PROCESO)


def test_determinismo_con_temas(tmp_path):
    e = lambda: con_grupos(ts.entrada(), {"008": (0, "clases | propone"), "005": (1, "claridad | visto")})  # noqa: E731
    sc.escribir_salidas_contrato(construir([e()]), tmp_path / "a")
    sc.escribir_salidas_contrato(construir([e()]), tmp_path / "b")
    assert sc.huella_archivos(tmp_path / "a") == sc.huella_archivos(tmp_path / "b")


# ── porcentajes sin solape ─────────────────────────────────────────────────


def test_pct_hdbscan_solo_temas_y_sin_aspecto_aparte():
    e = con_grupos(ts.entrada(), {"008": (0, "clases | propone")})
    s = construir([e])
    c, m = s.clausulas, s.metadatos.iloc[0]
    n = len(c)
    esperado_h = round(100 * ((c["aspecto_origen"] == "hdbscan") & (c["aspecto_id"] != "SIN_ASPECTO")).sum() / n, 2)
    esperado_s = round(100 * (c["aspecto_id"] == "SIN_ASPECTO").sum() / n, 2)
    assert (m["pct_aspecto_hdbscan"], m["pct_sin_aspecto"]) == (esperado_h, esperado_s) and esperado_h > 0 and esperado_s > 0
    total = float(m["pct_aspecto_clasificador"]) + float(m["pct_aspecto_hdbscan"]) + float(m["pct_aspecto_regla"]) + float(m["pct_sin_aspecto"])
    assert total == pytest.approx(100.0, abs=0.05)


# ── validador: reglas del punto 4 (pruebas negativas) ──────────────────────


@pytest.fixture()
def tablas():
    return {k: v.copy() for k, v in _releer(construir([con_grupos(ts.entrada(), {"008": (0, "clases | propone")})])).items()}


def _falla(t, mensaje):
    with pytest.raises(sc.ErrorSalidasNLP, match=mensaje):
        ts.validar(t)


def test_validador_pasa_con_el_esquema_nuevo(tablas):
    ts.validar(tablas)


def test_falla_sin_aspecto_con_origen_sistema(tablas):
    tablas["clausulas"].loc[tablas["clausulas"]["aspecto_id"] == "SIN_ASPECTO", "aspecto_origen"] = "sistema"
    _falla(tablas, "SIN_ASPECTO con aspecto_origen distinto de hdbscan")


def test_falla_catalogo_sin_aspecto_mal_marcado(tablas):
    cat = tablas["catalogo"]
    cat.loc[cat["aspecto_id"] == "SIN_ASPECTO", "grupo_visual"] = "Aspectos definidos"
    _falla(tablas, "SIN_ASPECTO debe ser hdbscan / Sin grupo visual")


def test_falla_id_emergente_antiguo_o_formato_invalido(tablas):
    cat, c = tablas["catalogo"], tablas["clausulas"]
    cat.loc[cat["aspecto_id"] == "CLASES", "aspecto_id"] = "EMERGENTE_2025-1_0"
    c.loc[c["aspecto_id"] == "CLASES", "aspecto_id"] = "EMERGENTE_2025-1_0"
    _falla(tablas, "formato inválido")


def test_falla_grupo_visual_fuera_de_dominio_y_emergente_sin_hdbscan(tablas):
    cat = tablas["catalogo"]
    cat.loc[cat["aspecto_id"] == "CLASES", "aspecto_origen"] = "clasificador"
    _falla(tablas, "temas emergentes deben tener aspecto_origen hdbscan")


def test_falla_pct_que_solapan(tablas):
    tablas["metadatos"].loc[0, "pct_sin_aspecto"] = "90"
    _falla(tablas, "no suman 100")


# ── spaCy real (si está disponible) ────────────────────────────────────────


def test_spacy_real_en_contexto():
    spacy = pytest.importorskip("spacy")
    if not spacy.util.is_package("es_core_news_lg"):
        pytest.skip("falta es_core_news_lg")
    pos = sc.crear_pos_spacy()
    grupos = [["dinámicas", "propone", "visto"], ["clases", "virtuales", "plataforma"], ["explica", "claro", "ejemplos"],
              ["tiempo", "tareas", "entrega"], ["material", "apoyo", "lecturas"], ["horario", "clases", "puntual"],
              ["evaluación", "justa", "notas"], ["docente", "respeto", "trato"], ["tecnología", "uso", "herramientas"],
              ["comunicación", "correos", "respuesta"]]
    ids = []
    for g in grupos:
        sust, hallado = sc.primer_sustantivo(g, pos)
        ids.append(sc.normalizar_id(sust or ""))
        assert sc.PATRON_ID.match(ids[-1])
    assert sum(1 for i in ids if i) == 10
