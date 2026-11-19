"""Fase 3 (NLP etapa 2): ingesta desde la Etapa 1 (respuestas.csv de la corrida más reciente).

Usa corridas sintéticas en un directorio temporal; la prueba con la corrida real se omite si no existe
(data/processed está fuera de git). No imprime textos de clientes.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

ft = pytest.importorskip("nlp_utilidades.modulos_nlp")

COLS = ["respuesta_id", "encuesta", "pregunta_codigo", "rol_respondiente", "periodo", "id_origen",
        "texto_original", "es_no_respuesta", "tiene_alerta", "marca_inconsistencia", "id_corrida_preparacion"]
ID1 = "20261116-100000_aaaaaaaaaa"
ID2 = "20261117-100000_bbbbbbbbbb"


def _fila(enc, preg, periodo, origen, texto, no_resp, corrida=ID2, alerta=False):
    return {"respuesta_id": f"{enc}|{periodo}|{origen}|{preg}", "encuesta": enc, "pregunta_codigo": preg,
            "rol_respondiente": "Estudiantes", "periodo": periodo, "id_origen": origen, "texto_original": texto,
            "es_no_respuesta": str(no_resp), "tiene_alerta": str(alerta), "marca_inconsistencia": "", "id_corrida_preparacion": corrida}


def _filas_base(corrida=ID2):
    return [
        _fila("evadoc_evaluacion", "obs19_aspectos_resaltar", "2025-1", "007-1", "Explica muy bien, pero llega tarde.", False, corrida),
        _fila("evadoc_evaluacion", "obs19_aspectos_resaltar", "2025-1", "008-1", "NA", False, corrida),  # «NA» es un comentario
        _fila("evadoc_evaluacion", "obs19_aspectos_resaltar", "2025-1", "009-1", "NR", True, corrida),
        _fila("evadoc_evaluacion", "obs20_oportunidades_mejora", "2024-2", "007-1", "Debe mejorar la puntualidad.", False, corrida),
        _fila("evadoc_evaluacion", "obs20_oportunidades_mejora", "2024-2", "010-1", "", True, corrida),
        _fila("autoevaluacion_acreditacion", "acred_sugerencias_complementarias", "2025-1", "55", "Mejorar la infraestructura.", False, corrida),
    ]


def _escribir(raiz: Path, filas, corrida=ID2, con_metadatos=True, catalogo=None):
    carpeta = raiz / corrida
    carpeta.mkdir(parents=True)
    pd.DataFrame(filas, columns=COLS).to_csv(carpeta / "respuestas.csv", index=False, encoding="utf-8-sig")
    meta = pd.DataFrame(filas, columns=COLS).groupby("pregunta_codigo").apply(
        lambda g: pd.Series({"respuestas": len(g), "respuestas_con_comentario": int((g["es_no_respuesta"] == "False").sum()),
                             "respuestas_con_alerta": int((g["tiene_alerta"] == "True").sum())}),
        include_groups=False).reset_index()
    if con_metadatos:
        meta.to_csv(carpeta / "metadatos_preparacion.csv", index=False, encoding="utf-8-sig")
    cat = catalogo if catalogo is not None else sorted({(f["pregunta_codigo"], f["encuesta"]) for f in filas})
    pd.DataFrame(cat, columns=["pregunta_codigo", "encuesta"]).to_csv(carpeta / "catalogo_preguntas.csv", index=False, encoding="utf-8-sig")
    return carpeta


def test_lee_la_corrida_y_filtra_no_respuestas(tmp_path):
    _escribir(tmp_path, _filas_base())
    enc = ft.cargar_encuestas_etapa1(tmp_path)
    assert sorted(enc) == ["autoevaluacion_acreditacion :: acred_sugerencias_complementarias",
                           "evadoc_evaluacion :: obs19_aspectos_resaltar", "evadoc_evaluacion :: obs20_oportunidades_mejora"]
    o19 = enc["evadoc_evaluacion :: obs19_aspectos_resaltar"]
    assert len(o19.texto_original) == 2 and "NR" not in o19.texto_original   # la no-respuesta se filtra
    assert "NA" in o19.texto_original                                         # «NA» no se pierde
    assert set(o19.marco["periodo"]) == {"2025-1"}                             # el periodo sigue siendo texto
    assert o19.marco["respuesta_id"].is_unique and (o19.marco["respuesta_id"] != "").all()
    assert o19.marco["respuesta_id"].iloc[0].endswith("|007-1|obs19_aspectos_resaltar")   # ceros a la izquierda intactos
    assert o19.columnas_llave == ["respuesta_id"] and o19.columna == "texto_original"
    assert o19.hoja == "evadoc_evaluacion" and o19.clave_stopwords == "evadocevaluacion"
    assert {e.id_corrida_preparacion for e in enc.values()} == {ID2}
    assert len(enc["evadoc_evaluacion :: obs20_oportunidades_mejora"].texto_original) == 1


def test_filtro_por_pregunta(tmp_path):
    _escribir(tmp_path, _filas_base())
    enc = ft.cargar_encuestas_etapa1(tmp_path, preguntas=["obs20_oportunidades_mejora"])
    assert list(enc) == ["evadoc_evaluacion :: obs20_oportunidades_mejora"]


def test_toma_la_corrida_mas_reciente(tmp_path):
    _escribir(tmp_path, _filas_base(ID1)[:1], corrida=ID1)
    _escribir(tmp_path, _filas_base(ID2), corrida=ID2)
    assert ft.leer_respuestas_etapa1(tmp_path)[0] == ID2


def test_sin_corridas_falla_con_mensaje_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="No hay corridas de la Etapa 1"):
        ft.cargar_encuestas_etapa1(tmp_path)
    (tmp_path / ID2).mkdir()  # carpeta sin respuestas.csv/metadatos: no cuenta como corrida
    with pytest.raises(FileNotFoundError, match="run_etapa1.py"):
        ft.cargar_encuestas_etapa1(tmp_path)


def test_dos_id_corrida_en_respuestas_falla(tmp_path):
    filas = _filas_base()
    filas[1]["id_corrida_preparacion"] = ID1
    _escribir(tmp_path, filas)
    with pytest.raises(ValueError, match="no traen todas el id de la corrida"):
        ft.cargar_encuestas_etapa1(tmp_path)


@pytest.mark.parametrize("cambio, mensaje", [
    (lambda f: f[1].update(es_no_respuesta="Sí"), "es_no_respuesta solo admite True/False"),
    (lambda f: f[1].update(tiene_alerta="Sí"), "tiene_alerta solo admite True/False"),
    (lambda f: f[1].update(respuesta_id=f[0]["respuesta_id"]), "repetidos"),
    (lambda f: f[0].update(respuesta_id=""), "vacío"),
])
def test_datos_invalidos_fallan(tmp_path, cambio, mensaje):
    filas = _filas_base()
    cambio(filas)
    _escribir(tmp_path, filas)
    with pytest.raises(ValueError, match=mensaje):
        ft.cargar_encuestas_etapa1(tmp_path)


def test_pregunta_fuera_del_catalogo_falla(tmp_path):
    _escribir(tmp_path, _filas_base(), catalogo=[("obs19_aspectos_resaltar", "evadoc_evaluacion")])
    with pytest.raises(ValueError, match="no están en catalogo_preguntas.csv"):
        ft.cargar_encuestas_etapa1(tmp_path)


def test_falta_una_columna_falla(tmp_path):
    carpeta = _escribir(tmp_path, _filas_base())
    pd.read_csv(carpeta / "respuestas.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig").drop(
        columns=["texto_original"]).to_csv(carpeta / "respuestas.csv", index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="texto_original"):
        ft.cargar_encuestas_etapa1(tmp_path)


def test_comparacion_con_metadatos(tmp_path):
    _escribir(tmp_path, _filas_base())
    enc = ft.cargar_encuestas_etapa1(tmp_path)
    cmp_ = ft.comparar_con_metadatos_etapa1(enc, tmp_path)
    assert (cmp_["diferencia"] == 0).all() and cmp_["leidas"].sum() == 4


def test_segmentacion_sobre_texto_original_con_respuesta_id(tmp_path):
    """20 respuestas sintéticas: cada cláusula es un fragmento literal de texto_original y lleva su respuesta_id."""
    base = ["Explica muy bien, pero llega tarde.", "Las clases son claras; la evaluación es justa.",
            "Excelente docente. Muy puntual", "No responde los correos y tampoco da retroalimentación.",
            "Dr. Pérez explica con ejemplos y motiva a los estudiantes."]
    filas = [_fila("evadoc_evaluacion", "obs19_aspectos_resaltar", "2025-1", f"{i:03d}-1", base[i % 5] + (" " * (i % 2)), False)
             for i in range(20)]
    _escribir(tmp_path, filas)
    enc = ft.cargar_encuestas_etapa1(tmp_path)["evadoc_evaluacion :: obs19_aspectos_resaltar"]
    recursos = ft.RecursosLexicos.cargar(RAIZ / "data" / "utilities")
    cfg = ft.Config()
    documentos, clausulas = ft.construir_unidades(enc, cfg, recursos, segmentador=ft.Segmentador(cfg, recursos))
    assert len(documentos) == 20 and documentos["respuesta_id"].is_unique
    # respuesta_id llega a las cláusulas por doc_id (llave única de documentos), como ya se une respuesta_completa
    unido = clausulas.merge(documentos[["doc_id", "respuesta_id"]], on="doc_id", how="left", validate="many_to_one")
    assert len(unido) == len(clausulas) and unido["respuesta_id"].notna().all()
    assert set(unido["respuesta_id"]) <= set(enc.marco["respuesta_id"])
    texto = dict(zip(enc.marco["respuesta_id"], enc.texto_original))
    no_literales = [c for c, r in zip(unido["clausula"], unido["respuesta_id"]) if c.strip() not in texto[r]]
    assert not no_literales, f"{len(no_literales)} cláusulas no son fragmentos literales"


CARPETA_REAL = RAIZ / "data" / "processed" / "etapa1"


@pytest.mark.skipif(not CARPETA_REAL.exists() or not any(CARPETA_REAL.glob("*/respuestas.csv")),
                    reason="no hay corridas de la Etapa 1 en data/processed/etapa1")
def test_corrida_real_coincide_con_metadatos():
    enc = ft.cargar_encuestas_etapa1(CARPETA_REAL)
    cmp_ = ft.comparar_con_metadatos_etapa1(enc, CARPETA_REAL)
    assert (cmp_["diferencia"] == 0).all(), cmp_.to_dict("records")
    for e in enc.values():
        assert e.marco["respuesta_id"].is_unique and len(e.texto_original) == len(e.marco)


# ── Punto 1: solo es_no_respuesta = False y tiene_alerta = False ───────────


def _filas_cuatro_combinaciones():
    """obs19 con las 4 combinaciones de es_no_respuesta/tiene_alerta, un periodo «2025-1» y un «NA»."""
    f = lambda o, t, nr, al: _fila("evadoc_evaluacion", "obs19_aspectos_resaltar", "2025-1", o, t, nr, alerta=al)  # noqa: E731
    return [
        f("001-1", "Explica muy bien.", False, False),   # procesable
        f("002-1", "NA", False, False),                  # procesable («NA» es un comentario)
        f("003-1", "NR", True, False),                   # no-respuesta
        f("004-1", "Sufrí acoso", False, True),          # con alerta
        f("005-1", "", True, False),                     # no-respuesta (celda vacía)
        _fila("evadoc_evaluacion", "obs20_oportunidades_mejora", "2024-2", "007-1", "Debe mejorar.", False),
        _fila("evadoc_evaluacion", "obs20_oportunidades_mejora", "2024-2", "008-1", "Hubo discriminación", False, alerta=True),
    ]


def test_solo_pasan_las_respuestas_sin_alerta_ni_no_respuesta(tmp_path):
    _escribir(tmp_path, _filas_cuatro_combinaciones())
    enc = ft.cargar_encuestas_etapa1(tmp_path)
    o19 = enc["evadoc_evaluacion :: obs19_aspectos_resaltar"]
    assert sorted(o19.marco["respuesta_id"].str.split("|").str[2]) == ["001-1", "002-1"]
    assert set(o19.marco["tiene_alerta"]) == {"False"} and set(o19.marco["es_no_respuesta"]) == {"False"}
    assert "NA" in o19.texto_original and set(o19.marco["periodo"]) == {"2025-1"}
    assert o19.conteos_ingesta == {"total": 5, "no_respuestas": 2, "con_alerta": 1, "procesadas": 2}
    o20 = enc["evadoc_evaluacion :: obs20_oportunidades_mejora"]
    assert len(o20.texto_original) == 1 and o20.conteos_ingesta == {"total": 2, "no_respuestas": 0, "con_alerta": 1, "procesadas": 1}


def test_ningun_texto_de_alerta_llega_al_corpus(tmp_path):
    _escribir(tmp_path, _filas_cuatro_combinaciones())
    enc = ft.cargar_encuestas_etapa1(tmp_path)
    textos = [t for e in enc.values() for t in e.texto_original]
    assert "Sufrí acoso" not in textos and "Hubo discriminación" not in textos


def test_comparacion_con_metadatos_descuenta_las_alertas(tmp_path):
    _escribir(tmp_path, _filas_cuatro_combinaciones())
    cmp_ = ft.comparar_con_metadatos_etapa1(ft.cargar_encuestas_etapa1(tmp_path), tmp_path).set_index("pregunta_codigo")
    assert (cmp_["diferencia"] == 0).all()
    assert cmp_.loc["obs19_aspectos_resaltar", ["total", "no_respuestas", "con_alerta", "leidas", "metadatos"]].tolist() == [5, 2, 1, 2, 2]


def test_falta_tiene_alerta_en_respuestas_pide_volver_a_correr_la_etapa1(tmp_path):
    carpeta = _escribir(tmp_path, _filas_base())
    pd.read_csv(carpeta / "respuestas.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig").drop(
        columns=["tiene_alerta"]).to_csv(carpeta / "respuestas.csv", index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="tiene_alerta.*run_etapa1.py"):
        ft.cargar_encuestas_etapa1(tmp_path)


def test_metadatos_sin_respuestas_con_alerta_piden_volver_a_correr(tmp_path):
    carpeta = _escribir(tmp_path, _filas_base())
    pd.read_csv(carpeta / "metadatos_preparacion.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig").drop(
        columns=["respuestas_con_alerta"]).to_csv(carpeta / "metadatos_preparacion.csv", index=False, encoding="utf-8-sig")
    enc = ft.cargar_encuestas_etapa1(tmp_path)
    with pytest.raises(ValueError, match="respuestas_con_alerta.*run_etapa1.py"):
        ft.comparar_con_metadatos_etapa1(enc, tmp_path)


def test_si_todas_tienen_alerta_no_queda_nada(tmp_path):
    f = _fila("evadoc_evaluacion", "obs19_aspectos_resaltar", "2025-1", "001-1", "Acoso", False, alerta=True)
    _escribir(tmp_path, [f])
    with pytest.raises(ValueError, match="no tiene respuestas procesables"):
        ft.cargar_encuestas_etapa1(tmp_path)


def test_no_respuesta_con_alerta_tambien_se_excluye(tmp_path):
    """La Etapa 1 solo evalúa alertas si es_no_respuesta = False; si llegara (True, True) igual no pasa."""
    filas = _filas_base()[:1] + [_fila("evadoc_evaluacion", "obs19_aspectos_resaltar", "2025-1", "099-1", "x", True, alerta=True)]
    _escribir(tmp_path, filas)
    o19 = ft.cargar_encuestas_etapa1(tmp_path)["evadoc_evaluacion :: obs19_aspectos_resaltar"]
    assert len(o19.texto_original) == 1
    assert o19.conteos_ingesta == {"total": 2, "no_respuestas": 1, "con_alerta": 0, "procesadas": 1}
