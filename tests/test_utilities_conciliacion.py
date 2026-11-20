"""Fase 1 (NLP etapa 2): conciliación de los Excel de data/utilities.

Usa los archivos reales de data/utilities (versionados). Las etiquetas manuales
(data/etiquetas_manuales) están fuera de git: esa prueba se omite si no existen.
No imprime textos de clientes: solo nombres de aspectos y conteos.
"""
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from limpieza.listas import cargar_listas  # noqa: E402
from nlp_utilidades.taxonomia import leer_taxonomia  # noqa: E402
from preparacion.configuracion import codigos_preguntas  # noqa: E402

UTIL = RAIZ / "data" / "utilities"
ETIQ = RAIZ / "data" / "etiquetas_manuales"
ENCUESTAS = ["evadoc_evaluacion", "evadoc_autoevaluacion", "calidad_servicios", "autoevaluacion_acreditacion"]
TIPOS = {"NEGADOR", "ADVERSATIVO", "CUANTIFICADOR_BAJO", "INTENSIFICADOR", "MODAL_ATENUANTE"}

# Pregunta -> (archivo, hoja) vigente de etiquetas manuales (decisiones Fase 1).
ETIQUETAS = {
    "obs19_aspectos_resaltar": ("--Prisma_Validacion_Humana-obs19 - auditado-27102026.xlsx", "obs19"),
    "obs20_oportunidades_mejora": ("--Prisma_Validacion_Humana-obs20 - auditado-10112026.xlsx", "obs20"),
    "obs23_percepcion_progreso": ("--Prisma_Validacion_Humana-obs23-13112026.xlsx", "obs23"),
    "obs24_aspectos_resaltar": ("--Prisma_Validacion_Humana-obs24-13112026.xlsx", "obs24"),
    "obs25_oportunidades_mejora": ("--Prisma_Validacion_Humana-obs25-13112026.xlsx", "obs25"),
    "acred_sugerencias_complementarias": ("--Prisma_Validacion_Humana-acreditacion-auditado-14112026.xlsx",
                                          "acreditacion_etiquetas"),
}


def _n(s: str) -> str:
    sin = "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
    return " ".join(sin.lower().split())


def _hojas(nombre: str) -> dict[str, pd.DataFrame]:
    return pd.read_excel(UTIL / nombre, sheet_name=None, dtype=str, keep_default_na=False)


def test_taxonomia_valida_y_conteos():
    tax = leer_taxonomia(UTIL / "taxonomia.xlsx", codigos_preguntas(UTIL / "preguntas_abiertas.xlsx"))
    conteo = tax.groupby("pregunta_codigo").size().to_dict()
    assert conteo == {
        "obs19_aspectos_resaltar": 17, "obs20_oportunidades_mejora": 17, "obs23_percepcion_progreso": 31,
        "obs24_aspectos_resaltar": 32, "obs25_oportunidades_mejora": 30, "acred_sugerencias_complementarias": 19,
    }
    voc = tax[tax["aspecto_id"] == "VOCACION"]
    assert set(voc["pregunta_codigo"]) == {"obs19_aspectos_resaltar", "obs20_oportunidades_mejora", "obs24_aspectos_resaltar"}


def test_negaciones_cinco_columnas_y_cinco_tipos():
    neg = _hojas("negaciones.xlsx")["negaciones"]
    assert list(neg.columns) == ["TERMINO", "TERMINO_SIN_TILDE", "TIPO", "ALCANCE", "NOTA"]
    assert set(neg["TIPO"]) == TIPOS and len(neg) == 44


def test_archivos_1_retirados_de_utilities():
    assert not (UTIL / "negaciones_1.xlsx").exists() and not (UTIL / "stopwords_1.xlsx").exists()


def test_etapa3_lee_stopwords_y_negaciones():
    listas = cargar_listas(UTIL / "stopwords.xlsx", UTIL / "negaciones.xlsx")
    for enc in ENCUESTAS:
        globales, _frases = listas.stopwords(enc)
        assert globales, enc
        assert {"dos", "tres", "queda", "quedan"} <= globales  # agregadas a globales (decisión Fase 1)


def test_stopwords_xlsx_estructura_intacta_y_valorativas_fuera_de_globales():
    h = _hojas("stopwords.xlsx")
    assert set(h) == {"base", "globales", "particulares", "adverbios"}
    assert list(h["adverbios"].columns) == ["encuesta", "palabra", "valora"]
    glob = set(h["globales"]["palabra"])
    assert not glob & {"bien", "mal", "mejor", "nunca", "peor", "siempre"}  # la Etapa 3 debe conservarlas
    assert len(h["globales"]) == 28
    part = h["particulares"]
    assert (part[part["palabra"] == "figri"]["encuesta"].str.startswith("stopwords_calidad")).sum() == 5


def test_lector_nlp_stopwords_claves_y_conteos():
    ft = pytest.importorskip("nlp_utilidades.modulos_nlp")
    globales, part = ft.RecursosLexicos._leer_stopwords(UTIL, {})
    assert len(globales) == 87  # = globales de stopwords_1.xlsx (decisión Fase 1, opción A)
    assert {"bien", "mal", "mejor", "nunca", "peor", "siempre"} <= globales
    assert not globales & set(ft.ADVERBIOS_NO_STOPWORD_NLP)
    assert not {"base", "adverbios"} & set(part)  # ya no entran hojas como "claves"
    # claves del catálogo (sin guion bajo, como compara stopwords_vectorizador) para las 4 encuestas
    for enc, esperada in {"evadoc_evaluacion": 31, "evadoc_autoevaluacion": 16}.items():
        assert len(part[enc.replace("_", "")]) == esperada, enc
    assert "figri" in part["calidadservicios"] and len(part["calidadservicios"]) > 10
    assert "acreditacion" not in "".join(part)  # sin particulares, igual que en la Etapa 3
    # claves antiguas (cuaderno actual) siguen resolviendo
    assert len(part["evaluaciondocente"]) == 31 and "figri" in part["calidaddocentes"]


@pytest.mark.skipif(not ETIQ.exists(), reason="data/etiquetas_manuales no está disponible")
@pytest.mark.parametrize("pregunta", sorted(ETIQUETAS))
def test_etiquetas_manuales_estan_en_la_taxonomia(pregunta):
    archivo, hoja = ETIQUETAS[pregunta]
    if not (ETIQ / archivo).exists():
        pytest.skip(f"falta {archivo}")
    tax = _hojas("taxonomia.xlsx")["aspectos"]
    validos = {_n(x) for x in tax[tax["pregunta_codigo"] == pregunta]["aspecto_nombre"]}
    d = pd.read_excel(ETIQ / archivo, sheet_name=hoja, dtype=str, keep_default_na=False)
    etiquetas = {_n(x) for x in d["aspecto_final_entrenamiento"].str.strip() if x.strip()}
    assert not etiquetas - validos, sorted(etiquetas - validos)
    assert not validos - etiquetas, f"aspectos sin etiquetas: {sorted(validos - etiquetas)}"
