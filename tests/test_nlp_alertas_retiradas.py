"""Punto 1 (NLP etapa 2): las alertas pasaron a la Etapa 1 v1.10 y ya no existen dentro del NLP.

Pruebas de que no quedan restos del código de alertas (módulo, salidas y cuaderno) y de que el filtro de entrada
es es_no_respuesta = False y tiene_alerta = False.
"""
import json
import re
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

ft = pytest.importorskip("nlp_utilidades.modulos_nlp")
from nlp_utilidades import salidas_contrato as sc  # noqa: E402

RESTOS = ("palabra_alerta", "MOTIVO_ALERTA", "DOC_ALERTA", "detectar_alerta", "_leer_alertas", "leer_alertas",
          "detectar_alertas", "leer_negadores", "huella_alertas", "n_alertas", "alerta_negacion", "alerta_reemplaza_aspecto")


@pytest.mark.parametrize("ruta", ["src/nlp_utilidades/modulos_nlp.py", "src/nlp_utilidades/salidas_contrato.py"])
def test_sin_restos_de_alertas_en_el_codigo(ruta):
    codigo = (RAIZ / ruta).read_text(encoding="utf-8")
    sin_comentarios = re.sub(r"#.*", "", codigo)
    encontrados = [r for r in RESTOS if r in sin_comentarios]
    assert not encontrados, encontrados


def test_api_publica_sin_alertas():
    assert not hasattr(ft, "ALERTA") and not hasattr(ft, "MOTIVO_ALERTA") and not hasattr(ft, "DOC_ALERTA")
    assert -4 not in ft.ETIQUETAS_ESPECIALES
    assert not hasattr(ft.RecursosLexicos, "detectar_alerta") and "alertas" not in ft.RecursosLexicos.__dataclass_fields__
    assert not hasattr(sc, "leer_alertas") and "alertas" not in sc.ARCHIVOS
    campos = sc.ParametrosSalidas.__dataclass_fields__
    assert not [c for c in campos if "alerta" in c or "negacion" in c]
    assert "tiene_alerta" in ft.COLUMNAS_RESPUESTAS_ETAPA1


def test_recursos_lexicos_cargan_sin_alertas_xlsx(tmp_path):
    """El NLP ya no lee alertas.xlsx: la huella de los léxicos no depende de él."""
    recursos = ft.RecursosLexicos.cargar(RAIZ / "data" / "utilities")
    assert recursos.huella() and len(recursos.stopwords_globales) > 0


def test_el_cuaderno_no_usa_alertas_del_nlp():
    nb = json.loads((RAIZ / "notebooks" / "Prisma_unificado.ipynb").read_text(encoding="utf-8"))
    patrones = [r"leer_alertas", r"detectar_alertas", r"leer_negadores", r"huella_alertas", r"n_alertas", r"palabra_alerta",
                r"MOTIVO_ALERTA", r"DOC_ALERTA", r"alerta_negacion", r"alerta_reemplaza", r"salidas\.alertas", r"alertas\.xlsx"]
    malas = {i: p for i, c in enumerate(nb["cells"]) for p in patrones if re.search(p, "".join(c["source"]))}
    assert not malas, malas
