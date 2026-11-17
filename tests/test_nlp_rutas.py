"""Fase 2 (NLP etapa 2): rutas del cuaderno Prisma_unificado.ipynb y de modulos_nlp.py.

Ejecuta SOLO las celdas de configuración y rutas del cuaderno (con EN_COLAB = False); no entrena ni carga modelos.
Las celdas se localizan por su contenido, no por su número. No imprime textos de clientes.
"""
import ast
import re
import sys
from pathlib import Path

import nbformat
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

CUADERNO = RAIZ / "notebooks" / "Prisma_unificado.ipynb"
ETIQ = RAIZ / "data" / "etiquetas_manuales"
CLAVES = [
    "obs19_aspectos_resaltar", "obs20_oportunidades_mejora", "obs23_percepcion_progreso",
    "obs24_aspectos_resaltar", "obs25_oportunidades_mejora", "acred_sugerencias_complementarias",
]
OBSOLETO = re.compile(r"finetuning_pruebas|datacleaning|Mod_Bertopic_Ft|topicos-unificado")


def _celdas():
    return nbformat.read(CUADERNO, as_version=4).cells


def _codigo(marcador: str) -> str:
    hallazgos = [c.source for c in _celdas() if c.cell_type == "code" and marcador in c.source]
    assert len(hallazgos) == 1, (marcador, len(hallazgos))
    return "\n".join(l for l in hallazgos[0].splitlines() if not l.lstrip().startswith(("%", "!")))


def _ejecutar_configuracion(clave: str) -> dict:
    ns: dict = {"__name__": "cuaderno"}
    exec("import sys", ns)  # lo importa la celda de entorno, anterior a la de rutas
    exec(_codigo("# ══ CONFIG ÚNICA").replace('CLAVE_FOCAL = "obs19_aspectos_resaltar"', f'CLAVE_FOCAL = "{clave}"', 1), ns)
    ns["EN_COLAB"] = False
    exec(_codigo("# ══ RUTAS"), ns)
    exec(_codigo('assert RUTAS["modulo"].exists()'), ns)  # celda que importa el módulo
    exec(_codigo('ENCODER = "potente"'), ns)
    return ns


def test_cuaderno_sin_referencias_a_rutas_viejas():
    for i, celda in enumerate(_celdas()):
        assert not OBSOLETO.search(celda.source), f"celda {i}: {OBSOLETO.search(celda.source).group(0)}"


def test_celdas_de_codigo_compilan():
    for i, c in enumerate(_celdas()):
        if c.cell_type == "code":
            lineas = [l[: len(l) - len(l.lstrip())] + "pass" if l.lstrip().startswith(("%", "!")) else l
                      for l in c.source.splitlines()]
            ast.parse("\n".join(lineas))


@pytest.mark.skipif(not ETIQ.exists(), reason="data/etiquetas_manuales no está disponible")
@pytest.mark.parametrize("clave", CLAVES)
def test_rutas_existen_y_etiquetas_se_resuelven(clave):
    ns = _ejecutar_configuracion(clave)
    ft = ns["ft"]
    for nombre, ruta in ns["RUTAS"].items():
        assert ruta.exists(), nombre
    assert ns["RUTAS"]["utilities"] == RAIZ / "data" / "utilities"
    assert ns["RAIZ_MODELOS"] == RAIZ / "models"
    p = ft.ParametrosU4.desde(ns["params_u4"](clave), clave)
    archivo = ft.resolver_archivo_etiquetas(ns["RAIZ_MODELOS"], p, clave)  # raiz = RAIZ_MODELOS, como en el cuaderno
    assert archivo.exists() and archivo.parent == ns["RUTAS"]["etiquetas"]
    assert ft.carpeta_modelos_u4(ns["RAIZ_MODELOS"], clave) == RAIZ / "models" / "modelos" / "u4_aspectos" / ft.slug(clave)


def test_col_respuesta_por_pregunta():
    ns = _ejecutar_configuracion("obs19_aspectos_resaltar")
    cols = {k: v["aspectos_u4"]["col_respuesta"] for k, v in ns["PERFILES"].items()}
    assert cols.pop("obs19_aspectos_resaltar") == "respuesta_completa_original"
    assert set(cols.values()) == {"respuesta_completa"}


def test_cargar_taxonomias_hoja_aspectos():
    ft = pytest.importorskip("nlp_utilidades.modulos_nlp")
    tax = ft.cargar_taxonomias(RAIZ / "data" / "utilities")
    assert {k: len(v) for k, v in tax.items()} == {
        "obs19_aspectos_resaltar": 17, "obs20_oportunidades_mejora": 17, "obs23_percepcion_progreso": 31,
        "obs24_aspectos_resaltar": 32, "obs25_oportunidades_mejora": 30, "acred_sugerencias_complementarias": 19,
    }
    assert "VOCACION" in tax["obs24_aspectos_resaltar"]


def test_cargar_taxonomias_formato_antiguo_sigue_funcionando(tmp_path):
    import pandas as pd

    ft = pytest.importorskip("nlp_utilidades.modulos_nlp")
    pd.DataFrame({"ENCUESTA": ["Evaluación Docente"], "CODIGO": ["CLARIDAD"], "DESCRIPCION": ["explica claro"]}).to_excel(
        tmp_path / "taxonomia.xlsx", sheet_name="taxonomia", index=False)
    assert ft.cargar_taxonomias(tmp_path) == {"Evaluación Docente": {"CLARIDAD": "explica claro"}}


def test_ejecutar_pipeline_acepta_rutas_separadas():
    import inspect

    ft = pytest.importorskip("nlp_utilidades.modulos_nlp")
    assert {"dir_lexicos", "raiz_modelos"} <= set(inspect.signature(ft.ejecutar_pipeline).parameters)
    assert ft.ParametrosU4().carpeta_etiquetas == "data/etiquetas_manuales"
    assert str(ft.Config().dir_cache).replace("\\", "/") == "models/cache/embeddings_ft"
