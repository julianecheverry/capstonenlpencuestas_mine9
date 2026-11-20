"""Los dos archivos de cláusulas no se confunden: modelado escribe ``clausulas_detalle.csv`` y el contrato ``clausulas.csv``.

``data/processed/nlp/modelado/<encuesta>/<run_id>/clausulas_detalle.csv`` (internas del modelo) frente a
``data/processed/nlp/<id_corrida_nlp>/clausulas.csv`` (tabla del contrato). Datos sintéticos; sin datos del cliente.
"""
import json
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
from test_nlp_salida_modelado import MODELADO, _resultado_sintetico  # noqa: E402
import test_nlp_salidas as ts  # noqa: E402


def test_nombres():
    assert ft.ARCHIVO_CLAUSULAS_DETALLE == "clausulas_detalle.csv" and ft.ARCHIVO_CLAUSULAS_DETALLE != sc.ARCHIVOS["clausulas"]
    assert ft.ARCHIVO_CLAUSULAS_DETALLE_PARQUET == "clausulas_detalle.parquet"


def test_modelado_escribe_clausulas_detalle_y_no_clausulas_csv(tmp_path):
    destino = ft.exportar_resultado(_resultado_sintetico(tmp_path / MODELADO))
    nombres = {f.name for f in destino.iterdir()}
    assert "clausulas_detalle.csv" in nombres and "clausulas.csv" not in nombres
    detalle = pd.read_csv(destino / "clausulas_detalle.csv")
    assert {"doc_id", "clausula_id", "topico", "sentimiento"} <= set(detalle.columns)    # columnas internas, no las del contrato
    assert set(detalle.columns).isdisjoint({"respuesta_id", "texto_clausula", "es_opinion"})


def test_el_contrato_sigue_escribiendo_clausulas_csv(tmp_path):
    s = ts.construir()
    sc.escribir_salidas_contrato(s, tmp_path / "nlp" / s.id_corrida_nlp)
    assert (tmp_path / "nlp" / s.id_corrida_nlp / "clausulas.csv").exists()
    destino = ft.exportar_resultado(_resultado_sintetico(tmp_path / "nlp" / "modelado"))
    nombres = {p.name for p in (tmp_path / "nlp").rglob("clausulas*.csv")}
    assert nombres == {"clausulas.csv", "clausulas_detalle.csv"}
    assert (tmp_path / "nlp" / s.id_corrida_nlp / "clausulas.csv") != destino / "clausulas_detalle.csv"


def test_ruta_con_nombre_legado_para_corridas_antiguas(tmp_path):
    assert ft.ruta_clausulas_detalle(tmp_path) == tmp_path / "clausulas_detalle.csv"            # nada: el nombre nuevo
    (tmp_path / "clausulas.csv").write_text("a\n1\n", encoding="utf-8")
    assert ft.ruta_clausulas_detalle(tmp_path) == tmp_path / "clausulas.csv"                    # solo el legado (corrida antigua)
    (tmp_path / "clausulas_detalle.csv").write_text("a\n1\n", encoding="utf-8")
    assert ft.ruta_clausulas_detalle(tmp_path) == tmp_path / "clausulas_detalle.csv"            # con ambos manda el nuevo


def test_la_recalificacion_lee_corridas_con_el_nombre_legado(tmp_path):
    """El lector de cláusulas (sin pyarrow cae al CSV) funciona con carpetas escritas antes del cambio de nombre."""
    carpeta = tmp_path / "enc" / "run1"
    carpeta.mkdir(parents=True)
    pd.DataFrame({"clausula": ["a", "b"], "sentimiento": ["POS", "NEG"]}).to_csv(carpeta / "clausulas.csv", index=False)
    leido = pd.read_csv(ft.ruta_clausulas_detalle(carpeta), usecols=["clausula", "sentimiento"])
    assert list(leido["sentimiento"]) == ["POS", "NEG"]


def test_cuaderno_y_modulo_no_leen_clausulas_csv_de_modelado():
    nb = json.loads((RAIZ / "notebooks" / "Prisma_unificado.ipynb").read_text(encoding="utf-8"))
    malas = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and re.search(r'CARPETA\s*/\s*"clausulas\.csv"', "".join(c["source"]))]
    assert not malas, malas
    codigo = (RAIZ / "src" / "nlp_utilidades" / "modulos_nlp.py").read_text(encoding="utf-8")
    assert not re.search(r'(carpeta|destino)\s*/\s*"clausulas\.csv"', codigo)


def test_ruta_del_parquet_con_nombre_legado_para_corridas_antiguas(tmp_path):
    assert ft.ruta_clausulas_detalle_parquet(tmp_path) == tmp_path / "clausulas_detalle.parquet"       # nada: el nombre nuevo
    (tmp_path / "clausulas.parquet").write_bytes(b"x")
    assert ft.ruta_clausulas_detalle_parquet(tmp_path) == tmp_path / "clausulas.parquet"               # solo el legado
    (tmp_path / "clausulas_detalle.parquet").write_bytes(b"x")
    assert ft.ruta_clausulas_detalle_parquet(tmp_path) == tmp_path / "clausulas_detalle.parquet"       # con ambos manda el nuevo


def test_modelado_no_escribe_clausulas_parquet(tmp_path):
    """Con pyarrow instalado escribe clausulas_detalle.parquet; sin él se omite. En ningún caso queda clausulas.parquet."""
    destino = ft.exportar_resultado(_resultado_sintetico(tmp_path / MODELADO))
    nombres = {f.name for f in destino.iterdir()}
    assert "clausulas.parquet" not in nombres
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        assert "clausulas_detalle.parquet" not in nombres
    else:
        assert "clausulas_detalle.parquet" in nombres


def test_el_modulo_y_los_cuadernos_no_usan_el_parquet_con_el_nombre_antiguo():
    codigo = (RAIZ / "src" / "nlp_utilidades" / "modulos_nlp.py").read_text(encoding="utf-8")
    usos = [l for l in codigo.splitlines() if re.search(r'"clausulas\.parquet"', l) and "LEGADO" not in l]   # literal de ruta (no docstrings)
    assert not usos, usos
    for nombre in ("Prisma_unificado.ipynb", "Prisma_anexos.ipynb"):
        nb = json.loads((RAIZ / "notebooks" / nombre).read_text(encoding="utf-8"))
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                for linea in "".join(c["source"]).splitlines():
                    if "clausulas.parquet" in linea:
                        assert "clausulas_detalle.parquet" in linea, (nombre, i)   # solo como alternativa de compatibilidad
