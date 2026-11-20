"""Punto 2 (NLP etapa 2): las salidas internas de modelado van a data/processed/nlp/modelado, no a outputs/.

No imprime textos de clientes; la exportación se prueba con un Resultado sintético mínimo en un directorio temporal.
"""
import json
import re
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

ft = pytest.importorskip("nlp_utilidades.modulos_nlp")
sys.path.insert(0, str(RAIZ / "tests"))
from test_nlp_rutas import HAY_CORRIDA, _celdas, _ejecutar_configuracion  # noqa: E402

MODELADO = Path("data/processed/nlp/modelado")


def test_config_por_defecto_no_escribe_en_outputs():
    assert ft.Config().output_dir == MODELADO
    assert "outputs" not in ft.Config().output_dir.parts


def test_sin_rutas_outputs_en_modulo_ni_cuaderno():
    codigo = (RAIZ / "src" / "nlp_utilidades" / "modulos_nlp.py").read_text(encoding="utf-8")
    assert not re.search(r"outputs/|[\"']outputs[\"']", codigo)
    for i, c in enumerate(_celdas()):
        if c.cell_type == "code":
            assert not re.search(r"[\"']outputs[\"']|outputs/|/outputs\b", c.source), f"celda {i}"
        else:
            assert "`outputs/" not in c.source, f"celda {i}"


@pytest.mark.skipif(not HAY_CORRIDA, reason="no hay corridas de la Etapa 1")
def test_celda_de_rutas_apunta_a_nlp_modelado():
    ns = _ejecutar_configuracion("obs19_aspectos_resaltar")
    salida, nlp = ns["SALIDA"], ns["RUTAS"]["nlp"]
    assert salida == RAIZ / MODELADO and nlp == RAIZ / "data" / "processed" / "nlp"
    assert salida.exists() and salida.parent == nlp and ns["CACHE"] == RAIZ / "models" / "cache" / "embeddings_ft"
    assert ns["RUTAS"]["salida"] == salida
    # NLP y modelado cuelgan de data/processed (ignorado por git)
    assert "data/processed/" in (RAIZ / ".gitignore").read_text(encoding="utf-8")


def _resultado_sintetico(carpeta_base: Path):
    cl = pd.DataFrame({"doc_id": [0, 1], "clausula_id": [0, 0], "topico": [0, -1], "sentimiento": ["POS", "NEG"],
                       "aspecto": ["Claridad", "Claridad"]})
    vacio = pd.DataFrame()
    return SimpleNamespace(
        cfg=SimpleNamespace(output_dir=carpeta_base, to_dict=lambda: {}), encuesta="evadoc_evaluacion :: obs19_aspectos_resaltar",
        run_id="20261119-000000_sintetico", clausulas=cl, documentos=pd.DataFrame({"doc_id": [0, 1]}), topicos=pd.DataFrame({"topico": [0]}),
        absa=pd.DataFrame({"x": [1]}), grid=vacio, modo=ft.MODO_BERTOPIC, huellas={}, metricas={}, validacion={},
        scorecard={"puertas": [], "nota": 5.0}, conciliacion=pd.DataFrame({"chequeo": ["a"], "valor": [1]}),
        muestra_pureza=vacio, muestra_manual=vacio, u4={}, topicos_residuales=vacio,
    )


def test_exportar_resultado_escribe_bajo_modelado(tmp_path):
    base = tmp_path / MODELADO
    destino = ft.exportar_resultado(_resultado_sintetico(base))
    assert destino == base / ft.slug("evadoc_evaluacion :: obs19_aspectos_resaltar") / "20261119-000000_sintetico"
    assert {"clausulas.csv", "manifest.json", "scorecard.json", "auditoria.xlsx"} <= {f.name for f in destino.iterdir()}
    assert not (tmp_path / "outputs").exists()
    # el glob que usan los Pasos 15-16 sigue funcionando apuntando a .../modelado
    assert [m.parent for m in base.glob("*/*/manifest.json")] == [destino]


def test_exportar_resultado_usa_el_default_de_config_si_no_se_da_raiz(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # el default es relativo al directorio de trabajo
    destino = ft.exportar_resultado(_resultado_sintetico(ft.Config().output_dir))
    assert MODELADO in destino.parents and (tmp_path / destino / "manifest.json").exists()
    assert not (tmp_path / "outputs").exists()


def test_comprimir_carpeta_no_se_incluye_a_si_mismo_ni_a_otros_zip(tmp_path):
    salida = tmp_path / "modelado"
    (salida / "enc" / "run1").mkdir(parents=True)
    (salida / "enc" / "run1" / "manifest.json").write_text(json.dumps({"ok": 1}), encoding="utf-8")
    (salida / "auditoria_ingesta_etapa1.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    anterior = ft.comprimir_carpeta(salida, salida / "outputs_full.zip")
    nuevo = ft.comprimir_carpeta(salida, salida / "outputs_full_recalificado.zip")  # el zip anterior no entra
    for z in (anterior, nuevo):
        with zipfile.ZipFile(z) as zf:
            assert sorted(zf.namelist()) == ["auditoria_ingesta_etapa1.csv", "enc/run1/manifest.json"]
            assert zf.testzip() is None
    # relleno en otra carpeta (Colab: /content): la ruta de destino se crea si no existe
    assert ft.comprimir_carpeta(salida, tmp_path / "otra" / "x.zip").exists()
