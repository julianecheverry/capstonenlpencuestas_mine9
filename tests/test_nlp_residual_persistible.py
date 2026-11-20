"""Tarea 9 (estabilidad): UMAP/HDBSCAN del residual reajustados con prediction_data=True, guardados y reutilizables.

UMAP y HDBSCAN no están instalados en este entorno: se usan dobles que imitan su interfaz, así que se prueba la
lógica propia (mapa de grupos, probabilidades, guardado y asignación de respuestas nuevas) y NO el algoritmo real.
La verificación con las librerías reales queda para Colab.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

ft = pytest.importorskip("nlp_utilidades.modulos_nlp")


def _crudas(x):
    return np.where(x[:, 0] > 0, 0, np.where(x[:, 0] < -1, 1, -1))


class FalsoUMAP:
    def __init__(self, **kw):
        self.kw = kw

    def fit(self, x):
        self.embedding_ = np.asarray(x)[:, :2]
        return self

    def transform(self, x):
        return np.asarray(x)[:, :2]


class FalsoHDBSCAN:
    def __init__(self, **kw):
        self.kw = kw

    def fit(self, x):
        self.labels_ = _crudas(np.asarray(x))
        self.probabilities_ = np.where(self.labels_ >= 0, 0.8, 0.0)
        return self


def _aproximar(modelo, x):
    return _crudas(np.asarray(x)), np.full(len(x), 0.6)


@pytest.fixture()
def libs_falsas(monkeypatch):
    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=FalsoUMAP))
    monkeypatch.setitem(sys.modules, "hdbscan", SimpleNamespace(HDBSCAN=FalsoHDBSCAN, approximate_predict=_aproximar))


def _modelador():
    # 5 textos en el grupo crudo 0, 3 en el 1 y 2 outliers (que la reasignación por centroide manda al tópico 1)
    emb = np.array([[1, 0, 0]] * 5 + [[-2, 0, 0]] * 3 + [[-0.5, 0, 0]] * 2, dtype=float)
    crudas = _crudas(emb)
    finales = np.where(crudas == -1, 1, crudas)
    return SimpleNamespace(embeddings=emb, etiquetas_hdbscan=ft.renumerar(crudas), etiquetas=finales)


def test_ajusta_con_prediction_data_y_mapa_de_grupos(libs_falsas):
    mod = ft.ajustar_residual_persistible(_modelador(), ft.Config())
    assert mod.mapa_final == {0: 0, 1: 1}
    assert mod.semilla == 42 and mod.n_textos == 10 and mod.version == ft.VERSION
    assert mod.hdbscan.kw["prediction_data"] is True                    # lo que habilita approximate_predict
    assert mod.umap.kw["random_state"] == 42 and mod.umap.kw["metric"] == "cosine"   # misma semilla/parámetros que el grid
    # pertenencia al grupo: NaN en los outliers de HDBSCAN que se reasignaron por centroide
    assert np.isnan(mod.probabilidades[-2:]).all() and (mod.probabilidades[:8] == 0.8).all()


def test_asigna_respuestas_nuevas_sin_reajustar(libs_falsas):
    mod = ft.ajustar_residual_persistible(_modelador(), ft.Config())
    nuevos = np.array([[3, 0, 0], [-4, 0, 0], [-0.1, 0, 0]], dtype=float)
    topicos, probs = mod.asignar(nuevos)
    assert topicos.tolist() == [0, 1, -1] and probs.tolist() == [0.6, 0.6, 0.6]


def test_guardar_y_cargar(libs_falsas, tmp_path):
    mod = ft.ajustar_residual_persistible(_modelador(), ft.Config())
    ruta = ft.guardar_modelo_residual(mod, tmp_path / "residual")
    assert ruta.name == "residual_umap_hdbscan.joblib" and ruta.exists()
    cargado = ft.cargar_modelo_residual(tmp_path / "residual")
    assert cargado.mapa_final == mod.mapa_final and cargado.config == mod.config and cargado.semilla == 42
    assert cargado.asignar(np.array([[3, 0, 0]], dtype=float))[0].tolist() == [0]


def test_avisa_si_el_reajuste_no_reproduce_la_particion(libs_falsas, caplog):
    m = _modelador()
    m.etiquetas_hdbscan = np.zeros(10, dtype=int)  # partición "original" distinta de la que da el reajuste
    with caplog.at_level("WARNING"):
        ft.ajustar_residual_persistible(m, ft.Config())
    assert "no reproduce la partición" in caplog.text


def test_etapas_u4_tiene_el_campo_persistible():
    assert "residual_persistible" in ft.EtapasU4.__dataclass_fields__
