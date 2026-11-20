"""Punto 8 (NLP etapa 2): Excel de métricas de U4 sobre el hold-out (``_test``) y sobre el entrenamiento (``_train``).

Sin torch: se prueban las piezas puras (escritor del Excel, tabla de predicciones, métricas) con datos sintéticos y el añadido de
diagnósticos de ``etiquetar_corpus_u4`` con un U4 simulado. ``entrenar_u4`` completo exige GPU y se prueba en Colab.
"""
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "tests"))

ft = pytest.importorskip("nlp_utilidades.modulos_nlp")
import test_nlp_opinion_general as og  # noqa: E402

CLASES = ["Claridad", "Puntualidad", "Opinión general"]
HOJAS_COMUNES = ["resumen", "recall_por_clase", "confusion_conteos", "confusion_normalizada", "predicciones", "datos_usados",
                 "errores_frecuentes", "config"]
ANTIGUO = "entrenamiento_u4.xlsx"


def datos_sinteticos(n=12):
    """DatosU4 mínimo: n fragmentos (8 entrenamiento, 2 validación interna, 2 hold-out) de 3 clases."""
    y = np.array([i % 3 for i in range(n)])
    roles = ["entrenamiento"] * 8 + ["validacion_interna"] * 2 + ["holdout"] * 2
    tabla = pd.DataFrame({
        "id": range(n), "revisor": "r", "respuesta_completa": [f"respuesta {i}" for i in range(n)],
        "texto": [f"fragmento {i}" for i in range(n)], "etiqueta": [CLASES[j] for j in y], "peso": 1.0, "tipo_fila": "limpia",
        "motivo": "", "grupo": range(n), "rol": roles, "y": y, "usa_evaluacion": True,
    })
    informe = {"archivo": "etiquetas.xlsx", "filas_usadas": n, "filas_descartadas_vacias": 0,
               "fragmentos_con_etiquetas_contradictorias": 0, "variantes_normalizadas": {}}
    return ft.DatosU4(tabla=tabla, clases=CLASES, ruta=Path("etiquetas.xlsx"), md5="abc123", informe=informe), y


def probabilidades(y, acierto):
    """Matriz de probabilidades que acierta con la proporción ``acierto`` de filas (determinista)."""
    P = np.full((len(y), 3), 0.1)
    for i, c in enumerate(y):
        P[i, c if (i % 10) < int(acierto * 10) else (c + 1) % 3] = 0.8
    return P / P.sum(1, keepdims=True)


def escribir_ambos(carpeta):
    datos, y = datos_sinteticos()
    d = datos.tabla
    config = {"clave": "obs19_aspectos_resaltar", "run_id": "r1", "modelo_base": "mpnet", "semilla": 42}
    info = {"mejor_epoca": 3, "epocas_ejecutadas": 6, "f1_macro_validacion_interna": 0.7}
    idx_te = np.flatnonzero(d["rol"] == "holdout")
    idx_tr = np.flatnonzero(d["rol"] == "entrenamiento")
    P_te, P_tr = probabilidades(y[idx_te], 0.0), probabilidades(y[idx_tr], 1.0)
    m_te, m_tr = ft._metricas_holdout(y[idx_te], P_te, CLASES, 1), ft._metricas_holdout(y[idx_tr], P_tr, CLASES, 1)
    pred_te = ft._tabla_predicciones_u4(d, idx_te, y, P_te, CLASES)
    pred_tr = ft._tabla_predicciones_u4(d, idx_tr, y, P_tr, CLASES)
    ft._escribir_excel_entrenamiento(carpeta, datos, m_te, pred_te, config, info)
    ft._escribir_excel_entrenamiento(carpeta, datos, m_tr, pred_tr, config, info, parte="train")
    return m_te, m_tr, pred_te, pred_tr


# ── 8.1: nombre nuevo ──────────────────────────────────────────────────────


def test_nombres_de_los_archivos():
    assert ft.ARCHIVO_ENTRENAMIENTO_TEST == "entrenamiento_u4_test.xlsx"
    assert ft.ARCHIVO_ENTRENAMIENTO_TRAIN == "entrenamiento_u4_train.xlsx"


def test_el_nombre_antiguo_ya_no_se_usa_como_archivo():
    codigo = (RAIZ / "src" / "nlp_utilidades" / "modulos_nlp.py").read_text(encoding="utf-8")
    usos = [l for l in codigo.splitlines() if ANTIGUO in l and not l.lstrip().startswith(("#", '"""')) and "antes" not in l]
    assert not usos, usos
    import json
    nb = json.loads((RAIZ / "notebooks" / "Prisma_unificado.ipynb").read_text(encoding="utf-8"))
    assert not [i for i, c in enumerate(nb["cells"]) if re.search(r"entrenamiento_u4\.xlsx", "".join(c["source"]))]


# ── 8.2: Excel de entrenamiento ────────────────────────────────────────────


def test_escribe_test_y_train_y_no_el_nombre_antiguo(tmp_path):
    escribir_ambos(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["entrenamiento_u4_test.xlsx", "entrenamiento_u4_train.xlsx"]
    assert not (tmp_path / ANTIGUO).exists()


def test_misma_estructura_salvo_el_marcador_de_diagnostico(tmp_path):
    escribir_ambos(tmp_path)
    te = pd.ExcelFile(tmp_path / "entrenamiento_u4_test.xlsx").sheet_names
    tr = pd.ExcelFile(tmp_path / "entrenamiento_u4_train.xlsx").sheet_names
    assert te == HOJAS_COMUNES + ["diagnostico_comentario_general"]
    assert tr == HOJAS_COMUNES
    for hoja in HOJAS_COMUNES:                                                  # mismas columnas hoja por hoja
        a = pd.read_excel(tmp_path / "entrenamiento_u4_test.xlsx", sheet_name=hoja, nrows=1)
        b = pd.read_excel(tmp_path / "entrenamiento_u4_train.xlsx", sheet_name=hoja, nrows=1)
        assert list(a.columns) == list(b.columns), hoja


def test_resumen_de_train_usa_los_campos_train_y_el_de_test_no_cambia(tmp_path):
    m_te, m_tr, *_ = escribir_ambos(tmp_path)
    te = dict(pd.read_excel(tmp_path / "entrenamiento_u4_test.xlsx", sheet_name="resumen").values)
    tr = dict(pd.read_excel(tmp_path / "entrenamiento_u4_train.xlsx", sheet_name="resumen").values)
    assert {"acierto_holdout", "f1_macro_holdout", "f1_macro_holdout_todas_las_clases", "n_holdout_evaluable"} <= set(te)
    assert not any(k.endswith("_train") or "train" in k for k in te)
    assert {"acierto_train", "f1_macro_train", "f1_macro_train_todas_las_clases", "n_train_evaluable"} <= set(tr)
    assert not any("holdout" in k for k in tr)
    assert te["acierto_holdout"] == pytest.approx(m_te["accuracy"]) and tr["acierto_train"] == pytest.approx(m_tr["accuracy"])
    assert tr["n_train_evaluable"] == 8 and te["n_holdout_evaluable"] == 2
    assert "NO miden generalización" in tr["nota"] and "hold-out" in te["nota"] and "NO miden" not in te["nota"]
    assert tr["acierto_train"] > te["acierto_holdout"]                          # el sintético acierta todo en train y nada en test
    for k in ("clave", "run_id", "modelo_base", "md5_auditado", "filas_usadas", "mejor_epoca"):   # el resto del resumen es el mismo
        assert te[k] == tr[k]


def test_las_predicciones_de_train_son_de_las_filas_de_entrenamiento(tmp_path):
    _, _, pred_te, pred_tr = escribir_ambos(tmp_path)
    datos, _ = datos_sinteticos()
    roles = datos.tabla.set_index("id")["rol"]
    assert set(roles[pred_tr["id"]]) == {"entrenamiento"} and set(roles[pred_te["id"]]) == {"holdout"}
    en_excel = pd.read_excel(tmp_path / "entrenamiento_u4_train.xlsx", sheet_name="predicciones")
    assert len(en_excel) == 8 and list(en_excel.columns) == list(pred_tr.columns) and en_excel["correcto"].all()


def test_tabla_de_predicciones():
    datos, y = datos_sinteticos()
    idx = np.array([1, 4, 11])
    P = probabilidades(y[idx], 1.0)
    t = ft._tabla_predicciones_u4(datos.tabla, idx, y, P, CLASES)
    assert list(t.columns) == ["id", "texto", "y_true", "y_pred", "p_max", "correcto", "p_Claridad", "p_Puntualidad", "p_Opinión general"]
    assert list(t["id"]) == [1, 4, 11] and list(t["y_true"]) == [CLASES[j] for j in y[idx]] and t["correcto"].all()
    assert t["p_max"].tolist() == pytest.approx(P.max(1).tolist())
    vacia = ft._tabla_predicciones_u4(datos.tabla, np.array([], dtype=int), y, np.zeros((0, 3)), CLASES)
    assert len(vacia) == 0 and "p_Claridad" in vacia.columns


def test_parte_invalida_falla(tmp_path):
    datos, y = datos_sinteticos()
    m = ft._metricas_holdout(y[:2], probabilidades(y[:2], 1.0), CLASES, 1)
    with pytest.raises(ValueError, match="parte debe ser"):
        ft._escribir_excel_entrenamiento(tmp_path, datos, m, pd.DataFrame(), {}, {}, parte="validacion")


def test_conjunto_de_entrenamiento_vacio_escribe_el_excel_sin_filas(tmp_path):
    datos, y = datos_sinteticos()
    m = ft._metricas_holdout(y[:0], np.zeros((0, 3)), CLASES, 1)
    pred = ft._tabla_predicciones_u4(datos.tabla, np.array([], dtype=int), y, np.zeros((0, 3)), CLASES)
    ft._escribir_excel_entrenamiento(tmp_path, datos, m, pred, {"clave": "x"}, {}, parte="train")
    assert (tmp_path / "entrenamiento_u4_train.xlsx").exists()


# ── los diagnósticos del corpus siguen yendo solo al archivo _test ─────────


def test_etiquetar_corpus_anade_los_diagnosticos_al_test_y_no_al_train(tmp_path, monkeypatch):
    escribir_ambos(tmp_path)
    antes_train = pd.ExcelFile(tmp_path / "entrenamiento_u4_train.xlsx").sheet_names
    monkeypatch.setattr(ft, "etiquetar_clausulas", lambda _m, textos: np.array([og.PROBS[t] for t in textos], dtype=float))
    modelo = SimpleNamespace(clases=og.CLASES, run_id="u4_falso", textos_vistos=set(), holdout=None, config={}, carpeta=tmp_path,
                             clases_especiales=["Opinión general"], metricas={})
    enc, docs, cl = og.corpus()
    cfg = ft.Config(modo_aspectos="u4", umbral_comentario_general=0.8, residual_hdbscan_activo=False)
    ft.etiquetar_corpus_u4(modelo, enc, docs, cl, cfg, escribir_diagnosticos=True)
    hojas_test = pd.ExcelFile(tmp_path / "entrenamiento_u4_test.xlsx").sheet_names
    assert any(h.startswith("diag_cg_") for h in hojas_test) and any(h.startswith("diag_res_") for h in hojas_test)
    assert HOJAS_COMUNES[0] in hojas_test                                       # las hojas de métricas siguen ahí
    assert pd.ExcelFile(tmp_path / "entrenamiento_u4_train.xlsx").sheet_names == antes_train
    assert not (tmp_path / ANTIGUO).exists()


def test_sin_el_archivo_test_no_se_crean_diagnosticos(tmp_path, monkeypatch):
    monkeypatch.setattr(ft, "etiquetar_clausulas", lambda _m, textos: np.array([og.PROBS[t] for t in textos], dtype=float))
    modelo = SimpleNamespace(clases=og.CLASES, run_id="u4_falso", textos_vistos=set(), holdout=None, config={}, carpeta=tmp_path,
                             clases_especiales=["Opinión general"], metricas={})
    enc, docs, cl = og.corpus()
    ft.etiquetar_corpus_u4(modelo, enc, docs, cl, ft.Config(modo_aspectos="u4", umbral_comentario_general=0.8, residual_hdbscan_activo=False),
                           escribir_diagnosticos=True)
    assert list(tmp_path.iterdir()) == []
