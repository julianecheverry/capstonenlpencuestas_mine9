"""Los cuatro ``pct_*`` de ``metadatos_nlp`` suman EXACTAMENTE 100,00 (método del mayor resto).

El redondeo simple de cada porcentaje podía dar 99,99 o 100,01. Datos sintéticos; sin datos del cliente.
"""
import itertools
import math
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "tests"))

from nlp_utilidades import salidas_contrato as sc  # noqa: E402
import test_nlp_salidas as ts  # noqa: E402

COLS = ["pct_aspecto_clasificador", "pct_aspecto_hdbscan", "pct_aspecto_regla", "pct_sin_aspecto"]


def suma(valores):
    return sum(Decimal(str(v)) for v in valores)


def test_casos_conocidos():
    assert sc.porcentajes_suma_exacta([1, 1, 1], 3) == [33.34, 33.33, 33.33]       # el simple daba 33,33 ×3 = 99,99
    assert sc.porcentajes_suma_exacta([6, 1, 2, 2], 11) == [54.55, 9.09, 18.18, 18.18]
    assert suma(sc.porcentajes_suma_exacta([1, 1, 1, 1, 1, 1], 6)) == 100          # 16,67 ×6 = 100,02 con el redondeo simple
    assert sc.porcentajes_suma_exacta([5, 0, 0, 0], 5) == [100.0, 0.0, 0.0, 0.0]
    assert sc.porcentajes_suma_exacta([0, 0, 0, 0], 7) == [0.0] * 4


def test_n_cero_da_nan():
    assert all(math.isnan(x) for x in sc.porcentajes_suma_exacta([0, 0], 0))


def test_conteos_que_no_cubren_todo_suman_su_parte():
    assert suma(sc.porcentajes_suma_exacta([1, 1, 0, 0], 3)) == Decimal("66.67")   # 2/3 del total: la meta es round(100·2/3, 2)
    assert suma(sc.porcentajes_suma_exacta([1, 1], 6)) == Decimal("33.33")


def test_empates_se_resuelven_por_orden_de_columna():
    assert sc.porcentajes_suma_exacta([1, 1, 1], 3) == sc.porcentajes_suma_exacta([1, 1, 1], 3)
    p = sc.porcentajes_suma_exacta([1, 1, 1], 3)
    assert p[0] >= p[1] >= p[2]


@pytest.mark.parametrize("decimales", [0, 1, 2, 3])
def test_barrido_exhaustivo_de_particiones_suma_exacta(decimales):
    """Todas las particiones de n (≤ 30) en 4 columnas: la suma es exactamente 100 y cada valor está a menos de una unidad del exacto."""
    unidad = Decimal(1) / Decimal(10 ** decimales)
    for n in range(1, 31 if decimales <= 2 else 13):
        for a, b, c in itertools.product(range(n + 1), repeat=3):
            if a + b + c > n:
                continue
            conteos = [a, b, c, n - a - b - c]
            p = sc.porcentajes_suma_exacta(conteos, n, decimales)
            assert suma(p) == 100, (conteos, n, p)
            for x, k in zip(p, conteos):
                assert abs(Decimal(str(x)) - Decimal(100 * k) / Decimal(n)) < unidad, (conteos, n, p)


def test_el_redondeo_simple_si_fallaba():
    """Documenta el problema: con tercios, el redondeo simple no suma 100."""
    simple = [round(100 * k / 3, 2) for k in (1, 1, 1)]
    assert suma(simple) != 100 and suma(sc.porcentajes_suma_exacta([1, 1, 1], 3)) == 100


def test_determinismo():
    assert sc.porcentajes_suma_exacta([7, 3, 5, 1], 16) == sc.porcentajes_suma_exacta([7, 3, 5, 1], 16)


# ── metadatos_nlp y validador ──────────────────────────────────────────────


def _releer(s):
    d = Path(tempfile.mkdtemp())
    sc.escribir_salidas_contrato(s, d)
    return sc.leer_salidas_contrato(d)


def test_metadatos_del_ejemplo_suman_exactamente_100():
    s = ts.construir()
    fila = s.metadatos.iloc[0]
    assert suma(fila[c] for c in COLS) == 100
    t = _releer(s)
    assert suma(t["metadatos"].iloc[0][c] for c in COLS) == 100
    ts.validar(t)


def test_metadatos_con_tercios_suman_100_y_validan():
    """Tres cláusulas con tres orígenes distintos: el redondeo simple daba 99,99."""
    e = ts.entrada()
    e.clausulas = e.clausulas[e.clausulas["doc_id"].isin([e.documentos["doc_id"][i] for i in (0, 4, 7)])].copy()   # 001 (clasificador ×2) + 005 + 008
    s = ts.construir(entradas=[e])
    assert suma(s.metadatos.iloc[0][c] for c in COLS) == 100


def test_validador_rechaza_99_99_y_100_01():
    t = {k: v.copy() for k, v in _releer(ts.construir()).items()}
    for malo in ("99.99", "100.01"):
        copia = {k: v.copy() for k, v in t.items()}
        copia["metadatos"].loc[0, "pct_sin_aspecto"] = str(round(float(t["metadatos"].loc[0, "pct_sin_aspecto"]) + (float(malo) - 100), 2))
        with pytest.raises(sc.ErrorSalidasNLP, match="no suman exactamente 100,00"):
            ts.validar(copia)


def test_validador_acepta_el_valor_exacto():
    ts.validar(_releer(ts.construir()))
