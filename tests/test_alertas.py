"""
Pruebas de las alertas de la Etapa 1 (src/preparacion/alertas.py, v1.10) con datos inventados.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preparacion.alertas import (  # noqa: E402
    DetectorAlertas, cargar_alertas, cargar_negaciones, detectar_alertas,
)
from preparacion.etapa1 import validar_alertas  # noqa: E402

TERMINOS = ["acoso", "grosero", "irrespetuoso", "ira", "abuso", "abuso de poder",
            "falta de respeto", "humillación", "salud mental"]
NEGACIONES = [  # TERMINO, TIPO, ALCANCE (subconjunto de data/utilities/negaciones.xlsx)
    ("no", "NEGADOR", "POSTERIOR"), ("sin", "NEGADOR", "POSTERIOR"),
    ("falta", "NEGADOR", "POSTERIOR"), ("nunca", "NEGADOR", "CLAUSULA"),
    ("nadie", "NEGADOR", "CLAUSULA"), ("pero", "ADVERSATIVO", "PIVOTE"),
    ("sin embargo", "ADVERSATIVO", "PIVOTE"), ("no obstante", "ADVERSATIVO", "PIVOTE"),
    ("muy", "INTENSIFICADOR", "POSTERIOR"),
]


def escribir_alertas(ruta: Path, terminos=TERMINOS) -> Path:
    df = pd.DataFrame({"TERMINO": terminos})
    df.to_excel(ruta, sheet_name="alertas", index=False)
    return ruta


def escribir_negaciones(ruta: Path) -> Path:
    df = pd.DataFrame(NEGACIONES, columns=["TERMINO", "TIPO", "ALCANCE"])
    df.insert(1, "TERMINO_SIN_TILDE", df["TERMINO"])
    df["NOTA"] = ""
    df.to_excel(ruta, sheet_name="negaciones", index=False)
    return ruta


@pytest.fixture
def detector(tmp_path):
    return DetectorAlertas(cargar_alertas(escribir_alertas(tmp_path / "alertas.xlsx")),
                           cargar_negaciones(escribir_negaciones(tmp_path / "negaciones.xlsx")))


def terminos(detector, texto):
    return [(c["termino"], c["negada"]) for c in detector.detectar(texto)]


# --- Coincidencia --------------------------------------------------------------

def test_palabra_completa_y_sin_importar_mayusculas_ni_tildes(detector):
    assert terminos(detector, "Hubo ACOSO en clase") == [("acoso", False)]
    assert terminos(detector, "Sentí humillacion") == [("humillación", False)]
    assert terminos(detector, "Su mirada y los acosadores") == []     # «ira» y «acoso» dentro de otra palabra


def test_gana_el_termino_mas_largo(detector):
    assert terminos(detector, "Hubo abuso de poder") == [("abuso de poder", False)]
    assert terminos(detector, "Hubo abuso. Y abuso de poder") == [("abuso", False), ("abuso de poder", False)]


def test_texto_coincidente_conserva_el_original(detector):
    [c] = detector.detectar("Fue una Falta de  Respeto total")
    assert c["texto_coincidente"] == "Falta de  Respeto"


def test_vacios_y_nan_no_tienen_alertas(detector):
    assert detector.detectar("") == [] and detector.detectar(None) == []
    assert detector.detectar(float("nan")) == []


# --- Negación ------------------------------------------------------------------

@pytest.mark.parametrize("texto", [
    "Nunca ha sido irrespetuoso con nadie",          # CLAUSULA antes
    "Es irrespetuoso nunca",                          # CLAUSULA después, mismo segmento
    "No es grosero",                                  # POSTERIOR dentro de la ventana
    "no fue muy grosero",                             # ventana de 3 palabras
    "Clase sin acoso",
])
def test_negada(detector, texto):
    assert [n for _, n in terminos(detector, texto)] == [True]


@pytest.mark.parametrize("texto", [
    "No llegó temprano y además fue grosero",         # «no» a más de 3 palabras
    "Nunca llega tarde. Es grosero",                  # el punto corta el segmento
    "Nunca falta, pero es grosero",                   # el adversativo corta el segmento
    "Sin embargo fue grosero",                        # «sin» de «sin embargo» no niega
    "No obstante, fue grosero",                       # «no» de «no obstante» no niega
    "Es una falta de respeto",                        # «falta» es parte del término
])
def test_no_negada(detector, texto):
    assert [n for _, n in terminos(detector, texto)] == [False]


# --- Carga de insumos ----------------------------------------------------------

def test_terminos_repetidos_tras_normalizar_se_cargan_una_vez(tmp_path):
    ruta = escribir_alertas(tmp_path / "a.xlsx", ["grosería", "groseria", "Grosería "])
    assert len(cargar_alertas(ruta)) == 1


def test_otras_columnas_de_la_lista_se_ignoran(tmp_path):
    ruta = tmp_path / "a.xlsx"
    pd.DataFrame({"TERMINO": ["acoso"], "CATEGORIA": ["x"]}).to_excel(ruta, index=False)
    [t] = cargar_alertas(ruta)
    assert t.termino == "acoso" and t.palabras == ("acoso",)


def test_alertas_sin_columna_termino_da_error_claro(tmp_path):
    ruta = tmp_path / "a.xlsx"
    pd.DataFrame({"palabra": ["acoso"]}).to_excel(ruta, index=False)
    with pytest.raises(ValueError, match="TERMINO"):
        cargar_alertas(ruta)


# --- Tabla de respuestas -------------------------------------------------------

def respuestas_ejemplo() -> pd.DataFrame:
    return pd.DataFrame({
        "respuesta_id": ["e|2025-1|1|p", "e|2025-1|2|p", "e|2025-1|3|p", "e|2025-1|4|p"],
        "texto_original": ["Hubo acoso y fue grosero", "Nunca fue grosero", "Explica bien", "acoso"],
        "es_no_respuesta": [False, False, False, True],   # la 4 es no-respuesta a propósito
    })


def test_detectar_alertas_marca_y_numera(detector):
    r = respuestas_ejemplo()
    tiene, alertas = detectar_alertas(r, detector)
    assert tiene.tolist() == [True, True, False, False]   # la negada también cuenta; la no-respuesta no
    assert alertas["alerta_id"].tolist() == ["e|2025-1|1|p|1", "e|2025-1|1|p|2", "e|2025-1|2|p|1"]
    assert alertas["negada"].tolist() == [False, False, True]
    assert list(alertas.columns) == ["alerta_id", "respuesta_id", "termino", "texto_coincidente",
                                     "negada"]


def test_validar_alertas(detector):
    r = respuestas_ejemplo()
    tiene, alertas = detectar_alertas(r, detector)
    r["tiene_alerta"] = tiene
    assert validar_alertas(r, alertas) == []
    r.loc[3, "tiene_alerta"] = True
    assert any("no-respuestas" in e for e in validar_alertas(r, alertas))


def test_tabla_vacia_tiene_las_columnas(detector):
    r = respuestas_ejemplo().iloc[2:3]
    _, alertas = detectar_alertas(r, detector)
    assert alertas.empty and "negada" in alertas.columns


# --- Ejecución completa ----------------------------------------------------------

def _ejecutar_con_alertas(tmp_path, terminos=TERMINOS):
    from test_preparacion import P20, _configuracion, base_ejemplo
    from preparacion.etapa1 import ejecutar_etapa1
    config = _configuracion(tmp_path)
    df = base_ejemplo()
    df.loc[2, P20] = "Fue grosero"          # 555002 / docente 27
    df.loc[3, P20] = "Nunca fue grosero"    # 555002 / docente 31: negada
    df.to_excel(tmp_path / "raw" / "base.xlsx", sheet_name="EvaDoc", index=False)
    return ejecutar_etapa1(config=config, carpeta_cruda=tmp_path / "raw",
                           carpeta_salida=tmp_path / "salida",
                           alertas=escribir_alertas(tmp_path / f"al_{len(terminos)}.xlsx", terminos),
                           negaciones=escribir_negaciones(tmp_path / "ng.xlsx"))


def test_ejecucion_escribe_alertas_y_metadatos(tmp_path):
    tablas = _ejecutar_con_alertas(tmp_path)
    r = tablas["respuestas"]
    columnas = list(r.columns)
    assert columnas[columnas.index("es_no_respuesta") + 1] == "tiene_alerta"
    assert int(r["tiene_alerta"].sum()) == 2
    m = tablas["metadatos_preparacion"].set_index("pregunta_codigo")
    assert m.loc["obs20", ["respuestas_con_alerta", "alertas",
                                                "alertas_negadas"]].tolist() == [2, 2, 1]
    assert m.loc["obs19", "alertas"] == 0
    assert m["huella_alertas"].str.len().eq(10).all()
    carpeta = tmp_path / "salida" / r["id_corrida_preparacion"].iloc[0]
    leido = pd.read_csv(carpeta / "alertas.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert sorted(leido["negada"]) == ["False", "True"]


def test_cambiar_la_lista_de_alertas_cambia_la_huella_de_la_corrida(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = _ejecutar_con_alertas(tmp_path / "a")
    b = _ejecutar_con_alertas(tmp_path / "b", TERMINOS + ["agresivo"])
    huella = lambda t: t["metadatos_preparacion"]["id_corrida_preparacion"].iloc[0].split("_")[1]  # noqa: E731
    assert huella(a) != huella(b)
