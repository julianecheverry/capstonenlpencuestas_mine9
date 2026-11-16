"""
Pruebas de la Etapa 3 con listas y cláusulas inventadas (nunca datos del cliente).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from limpieza.etapa3 import ejecutar_limpieza  # noqa: E402
from limpieza.listas import cargar_listas  # noqa: E402
from limpieza.textos import limpiar, ngramas, tokenizar  # noqa: E402

EV = "evadoc_evaluacion|2024-2|555001-27|obs19_aspectos_resaltar"


def _utilidades(carpeta: Path, quitar_hoja: str | None = None, base=("de", "la", "las", "los", "más",
                                                                      "también", "no", "y", "el", "poco", "muy")):
    carpeta.mkdir(parents=True, exist_ok=True)
    hojas = {
        "base": pd.DataFrame({"encuesta": "base", "palabra": list(base)}),
        "globales": pd.DataFrame({"encuesta": "stopwords_global", "palabra": ["sin embargo", "por favor", "nada"]}),
        "particulares": pd.DataFrame({"encuesta": "stopwords_evaluaciondocente",
                                      "palabra": ["Profesor", "docente"]}),
        "adverbios": pd.DataFrame({"encuesta": "adverbios", "palabra": ["bien", "muy", "siempre", "ahora"],
                                   "valora": ["si", "no", "si", "no"]}),
    }
    with pd.ExcelWriter(carpeta / "stopwords.xlsx") as w:
        for nombre, df in hojas.items():
            if nombre != quitar_hoja:
                df.to_excel(w, sheet_name=nombre, index=False)
    with pd.ExcelWriter(carpeta / "negaciones.xlsx") as w:
        pd.DataFrame({"TERMINO": ["no", "nunca", "sin", "nada", "poco", "pero", "muy"],
                      "TIPO": ["NEGADOR", "NEGADOR", "NEGADOR", "NEGADOR", "CUANTIFICADOR_BAJO",
                               "ADVERSATIVO", "INTENSIFICADOR"]}).to_excel(w, sheet_name="negaciones", index=False)
    return carpeta


def _nlp(carpeta: Path, textos: list[str], opinion: list[bool] | None = None):
    carpeta.mkdir(parents=True, exist_ok=True)
    opinion = opinion or [True] * len(textos)
    pd.DataFrame({
        "clausula_id": [f"{EV}|{i + 1}" for i in range(len(textos))],
        "respuesta_id": EV, "orden": [str(i + 1) for i in range(len(textos))],
        "texto_clausula": textos, "es_opinion": [str(o) for o in opinion],
    }).to_csv(carpeta / "clausulas.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"id_corrida_nlp": "20261115-000000_prueba"}]).to_csv(
        carpeta / "metadatos_nlp.csv", index=False, encoding="utf-8-sig")
    return carpeta


def _correr(tmp_path, textos, opinion=None, **kw):
    return ejecutar_limpieza(carpeta_nlp=_nlp(tmp_path / "nlp", textos, opinion), carpeta_salida=None,
                             utilidades=_utilidades(tmp_path / "util", **kw))


def _texto(tablas, i=0):
    return tablas["clausulas_texto"]["texto_sin_stopwords"].iloc[i]


def test_negacion_se_conserva(tmp_path):
    t = _correr(tmp_path, ["No explica bien"])
    assert _texto(t) == "no explica bien"
    assert "no explica" in set(t["ngramas"]["ngrama"])


def test_stopword_de_varias_palabras(tmp_path):
    t = _correr(tmp_path, ["Sin embargo, explica; por favor más ejemplos"])
    assert _texto(t) == "explica ejemplos"
    assert not t["ngramas"]["ngrama"].str.contains("sin embargo|por favor").any()


def test_sin_suelto_es_negacion(tmp_path):
    assert _texto(_correr(tmp_path, ["Clases sin ejemplos"])) == "clases sin ejemplos"


def test_stopword_con_tilde(tmp_path):
    assert _texto(_correr(tmp_path, ["Más ejemplos también"])) == "ejemplos"


def test_stopword_particular_de_la_encuesta(tmp_path):
    assert _texto(_correr(tmp_path, ["El profesor explica"])) == "explica"


def test_lista_vacia_o_faltante_detiene(tmp_path):
    with pytest.raises(ValueError, match="particulares"):
        _correr(tmp_path, ["Explica"], quitar_hoja="particulares")


def test_negacion_nunca_es_stopword(tmp_path):
    listas = cargar_listas(*(_utilidades(tmp_path / "u") / f for f in
                             ("stopwords.xlsx", "negaciones.xlsx")))
    assert "no" not in listas.palabras_comunes and "nada" not in listas.palabras_comunes
    assert "poco" not in listas.palabras_comunes          # cuantificador bajo: protegido
    assert "muy" in listas.palabras_comunes               # intensificador: no se protege
    assert "pero" not in listas.protegidas


def test_adverbio_que_valora_queda_en_nubes(tmp_path):
    t = _correr(tmp_path, ["Siempre explica muy bien ahora"])
    fila = t["clausulas_texto"].iloc[0]
    assert fila["texto_sin_stopwords"] == "siempre explica bien ahora"   # «muy» es stopword base
    assert fila["texto_nubes"] == "siempre explica bien"


def test_ngramas_y_frecuencia(tmp_path):
    t = _correr(tmp_path, ["Explica explica claro"])
    ng = t["ngramas"].set_index(["n", "ngrama"])
    assert ng.loc[(1, "explica"), "frecuencia"] == 2
    assert ng.loc[(2, "explica claro"), "frecuencia"] == 1
    assert ng.loc[(3, "explica explica claro"), "frecuencia"] == 1
    assert set(t["ngramas"]["n"]) == {1, 2, 3}


def test_no_suelto_no_va_a_las_nubes(tmp_path):
    ng = _correr(tmp_path, ["No explica"])["ngramas"].set_index("ngrama")
    assert not ng.loc["no", "es_nube"]
    assert ng.loc["no explica", "es_nube"]


def test_ngrama_mostrar_toma_la_forma_con_tildes(tmp_path):
    ng = _correr(tmp_path, ["Excelente metodología", "excelente metodologia", "Excelente Metodología"])["ngramas"]
    assert set(ng.loc[ng["ngrama"] == "metodologia", "ngrama_mostrar"]) == {"metodología"}


def test_solo_clausulas_con_opinion(tmp_path):
    t = _correr(tmp_path, ["Buenas tardes", "Explica"], opinion=[False, True])
    assert list(t["clausulas_texto"]["clausula_id"]) == [f"{EV}|2"]
    assert set(t["ngramas"]["clausula_id"]) == {f"{EV}|2"}


def test_tokenizar_conserva_la_forma_original():
    assert tokenizar("¡Excelente, metodología!") == [("excelente", "excelente"), ("metodologia", "metodología")]


def test_limpiar_quita_frases_antes_que_palabras():
    tokens = tokenizar("sin embargo sin ejemplos")
    assert [t[0] for t in limpiar(tokens, frozenset({"de"}), frozenset({("sin", "embargo")}))] == ["sin", "ejemplos"]
    assert ngramas(tokenizar("a b"))[(2, "a b", "a b")] == 1


def test_salida_estable(tmp_path):
    a = _correr(tmp_path / "a", ["No explica bien", "Excelente metodología"])
    b = _correr(tmp_path / "b", ["No explica bien", "Excelente metodología"])
    pd.testing.assert_frame_equal(a["ngramas"], b["ngramas"])
    pd.testing.assert_frame_equal(a["clausulas_texto"], b["clausulas_texto"])


def test_cuantificador_bajo_se_conserva(tmp_path):
    t = _correr(tmp_path, ["Muy poco claro"])
    assert _texto(t) == "poco claro"
    ng = t["ngramas"].set_index("ngrama")
    assert ng.loc["poco claro", "es_nube"] and not ng.loc["poco", "es_nube"]


def test_lexico_con_tipo_desconocido_detiene(tmp_path):
    util = _utilidades(tmp_path / "util")
    with pd.ExcelWriter(util / "negaciones.xlsx") as w:
        pd.DataFrame({"TERMINO": ["no"], "TIPO": ["OTRO"]}).to_excel(w, sheet_name="negaciones", index=False)
    with pytest.raises(ValueError, match="TIPO desconocidos"):
        cargar_listas(util / "stopwords.xlsx", util / "negaciones.xlsx")
