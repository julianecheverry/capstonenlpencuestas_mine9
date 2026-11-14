"""
Pruebas de la Etapa 1 (src/preparacion) con datos inventados.

Reproducen los casos difíciles encontrados en la base real (ver el
"Ejemplo trazado" del contrato de salida): copias por facultad de oferta,
no-respuesta escrita de varias formas, un mismo ID de respuesta en
evaluaciones de docentes distintos, autoevaluación repetida por estudiante y
no-respuesta registrada como "0".
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preparacion.etapa1 import ejecutar_etapa1, validar_salida  # noqa: E402
from preparacion.normalizacion import es_no_respuesta, nombre_columna, normalizar  # noqa: E402
from preparacion.respuestas import ConfigPregunta, codificar, construir_respuestas  # noqa: E402

SAL = "sal-de-prueba"
P19 = "Observaciones - 19. Aspectos para resaltar del docente y/o del espacio académico"
P20 = "Observaciones - 20. Oportunidades de mejora del docente y/o del espacio académico"
P24 = "Observaciones - 24. Aspectos para resaltar de su labor como docente y/o del espacio académico"
CRUCE = "Código Cruce Autoevaluación"


def base_ejemplo() -> pd.DataFrame:
    """Las cuatro filas del Ejemplo trazado (2024-2), más un caso de '0'."""
    return pd.DataFrame([
        # Fila 1 y 2: misma respuesta, una copia por facultad de oferta.
        {"Periodo": "2024-2", "ID de respuesta": "555001", "Id Docente": "27",
         "Facultad de oferta": "Derecho", CRUCE: "X1", "Codigo del Grupo": "G1",
         P19: "Explica muy bien", P20: "NR", P24: "Logro que participen"},
        {"Periodo": "2024-2", "ID de respuesta": "555001", "Id Docente": "27",
         "Facultad de oferta": "Economía", CRUCE: "X2", "Codigo del Grupo": "G1",
         P19: "Explica muy bien", P20: "Ninguna", P24: "Promuevo el debate"},
        # Fila 3: otro estudiante del docente 27.
        {"Periodo": "2024-2", "ID de respuesta": "555002", "Id Docente": "27",
         "Facultad de oferta": "Derecho", CRUCE: "X1", "Codigo del Grupo": "G1",
         P19: "NR", P20: "A veces llega tarde", P24: "Logro que participen"},
        # Fila 4: mismo ID de respuesta, pero es otra evaluación (docente 31).
        {"Periodo": "2024-2", "ID de respuesta": "555002", "Id Docente": "31",
         "Facultad de oferta": "Derecho", CRUCE: "X5", "Codigo del Grupo": "G2",
         P19: "Gran profesor", P20: "NR", P24: "0"},
    ])


def cfg_estudiante(columna_texto: str, codigo: str) -> ConfigPregunta:
    return ConfigPregunta(
        encuesta="evadoc_evaluacion", pregunta_codigo=codigo, columna_texto=columna_texto,
        llaves=["Periodo", "ID de respuesta", "Id Docente"],
        contexto=["ID de respuesta", "Id Docente", CRUCE, "Codigo del Grupo"],
        oferta=["Facultad de oferta", CRUCE], desempate=["Facultad de oferta"],
        codificar=[CRUCE], rol="estudiante",
    )


def cfg_docente(columna_texto: str, codigo: str) -> ConfigPregunta:
    return ConfigPregunta(
        encuesta="evadoc_autoevaluacion", pregunta_codigo=codigo, columna_texto=columna_texto,
        llaves=["Periodo", CRUCE], contexto=["Id Docente", "Codigo del Grupo"],
        oferta=["Facultad de oferta"], desempate=["Facultad de oferta"],
        codificar=[CRUCE], rol="docente",
    )


# --- Normalización y no-respuesta --------------------------------------------

@pytest.mark.parametrize("texto", [
    "NR", "nr", "", None, float("nan"), "Ninguna", "Ninguna.", "NINGUNA", "Ningúno",
    "Nada.", "N/A", "No aplica", "no", "No hay", "no tengo ninguna", "No sé", "0", "  .  ",
])
def test_no_respuesta_reconoce_variantes(texto):
    assert es_no_respuesta(texto)


@pytest.mark.parametrize("texto", ["bien", "Todo bien", "es la mejor", "No explica nada", "10/10"])
def test_comentarios_cortos_genericos_se_conservan(texto):
    assert not es_no_respuesta(texto)


def test_normalizar_y_nombre_columna():
    assert normalizar("  ¡Muy   BUENA clase! ") == "muy buena clase"
    assert nombre_columna(CRUCE) == "codigo_cruce_autoevaluacion"


def test_codificar_es_estable_y_no_expone_el_valor():
    assert codificar("1018426060", SAL) == codificar("1018426060", SAL)
    assert codificar("1018426060", SAL) != codificar("1018426060", "otra-sal")
    assert "1018426060" not in codificar("1018426060", SAL)
    assert len(codificar("X1", SAL)) == 12
    assert codificar("", SAL) == "" and codificar(None, SAL) == ""


# --- Estudiantes ---------------------------------------------------------------

def test_estudiante_una_respuesta_por_llave():
    r = construir_respuestas(base_ejemplo(), cfg_estudiante(P19, "obs19"), SAL)
    assert len(r.respuestas) == 3                      # 555001-27, 555002-27, 555002-31
    assert r.respuestas["respuesta_id"].is_unique
    assert r.conteos["copias_descartadas"] == 1


def test_estudiante_mismo_id_otro_docente_no_se_fusiona():
    r = construir_respuestas(base_ejemplo(), cfg_estudiante(P19, "obs19"), SAL)
    ids = set(r.respuestas["id_origen"])
    assert {"555002-27", "555002-31"} <= ids
    gran = r.respuestas.set_index("id_origen").loc["555002-31"]
    assert gran["texto_original"] == "Gran profesor"
    assert not gran["es_no_respuesta"]


def test_respuesta_id_tiene_el_formato_del_contrato():
    r = construir_respuestas(base_ejemplo(), cfg_estudiante(P19, "obs19"), SAL)
    assert "evadoc_evaluacion|2024-2|555001-27|obs19" in set(r.respuestas["respuesta_id"])


def test_copias_por_facultad_quedan_en_oferta():
    r = construir_respuestas(base_ejemplo(), cfg_estudiante(P19, "obs19"), SAL)
    rid = "evadoc_evaluacion|2024-2|555001-27|obs19"
    facultades = set(r.oferta.loc[r.oferta["respuesta_id"] == rid, "facultad_de_oferta"])
    assert facultades == {"Derecho", "Economía"}
    assert set(r.oferta["respuesta_id"]) == set(r.respuestas["respuesta_id"])


def test_no_respuesta_se_marca_no_se_elimina():
    r = construir_respuestas(base_ejemplo(), cfg_estudiante(P20, "obs20"), SAL)
    fila = r.respuestas.set_index("id_origen").loc["555001-27"]
    assert fila["es_no_respuesta"]
    assert r.conteos["no_respuestas"] == 2             # 555001-27 y 555002-31
    assert r.conteos["respuestas_con_comentario"] == 1


def test_eleccion_prefiere_comentario_y_luego_el_mas_largo():
    df = pd.DataFrame([
        {"Periodo": "2025-1", "ID de respuesta": "1", "Id Docente": "9", "Facultad de oferta": "A",
         CRUCE: "c", "Codigo del Grupo": "g", P19: "NR", P20: "NR", P24: ""},
        {"Periodo": "2025-1", "ID de respuesta": "1", "Id Docente": "9", "Facultad de oferta": "B",
         CRUCE: "c", "Codigo del Grupo": "g", P19: "ok", P20: "NR", P24: ""},
        {"Periodo": "2025-1", "ID de respuesta": "1", "Id Docente": "9", "Facultad de oferta": "C",
         CRUCE: "c", "Codigo del Grupo": "g", P19: "muy buen curso", P20: "NR", P24: ""},
    ])
    r = construir_respuestas(df, cfg_estudiante(P19, "obs19"), SAL)
    assert r.respuestas.loc[0, "texto_original"] == "muy buen curso"
    assert r.conteos["llaves_con_comentarios_distintos"] == 1


def test_desempate_es_deterministico_por_facultad():
    filas = [
        {"Periodo": "2025-1", "ID de respuesta": "1", "Id Docente": "9", "Facultad de oferta": fac,
         CRUCE: "c", "Codigo del Grupo": "g", P19: "NR", P20: txt, P24: ""}
        for fac, txt in [("Zeta", "Ninguno"), ("Alfa", "Ninguna")]
    ]
    a = construir_respuestas(pd.DataFrame(filas), cfg_estudiante(P20, "obs20"), SAL)
    b = construir_respuestas(pd.DataFrame(filas[::-1]), cfg_estudiante(P20, "obs20"), SAL)
    assert a.respuestas.loc[0, "texto_original"] == b.respuestas.loc[0, "texto_original"] == "Ninguna"


def test_codigo_cruce_sale_codificado():
    r = construir_respuestas(base_ejemplo(), cfg_estudiante(P19, "obs19"), SAL)
    assert "X1" not in set(r.respuestas["codigo_cruce_autoevaluacion"])
    assert codificar("X1", SAL) in set(r.respuestas["codigo_cruce_autoevaluacion"])


def test_filas_sin_llave_se_excluyen_y_se_cuentan():
    df = base_ejemplo()
    df.loc[0, "ID de respuesta"] = None
    r = construir_respuestas(df, cfg_estudiante(P19, "obs19"), SAL)
    assert r.conteos["filas_sin_llave"] == 1
    assert r.respuestas["respuesta_id"].is_unique


# --- Docentes ------------------------------------------------------------------

def test_autoevaluacion_una_vez_por_codigo_cruce():
    r = construir_respuestas(base_ejemplo(), cfg_docente(P24, "obs24"), SAL)
    assert len(r.respuestas) == 3                      # X1, X2, X5
    x1 = codificar("X1", SAL)
    fila = r.respuestas.set_index("id_origen").loc[x1]
    assert fila["texto_original"] == "Logro que participen"
    assert r.conteos["copias_descartadas"] == 1


def test_autoevaluacion_cero_es_no_respuesta():
    r = construir_respuestas(base_ejemplo(), cfg_docente(P24, "obs24"), SAL)
    fila = r.respuestas.set_index("id_origen").loc[codificar("X5", SAL)]
    assert fila["es_no_respuesta"]


# --- Configuración y orquestador ------------------------------------------------

def test_columna_faltante_da_error_claro():
    cfg = cfg_estudiante(P19, "obs19")
    cfg.contexto = ["Columna que no existe"]
    with pytest.raises(ValueError, match="Faltan columnas"):
        construir_respuestas(base_ejemplo(), cfg, SAL)


def _configuracion(tmp_path: Path) -> Path:
    preguntas = pd.DataFrame([
        {"encuesta": "evadoc_evaluacion", "clave_logica": "obs19", "alias_redaccion": P19,
         "hoja_excel": "EvaDoc", "archivo": "base.xlsx", "activa": "Si"},
        {"encuesta": "evadoc_evaluacion", "clave_logica": "obs20", "alias_redaccion": P20,
         "hoja_excel": "EvaDoc", "archivo": "base.xlsx", "activa": "Si"},
        {"encuesta": "evadoc_autoevaluacion", "clave_logica": "obs24", "alias_redaccion": P24,
         "hoja_excel": "EvaDoc", "archivo": "base.xlsx", "activa": "Si"},
    ])
    archivos = pd.DataFrame([
        {"encuesta": "evadoc_evaluacion", "alias_archivo": "base.xlsx"},
        {"encuesta": "evadoc_autoevaluacion", "alias_archivo": "base.xlsx"},
    ])
    etapa1 = pd.DataFrame(
        [("evadoc_evaluacion", t, v) for t, v in [
            ("llave", "Periodo"), ("llave", "ID de respuesta"), ("llave", "Id Docente"),
            ("contexto", "ID de respuesta"), ("contexto", "Id Docente"), ("contexto", CRUCE),
            ("oferta", "Facultad de oferta"), ("oferta", CRUCE),
            ("desempate", "Facultad de oferta"), ("codificar", CRUCE), ("rol", "estudiante")]]
        + [("evadoc_autoevaluacion", t, v) for t, v in [
            ("llave", "Periodo"), ("llave", "codigo cruce autoevaluacion"),  # sin tildes: se normaliza
            ("contexto", "Id Docente"), ("oferta", "Facultad de oferta"),
            ("desempate", "Facultad de oferta"), ("codificar", CRUCE), ("rol", "docente")]],
        columns=["encuesta", "tipo", "valor"],
    )
    ruta = tmp_path / "preguntas_abiertas.xlsx"
    with pd.ExcelWriter(ruta) as w:
        preguntas.to_excel(w, sheet_name="preguntas", index=False)
        archivos.to_excel(w, sheet_name="archivos", index=False)
        etapa1.to_excel(w, sheet_name="etapa1", index=False)
    (tmp_path / "raw").mkdir()
    base_ejemplo().to_excel(tmp_path / "raw" / "base.xlsx", sheet_name="EvaDoc", index=False)
    return ruta


def test_ejecucion_completa_cumple_el_contrato(tmp_path):
    config = _configuracion(tmp_path)
    salida = tmp_path / "salida"
    tablas = ejecutar_etapa1(config=config, carpeta_cruda=tmp_path / "raw",
                             carpeta_salida=salida, sal=SAL)
    r = tablas["respuestas"]
    assert len(r) == 3 + 3 + 3                          # obs19, obs20, obs24
    assert validar_salida(r, tablas["respuestas_oferta"]) == []
    assert r["id_corrida_preparacion"].nunique() == 1
    for nombre in ["respuestas", "respuestas_oferta", "catalogo_preguntas", "metadatos_preparacion"]:
        assert (salida / f"{nombre}.csv").exists()
    leido = pd.read_csv(salida / "metadatos_preparacion.csv")
    assert "sha256_archivo" in leido.columns and SAL not in leido.to_csv()


def test_dos_corridas_dan_el_mismo_resultado(tmp_path):
    config = _configuracion(tmp_path)
    a = ejecutar_etapa1(config=config, carpeta_cruda=tmp_path / "raw", carpeta_salida=None, sal=SAL)
    b = ejecutar_etapa1(config=config, carpeta_cruda=tmp_path / "raw", carpeta_salida=None, sal=SAL)
    cols = [c for c in a["respuestas"].columns if c != "id_corrida_preparacion"]
    pd.testing.assert_frame_equal(a["respuestas"][cols], b["respuestas"][cols])


def test_sin_sal_falla_con_mensaje_claro(tmp_path, monkeypatch):
    monkeypatch.delenv("ETAPA1_SAL", raising=False)
    config = _configuracion(tmp_path)
    with pytest.raises(ValueError, match="ETAPA1_SAL"):
        ejecutar_etapa1(config=config, carpeta_cruda=tmp_path / "raw", carpeta_salida=None,
                        archivo_sal=tmp_path / "no_existe")


def test_sal_desde_archivo_local(tmp_path, monkeypatch):
    monkeypatch.delenv("ETAPA1_SAL", raising=False)
    config = _configuracion(tmp_path)
    archivo = tmp_path / ".etapa1_sal"
    archivo.write_text(SAL + "\n", encoding="utf-8")
    tablas = ejecutar_etapa1(config=config, carpeta_cruda=tmp_path / "raw",
                             carpeta_salida=None, archivo_sal=archivo)
    assert codificar("X1", SAL) in set(tablas["respuestas"]["codigo_cruce_autoevaluacion"])
