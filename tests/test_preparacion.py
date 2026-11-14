"""
Pruebas de la Etapa 1 (src/preparacion) con datos inventados.

Reproducen los casos difíciles encontrados en la base real (ver el
"Ejemplo trazado" del contrato de salida): copias por facultad de oferta,
no-respuesta escrita de varias formas, un mismo ID de respuesta en
evaluaciones de docentes distintos, autoevaluación repetida por estudiante,
no-respuesta registrada como "0" y roles escritos de formas distintas.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preparacion.etapa1 import ejecutar_etapa1, validar_salida  # noqa: E402
from preparacion.normalizacion import (  # noqa: E402
    NO_RESPUESTA_POR_DEFECTO, es_no_respuesta, nombre_columna, normalizar,
)
from preparacion.respuestas import ConfigPregunta, codificar, construir_respuestas  # noqa: E402

SAL = "sal-de-prueba"
P19 = "Observaciones - 19. Aspectos para resaltar del docente y/o del espacio académico"
P20 = "Observaciones - 20. Oportunidades de mejora del docente y/o del espacio académico"
P24 = "Observaciones - 24. Aspectos para resaltar de su labor como docente y/o del espacio académico"
CRUCE = "Código Cruce Autoevaluación"
ROLES = {"estudiantes": "Estudiantes", "docentes": "Docentes",
         "directivos y coordinadores": "Directivos y coordinadores"}


def base_ejemplo() -> pd.DataFrame:
    """Las cuatro filas del Ejemplo trazado (2024-2), más un caso de '0'."""
    comun = {"Periodo": "2024-2", "Facultad Estudiante": "Derecho",
             "Evaluación": "Si", "Autoevaluación": "SI"}
    return pd.DataFrame([
        # Filas 1 y 2: misma respuesta, una copia por facultad de oferta.
        {**comun, "ID de respuesta": "555001", "Id Docente": "27", "Facultad de oferta": "Derecho",
         CRUCE: "X1", "Codigo del Grupo": "G1", P19: "Explica muy bien", P20: "NR",
         P24: "Logro que participen"},
        {**comun, "ID de respuesta": "555001", "Id Docente": "27", "Facultad de oferta": "Economía",
         CRUCE: "X2", "Codigo del Grupo": "G1", P19: "Explica muy bien", P20: "Ninguna",
         P24: "Promuevo el debate"},
        # Fila 3: otro estudiante del docente 27.
        {**comun, "ID de respuesta": "555002", "Id Docente": "27", "Facultad de oferta": "Derecho",
         CRUCE: "X1", "Codigo del Grupo": "G1", P19: "NR", P20: "A veces llega tarde",
         P24: "Logro que participen"},
        # Fila 4: mismo ID de respuesta, pero es otra evaluación (docente 31).
        {**comun, "ID de respuesta": "555002", "Id Docente": "31", "Facultad de oferta": "Derecho",
         CRUCE: "X5", "Codigo del Grupo": "G2", P19: "Gran profesor", P20: "NR", P24: "0"},
    ])


def cfg_estudiante(columna_texto: str, codigo: str, **extra) -> ConfigPregunta:
    return ConfigPregunta(
        encuesta="evadoc_evaluacion", pregunta_codigo=codigo, columna_texto=columna_texto,
        llaves=["Periodo", "ID de respuesta", "Id Docente"],
        contexto=["ID de respuesta", "Id Docente", CRUCE, "Codigo del Grupo", "Facultad Estudiante"],
        oferta=["Facultad de oferta", CRUCE], desempate=["Facultad de oferta"],
        rol="Estudiantes", roles=ROLES, **extra,
    )


def cfg_docente(columna_texto: str, codigo: str) -> ConfigPregunta:
    return ConfigPregunta(
        encuesta="evadoc_autoevaluacion", pregunta_codigo=codigo, columna_texto=columna_texto,
        llaves=["Periodo", CRUCE], contexto=[CRUCE, "Id Docente", "Codigo del Grupo"],
        oferta=["Facultad de oferta"], desempate=["Facultad de oferta"],
        rol="docentes", roles=ROLES,
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


def test_lista_de_no_respuesta_es_configurable():
    lista = NO_RESPUESTA_POR_DEFECTO | {"sin comentarios"}
    df = base_ejemplo()
    df.loc[3, P19] = "Sin comentarios."
    r = construir_respuestas(df, cfg_estudiante(P19, "obs19", no_respuesta=lista))
    assert r.respuestas.set_index("id_origen").loc["555002-31", "es_no_respuesta"]


# --- Estudiantes ---------------------------------------------------------------

def test_estudiante_una_respuesta_por_llave():
    r = construir_respuestas(base_ejemplo(), cfg_estudiante(P19, "obs19"))
    assert len(r.respuestas) == 3                      # 555001-27, 555002-27, 555002-31
    assert r.respuestas["respuesta_id"].is_unique
    assert r.conteos["copias_descartadas"] == 1


def test_nucleo_no_tiene_columnas_de_contexto():
    r = construir_respuestas(base_ejemplo(), cfg_estudiante(P19, "obs19"))
    assert list(r.respuestas.columns) == [
        "respuesta_id", "encuesta", "pregunta_codigo", "rol_respondiente", "periodo",
        "id_origen", "texto_original", "es_no_respuesta", "marca_inconsistencia"]
    assert list(r.contexto["respuesta_id"]) == list(r.respuestas["respuesta_id"])
    assert "facultad_estudiante" in r.contexto.columns
    assert (r.contexto.drop(columns="respuesta_id") != "").all().all()   # sin columnas vacías


def test_estudiante_mismo_id_otro_docente_no_se_fusiona():
    r = construir_respuestas(base_ejemplo(), cfg_estudiante(P19, "obs19"))
    assert {"555002-27", "555002-31"} <= set(r.respuestas["id_origen"])
    gran = r.respuestas.set_index("id_origen").loc["555002-31"]
    assert gran["texto_original"] == "Gran profesor"
    assert not gran["es_no_respuesta"]


def test_respuesta_id_tiene_el_formato_del_contrato():
    r = construir_respuestas(base_ejemplo(), cfg_estudiante(P19, "obs19"))
    assert "evadoc_evaluacion|2024-2|555001-27|obs19" in set(r.respuestas["respuesta_id"])


def test_periodo_se_conserva_como_texto():
    r = construir_respuestas(base_ejemplo(), cfg_estudiante(P19, "obs19"))
    assert set(r.respuestas["periodo"]) == {"2024-2"}


def test_copias_por_facultad_quedan_en_oferta():
    r = construir_respuestas(base_ejemplo(), cfg_estudiante(P19, "obs19"))
    rid = "evadoc_evaluacion|2024-2|555001-27|obs19"
    facultades = set(r.oferta.loc[r.oferta["respuesta_id"] == rid, "facultad_de_oferta"])
    assert facultades == {"Derecho", "Economía"}
    assert set(r.oferta["respuesta_id"]) == set(r.respuestas["respuesta_id"])


def test_no_respuesta_se_marca_no_se_elimina():
    r = construir_respuestas(base_ejemplo(), cfg_estudiante(P20, "obs20"))
    assert r.respuestas.set_index("id_origen").loc["555001-27", "es_no_respuesta"]
    assert r.conteos["no_respuestas"] == 2             # 555001-27 y 555002-31
    assert r.conteos["respuestas_con_comentario"] == 1


def test_eleccion_prefiere_comentario_y_luego_el_mas_largo():
    base = {"Periodo": "2025-1", "ID de respuesta": "1", "Id Docente": "9", CRUCE: "c",
            "Codigo del Grupo": "g", "Facultad Estudiante": "f", P20: "NR", P24: ""}
    df = pd.DataFrame([
        {**base, "Facultad de oferta": "A", P19: "NR"},
        {**base, "Facultad de oferta": "B", P19: "ok"},
        {**base, "Facultad de oferta": "C", P19: "muy buen curso"},
    ])
    r = construir_respuestas(df, cfg_estudiante(P19, "obs19"))
    assert r.respuestas.loc[0, "texto_original"] == "muy buen curso"
    assert r.conteos["llaves_con_comentarios_distintos"] == 1
    assert r.respuestas.loc[0, "marca_inconsistencia"] == "comentarios_distintos_en_copias"


def test_desempate_es_deterministico_por_facultad():
    base = {"Periodo": "2025-1", "ID de respuesta": "1", "Id Docente": "9", CRUCE: "c",
            "Codigo del Grupo": "g", "Facultad Estudiante": "f", P19: "NR", P24: ""}
    filas = [{**base, "Facultad de oferta": fac, P20: txt}
             for fac, txt in [("Zeta", "Ninguno"), ("Alfa", "Ninguna")]]
    a = construir_respuestas(pd.DataFrame(filas), cfg_estudiante(P20, "obs20"))
    b = construir_respuestas(pd.DataFrame(filas[::-1]), cfg_estudiante(P20, "obs20"))
    assert a.respuestas.loc[0, "texto_original"] == b.respuestas.loc[0, "texto_original"] == "Ninguna"


def test_codigo_cruce_original_por_defecto():
    r = construir_respuestas(base_ejemplo(), cfg_estudiante(P19, "obs19"))
    assert {"X1", "X5"} <= set(r.contexto["codigo_cruce_autoevaluacion"])


def test_codificar_es_opcional_y_exige_sal():
    cfg = cfg_estudiante(P19, "obs19", codificar=[CRUCE])
    with pytest.raises(ValueError, match="frase secreta"):
        construir_respuestas(base_ejemplo(), cfg)
    r = construir_respuestas(base_ejemplo(), cfg, SAL)
    assert "X1" not in set(r.contexto["codigo_cruce_autoevaluacion"])
    assert codificar("X1", SAL) in set(r.contexto["codigo_cruce_autoevaluacion"])


def test_filas_sin_llave_se_excluyen_y_se_cuentan():
    df = base_ejemplo()
    df.loc[0, "ID de respuesta"] = None
    r = construir_respuestas(df, cfg_estudiante(P19, "obs19"))
    assert r.conteos["filas_sin_llave"] == 1
    assert r.respuestas["respuesta_id"].is_unique


# --- Roles ---------------------------------------------------------------------

def test_roles_se_unifican():
    r = construir_respuestas(base_ejemplo(), cfg_docente(P24, "obs24"))
    assert set(r.respuestas["rol_respondiente"]) == {"Docentes"}


def test_rol_desde_columna_con_variantes():
    df = base_ejemplo()
    df["Población"] = ["Docentes", "Directivos y Coordinadores", "Directivos y coordinadores", "docentes"]
    cfg = cfg_estudiante(P19, "obs19")
    cfg.rol, cfg.columna_rol = None, "Población"
    r = construir_respuestas(df, cfg)
    assert set(r.respuestas["rol_respondiente"]) == {"Docentes", "Directivos y coordinadores"}


def test_rol_sin_equivalencia_da_error_claro():
    df = base_ejemplo()
    df["Población"] = "Visitantes"
    cfg = cfg_estudiante(P19, "obs19")
    cfg.rol, cfg.columna_rol = None, "Población"
    with pytest.raises(ValueError, match="sin equivalencia"):
        construir_respuestas(df, cfg)


# --- Docentes ------------------------------------------------------------------

def test_autoevaluacion_una_vez_por_codigo_cruce():
    r = construir_respuestas(base_ejemplo(), cfg_docente(P24, "obs24"))
    assert len(r.respuestas) == 3                      # X1, X2, X5
    fila = r.respuestas.set_index("id_origen").loc["X1"]
    assert fila["texto_original"] == "Logro que participen"
    assert r.conteos["copias_descartadas"] == 1


def test_autoevaluacion_cero_es_no_respuesta():
    r = construir_respuestas(base_ejemplo(), cfg_docente(P24, "obs24"))
    assert r.respuestas.set_index("id_origen").loc["X5", "es_no_respuesta"]


# --- Configuración y orquestador ------------------------------------------------

def test_columna_faltante_da_error_claro():
    cfg = cfg_estudiante(P19, "obs19")
    cfg.contexto = ["Columna que no existe"]
    with pytest.raises(ValueError, match="Faltan columnas"):
        construir_respuestas(base_ejemplo(), cfg)


def _configuracion(tmp_path: Path, codificar_cruce: bool = False) -> Path:
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
    evaluacion = [
        ("llave", "Periodo"), ("llave", "ID de respuesta"), ("llave", "Id Docente"),
        ("contexto", "ID de respuesta"), ("contexto", "Id Docente"), ("contexto", CRUCE),
        ("contexto", "Facultad Estudiante"), ("oferta", "Facultad de oferta"), ("oferta", CRUCE),
        ("desempate", "Facultad de oferta"), ("rol", "Estudiantes"),
        ("filtro", "Evaluación = Sí")]
    if codificar_cruce:
        evaluacion.append(("codificar", CRUCE))
    autoevaluacion = [
        ("llave", "Periodo"), ("llave", "codigo cruce autoevaluacion"),  # sin tildes: se normaliza
        ("contexto", CRUCE), ("contexto", "Id Docente"), ("oferta", "Facultad de oferta"),
        ("desempate", "Facultad de oferta"), ("rol", "Docentes"),
        ("filtro", "Autoevaluación = Sí")]
    etapa1 = pd.DataFrame(
        [("evadoc_evaluacion", t, v) for t, v in evaluacion]
        + [("evadoc_autoevaluacion", t, v) for t, v in autoevaluacion],
        columns=["encuesta", "tipo", "valor"])
    no_respuesta = pd.DataFrame({"valor": sorted(NO_RESPUESTA_POR_DEFECTO - {""})})
    roles = pd.DataFrame([{"valor_original": o, "rol": r} for o, r in ROLES.items()])
    ruta = tmp_path / "preguntas_abiertas.xlsx"
    with pd.ExcelWriter(ruta) as w:
        preguntas.to_excel(w, sheet_name="preguntas", index=False)
        archivos.to_excel(w, sheet_name="archivos", index=False)
        etapa1.to_excel(w, sheet_name="etapa1", index=False)
        no_respuesta.to_excel(w, sheet_name="no_respuesta", index=False)
        roles.to_excel(w, sheet_name="roles", index=False)
        pd.DataFrame({"valor": ["De acuerdo", "En desacuerdo", "N/S"]}).to_excel(
            w, sheet_name="valores_cerrados", index=False)
    (tmp_path / "raw").mkdir(exist_ok=True)
    base_ejemplo().to_excel(tmp_path / "raw" / "base.xlsx", sheet_name="EvaDoc", index=False)
    return ruta


def _ejecutar(tmp_path, **kw):
    return ejecutar_etapa1(config=kw.pop("config"), carpeta_cruda=tmp_path / "raw",
                           archivo_sal=tmp_path / "no_existe", **kw)


def test_ejecucion_completa_cumple_el_contrato(tmp_path):
    config = _configuracion(tmp_path)
    salida = tmp_path / "salida"
    tablas = _ejecutar(tmp_path, config=config, carpeta_salida=salida)
    r = tablas["respuestas"]
    assert len(r) == 3 + 3 + 3                          # obs19, obs20, obs24
    contextos = {k.removeprefix("contexto_"): v for k, v in tablas.items() if k.startswith("contexto_")}
    assert set(contextos) == {"evadoc_evaluacion", "evadoc_autoevaluacion"}
    copias = {"respuestas_oferta": (tablas["respuestas_oferta"],
                                    {"evadoc_evaluacion", "evadoc_autoevaluacion"}),
              "respuestas_programa": (tablas["respuestas_programa"], set())}
    assert validar_salida(r, contextos, copias) == []
    assert r["id_corrida_preparacion"].nunique() == 1
    assert set(r["rol_respondiente"]) == {"Estudiantes", "Docentes"}
    for nombre in ["respuestas", "respuestas_oferta", "respuestas_programa", "catalogo_preguntas", "metadatos_preparacion",
                   "contexto_evadoc_evaluacion", "contexto_evadoc_autoevaluacion"]:
        assert (salida / f"{nombre}.csv").exists()
    leido = pd.read_csv(salida / "respuestas.csv", dtype=str)
    assert set(leido["periodo"]) == {"2024-2"}


def test_dos_corridas_dan_el_mismo_resultado(tmp_path):
    config = _configuracion(tmp_path)
    a = _ejecutar(tmp_path, config=config, carpeta_salida=None)
    b = _ejecutar(tmp_path, config=config, carpeta_salida=None)
    cols = [c for c in a["respuestas"].columns if c != "id_corrida_preparacion"]
    pd.testing.assert_frame_equal(a["respuestas"][cols], b["respuestas"][cols])


def test_sin_sal_y_con_columnas_a_codificar_falla(tmp_path, monkeypatch):
    monkeypatch.delenv("ETAPA1_SAL", raising=False)
    config = _configuracion(tmp_path, codificar_cruce=True)
    with pytest.raises(ValueError, match="ETAPA1_SAL"):
        _ejecutar(tmp_path, config=config, carpeta_salida=None)


def test_sal_desde_archivo_local(tmp_path, monkeypatch):
    monkeypatch.delenv("ETAPA1_SAL", raising=False)
    config = _configuracion(tmp_path, codificar_cruce=True)
    archivo = tmp_path / ".etapa1_sal"
    archivo.write_text(SAL + "\n", encoding="utf-8")
    tablas = ejecutar_etapa1(config=config, carpeta_cruda=tmp_path / "raw",
                             carpeta_salida=None, archivo_sal=archivo)
    ctx = tablas["contexto_evadoc_evaluacion"]
    assert codificar("X1", SAL) in set(ctx["codigo_cruce_autoevaluacion"])


def test_na_y_n_a_en_la_configuracion_no_se_pierden(tmp_path):
    from preparacion.configuracion import lista_no_respuesta
    ruta = tmp_path / "c.xlsx"
    pd.DataFrame({"valor": ["NA", "N/A", "Ninguna"]}).to_excel(ruta, sheet_name="no_respuesta", index=False)
    assert {"na", "n a", "ninguna", ""} == set(lista_no_respuesta(ruta))


def test_copias_con_el_mismo_comentario_no_se_marcan():
    r = construir_respuestas(base_ejemplo(), cfg_estudiante(P19, "obs19"))
    assert set(r.respuestas["marca_inconsistencia"]) == {""}      # 555001: dos copias iguales


def test_autoevaluacion_con_textos_distintos_se_marca():
    df = base_ejemplo()
    df.loc[2, P24] = "Uso estudios de caso en clase"           # X1: fila 1 dice otra cosa
    r = construir_respuestas(df, cfg_docente(P24, "obs24"))
    fila = r.respuestas.set_index("id_origen").loc["X1"]
    assert fila["marca_inconsistencia"] == "comentarios_distintos_en_copias"
    assert fila["texto_original"] == "Uso estudios de caso en clase"   # el más largo
    assert r.conteos["respuestas_con_marca_inconsistencia"] == 1


def test_inconsistencias_csv_se_genera(tmp_path):
    config = _configuracion(tmp_path)
    salida = tmp_path / "salida"
    tablas = _ejecutar(tmp_path, config=config, carpeta_salida=salida)
    assert (salida / "inconsistencias.csv").exists()
    assert list(tablas["inconsistencias"].columns) == [
        "respuesta_id", "encuesta", "pregunta_codigo", "periodo", "id_origen",
        "marca_inconsistencia", "detalle"]



# --- Filtro de respondientes efectivos -------------------------------------------

def test_interpretar_filtros():
    from preparacion.configuracion import interpretar_filtros
    assert interpretar_filtros(["Evaluación = Sí", "Nivel = Pregrado; Posgrado"]) == {
        "Evaluación": {"si"}, "Nivel": {"pregrado", "posgrado"}}
    with pytest.raises(ValueError, match="mal escrito"):
        interpretar_filtros(["Evaluación Sí"])


def test_filtro_excluye_a_quien_no_respondio():
    df = base_ejemplo()
    df.loc[3, "Evaluación"] = "NO"                       # 555002-31 no evaluó
    cfg = cfg_estudiante(P19, "obs19", filtros={"Evaluación": {"si"}})
    r = construir_respuestas(df, cfg)
    assert "555002-31" not in set(r.respuestas["id_origen"])
    assert r.conteos["filas_excluidas_por_filtro"] == 1
    assert len(r.respuestas) == 2


def test_filtro_con_marca_mixta_conserva_la_respuesta():
    df = base_ejemplo()
    df.loc[2, "Autoevaluación"] = "No"                   # X1: fila 1 Sí, fila 3 No
    cfg = cfg_docente(P24, "obs24")
    cfg.filtros = {"Autoevaluación": {"si"}}
    r = construir_respuestas(df, cfg)
    assert "X1" in set(r.respuestas["id_origen"])
    assert r.conteos["filas_excluidas_por_filtro"] == 1


def test_ejecucion_aplica_filtros_de_la_configuracion(tmp_path):
    df = base_ejemplo()
    df.loc[3, ["Evaluación", "Autoevaluación"]] = ["No", "No"]
    config = _configuracion(tmp_path)
    df.to_excel(tmp_path / "raw" / "base.xlsx", sheet_name="EvaDoc", index=False)
    tablas = _ejecutar(tmp_path, config=config, carpeta_salida=None)
    ids = set(tablas["respuestas"]["id_origen"])
    assert "555002-31" not in ids and "X5" not in ids
    meta = tablas["metadatos_preparacion"]
    assert meta["filas_excluidas_por_filtro"].sum() == 3     # obs19, obs20, obs24


# --- Acreditación: una respuesta por envío y valores cerrados (v1.4) ----------------

ACRED = "FNACNAANAi181i."
CERRADOS = frozenset({"de acuerdo", "en desacuerdo", "n s"})


def cfg_acreditacion(**extra) -> ConfigPregunta:
    return ConfigPregunta(
        encuesta="autoevaluacion_acreditacion", pregunta_codigo="acred_sug", columna_texto=ACRED,
        llaves=["Periodo", "ID de respuesta"], contexto=["ID de respuesta", "Población"],
        programa=["llave", "Programa", "SNIES"], renombrar={"llave": "llave_oac"},
        columna_rol="Población", roles={"docentes": "Docentes"},
        valores_cerrados=CERRADOS, **extra,
    )


def base_acreditacion(textos: list[str]) -> pd.DataFrame:
    return pd.DataFrame([
        {"Periodo": "2024-1", "ID de respuesta": "900", "Población": "Docentes",
         "llave": f"L{i}", "Programa": f"Prog {i}", "SNIES": f"{100 + i}", ACRED: t}
        for i, t in enumerate(textos)])


def test_acreditacion_una_respuesta_por_envio_y_programas_aparte():
    r = construir_respuestas(base_acreditacion(["", "Faltan laboratorios", ""]), cfg_acreditacion())
    assert len(r.respuestas) == 1
    assert r.respuestas.loc[0, "id_origen"] == "900"
    assert r.respuestas.loc[0, "texto_original"] == "Faltan laboratorios"
    assert list(r.programa.columns) == ["respuesta_id", "llave_oac", "programa", "snies"]
    assert len(r.programa) == 3
    assert list(r.contexto.columns) == ["respuesta_id", "id_de_respuesta", "poblacion"]
    assert r.oferta.empty


def test_valor_cerrado_es_no_respuesta_y_se_marca():
    r = construir_respuestas(base_acreditacion(["De acuerdo", ""]), cfg_acreditacion())
    fila = r.respuestas.iloc[0]
    assert bool(fila["es_no_respuesta"])
    assert fila["marca_inconsistencia"] == "valor_cerrado_en_pregunta_abierta"
    assert r.conteos["llaves_con_valor_cerrado"] == 1


def test_valor_cerrado_no_compite_con_el_texto_real():
    r = construir_respuestas(base_acreditacion(["En desacuerdo", "Faltan laboratorios", "N/S"]),
                             cfg_acreditacion())
    fila = r.respuestas.iloc[0]
    assert fila["texto_original"] == "Faltan laboratorios"
    assert not fila["es_no_respuesta"]
    assert fila["marca_inconsistencia"] == "valor_cerrado_en_pregunta_abierta"   # no "comentarios distintos"


def test_valor_cerrado_dentro_de_una_frase_es_comentario():
    r = construir_respuestas(base_acreditacion(["De acuerdo, pero faltan salones"]), cfg_acreditacion())
    assert not r.respuestas.loc[0, "es_no_respuesta"]
    assert r.respuestas.loc[0, "marca_inconsistencia"] == ""


def test_varias_marcas_se_separan_y_validan():
    r = construir_respuestas(base_acreditacion(["Faltan salones", "Faltan laboratorios", "De acuerdo"]),
                             cfg_acreditacion())
    marca = r.respuestas.loc[0, "marca_inconsistencia"]
    assert marca == "comentarios_distintos_en_copias; valor_cerrado_en_pregunta_abierta"
    contextos = {"autoevaluacion_acreditacion": r.contexto}
    copias = {"respuestas_programa": (r.programa, {"autoevaluacion_acreditacion"})}
    assert validar_salida(r.respuestas, contextos, copias) == []


def test_validar_salida_detecta_respuestas_sin_programa():
    r = construir_respuestas(base_acreditacion(["Faltan salones"]), cfg_acreditacion())
    copias = {"respuestas_programa": (r.programa.iloc[0:0], {"autoevaluacion_acreditacion"})}
    errores = validar_salida(r.respuestas, {"autoevaluacion_acreditacion": r.contexto}, copias)
    assert any("sin fila en respuestas_programa" in e for e in errores)


def test_interpretar_renombres():
    from preparacion.configuracion import interpretar_renombres
    assert interpretar_renombres(["llave = llave_oac"]) == {"llave": "llave_oac"}
    with pytest.raises(ValueError):
        interpretar_renombres(["llave llave_oac"])


def test_inconsistencias_csv_una_fila_por_codigo(tmp_path):
    config = _configuracion(tmp_path)
    base = base_ejemplo()
    base.loc[0, P19] = "De acuerdo"           # copia 1 de 555001: valor cerrado
    base.loc[1, P19] = "Explica muy bien, gracias"   # copia 2: otro texto
    base.to_excel(tmp_path / "raw" / "base.xlsx", sheet_name="EvaDoc", index=False)
    tablas = _ejecutar(tmp_path, config=config, carpeta_salida=None)
    inc = tablas["inconsistencias"]
    assert set(inc["marca_inconsistencia"]) == {"valor_cerrado_en_pregunta_abierta"}
    assert inc["detalle"].notna().all()

