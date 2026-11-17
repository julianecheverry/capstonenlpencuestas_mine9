"""Pruebas del informe de inconsistencias de la Etapa 1 (src/preparacion/informe.py)."""
import sys
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preparacion.informe import OCULTO, RevisionEvaDoc, Textos, generar_informe, revisar_bases  # noqa: E402
from preparacion.normalizacion import NO_RESPUESTA_POR_DEFECTO  # noqa: E402

CERRADOS = frozenset({"de acuerdo", "en desacuerdo", "n s"})
NO_RESPUESTA = NO_RESPUESTA_POR_DEFECTO | {"gracias"}
TEXTOS = Textos(NO_RESPUESTA, CERRADOS)
COMENTARIO = "El profesor explica muy bien"          # nunca debe salir en el informe
ITEMS = [f"Ítem {n}" for n in range(1, 19)]


def fila_evadoc(**k) -> dict:
    fila = {"Periodo": "2025-1", "ID de respuesta": "1", "Marca de tiempo (mm/dd/yyyy)": "2025-05-01 10:00:00",
            "Tiempo necesario para completar (segundos)": "100", "Codigo del Grupo": "G1", "Grupo": "GRUPO 1",
            "Materia": "MAT", "Codigo Materia": "M1", "Facultad Estudiante": "F"}
    fila.update({i: "5" for i in ITEMS})
    fila.update({"Observaciones - 19. Resaltar": COMENTARIO, "Observaciones - 20. Mejorar": "NR",
                 "Código Cruce Autoevaluación": "C1", "Observaciones - 23. Progreso": "NR",
                 "Observaciones - 24. Resaltar": "NR", "Observaciones - 25. Mejorar": "NR",
                 "Código Conteo": "K1", "N° Inscritos": "20", "Observaciones eval": "", "Observaciones Auto": "",
                 "Promedio Evaluación - General": "5", "Facultad de oferta": "FAC A", "Evaluación": "Si",
                 "Autoevaluación": "Si", "Id Estudiante": "E1", "Id Docente": "D1"})
    fila.update(k)
    return fila


def base_evadoc() -> pd.DataFrame:
    filas = [
        fila_evadoc(),                                                     # copia 1 por facultad
        fila_evadoc(**{"Facultad de oferta": "FAC B", "Observaciones - 19. Resaltar": "Ninguna"}),
        fila_evadoc(**{"ID de respuesta": "2", "Id Estudiante": "E2", "Observaciones - 23. Progreso": "0"}),
        fila_evadoc(**{"ID de respuesta": "3", "Id Estudiante": "E3", "Ítem 5": "NR",
                       "Promedio Evaluación - General": ""}),
    ]
    return pd.DataFrame(filas)


def base_acreditacion() -> pd.DataFrame:
    comun = {"Año": "2024", "Periodo": "2024-1", "Tiempo necesario para completar (segundos)": "50",
             "SNIES": "1", "Facultad": "F", "Departamento": "D", "Programa": "P", "Ciudad/Énfasis": "B",
             "Nivel de Formación": "U", "Nivel Académico": "Pregrado", "Tamaño de muestra": "10",
             "F1i1a.": "De acuerdo", "F12i180h.": "De acuerdo"}
    return pd.DataFrame([
        {**comun, "ID de respuesta": "10", "Población": "Docentes", "llave": "L1", "FNACi181i.": "De acuerdo"},
        {**comun, "ID de respuesta": "10", "Población": "Docentes", "llave": "L2", "FNACi181i.": COMENTARIO,
         "F1i1a.": "En desacuerdo"},
        {**comun, "ID de respuesta": "11", "Población": "Docentes", "llave": "L1", "FNACi181i.": "NR"},
        {**comun, "ID de respuesta": "12", "Población": "docentes", "llave": "L1", "FNACi181i.": "NR"},
    ])


def base_calidad() -> pd.DataFrame:
    return pd.DataFrame([
        {"ID de Acceso": "a1", "Año": "2021", "Población": "GRADUADOS", "Facultad/Departamento/Área": " DERECHO",
         "Pregunta": "¿Sugerencias?", "Respuesta": COMENTARIO},
        {"ID de Acceso": "a2", "Año": "2023", "Población": "GRADUADOS", "Facultad/Departamento/Área": "SIN INFORMACIÓN",
         "Pregunta": "¿Sugerencias?", "Respuesta": "Gracias"},
    ])


@pytest.fixture
def config(tmp_path, monkeypatch):
    """La configuración solo se usa para ubicar preguntas; se reemplaza por funciones simples."""
    import preparacion.informe as informe
    monkeypatch.setattr(informe, "preguntas_activas",
                        lambda cfg, enc: [{"pregunta_codigo": "p", "alias": ["FNACi181i."]}])
    monkeypatch.setattr(informe, "asignar_preguntas",
                        lambda serie, preguntas, enc: pd.Series(["cal_sugerencias"] * len(serie), index=serie.index))
    return tmp_path / "config.xlsx"


def _hojas(config, **crudas):
    return {h.codigo: h for h in revisar_bases(crudas, config, NO_RESPUESTA, CERRADOS)}


def test_ocultar_muestra_solo_no_respuesta_y_valores_cerrados():
    s = pd.Series([COMENTARIO, "NR", "Ninguna.", "De acuerdo", ""])
    assert TEXTOS.ocultar(s).tolist() == [OCULTO, "NR", "Ninguna.", "De acuerdo", ""]
    assert TEXTOS.largo(s).tolist() == [len(COMENTARIO), 0, 0, len("De acuerdo"), 0]


def test_evadoc_detecta_copias_promedio_y_cero(config):
    h = _hojas(config, evadoc_evaluacion=base_evadoc())
    assert len(h["I01"].detalle) == 2                      # una respuesta en dos filas
    assert len(h["I02"].detalle) == 2                      # P19 distinto entre copias
    assert len(h["I08"].detalle) == 1                      # ítem NR y promedio vacío
    assert len(h["I12"].detalle) == 1                      # «0» en P23
    assert not any(x.error for x in h.values())


def test_columna_que_falta_no_detiene_las_demas_revisiones(config):
    h = _hojas(config, evadoc_evaluacion=base_evadoc().drop(columns=["Código Conteo"]))
    assert "Código Conteo" in h["I06"].error
    assert h["I01"].error == "" and len(h["I01"].detalle) == 2


def test_acreditacion_valor_cerrado_y_copias_distintas(config):
    h = _hojas(config, autoevaluacion_acreditacion=base_acreditacion())
    i16 = h["I16"].detalle
    assert len(i16) == 1 and i16["¿Otra copia del envío trae un comentario?"].tolist() == ["Sí"]
    assert h["I17"].detalle["ID de respuesta"].tolist() == ["10"]
    assert len(h["I15"].detalle) == 1                      # «docentes» frente a «Docentes»


def test_calidad_area_y_no_respuesta(config):
    h = _hojas(config, calidad_servicios=base_calidad())
    assert len(h["I18"].detalle) == 2                      # espacio inicial y «SIN INFORMACIÓN»
    assert h["I21"].detalle["Valor (normalizado)"].tolist() == ["gracias"]


def test_informe_completo_sin_comentarios(config, tmp_path):
    ruta, hojas = generar_informe(
        {"evadoc_evaluacion": base_evadoc(), "autoevaluacion_acreditacion": base_acreditacion(),
         "calidad_servicios": base_calidad()},
        config, NO_RESPUESTA, CERRADOS, "20261116-000000_abc", "2026-11-16 00:00",
        ["prueba"], tmp_path / "reportes")
    assert ruta.name == "Informe_inconsistencias_corrida_20261116-000000_abc.xlsx"
    wb = load_workbook(ruta)
    assert wb.sheetnames[0] == "Resumen" and len(hojas) == 21
    textos = [str(c.value) for ws in wb.worksheets for fila in ws.iter_rows() for c in fila if c.value is not None]
    assert not any(COMENTARIO in t for t in textos)
    assert any(t == OCULTO for t in textos)


def test_receta_del_cruce_se_calcula_por_periodo():
    filas = []
    for i, (grupo, materia, cruce) in enumerate([("A", "M1", "C1"), ("A", "M1", "C1"), ("B", "M2", "C2")]):
        filas.append(fila_evadoc(**{"ID de respuesta": str(i), "Grupo": grupo, "Materia": materia,
                                    "Codigo del Grupo": grupo, "Código Cruce Autoevaluación": cruce}))
    h = RevisionEvaDoc(pd.DataFrame(filas), TEXTOS).i05()
    assert h.detalle.empty                                  # una receta lo reproduce sin excepciones
    assert "2025-1: Código del Grupo + Id Docente" in h.observado
