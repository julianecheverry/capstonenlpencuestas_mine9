"""
Lista fija de aspectos por pregunta (data/utilities/taxonomia.xlsx).

La hoja `aspectos` tiene una fila por pregunta y aspecto:
  pregunta_codigo  código de la hoja `preguntas` de preguntas_abiertas.xlsx
  aspecto_id       código estable en MAYÚSCULAS (une los periodos)
  aspecto_nombre   nombre para mostrar; coincide con la etiqueta del clasificador
  descripcion      qué incluye el aspecto

Un mismo `aspecto_id` en varias preguntas es el mismo aspecto: debe llevar el
mismo nombre y la misma descripción. Así obs19 y obs20 comparten códigos y
un aspecto de otra lista con otro sentido necesita otro código
(p. ej. TECNOLOGIA frente a TECNOLOGIA_DOCENTE).

La Etapa 1 valida el archivo y lo publica como `taxonomia.csv`; el NLP lo usa
como lista de etiquetas y Power BI como dimensión de aspectos.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from preparacion.normalizacion import normalizar

COLUMNAS = ["pregunta_codigo", "aspecto_id", "aspecto_nombre", "descripcion"]
PATRON_ID = re.compile(r"^[A-Z0-9]+(_[A-Z0-9]+)*$")
RESERVADOS = {"SIN_ASPECTO"}  # los pone el NLP, no la lista


def version_taxonomia(ruta: str | Path) -> str:
    """Huella corta del archivo: cambia si cambia cualquier fila."""
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()[:10]


def leer_taxonomia(ruta: str | Path, preguntas_validas: set[str]) -> pd.DataFrame:
    """Lee y valida la taxonomía.

    Raises:
        ValueError: con todos los problemas encontrados, si hay alguno.
    """
    try:
        df = pd.read_excel(ruta, sheet_name="aspectos", dtype=str, keep_default_na=False)
    except ValueError as exc:
        raise ValueError(f"No existe la hoja 'aspectos' en {ruta}.") from exc
    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan las columnas {faltan} en la hoja 'aspectos' de {ruta}.")
    df = df[COLUMNAS].apply(lambda col: col.str.strip())
    df = df[(df != "").any(axis=1)].reset_index(drop=True)

    errores: list[str] = []
    vacias = df[(df[["pregunta_codigo", "aspecto_id", "aspecto_nombre"]] == "").any(axis=1)]
    if len(vacias):
        errores.append(f"{len(vacias)} filas sin pregunta_codigo, aspecto_id o aspecto_nombre "
                       f"(filas de Excel {[i + 2 for i in vacias.index[:10]]}).")
    df = df.drop(vacias.index)

    desconocidas = sorted(set(df["pregunta_codigo"]) - preguntas_validas)
    if desconocidas:
        errores.append(f"Preguntas que no están en la hoja 'preguntas': {desconocidas}.")

    malos = sorted({i for i in df["aspecto_id"] if not PATRON_ID.match(i)})
    if malos:
        errores.append(f"aspecto_id con formato inválido (solo MAYÚSCULAS, números y _): {malos[:10]}.")
    reservados = sorted(set(df["aspecto_id"]) & RESERVADOS)
    if reservados:
        errores.append(f"aspecto_id reservados para el NLP: {reservados}.")

    repetidos = df[df.duplicated(["pregunta_codigo", "aspecto_id"], keep=False)]
    if len(repetidos):
        errores.append("aspecto_id repetido dentro de una pregunta: "
                       f"{sorted(set(zip(repetidos.pregunta_codigo, repetidos.aspecto_id)))[:10]}.")

    nombres = df.assign(_n=df["aspecto_nombre"].map(normalizar))
    choques = nombres[nombres.duplicated(["pregunta_codigo", "_n"], keep=False)
                      & ~nombres.duplicated(["pregunta_codigo", "aspecto_id"], keep=False)]
    if len(choques):
        errores.append("Dos aspectos de una misma pregunta con el mismo nombre (sin contar tildes "
                       f"ni mayúsculas): {sorted(set(zip(choques.pregunta_codigo, choques.aspecto_nombre)))[:10]}.")

    distintos = df.groupby("aspecto_id")[["aspecto_nombre", "descripcion"]].nunique()
    distintos = distintos[(distintos > 1).any(axis=1)].index.tolist()
    if distintos:
        errores.append("Un mismo aspecto_id con nombre o descripción distintos entre preguntas "
                       f"(si es otro aspecto, necesita otro código): {distintos[:10]}.")

    if errores:
        raise ValueError(f"La taxonomía {ruta} tiene problemas:\n- " + "\n- ".join(errores))
    return df.reset_index(drop=True)
