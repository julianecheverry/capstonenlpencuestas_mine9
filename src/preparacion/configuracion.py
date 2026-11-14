"""
Lectura de la configuración de la Etapa 1 desde preguntas_abiertas.xlsx.

Reutiliza las hojas que ya existen (`preguntas`, `archivos`) y agrega una
hoja nueva, `etapa1`, con tres columnas: encuesta | tipo | valor.

Tipos admitidos en la hoja `etapa1`:
  llave       Columnas que identifican una respuesta, en orden. La primera
              debe ser el periodo. Ej.: Periodo, ID de respuesta, Id Docente.
  contexto    Columnas que van a la tabla `contexto_<encuesta>` (de la copia elegida).
  oferta      Columnas que se registran por copia en `respuestas_oferta`.
  desempate   Columnas para desempatar copias (orden alfabético).
  codificar   OPCIONAL. Columnas que se reemplazan por un código (hash con sal),
              p. ej. para copias de trabajo que circulan fuera del cliente.
              En la entrega al cliente no se usa (decisión del 14-nov-2026).
  rol         Rol fijo del respondiente (ej.: Estudiantes, Docentes).
  columna_rol Columna de la base cruda que trae el rol (ej.: Población).

Otras dos hojas nuevas:
  no_respuesta  columna `valor`: textos que cuentan como no-respuesta. Se
                comparan normalizados (sin tildes, mayúsculas ni puntuación),
                así que "Ninguna." y "NINGUNA" equivalen a "ninguna".
  roles         columnas `valor_original` y `rol`: traduce cualquier forma
                del rol (fijo o tomado de una columna) a un vocabulario único.
                El vocabulario es el de la columna Población de la encuesta de
                acreditación (Estudiantes, Docentes, Administrativos,
                Directivos y coordinadores, Graduados); la evaluación docente
                se adapta a él (decisión del 14-nov-2026).

Los nombres de columna se comparan normalizados (sin tildes, mayúsculas ni
puntuación), igual que las redacciones de las preguntas.

Nota de diseño: estas funciones duplican de forma deliberada una parte
pequeña de `datacleaning/preguntas.py`, porque ese módulo importa la limpieza
de texto (nltk, descargas de corpus) y la Etapa 1 no debe depender de ella.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from preparacion.normalizacion import normalizar

TIPOS_ETAPA1 = {"llave", "contexto", "oferta", "desempate", "codificar", "rol", "columna_rol"}


def _leer_hoja(config: str | Path, hoja: str, columnas: set[str]) -> pd.DataFrame:
    try:
        # keep_default_na=False: sin esto pandas convierte "NA" y "N/A" en vacío,
        # y esos son justamente valores de la lista de no-respuesta.
        df = pd.read_excel(config, sheet_name=hoja, dtype=str, keep_default_na=False)
        df = df.apply(lambda col: col.str.strip()).replace("", pd.NA)
    except ValueError as exc:
        raise ValueError(f"No existe la hoja '{hoja}' en {config}.") from exc
    faltan = columnas - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan las columnas {faltan} en la hoja '{hoja}' de {config}.")
    return df


def encuestas_configuradas(config: str | Path) -> list[str]:
    """Encuestas que tienen configuración en la hoja `etapa1`, en orden."""
    df = _leer_hoja(config, "etapa1", {"encuesta", "tipo", "valor"})
    return list(dict.fromkeys(df["encuesta"].dropna().str.strip()))


def preguntas_activas(config: str | Path, encuesta: str) -> list[dict]:
    """Preguntas activas de una encuesta (hoja `preguntas`)."""
    df = _leer_hoja(
        config, "preguntas",
        {"encuesta", "clave_logica", "alias_redaccion", "hoja_excel", "archivo", "activa"},
    )
    df = df[(df["encuesta"] == encuesta) & (df["activa"].map(normalizar) == "si")]
    if df.empty:
        raise ValueError(f"La encuesta '{encuesta}' no tiene preguntas activas en {config}.")
    preguntas = []
    for clave, grupo in df.groupby("clave_logica", sort=False):
        preguntas.append({
            "pregunta_codigo": clave,
            "alias": grupo["alias_redaccion"].dropna().tolist(),
            "hoja": grupo["hoja_excel"].iloc[0],
        })
    return preguntas


def archivo_de_encuesta(config: str | Path, encuesta: str, carpeta: str | Path) -> Path:
    """Primer nombre de archivo registrado (hoja `archivos`) que existe en la carpeta."""
    df = _leer_hoja(config, "archivos", {"encuesta", "alias_archivo"})
    candidatos = df.loc[df["encuesta"] == encuesta, "alias_archivo"].dropna().tolist()
    for nombre in candidatos:
        ruta = Path(carpeta) / nombre
        if ruta.exists():
            return ruta
    raise FileNotFoundError(
        f"Ninguno de los archivos registrados para '{encuesta}' existe en {carpeta}: "
        f"{candidatos}. Si el archivo cambió de nombre, agregue una fila en la hoja "
        f"'archivos' de preguntas_abiertas.xlsx."
    )


def resolver_columna(columnas: list[str], alias: list[str]) -> str:
    """Columna real que coincide (normalizada) con alguno de los alias."""
    reales = {normalizar(c): c for c in columnas}
    for a in alias:
        if normalizar(a) in reales:
            return reales[normalizar(a)]
    raise ValueError(
        f"Ninguna redacción conocida coincide con las columnas del archivo.\n"
        f"Redacciones probadas: {alias}"
    )


def lista_no_respuesta(config: str | Path) -> frozenset[str]:
    """Valores de no-respuesta (hoja `no_respuesta`), normalizados. Incluye siempre el vacío."""
    df = _leer_hoja(config, "no_respuesta", {"valor"})
    valores = {normalizar(v) for v in df["valor"].dropna()}
    return frozenset(valores | {""})


def equivalencias_roles(config: str | Path) -> dict[str, str]:
    """Hoja `roles`: {valor original normalizado: rol unificado}."""
    df = _leer_hoja(config, "roles", {"valor_original", "rol"}).dropna()
    return {normalizar(o): r.strip() for o, r in zip(df["valor_original"], df["rol"])}


def configuracion_encuesta(config: str | Path, encuesta: str) -> dict[str, list[str]]:
    """Lee la hoja `etapa1` para una encuesta: {tipo: [valores en orden]}."""
    df = _leer_hoja(config, "etapa1", {"encuesta", "tipo", "valor"})
    df = df[df["encuesta"] == encuesta].dropna(subset=["tipo", "valor"])
    tipos_raros = set(df["tipo"].str.strip()) - TIPOS_ETAPA1
    if tipos_raros:
        raise ValueError(f"Tipos no reconocidos en la hoja 'etapa1': {tipos_raros}")
    resultado: dict[str, list[str]] = {t: [] for t in TIPOS_ETAPA1}
    for tipo, valor in zip(df["tipo"].str.strip(), df["valor"].str.strip()):
        resultado[tipo].append(valor)
    if not resultado["llave"]:
        raise ValueError(f"La encuesta '{encuesta}' no tiene llaves en la hoja 'etapa1'.")
    return resultado
