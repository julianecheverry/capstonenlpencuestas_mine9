"""
Lectura de la configuración de la Etapa 1 desde preguntas_abiertas.xlsx.

Reutiliza las hojas que ya existen (`preguntas`, `archivos`) y agrega una
hoja nueva, `etapa1`, con tres columnas: encuesta | tipo | valor.

Tipos admitidos en la hoja `etapa1`:
  llave       Columnas que identifican una respuesta, en orden. La primera
              debe ser el periodo. Ej.: Periodo, ID de respuesta, Id Docente.
  contexto    Columnas que van a la tabla `contexto_<encuesta>` (de la copia elegida).
  oferta      Columnas que se registran por copia en `respuestas_oferta`.
  programa    Columnas que se registran por copia en `respuestas_programa`
              (encuestas que repiten el envío por programa: acreditación).
  renombrar   Nombre de una columna en las salidas, con la forma
              "Columna = nombre_salida" (ej.: "llave = llave_oac"). Sin esto,
              el nombre es la columna en snake_case.
  desempate   Columnas para desempatar copias (orden alfabético).
  formato     OPCIONAL. «ancho» (por defecto: cada pregunta es una columna) o
              «largo» (cada fila es una respuesta y la pregunta viene como
              texto en una columna, como en la encuesta de calidad).
  columna_pregunta  Solo formato largo: columna que trae el texto de la pregunta.
  columna_texto     Solo formato largo: columna que trae la respuesta.
  rol         Rol fijo del respondiente (ej.: Estudiantes, Docentes).
  columna_rol Columna de la base cruda que trae el rol (ej.: Población).
  filtro      Condición para incluir una fila, con la forma "Columna = valor"
              o "Columna = valor1; valor2" (ej.: "Evaluación = Sí"). Los
              valores se comparan normalizados (Si, SI y Sí son iguales).

Otras hojas nuevas:
  no_respuesta  columna `valor`: textos que cuentan como no-respuesta. Se
                comparan normalizados (sin tildes, mayúsculas ni puntuación),
                así que "Ninguna." y "NINGUNA" equivalen a "ninguna".
  valores_cerrados  columna `valor`: valores de escala cerrada (De acuerdo,
                N/S…) que, si aparecen en la columna de una pregunta abierta,
                se tratan como no-respuesta y se marcan como inconsistencia.
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

TIPOS_ETAPA1 = {"llave", "contexto", "oferta", "programa", "renombrar", "desempate",
                "formato", "columna_pregunta", "columna_texto", "rol", "columna_rol", "filtro"}


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
        opcional = lambda col: (grupo[col].dropna().iloc[0] if col in grupo and grupo[col].notna().any()
                                else "")
        preguntas.append({
            "pregunta_codigo": clave,
            "alias": grupo["alias_redaccion"].dropna().tolist(),
            "hoja": grupo["hoja_excel"].iloc[0],
            "enfoque": opcional("enfoque"),        # columnas opcionales de la hoja
            "par_espejo": opcional("par_espejo"),
        })
    return preguntas


def asignar_preguntas(redacciones: pd.Series, preguntas: list[dict], encuesta: str) -> pd.Series:
    """Formato largo: código de pregunta de cada fila según su redacción.

    Compara las redacciones normalizadas (sin tildes, mayúsculas, puntuación ni
    espacios repetidos) con los alias de la hoja `preguntas`. Si alguna fila trae
    una redacción no registrada, se detiene: nada nuevo se procesa por error.
    """
    mapa: dict[str, str] = {}
    for preg in preguntas:
        for alias in preg["alias"]:
            clave = normalizar(alias)
            if clave in mapa and mapa[clave] != preg["pregunta_codigo"]:
                raise ValueError(f"[{encuesta}] La redacción «{alias}» está registrada en dos "
                                 f"códigos: {mapa[clave]} y {preg['pregunta_codigo']}.")
            mapa[clave] = preg["pregunta_codigo"]
    codigos = redacciones.map(lambda v: mapa.get(normalizar(v)))
    sin_codigo = redacciones[codigos.isna()]
    if len(sin_codigo):
        conteo = sin_codigo.str.strip().value_counts()
        lista = "\n".join(f"  - ({n} filas) {r}" for r, n in conteo.head(10).items())
        raise ValueError(
            f"[{encuesta}] {len(sin_codigo)} filas traen una pregunta que no está registrada en la "
            f"hoja 'preguntas' ({len(conteo)} redacciones distintas). Agréguelas con su código:\n{lista}"
        )
    return codigos


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


def lista_valores_cerrados(config: str | Path) -> frozenset[str]:
    """Valores de escala cerrada (hoja `valores_cerrados`), normalizados."""
    df = _leer_hoja(config, "valores_cerrados", {"valor"})
    return frozenset(normalizar(v) for v in df["valor"].dropna() if normalizar(v))


def interpretar_renombres(renombres: list[str]) -> dict[str, str]:
    """Convierte ["llave = llave_oac"] en {"llave": "llave_oac"} (columna sin resolver)."""
    resultado: dict[str, str] = {}
    for item in renombres:
        if "=" not in item:
            raise ValueError(f"Renombre mal escrito en la hoja 'etapa1': '{item}'. "
                             f"Use la forma 'Columna = nombre_salida'.")
        columna, nuevo = (x.strip() for x in item.split("=", 1))
        if not columna or not nuevo:
            raise ValueError(f"Renombre incompleto en la hoja 'etapa1': '{item}'.")
        resultado[columna] = nuevo
    return resultado


def equivalencias_roles(config: str | Path) -> dict[str, str]:
    """Hoja `roles`: {valor original normalizado: rol unificado}."""
    df = _leer_hoja(config, "roles", {"valor_original", "rol"}).dropna()
    return {normalizar(o): r.strip() for o, r in zip(df["valor_original"], df["rol"])}


def interpretar_filtros(filtros: list[str]) -> dict[str, set[str]]:
    """Convierte ["Evaluación = Sí"] en {"Evaluación": {"si"}} (columna sin resolver)."""
    resultado: dict[str, set[str]] = {}
    for filtro in filtros:
        if "=" not in filtro:
            raise ValueError(f"Filtro mal escrito en la hoja 'etapa1': '{filtro}'. "
                             f"Use la forma 'Columna = valor'.")
        columna, valores = filtro.split("=", 1)
        aceptados = {normalizar(v) for v in valores.split(";") if v.strip()}
        if not aceptados:
            raise ValueError(f"El filtro '{filtro}' no tiene valores aceptados.")
        resultado.setdefault(columna.strip(), set()).update(aceptados)
    return resultado


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
