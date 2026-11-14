"""
Construcción de la tabla `respuestas` y de `respuestas_oferta` para UNA
pregunta abierta.

Reglas (contrato de salida, sección "respuestas"):

1. Una fila por llave. Las llaves se configuran por encuesta en la hoja
   `etapa1` de preguntas_abiertas.xlsx. Evaluación docente:
   Periodo + ID de respuesta + Id Docente. Autoevaluación:
   Periodo + Código Cruce Autoevaluación.
2. Si la base cruda trae varias copias de la misma llave (p. ej. una por
   facultad de oferta), se elige UNA, por pregunta:
   a. la que tiene comentario (no es no-respuesta);
   b. entre esas, la de texto más largo;
   c. desempate determinístico por la columna configurada (Facultad de
      oferta, orden alfabético) y luego por el orden en la base cruda.
3. Todas las copias quedan registradas en `respuestas_oferta`.
4. La no-respuesta se marca (`es_no_respuesta`), no se elimina.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pandas as pd

from preparacion.normalizacion import NO_RESPUESTA, nombre_columna, normalizar

SEPARADOR_ID = "|"
SEPARADOR_ORIGEN = "-"


def codificar(valor: object, sal: str) -> str:
    """Código de 12 caracteres (SHA-256 con sal). Vacío si no hay valor.

    El mismo valor con la misma sal siempre da el mismo código, así que las
    uniones y los conteos se conservan sin exponer el valor original.
    """
    if valor is None or (isinstance(valor, float) and valor != valor):
        return ""
    texto = str(valor).strip()
    if texto == "":
        return ""
    return hashlib.sha256((sal + texto).encode("utf-8")).hexdigest()[:12]


@dataclass
class ConfigPregunta:
    """Parámetros para construir las respuestas de una pregunta."""

    encuesta: str
    pregunta_codigo: str
    columna_texto: str
    llaves: list[str]                      # la primera debe ser el periodo
    contexto: list[str] = field(default_factory=list)
    oferta: list[str] = field(default_factory=list)
    desempate: list[str] = field(default_factory=list)
    codificar: list[str] = field(default_factory=list)
    rol: str | None = None                 # rol fijo (p. ej. "estudiante")
    columna_rol: str | None = None         # o columna que trae el rol


@dataclass
class ResultadoPregunta:
    respuestas: pd.DataFrame
    oferta: pd.DataFrame
    conteos: dict


def _validar_columnas(df: pd.DataFrame, cfg: ConfigPregunta) -> None:
    requeridas = (
        [cfg.columna_texto] + cfg.llaves + cfg.contexto + cfg.oferta
        + cfg.desempate + cfg.codificar
        + ([cfg.columna_rol] if cfg.columna_rol else [])
    )
    faltan = sorted({c for c in requeridas if c not in df.columns})
    if faltan:
        raise ValueError(
            f"[{cfg.encuesta} / {cfg.pregunta_codigo}] Faltan columnas en la "
            f"base cruda: {faltan}. Revise la hoja 'etapa1' de "
            f"preguntas_abiertas.xlsx."
        )
    if not cfg.rol and not cfg.columna_rol:
        raise ValueError(
            f"[{cfg.encuesta}] Falta definir el rol del respondiente "
            f"('rol' o 'columna_rol' en la hoja 'etapa1')."
        )


def construir_respuestas(df: pd.DataFrame, cfg: ConfigPregunta, sal: str) -> ResultadoPregunta:
    """Construye `respuestas` y `respuestas_oferta` para una pregunta.

    Args:
        df: base cruda (una hoja), idealmente leída con dtype=str.
        cfg: configuración de la pregunta.
        sal: frase secreta para codificar las columnas de `cfg.codificar`.
    """
    _validar_columnas(df, cfg)

    trabajo = df.copy()
    trabajo["_orden_crudo"] = range(len(trabajo))

    for col in cfg.codificar:
        trabajo[col] = trabajo[col].map(lambda v: codificar(v, sal))

    # Llaves como texto limpio; filas sin llave completa se excluyen y se cuentan.
    for col in cfg.llaves:
        trabajo[col] = trabajo[col].map(
            lambda v: "" if v is None or (isinstance(v, float) and v != v) else str(v).strip()
        )
    sin_llave = (trabajo[cfg.llaves] == "").any(axis=1)
    filas_sin_llave = int(sin_llave.sum())
    trabajo = trabajo.loc[~sin_llave]

    texto = trabajo[cfg.columna_texto].map(
        lambda v: "" if v is None or (isinstance(v, float) and v != v) else str(v).strip()
    )
    normalizado = texto.map(normalizar)
    trabajo["_texto"] = texto
    trabajo["_tiene_comentario"] = ~normalizado.isin(NO_RESPUESTA)
    trabajo["_largo"] = texto.str.len()
    trabajo["_texto_norm"] = normalizado

    periodo_col = cfg.llaves[0]
    origen_cols = cfg.llaves[1:]
    trabajo["_id_origen"] = trabajo[origen_cols].agg(SEPARADOR_ORIGEN.join, axis=1)
    trabajo["respuesta_id"] = (
        cfg.encuesta + SEPARADOR_ID + trabajo[periodo_col] + SEPARADOR_ID
        + trabajo["_id_origen"] + SEPARADOR_ID + cfg.pregunta_codigo
    )

    # --- Elección de copia -------------------------------------------------
    desempate = [c for c in cfg.desempate if c not in cfg.llaves]
    for col in desempate:
        trabajo[col] = trabajo[col].fillna("").astype(str)
    orden = cfg.llaves + ["_tiene_comentario", "_largo"] + desempate + ["_orden_crudo"]
    ascendente = [True] * len(cfg.llaves) + [False, False] + [True] * len(desempate) + [True]
    elegidas = (
        trabajo.sort_values(orden, ascending=ascendente, kind="mergesort")
        .drop_duplicates(cfg.llaves, keep="first")
        .sort_values("_orden_crudo", kind="mergesort")
    )

    # --- Tabla respuestas --------------------------------------------------
    if cfg.columna_rol:
        rol = elegidas[cfg.columna_rol].fillna("").astype(str).str.strip()
    else:
        rol = pd.Series(cfg.rol, index=elegidas.index)

    respuestas = pd.DataFrame({
        "respuesta_id": elegidas["respuesta_id"],
        "encuesta": cfg.encuesta,
        "pregunta_codigo": cfg.pregunta_codigo,
        "rol_respondiente": rol,
        "periodo": elegidas[periodo_col],
        "id_origen": elegidas["_id_origen"],
    })
    for col in cfg.contexto:
        respuestas[nombre_columna(col)] = elegidas[col].values
    respuestas["texto_original"] = elegidas["_texto"].values
    respuestas["es_no_respuesta"] = (~elegidas["_tiene_comentario"]).values
    respuestas = respuestas.reset_index(drop=True)

    # --- Tabla respuestas_oferta (todas las copias) -------------------------
    if cfg.oferta:
        oferta = (
            trabajo[["respuesta_id"] + cfg.oferta]
            .drop_duplicates()
            .rename(columns={c: nombre_columna(c) for c in cfg.oferta})
            .reset_index(drop=True)
        )
    else:
        oferta = pd.DataFrame({"respuesta_id": respuestas["respuesta_id"]})

    # --- Conteos de control --------------------------------------------------
    copias = trabajo.groupby(cfg.llaves, sort=False).size()
    con_comentario = trabajo.loc[trabajo["_tiene_comentario"]]
    textos_distintos = (
        con_comentario.groupby(cfg.llaves, sort=False)["_texto_norm"].nunique() > 1
    )
    conteos = {
        "encuesta": cfg.encuesta,
        "pregunta_codigo": cfg.pregunta_codigo,
        "filas_crudas": int(len(df)),
        "filas_sin_llave": filas_sin_llave,
        "respuestas": int(len(respuestas)),
        "respuestas_con_comentario": int((~respuestas["es_no_respuesta"]).sum()),
        "no_respuestas": int(respuestas["es_no_respuesta"].sum()),
        "copias_descartadas": int(len(trabajo) - len(respuestas)),
        "llaves_con_varias_copias": int((copias > 1).sum()),
        "llaves_con_comentarios_distintos": int(textos_distintos.sum()),
    }
    return ResultadoPregunta(respuestas=respuestas, oferta=oferta, conteos=conteos)
