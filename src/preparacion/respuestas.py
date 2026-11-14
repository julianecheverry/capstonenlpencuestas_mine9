"""
Construcción de las tablas de la Etapa 1 para UNA pregunta abierta.

Salidas (contrato de salida, sección "Estado: Etapa 1 implementada"):

  respuestas      núcleo común a todas las encuestas: una fila por respuesta
                  (persona × pregunta) con llave, encuesta, pregunta, periodo,
                  rol, texto y marca de no-respuesta.
  contexto        columnas propias de la encuesta (grupo, materia, facultad,
                  programa…), una fila por respuesta, unida por respuesta_id.
  oferta          una fila por copia original de cada respuesta (p. ej. una
                  por facultad de oferta), solo si la encuesta lo configura.

Reglas:

1. Una fila por llave. Las llaves se configuran por encuesta en la hoja
   `etapa1` de preguntas_abiertas.xlsx. Evaluación docente:
   Periodo + ID de respuesta + Id Docente. Autoevaluación:
   Periodo + Código Cruce Autoevaluación.
2. Si hay varias copias de la misma llave se elige UNA, por pregunta:
   a. la que tiene comentario (no es no-respuesta);
   b. entre esas, la de texto más largo;
   c. desempate determinístico por las columnas configuradas (Facultad de
      oferta, orden alfabético) y luego por el orden en la base cruda.
3. Si las copias de una misma respuesta traen comentarios distintos, se marca
   en `marca_inconsistencia` (código `comentarios_distintos_en_copias`).
4. La no-respuesta se marca (`es_no_respuesta`), no se elimina. La lista de
   valores de no-respuesta se configura en la hoja `no_respuesta`.
5. El rol se traduce a un vocabulario único con la hoja `roles`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pandas as pd

from preparacion.normalizacion import NO_RESPUESTA_POR_DEFECTO, nombre_columna, normalizar

SEPARADOR_ID = "|"
SEPARADOR_ORIGEN = "-"

# Códigos posibles de la columna `marca_inconsistencia` (vacío = sin inconsistencia).
MARCAS = {
    "comentarios_distintos_en_copias": (
        "La base cruda trae varias copias de esta respuesta con comentarios distintos; "
        "se conservó el más largo."
    ),
}


def codificar(valor: object, sal: str) -> str:
    """Código de 12 caracteres (SHA-256 con sal). Vacío si no hay valor.

    Opcional: solo se usa para las columnas listadas como `codificar` en la
    hoja `etapa1` (p. ej. en copias de trabajo que circulan fuera del cliente).
    """
    if valor is None or (isinstance(valor, float) and valor != valor):
        return ""
    texto = str(valor).strip()
    if texto == "":
        return ""
    return hashlib.sha256((sal + texto).encode("utf-8")).hexdigest()[:12]


def _texto(valor: object) -> str:
    if valor is None or (isinstance(valor, float) and valor != valor):
        return ""
    return str(valor).strip()


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
    rol: str | None = None                 # rol fijo (p. ej. "Estudiantes")
    columna_rol: str | None = None         # o columna que trae el rol
    roles: dict[str, str] | None = None    # valor normalizado -> rol unificado
    no_respuesta: frozenset[str] = NO_RESPUESTA_POR_DEFECTO


@dataclass
class ResultadoPregunta:
    respuestas: pd.DataFrame
    contexto: pd.DataFrame
    oferta: pd.DataFrame
    conteos: dict


def _validar(df: pd.DataFrame, cfg: ConfigPregunta, sal: str | None) -> None:
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
    if len(cfg.llaves) < 2:
        raise ValueError(f"[{cfg.encuesta}] Se necesitan al menos dos llaves (periodo + identificador).")
    if not cfg.rol and not cfg.columna_rol:
        raise ValueError(
            f"[{cfg.encuesta}] Falta definir el rol del respondiente "
            f"('rol' o 'columna_rol' en la hoja 'etapa1')."
        )
    if cfg.codificar and not sal:
        raise ValueError(
            f"[{cfg.encuesta}] Hay columnas para codificar {cfg.codificar} pero no hay "
            f"frase secreta (ETAPA1_SAL o .etapa1_sal)."
        )


def _unificar_roles(valores: pd.Series, cfg: ConfigPregunta) -> pd.Series:
    if cfg.roles is None:
        return valores
    normalizados = valores.map(normalizar)
    sin_equivalencia = sorted(set(valores[~normalizados.isin(cfg.roles)]))
    if sin_equivalencia:
        raise ValueError(
            f"[{cfg.encuesta}] Valores de rol sin equivalencia en la hoja 'roles': "
            f"{sin_equivalencia}. Agréguelos a preguntas_abiertas.xlsx."
        )
    return normalizados.map(cfg.roles)


def construir_respuestas(df: pd.DataFrame, cfg: ConfigPregunta, sal: str | None = None) -> ResultadoPregunta:
    """Construye respuestas, contexto y oferta para una pregunta.

    Args:
        df: base cruda (una hoja), idealmente leída con dtype=str.
        cfg: configuración de la pregunta.
        sal: frase secreta; solo se exige si `cfg.codificar` no está vacío.
    """
    _validar(df, cfg, sal)

    trabajo = df.copy()
    trabajo["_orden_crudo"] = range(len(trabajo))

    for col in cfg.codificar:
        trabajo[col] = trabajo[col].map(lambda v: codificar(v, sal))

    # Llaves como texto limpio; filas sin llave completa se excluyen y se cuentan.
    for col in cfg.llaves:
        trabajo[col] = trabajo[col].map(_texto)
    sin_llave = (trabajo[cfg.llaves] == "").any(axis=1)
    filas_sin_llave = int(sin_llave.sum())
    trabajo = trabajo.loc[~sin_llave]

    texto = trabajo[cfg.columna_texto].map(_texto)
    normalizado = texto.map(normalizar)
    trabajo["_texto"] = texto
    trabajo["_tiene_comentario"] = ~normalizado.isin(cfg.no_respuesta)
    trabajo["_largo"] = texto.str.len()
    trabajo["_texto_norm"] = normalizado

    periodo_col = cfg.llaves[0]
    trabajo["_id_origen"] = trabajo[cfg.llaves[1:]].agg(SEPARADOR_ORIGEN.join, axis=1)
    trabajo["respuesta_id"] = (
        cfg.encuesta + SEPARADOR_ID + trabajo[periodo_col] + SEPARADOR_ID
        + trabajo["_id_origen"] + SEPARADOR_ID + cfg.pregunta_codigo
    )

    # --- Elección de copia -------------------------------------------------
    desempate = [c for c in cfg.desempate if c not in cfg.llaves]
    for col in desempate:
        trabajo[col] = trabajo[col].map(_texto)
    orden = cfg.llaves + ["_tiene_comentario", "_largo"] + desempate + ["_orden_crudo"]
    ascendente = [True] * len(cfg.llaves) + [False, False] + [True] * len(desempate) + [True]
    elegidas = (
        trabajo.sort_values(orden, ascending=ascendente, kind="mergesort")
        .drop_duplicates(cfg.llaves, keep="first")
        .sort_values("_orden_crudo", kind="mergesort")
    )

    # --- Marcas de inconsistencia --------------------------------------------
    con_comentario = trabajo.loc[trabajo["_tiene_comentario"]]
    distintos = con_comentario.groupby("respuesta_id")["_texto_norm"].nunique()
    ids_distintos = set(distintos[distintos > 1].index)
    marca = elegidas["respuesta_id"].map(
        lambda rid: "comentarios_distintos_en_copias" if rid in ids_distintos else ""
    )

    # --- Núcleo: respuestas ------------------------------------------------
    if cfg.columna_rol:
        rol = elegidas[cfg.columna_rol].map(_texto)
    else:
        rol = pd.Series(cfg.rol, index=elegidas.index)
    rol = _unificar_roles(rol, cfg)

    respuestas = pd.DataFrame({
        "respuesta_id": elegidas["respuesta_id"].values,
        "encuesta": cfg.encuesta,
        "pregunta_codigo": cfg.pregunta_codigo,
        "rol_respondiente": rol.values,
        "periodo": elegidas[periodo_col].values,
        "id_origen": elegidas["_id_origen"].values,
        "texto_original": elegidas["_texto"].values,
        "es_no_respuesta": (~elegidas["_tiene_comentario"]).values,
        "marca_inconsistencia": marca.values,
    })

    # --- Contexto propio de la encuesta ---------------------------------------
    contexto = pd.DataFrame({"respuesta_id": elegidas["respuesta_id"].values})
    for col in cfg.contexto:
        contexto[nombre_columna(col)] = elegidas[col].map(_texto).values

    # --- Oferta: todas las copias (solo si la encuesta lo configura) -----------
    if cfg.oferta:
        oferta = trabajo[["respuesta_id"] + cfg.oferta].copy()
        for col in cfg.oferta:
            oferta[col] = oferta[col].map(_texto)
        oferta = (
            oferta.drop_duplicates()
            .rename(columns={c: nombre_columna(c) for c in cfg.oferta})
            .reset_index(drop=True)
        )
    else:
        oferta = pd.DataFrame(columns=["respuesta_id"])

    # --- Conteos de control --------------------------------------------------
    copias = trabajo.groupby(cfg.llaves, sort=False).size()
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
        "llaves_con_comentarios_distintos": len(ids_distintos),
        "respuestas_con_marca_inconsistencia": int((marca != "").sum()),
    }
    return ResultadoPregunta(respuestas=respuestas, contexto=contexto, oferta=oferta, conteos=conteos)
