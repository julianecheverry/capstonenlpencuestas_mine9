"""
Orquestador de la Etapa 1 — Preparación de datos.

Lee la configuración, procesa todas las preguntas activas de cada encuesta y
escribe en `data/processed/etapa1/`:

  respuestas.csv                 núcleo: una fila por respuesta (persona × pregunta)
  contexto_<encuesta>.csv        columnas propias de cada encuesta, por respuesta
  respuestas_oferta.csv          una fila por copia original por oferta (evaluación docente)
  respuestas_programa.csv        una fila por copia original por programa (acreditación)
  catalogo_preguntas.csv         una fila por pregunta abierta
  inconsistencias.csv            una fila por respuesta y código de inconsistencia
  metadatos_preparacion.csv      una fila por pregunta con los conteos de control

El Código Cruce y las demás columnas se usan tal como vienen en la base que se
recibe. Si una copia de los datos debe salir del cliente, se anonimiza la base
cruda antes de correr la Etapa 1.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from preparacion import VERSION_ETAPA1
from preparacion.configuracion import (
    archivo_de_encuesta,
    asignar_preguntas,
    configuracion_encuesta,
    encuestas_configuradas,
    equivalencias_roles,
    interpretar_filtros,
    interpretar_renombres,
    lista_no_respuesta,
    lista_valores_cerrados,
    preguntas_activas,
    resolver_columna,
)
from preparacion.normalizacion import normalizar
from preparacion.respuestas import MARCAS, SEPARADOR_MARCAS, ConfigPregunta, construir_respuestas

# Tablas de copias: nombre del archivo -> tipo de la hoja `etapa1` que las alimenta.
TABLAS_COPIAS = {"respuestas_oferta": "oferta", "respuestas_programa": "programa"}

RAIZ = Path(__file__).resolve().parents[2]
CONFIG = RAIZ / "data" / "utilities" / "preguntas_abiertas.xlsx"
CARPETA_CRUDA = RAIZ / "data" / "raw"
CARPETA_SALIDA = RAIZ / "data" / "processed" / "etapa1"
CODIFICACION = "utf-8-sig"  # se abre bien en Excel y en Power BI


def _sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _resolver(columnas: list[str], nombres: list[str]) -> list[str]:
    return [resolver_columna(columnas, [n]) for n in nombres]


def validar_salida(respuestas: pd.DataFrame, contextos: dict[str, pd.DataFrame],
                   copias: dict[str, tuple[pd.DataFrame, set[str]]]) -> list[str]:
    """Reglas de validación del contrato que aplican a la Etapa 1.

    Args:
        copias: {nombre de tabla: (tabla, encuestas que la configuran)}, p. ej.
            {"respuestas_oferta": (oferta, {"evaluacion_docente_estudiantes"})}.
    """
    errores = []
    duplicados = respuestas["respuesta_id"].duplicated().sum()
    if duplicados:
        errores.append(f"respuesta_id repetido en {duplicados} filas.")
    obligatorias = ["respuesta_id", "encuesta", "pregunta_codigo", "rol_respondiente",
                    "periodo", "id_origen", "es_no_respuesta"]
    for col in obligatorias:
        vacias = (respuestas[col].isna() | (respuestas[col].astype(str).str.strip() == "")).sum()
        if vacias:
            errores.append(f"La columna obligatoria '{col}' tiene {vacias} valores vacíos.")
    codigos = {c for m in respuestas["marca_inconsistencia"] for c in str(m).split(SEPARADOR_MARCAS) if c}
    marcas_raras = codigos - set(MARCAS)
    if marcas_raras:
        errores.append(f"Valores no permitidos en marca_inconsistencia: {marcas_raras}.")
    for encuesta, ctx in contextos.items():
        esperadas = set(respuestas.loc[respuestas["encuesta"] == encuesta, "respuesta_id"])
        if set(ctx["respuesta_id"]) != esperadas or ctx["respuesta_id"].duplicated().any():
            errores.append(f"contexto_{encuesta} no tiene exactamente una fila por respuesta.")
    for nombre, (tabla, encuestas_tabla) in copias.items():
        huerfanas = ~tabla["respuesta_id"].isin(respuestas["respuesta_id"])
        if huerfanas.any():
            errores.append(f"{huerfanas.sum()} filas de {nombre} sin respuesta asociada.")
        deberian = respuestas["encuesta"].isin(encuestas_tabla)
        faltan = deberian & ~respuestas["respuesta_id"].isin(tabla["respuesta_id"])
        if faltan.any():
            errores.append(f"{faltan.sum()} respuestas sin fila en {nombre}.")
        ajenas = ~tabla["respuesta_id"].isin(respuestas.loc[deberian, "respuesta_id"])
        if ajenas.any():
            errores.append(f"{ajenas.sum()} filas de {nombre} de encuestas que no la configuran.")
    return errores


def ejecutar_etapa1(
    encuestas: list[str] | None = None,
    config: str | Path = CONFIG,
    carpeta_cruda: str | Path = CARPETA_CRUDA,
    carpeta_salida: str | Path | None = CARPETA_SALIDA,
) -> dict[str, pd.DataFrame]:
    """Ejecuta la Etapa 1 y devuelve las tablas generadas.

    Args:
        encuestas: encuestas a procesar; por defecto, todas las de la hoja `etapa1`.
        carpeta_salida: dónde escribir los CSV; None para no escribir (pruebas).

    Raises:
        ValueError: si la configuración está incompleta o la salida no cumple el contrato.
    """
    encuestas = encuestas or encuestas_configuradas(config)
    no_respuesta = lista_no_respuesta(config)
    valores_cerrados = lista_valores_cerrados(config)
    roles = equivalencias_roles(config)
    fecha = datetime.now(timezone.utc)

    lista_respuestas, lista_catalogo, lista_meta = [], [], []
    listas_copias: dict[str, list[pd.DataFrame]] = {n: [] for n in TABLAS_COPIAS}
    encuestas_copias: dict[str, set[str]] = {n: set() for n in TABLAS_COPIAS}
    contextos: dict[str, list[pd.DataFrame]] = {}
    hojas_leidas: dict[tuple[Path, str], pd.DataFrame] = {}

    for encuesta in encuestas:
        cfg_enc = configuracion_encuesta(config, encuesta)
        archivo = archivo_de_encuesta(config, encuesta, carpeta_cruda)
        huella = _sha256_archivo(archivo)
        libro = pd.ExcelFile(archivo)
        for nombre, tipo in TABLAS_COPIAS.items():
            if cfg_enc[tipo]:
                encuestas_copias[nombre].add(encuesta)

        formato = normalizar((cfg_enc["formato"] or ["ancho"])[0])
        if formato not in {"ancho", "largo"}:
            raise ValueError(f"[{encuesta}] Formato no reconocido en la hoja 'etapa1': {formato}. "
                             f"Use «ancho» o «largo».")
        if formato == "largo" and not (cfg_enc["columna_pregunta"] and cfg_enc["columna_texto"]):
            raise ValueError(f"[{encuesta}] El formato largo necesita 'columna_pregunta' y "
                             f"'columna_texto' en la hoja 'etapa1'.")
        preguntas = preguntas_activas(config, encuesta)
        codigos_largo = None

        for preg in preguntas:
            hoja = preg["hoja"]
            if hoja not in libro.sheet_names:
                raise ValueError(f"La hoja '{hoja}' no existe en {archivo.name}.")
            if (archivo, hoja) not in hojas_leidas:
                print(f"Leyendo {archivo.name} / {hoja} ...")
                hojas_leidas[(archivo, hoja)] = pd.read_excel(libro, sheet_name=hoja, dtype=str,
                                                             keep_default_na=False)
            df = hojas_leidas[(archivo, hoja)]
            columnas = list(df.columns)

            if formato == "largo":
                # Cada fila ya es una respuesta: se toman las filas cuya redacción
                # corresponde a esta pregunta y el texto sale de `columna_texto`.
                if codigos_largo is None:
                    col_pregunta = resolver_columna(columnas, cfg_enc["columna_pregunta"][:1])
                    codigos_largo = asignar_preguntas(df[col_pregunta], preguntas, encuesta)
                df = df.loc[codigos_largo == preg["pregunta_codigo"]].reset_index(drop=True)
                columna_texto = resolver_columna(columnas, cfg_enc["columna_texto"][:1])
                texto_pregunta = preg["alias"][0].strip()
                if df.empty:
                    print(f"  Aviso: {preg['pregunta_codigo']} no tiene filas en {archivo.name}.")
                    continue
            else:
                columna_texto = resolver_columna(columnas, preg["alias"])
                texto_pregunta = columna_texto.strip()

            cfg = ConfigPregunta(
                encuesta=encuesta,
                pregunta_codigo=preg["pregunta_codigo"],
                columna_texto=columna_texto,
                llaves=_resolver(columnas, cfg_enc["llave"]),
                contexto=_resolver(columnas, cfg_enc["contexto"]),
                oferta=_resolver(columnas, cfg_enc["oferta"]),
                programa=_resolver(columnas, cfg_enc["programa"]),
                renombrar={resolver_columna(columnas, [col]): nuevo
                           for col, nuevo in interpretar_renombres(cfg_enc["renombrar"]).items()},
                desempate=_resolver(columnas, cfg_enc["desempate"]),
                rol=(cfg_enc["rol"] or [None])[0],
                columna_rol=(_resolver(columnas, cfg_enc["columna_rol"]) or [None])[0],
                roles=roles,
                no_respuesta=no_respuesta,
                valores_cerrados=valores_cerrados,
                filtros={resolver_columna(columnas, [col]): valores
                         for col, valores in interpretar_filtros(cfg_enc["filtro"]).items()},
            )
            resultado = construir_respuestas(df, cfg)
            lista_respuestas.append(resultado.respuestas)
            contextos.setdefault(encuesta, []).append(resultado.contexto)
            listas_copias["respuestas_oferta"].append(resultado.oferta)
            listas_copias["respuestas_programa"].append(resultado.programa)
            lista_catalogo.append({
                "pregunta_codigo": cfg.pregunta_codigo,
                "encuesta": encuesta,
                "texto_pregunta": texto_pregunta,
                "rol_respondiente": cfg.rol or f"según columna '{cfg.columna_rol}'",
                "hoja": hoja,
                "enfoque": preg["enfoque"],
                "par_espejo": preg["par_espejo"],
            })
            lista_meta.append({
                **resultado.conteos,
                "archivo": archivo.name,
                "sha256_archivo": huella,
                "hoja": hoja,
                "columna_texto": cfg.columna_texto.strip()
            })
            c = resultado.conteos
            print(f"  {cfg.pregunta_codigo}: {c['respuestas']:,} respuestas "
                  f"({c['respuestas_con_comentario']:,} con comentario), "
                  f"{c['copias_descartadas']:,} copias descartadas, "
                  f"{c['filas_excluidas_por_filtro']:,} filas excluidas por filtro")

    respuestas = pd.concat(lista_respuestas, ignore_index=True)
    contexto_final = {e: pd.concat(t, ignore_index=True) for e, t in contextos.items()}
    tablas_copias = {}
    for nombre, lista in listas_copias.items():
        no_vacias = [t for t in lista if len(t)]
        tablas_copias[nombre] = (pd.concat(no_vacias, ignore_index=True) if no_vacias
                                 else pd.DataFrame(columns=["respuesta_id"]))
    catalogo = pd.DataFrame(lista_catalogo)
    metadatos = pd.DataFrame(lista_meta)

    errores = validar_salida(respuestas, contexto_final,
                             {n: (tablas_copias[n], encuestas_copias[n]) for n in TABLAS_COPIAS})
    if errores:
        raise ValueError("La salida de la Etapa 1 no cumple el contrato:\n- " + "\n- ".join(errores))

    huella_corrida = hashlib.sha256(
        "".join(sorted(set(metadatos["sha256_archivo"]))).encode()
        + _sha256_archivo(Path(config)).encode() + VERSION_ETAPA1.encode()
    ).hexdigest()[:10]
    id_corrida = f"{fecha:%Y%m%d-%H%M%S}_{huella_corrida}"
    metadatos.insert(0, "id_corrida_preparacion", id_corrida)
    metadatos.insert(1, "fecha_utc", fecha.isoformat(timespec="seconds"))
    metadatos.insert(2, "version_etapa1", VERSION_ETAPA1)
    respuestas["id_corrida_preparacion"] = id_corrida

    marcadas = respuestas[respuestas["marca_inconsistencia"] != ""]
    inconsistencias = marcadas[["respuesta_id", "encuesta", "pregunta_codigo", "periodo",
                                "id_origen", "marca_inconsistencia"]].copy()
    # Una fila por respuesta y código (una respuesta puede tener varios).
    inconsistencias["marca_inconsistencia"] = inconsistencias["marca_inconsistencia"].str.split(SEPARADOR_MARCAS)
    inconsistencias = inconsistencias.explode("marca_inconsistencia", ignore_index=True)
    inconsistencias["detalle"] = inconsistencias["marca_inconsistencia"].map(MARCAS)
    if len(inconsistencias):
        print(f"\nAviso: {len(inconsistencias):,} respuestas con marca de inconsistencia "
              f"(ver inconsistencias.csv): "
              f"{inconsistencias['marca_inconsistencia'].value_counts().to_dict()}")

    if carpeta_salida is not None:
        salida = Path(carpeta_salida)
        salida.mkdir(parents=True, exist_ok=True)
        actuales = {f"contexto_{e}.csv" for e in contexto_final}
        sobrantes = sorted(p.name for p in salida.glob("contexto_*.csv") if p.name not in actuales)
        if sobrantes:
            print(f"Aviso: en {salida} hay archivos de contexto de encuestas que no se "
                  f"procesaron en esta corrida (no se modifican): {sobrantes}")
        respuestas.to_csv(salida / "respuestas.csv", index=False, encoding=CODIFICACION)
        for encuesta, ctx in contexto_final.items():
            ctx.to_csv(salida / f"contexto_{encuesta}.csv", index=False, encoding=CODIFICACION)
        for nombre, tabla in tablas_copias.items():
            tabla.to_csv(salida / f"{nombre}.csv", index=False, encoding=CODIFICACION)
        catalogo.to_csv(salida / "catalogo_preguntas.csv", index=False, encoding=CODIFICACION)
        inconsistencias.to_csv(salida / "inconsistencias.csv", index=False, encoding=CODIFICACION)
        metadatos.to_csv(salida / "metadatos_preparacion.csv", index=False, encoding=CODIFICACION)
        print(f"\nEtapa 1 lista: corrida {id_corrida}, archivos en {salida}")

    tablas = {"respuestas": respuestas, **tablas_copias,
              "catalogo_preguntas": catalogo, "metadatos_preparacion": metadatos,
              "inconsistencias": inconsistencias}
    tablas.update({f"contexto_{e}": c for e, c in contexto_final.items()})
    return tablas
