"""
Orquestador de la Etapa 1 — Preparación de datos.

Lee la configuración, procesa todas las preguntas activas de cada encuesta y
escribe cada corrida en su propia carpeta, `data/processed/etapa1/<id_corrida_preparacion>/`
(nunca sobrescribe una corrida anterior):

  respuestas.csv                 núcleo: una fila por respuesta (persona × pregunta)
  alertas.csv                    una fila por término de alerta encontrado (v1.10)
  contexto_<encuesta>.csv        columnas propias de cada encuesta, por respuesta
  respuestas_oferta.csv          una fila por copia original por oferta (evaluación docente)
  respuestas_programa.csv        una fila por copia original por programa (acreditación)
  catalogo_preguntas.csv         una fila por pregunta abierta
  inconsistencias.csv            una fila por respuesta y código de inconsistencia
  metadatos_preparacion.csv      una fila por pregunta con los conteos de control

Además escribe el informe de inconsistencias para el cliente (ver
`preparacion.informe`) en `reportes/etapa 1 preparación/`. Si el informe falla,
la corrida sigue y solo se avisa: el informe no es insumo de ninguna etapa.

No hay aprobación del cliente: el NLP toma la corrida más reciente
(`preparacion.corridas.ultima_corrida`) y registra su id. Las inconsistencias se
registran, no detienen el flujo.

Alertas (v1.10, decisión del 19-nov-2026): después de marcar la no-respuesta, se
buscan los términos de `alertas.xlsx` en las respuestas con comentario
(`preparacion.alertas`). Las respuestas con `tiene_alerta = True`, negadas o no,
no pasan al NLP: el NLP toma solo `es_no_respuesta = False` y `tiene_alerta = False`.

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
from preparacion.alertas import (
    DetectorAlertas,
    cargar_alertas,
    cargar_negaciones,
    detectar_alertas,
)
from preparacion.informe import CARPETA_INFORME, generar_informe
from preparacion.normalizacion import normalizar
from preparacion.respuestas import MARCAS, SEPARADOR_MARCAS, ConfigPregunta, construir_respuestas

# Tablas de copias: nombre del archivo -> tipo de la hoja `etapa1` que las alimenta.
TABLAS_COPIAS = {"respuestas_oferta": "oferta", "respuestas_programa": "programa"}

RAIZ = Path(__file__).resolve().parents[2]
CONFIG = RAIZ / "data" / "utilities" / "preguntas_abiertas.xlsx"
ALERTAS = RAIZ / "data" / "utilities" / "alertas.xlsx"
NEGACIONES = RAIZ / "data" / "utilities" / "negaciones.xlsx"
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


def validar_alertas(respuestas: pd.DataFrame, alertas: pd.DataFrame) -> list[str]:
    """Reglas del contrato para tiene_alerta y alertas.csv."""
    errores = []
    if "tiene_alerta" not in respuestas.columns:
        return ["Falta la columna tiene_alerta en respuestas."]
    tiene = respuestas["tiene_alerta"].astype(bool)
    no_resp = respuestas["es_no_respuesta"].astype(bool)
    if (tiene & no_resp).any():
        errores.append(f"{(tiene & no_resp).sum()} no-respuestas marcadas con tiene_alerta.")
    if alertas["alerta_id"].duplicated().any():
        errores.append("alerta_id repetido en alertas.")
    con_alerta = set(respuestas.loc[tiene, "respuesta_id"])
    en_tabla = set(alertas["respuesta_id"])
    if en_tabla - set(respuestas["respuesta_id"]):
        errores.append(f"{len(en_tabla - set(respuestas['respuesta_id']))} respuesta_id de alertas "
                       f"no existen en respuestas.")
    if con_alerta != en_tabla:
        errores.append("tiene_alerta no coincide con las respuestas que aparecen en alertas.")
    return errores


def ejecutar_etapa1(
    encuestas: list[str] | None = None,
    config: str | Path = CONFIG,
    carpeta_cruda: str | Path = CARPETA_CRUDA,
    carpeta_salida: str | Path | None = CARPETA_SALIDA,
    carpeta_informe: str | Path | None = None,
    alertas: str | Path = ALERTAS,
    negaciones: str | Path = NEGACIONES,
) -> dict[str, pd.DataFrame]:
    """Ejecuta la Etapa 1 y devuelve las tablas generadas.

    Args:
        encuestas: encuestas a procesar; por defecto, todas las de la hoja `etapa1`.
        carpeta_salida: carpeta de las corridas; cada corrida va en una subcarpeta con su
            id_corrida_preparacion. None para no escribir (pruebas).
        carpeta_informe: dónde escribir el informe de inconsistencias. Por defecto,
            `reportes/etapa 1 preparación/` si la salida es la carpeta normal del
            proyecto, y la carpeta de la corrida si es otra (pruebas). No se escribe
            si carpeta_salida es None.
        alertas: lista de términos de alerta (columna TERMINO).
        negaciones: léxico de negaciones; se usan sus NEGADOR y ADVERSATIVO.

    Raises:
        ValueError: si la configuración está incompleta o la salida no cumple el contrato.
    """
    encuestas = encuestas or encuestas_configuradas(config)
    no_respuesta = lista_no_respuesta(config)
    valores_cerrados = lista_valores_cerrados(config)
    roles = equivalencias_roles(config)
    detector = DetectorAlertas(cargar_alertas(alertas), cargar_negaciones(negaciones))
    fecha = datetime.now(timezone.utc)

    lista_respuestas, lista_catalogo, lista_meta = [], [], []
    listas_copias: dict[str, list[pd.DataFrame]] = {n: [] for n in TABLAS_COPIAS}
    encuestas_copias: dict[str, set[str]] = {n: set() for n in TABLAS_COPIAS}
    contextos: dict[str, list[pd.DataFrame]] = {}
    hojas_leidas: dict[tuple[Path, str], pd.DataFrame] = {}
    crudas: dict[str, pd.DataFrame] = {}  # hoja cruda completa por encuesta (para el informe)

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
            crudas.setdefault(encuesta, df)
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

    tiene_alerta, tabla_alertas = detectar_alertas(respuestas, detector)
    respuestas.insert(respuestas.columns.get_loc("es_no_respuesta") + 1, "tiene_alerta", tiene_alerta)

    errores = validar_salida(respuestas, contexto_final,
                             {n: (tablas_copias[n], encuestas_copias[n]) for n in TABLAS_COPIAS})
    errores += validar_alertas(respuestas, tabla_alertas)
    if errores:
        raise ValueError("La salida de la Etapa 1 no cumple el contrato:\n- " + "\n- ".join(errores))

    huella_corrida = hashlib.sha256(
        "".join(sorted(set(metadatos["sha256_archivo"]))).encode()
        + _sha256_archivo(Path(config)).encode() + VERSION_ETAPA1.encode()
        + _sha256_archivo(Path(alertas)).encode() + _sha256_archivo(Path(negaciones)).encode()
    ).hexdigest()[:10]
    id_corrida = f"{fecha:%Y%m%d-%H%M%S}_{huella_corrida}"
    metadatos.insert(0, "id_corrida_preparacion", id_corrida)
    metadatos.insert(1, "fecha_utc", fecha.isoformat(timespec="seconds"))
    metadatos.insert(2, "version_etapa1", VERSION_ETAPA1)
    # Conteos de alertas por pregunta (v1.10), antes de las columnas del archivo.
    por_pregunta = respuestas.assign(_alerta=respuestas["tiene_alerta"]).groupby(
        ["encuesta", "pregunta_codigo"])["_alerta"].sum()
    preg_alerta = tabla_alertas.merge(respuestas[["respuesta_id", "encuesta", "pregunta_codigo"]],
                                      on="respuesta_id")
    n_alertas = preg_alerta.groupby(["encuesta", "pregunta_codigo"]).size()
    n_negadas = preg_alerta[preg_alerta["negada"]].groupby(["encuesta", "pregunta_codigo"]).size()
    llave = list(zip(metadatos["encuesta"], metadatos["pregunta_codigo"]))
    posicion = metadatos.columns.get_loc("archivo")
    for nombre, serie in [("respuestas_con_alerta", por_pregunta), ("alertas", n_alertas),
                          ("alertas_negadas", n_negadas)]:
        metadatos.insert(posicion, nombre, [int(serie.get(k, 0)) for k in llave])
        posicion += 1
    metadatos["huella_alertas"] = _sha256_archivo(Path(alertas))[:10]
    metadatos["huella_negaciones"] = _sha256_archivo(Path(negaciones))[:10]
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

    if len(tabla_alertas):
        print(f"\nAlertas: {int(respuestas['tiene_alerta'].sum()):,} respuestas con alerta "
              f"({len(tabla_alertas):,} términos, {int(tabla_alertas['negada'].sum()):,} negados); "
              f"no pasan al NLP.")

    if carpeta_salida is not None:
        salida = Path(carpeta_salida) / id_corrida
        if salida.exists():
            raise FileExistsError(f"Ya existe la carpeta de la corrida {id_corrida}; "
                                  f"espere un segundo y vuelva a correr.")
        salida.mkdir(parents=True)
        respuestas.to_csv(salida / "respuestas.csv", index=False, encoding=CODIFICACION)
        tabla_alertas.to_csv(salida / "alertas.csv", index=False, encoding=CODIFICACION)
        for encuesta, ctx in contexto_final.items():
            ctx.to_csv(salida / f"contexto_{encuesta}.csv", index=False, encoding=CODIFICACION)
        for nombre, tabla in tablas_copias.items():
            tabla.to_csv(salida / f"{nombre}.csv", index=False, encoding=CODIFICACION)
        catalogo.to_csv(salida / "catalogo_preguntas.csv", index=False, encoding=CODIFICACION)
        inconsistencias.to_csv(salida / "inconsistencias.csv", index=False, encoding=CODIFICACION)
        metadatos.to_csv(salida / "metadatos_preparacion.csv", index=False, encoding=CODIFICACION)
        print(f"\nEtapa 1 lista: corrida {id_corrida}, archivos en {salida}")

        if carpeta_informe is None:
            mismo = Path(carpeta_salida).resolve() == CARPETA_SALIDA.resolve()
            carpeta_informe = CARPETA_INFORME if mismo else salida
        bases = sorted({f"{a.name} / {h} ({len(t):,} filas)" for (a, h), t in hojas_leidas.items()})
        try:
            ruta, hojas = generar_informe(crudas, config, no_respuesta, valores_cerrados, id_corrida,
                                          f"{fecha:%Y-%m-%d %H:%M}", bases, carpeta_informe)
            fallidas = [h.codigo for h in hojas if h.error]
            print(f"Informe de inconsistencias: {ruta} ({len(hojas)} revisiones"
                  + (f"; no se pudieron calcular: {', '.join(fallidas)})" if fallidas else ")"))
        except Exception as exc:  # el informe no detiene la corrida
            print(f"Aviso: no se pudo generar el informe de inconsistencias ({type(exc).__name__}: {exc}). "
                  f"La corrida {id_corrida} quedó completa.")

    tablas = {"respuestas": respuestas, "alertas": tabla_alertas, **tablas_copias,
              "catalogo_preguntas": catalogo, "metadatos_preparacion": metadatos,
              "inconsistencias": inconsistencias}
    tablas.update({f"contexto_{e}": c for e, c in contexto_final.items()})
    return tablas
