"""
Orquestador de la Etapa 1 — Preparación de datos.

Lee la configuración, procesa todas las preguntas activas de cada encuesta y
escribe en `data/processed/etapa1/`:

  respuestas.csv                 núcleo: una fila por respuesta (persona × pregunta)
  contexto_<encuesta>.csv        columnas propias de cada encuesta, por respuesta
  respuestas_oferta.csv          una fila por copia original (encuestas que lo configuran)
  catalogo_preguntas.csv         una fila por pregunta abierta
  inconsistencias.csv            respuestas con `marca_inconsistencia`, para revisión
  metadatos_preparacion.csv      una fila por pregunta con los conteos de control

Si la hoja `etapa1` lista columnas para codificar, la frase secreta se toma
de la variable de entorno ETAPA1_SAL o del archivo `.etapa1_sal`.
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from preparacion import VERSION_ETAPA1
from preparacion.configuracion import (
    archivo_de_encuesta,
    configuracion_encuesta,
    encuestas_configuradas,
    equivalencias_roles,
    lista_no_respuesta,
    preguntas_activas,
    resolver_columna,
)
from preparacion.respuestas import MARCAS, ConfigPregunta, construir_respuestas

RAIZ = Path(__file__).resolve().parents[2]
CONFIG = RAIZ / "data" / "utilities" / "preguntas_abiertas.xlsx"
CARPETA_CRUDA = RAIZ / "data" / "raw"
CARPETA_SALIDA = RAIZ / "data" / "processed" / "etapa1"
CODIFICACION = "utf-8-sig"  # se abre bien en Excel y en Power BI
ARCHIVO_SAL = RAIZ / ".etapa1_sal"


def _sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _resolver(columnas: list[str], nombres: list[str]) -> list[str]:
    return [resolver_columna(columnas, [n]) for n in nombres]


def _obtener_sal(sal: str | None, archivo_sal: str | Path) -> str | None:
    if sal:
        return sal
    if os.environ.get("ETAPA1_SAL"):
        return os.environ["ETAPA1_SAL"]
    if Path(archivo_sal).exists():
        return Path(archivo_sal).read_text(encoding="utf-8").strip() or None
    return None


def validar_salida(respuestas: pd.DataFrame, contextos: dict[str, pd.DataFrame],
                   oferta: pd.DataFrame, encuestas_con_oferta: set[str]) -> list[str]:
    """Reglas de validación del contrato que aplican a la Etapa 1."""
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
    marcas_raras = set(respuestas["marca_inconsistencia"]) - set(MARCAS) - {""}
    if marcas_raras:
        errores.append(f"Valores no permitidos en marca_inconsistencia: {marcas_raras}.")
    for encuesta, ctx in contextos.items():
        esperadas = set(respuestas.loc[respuestas["encuesta"] == encuesta, "respuesta_id"])
        if set(ctx["respuesta_id"]) != esperadas or ctx["respuesta_id"].duplicated().any():
            errores.append(f"contexto_{encuesta} no tiene exactamente una fila por respuesta.")
    huerfanas = ~oferta["respuesta_id"].isin(respuestas["respuesta_id"])
    if huerfanas.any():
        errores.append(f"{huerfanas.sum()} filas de respuestas_oferta sin respuesta asociada.")
    con_oferta = respuestas["encuesta"].isin(encuestas_con_oferta)
    sin_oferta = con_oferta & ~respuestas["respuesta_id"].isin(oferta["respuesta_id"])
    if sin_oferta.any():
        errores.append(f"{sin_oferta.sum()} respuestas sin fila en respuestas_oferta.")
    return errores


def ejecutar_etapa1(
    encuestas: list[str] | None = None,
    config: str | Path = CONFIG,
    carpeta_cruda: str | Path = CARPETA_CRUDA,
    carpeta_salida: str | Path | None = CARPETA_SALIDA,
    sal: str | None = None,
    archivo_sal: str | Path = ARCHIVO_SAL,
) -> dict[str, pd.DataFrame]:
    """Ejecuta la Etapa 1 y devuelve las tablas generadas.

    Args:
        encuestas: encuestas a procesar; por defecto, todas las de la hoja `etapa1`.
        carpeta_salida: dónde escribir los CSV; None para no escribir (pruebas).
        sal: frase secreta, solo necesaria si se configuran columnas a codificar.
        archivo_sal: archivo local con la frase (ignorado por git).

    Raises:
        ValueError: si la configuración está incompleta o la salida no cumple el contrato.
    """
    sal = _obtener_sal(sal, archivo_sal)
    encuestas = encuestas or encuestas_configuradas(config)
    no_respuesta = lista_no_respuesta(config)
    roles = equivalencias_roles(config)
    fecha = datetime.now(timezone.utc)

    lista_respuestas, lista_oferta, lista_catalogo, lista_meta = [], [], [], []
    contextos: dict[str, list[pd.DataFrame]] = {}
    encuestas_con_oferta: set[str] = set()
    hojas_leidas: dict[tuple[Path, str], pd.DataFrame] = {}

    for encuesta in encuestas:
        cfg_enc = configuracion_encuesta(config, encuesta)
        archivo = archivo_de_encuesta(config, encuesta, carpeta_cruda)
        huella = _sha256_archivo(archivo)
        libro = pd.ExcelFile(archivo)
        if cfg_enc["oferta"]:
            encuestas_con_oferta.add(encuesta)

        for preg in preguntas_activas(config, encuesta):
            hoja = preg["hoja"]
            if hoja not in libro.sheet_names:
                raise ValueError(f"La hoja '{hoja}' no existe en {archivo.name}.")
            if (archivo, hoja) not in hojas_leidas:
                print(f"Leyendo {archivo.name} / {hoja} ...")
                hojas_leidas[(archivo, hoja)] = pd.read_excel(libro, sheet_name=hoja, dtype=str,
                                                             keep_default_na=False)
            df = hojas_leidas[(archivo, hoja)]
            columnas = list(df.columns)

            cfg = ConfigPregunta(
                encuesta=encuesta,
                pregunta_codigo=preg["pregunta_codigo"],
                columna_texto=resolver_columna(columnas, preg["alias"]),
                llaves=_resolver(columnas, cfg_enc["llave"]),
                contexto=_resolver(columnas, cfg_enc["contexto"]),
                oferta=_resolver(columnas, cfg_enc["oferta"]),
                desempate=_resolver(columnas, cfg_enc["desempate"]),
                codificar=_resolver(columnas, cfg_enc["codificar"]),
                rol=(cfg_enc["rol"] or [None])[0],
                columna_rol=(_resolver(columnas, cfg_enc["columna_rol"]) or [None])[0],
                roles=roles,
                no_respuesta=no_respuesta,
            )
            resultado = construir_respuestas(df, cfg, sal)
            lista_respuestas.append(resultado.respuestas)
            contextos.setdefault(encuesta, []).append(resultado.contexto)
            lista_oferta.append(resultado.oferta)
            lista_catalogo.append({
                "pregunta_codigo": cfg.pregunta_codigo,
                "encuesta": encuesta,
                "texto_pregunta": cfg.columna_texto.strip(),
                "rol_respondiente": cfg.rol or f"según columna '{cfg.columna_rol}'",
                "hoja": hoja,
            })
            lista_meta.append({
                **resultado.conteos,
                "archivo": archivo.name,
                "sha256_archivo": huella,
                "hoja": hoja,
                "columna_texto": cfg.columna_texto.strip(),
                "columnas_codificadas": "; ".join(cfg.codificar),
            })
            c = resultado.conteos
            print(f"  {cfg.pregunta_codigo}: {c['respuestas']:,} respuestas "
                  f"({c['respuestas_con_comentario']:,} con comentario), "
                  f"{c['copias_descartadas']:,} copias descartadas")

    respuestas = pd.concat(lista_respuestas, ignore_index=True)
    contexto_final = {e: pd.concat(t, ignore_index=True) for e, t in contextos.items()}
    oferta = pd.concat(lista_oferta, ignore_index=True)
    catalogo = pd.DataFrame(lista_catalogo)
    metadatos = pd.DataFrame(lista_meta)

    errores = validar_salida(respuestas, contexto_final, oferta, encuestas_con_oferta)
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
        oferta.to_csv(salida / "respuestas_oferta.csv", index=False, encoding=CODIFICACION)
        catalogo.to_csv(salida / "catalogo_preguntas.csv", index=False, encoding=CODIFICACION)
        inconsistencias.to_csv(salida / "inconsistencias.csv", index=False, encoding=CODIFICACION)
        metadatos.to_csv(salida / "metadatos_preparacion.csv", index=False, encoding=CODIFICACION)
        print(f"\nEtapa 1 lista: corrida {id_corrida}, archivos en {salida}")

    tablas = {"respuestas": respuestas, "respuestas_oferta": oferta,
              "catalogo_preguntas": catalogo, "metadatos_preparacion": metadatos,
              "inconsistencias": inconsistencias}
    tablas.update({f"contexto_{e}": c for e, c in contexto_final.items()})
    return tablas
