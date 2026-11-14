"""
Orquestador de la Etapa 1 — Preparación de datos.

Lee la configuración, procesa todas las preguntas activas de cada encuesta y
escribe en `data/processed/etapa1/`:

  respuestas.csv            una fila por respuesta (persona × pregunta)
  respuestas_oferta.csv     una fila por copia original de cada respuesta
  catalogo_preguntas.csv    una fila por pregunta abierta
  metadatos_preparacion.csv una fila por pregunta con los conteos de control

La frase secreta para codificar el Código Cruce se toma de la variable de
entorno ETAPA1_SAL (o del argumento `sal`). Nunca se escribe en los archivos.
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
    preguntas_activas,
    resolver_columna,
)
from preparacion.respuestas import ConfigPregunta, construir_respuestas

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


def validar_salida(respuestas: pd.DataFrame, oferta: pd.DataFrame) -> list[str]:
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
    huerfanas = ~oferta["respuesta_id"].isin(respuestas["respuesta_id"])
    if huerfanas.any():
        errores.append(f"{huerfanas.sum()} filas de respuestas_oferta sin respuesta asociada.")
    sin_oferta = ~respuestas["respuesta_id"].isin(oferta["respuesta_id"])
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
        sal: frase secreta para codificar; si es None se lee de la variable de
            entorno ETAPA1_SAL y, si no existe, del archivo `archivo_sal`.
        archivo_sal: archivo local con la frase (por defecto `.etapa1_sal` en la
            raíz del repositorio, ignorado por git).

    Raises:
        ValueError: si falta la sal o la salida no cumple las validaciones.
    """
    sal = sal if sal is not None else os.environ.get("ETAPA1_SAL")
    if not sal and Path(archivo_sal).exists():
        sal = Path(archivo_sal).read_text(encoding="utf-8").strip()
    if not sal:
        raise ValueError(
            "Falta la frase secreta para codificar el Código Cruce. Defina la variable "
            f"de entorno ETAPA1_SAL o guárdela en {archivo_sal} (ese archivo está en "
            ".gitignore; nunca la escriba en el código ni la suba al repositorio)."
        )

    encuestas = encuestas or encuestas_configuradas(config)
    fecha = datetime.now(timezone.utc)
    lista_respuestas, lista_oferta, lista_catalogo, lista_meta = [], [], [], []
    hojas_leidas: dict[tuple[Path, str], pd.DataFrame] = {}

    for encuesta in encuestas:
        cfg_enc = configuracion_encuesta(config, encuesta)
        archivo = archivo_de_encuesta(config, encuesta, carpeta_cruda)
        huella = _sha256_archivo(archivo)
        libro = pd.ExcelFile(archivo)

        for preg in preguntas_activas(config, encuesta):
            hoja = preg["hoja"]
            if hoja not in libro.sheet_names:
                raise ValueError(f"La hoja '{hoja}' no existe en {archivo.name}.")
            if (archivo, hoja) not in hojas_leidas:
                print(f"Leyendo {archivo.name} / {hoja} ...")
                hojas_leidas[(archivo, hoja)] = pd.read_excel(libro, sheet_name=hoja, dtype=str)
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
            )
            resultado = construir_respuestas(df, cfg, sal)
            lista_respuestas.append(resultado.respuestas)
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
            })
            c = resultado.conteos
            print(f"  {cfg.pregunta_codigo}: {c['respuestas']:,} respuestas "
                  f"({c['respuestas_con_comentario']:,} con comentario), "
                  f"{c['copias_descartadas']:,} copias descartadas")

    respuestas = pd.concat(lista_respuestas, ignore_index=True)
    oferta = pd.concat(lista_oferta, ignore_index=True)
    catalogo = pd.DataFrame(lista_catalogo)
    metadatos = pd.DataFrame(lista_meta)

    errores = validar_salida(respuestas, oferta)
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

    if carpeta_salida is not None:
        salida = Path(carpeta_salida)
        salida.mkdir(parents=True, exist_ok=True)
        respuestas.to_csv(salida / "respuestas.csv", index=False, encoding=CODIFICACION)
        oferta.to_csv(salida / "respuestas_oferta.csv", index=False, encoding=CODIFICACION)
        catalogo.to_csv(salida / "catalogo_preguntas.csv", index=False, encoding=CODIFICACION)
        metadatos.to_csv(salida / "metadatos_preparacion.csv", index=False, encoding=CODIFICACION)
        print(f"\nEtapa 1 lista: corrida {id_corrida}, archivos en {salida}")

    return {"respuestas": respuestas, "respuestas_oferta": oferta,
            "catalogo_preguntas": catalogo, "metadatos_preparacion": metadatos}
