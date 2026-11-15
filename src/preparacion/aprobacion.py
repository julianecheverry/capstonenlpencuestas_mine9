"""
Aprobación de corridas de la Etapa 1 (punto de control antes del NLP).

Cada corrida de la Etapa 1 queda en `data/processed/etapa1/<id_corrida_preparacion>/`.
Cuando el cliente aprueba una (por correo), se registra con `aprobar(...)`
o `python run_aprobar_corrida.py`, que agrega una fila a
`data/processed/etapa1/aprobaciones.csv`. El NLP procesa la última corrida
aprobada: `ultima_aprobada(...)`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ARCHIVO = "aprobaciones.csv"
COLUMNAS = ["id_corrida_preparacion", "fecha_aprobacion_utc", "aprobado_por", "referencia"]
CODIFICACION = "utf-8-sig"


def corridas(carpeta: str | Path) -> list[str]:
    """Corridas guardadas, de la más antigua a la más reciente (el id empieza con la fecha)."""
    return sorted(p.name for p in Path(carpeta).iterdir()
                  if p.is_dir() and (p / "metadatos_preparacion.csv").exists())


def ultima_corrida(carpeta: str | Path) -> Path:
    """Carpeta de la corrida más reciente, aprobada o no (para revisiones como la conciliación)."""
    todas = corridas(carpeta)
    if not todas:
        raise FileNotFoundError(f"No hay corridas de la Etapa 1 en {carpeta}.")
    return Path(carpeta) / todas[-1]


def leer_aprobaciones(carpeta: str | Path) -> pd.DataFrame:
    ruta = Path(carpeta) / ARCHIVO
    if not ruta.exists():
        return pd.DataFrame(columns=COLUMNAS)
    return pd.read_csv(ruta, dtype=str, keep_default_na=False, encoding=CODIFICACION)


def aprobar(id_corrida: str, aprobado_por: str, referencia: str, carpeta: str | Path) -> pd.Series:
    """Registra la aprobación de una corrida y devuelve la fila agregada.

    Raises:
        FileNotFoundError: si la corrida no existe en `carpeta`.
        ValueError: si faltan datos, la carpeta no corresponde a ese id o ya estaba aprobada.
    """
    carpeta = Path(carpeta)
    if not aprobado_por.strip() or not referencia.strip():
        raise ValueError("Indique quién aprobó y la referencia del correo (asunto o fecha).")
    corrida = carpeta / id_corrida
    meta = corrida / "metadatos_preparacion.csv"
    if not meta.exists():
        raise FileNotFoundError(f"No existe la corrida {id_corrida} en {carpeta}. "
                                f"Corridas disponibles: {corridas(carpeta) if carpeta.exists() else []}")
    ids = set(pd.read_csv(meta, dtype=str, usecols=["id_corrida_preparacion"])["id_corrida_preparacion"])
    if ids != {id_corrida}:
        raise ValueError(f"La carpeta {corrida} trae otro id_corrida_preparacion: {sorted(ids)}.")
    aprobaciones = leer_aprobaciones(carpeta)
    if id_corrida in set(aprobaciones["id_corrida_preparacion"]):
        raise ValueError(f"La corrida {id_corrida} ya estaba aprobada.")
    fila = pd.Series({
        "id_corrida_preparacion": id_corrida,
        "fecha_aprobacion_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "aprobado_por": aprobado_por.strip(),
        "referencia": referencia.strip(),
    })
    pd.concat([aprobaciones, fila.to_frame().T], ignore_index=True)[COLUMNAS].to_csv(
        carpeta / ARCHIVO, index=False, encoding=CODIFICACION)
    return fila


def ultima_aprobada(carpeta: str | Path) -> tuple[str, Path]:
    """Id y carpeta de la última corrida aprobada.

    Raises:
        ValueError: si ninguna corrida está aprobada.
        FileNotFoundError: si la carpeta de la corrida aprobada ya no existe.
    """
    aprobaciones = leer_aprobaciones(carpeta)
    if aprobaciones.empty:
        raise ValueError(f"Ninguna corrida de la Etapa 1 está aprobada ({Path(carpeta) / ARCHIVO}). "
                         f"El NLP no debe correr sin una corrida aprobada.")
    id_corrida = aprobaciones["id_corrida_preparacion"].iloc[-1]
    corrida = Path(carpeta) / id_corrida
    if not (corrida / "respuestas.csv").exists():
        raise FileNotFoundError(f"La corrida aprobada {id_corrida} no está en {carpeta}.")
    return id_corrida, corrida
