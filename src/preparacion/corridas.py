"""
Corridas guardadas de la Etapa 1.

Cada corrida queda en `data/processed/etapa1/<id_corrida_preparacion>/` y nunca se
sobrescribe. No hay aprobación del cliente (decisión del equipo, 16-nov): el NLP
toma la corrida más reciente con `ultima_corrida(...)` y registra su
`id_corrida_preparacion` en `metadatos_nlp`, para saber siempre qué datos usó.
Las inconsistencias quedan registradas en `inconsistencias.csv` y en el informe
para el cliente, sin detener el flujo.
"""
from __future__ import annotations

from pathlib import Path


def corridas(carpeta: str | Path) -> list[str]:
    """Corridas completas, de la más antigua a la más reciente (el id empieza con la fecha)."""
    carpeta = Path(carpeta)
    if not carpeta.exists():
        return []
    return sorted(p.name for p in carpeta.iterdir()
                  if p.is_dir() and (p / "metadatos_preparacion.csv").exists()
                  and (p / "respuestas.csv").exists())


def ultima_corrida(carpeta: str | Path) -> tuple[str, Path]:
    """Id y carpeta de la corrida más reciente.

    Raises:
        FileNotFoundError: si no hay ninguna corrida completa en `carpeta`.
    """
    todas = corridas(carpeta)
    if not todas:
        raise FileNotFoundError(f"No hay corridas de la Etapa 1 en {carpeta}. Corra run_etapa1.py.")
    return todas[-1], Path(carpeta) / todas[-1]
