"""
Normalización de texto y detección de no-respuesta.

La no-respuesta se decide SIEMPRE sobre el texto original normalizado, nunca
después de quitar stopwords: así un elogio corto como "es la mejor" no se
pierde, y variantes como "Ninguna." o "Ningúno" se reconocen igual que "NR".
"""
from __future__ import annotations

import re
import unicodedata

# Valores que, una vez normalizados, se consideran no-respuesta.
# Incluye "0": en la autoevaluación de 2026-1 el cliente registró así la
# no-respuesta del docente. Los comentarios cortos genéricos ("bien",
# "todo bien", "es la mejor") NO están aquí a propósito: se conservan como
# comentarios y el NLP decide su aspecto (decisión del 14-nov-2026).
NO_RESPUESTA = frozenset({
    "",
    "nr",
    "na",
    "n a",          # "N/A" normalizado
    "ninguna",
    "ninguno",
    "nada",
    "no aplica",
    "no",
    "no hay",
    "no tengo",
    "no tengo ninguna",
    "no se",
    "0",
})


def normalizar(texto: object) -> str:
    """Minúsculas, sin tildes, sin puntuación y con espacios simples.

    >>> normalizar("  Ningúno. ")
    'ninguno'
    >>> normalizar("N/A")
    'n a'
    """
    if texto is None:
        return ""
    if isinstance(texto, float) and texto != texto:  # NaN
        return ""
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-z0-9ñ ]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def es_no_respuesta(texto: object) -> bool:
    """True si el texto, normalizado, está vacío o está en NO_RESPUESTA."""
    return normalizar(texto) in NO_RESPUESTA


def nombre_columna(nombre: str) -> str:
    """Convierte un nombre de columna del cliente a snake_case sin tildes.

    >>> nombre_columna("Código Cruce Autoevaluación")
    'codigo_cruce_autoevaluacion'
    """
    return normalizar(nombre).replace(" ", "_")
