"""
Normalización de texto y detección de no-respuesta.

La no-respuesta se decide SIEMPRE sobre el texto original normalizado, nunca
después de quitar stopwords: así un elogio corto como "es la mejor" no se
pierde, y variantes como "Ninguna." o "Ningúno" se reconocen igual que "NR".
"""
from __future__ import annotations

import re
import unicodedata

# Lista POR DEFECTO de valores de no-respuesta (ya normalizados).
# La lista que usa la Etapa 1 se configura en la hoja `no_respuesta` de
# preguntas_abiertas.xlsx; esta solo sirve de valor inicial y para usar el
# módulo sin archivo de configuración (p. ej. en pruebas).
# Incluye "0": en la autoevaluación de 2026-1 el cliente registró así la
# no-respuesta del docente. Los comentarios cortos genéricos ("bien",
# "todo bien", "es la mejor") NO están a propósito: se conservan como
# comentarios y el NLP decide su aspecto (decisión del 14-nov-2026).
NO_RESPUESTA_POR_DEFECTO = frozenset({
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


def es_no_respuesta(texto: object, lista: frozenset[str] = NO_RESPUESTA_POR_DEFECTO) -> bool:
    """True si el texto, normalizado, está vacío o está en la lista de no-respuesta."""
    return normalizar(texto) in lista


def nombre_columna(nombre: str) -> str:
    """Convierte un nombre de columna del cliente a snake_case sin tildes.

    >>> nombre_columna("Código Cruce Autoevaluación")
    'codigo_cruce_autoevaluacion'
    """
    return normalizar(nombre).replace(" ", "_")
