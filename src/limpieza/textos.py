"""
Limpieza del texto de una cláusula y n-gramas.

Cada palabra se guarda en dos formas: la normalizada (para comparar con las
listas y contar) y la original en minúsculas con tildes (para mostrar). Así
`ngrama_mostrar` sale del mismo texto de la cláusula, sin reconstruirlo.
"""
from __future__ import annotations

import re
from collections import Counter

from preparacion.normalizacion import normalizar

_PALABRA = re.compile(r"[^\W_]+")

Token = tuple[str, str]  # (normalizada, original en minúsculas)


def tokenizar(texto: str) -> list[Token]:
    """Palabras del texto, cada una con su forma normalizada y la original."""
    tokens: list[Token] = []
    for original in _PALABRA.findall(str(texto).lower()):
        for parte in normalizar(original).split():
            tokens.append((parte, original))
    return tokens


def quitar_frases(tokens: list[Token], frases: frozenset[tuple[str, ...]]) -> list[Token]:
    """Quita las stopwords de varias palabras, como palabras completas y empezando por la más larga."""
    if not frases:
        return tokens
    largos = sorted({len(f) for f in frases}, reverse=True)
    resultado, i = [], 0
    while i < len(tokens):
        for largo in largos:
            if tuple(t[0] for t in tokens[i:i + largo]) in frases:
                i += largo
                break
        else:
            resultado.append(tokens[i])
            i += 1
    return resultado


def limpiar(tokens: list[Token], palabras: frozenset[str], frases: frozenset[tuple[str, ...]]) -> list[Token]:
    """Quita primero las frases y luego las palabras sueltas (las negaciones ya no están en `palabras`)."""
    return [t for t in quitar_frases(tokens, frases) if t[0] not in palabras]


def ngramas(tokens: list[Token], n_max: int = 3) -> Counter:
    """Cuenta los n-gramas de 1 a n_max: clave (n, normalizado, mostrar)."""
    cuenta: Counter = Counter()
    for n in range(1, n_max + 1):
        for i in range(len(tokens) - n + 1):
            trozo = tokens[i:i + n]
            cuenta[(n, " ".join(t[0] for t in trozo), " ".join(t[1] for t in trozo))] += 1
    return cuenta
