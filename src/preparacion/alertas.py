"""
Detección de alertas en la Etapa 1 (decisión del equipo, 19-nov-2026).

Una alerta es una respuesta con comentario que contiene un término de
`data/utilities/alertas.xlsx` (acoso, discriminación, salud mental…). La Etapa 1:

- marca la respuesta con `tiene_alerta = True` en respuestas.csv. Solo se evalúa
  si `es_no_respuesta = False`; una no-respuesta nunca tiene alerta;
- escribe una fila por término encontrado en alertas.csv, con `negada = True` si
  una negación cercana lo anula («nunca ha sido irrespetuoso»).

`tiene_alerta` es True aunque todas las coincidencias estén negadas: las
respuestas con alerta, negada o no, NO pasan al NLP. `negada` sirve para decidir
qué se muestra en el tablero.

Reglas de coincidencia (sobre el texto normalizado, como la no-respuesta):
- Palabras completas: «ira» no coincide con «mirada»; «acoso» no coincide con
  «acosadores» salvo que la lista tenga esa forma.
- Si dos términos se solapan, gana el más largo: «abuso de poder» es una sola
  alerta, no dos («abuso» y «abuso de poder»).

Negación, con los términos de tipo NEGADOR de `data/utilities/negaciones.xlsx`:
- alcance POSTERIOR («no», «sin», «falta»): niega si está hasta VENTANA_NEGACION
  palabras antes del término;
- alcance CLAUSULA («nunca», «nadie»): niega si está en el mismo segmento, antes
  o después del término.
Un segmento termina en punto, punto y coma, dos puntos, signos de cierre de
exclamación o interrogación, salto de línea o un ADVERSATIVO («pero», «sin
embargo»). Las palabras de un adversativo de varias palabras no cuentan como
negadores («sin» de «sin embargo», «no» de «no obstante»), y las palabras del
propio término tampoco («falta» de «falta de respeto»).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from preparacion.normalizacion import nombre_columna, normalizar

VENTANA_NEGACION = 3
_PALABRA = re.compile(r"[^\W_]+")
_CORTE = re.compile(r"[.;:!?\n\r]")
_COLUMNAS_TERMINO = ("termino",)


@dataclass(frozen=True)
class TerminoAlerta:
    termino: str               # tal como está en alertas.xlsx
    palabras: tuple[str, ...]  # normalizadas


@dataclass(frozen=True)
class Negaciones:
    posterior: frozenset[str]                  # negadores de una palabra con alcance POSTERIOR
    clausula: frozenset[str]                   # negadores de una palabra con alcance CLAUSULA
    adversativos: tuple[tuple[str, ...], ...]  # cortes de segmento (normalizados)


def _columna(df: pd.DataFrame, candidatas: tuple[str, ...]) -> str | None:
    reales = {nombre_columna(str(c)): c for c in df.columns}
    for c in candidatas:
        if c in reales:
            return reales[c]
    return None


def cargar_alertas(ruta: str | Path) -> list[TerminoAlerta]:
    """Términos de alertas.xlsx (primera hoja), sin repetidos tras normalizar.

    Solo se usa la columna TERMINO. No hay categorías (decisión del 19-nov-2026): el
    tablero muestra el comentario completo en la sección de alertas del docente.
    """
    df = pd.read_excel(ruta, dtype=str, keep_default_na=False)
    col_termino = _columna(df, _COLUMNAS_TERMINO)
    if col_termino is None:
        raise ValueError(f"{ruta} no tiene la columna TERMINO.")
    terminos, vistos = [], set()
    for _, fila in df.iterrows():
        palabras = tuple(normalizar(fila[col_termino]).split())
        if not palabras or palabras in vistos:
            continue
        vistos.add(palabras)
        terminos.append(TerminoAlerta(str(fila[col_termino]).strip(), palabras))
    if not terminos:
        raise ValueError(f"{ruta} no tiene términos de alerta.")
    return terminos


def cargar_negaciones(ruta: str | Path) -> Negaciones:
    """Negadores y adversativos de negaciones.xlsx (hoja `negaciones`)."""
    df = pd.read_excel(ruta, sheet_name="negaciones", dtype=str, keep_default_na=False)
    faltan = {"termino", "tipo", "alcance"} - {nombre_columna(str(c)) for c in df.columns}
    if faltan:
        raise ValueError(f"Faltan las columnas {faltan} en la hoja 'negaciones' de {ruta}.")
    df.columns = [nombre_columna(str(c)) for c in df.columns]
    df["_norm"] = df["termino"].map(normalizar)
    tipo, alcance = df["tipo"].str.strip().str.upper(), df["alcance"].str.strip().str.upper()
    negador = (tipo == "NEGADOR") & (df["_norm"] != "")
    simples = negador & ~df["_norm"].str.contains(" ")
    return Negaciones(
        posterior=frozenset(df.loc[simples & (alcance == "POSTERIOR"), "_norm"]),
        clausula=frozenset(df.loc[simples & (alcance == "CLAUSULA"), "_norm"]),
        adversativos=tuple(sorted({tuple(t.split()) for t in df.loc[(tipo == "ADVERSATIVO")
                                                                 & (df["_norm"] != ""), "_norm"]},
                                  key=len, reverse=True)),
    )


class DetectorAlertas:
    """Busca los términos de alerta en un texto y decide si cada uno está negado."""

    def __init__(self, terminos: list[TerminoAlerta], negaciones: Negaciones,
                 ventana: int = VENTANA_NEGACION):
        self.negaciones = negaciones
        self.ventana = ventana
        self._por_primera: dict[str, list[TerminoAlerta]] = {}
        for t in sorted(terminos, key=lambda t: len(t.palabras), reverse=True):
            self._por_primera.setdefault(t.palabras[0], []).append(t)

    @staticmethod
    def _tokens(texto: str) -> tuple[list[str], list[tuple[int, int]], list[int]]:
        """Palabras normalizadas, su posición en el texto original y su segmento por puntuación."""
        palabras, spans, segmentos = [], [], []
        segmento, fin_anterior = 0, 0
        for m in _PALABRA.finditer(texto):
            if _CORTE.search(texto, fin_anterior, m.start()):
                segmento += 1
            fin_anterior = m.end()
            for parte in normalizar(m.group()).split():
                palabras.append(parte)
                spans.append(m.span())
                segmentos.append(segmento)
        return palabras, spans, segmentos

    def _cortar_por_adversativos(self, palabras: list[str],
                                 segmentos: list[int]) -> tuple[list[tuple[int, int]], set[int]]:
        """Segmento final = (segmento por puntuación, n.º de adversativos vistos antes).

        Devuelve también las posiciones de las palabras que forman los adversativos.
        """
        nuevos, dentro_adversativo, vistos, i = [], set(), 0, 0
        while i < len(palabras):
            largo = next((len(a) for a in self.negaciones.adversativos
                          if tuple(palabras[i:i + len(a)]) == a), 0)
            if largo:
                vistos += 1
                dentro_adversativo.update(range(i, i + largo))
            for j in range(i, min(i + max(largo, 1), len(palabras))):
                nuevos.append((segmentos[j], vistos))
            i += max(largo, 1)
        return nuevos, dentro_adversativo

    def detectar(self, texto: object) -> list[dict]:
        """Coincidencias en orden de aparición: termino, texto_coincidente, negada."""
        if texto is None or (isinstance(texto, float) and texto != texto):
            return []
        texto = str(texto)
        palabras, spans, segmentos = self._tokens(texto)
        if not palabras:
            return []
        segmentos, adversativas = self._cortar_por_adversativos(palabras, segmentos)
        encontradas, i = [], 0
        while i < len(palabras):
            termino = next((t for t in self._por_primera.get(palabras[i], [])
                            if tuple(palabras[i:i + len(t.palabras)]) == t.palabras), None)
            if termino is None:
                i += 1
                continue
            fin = i + len(termino.palabras)
            encontradas.append((termino, i, fin))
            i = fin
        propias = {j for _, a, b in encontradas for j in range(a, b)}
        resultado = []
        for termino, a, b in encontradas:
            seg = segmentos[a]
            negada = False
            for j, palabra in enumerate(palabras):
                if j in propias or j in adversativas or segmentos[j] != seg:
                    continue
                if palabra in self.negaciones.clausula:
                    negada = True
                elif palabra in self.negaciones.posterior and a - self.ventana <= j < a:
                    negada = True
                if negada:
                    break
            resultado.append({
                "termino": termino.termino,
                "texto_coincidente": texto[spans[a][0]:spans[b - 1][1]],
                "negada": negada,
            })
        return resultado


def detectar_alertas(respuestas: pd.DataFrame, detector: DetectorAlertas) -> tuple[pd.Series, pd.DataFrame]:
    """Devuelve (tiene_alerta por respuesta, tabla alertas).

    Solo revisa las respuestas con `es_no_respuesta = False`.
    alerta_id = respuesta_id|n, con n desde 1 en el orden de aparición en el texto.
    """
    tiene = pd.Series(False, index=respuestas.index, name="tiene_alerta")
    filas = []
    con_comentario = respuestas.loc[~respuestas["es_no_respuesta"].astype(bool)]
    for idx, rid, texto in zip(con_comentario.index, con_comentario["respuesta_id"],
                               con_comentario["texto_original"]):
        coincidencias = detector.detectar(texto)
        if not coincidencias:
            continue
        tiene.at[idx] = True
        for n, c in enumerate(coincidencias, start=1):
            filas.append({"alerta_id": f"{rid}|{n}", "respuesta_id": rid, **c})
    alertas = pd.DataFrame(filas, columns=["alerta_id", "respuesta_id", "termino",
                                           "texto_coincidente", "negada"])
    alertas["negada"] = alertas["negada"].astype(bool)
    return tiene, alertas
