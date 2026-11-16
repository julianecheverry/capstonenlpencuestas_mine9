"""
Listas de palabras de la Etapa 3: stopwords, adverbios y palabras que invierten la polaridad.

Las tildes no usan lista: `normalizar` las quita con la normalización Unicode
(NFKD), igual que la Etapa 1 (decisión del 16-nov; accents.xlsx ya no se usa).

Reglas (especificación de la Etapa 3, tarea 2):
- Si falta una hoja o una lista está vacía, la corrida se detiene con un
  mensaje que la nombra; nunca sigue con una lista vacía.
- Cada entrada se normaliza con `preparacion.normalizacion.normalizar`, igual
  que el texto, para que «más» o «También» coincidan con «mas» y «tambien».
- Las stopwords de una palabra se separan de las de varias («sin embargo»).
- Las palabras que invierten la polaridad nunca son stopwords de una palabra, ni
  adverbios que se quiten: «no explica» no es «explica» y «poco claro» no es «claro».
  Salen de negaciones.xlsx, el léxico compartido con el NLP (dueño: Etapa 2),
  filas con TIPO en TIPOS_PROTEGIDOS. Los intensificadores, adversativos y modales
  no se protegen: no invierten la polaridad.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from preparacion.normalizacion import normalizar

HOJAS_STOPWORDS = ("base", "globales", "particulares", "adverbios")

# Léxico compartido con la Etapa 2 (negaciones.xlsx, hoja `negaciones`).
TIPOS_LEXICO = {"NEGADOR", "ADVERSATIVO", "CUANTIFICADOR_BAJO", "INTENSIFICADOR", "MODAL_ATENUANTE"}
TIPOS_PROTEGIDOS = ("NEGADOR", "CUANTIFICADOR_BAJO")

# Nombre de la lista particular en stopwords.xlsx → código de encuesta del contrato.
# Calidad tiene una lista por población; como la Etapa 3 no recibe la población,
# se aplican todas juntas (decisión abierta, ver la especificación).
PARTICULARES_POR_ENCUESTA = {
    "evadoc_evaluacion": ["stopwords_evaluaciondocente"],
    "evadoc_autoevaluacion": ["stopwords_autoevaluaciondocente"],
    "calidad_servicios": [
        "stopwords_calidadadministrativos", "stopwords_calidaddirectivos",
        "stopwords_calidadestudiantes", "stopwords_calidadegresados",
        "stopwords_calidaddocentes",
    ],
    "autoevaluacion_acreditacion": [],
}


def huella(ruta: str | Path) -> str:
    """Primeros 10 caracteres del SHA-256 del archivo."""
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()[:10]


@dataclass
class Listas:
    """Listas ya normalizadas y listas para comparar con el texto normalizado."""
    palabras_comunes: frozenset[str]              # base + globales, una palabra
    frases_comunes: frozenset[tuple[str, ...]]    # base + globales, varias palabras
    palabras_particulares: dict[str, frozenset[str]]
    frases_particulares: dict[str, frozenset[tuple[str, ...]]]
    adverbios_no_valoran: frozenset[str]
    protegidas: frozenset[str]                   # negadores y cuantificadores bajos
    huella_stopwords: str
    huella_negaciones: str
    avisos: list[str] = field(default_factory=list)

    def stopwords(self, encuesta: str) -> tuple[frozenset[str], frozenset[tuple[str, ...]]]:
        """Palabras y frases que se quitan en las cláusulas de una encuesta."""
        return (self.palabras_comunes | self.palabras_particulares.get(encuesta, frozenset()),
                self.frases_comunes | self.frases_particulares.get(encuesta, frozenset()))


def _hoja(ruta: Path, hoja: str, columnas: set[str]) -> pd.DataFrame:
    try:
        df = pd.read_excel(ruta, sheet_name=hoja, dtype=str)
    except ValueError as e:
        raise ValueError(f"Falta la hoja '{hoja}' en {ruta.name}.") from e
    faltan = sorted(columnas - set(df.columns))
    if faltan:
        raise ValueError(f"La hoja '{hoja}' de {ruta.name} no tiene las columnas {faltan}.")
    if df.dropna(how="all").empty:
        raise ValueError(f"La hoja '{hoja}' de {ruta.name} está vacía.")
    return df


def _separar(valores, negaciones: frozenset[str], avisos: list[str], origen: str):
    """Normaliza y separa en palabras sueltas y frases. Quita las palabras protegidas sueltas."""
    palabras, frases = set(), set()
    for v in valores:
        n = normalizar(v)
        if not n:
            continue
        partes = tuple(n.split())
        if len(partes) == 1:
            if partes[0] in negaciones:
                continue
            palabras.add(partes[0])
        else:
            frases.add(partes)
            if any(p in negaciones for p in partes):
                avisos.append(f"{origen}: la frase «{n}» contiene una palabra protegida y se quita completa.")
    return frozenset(palabras), frozenset(frases)


def cargar_listas(ruta_stopwords: str | Path, ruta_negaciones: str | Path) -> Listas:
    """Lee y valida las listas de stopwords y el léxico de polaridad (negaciones.xlsx).

    Raises:
        ValueError: si falta una hoja o una columna, o una lista está vacía.
    """
    ruta_stopwords, ruta_negaciones = Path(ruta_stopwords), Path(ruta_negaciones)
    avisos: list[str] = []

    lex = _hoja(ruta_negaciones, "negaciones", {"TERMINO", "TIPO"}).dropna(subset=["TERMINO"])
    tipos = set(lex["TIPO"].dropna().str.strip().str.upper())
    desconocidos = sorted(tipos - TIPOS_LEXICO)
    if desconocidos:
        raise ValueError(f"La hoja 'negaciones' de {ruta_negaciones.name} tiene TIPO desconocidos: {desconocidos}.")
    protegidos = lex.loc[lex["TIPO"].str.strip().str.upper().isin(TIPOS_PROTEGIDOS), "TERMINO"]
    negaciones = frozenset(n for n in (normalizar(v) for v in protegidos) if n)
    if not negaciones:
        raise ValueError(f"La hoja 'negaciones' de {ruta_negaciones.name} no tiene términos de tipo {TIPOS_PROTEGIDOS}.")

    hojas = {h: _hoja(ruta_stopwords, h, {"encuesta", "palabra"}) for h in HOJAS_STOPWORDS}
    comunes = pd.concat([hojas["base"], hojas["globales"]])["palabra"].dropna()
    palabras_comunes, frases_comunes = _separar(comunes, negaciones, avisos, "base/globales")

    part = hojas["particulares"].dropna(subset=["palabra"])
    pal_part, fra_part = {}, {}
    conocidas = {l for ls in PARTICULARES_POR_ENCUESTA.values() for l in ls}
    sin_uso = sorted(set(part["encuesta"]) - conocidas)
    if sin_uso:
        avisos.append(f"particulares: listas sin encuesta asignada, no se usan: {sin_uso}.")
    for encuesta, nombres in PARTICULARES_POR_ENCUESTA.items():
        valores = part.loc[part["encuesta"].isin(nombres), "palabra"]
        pal_part[encuesta], fra_part[encuesta] = _separar(valores, negaciones, avisos, f"particulares/{encuesta}")

    adv = hojas["adverbios"]
    if "valora" not in adv.columns:
        raise ValueError(f"La hoja 'adverbios' de {ruta_stopwords.name} no tiene la columna 'valora' (si/no).")
    no_valoran = adv.loc[adv["valora"].map(normalizar) != "si", "palabra"].dropna()
    adverbios_no_valoran = frozenset(n for n in map(normalizar, no_valoran) if n and n not in negaciones)

    return Listas(palabras_comunes, frases_comunes, pal_part, fra_part, adverbios_no_valoran,
                  negaciones, huella(ruta_stopwords), huella(ruta_negaciones),
                  avisos)
