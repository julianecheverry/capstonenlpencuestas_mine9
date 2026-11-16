"""
Listas de palabras de la Etapa 3: stopwords, adverbios, negaciones y acentos.

Reglas (especificación de la Etapa 3, tarea 2):
- Si falta una hoja o una lista está vacía, la corrida se detiene con un
  mensaje que la nombra; nunca sigue con una lista vacía.
- Cada entrada se normaliza con `preparacion.normalizacion.normalizar`, igual
  que el texto, para que «más» o «También» coincidan con «mas» y «tambien».
- Las stopwords de una palabra se separan de las de varias («sin embargo»).
- Las negaciones nunca son stopwords de una palabra, ni adverbios que se
  quiten: «no explica» no debe confundirse con «explica».
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from preparacion.normalizacion import normalizar

HOJAS_STOPWORDS = ("base", "globales", "particulares", "adverbios")

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
    negaciones: frozenset[str]
    huella_stopwords: str
    huella_acentos: str
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
    """Normaliza y separa en palabras sueltas y frases. Quita negaciones sueltas."""
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
                avisos.append(f"{origen}: la frase «{n}» contiene una negación y se quita completa.")
    return frozenset(palabras), frozenset(frases)


def cargar_listas(ruta_stopwords: str | Path, ruta_negaciones: str | Path,
                  ruta_acentos: str | Path) -> Listas:
    """Lee y valida las tres fuentes de listas.

    Raises:
        ValueError: si falta una hoja o una columna, o una lista está vacía.
    """
    ruta_stopwords, ruta_negaciones, ruta_acentos = map(Path, (ruta_stopwords, ruta_negaciones, ruta_acentos))
    avisos: list[str] = []

    neg = _hoja(ruta_negaciones, "negaciones", {"valor"})
    negaciones = frozenset(n for n in (normalizar(v) for v in neg["valor"].dropna()) if n)
    if not negaciones:
        raise ValueError(f"La hoja 'negaciones' de {ruta_negaciones.name} está vacía.")

    # accents.xlsx solo se valida y se registra su huella: `normalizar` ya quita
    # las tildes igual que la Etapa 1, así que no hace falta aplicarlo aparte.
    _hoja(ruta_acentos, "accents", {"accented", "not_accented"})

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
                  negaciones, huella(ruta_stopwords), huella(ruta_acentos), huella(ruta_negaciones),
                  avisos)
