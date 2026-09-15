"""
Mod_Bertopic.py — Tópicos + Sentimiento + ABSA para encuestas abiertas
=======================================================================

Proyecto: Capstone NLP — Encuestas Universitarias (MINE009)
Versión: MVP 2 — corrige los tres defectos detectados en la primera corrida
real sobre el corpus (136 tópicos, 40% de outliers, 65% de corpus "perdido").

QUÉ CORRIGE ESTA VERSIÓN
-------------------------
1. HIGIENE OPACA. El log anterior decía "65,1% descartados", lo que parecía
   una pérdida masiva de datos. En realidad la mayoría son CELDAS VACÍAS: la
   pregunta abierta es opcional y mucha gente no la contesta. Ahora el
   descarte se desglosa en VACÍA / RUIDO / CORTA y los porcentajes se
   calculan sobre respuestas CONTESTADAS, no sobre filas del archivo.

2. HIPERPARÁMETROS EN ESCALA ABSOLUTA. `min_topic_size=25` sobre 20.000
   documentos únicos es el 0,125% del corpus y admite hasta 800 tópicos: por
   eso salieron 136. Ahora el tamaño mínimo se expresa como FRACCIÓN del
   corpus, que sí es interpretable y escala sola entre encuestas.

3. min_samples DEMASIADO ALTO. Con `min_cluster_size=25` y `min_samples=10`
   el ratio es 0,40. HDBSCAN se vuelve tan conservador que manda el 40% de
   los documentos a outlier. La regla práctica es un ratio cercano a 0,05.
   Ahora se deriva solo, y el módulo avisa si se sobrepasa 0,20.

Además se añade `GridSearchTopicos`: una búsqueda por etapas que encuentra
los hiperparámetros empíricamente contra criterios de negocio explícitos.

PRINCIPIO DE SEPARACIÓN DE RAMAS
---------------------------------
  RAMA TÓPICOS                      RAMA SENTIMIENTO / ABSA
  --------------------------        --------------------------
  texto _clean (SIN tildes)         texto original (CON tildes)
  stopwords.xlsx: SÍ, en el         stopwords.xlsx: NUNCA
    CountVectorizer, DESPUÉS          (quitar "no" invierte la
    de la clusterización               polaridad de la frase)
  accents.xlsx: SÍ, sólo para       accents.xlsx: NUNCA
    normalizar la LISTA               (los modelos de español
  negaciones.xlsx: NO                  están entrenados con tildes)
                                    negaciones.xlsx: SÍ

Lo único compartido es el encoder, que se calcula una vez y se reutiliza.

POR QUÉ LAS STOPWORDS VAN DESPUÉS DEL CLUSTERING
-------------------------------------------------
  Etapa 1 (agrupar):   embeddings -> UMAP -> HDBSCAN
                       Necesita texto NATURAL: el transformer usa el orden y
                       las preposiciones. Sin stopwords queda ciego.
  Etapa 2 (etiquetar): c-TF-IDF sobre documentos agregados POR TÓPICO.
                       Esto sí es bolsa de palabras. Aquí las stopwords son
                       ruido y aquí se eliminan.

RECURSOS EXTERNOS (ningún léxico vive en el código)
----------------------------------------------------
  data/Corpus Ejemplo PLN.xlsx    corpus de las encuestas
  data/utilities/stopwords.xlsx   rama TÓPICOS
  data/utilities/accents.xlsx     rama TÓPICOS
  data/utilities/negaciones.xlsx  rama SENTIMIENTO / ABSA

USO
---
    from Mod_Bertopic import ejecutar_pipeline
    resultados = ejecutar_pipeline()
    resultados["Evaluación Docente"].absa
"""

# pylint: disable=too-many-lines  # módulo cohesivo, una sola responsabilidad

from __future__ import annotations

# ── Librería estándar ──────────────────────────────────────────────────────
import hashlib  # clave de caché de embeddings
import json  # manifiesto de la corrida
import logging  # trazas de progreso
import os  # listdir: detección robusta frente a OneDrive
import re  # segmentación y limpieza mínima
import unicodedata  # respaldo si falta accents.xlsx
from collections.abc import Iterable, Sequence  # anotaciones de tipo
from dataclasses import asdict, dataclass, field  # configuración tipada
from datetime import datetime  # marca temporal
from pathlib import Path  # rutas portables
from typing import Any  # anotaciones de tipo

# ── Terceros ───────────────────────────────────────────────────────────────
import numpy as np  # álgebra de embeddings
import pandas as pd  # manejo tabular completo

# ── Logger ─────────────────────────────────────────────────────────────────
logger = logging.getLogger("Mod_Bertopic")
if not logger.handlers:  # evita mensajes duplicados al reimportar
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
        )
    )
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════════════════════
# 1. CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════

OUTLIER: int = -1  # HDBSCAN no pudo asignarlo a ningún cluster
EXCLUIDO: int = -2  # nunca entró al modelo (vacío, ruido o demasiado corto)
POS, NEU, NEG = "POS", "NEU", "NEG"  # etiquetas del modelo de sentimiento


# ═══════════════════════════════════════════════════════════════════════════
# 2. LOCALIZACIÓN DE ARCHIVOS
# ═══════════════════════════════════════════════════════════════════════════


def encontrar_raiz(marcador: str = "pyproject.toml") -> Path:
    """Sube por el árbol hasta hallar la raíz del proyecto.

    El kernel de VS Code arranca desde directorios variables: no se puede
    asumir que el directorio actual sea la raíz.
    """
    for candidato in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (candidato / marcador).exists():
            return candidato
    raise FileNotFoundError(f"No se encontró '{marcador}' sobre {Path.cwd()}")


def buscar_archivo(
    directorio: Path, *fragmentos: str, extensiones: tuple[str, ...] = (".xlsx", ".xls")
) -> Path:
    """Localiza un archivo por fragmentos del nombre, no por nombre exacto.

    Se usa `os.listdir` a propósito: con OneDrive en modo "solo en la nube",
    comprobar una ruta fija produce FileNotFoundError silenciosos.
    """
    if not directorio.exists():
        raise FileNotFoundError(f"No existe el directorio: {directorio}")
    candidatos = [
        directorio / nombre
        for nombre in sorted(os.listdir(directorio))
        if nombre.lower().endswith(extensiones)
        and all(f.lower() in nombre.lower() for f in fragmentos)
    ]
    if not candidatos:
        raise FileNotFoundError(
            f"Sin coincidencias para {fragmentos} en {directorio}.\n"
            f"Archivos disponibles: {sorted(os.listdir(directorio))}"
        )
    if len(candidatos) > 1:
        logger.warning(
            "%d coincidencias para %s; se usa '%s'",
            len(candidatos),
            fragmentos,
            candidatos[0].name,
        )
    return candidatos[0]


# ═══════════════════════════════════════════════════════════════════════════
# 3. RECURSOS LÉXICOS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class RecursosLexicos:
    """Los tres léxicos, cargados de disco y separados por rama."""

    stopwords: frozenset[str] = field(default_factory=frozenset)
    mapa_acentos: dict[str, str] = field(default_factory=dict)
    negaciones: pd.DataFrame = field(default_factory=pd.DataFrame)

    @classmethod
    def cargar(cls, dir_utilities: Path) -> RecursosLexicos:
        """Lee stopwords.xlsx, accents.xlsx y negaciones.xlsx."""
        logger.info("Cargando recursos léxicos desde %s", dir_utilities)
        # accents.xlsx primero: las stopwords se normalizan con ese mapa.
        mapa = cls._leer_acentos(dir_utilities)
        crudas = cls._leer_stopwords(dir_utilities)
        # Sin esta normalización, "más" o "también" nunca coinciden con el
        # texto _clean (que va sin tildes) y quedan decorativas en la lista.
        normalizadas = {cls._sin_acentos(s, mapa) for s in crudas}
        negaciones = cls._leer_negaciones(dir_utilities)
        logger.info(
            "  stopwords: %d | acentos: %d pares | negaciones: %d términos",
            len(normalizadas),
            len(mapa),
            len(negaciones),
        )
        return cls(frozenset(normalizadas), mapa, negaciones)

    @staticmethod
    def _leer_stopwords(directorio: Path) -> set[str]:
        """Extrae stopwords de todas las hojas y columnas del archivo."""
        hojas = pd.read_excel(buscar_archivo(directorio, "stopword"), sheet_name=None)
        terminos: set[str] = set()
        for hoja in hojas.values():
            for columna in hoja.columns:
                terminos |= {
                    str(v).strip().lower()
                    for v in hoja[columna].dropna()
                    if str(v).strip()
                }
        return terminos

    @staticmethod
    def _leer_acentos(directorio: Path) -> dict[str, str]:
        """Mapa {carácter acentuado -> carácter plano} de dos columnas."""
        try:
            ruta = buscar_archivo(directorio, "accent")
        except FileNotFoundError:
            logger.warning("accents.xlsx no encontrado; se usará unicodedata")
            return {}
        hoja = pd.read_excel(ruta)
        if hoja.shape[1] < 2:
            logger.warning("accents.xlsx no tiene dos columnas; se ignora")
            return {}
        return {
            str(a): str(b)
            for a, b in zip(hoja.iloc[:, 0], hoja.iloc[:, 1])
            if pd.notna(a) and pd.notna(b)
        }

    @staticmethod
    def _leer_negaciones(directorio: Path) -> pd.DataFrame:
        """Lee negaciones.xlsx, hoja 'negaciones', y valida su esquema."""
        tabla = pd.read_excel(
            buscar_archivo(directorio, "negacion"), sheet_name="negaciones"
        )
        tabla.columns = [str(c).strip().upper() for c in tabla.columns]
        faltantes = {"TERMINO", "TERMINO_SIN_TILDE", "TIPO", "ALCANCE"} - set(
            tabla.columns
        )
        if faltantes:
            raise KeyError(f"negaciones.xlsx: faltan columnas {faltantes}")
        return tabla

    @staticmethod
    def _sin_acentos(texto: str, mapa: dict[str, str]) -> str:
        """Quita tildes con el mapa del proyecto, o unicodedata si no hay."""
        if mapa:  # ruta preferente: misma fuente que limpieza.py
            for origen, destino in mapa.items():
                texto = texto.replace(origen, destino)
            return texto
        return "".join(
            c
            for c in unicodedata.normalize("NFD", texto)
            if unicodedata.category(c) != "Mn"
        )

    def sin_acentos(self, texto: str) -> str:
        """Normalización pública. SOLO para la rama TÓPICOS."""
        return self._sin_acentos(texto, self.mapa_acentos)

    def stopwords_para_vectorizador(self, token_pattern: str) -> list[str]:
        """Stopwords que el tokenizador puede realmente producir.

        scikit-learn no avisa cuando una stopword es imposible de generar con
        el `token_pattern` en uso (p. ej. "de" con un patrón de 3+ letras).
        Esas entradas quedan sin efecto y dan falsa sensación de cobertura.
        """
        compilado = re.compile(token_pattern)
        return sorted(s for s in self.stopwords if compilado.fullmatch(s))

    def terminos_por_tipo(self, *tipos: str, con_tilde: bool = True) -> list[str]:
        """Términos de negaciones.xlsx filtrados por TIPO.

        `con_tilde=True` para la rama SENTIMIENTO (texto acentuado);
        `False` para la rama TÓPICOS (texto _clean).
        """
        columna = "TERMINO" if con_tilde else "TERMINO_SIN_TILDE"
        sub = self.negaciones[self.negaciones["TIPO"].isin(tipos)]
        return sorted({str(t).strip().lower() for t in sub[columna]})


# ═══════════════════════════════════════════════════════════════════════════
# 4. TOKENIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════


def construir_token_pattern(preservar: Iterable[str], longitud_minima: int = 3) -> str:
    """Patrón de tokenización que NO borra las negaciones cortas.

    Un patrón ingenuo `\\b[a-zñ]{3,}\\b` elimina "no" y "ni" por longitud.
    Como son las negaciones más frecuentes del español, el vectorizador
    produce el mismo vector para "no explica bien" y "explica bien".
    """
    cortos = sorted(
        {t for t in preservar if 0 < len(t) < longitud_minima}, key=len, reverse=True
    )  # los largos primero: la alternancia regex es perezosa
    alternativas = [re.escape(t) for t in cortos]
    alternativas.append(f"[a-zñ]{{{longitud_minima},}}")
    return r"(?u)\b(?:" + "|".join(alternativas) + r")\b"


# ═══════════════════════════════════════════════════════════════════════════
# 5. CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

# Encoder multilingüe. El `all-MiniLM-L6-v2` por defecto de BERTopic es
# MONOLINGÜE EN INGLÉS: sobre español agrupa por accidente.
ENCODER_POR_DEFECTO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Clasificador de polaridad en español (familia RoBERTa, entrenado en TASS).
MODELO_SENTIMIENTO_POR_DEFECTO = "pysentimiento/robertuito-sentiment-analysis"


@dataclass
class Config:
    """Hiperparámetros de una corrida, serializables al manifiesto."""

    # --- Identidad ---
    nombre: str = "mvp"
    random_state: int = 42  # UMAP es estocástico: sin esto no hay réplica

    # --- Encoders ---
    encoder: str = ENCODER_POR_DEFECTO
    modelo_sentimiento: str = MODELO_SENTIMIENTO_POR_DEFECTO
    batch_size: int = 64
    usar_cache: bool = True  # embeddings persistidos en disco

    # --- Higiene ---
    min_tokens: int = 3  # respuestas más cortas no sostienen un tópico
    deduplicar: bool = True

    # --- UMAP ---
    # 30 y no 15: con 15, UMAP preserva estructura muy local y HDBSCAN
    # encuentra decenas de micro-grupos. Subirlo agrupa por temas amplios,
    # que es lo que necesita un informe.
    umap_n_neighbors: int = 30
    umap_n_components: int = 5  # dimensiones destino para HDBSCAN
    umap_min_dist: float = 0.0  # 0 = clusters compactos

    # --- HDBSCAN ---
    # None => se deriva de `fraccion_topico_minimo`. Un valor absoluto no
    # significa nada sin saber cuántos documentos hay: 25 sobre 20.000
    # únicos es el 0,125% del corpus y admite 800 tópicos.
    min_topic_size: int | None = None
    fraccion_topico_minimo: float = 0.025  # 2,5% del corpus único
    # None => se deriva con `ratio_min_samples`. Un min_samples alto
    # respecto a min_cluster_size vuelve a HDBSCAN muy conservador y dispara
    # los outliers: con ratio 0,40 se obtuvo 40% de outliers.
    min_samples: int | None = None
    ratio_min_samples: float = 0.05
    # "eom" (excess of mass) favorece clusters grandes y estables; "leaf"
    # corta el árbol en las hojas y produce tópicos más homogéneos y
    # parejos. Cuando un par de tópicos absorbe la mayoría del corpus,
    # este es el primer parámetro que hay que mover.
    cluster_selection_method: str = "eom"

    # --- Vectorizador (etapa de ETIQUETADO, después del clustering) ---
    ngram_range: tuple[int, int] = (1, 2)
    top_n_words: int = 10
    mmr_diversity: float = 0.35  # 0 = redundante, 1 = disperso

    # --- Post-proceso ---
    reducir_outliers: bool = True
    # Cascada en orden. "embeddings" es la más eficaz en textos cortos porque
    # compara vectores densos; "c-tf-idf" sola apenas recupera nada cuando la
    # respuesta tiene 6 palabras.
    cascada_outliers: tuple[str, ...] = ("embeddings", "distributions", "c-tf-idf")
    # Similitud mínima exigida para reasignar. 0.0 reasigna SIEMPRE al tópico
    # más cercano, lo que deja 0% de outliers — pero eso no es un logro: es
    # asignación forzada, y mete respuestas genuinamente ajenas dentro de un
    # tópico. Un valor entre 0.15 y 0.30 conserva como outlier lo que de
    # verdad no encaja. Cero outliers debe leerse con sospecha, no con alivio.
    umbral_outliers: float = 0.20
    # Tope de crecimiento por tópico durante la cascada, como fracción de su
    # tamaño original. Sin tope, los tópicos grandes actúan como imanes: cada
    # outlier va al centroide más cercano y la masa se concentra todavía más.
    # Medido sobre el corpus real, la cascada sin tope subió la
    # concentración en los dos tópicos mayores de 15,4% a 27,2%.
    tope_crecimiento_cascada: float = 0.35
    # Fusión de tópicos redundantes tras el ajuste. "auto" usa la jerarquía;
    # un entero fuerza ese número exacto. None = no fusionar.
    nr_topics: int | str | None = None

    # --- ABSA ---
    # 2 y no 3: fragmentos como "llega tarde" o "poco claro" llevan polaridad
    # real; perderlos sesga toda la matriz ABSA hacia lo positivo.
    min_tokens_clausula: int = 2
    max_clausulas_doc: int = 6

    # --- Salidas ---
    output_dir: Path = Path("outputs/mvp")

    def to_dict(self) -> dict[str, Any]:
        """Serializa la configuración a tipos compatibles con JSON."""
        datos = asdict(self)
        datos["output_dir"] = str(self.output_dir)
        datos["ngram_range"] = list(self.ngram_range)
        datos["cascada_outliers"] = list(self.cascada_outliers)
        return datos


# ═══════════════════════════════════════════════════════════════════════════
# 6. CARGA DEL CORPUS
# ═══════════════════════════════════════════════════════════════════════════

ENCUESTAS: dict[str, str] = {
    # Nombres verificados contra el corpus real. La columna "Observaciones"
    # que se asumía en versiones previas NO EXISTE en ninguna hoja.
    "Evaluación Docente": (
        "Observaciones - 19. Aspectos para resaltar del docente "
        "y/o del espacio académico"
    ),
    "Autoevaluación Docente": (
        "Aspectos para resaltar de su labor como docente y/o del espacio " "académico"
    ),
    # Las tres encuestas de Calidad tienen entre 36 y 47 respuestas de texto.
    # Se dejan declaradas para trazabilidad, pero `cargar_encuestas` las
    # marcará como NO MODELABLES: con ese volumen no hay topic modeling
    # posible y forzarlo produciría tópicos sin sustento estadístico.
    "Calidad Docentes": (
        "Si desea complementar sus respuestas o hacer sugerencias para "
        "nuestro mejoramiento continuo, por favor inclúyalas a continuación "
        "(si no tiene comentarios, dejar este espacio en blanco):"
    ),
    "Calidad Administrativos": (
        "Si desea complementar sus respuestas o hacer sugerencias para "
        "nuestro mejoramiento continuo, por favor inclúyalas a continuación "
        "(si no tiene comentarios, dejar este espacio en blanco):"
    ),
    "Calidad Estudiantes": (
        "Si desea complementar sus respuestas o hacer sugerencias para "
        "nuestro mejoramiento continuo, por favor inclúyalas a continuación "
        "(si no tiene comentarios, dejar este espacio en blanco):"
    ),
}

#: Columnas de texto abierto adicionales, por hoja. El corpus tiene más de
#: una por encuesta y cada una se modela por separado: "aspectos a resaltar"
#: y "oportunidades de mejora" tienen distribuciones opuestas y mezclarlas
#: destruiría la señal de ambas.
COLUMNAS_ADICIONALES: dict[str, list[str]] = {
    "Evaluación Docente": [
        "Observaciones - 20. Oportunidades de mejora del docente "
        "y/o del espacio académico",
        "Observaciones - 24. Aspectos para resaltar de su labor como docente "
        "y/o del espacio académico",
        "Observaciones - 25. Oportunidades de mejora de su labor como docente "
        "y/o del espacio académico",
    ],
    "Autoevaluación Docente": [
        "Oportunidades de mejora de su labor como docente y/o del espacio " "académico",
    ],
}

#: Mínimo de respuestas modelables para que una hoja admita topic modeling.
#: Por debajo de esto el resultado no tiene sustento y debe analizarse
#: cualitativamente.
MINIMO_MODELABLES: int = 300

#: Columna numérica usada como validación externa cuando no hay etiquetado
#: manual. No es una etiqueta de tópico, pero sí una medida independiente
#: del texto contra la cual contrastar la polaridad detectada.
COLUMNA_VALIDACION: str = "Promedio Evaluación - General"


# Respuestas que no son opinión. No deben entrar al modelo ni contarse como
# outliers: son una categoría distinta de descarte.
RUIDO = frozenset(
    {
        "nr",
        "na",
        "ns",
        "nc",
        "n/a",
        "no aplica",
        "no responde",
        "ninguno",
        "ninguna",
        "sin comentarios",
        "sin comentario",
        "x",
        "xx",
        "xxx",
        "-",
        "--",
        ".",
        "..",
        "...",
        "0",
        "nada",
        "no",
    }
)


@dataclass
class Encuesta:
    """Una columna de texto abierto de una hoja, con sus dos versiones."""

    nombre: str
    columna: str
    marco: pd.DataFrame
    texto_original: list[str]  # CON tildes -> rama SENTIMIENTO / ABSA
    texto_clean: list[str]  # SIN tildes -> rama TÓPICOS
    modelable: bool = True  # False si no hay volumen suficiente
    n_modelables: int = 0  # respuestas que pasarían la higiene


def _normalizar_minimo(texto: str, recursos: RecursosLexicos) -> str:
    """Produce la versión `_clean`: minúsculas, sin tildes, sin símbolos.

    CONSERVA stopwords, negaciones y orden de palabras. Esa es toda la
    diferencia con `_no_stopwords_synonyms`, y es la razón por la que el
    encoder puede trabajar: necesita la frase, no un telegrama.
    """
    if not isinstance(texto, str):  # celdas vacías llegan como NaN
        return ""
    plano = recursos.sin_acentos(texto.lower())
    plano = re.sub(r"[^\w\sñ.,;:¿?¡!]", " ", plano)  # conserva puntuación
    return re.sub(r"\s+", " ", plano).strip()


def cargar_encuestas(
    ruta_corpus: Path,
    recursos: RecursosLexicos,
    encuestas: dict[str, str] | None = None,
    incluir_adicionales: bool = True,
    minimo_modelables: int = MINIMO_MODELABLES,
) -> dict[str, Encuesta]:
    """Carga cada columna de texto abierto como una unidad de modelado propia.

    Dos decisiones de diseño verificadas contra el corpus real:

    1. **Una columna, un modelo.** "Aspectos para resaltar" y "Oportunidades
       de mejora" tienen distribuciones opuestas — en el corpus, quien
       escribe en la segunda califica 0,44 puntos más bajo. Mezclarlas
       destruiría la señal de ambas.
    2. **Volumen mínimo.** Las tres encuestas de Calidad tienen entre 36 y 47
       respuestas de texto. Se cargan y se marcan `modelable=False` en vez de
       forzar un modelo sin sustento estadístico.

    Returns:
        Mapa "Hoja :: columna abreviada" -> :class:`Encuesta`.
    """
    mapa = encuestas or ENCUESTAS
    libro = pd.read_excel(ruta_corpus, sheet_name=None)
    logger.info("Hojas en el corpus: %s", list(libro))

    cargadas: dict[str, Encuesta] = {}
    for hoja, columna_principal in mapa.items():
        if hoja not in libro:
            logger.warning("Hoja '%s' no está en el corpus; se omite", hoja)
            continue
        marco = libro[hoja]
        columnas = [columna_principal]
        if incluir_adicionales:
            columnas += COLUMNAS_ADICIONALES.get(hoja, [])

        for columna in columnas:
            if columna not in marco.columns:
                logger.warning("Columna ausente en '%s': '%s...'", hoja, columna[:55])
                continue
            original = marco[columna].fillna("").astype(str).tolist()
            clean = [_normalizar_minimo(t, recursos) for t in original]
            # Conteo previo: cuántas sobrevivirían a la higiene.
            modelables = sum(
                1 for t in clean if t and t not in RUIDO and len(t.split()) >= 3
            )
            etiqueta = _etiqueta_columna(hoja, columna)
            es_modelable = modelables >= minimo_modelables
            cargadas[etiqueta] = Encuesta(
                nombre=etiqueta,
                columna=columna,
                marco=marco,
                texto_original=original,
                texto_clean=clean,
                modelable=es_modelable,
                n_modelables=modelables,
            )
            logger.info(
                "  %-46s %7d filas | %6d modelables %s",
                etiqueta[:46],
                len(original),
                modelables,
                "" if es_modelable else "-> NO MODELABLE (volumen insuficiente)",
            )

    if not cargadas:
        raise ValueError(
            "Ninguna columna pudo cargarse. Revise ENCUESTAS contra la salida."
        )
    n_ok = sum(1 for e in cargadas.values() if e.modelable)
    logger.info(
        "Total: %d columnas de texto | %d modelables | %d descartadas por volumen",
        len(cargadas),
        n_ok,
        len(cargadas) - n_ok,
    )
    return cargadas


def _etiqueta_columna(hoja: str, columna: str) -> str:
    """Nombre corto y estable para identificar hoja + columna."""
    numero = re.match(r"Observaciones - (\d+)\.", columna)
    if numero:
        return f"{hoja} :: P{numero.group(1)}"
    return f"{hoja} :: {columna[:38].strip()}"


def sugerir_columnas_texto(
    marco: pd.DataFrame, min_palabras_media: float = 3.0
) -> pd.DataFrame:
    """Sugiere qué columnas de una hoja son candidatas a texto abierto.

    NO decide nada por sí sola: `ENCUESTAS` sigue siendo una elección
    manual, porque el nombre exacto hay que revisarlo. Esta función sólo
    reduce el trabajo de mirar 148 columnas para encontrar la que interesa —
    y el corpus real tiene hojas con exactamente ese número.

    La heurística usa UN solo criterio duro, a propósito:
      - Las columnas NUMÉRICAS se descartan directo: son calificaciones.
      - Entre las de texto, decide `palabras_media`. Una escala corta tiene
        un puñado de palabras por celda; una respuesta abierta tiene frases.

    `pct_unicos` se calcula y se muestra, pero NO se exige alto: en el
    corpus real el 60,7% de Punto 19 es la cadena "NR" repetida, y la
    columna sigue siendo texto abierto legítimo.
    """
    filas: list[dict[str, Any]] = []
    for columna in marco.columns:
        serie = marco[columna]
        if pd.api.types.is_numeric_dtype(serie):
            continue
        texto = serie.dropna().astype(str)
        texto = texto[texto.str.strip() != ""]
        if texto.empty:
            continue
        palabras = texto.str.split().str.len()
        palabras_media = round(float(palabras.mean()), 2)
        filas.append(
            {
                "columna": columna,
                "filas_no_vacias": len(texto),
                "pct_no_vacias": round(100 * len(texto) / len(serie), 1),
                "palabras_media": palabras_media,
                "palabras_mediana": float(palabras.median()),
                "valores_unicos": int(texto.nunique()),
                "pct_unicos": round(100 * texto.nunique() / max(len(texto), 1), 1),
                "es_candidata_texto_abierto": palabras_media >= min_palabras_media,
            }
        )
    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return tabla
    return tabla.sort_values("palabras_media", ascending=False).reset_index(drop=True)


def sugerir_encuestas(ruta_corpus: Path, **kwargs: Any) -> dict[str, pd.DataFrame]:
    """Aplica :func:`sugerir_columnas_texto` a todas las hojas del corpus.

    Uso típico: revisar la salida y copiar los nombres elegidos a
    `ENCUESTAS` y `COLUMNAS_ADICIONALES`. Nunca escribe en ellos.
    """
    libro = pd.read_excel(ruta_corpus, sheet_name=None)
    return {
        hoja: sugerir_columnas_texto(marco, **kwargs) for hoja, marco in libro.items()
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7. RAMA A — MODELADO DE TÓPICOS
# ═══════════════════════════════════════════════════════════════════════════


class ModeladorTopicos:
    """BERTopic configurado para respuestas abiertas cortas en español."""

    def __init__(self, cfg: Config, recursos: RecursosLexicos) -> None:
        self.cfg = cfg
        self.recursos = recursos
        self.modelo: Any = None
        self.docs_originales: list[str] = []
        self.docs_modelo: list[str] = []  # tras higiene y deduplicación
        self.embeddings: np.ndarray | None = None
        self.topicos: list[int] = []  # alineado con docs_originales
        self.higiene: dict[str, Any] = {}  # desglose del descarte
        self.crecimiento_cascada: dict[int, float] = {}  # % por tópico
        self._idx_validos: np.ndarray | None = None
        self._mapa_dedup: np.ndarray | None = None
        self._min_topic_size: int = 0  # valor efectivo
        self._min_samples: int = 0  # valor efectivo

        # El patrón sale de negaciones.xlsx, para que las etiquetas de tópico
        # puedan mostrar "no explica" como bigrama.
        self.token_pattern = construir_token_pattern(
            recursos.terminos_por_tipo("NEGADOR", con_tilde=False)
        )

    # ── Hiperparámetros adaptativos ────────────────────────────────────────

    def _resolver_hiperparametros(self, n_unicos: int) -> None:
        """Fija min_topic_size y min_samples según el tamaño real del corpus.

        Es la corrección central de esta versión. Un `min_topic_size`
        absoluto no dice nada sin saber cuántos documentos hay; la fracción
        sí es interpretable y escala sola entre encuestas de tamaños muy
        distintos.
        """
        if self.cfg.min_topic_size is not None:
            self._min_topic_size = int(self.cfg.min_topic_size)
        else:
            # Fracción del corpus, con un suelo para corpus pequeños.
            self._min_topic_size = max(
                25, int(round(n_unicos * self.cfg.fraccion_topico_minimo))
            )

        if self.cfg.min_samples is not None:
            self._min_samples = int(self.cfg.min_samples)
        else:
            self._min_samples = max(
                2, int(round(self._min_topic_size * self.cfg.ratio_min_samples))
            )

        ratio = self._min_samples / max(self._min_topic_size, 1)
        logger.info(
            "  hiperparámetros: min_topic_size=%d (%.2f%% del corpus único) | "
            "min_samples=%d (ratio %.2f) | techo teórico ~%d tópicos",
            self._min_topic_size,
            100 * self._min_topic_size / max(n_unicos, 1),
            self._min_samples,
            ratio,
            n_unicos // max(self._min_topic_size, 1),
        )
        if ratio > 0.20:
            logger.warning(
                "  ratio min_samples/min_topic_size = %.2f (>0.20): HDBSCAN se "
                "vuelve conservador y la tasa de outliers se dispara.",
                ratio,
            )

    # ── Higiene ────────────────────────────────────────────────────────────

    def _higiene(self, docs: Sequence[str]) -> tuple[list[str], np.ndarray]:
        """Separa las respuestas modelables, desglosando el descarte.

        Tres causas distintas, que confundidas hacen parecer que el modelo
        tira datos cuando la mayoría del corpus son celdas vacías:

          VACÍA  no contestó la pregunta abierta (es opcional). No es una
                 pérdida: nunca hubo texto. Es la tasa de respuesta del
                 instrumento, un dato de la encuesta y no del modelo.
          RUIDO  contestó algo que no es opinión: "NR", "n/a", "-".
          CORTA  contestó menos de `min_tokens` palabras: "excelente".
                 Es la única categoría con información real que el modelo de
                 tópicos no puede aprovechar.
        """
        validos: list[str] = []
        indices: list[int] = []
        n_vacias = n_ruido = n_cortas = 0
        ejemplos_cortas: list[str] = []

        for i, doc in enumerate(docs):
            texto = str(doc).strip()
            if not texto:  # la pregunta abierta es opcional
                n_vacias += 1
                continue
            if texto.lower() in RUIDO:
                n_ruido += 1
                continue
            if len(texto.split()) < self.cfg.min_tokens:
                n_cortas += 1
                if len(ejemplos_cortas) < 200:
                    ejemplos_cortas.append(texto)
                continue
            validos.append(texto)
            indices.append(i)

        # Denominador honesto: sobre respuestas CONTESTADAS, no sobre filas.
        contestadas = len(docs) - n_vacias
        self.higiene = {
            "filas_totales": len(docs),
            "vacias": n_vacias,
            "pct_vacias": round(100 * n_vacias / max(len(docs), 1), 2),
            "contestadas": contestadas,
            "tasa_respuesta": round(100 * contestadas / max(len(docs), 1), 2),
            "ruido": n_ruido,
            "cortas": n_cortas,
            "modelables": len(validos),
            "pct_modelables_de_contestadas": round(
                100 * len(validos) / max(contestadas, 1), 2
            ),
            "ejemplos_cortas": ejemplos_cortas[:20],
        }
        logger.info(
            "  higiene: %d filas -> %d vacías (%.1f%%, pregunta opcional) | "
            "%d contestadas: %d ruido, %d cortas, %d MODELABLES (%.1f%% de las "
            "contestadas)",
            len(docs),
            n_vacias,
            self.higiene["pct_vacias"],
            contestadas,
            n_ruido,
            n_cortas,
            len(validos),
            self.higiene["pct_modelables_de_contestadas"],
        )
        return validos, np.array(indices, dtype=int)

    @staticmethod
    def _deduplicar(docs: Sequence[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Colapsa respuestas idénticas conservando la PRIMERA aparición.

        Ajustar sobre duplicados da a HDBSCAN densidad artificial: mil veces
        "excelente profesor" parece un cluster densísimo que no lo es.
        """
        vistos: dict[str, int] = {}
        unicos: list[str] = []
        primeras: list[int] = []
        mapa = np.empty(len(docs), dtype=int)
        for i, doc in enumerate(docs):
            idx = vistos.get(doc)
            if idx is None:
                idx = len(unicos)
                vistos[doc] = idx
                unicos.append(doc)
                primeras.append(i)
            mapa[i] = idx
        return unicos, mapa, np.array(primeras, dtype=int)

    # ── Embeddings ─────────────────────────────────────────────────────────

    def codificar(self, docs: Sequence[str], etiqueta: str = "") -> np.ndarray:
        """Codifica texto a vectores densos, con caché en disco.

        La caché es lo que hace viable el grid search: codificar 20.000
        respuestas tarda minutos; recuperarlas del disco, milisegundos.
        """
        firma = hashlib.sha256()
        firma.update(self.cfg.encoder.encode())
        firma.update(str(len(docs)).encode())
        for doc in docs[:: max(1, len(docs) // 500)]:  # muestreo estable
            firma.update(doc.encode("utf-8", "ignore"))
        ruta = Path("cache/embeddings") / f"emb_{firma.hexdigest()[:16]}.npy"

        if self.cfg.usar_cache and ruta.exists():
            logger.info("  embeddings %s recuperados de caché", etiqueta)
            return np.load(ruta)

        from sentence_transformers import SentenceTransformer

        logger.info("  codificando %d textos %s...", len(docs), etiqueta)
        encoder = SentenceTransformer(self.cfg.encoder)
        matriz = np.asarray(
            encoder.encode(
                list(docs),
                batch_size=self.cfg.batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,  # coseno == producto punto
            ),
            dtype=np.float32,
        )
        if self.cfg.usar_cache:
            ruta.parent.mkdir(parents=True, exist_ok=True)
            np.save(ruta, matriz)
        return matriz

    # ── Construcción ───────────────────────────────────────────────────────

    def _construir(self) -> Any:
        """Ensambla los cinco sub-modelos de BERTopic."""
        from bertopic import BERTopic
        from bertopic.representation import MaximalMarginalRelevance
        from bertopic.vectorizers import ClassTfidfTransformer
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP

        # (1) Reducción: 384 dimensiones son demasiadas para una métrica de
        #     densidad. HDBSCAN trabaja bien en 5.
        umap_model = UMAP(
            n_neighbors=self.cfg.umap_n_neighbors,
            n_components=self.cfg.umap_n_components,
            min_dist=self.cfg.umap_min_dist,
            metric="cosine",
            random_state=self.cfg.random_state,
        )
        # (2) Clustering por densidad: no hay que fijar k.
        hdbscan_model = HDBSCAN(
            min_cluster_size=self._min_topic_size,
            min_samples=self._min_samples,
            metric="euclidean",  # ya estamos en el espacio UMAP
            cluster_selection_method="eom",
            prediction_data=True,  # necesario para transform() y ABSA
        )
        # (3) ★ AQUÍ ENTRAN LAS STOPWORDS ★ — después del clustering.
        vectorizer_model = CountVectorizer(
            stop_words=self.recursos.stopwords_para_vectorizador(self.token_pattern),
            ngram_range=self.cfg.ngram_range,
            # min_df/max_df neutros a propósito: BERTopic ajusta este
            # vectorizador sobre documentos AGREGADOS POR TÓPICO. min_df=5
            # exigiría presencia en 5 TÓPICOS y borraría el vocabulario
            # distintivo, que es justo lo que se busca conservar.
            min_df=1,
            max_df=1.0,
            token_pattern=self.token_pattern,
        )
        # (4) c-TF-IDF: TF-IDF a nivel de tópico en vez de documento.
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        # (5) MMR evita que los 10 términos sean sinónimos entre sí.
        representation_model = MaximalMarginalRelevance(
            diversity=self.cfg.mmr_diversity
        )

        return BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,
            embedding_model=self.cfg.encoder,  # imprescindible para MMR
            top_n_words=self.cfg.top_n_words,
            calculate_probabilities=False,
            verbose=False,
        )

    # ── Entrenamiento ──────────────────────────────────────────────────────

    def entrenar(self, docs: Sequence[str]) -> ModeladorTopicos:
        """Ejecuta el pipeline completo de tópicos."""
        self.docs_originales = [str(d) for d in docs]

        # 1. Separar lo modelable de lo que no lo es.
        validos, self._idx_validos = self._higiene(self.docs_originales)
        if not validos:
            raise ValueError("La higiene descartó todo. Revise min_tokens.")

        # 2. Colapsar duplicados exactos.
        if self.cfg.deduplicar:
            unicos, self._mapa_dedup, _ = self._deduplicar(validos)
            logger.info(
                "  deduplicación: %d -> %d únicos (%.1f%% de ahorro)",
                len(validos),
                len(unicos),
                100 * (1 - len(unicos) / len(validos)),
            )
        else:
            unicos = validos
            self._mapa_dedup = np.arange(len(validos))
        self.docs_modelo = unicos

        # 3. Fijar hiperparámetros AHORA, cuando ya se sabe el tamaño real.
        self._resolver_hiperparametros(len(unicos))

        # 4. Codificar sólo los únicos.
        self.embeddings = self.codificar(unicos, "(tópicos)")

        # 5. Ajustar.
        logger.info(
            "  entrenando BERTopic (min_topic_size=%d, min_samples=%d, "
            "n_neighbors=%d)...",
            self._min_topic_size,
            self._min_samples,
            self.cfg.umap_n_neighbors,
        )
        self.modelo = self._construir()
        asignaciones = np.asarray(self.modelo.fit_transform(unicos, self.embeddings)[0])
        logger.info(
            "  partición inicial: %d tópicos, %d outliers (%.1f%%)",
            len({t for t in asignaciones if t >= 0}),
            int((asignaciones == OUTLIER).sum()),
            100 * float((asignaciones == OUTLIER).mean()),
        )

        # 6. Fusionar tópicos redundantes, si se pidió.
        if self.cfg.nr_topics is not None:
            self.modelo.reduce_topics(unicos, nr_topics=self.cfg.nr_topics)
            asignaciones = np.asarray(self.modelo.topics_)
            logger.info(
                "  tras fusión: %d tópicos", len({t for t in asignaciones if t >= 0})
            )

        # 7. Recuperar outliers en cascada.
        if self.cfg.reducir_outliers and (asignaciones == OUTLIER).any():
            asignaciones = self._reducir_outliers(unicos, asignaciones)

        # 8. Proyectar al corpus completo.
        self.topicos = self._proyectar(asignaciones)
        self._resumen()
        return self

    def _reducir_outliers(self, docs: list[str], topicos: np.ndarray) -> np.ndarray:
        """Cascada de estrategias de reasignación, con tope de crecimiento.

        Cada estrategia ataca los outliers que dejó la anterior. "c-tf-idf"
        sola es casi inútil en respuestas cortas: el vector disperso de la
        respuesta apenas solapa con el del tópico. "embeddings" compara
        vectores densos y recupera mucho más.

        El tope es la corrección clave. Sin él, cada outlier acaba en el
        centroide más cercano y los tópicos grandes actúan como imanes:
        medido sobre el corpus real de Punto 19, la cascada sin tope subió la
        concentración en los dos tópicos mayores del 15,4% al 27,2%. Con
        `tope_crecimiento_cascada` un tópico no puede crecer más allá de esa
        fracción de su tamaño original; lo que no cabe se queda como outlier,
        que es la respuesta honesta.
        """
        n_inicial = int((topicos == OUTLIER).sum())
        tamano_original = pd.Series(topicos[topicos >= 0]).value_counts().to_dict()
        nuevos = np.asarray(topicos)

        for estrategia in self.cfg.cascada_outliers:
            restantes = int((nuevos == OUTLIER).sum())
            if restantes == 0:
                break
            argumentos: dict[str, Any] = {
                "strategy": estrategia,
                "threshold": self.cfg.umbral_outliers,
            }
            if estrategia == "embeddings":
                argumentos["embeddings"] = self.embeddings
            try:
                candidatos = np.asarray(
                    self.modelo.reduce_outliers(docs, list(nuevos), **argumentos)
                )
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                logger.warning("    '%s' falló: %s", estrategia, exc)
                continue
            candidatos = self._aplicar_tope(nuevos, candidatos, tamano_original)
            quedan = int((candidatos == OUTLIER).sum())
            logger.info(
                "    '%s': %d -> %d outliers (recupera %d)",
                estrategia,
                restantes,
                quedan,
                restantes - quedan,
            )
            nuevos = candidatos

        self.modelo.update_topics(
            docs,
            topics=list(nuevos),
            vectorizer_model=self.modelo.vectorizer_model,
            ctfidf_model=self.modelo.ctfidf_model,
            representation_model=self.modelo.representation_model,
            top_n_words=self.cfg.top_n_words,
        )
        n_final = int((nuevos == OUTLIER).sum())
        self.crecimiento_cascada = {
            int(t): round(100 * (int((nuevos == t).sum()) / max(n, 1) - 1), 1)
            for t, n in tamano_original.items()
        }
        logger.info(
            "  outliers: %d -> %d (recuperado %.1f%%) | crecimiento máximo de "
            "un tópico: %+.1f%%",
            n_inicial,
            n_final,
            100 * (n_inicial - n_final) / max(n_inicial, 1),
            max(self.crecimiento_cascada.values(), default=0.0),
        )
        return nuevos

    def _aplicar_tope(
        self,
        antes: np.ndarray,
        despues: np.ndarray,
        tamano_original: dict[Any, int],
    ) -> np.ndarray:
        """Revierte las reasignaciones que exceden el tope de crecimiento.

        Se recorren los documentos recién asignados y se aceptan mientras el
        tópico destino no supere su cupo; el resto vuelve a OUTLIER.
        """
        if self.cfg.tope_crecimiento_cascada <= 0:
            return despues
        cupo = {
            t: int(round(n * self.cfg.tope_crecimiento_cascada))
            for t, n in tamano_original.items()
        }
        usado: dict[Any, int] = dict.fromkeys(cupo, 0)
        salida = np.asarray(despues).copy()
        recien = np.where((antes == OUTLIER) & (despues >= 0))[0]
        revertidos = 0
        for i in recien:
            destino = int(despues[i])
            if usado.get(destino, 0) < cupo.get(destino, 0):
                usado[destino] = usado.get(destino, 0) + 1
            else:
                salida[i] = OUTLIER
                revertidos += 1
        if revertidos:
            logger.info(
                "      tope de crecimiento: %d reasignaciones revertidas a outlier",
                revertidos,
            )
        return salida

    def _proyectar(self, asignaciones: np.ndarray) -> list[int]:
        """Lleva las asignaciones de los únicos al corpus original."""
        # Se inicializa en EXCLUIDO y luego se rellenan los válidos.
        salida = np.full(len(self.docs_originales), EXCLUIDO, dtype=int)
        salida[self._idx_validos] = asignaciones[self._mapa_dedup]
        return salida.tolist()

    def _resumen(self) -> None:
        """Traza final con denominadores explícitos."""
        arr = np.asarray(self.topicos)
        n_exc = int((arr == EXCLUIDO).sum())
        n_out = int((arr == OUTLIER).sum())
        modelado = len(arr) - n_exc
        logger.info(
            "  RESULTADO: %d tópicos | %d respuestas clasificadas | outliers %d "
            "(%.1f%% de lo modelado) | fuera del modelo %d (vacías/ruido/cortas)",
            len({t for t in arr if t >= 0}),
            modelado - n_out,
            n_out,
            100 * n_out / max(modelado, 1),
            n_exc,
        )

    # ── Resultados ─────────────────────────────────────────────────────────

    def metricas(self) -> dict[str, Any]:
        """Métricas estructurales de la corrida."""
        arr = np.asarray(self.topicos)
        n_exc = int((arr == EXCLUIDO).sum())
        modelado = len(arr) - n_exc
        tamanos = pd.Series(arr[arr >= 0]).value_counts()
        return {
            "filas": len(arr),
            "contestadas": self.higiene.get("contestadas", 0),
            "modelables": self.higiene.get("modelables", 0),
            "unicos": len(self.docs_modelo),
            "min_topic_size": self._min_topic_size,
            "min_samples": self._min_samples,
            "n_topicos": int(tamanos.size),
            "pct_outliers": round(
                100 * int((arr == OUTLIER).sum()) / max(modelado, 1), 2
            ),
            "tamano_mediano": float(tamanos.median()) if tamanos.size else 0.0,
            "tamano_min": int(tamanos.min()) if tamanos.size else 0,
            "tamano_max": int(tamanos.max()) if tamanos.size else 0,
            "diversidad": round(self.diversidad(), 4),
            "pares_redundantes": len(self.redundancia()),
            **concentracion(arr),
            "crecimiento_max_cascada": round(
                max(self.crecimiento_cascada.values(), default=0.0), 1
            ),
        }

    def diversidad(self, top_n: int = 10) -> float:
        """Proporción de términos únicos entre los top-N de cada tópico.

        Baja diversidad = los tópicos se describen con las mismas palabras,
        señal de fragmentación.
        """
        grupos = [
            [w for w, _ in (self.modelo.get_topic(t) or [])][:top_n]
            for t in sorted(k for k in self.modelo.get_topics() if k != OUTLIER)
        ]
        todos = [w for g in grupos for w in g]
        return len(set(todos)) / len(todos) if todos else 0.0

    def redundancia(self, umbral: float = 0.20) -> pd.DataFrame:
        """Pares de tópicos casi idénticos según distancia c-TF-IDF."""
        from sklearn.metrics.pairwise import cosine_distances

        columnas = ["topico_a", "topico_b", "distancia"]
        matriz = getattr(self.modelo, "c_tf_idf_", None)
        if matriz is None:
            return pd.DataFrame(columns=columnas)
        orden = sorted(self.modelo.get_topics())
        denso = np.asarray(matriz.todense())
        posicion = {t: i for i, t in enumerate(orden) if i < denso.shape[0]}
        ids = [t for t in orden if t != OUTLIER and t in posicion]
        if len(ids) < 2:
            return pd.DataFrame(columns=columnas)
        distancias = cosine_distances(denso[[posicion[t] for t in ids]])
        filas = [
            {
                "topico_a": ids[i],
                "topico_b": ids[j],
                "distancia": round(float(distancias[i, j]), 4),
            }
            for i in range(len(ids))
            for j in range(i + 1, len(ids))
            if distancias[i, j] <= umbral
        ]
        return (
            pd.DataFrame(filas).sort_values("distancia").reset_index(drop=True)
            if filas
            else pd.DataFrame(columns=columnas)
        )

    def tabla_topicos(self) -> pd.DataFrame:
        """Tabla de tópicos con tamaño en únicos y en corpus completo."""
        info = self.modelo.get_topic_info()
        info = info[info["Topic"] != OUTLIER].copy()
        # `Count` cuenta documentos ÚNICOS; `Respuestas` cuenta respuestas
        # reales, que es la cifra de prevalencia que va al informe.
        conteo = pd.Series(self.topicos).value_counts()
        info["Respuestas"] = info["Topic"].map(conteo).fillna(0).astype(int)
        info["Terminos"] = info["Topic"].map(
            lambda t: " | ".join(w for w, _ in (self.modelo.get_topic(t) or [])[:8])
        )
        return info[["Topic", "Count", "Respuestas", "Name", "Terminos"]].reset_index(
            drop=True
        )

    def asignar(self, textos: Sequence[str], embeddings: np.ndarray) -> np.ndarray:
        """Asigna textos NUEVOS a los tópicos ya aprendidos.

        Es la pieza que permite ABSA: las cláusulas no estaban en el
        entrenamiento, pero se proyectan al mismo espacio.
        """
        asignados, _ = self.modelo.transform(list(textos), embeddings)
        return np.asarray(asignados)


def concentracion(asignaciones: np.ndarray) -> dict[str, float]:
    """Mide cuánta masa del corpus absorben los tópicos más grandes.

    Es la métrica que faltaba. Un modelo puede tener 16 tópicos, pocos
    outliers y coherencia alta, y aun así ser inservible si dos de esos
    tópicos se llevan el 79% de las respuestas: los otros catorce serían
    residuos decorativos.

    Devuelve:
      pct_top1, pct_top2, pct_top3  masa acumulada en los mayores
      gini                          0 = tópicos del mismo tamaño,
                                    1 = uno se lo lleva todo
    """
    asignados = np.asarray(asignaciones)
    asignados = asignados[asignados >= 0]
    if asignados.size == 0:
        return {"pct_top1": 100.0, "pct_top2": 100.0, "pct_top3": 100.0, "gini": 1.0}
    tam = np.sort(pd.Series(asignados).value_counts().to_numpy())[::-1]
    total = tam.sum()

    # Gini sobre los tamaños de tópico.
    orden = np.sort(tam)
    n = orden.size
    indices = np.arange(1, n + 1)
    gini = float((2 * (indices * orden).sum()) / (n * orden.sum()) - (n + 1) / n)

    return {
        "pct_top1": round(100 * float(tam[:1].sum() / total), 1),
        "pct_top2": round(100 * float(tam[:2].sum() / total), 1),
        "pct_top3": round(100 * float(tam[:3].sum() / total), 1),
        "gini": round(gini, 3),
    }


def validar_contra_nota(
    clausulas: pd.DataFrame, notas: Sequence[float]
) -> dict[str, Any]:
    """Contrasta la polaridad detectada contra una calificación numérica.

    Es el sustituto del etiquetado manual cuando no hay anotadores. La
    calificación de 1 a 5 es una medida independiente del texto, producida
    por la misma persona en el mismo formulario: si el modelo de sentimiento
    funciona en este dominio, las cláusulas NEG deben concentrarse en notas
    bajas y las POS en notas altas.

    No reemplaza la anotación de TÓPICO — para eso no hay atajo — pero sí
    valida la rama de sentimiento con evidencia externa y reproducible.

    Args:
        clausulas: debe traer `doc_id` y `sentimiento`.
        notas: calificación por documento del corpus original.

    Returns:
        Nota media por polaridad, separación POS-NEG y correlación.
    """
    from scipy.stats import kruskal, spearmanr

    serie = pd.Series(list(notas), dtype="float64")
    marco = clausulas.copy()
    marco["nota"] = marco["doc_id"].map(serie)
    marco = marco.dropna(subset=["nota"])
    if marco.empty:
        return {"n": 0, "advertencia": "Ninguna cláusula tiene nota asociada."}

    por_polaridad = marco.groupby("sentimiento")["nota"].agg(
        ["mean", "median", "count"]
    )
    media = por_polaridad["mean"].to_dict()
    separacion = float(media.get(POS, np.nan) - media.get(NEG, np.nan))

    # Correlación entre polaridad ordinal (-1, 0, 1) y nota.
    escala = {NEG: -1, NEU: 0, POS: 1}
    ordinal = marco["sentimiento"].map(escala).astype("float64")
    rho, p_rho = spearmanr(ordinal, marco["nota"])

    grupos = [
        g["nota"].to_numpy() for _, g in marco.groupby("sentimiento") if len(g) > 1
    ]
    h, p_h = kruskal(*grupos) if len(grupos) >= 2 else (np.nan, np.nan)

    return {
        "n_clausulas_con_nota": int(len(marco)),
        "nota_media_POS": round(float(media.get(POS, np.nan)), 3),
        "nota_media_NEU": round(float(media.get(NEU, np.nan)), 3),
        "nota_media_NEG": round(float(media.get(NEG, np.nan)), 3),
        "separacion_POS_menos_NEG": round(separacion, 3),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(p_rho),
        "kruskal_H": round(float(h), 2),
        "kruskal_p": float(p_h),
        "detalle": por_polaridad.round(3),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 8. BÚSQUEDA DE HIPERPARÁMETROS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CriteriosNegocio:
    """Qué se considera un modelo aceptable para un informe.

    No se optimiza coherencia. Un modelo de 2 tópicos gigantes puede tener
    coherencia altísima y no decir nada; uno de 136 tópicos con 40% de
    outliers tampoco sirve. Lo que hace útil un modelo de tópicos en un
    informe es que quepa en una página y cubra el corpus.
    """

    n_topicos_min: int = 8  # menos de esto, no discrimina
    n_topicos_max: int = 25  # más de esto, nadie lo lee en una reunión
    pct_outliers_max: float = 15.0  # techo de opiniones sin clasificar
    # Ningún tópico debe absorber más de esta fracción: si uno se lleva el
    # 70% del corpus, el modelo no separó nada.
    pct_topico_mayor_max: float = 40.0
    # Techo de masa acumulada en los DOS tópicos mayores. Es el criterio que
    # detecta el colapso: 16 tópicos con el 79% del corpus en dos de ellos
    # pasa todos los demás filtros y aun así no sirve.
    pct_top2_max: float = 45.0
    # Pesos del puntaje compuesto. Suman 1.
    peso_topicos: float = 0.25
    peso_outliers: float = 0.25
    peso_equilibrio: float = 0.25
    peso_diversidad: float = 0.15
    peso_redundancia: float = 0.10


def _decae(valor: float, limite: float, escala: float) -> float:
    """Puntúa 1 dentro del límite y decae linealmente fuera de él."""
    if valor <= limite:
        return 1.0
    return float(max(0.0, 1.0 - (valor - limite) / escala))


def puntuar(metricas: dict[str, Any], criterios: CriteriosNegocio) -> dict[str, float]:
    """Convierte las métricas de una corrida en un puntaje de 0 a 1.

    Devuelve cada componente por separado para que la elección del modelo
    sea auditable y no una caja negra.
    """
    n_topicos = int(metricas.get("n_topicos", 0))
    pct_out = float(metricas.get("pct_outliers", 100.0))
    pct_mayor = float(metricas.get("pct_topico_mayor", 100.0))

    # (1) Número de tópicos dentro del rango legible.
    if criterios.n_topicos_min <= n_topicos <= criterios.n_topicos_max:
        p_topicos = 1.0
    elif n_topicos < criterios.n_topicos_min:
        # Penaliza fuerte: 2 tópicos no discriminan nada.
        p_topicos = _decae(
            criterios.n_topicos_min - n_topicos, 0, criterios.n_topicos_min
        )
    else:
        # Escala generosa: 30 tópicos no es tan grave como 130.
        p_topicos = _decae(n_topicos, criterios.n_topicos_max, 60.0)

    # (2) Outliers por encima del techo.
    p_outliers = _decae(pct_out, criterios.pct_outliers_max, 35.0)
    # (3) Equilibrio: ni un tópico ni la pareja mayor deben tragarse el
    #     corpus. Se toma el peor de los dos, no el promedio: si cualquiera
    #     de las dos condiciones falla, el modelo no sirve.
    pct_top2 = float(metricas.get("pct_top2", 100.0))
    p_equilibrio = min(
        _decae(pct_mayor, criterios.pct_topico_mayor_max, 40.0),
        _decae(pct_top2, criterios.pct_top2_max, 45.0),
    )
    # (4) Diversidad léxica entre tópicos.
    p_diversidad = float(metricas.get("diversidad", 0.0))
    # (5) Redundancia normalizada por el número de tópicos.
    p_redundancia = _decae(
        float(metricas.get("pares_redundantes", 0)), 0, max(n_topicos, 1)
    )

    puntaje = (
        criterios.peso_topicos * p_topicos
        + criterios.peso_outliers * p_outliers
        + criterios.peso_equilibrio * p_equilibrio
        + criterios.peso_diversidad * p_diversidad
        + criterios.peso_redundancia * p_redundancia
    )
    cumple = (
        criterios.n_topicos_min <= n_topicos <= criterios.n_topicos_max
        and pct_out <= criterios.pct_outliers_max
        and pct_mayor <= criterios.pct_topico_mayor_max
        and pct_top2 <= criterios.pct_top2_max
    )
    return {
        "p_topicos": round(p_topicos, 3),
        "p_outliers": round(p_outliers, 3),
        "p_equilibrio": round(p_equilibrio, 3),
        "p_diversidad": round(p_diversidad, 3),
        "p_redundancia": round(p_redundancia, 3),
        "puntaje": round(puntaje, 4),
        "cumple": int(cumple),
    }


class GridSearchTopicos:
    """Búsqueda de hiperparámetros por etapas sobre UMAP y HDBSCAN.

    EL TRUCO QUE LA HACE VIABLE
    ---------------------------
    UMAP es lento (minutos sobre 20.000 documentos); HDBSCAN es rápido
    (segundos). Un grid ingenuo recalcula UMAP en cada combinación y tarda
    horas — que es exactamente lo que ocurrió en la primera corrida, donde
    cada punto tardaba más de un minuto.

    Aquí la proyección UMAP se calcula UNA vez por configuración de UMAP y
    se reutiliza para todas las combinaciones de HDBSCAN que cuelgan de
    ella. Con 3 configuraciones de UMAP y 12 de HDBSCAN son 36 modelos pero
    sólo 3 proyecciones.

    Uso:
        busqueda = GridSearchTopicos(cfg, recursos).preparar(docs)
        tabla = busqueda.ejecutar()
        mejor = busqueda.mejor_config()
    """

    def __init__(self, cfg: Config, recursos: RecursosLexicos) -> None:
        self.cfg = cfg
        self.recursos = recursos
        self.resultados: pd.DataFrame = pd.DataFrame()
        self.docs_modelo: list[str] = []
        self.embeddings: np.ndarray | None = None
        self.higiene: dict[str, Any] = {}

    # ── Preparación: se hace UNA vez para todo el grid ─────────────────────

    def preparar(self, docs: Sequence[str]) -> GridSearchTopicos:
        """Higiene, deduplicación y codificación, compartidas por el grid."""
        base = ModeladorTopicos(self.cfg, self.recursos)
        validos, _ = base._higiene(docs)  # pylint: disable=protected-access
        unicos, _, _ = base._deduplicar(validos)  # pylint: disable=protected-access
        self.docs_modelo = unicos
        self.higiene = base.higiene
        self.embeddings = base.codificar(unicos, "(grid search)")
        logger.info(
            "Grid preparado: %d documentos únicos, embeddings %s",
            len(unicos),
            self.embeddings.shape,
        )
        return self

    # ── Rejilla por defecto, calibrada al tamaño del corpus ────────────────

    def rejilla_sugerida(
        self, criterios: CriteriosNegocio | None = None
    ) -> tuple[list[dict[str, int]], list[dict[str, Any]]]:
        """Propone una rejilla razonable para este corpus.

        El rango de `min_topic_size` se deriva del número de tópicos que se
        quiere: si el objetivo son entre 8 y 25 tópicos sobre N documentos,
        el tamaño mínimo útil ronda N/(2·n_max) hasta N/(n_min/2).
        """
        crit = criterios or CriteriosNegocio()
        n = len(self.docs_modelo)
        if n == 0:
            raise RuntimeError("Llame a preparar() antes de rejilla_sugerida().")

        # Anclas derivadas del objetivo, no elegidas a ojo.
        bajo = max(50, int(n / (3 * crit.n_topicos_max)))
        alto = max(bajo * 2, int(n / max(crit.n_topicos_min, 1)))
        tamanos = sorted({int(round(v)) for v in np.geomspace(bajo, alto, 5)})

        rejilla_umap = [
            {"n_neighbors": 15, "n_components": 5},  # estructura local
            {"n_neighbors": 30, "n_components": 5},  # equilibrio
            {"n_neighbors": 50, "n_components": 5},  # estructura global
        ]
        # Se barre también `cluster_selection_method`: "leaf" corta el árbol
        # en las hojas y produce tópicos más parejos que "eom", que favorece
        # clusters grandes. Es el primer parámetro a mover cuando dos tópicos
        # absorben la mayoría del corpus.
        rejilla_hdbscan = [
            {"min_topic_size": t, "ratio_min_samples": r, "metodo": m}
            for t in tamanos
            for r in (0.02, 0.05)
            for m in ("eom", "leaf")
        ]
        logger.info(
            "Rejilla sugerida para %d documentos: min_topic_size %s x 3 ratios "
            "x 3 UMAP = %d combinaciones",
            n,
            tamanos,
            len(rejilla_umap) * len(rejilla_hdbscan),
        )
        return rejilla_umap, rejilla_hdbscan

    # ── Modelo sobre embeddings ya reducidos ───────────────────────────────

    def _modelo_sin_umap(
        self, min_topic_size: int, min_samples: int, metodo: str = "eom"
    ) -> Any:
        """BERTopic con la reducción dimensional desactivada.

        Recibe los embeddings YA proyectados por UMAP, de modo que sólo
        ejecuta HDBSCAN y la representación. Es lo que permite reutilizar
        una única proyección entre muchas combinaciones.
        """
        from bertopic import BERTopic
        from bertopic.dimensionality import BaseDimensionalityReduction
        from bertopic.representation import MaximalMarginalRelevance
        from bertopic.vectorizers import ClassTfidfTransformer
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer

        token_pattern = construir_token_pattern(
            self.recursos.terminos_por_tipo("NEGADOR", con_tilde=False)
        )
        return BERTopic(
            umap_model=BaseDimensionalityReduction(),  # paso directo
            hdbscan_model=HDBSCAN(
                min_cluster_size=min_topic_size,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_method=metodo,
                prediction_data=True,
            ),
            vectorizer_model=CountVectorizer(
                stop_words=self.recursos.stopwords_para_vectorizador(token_pattern),
                ngram_range=self.cfg.ngram_range,
                min_df=1,
                max_df=1.0,
                token_pattern=token_pattern,
            ),
            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
            representation_model=MaximalMarginalRelevance(
                diversity=self.cfg.mmr_diversity
            ),
            embedding_model=self.cfg.encoder,
            top_n_words=self.cfg.top_n_words,
            calculate_probabilities=False,
            verbose=False,
        )

    @staticmethod
    def _medir(modelo: Any, asignaciones: np.ndarray) -> dict[str, Any]:
        """Métricas estructurales de una combinación del grid."""
        from sklearn.metrics.pairwise import cosine_distances

        tamanos = pd.Series(asignaciones[asignaciones >= 0]).value_counts()
        n_topicos = int(tamanos.size)

        grupos = [
            [w for w, _ in (modelo.get_topic(t) or [])][:10]
            for t in sorted(k for k in modelo.get_topics() if k != OUTLIER)
        ]
        todos = [w for g in grupos for w in g]
        diversidad = len(set(todos)) / len(todos) if todos else 0.0

        pares = 0
        matriz = getattr(modelo, "c_tf_idf_", None)
        if matriz is not None and n_topicos >= 2:
            orden = sorted(modelo.get_topics())
            denso = np.asarray(matriz.todense())
            filas = [i for i, t in enumerate(orden) if t != OUTLIER and i < len(denso)]
            if len(filas) >= 2:
                distancias = cosine_distances(denso[filas])
                pares = int(
                    (distancias[np.triu_indices(len(filas), k=1)] <= 0.20).sum()
                )

        return {
            "n_topicos": n_topicos,
            "pct_outliers": round(100 * float((asignaciones == OUTLIER).mean()), 2),
            "tamano_mediano": float(tamanos.median()) if n_topicos else 0.0,
            "tamano_min": int(tamanos.min()) if n_topicos else 0,
            "pct_topico_mayor": (
                round(100 * float(tamanos.max() / len(asignaciones)), 2)
                if n_topicos
                else 100.0
            ),
            "diversidad": round(float(diversidad), 4),
            "pares_redundantes": pares,
        }

    # ── Ejecución ──────────────────────────────────────────────────────────

    def ejecutar(
        self,
        rejilla_umap: Sequence[dict[str, int]] | None = None,
        rejilla_hdbscan: Sequence[dict[str, Any]] | None = None,
        criterios: CriteriosNegocio | None = None,
        reducir_outliers: bool = True,
    ) -> pd.DataFrame:
        """Recorre el grid y devuelve una fila por combinación.

        Args:
            rejilla_umap: dicts con `n_neighbors` y `n_components`.
            rejilla_hdbscan: dicts con `min_topic_size` y `ratio_min_samples`
                o `min_samples`.
            criterios: qué se considera aceptable.
            reducir_outliers: aplica la estrategia "embeddings" antes de
                medir, para que la cifra sea la que realmente se obtendrá.

        Returns:
            DataFrame ordenado por cumplimiento y puntaje.
        """
        if self.embeddings is None:
            raise RuntimeError("Llame a preparar() antes de ejecutar().")
        from umap import UMAP

        crit = criterios or CriteriosNegocio()
        if rejilla_umap is None or rejilla_hdbscan is None:
            sugerida_umap, sugerida_hdbscan = self.rejilla_sugerida(crit)
            rejilla_umap = rejilla_umap or sugerida_umap
            rejilla_hdbscan = rejilla_hdbscan or sugerida_hdbscan

        n_unicos = len(self.docs_modelo)
        total = len(rejilla_umap) * len(rejilla_hdbscan)
        filas: list[dict[str, Any]] = []
        contador = 0

        for config_umap in rejilla_umap:
            # UMAP exige n_neighbors < n_documentos. En una encuesta chica
            # (algunas rondan unos cientos de respuestas únicas) una rejilla
            # pensada para corpus grandes puede pedir más vecinos de los que
            # hay documentos. Se salta esa configuración en vez de fallar.
            if config_umap["n_neighbors"] >= n_unicos:
                logger.warning(
                    "  se omite n_neighbors=%d: el corpus sólo tiene %d "
                    "documentos únicos (se necesita n_neighbors < n_unicos).",
                    config_umap["n_neighbors"],
                    n_unicos,
                )
                continue
            # ── UNA proyección por configuración de UMAP ──
            logger.info(
                "UMAP n_neighbors=%d n_components=%d — proyectando %d docs "
                "(esta es la parte lenta, se hace una sola vez)...",
                config_umap["n_neighbors"],
                config_umap["n_components"],
                n_unicos,
            )
            reducidos = UMAP(
                n_neighbors=config_umap["n_neighbors"],
                n_components=config_umap["n_components"],
                min_dist=self.cfg.umap_min_dist,
                metric="cosine",
                random_state=self.cfg.random_state,
            ).fit_transform(self.embeddings)

            for config_hdbscan in rejilla_hdbscan:
                contador += 1
                min_topic_size = int(config_hdbscan["min_topic_size"])
                min_samples = int(
                    config_hdbscan.get(
                        "min_samples",
                        max(
                            2,
                            round(
                                min_topic_size
                                * config_hdbscan.get("ratio_min_samples", 0.05)
                            ),
                        ),
                    )
                )
                metodo = str(config_hdbscan.get("metodo", "eom"))
                modelo = self._modelo_sin_umap(min_topic_size, min_samples, metodo)
                asignaciones = np.asarray(
                    modelo.fit_transform(self.docs_modelo, reducidos)[0]
                )
                if reducir_outliers and (asignaciones == OUTLIER).any():
                    try:
                        asignaciones = np.asarray(
                            modelo.reduce_outliers(
                                self.docs_modelo,
                                list(asignaciones),
                                strategy="embeddings",
                                embeddings=self.embeddings,
                                threshold=0.0,
                            )
                        )
                    except (ValueError, TypeError, KeyError, IndexError):
                        pass  # se mide la partición sin reducir

                medidas = self._medir(modelo, asignaciones)
                fila = {
                    "n_neighbors": config_umap["n_neighbors"],
                    "n_components": config_umap["n_components"],
                    "min_topic_size": min_topic_size,
                    "pct_corpus": round(100 * min_topic_size / n_unicos, 3),
                    "min_samples": min_samples,
                    "ratio_ms": round(min_samples / min_topic_size, 3),
                    "metodo": metodo,
                    **medidas,
                    **puntuar(medidas, crit),
                }
                filas.append(fila)
                logger.info(
                    "  [%2d/%2d] mts=%4d ms=%3d -> %3d tópicos | %5.1f%% outliers "
                    "| mayor %4.1f%% | puntaje %.3f%s",
                    contador,
                    total,
                    min_topic_size,
                    min_samples,
                    medidas["n_topicos"],
                    medidas["pct_outliers"],
                    medidas["pct_topico_mayor"],
                    fila["puntaje"],
                    "  <-- CUMPLE" if fila["cumple"] else "",
                )

        if not filas:
            raise RuntimeError(
                f"Ninguna combinación de la rejilla fue evaluable: el corpus "
                f"tiene {n_unicos} documentos únicos y todos los valores de "
                "n_neighbors lo superaron. Reduzca n_neighbors en la rejilla "
                "de UMAP o revise si esta encuesta tiene datos suficientes."
            )
        self.resultados = (
            pd.DataFrame(filas)
            .sort_values(["cumple", "puntaje"], ascending=False)
            .reset_index(drop=True)
        )
        cumplen = int(self.resultados["cumple"].sum())
        logger.info(
            "Grid terminado: %d combinaciones evaluadas de %d posibles, %d "
            "cumplen los criterios",
            len(filas),
            total,
            cumplen,
        )
        return self.resultados

    # ── Lectura del resultado ──────────────────────────────────────────────

    def mejor_config(self, indice: int = 0) -> Config:
        """Devuelve una `Config` lista para entrenar el modelo definitivo.

        El puntaje suele empatar entre modelos legítimamente distintos (uno
        de 6 tópicos gruesos y otro de 20 finos pueden ser ambos válidos).
        Por eso `indice` permite tomar cualquier fila de la tabla ordenada en
        lugar de imponer la primera: la decisión final es editorial, no
        aritmética.
        """
        if self.resultados.empty:
            raise RuntimeError("No hay resultados: ejecute ejecutar() primero.")
        mejor = self.resultados.iloc[indice]
        if not mejor["cumple"]:
            logger.warning(
                "Ninguna combinación cumple los criterios. Se devuelve la de mayor "
                "puntaje (%d tópicos, %.1f%% outliers). Considere ampliar la "
                "rejilla hacia min_topic_size mayores o relajar los criterios.",
                int(mejor["n_topicos"]),
                mejor["pct_outliers"],
            )
        nueva = Config(**{**asdict(self.cfg)})
        nueva.output_dir = Path(self.cfg.output_dir)
        nueva.umap_n_neighbors = int(mejor["n_neighbors"])
        nueva.umap_n_components = int(mejor["n_components"])
        nueva.min_topic_size = int(mejor["min_topic_size"])
        nueva.min_samples = int(mejor["min_samples"])
        if "metodo" in mejor:
            nueva.cluster_selection_method = str(mejor["metodo"])
        logger.info(
            "Mejor configuración: n_neighbors=%d, min_topic_size=%d, "
            "min_samples=%d -> %d tópicos, %.1f%% outliers",
            nueva.umap_n_neighbors,
            nueva.min_topic_size,
            nueva.min_samples,
            int(mejor["n_topicos"]),
            mejor["pct_outliers"],
        )
        return nueva


def buscar_config_por_encuesta(
    cargadas: dict[str, Encuesta],
    cfg_base: Config,
    recursos: RecursosLexicos,
    criterios: CriteriosNegocio | None = None,
) -> dict[str, Config]:
    """Corre `GridSearchTopicos` de forma INDEPENDIENTE para cada encuesta.

    Por qué no basta con una sola búsqueda: compartir la fracción y el ratio
    ganadores de una encuesta con las demás asume que todas tienen la misma
    estructura temática. Dos encuestas de tamaños parecidos pueden tener
    números de tópicos naturales muy distintos — una centrada en un tema
    puntual, otra cubriendo diez aspectos institucionales — y la fracción que
    a una le da 15 tópicos limpios a la otra le puede dar 3 o 40.

    Esta función repite el Paso 6 del cuaderno para cada hoja del corpus y
    devuelve una `Config` propia por encuesta, lista para `ejecutar_pipeline`.

    Args:
        cargadas: salida de `cargar_encuestas`.
        cfg_base: configuración base (encoder, modelo de sentimiento, etc.).
            Sólo se sobrescriben los campos que decide el grid: UMAP,
            `min_topic_size`, `min_samples`.
        recursos: léxicos ya cargados.
        criterios: mismos criterios de negocio para todas las encuestas. Si
            una encuesta necesita un rango distinto (p. ej. porque es mucho
            más pequeña), llame a `GridSearchTopicos` directamente para esa
            encuesta con su propio `CriteriosNegocio`.

    Returns:
        Mapa nombre de encuesta -> `Config` con los hiperparámetros óptimos
        encontrados para ESA encuesta.
    """
    criterios = criterios or CriteriosNegocio()
    configs: dict[str, Config] = {}
    for nombre, encuesta in cargadas.items():
        if not encuesta.modelable:
            logger.info("Se omite el grid de '%s': volumen insuficiente.", nombre)
            continue
        logger.info("=" * 74)
        logger.info("GRID SEARCH — %s", nombre)
        logger.info("=" * 74)
        try:
            base_encuesta = Config(**{**asdict(cfg_base), "nombre": nombre})
            base_encuesta.output_dir = Path(cfg_base.output_dir)
            busqueda = GridSearchTopicos(base_encuesta, recursos).preparar(
                encuesta.texto_clean
            )
            busqueda.ejecutar(criterios=criterios)
            configs[nombre] = busqueda.mejor_config()
            configs[nombre].nombre = nombre
        except (ValueError, RuntimeError, KeyError, TypeError) as exc:
            # TypeError incluido a propósito: UMAP/scipy la lanzan (en vez de
            # un error propio) cuando una encuesta tiene muy pocos documentos
            # únicos para la rejilla de n_neighbors pedida.
            logger.error(
                "Grid search de '%s' falló (%s: %s); se usa cfg_base sin ajustar.",
                nombre,
                type(exc).__name__,
                exc,
            )
            configs[nombre] = Config(**{**asdict(cfg_base), "nombre": nombre})
    return configs


# ═══════════════════════════════════════════════════════════════════════════
# 9. RAMA B — SEGMENTACIÓN EN CLÁUSULAS
# ═══════════════════════════════════════════════════════════════════════════


class Segmentador:
    """Parte una respuesta en unidades de opinión independientes.

    Es el paso que hace posible ABSA. Una respuesta como

        "el profesor explica muy bien PERO nunca responde los correos"

    tiene UNA valoración positiva y UNA negativa sobre DOS aspectos
    distintos. Darle una sola polaridad sería falso en la mitad del texto.

    Los puntos de corte salen de negaciones.xlsx, tipo ADVERSATIVO.
    """

    def __init__(self, cfg: Config, recursos: RecursosLexicos) -> None:
        self.cfg = cfg
        adversativos = recursos.terminos_por_tipo("ADVERSATIVO", con_tilde=True)
        # Los largos primero: "sin embargo" debe ganar a "sin".
        adversativos.sort(key=len, reverse=True)
        alternancia = "|".join(re.escape(a) for a in adversativos)
        # Corta por puntuación fuerte o por conector adversativo.
        self._patron = re.compile(
            r"(?:[.;!?]+|\s+(?:" + alternancia + r")\s+)", flags=re.IGNORECASE
        )
        # Un conector puede quedar al INICIO cuando el corte anterior lo hizo
        # la puntuación: "...clase. Sin embargo, los parciales...".
        self._patron_inicial = re.compile(
            r"^\s*(?:" + alternancia + r")\b[\s,]*", flags=re.IGNORECASE
        )

    def segmentar(self, texto: str) -> list[str]:
        """Devuelve las cláusulas útiles de una respuesta."""
        if not isinstance(texto, str) or not texto.strip():
            return []
        crudas: list[str] = []
        for fragmento in self._patron.split(texto):
            # Un concesivo ANTEPUESTO ("Aunque llega tarde, sus clases son
            # excelentes") no lo captura el patrón principal porque no tiene
            # espacio delante. La coma que cierra la subordinada es el corte.
            sin_conector = self._patron_inicial.sub("", fragmento)
            if sin_conector != fragmento and "," in sin_conector:
                izquierda, derecha = sin_conector.split(",", 1)
                crudas.extend([izquierda.strip(" ,;:"), derecha.strip(" ,;:")])
            else:
                crudas.append(sin_conector.strip(" ,;:"))
        utiles = [c for c in crudas if len(c.split()) >= self.cfg.min_tokens_clausula]
        return utiles[: self.cfg.max_clausulas_doc]  # tope defensivo

    def segmentar_corpus(self, docs: Sequence[str]) -> pd.DataFrame:
        """Aplica la segmentación a todo el corpus."""
        filas = [
            {"doc_id": doc_id, "clausula_id": i, "clausula": clausula}
            for doc_id, texto in enumerate(docs)
            for i, clausula in enumerate(self.segmentar(texto))
        ]
        marco = pd.DataFrame(filas)
        contestadas = sum(1 for t in docs if str(t).strip())
        logger.info(
            "  segmentación: %d contestadas -> %d cláusulas (%.2f por respuesta)",
            contestadas,
            len(marco),
            len(marco) / max(contestadas, 1),
        )
        return marco


# ═══════════════════════════════════════════════════════════════════════════
# 10. RAMA B — SENTIMIENTO
# ═══════════════════════════════════════════════════════════════════════════


class AnalizadorSentimiento:
    """Clasifica polaridad con un modelo de transformers en español.

    Recibe el texto ORIGINAL, con tildes y con stopwords. No se le aplica
    nada de la rama de tópicos: quitarle "no" invertiría el resultado.
    """

    def __init__(self, cfg: Config, recursos: RecursosLexicos) -> None:
        self.cfg = cfg
        self.recursos = recursos
        self._pipeline: Any = None  # carga perezosa

    def _cargar(self) -> Any:
        """Carga el modelo la primera vez que se necesita."""
        if self._pipeline is not None:
            return self._pipeline
        from transformers import pipeline

        logger.info("  cargando '%s'...", self.cfg.modelo_sentimiento)
        self._pipeline = pipeline(  # type: ignore[call-overload]
            task="sentiment-analysis",
            model=self.cfg.modelo_sentimiento,
            truncation=True,
            max_length=128,  # las cláusulas son cortas; acelera mucho
        )
        return self._pipeline

    def clasificar(self, textos: Sequence[str]) -> pd.DataFrame:
        """Polaridad y confianza para cada texto."""
        clasificador = self._cargar()
        logger.info("  clasificando %d cláusulas...", len(textos))
        resultados: list[dict[str, Any]] = []
        # Por lotes: llamar uno a uno es órdenes de magnitud más lento.
        for inicio in range(0, len(textos), self.cfg.batch_size):
            for salida in clasificador(
                list(textos[inicio : inicio + self.cfg.batch_size])
            ):
                resultados.append(
                    {
                        "sentimiento": str(salida["label"]).upper(),
                        "confianza": round(float(salida["score"]), 4),
                    }
                )
        return pd.DataFrame(resultados)

    def marcar_negacion(self, textos: Sequence[str]) -> pd.DataFrame:
        """Señala qué textos contienen negadores, y de qué tipo.

        No corrige la predicción: sirve para auditarla. Una cláusula con
        negador explícito clasificada POS con alta confianza es candidata a
        revisión manual.
        """
        patrones = {
            tipo: re.compile(
                r"\b(?:"
                + "|".join(
                    re.escape(t)
                    for t in self.recursos.terminos_por_tipo(tipo, con_tilde=True)
                )
                + r")\b",
                flags=re.IGNORECASE,
            )
            for tipo in ("NEGADOR", "CUANTIFICADOR_BAJO", "MODAL_ATENUANTE")
        }
        return pd.DataFrame(
            {
                f"tiene_{tipo.lower()}": [bool(p.search(str(t))) for t in textos]
                for tipo, p in patrones.items()
            }
        )


# ═══════════════════════════════════════════════════════════════════════════
# 11. FUSIÓN — ABSA
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Resultado:
    """Todo lo que produce el pipeline para UNA encuesta."""

    encuesta: str
    modelador: ModeladorTopicos
    topicos: pd.DataFrame
    clausulas: pd.DataFrame
    absa: pd.DataFrame
    metricas: dict[str, Any]


def construir_absa(
    clausulas: pd.DataFrame, tabla_topicos: pd.DataFrame
) -> pd.DataFrame:
    """Agrega las cláusulas en la matriz aspecto x polaridad.

    Es el entregable de valor: por cada tópico (aspecto), cuánta gente habla
    de él y en qué proporción lo valora mal. El cruce volumen x negatividad
    convierte un modelo descriptivo en una herramienta de priorización.
    """
    asignadas = clausulas[clausulas["topico"] >= 0]
    if asignadas.empty:
        return pd.DataFrame()

    conteos = pd.crosstab(asignadas["topico"], asignadas["sentimiento"])
    for etiqueta in (POS, NEU, NEG):  # garantizar las tres columnas
        if etiqueta not in conteos.columns:
            conteos[etiqueta] = 0
    absa = conteos[[POS, NEU, NEG]].copy()
    absa["total_clausulas"] = absa.sum(axis=1)
    absa["pct_positivo"] = (100 * absa[POS] / absa["total_clausulas"]).round(1)
    absa["pct_neutro"] = (100 * absa[NEU] / absa["total_clausulas"]).round(1)
    absa["pct_negativo"] = (100 * absa[NEG] / absa["total_clausulas"]).round(1)
    # Polaridad neta en [-1, 1]: resume la valoración en un número.
    absa["polaridad_neta"] = ((absa[POS] - absa[NEG]) / absa["total_clausulas"]).round(
        3
    )
    absa["terminos"] = absa.index.map(tabla_topicos.set_index("Topic")["Terminos"])
    # Prioridad = volumen x negatividad. Ordena la agenda de intervención.
    absa["prioridad"] = (absa["total_clausulas"] * absa["pct_negativo"] / 100).round(1)
    return absa.sort_values("prioridad", ascending=False)


def ejemplos_por_topico(
    clausulas: pd.DataFrame, topico: int, sentimiento: str, n: int = 3
) -> list[str]:
    """Cláusulas literales de un tópico y una polaridad.

    Es la trazabilidad: cada celda de la matriz ABSA puede respaldarse con
    las frases reales que la produjeron.
    """
    seleccion = clausulas[
        (clausulas["topico"] == topico) & (clausulas["sentimiento"] == sentimiento)
    ]
    return (
        seleccion.sort_values("confianza", ascending=False)["clausula"].head(n).tolist()
    )


# ═══════════════════════════════════════════════════════════════════════════
# 12. ORQUESTADOR
# ═══════════════════════════════════════════════════════════════════════════


def procesar_encuesta(
    encuesta: Encuesta, cfg: Config, recursos: RecursosLexicos
) -> Resultado:
    """Ejecuta ambas ramas sobre una encuesta y las fusiona en ABSA."""
    logger.info("=" * 74)
    logger.info("ENCUESTA: %s (%d filas)", encuesta.nombre, len(encuesta.texto_clean))
    logger.info("=" * 74)

    # ── RAMA A: TÓPICOS (texto _clean, sin tildes) ─────────────────────────
    logger.info("[A] Modelado de tópicos")
    modelador = ModeladorTopicos(cfg, recursos)
    modelador.entrenar(encuesta.texto_clean)
    tabla = modelador.tabla_topicos()

    # ── RAMA B: SENTIMIENTO (texto original, con tildes) ───────────────────
    logger.info("[B] Segmentación y sentimiento")
    segmentador = Segmentador(cfg, recursos)
    clausulas = segmentador.segmentar_corpus(encuesta.texto_original)
    if clausulas.empty:
        raise ValueError(f"'{encuesta.nombre}': no se obtuvo ninguna cláusula.")

    analizador = AnalizadorSentimiento(cfg, recursos)
    clausulas = pd.concat(
        [
            clausulas.reset_index(drop=True),
            analizador.clasificar(clausulas["clausula"].tolist()),
            analizador.marcar_negacion(clausulas["clausula"].tolist()),
        ],
        axis=1,
    )

    # ── FUSIÓN ─────────────────────────────────────────────────────────────
    logger.info("[C] Fusión ABSA: cláusula -> tópico -> polaridad")
    # Las cláusulas se normalizan al espacio de la rama de tópicos SÓLO para
    # poder proyectarlas. El sentimiento ya se calculó antes, sobre el texto
    # original: las dos ramas no se contaminan.
    clausulas_clean = [_normalizar_minimo(c, recursos) for c in clausulas["clausula"]]
    emb = modelador.codificar(clausulas_clean, "(cláusulas)")
    clausulas["topico"] = modelador.asignar(clausulas_clean, emb)
    absa = construir_absa(clausulas, tabla)

    distribucion = clausulas["sentimiento"].value_counts(normalize=True)
    metricas = {
        "encuesta": encuesta.nombre,
        **modelador.metricas(),
        **{k: v for k, v in modelador.higiene.items() if k != "ejemplos_cortas"},
        "n_clausulas": len(clausulas),
        "pct_clausulas_asignadas": round(
            100 * float((clausulas["topico"] >= 0).mean()), 2
        ),
        "pct_positivo": round(100 * float(distribucion.get(POS, 0)), 2),
        "pct_neutro": round(100 * float(distribucion.get(NEU, 0)), 2),
        "pct_negativo": round(100 * float(distribucion.get(NEG, 0)), 2),
        "confianza_media": round(float(clausulas["confianza"].mean()), 4),
    }
    logger.info(
        "[OK] %s: %d tópicos | %.1f%% outliers | %d cláusulas | %.1f%% neg",
        encuesta.nombre,
        metricas["n_topicos"],
        metricas["pct_outliers"],
        metricas["n_clausulas"],
        metricas["pct_negativo"],
    )
    return Resultado(encuesta.nombre, modelador, tabla, clausulas, absa, metricas)


def ejecutar_pipeline(
    cfg: Config | None = None,
    encuestas: dict[str, str] | None = None,
    raiz: Path | None = None,
    configs_por_encuesta: dict[str, Config] | None = None,
) -> dict[str, Resultado]:
    """Corre el MVP completo sobre las encuestas del corpus.

    Cada encuesta se modela de forma INDEPENDIENTE: nunca se concatenan los
    textos de varias hojas en un solo corpus. `procesar_encuesta` recibe una
    sola encuesta y le entrena su propio `ModeladorTopicos`.

    Args:
        cfg: configuración ÚNICA aplicada a todas las encuestas. Se usa
            cuando `configs_por_encuesta` es None.
        encuestas: mapa hoja -> columna de texto; se usa `ENCUESTAS` si es
            None.
        raiz: raíz del proyecto; se autodetecta si es None.
        configs_por_encuesta: mapa nombre de encuesta -> `Config` propia,
            típicamente la salida de `buscar_config_por_encuesta`. Cuando se
            pasa, tiene prioridad sobre `cfg` y cada encuesta se entrena con
            SUS PROPIOS hiperparámetros en vez de compartir una fracción.

    Returns:
        Un `Resultado` por encuesta, indexado por nombre de hoja.
    """
    cfg = cfg or Config()
    raiz = raiz or encontrar_raiz()
    dir_data = raiz / "data"

    recursos = RecursosLexicos.cargar(dir_data / "utilities")
    ruta_corpus = buscar_archivo(dir_data, "corpus")
    logger.info("Corpus: %s", ruta_corpus.name)
    cargadas = cargar_encuestas(ruta_corpus, recursos, encuestas)

    resultados: dict[str, Resultado] = {}
    for nombre, encuesta in cargadas.items():
        if not encuesta.modelable:
            logger.warning(
                "Se omite '%s': sólo %d respuestas modelables (mínimo %d). "
                "Analícela cualitativamente.",
                nombre,
                encuesta.n_modelables,
                MINIMO_MODELABLES,
            )
            continue
        # Config propia si se proveyó; si no, la única compartida.
        cfg_encuesta = (configs_por_encuesta or {}).get(nombre, cfg)
        try:
            resultados[nombre] = procesar_encuesta(encuesta, cfg_encuesta, recursos)
        except (ValueError, KeyError, RuntimeError) as exc:
            logger.error("'%s' falló: %s: %s", nombre, type(exc).__name__, exc)

    if resultados:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        manifiesto = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "corpus": ruta_corpus.name,
            "configuracion_por_defecto": cfg.to_dict(),
            "configuraciones_por_encuesta": (
                {n: c.to_dict() for n, c in configs_por_encuesta.items()}
                if configs_por_encuesta
                else None
            ),
            "n_stopwords": len(recursos.stopwords),
            "n_negaciones": len(recursos.negaciones),
            "metricas_por_encuesta": [r.metricas for r in resultados.values()],
        }
        with open(Path(cfg.output_dir) / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifiesto, f, ensure_ascii=False, indent=2, default=str)
        logger.info("Manifiesto en %s", Path(cfg.output_dir) / "manifest.json")
    return resultados


def resumen_comparativo(resultados: dict[str, Resultado]) -> pd.DataFrame:
    """Una fila por encuesta, para comparar de un vistazo."""
    columnas = [
        "encuesta",
        "filas",
        "tasa_respuesta",
        "contestadas",
        "modelables",
        "unicos",
        "min_topic_size",
        "min_samples",
        "n_topicos",
        "pct_outliers",
        "tamano_mediano",
        "diversidad",
        "pares_redundantes",
        "n_clausulas",
        "pct_positivo",
        "pct_neutro",
        "pct_negativo",
    ]
    marco = pd.DataFrame([r.metricas for r in resultados.values()])
    return marco[[c for c in columnas if c in marco.columns]].set_index("encuesta")


if __name__ == "__main__":
    SALIDA = ejecutar_pipeline()
    print("\n" + "=" * 74)
    print("RESUMEN COMPARATIVO")
    print("=" * 74)
    print(resumen_comparativo(SALIDA).to_string())
