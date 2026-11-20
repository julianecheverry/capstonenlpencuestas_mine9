"""
Mod_Bertopic_Ft.py — Tópicos + Sentimiento + ABSA para preguntas abiertas de encuestas
======================================================================================

Proyecto: Capstone NLP — Encuestas Universitarias (MINE009) · Proyecto Prisma
Versión: ft-r4.15 (ft-r4.14 + aspectos SUPERVISADOS «U4» como alternativa a UMAP/HDBSCAN: ``Config.modo_aspectos``,
         sección «ASPECTOS SUPERVISADOS · U4»; con ``modo_aspectos="bertopic"`` se comporta exactamente como ft-r4.14)
         ft-r4.14 (ft-r4.12 + opciones de Config: politica_fusion, ari_semillas, n_finalistas_estabilidad
         + reglas de reparación de fragmentos R1–R5 (ft-r4.13) y R7–R15 (ft-r4.14, comentarios de Julián 11-nov-2026):
         reglas_fragmentos, r2_max_contenido, r3_unir_protasis, r4_estricto, r4b_cierres, r12_max_palabras,
         r12_guarda_adversativa, puntuacion_original)

FLUJO
-----
1. Ingesta: un CSV por encuesta::pregunta, catalogados en ``preguntas_abiertas.xlsx``
   (hojas "preguntas" y "llaves"). Se usa solo la columna de texto original (con tildes y
   puntuación) y se conservan las llaves compuestas (periodo, ID de respuesta, etc.).
2. Alertas: ya NO son del NLP (Etapa 1 v1.10). Solo entran las respuestas con ``es_no_respuesta = False`` y
   ``tiene_alerta = False`` (``cargar_encuestas_etapa1``); las alertas las publica la Etapa 1.
3. Segmentación en cláusulas con el analizador de dependencias de spaCy
   (``es_core_news_lg``, ``SegmentadorDependencias``) y clasificación de cada cláusula
   en ASPECTO, VALORACION_GLOBAL, NO_OPINION o RUIDO.
4. Rama de tópicos: texto de aspecto neutralizado (``neutralizacion.xlsx``), embeddings
   (mpnet), UMAP + HDBSCAN elegidos por búsqueda con puertas duras y estabilidad, fusión
   iterativa de tópicos redundantes, reasignación de outliers y etiquetas c-TF-IDF.
5. Rama de sentimiento: RoBERTuito sobre la cláusula original; calibrador opcional
   aprendido de anotaciones humanas.
6. ABSA: matriz aspecto × sentimiento con prioridad = límite inferior de Wilson × log(volumen).
7. Scorecard de puertas G1–G11, exportación verificada y recalificación con auditoría humana.

MODO DE ASPECTOS (ft-r4.15)
---------------------------
``Config.modo_aspectos`` decide cómo se asigna el aspecto de cada cláusula con motivo ASPECTO:

* ``"bertopic"`` (por defecto en ``Config``): UMAP + HDBSCAN + c-TF-IDF, EXACTAMENTE como ft-r4.14
  (la huella de ``Config`` es idéntica a la de ft-r4.14).
* ``"u4"``: un modelo supervisado por pregunta (encoder mpnet con las últimas 4 capas descongeladas + cabeza lineal,
  entrenado SOLO con etiquetas auditadas a mano) decide el aspecto. Ver la sección «ASPECTOS SUPERVISADOS · U4»:
  ``entrenar_u4`` → ``cargar_u4`` → ``asignar_aspectos_u4`` (lo llama ``procesar_encuesta``). Reglas: en respuestas cortas (<= ``max_palabras_respuesta_corta`` palabras) una probabilidad máxima menor que
  ``umbral_comentario_general`` da «Comentario General»; las cláusulas largas de baja probabilidad forman el
  «residual», que SÍ pasa por UMAP/HDBSCAN para descubrir aspectos nuevos y queda como «Por descubrir».

GLOSARIO: QUÉ ES UNA CLÁUSULA
-----------------------------
Una respuesta abierta suele valorar varias cosas a la vez:

    "Explica muy bien los temas, pero nunca responde los correos."

Esa respuesta tiene DOS unidades de opinión: una positiva sobre la explicación y una
negativa sobre la comunicación. Cada unidad es una CLÁUSULA, la unidad de análisis de todo
el pipeline: se le asigna un aspecto (tópico) y una polaridad.

DECISIONES DE DISEÑO
--------------------
H1  Todas las salidas se escriben desde un solo DataFrame, en una carpeta por corrida
    identificada por la huella de configuración, léxicos y corpus; la exportación relee y
    verifica.
H2  Los tópicos se modelan directamente sobre cláusulas; la matriz ABSA usa exactamente las
    mismas asignaciones.
H3  Texto de aspecto: la cláusula sin valorativos, persona, negadores ni nombres propios.
    Lo que queda sin contenido ("excelente profesor") es VALORACIÓN GLOBAL (-3).
H4  Filtro de NO OPINIÓN antes de modelar; tampoco entra al sentimiento.
H5  Stopwords NLTK + stopwords.xlsx (globales + particulares de la encuesta).
H6  Búsqueda con puertas duras, frente de Pareto y estabilidad (ARI entre semillas). Si
    nada cumple, ``mejor_config`` falla de forma explícita.
H7  Una única función de reasignación de outliers en la búsqueda y en el modelo final.
H8  Probabilidades completas, polaridad continua, ajuste de NEU por margen y validación
    contra la nota y contra etiquetas humanas.
H9  Prioridad = límite inferior de Wilson de %NEG × log(volumen), con n mínimo.
H10 Protección de abreviaturas, decimales e iniciales al segmentar.
H13 F1 reponderado a la distribución poblacional.
H14 ``autor_texto``: el texto escrito por el docente no se valida contra la nota de los
    estudiantes (NO_APLICA).
H15 ``Calibrador``: regresión logística sobre las probabilidades del modelo y el tipo de
    pregunta, medida con validación cruzada. H27: modelo propio por tipo de pregunta solo si
    no rinde peor que el compartido.
H16 Con menos de MINIMO_MODELABLES cláusulas con aspecto: solo sentimiento; las puertas de
    aspectos quedan NO_APLICA.
H17 ``confiable``: un tópico de cohesión baja y con ≥ 2× palabras sueltas del corpus baja al
    final de la priorización.
H20 Columna ``asignacion`` (directa / reasignada / sin_topico) y ``cobertura_directa``.
H24 La fusión de tópicos redundantes se repite hasta que ningún par supere
    ``umbral_fusion``.
H25 ``HUELLA_ARCHIVO`` permite verificar que el módulo en memoria es el del disco.
H29 Reglas de reparación de fragmentos (ft-r4.13 y ft-r4.14), probadas en ``Prisma_reglas_fragmentos.ipynb`` contra
    la validación humana: se aplican en ``SegmentadorDependencias`` sobre los fragmentos de spaCy, en
    este orden y activables con ``Config.reglas_fragmentos`` (() = comportamiento de ft-r4.12):
    R1 huérfano (``, ; :`` inicial se une al anterior; ``, pero``/``, sin embargo``/``, aunque`` NO se
    unen y NO se limpian), R2 subordinada adherida (solo si tiene < ``r2_max_contenido`` palabras de
    contenido), R3 prótasis + apódosis (``r3_unir_protasis``), R4 adjunto sin verbo finito, R4B fragmento
    colgante («…incluso | si…»), R5 incisos/paréntesis. ft-r4.14 añade, por los comentarios de Julián
    (11-nov-2026): R11 una unidad por numeral, R7 abreviatura final («Ext. | Cali»), R9 minúscula tras punto
    suelto, R13 coma + minúscula, R10 anáfora inicial (solo tras . ? !), R8 «pero» tras punto, R14 «tal como»,
    R15 minúscula sin puntuación previa y R12 unidad única por longitud (≤ ``r12_max_palabras``); R4B amplía sus
    cierres («teniendo en cuenta»…). ``puntuacion_original`` devuelve cada unidad como el tramo del texto
    ORIGINAL (sin perder ni mover signos). Quedaron FUERA R0 (normalizar puntuación) y R6 (contenido mínimo).
    Por defecto TODAS las reglas están activas (decisión del equipo); ``reglas_fragmentos=()`` = ft-r4.12.
    Para reproducir ft-r4.13: ``Config(reglas_fragmentos=REGLAS_FRAGMENTOS_FT413, r4b_cierres=R4B_CIERRES_FT413,
    puntuacion_original=False)``.

PRINCIPIO DE SEPARACIÓN DE RAMAS
--------------------------------
  RAMA TÓPICOS (aspecto)             RAMA SENTIMIENTO
  -------------------------------    -------------------------------
  cláusula _clean, SIN tildes        cláusula ORIGINAL, CON tildes
  sin valorativos ni negadores       con TODO: "no" invierte polaridad
  stopwords: sólo en c-TF-IDF        stopwords.xlsx: NUNCA
  accents.xlsx: SÍ                   accents.xlsx: NUNCA
  negaciones.xlsx: para NEUTRALIZAR  negaciones.xlsx: cortes y auditoría

USO
---
    from nlp_utilidades.modulos_nlp import ejecutar_pipeline
    resultados = ejecutar_pipeline()
    resultados["evadoc_evaluacion :: obs19_aspectos_resaltar"].absa
"""

# pylint: disable=too-many-lines  # módulo cohesivo: un pipeline, un contrato

from __future__ import annotations

# ── Librería estándar ──────────────────────────────────────────────────────
import copy
import hashlib
import json
import math
import logging
import os
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

# ── Terceros (ligeros; los pesados se importan dentro de cada función) ─────
import numpy as np
import pandas as pd

# ── Logger ─────────────────────────────────────────────────────────────────
logger = logging.getLogger("Mod_Bertopic_Ft")
if not logger.handlers:  # evita mensajes duplicados al reimportar
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
        )
    )
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

VERSION: str = "ft-r4.15"
try:
    HUELLA_ARCHIVO: str = hashlib.md5(Path(__file__).read_bytes()).hexdigest()
except (NameError, OSError):  # ejecución sin archivo (p. ej. exec de un string)
    HUELLA_ARCHIVO = "desconocida"

# ═══════════════════════════════════════════════════════════════════════════
# 1. CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════

OUTLIER: int = -1  # opinión con aspecto, pero ningún tópico/aspecto la acoge
EXCLUIDO: int = -2  # no es opinión: vacía, ruido o "sin comentarios"
VALORACION_GLOBAL: int = -3  # opinión sin aspecto: "excelente profesor"
# El código -4 (ALERTA) se retiró: las alertas pasaron a la Etapa 1 (v1.10) y no llegan al NLP. No se reutiliza.
POS, NEU, NEG = "POS", "NEU", "NEG"
MODO_BERTOPIC, MODO_NINGUNO = "bertopic", "ninguno"
#: ft-r4.15: aspectos asignados por el clasificador supervisado U4 (ver sección «ASPECTOS SUPERVISADOS · U4»).
MODO_U4 = "u4"
#: ft-r4.15: códigos reservados (negativos: nunca cuentan como aspecto de la lista en el ABSA ni en «asignadas»).
COMENTARIO_GENERAL: int = -5  # respuesta corta con probabilidad U4 baja
POR_DESCUBRIR: int = -6  # residual de U4 (cláusula larga con probabilidad baja), modelado con UMAP/HDBSCAN
#: Plan v2, punto 7 (etapa A): VALORACIÓN GLOBAL (-3), COMENTARIO GENERAL (-5) y la clase «Opinión general» de U4 son UNA sola
#: categoría para el usuario. Los códigos internos -3/-5 se conservan (scorecard, ABSA); lo que las distingue es la columna
#: columna ``motivo_opinion_general`` de ``clausulas`` (valores en ``MOTIVOS_OPINION_GENERAL``); ``salidas_contrato`` la publica
#: también en ``clausulas.csv`` del contrato.
NOMBRE_OPINION_GENERAL = "Opinión general"
NOMBRE_COMENTARIO_GENERAL = NOMBRE_OPINION_GENERAL  # alias de compatibilidad: el nombre ya no se distingue
MOTIVO_OG_SIN_CONTENIDO = "sin_contenido"        # antes VALORACIÓN GLOBAL: la cláusula solo valora, no dice de qué habla
MOTIVO_OG_COMENTARIO_CORTO = "comentario_corto"  # antes COMENTARIO GENERAL: respuesta corta con probabilidad baja de U4
MOTIVO_OG_CLASIFICADOR = "clasificador"          # U4 predijo «Opinión general» (o «Sin aspecto» / «Ninguno de la lista»)
MOTIVOS_OPINION_GENERAL = (MOTIVO_OG_SIN_CONTENIDO, MOTIVO_OG_COMENTARIO_CORTO, MOTIVO_OG_CLASIFICADOR)
NOMBRE_POR_DESCUBRIR = "Por descubrir"

#: Motivo de cada cláusula; explica cualquier código negativo de tópico.
MOTIVO_ASPECTO = "ASPECTO"
MOTIVO_GLOBAL = "VALORACION_GLOBAL"
MOTIVO_NO_OPINION = "NO_OPINION"
MOTIVO_RUIDO = "RUIDO"

#: Estado de cada fila (documento) del corpus.
DOC_VACIA = "VACIA"
DOC_RUIDO = "RUIDO"
DOC_NO_OPINION = "NO_OPINION"
DOC_CON_OPINION = "CON_OPINION"

ETIQUETAS_ESPECIALES: dict[int, str] = {
    OUTLIER: "SIN TÓPICO (outlier)",
    EXCLUIDO: "NO OPINIÓN",
    VALORACION_GLOBAL: NOMBRE_OPINION_GENERAL,
    COMENTARIO_GENERAL: NOMBRE_OPINION_GENERAL,
    POR_DESCUBRIR: NOMBRE_POR_DESCUBRIR,
}


# ═══════════════════════════════════════════════════════════════════════════
# 2. LOCALIZACIÓN DE ARCHIVOS
# ═══════════════════════════════════════════════════════════════════════════


def encontrar_raiz(marcador: str = "pyproject.toml") -> Path:
    """Sube por el árbol de directorios hasta hallar la raíz del proyecto.

    Args:
        marcador: Archivo que identifica la raíz.

    Returns:
        Ruta de la raíz.

    Raises:
        FileNotFoundError: Si ningún ancestro contiene el marcador.
    """
    actual = Path.cwd().resolve()
    for candidato in (actual, *actual.parents):
        if (candidato / marcador).exists():
            return candidato
    raise FileNotFoundError(f"No se encontró '{marcador}' sobre {actual}")


def buscar_archivo(
    directorio: Path,
    *fragmentos: str,
    extensiones: tuple[str, ...] = (".xlsx", ".xls"),
) -> Path:
    """Localiza un archivo por fragmentos de su nombre.

    Usa ``os.listdir`` a propósito: con OneDrive en modo "solo en la nube"
    comprobar una ruta fija produce ``FileNotFoundError`` silenciosos.

    Args:
        directorio: Carpeta donde buscar.
        *fragmentos: Subcadenas que deben aparecer en el nombre.
        extensiones: Extensiones admitidas.

    Returns:
        Primera coincidencia en orden alfabético.

    Raises:
        FileNotFoundError: Si no existe la carpeta o no hay coincidencias.
    """
    if not directorio.exists():
        raise FileNotFoundError(f"No existe el directorio: {directorio}")
    candidatos = [
        directorio / nombre
        for nombre in sorted(os.listdir(directorio))
        if nombre.lower().endswith(extensiones)
        and not nombre.startswith("~$")  # bloqueo temporal de Excel
        and all(f.lower() in nombre.lower() for f in fragmentos)
    ]
    if not candidatos:
        raise FileNotFoundError(
            f"Sin coincidencias para {fragmentos} en {directorio}. "
            f"Disponibles: {sorted(os.listdir(directorio))}"
        )
    if len(candidatos) > 1:
        logger.warning(
            "%d coincidencias para %s; se usa '%s'",
            len(candidatos),
            fragmentos,
            candidatos[0].name,
        )
    return candidatos[0]


def slug(texto: str) -> str:
    """Identificador ASCII en minúsculas y sin espacios ni símbolos.

    Args:
        texto: Texto libre (p. ej. nombre de hoja).

    Returns:
        Slug estable, apto para rutas y para las claves de stopwords.xlsx.
    """
    plano = "".join(
        c
        for c in unicodedata.normalize("NFD", str(texto))
        if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"[^a-z0-9]+", "_", plano.lower()).strip("_")


def huella_texto(*partes: Any) -> str:
    """SHA-256 de un conjunto de objetos serializados a JSON.

    Args:
        *partes: Objetos serializables (``default=str`` para el resto).

    Returns:
        Hash hexadecimal completo.
    """
    firma = hashlib.sha256()
    for parte in partes:
        firma.update(
            json.dumps(parte, sort_keys=True, ensure_ascii=False, default=str).encode(
                "utf-8"
            )
        )
    return firma.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# 3. LÉXICOS DE NEUTRALIZACIÓN Y NO OPINIÓN
# ═══════════════════════════════════════════════════════════════════════════
# Valores por defecto versionados con el módulo. Pueden ampliarse SIN tocar
# código con un archivo opcional `neutralizacion.xlsx` en data/utilities,
# columnas TERMINO | CATEGORIA (VALORATIVO, PERSONA, GENERICO).
# Todos van en minúsculas y SIN tildes: se aplican sobre texto _clean.


def _palabras(bloque: str) -> list[str]:
    """Palabras de un bloque de texto (léxicos legibles como párrafo)."""
    return bloque.split()


#: Respuestas completas que no son opinión (comparación exacta, texto clean).
RUIDO: frozenset[str] = frozenset(
    {
        "nr",
        "na",
        "ns",
        "nc",
        "n a",
        "no aplica",
        "no responde",
        "x",
        "xx",
        "xxx",
        "0",
        "no",
        "si",
        "ok",
        "nada",
        "ninguno",
        "ninguna",
        "ningun",
        "no se",
        "ns nr",
        "nn",
        "n",
        ".",
        "-",
    }
)

#: Patrones (full match sobre texto clean sin puntuación) de NO OPINIÓN:
#: la persona escribió algo, pero no evaluó nada.
PATRONES_NO_OPINION: tuple[str, ...] = (
    r"(?:sin|no hay|no tengo|no tiene|no tendria|ningun[oa]?|nada|ninguno)"
    r"(?: (?:otro|otra|otros|otras|mas|mayor|mayores|particular|particulares"
    r"|especial|especiales|adicional|adicionales|en especial))*"
    r"(?: (?:comentario|comentarios|observacion|observaciones|aspecto"
    r"|aspectos|sugerencia|sugerencias|recomendacion|recomendaciones"
    r"|opinion|novedad|novedades|queja|quejas|reclamo|reclamos))?"
    r"(?: (?:mas|adicional|adicionales|particular|particulares|especial"
    r"|especiales|relevante|relevantes|importante|importantes))*"
    r"(?: (?:que|para|por) (?:agregar|decir|destacar|resaltar|comentar"
    r"|anadir|observar|senalar|sugerir|recomendar|objetar))?"
    r"(?: (?:en particular|al respecto|por ahora|por el momento"
    r"|hasta el momento|de momento|por mi parte))?",
    # "no tengo clase con este docente", "aun no he visto clase con el"
    r"(?:aun |todavia )?no (?:tengo|he tenido|e tenido|eh tenido|tuve|he visto"
    r"|e visto|eh visto|vi|recibi|he recibido) (?:clase|clases)(?: .*)?",
    r"(?:aun |todavia )?no (?:lo|la|le|los|las) (?:conozco|conoci|he visto"
    r"|e visto|vi|he tenido)(?: .*)?",
    r".*no (?:entiendo|se) (?:por que|porque) me (?:habilitan|habilitaron"
    r"|aparece|aparecio|salio|sale) .*",
    r"(?:aun |todavia )?no (?:he|e|eh) (?:visto|tenido) (?:clase |clases )?"
    r"con (?:el|ella|usted|el docente|la docente|este docente|esta docente)"
    r"(?: .*)?",
    r"no (?:me )?(?:dio|dicto|ha dado|ha dictado) (?:clase|clases)(?: .*)?",
    r"(?:no )?aplica(?: .*)?",
)


def _normalizar_acentos(texto: str, mapa: dict[str, str]) -> str:
    """Quita tildes con el mapa del proyecto o, si no hay, con unicodedata.

    Args:
        texto: Texto de entrada.
        mapa: Pares {acentuado: plano} de accents.xlsx.

    Returns:
        Texto sin tildes (la ñ se conserva si el mapa no la toca).
    """
    if mapa:
        for origen, destino in mapa.items():
            texto = texto.replace(origen, destino)
        return texto
    salida: list[str] = []
    for caracter in texto:
        if caracter in "ñÑ":  # la eñe es letra, no letra con tilde
            salida.append(caracter)
            continue
        salida.extend(
            c
            for c in unicodedata.normalize("NFD", caracter)
            if unicodedata.category(c) != "Mn"
        )
    return "".join(salida)


#: Hoja ``particulares`` de stopwords.xlsx: clave del Excel -> código de encuesta del
#: catálogo (sin guion bajo, como lo compara ``stopwords_vectorizador``). Calidad tiene
#: una lista por población; como es una sola encuesta, se unen. Acreditación no tiene
#: particulares (igual que en la Etapa 3: ``src/limpieza/listas.py``).
CLAVES_PARTICULARES_NLP: dict[str, str] = {
    "stopwords_evaluaciondocente": "evadocevaluacion",
    "stopwords_autoevaluaciondocente": "evadocautoevaluacion",
    "stopwords_calidadadministrativos": "calidadservicios",
    "stopwords_calidaddirectivos": "calidadservicios",
    "stopwords_calidadestudiantes": "calidadservicios",
    "stopwords_calidadegresados": "calidadservicios",
    "stopwords_calidaddocentes": "calidadservicios",
}

#: Adverbios con ``valora = no`` en stopwords.xlsx que el NLP NO quita de las etiquetas
#: (decisión de ``stopwords_1.xlsx``, hoja ``_adverbios_excluidos``: informan tiempo o modo).
ADVERBIOS_NO_STOPWORD_NLP: tuple[str, ...] = (
    "claramente", "deprisa", "despacio", "facilmente", "lentamente",
    "manana", "pronto", "rapidamente", "tarde", "temprano",
)

#: Palabras que solo el c-TF-IDF del NLP quita de las etiquetas de tópicos. NO van a la hoja
#: ``globales`` de stopwords.xlsx porque la Etapa 3 debe conservarlas (valoran; contrato 6.2).
GLOBALES_SOLO_NLP: tuple[str, ...] = ("bien", "mal", "mejor", "nunca", "peor", "siempre")


#: Claves de stopwords sin particulares ya avisadas (un aviso por clave).
_CLAVES_STOPWORDS_AVISADAS: set[str] = set()


@dataclass
class RecursosLexicos:
    """Léxicos cargados de disco, separados por rama.

    Attributes:
        stopwords_globales: Hoja ``globales`` de stopwords.xlsx.
        stopwords_por_encuesta: Hoja ``particulares`` agrupada por clave.
        stopwords_nltk: Lista española de NLTK (sin tildes).
        mapa_acentos: Pares de accents.xlsx.
        negaciones: Tabla de negaciones.xlsx.
        valorativos: Léxico de valoración pura (neutralización).
        persona: Léxico de persona y tratamientos (neutralización).
        genericos: Contenido no aspectual (neutralización).
    """

    stopwords_globales: frozenset[str] = field(default_factory=frozenset)
    stopwords_por_encuesta: dict[str, frozenset[str]] = field(default_factory=dict)
    stopwords_nltk: frozenset[str] = field(default_factory=frozenset)
    mapa_acentos: dict[str, str] = field(default_factory=dict)
    negaciones: pd.DataFrame = field(default_factory=pd.DataFrame)
    valorativos: frozenset[str] = field(default_factory=frozenset)
    persona: frozenset[str] = field(default_factory=frozenset)
    genericos: frozenset[str] = field(default_factory=frozenset)

    _COLUMNAS_NEGACION = frozenset({"TERMINO", "TERMINO_SIN_TILDE", "TIPO"})

    # ── Carga ──────────────────────────────────────────────────────────────

    @classmethod
    def cargar(cls, dir_utilities: Path) -> RecursosLexicos:
        """Lee stopwords.xlsx, accents.xlsx, negaciones.xlsx y NLTK.

        Args:
            dir_utilities: Carpeta ``data/utilities``.

        Returns:
            Recursos listos para ambas ramas.
        """
        logger.info("Cargando recursos léxicos desde %s", dir_utilities)
        mapa = cls._leer_acentos(dir_utilities)
        globales, particulares = cls._leer_stopwords(dir_utilities, mapa)
        negaciones = cls._leer_negaciones(dir_utilities)
        recursos = cls(
            stopwords_globales=globales,
            stopwords_por_encuesta=particulares,
            stopwords_nltk=cls._leer_nltk(mapa),
            mapa_acentos=mapa,
            negaciones=negaciones,
        )
        recursos._leer_neutralizacion(dir_utilities)
        logger.info(
            "  stopwords: %d NLTK + %d globales + %d claves de encuesta | "
            "acentos: %d | negaciones: %d | neutralización: %d términos",
            len(recursos.stopwords_nltk),
            len(globales),
            len(particulares),
            len(mapa),
            len(negaciones),
            len(recursos.terminos_neutralizados()),
        )
        return recursos

    @staticmethod
    def _leer_acentos(directorio: Path) -> dict[str, str]:
        """Mapa {acentuado: plano}; vacío si falta el archivo."""
        try:
            ruta = buscar_archivo(directorio, "accent")
        except FileNotFoundError:
            logger.info("accents.xlsx no se usa (decisión 16-nov): se normaliza con unicodedata")
            return {}
        hoja = pd.read_excel(ruta)
        if hoja.shape[1] < 2:
            logger.warning("accents.xlsx no tiene dos columnas; se ignora")
            return {}
        return {
            str(a): str(b)
            for a, b in zip(hoja.iloc[:, 0], hoja.iloc[:, 1], strict=False)
            if pd.notna(a) and pd.notna(b)
        }

    @staticmethod
    def _leer_stopwords(
        directorio: Path, mapa: dict[str, str]
    ) -> tuple[frozenset[str], dict[str, frozenset[str]]]:
        """Lee ``stopwords.xlsx`` (el mismo archivo que usa la Etapa 3), sin cambiarlo.

        Lee las hojas por nombre, no todas a ciegas:

        * ``globales``: stopwords globales.
        * ``adverbios``: las filas con ``valora = no`` (relleno) pasan a globales,
          salvo ``ADVERBIOS_NO_STOPWORD_NLP``. Las filas con ``valora = si`` no.
        * ``particulares``: por encuesta. Las claves del Excel (``stopwords_evaluaciondocente``)
          se traducen con ``CLAVES_PARTICULARES_NLP`` al código de encuesta del catálogo
          (sin guion bajo: ``evadocevaluacion``). Se conserva además la clave antigua para
          que el cuaderno actual (clave tomada de la hoja) siga funcionando.
        * ``base`` (NLTK) y las hojas que empiezan con ``_`` se ignoran: NLTK se lee aparte.

        Se añade ``GLOBALES_SOLO_NLP`` (palabras que solo el c-TF-IDF quita; la Etapa 3
        las conserva porque valoran).
        """
        hojas = pd.read_excel(buscar_archivo(directorio, "stopword"), sheet_name=None)
        hojas = {slug(str(n)): m for n, m in hojas.items() if not str(n).startswith("_")}

        def limpiar(marco: pd.DataFrame, nombre: str) -> pd.DataFrame | None:
            marco = marco.copy()
            marco.columns = [str(c).strip().lower() for c in marco.columns]
            if "palabra" not in marco.columns:
                logger.warning("stopwords.xlsx hoja '%s' sin columna 'palabra'", nombre)
                return None
            return marco

        def terminos(palabras: Iterable) -> set[str]:
            return {
                _normalizar_acentos(str(v).strip().lower(), mapa)
                for v in palabras
                if pd.notna(v) and str(v).strip()
            }

        globales: set[str] = set()
        if (marco := limpiar(hojas.get("globales", pd.DataFrame()), "globales")) is not None:
            globales |= terminos(marco["palabra"])
        adverbios = limpiar(hojas.get("adverbios", pd.DataFrame()), "adverbios")
        if adverbios is not None and "valora" in adverbios.columns:
            relleno = adverbios[adverbios["valora"].astype(str).str.strip().str.lower() == "no"]
            globales |= terminos(relleno["palabra"])
        elif adverbios is not None:
            logger.warning("stopwords.xlsx hoja 'adverbios' sin columna 'valora': no se usa")
        globales -= terminos(ADVERBIOS_NO_STOPWORD_NLP)
        globales |= terminos(GLOBALES_SOLO_NLP)

        particulares: dict[str, set[str]] = {}
        marco = limpiar(hojas.get("particulares", pd.DataFrame()), "particulares")
        if marco is not None and "encuesta" in marco.columns:
            for clave, grupo in marco.groupby("encuesta"):
                antigua = slug(str(clave)).removeprefix("stopwords_").replace("_", "")
                palabras = terminos(grupo["palabra"])
                particulares.setdefault(antigua, set()).update(palabras)
                nueva = CLAVES_PARTICULARES_NLP.get(str(clave).strip())
                if nueva:
                    particulares.setdefault(nueva, set()).update(palabras)
                else:
                    logger.warning("stopwords.xlsx particulares: clave '%s' sin traducción a código de encuesta", clave)
        return frozenset(globales), {k: frozenset(v) for k, v in particulares.items()}

    @staticmethod
    def _leer_nltk(mapa: dict[str, str]) -> frozenset[str]:
        """Stopwords españolas de NLTK sin tildes; descarga si faltan."""
        import nltk  # pylint: disable=import-outside-toplevel

        try:
            palabras = nltk.corpus.stopwords.words("spanish")
        except LookupError:
            nltk.download("stopwords", quiet=True)
            palabras = nltk.corpus.stopwords.words("spanish")
        return frozenset(_normalizar_acentos(p.lower(), mapa) for p in palabras)

    @classmethod
    def _leer_negaciones(cls, directorio: Path) -> pd.DataFrame:
        """Lee negaciones.xlsx (hoja 'negaciones') y valida su esquema."""
        tabla = pd.read_excel(
            buscar_archivo(directorio, "negacion"), sheet_name="negaciones"
        )
        tabla.columns = [str(c).strip().upper() for c in tabla.columns]
        faltantes = cls._COLUMNAS_NEGACION - set(tabla.columns)
        if faltantes:
            raise KeyError(f"negaciones.xlsx: faltan columnas {sorted(faltantes)}")
        return tabla

    #: Categorías válidas de neutralizacion.xlsx -> atributo que llenan.
    _CATEGORIAS_NEUTRALIZACION = {
        "VALORATIVO": "valorativos",
        "PERSONA": "persona",
        "GENERICO": "genericos",
    }

    def _leer_neutralizacion(self, directorio: Path) -> None:
        """Carga neutralizacion.xlsx: ÚNICA fuente de los términos
        que se retiran (VALORATIVO, PERSONA) o no cuentan (GENERICO) en el
        texto de aspecto.

        Es OBLIGATORIO: sin él, el texto de aspecto conservaría "excelente",
        "profesora"... y los tópicos se armarían por tono o por género (H3)
        sin que nadie lo note. Mismo criterio que taxonomia.xlsx: fallar
        explícito antes que degradar en silencio.

        Raises:
            FileNotFoundError: Si no existe el archivo.
            KeyError: Si faltan las columnas TERMINO o CATEGORIA.
            ValueError: Si hay una CATEGORIA desconocida (p. ej. "DUDA" o un
                error de digitación) o una categoría queda vacía.
        """
        try:
            ruta = buscar_archivo(directorio, "neutraliza")
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Falta neutralizacion.xlsx en {directorio}. Desde "
                "ft-r4.10 es obligatorio (reemplaza las listas que antes estaban "
                "en el código). Cópielo junto a los demás léxicos."
            ) from error
        tabla = pd.read_excel(ruta, sheet_name=0)
        tabla.columns = [str(c).strip().upper() for c in tabla.columns]
        faltantes = {"TERMINO", "CATEGORIA"} - set(tabla.columns)
        if faltantes:
            raise KeyError(f"neutralizacion.xlsx: faltan columnas {sorted(faltantes)}")
        tabla = tabla.dropna(subset=["TERMINO"])
        tabla["CATEGORIA"] = tabla["CATEGORIA"].astype(str).str.strip().str.upper()
        desconocidas = set(tabla["CATEGORIA"]) - set(self._CATEGORIAS_NEUTRALIZACION)
        if desconocidas:
            raise ValueError(
                f"neutralizacion.xlsx: categorías no válidas {sorted(desconocidas)}. "
                f"Use solo {sorted(self._CATEGORIAS_NEUTRALIZACION)}."
            )
        for categoria, atributo in self._CATEGORIAS_NEUTRALIZACION.items():
            terminos = frozenset(
                self.sin_acentos(str(t).strip().lower())
                for t in tabla.loc[tabla["CATEGORIA"] == categoria, "TERMINO"]
                if str(t).strip()
            )
            if not terminos:
                raise ValueError(f"neutralizacion.xlsx: la categoría {categoria} está vacía")
            setattr(self, atributo, terminos)
        logger.info(
            "  neutralizacion.xlsx: %d valorativos | %d persona | %d genéricos (%s)",
            len(self.valorativos),
            len(self.persona),
            len(self.genericos),
            ruta.name,
        )


    # ── Consultas ──────────────────────────────────────────────────────────

    def sin_acentos(self, texto: str) -> str:
        """Normalización de acentos. SOLO para la rama TÓPICOS."""
        return _normalizar_acentos(texto, self.mapa_acentos)

    def terminos_por_tipo(self, *tipos: str, con_tilde: bool = True) -> list[str]:
        """Términos de negaciones.xlsx filtrados por TIPO.

        Args:
            *tipos: Valores de la columna TIPO.
            con_tilde: True para SENTIMIENTO, False para TÓPICOS.

        Returns:
            Términos únicos en minúsculas, ordenados.
        """
        if self.negaciones.empty:
            return []
        columna = "TERMINO" if con_tilde else "TERMINO_SIN_TILDE"
        sub = self.negaciones[self.negaciones["TIPO"].isin(tipos)]
        return sorted({str(t).strip().lower() for t in sub[columna].dropna()})

    def terminos_neutralizados(self) -> frozenset[str]:
        """Todo lo que se retira del texto de aspecto (palabras sueltas)."""
        modificadores = {
            palabra
            for termino in self.terminos_por_tipo(
                "NEGADOR",
                "ADVERSATIVO",
                "CUANTIFICADOR_BAJO",
                "INTENSIFICADOR",
                "MODAL_ATENUANTE",
                con_tilde=False,
            )
            for palabra in termino.split()
        }
        return frozenset(self.valorativos | self.persona | modificadores)

    def stopwords_vectorizador(self, clave_encuesta: str | None) -> list[str]:
        """Stopwords del c-TF-IDF: NLTK + globales + encuesta + neutralizados.

        Args:
            clave_encuesta: Slug de la hoja (p. ej. ``evaluaciondocente``).

        Returns:
            Lista ordenada de palabras sueltas (sin frases).
        """
        particulares: frozenset[str] = frozenset()
        if clave_encuesta:
            particulares = self.stopwords_por_encuesta.get(
                clave_encuesta.replace("_", ""), frozenset()
            )
            if not particulares and clave_encuesta not in _CLAVES_STOPWORDS_AVISADAS:
                _CLAVES_STOPWORDS_AVISADAS.add(clave_encuesta)
                logger.warning(
                    "stopwords.xlsx no tiene particulares para la clave "
                    "'%s' (claves disponibles: %s). Las etiquetas usarán solo NLTK + "
                    "globales. Agregue filas 'stopwords_%s' en la hoja 'particulares'.",
                    clave_encuesta,
                    sorted(self.stopwords_por_encuesta),
                    clave_encuesta,
                )
        todas = (
            self.stopwords_nltk
            | self.stopwords_globales
            | particulares
            | self.terminos_neutralizados()
            | self.genericos
        )
        return sorted(p for p in todas if p and " " not in p)

    def huella(self) -> str:
        """Hash de todos los léxicos: cambia si se edita cualquier Excel."""
        return huella_texto(
            sorted(self.stopwords_globales),
            {k: sorted(v) for k, v in sorted(self.stopwords_por_encuesta.items())},
            sorted(self.stopwords_nltk),
            self.mapa_acentos,
            self.negaciones.astype(str).to_dict(orient="list"),
            sorted(self.valorativos),
            sorted(self.persona),
            sorted(self.genericos),
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. CONFIGURACIÓN Y CRITERIOS
# ═══════════════════════════════════════════════════════════════════════════

#: Encoders multilingües (el `all-MiniLM-L6-v2` de BERTopic es inglés).
#: "potente" (mpnet, 768 dims) separa mejor aspectos en español; "rapido"
#: (MiniLM, 384 dims) es 3x más veloz. Se eligen en Config.encoder.
ENCODERS: dict[str, str] = {
    "rapido": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "potente": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}
ENCODER_POR_DEFECTO = ENCODERS["potente"]
#: Clasificador de polaridad en español (RoBERTuito, familia RoBERTa).
MODELO_SENTIMIENTO_POR_DEFECTO = "pysentimiento/robertuito-sentiment-analysis"
#: Columnas exigidas de taxonomia.xlsx (hoja "taxonomia").
_COLUMNAS_TAXONOMIA: frozenset[str] = frozenset({"ENCUESTA", "CODIGO", "DESCRIPCION"})
_COLUMNAS_TAXONOMIA_ASPECTOS: frozenset[str] = frozenset({"pregunta_codigo", "aspecto_id", "descripcion"})


def cargar_taxonomias(directorio: Path) -> dict[str, dict[str, str]]:
    """Taxonomía de aspectos por pregunta, desde ``taxonomia.xlsx``.

    Fuente de la ayuda de nombrado de tópicos (``cfg.taxonomia``, modo ``bertopic``).

    Formato vigente (hoja ``aspectos``, el que valida ``run_validar_taxonomia.py``): columnas
    ``pregunta_codigo``, ``aspecto_id``, ``aspecto_nombre``, ``descripcion``. La clave del
    resultado es ``pregunta_codigo`` (la clave lógica, p. ej. ``obs19_aspectos_resaltar``) y
    el valor ``{aspecto_id: descripcion}``.

    Formato antiguo (hoja ``taxonomia``: ``ENCUESTA``, ``CODIGO``, ``DESCRIPCION``): solo si el
    archivo no trae la hoja ``aspectos``; la clave es ``ENCUESTA``.

    A propósito NO rellena con ninguna taxonomía genérica si una pregunta no aparece: el
    llamador decide qué hacer.

    Args:
        directorio: Carpeta donde buscar taxonomia.xlsx (``data/utilities``).

    Returns:
        Mapa ``clave -> {codigo: descripcion}``. Vacío si el archivo no existe.

    Raises:
        KeyError: Si el archivo existe pero le faltan columnas requeridas.
    """
    try:
        ruta = buscar_archivo(directorio, "taxonomia")
    except FileNotFoundError:
        logger.warning(
            "No se encontró taxonomia.xlsx en %s -- ninguna encuesta "
            "tendrá ayuda de nombrado hasta que se agregue.",
            directorio,
        )
        return {}
    hojas = pd.ExcelFile(ruta).sheet_names
    if "aspectos" in hojas:
        tabla = pd.read_excel(ruta, sheet_name="aspectos", dtype=str, keep_default_na=False)
        tabla.columns = [str(c).strip().lower() for c in tabla.columns]
        faltantes = _COLUMNAS_TAXONOMIA_ASPECTOS - set(tabla.columns)
        if faltantes:
            raise KeyError(f"taxonomia.xlsx (hoja aspectos): faltan columnas {sorted(faltantes)}")
        return {
            str(pregunta).strip(): {
                str(codigo).strip(): str(desc).strip()
                for codigo, desc in zip(grupo["aspecto_id"], grupo["descripcion"])
                if str(codigo).strip()
            }
            for pregunta, grupo in tabla.groupby("pregunta_codigo")
        }
    tabla = pd.read_excel(ruta, sheet_name="taxonomia")
    tabla.columns = [str(c).strip().upper() for c in tabla.columns]
    faltantes = _COLUMNAS_TAXONOMIA - set(tabla.columns)
    if faltantes:
        raise KeyError(f"taxonomia.xlsx: faltan columnas {sorted(faltantes)}")
    salida: dict[str, dict[str, str]] = {}
    for encuesta_nombre, grupo in tabla.groupby("ENCUESTA"):
        salida[str(encuesta_nombre).strip()] = {
            str(codigo).strip(): str(desc).strip()
            for codigo, desc in zip(grupo["CODIGO"], grupo["DESCRIPCION"])
            if str(codigo).strip() and str(codigo).strip().lower() != "nan"
        }
    return salida


#: Campos de ``Config`` que solo entran en ``huella()`` si difieren del default.
_CAMPOS_OPCIONALES_HUELLA: tuple[str, ...] = (
    "politica_fusion",
    "fusion_umbral_raiz",
    "fusion_top_terminos",
    "fusion_raices_excluidas",
    "fusion_tope_pct",
    "ari_semillas",
    "n_finalistas_estabilidad",
)

#: Campos de ``Config`` de las reglas de fragmentos. Con ``reglas_fragmentos=()`` (= ft-r4.12) NO entran en
#: ``huella()``; con cualquier regla activa SÍ (el default las activa, así que la huella por defecto cambia).
_CAMPOS_REGLAS_FRAGMENTOS: tuple[str, ...] = (
    "reglas_fragmentos",
    "r2_max_contenido",
    "r3_unir_protasis",
    "r4_estricto",
    "r4b_cierres",
    "r12_max_palabras",
    "r12_guarda_adversativa",
    "puntuacion_original",
)

#: ft-r4.15: campos de ``Config`` de los aspectos supervisados U4. Con ``modo_aspectos="bertopic"`` NO entran en
#: ``huella()`` (la huella es idéntica a la de ft-r4.14).
_CAMPOS_U4: tuple[str, ...] = (
    "modo_aspectos",
    "ruta_modelo_u4",
    "semilla_u4",
    "max_palabras_respuesta_corta",
    "umbral_comentario_general",
    "sin_contenido_en_cortas",
)
#: Campos del residual (UMAP/HDBSCAN sobre las cláusulas largas de baja probabilidad). Con
#: ``residual_hdbscan_activo=False`` NO entran en ``huella()`` (idéntica a la versión sin residual).
_CAMPOS_RESIDUAL: tuple[str, ...] = (
    "umbral_residual_u4",
    "min_palabras_residual",
    "residual_hdbscan_activo",
    "residual_rejilla",
    "residual_minimo",
    "residual_criterios",
)


@dataclass
class Config:  # pylint: disable=too-many-instance-attributes
    """Parámetros de una corrida. Todo lo que cambia el resultado vive aquí.

    La huella (:meth:`huella`) identifica la corrida: misma huella, mismos
    léxicos y mismo corpus producen los mismos artefactos.
    """

    # --- Identidad y reproducibilidad ---
    nombre: str = "ft"
    random_state: int = 42

    # --- Encoders ---
    encoder: str = ENCODER_POR_DEFECTO
    modelo_sentimiento: str = MODELO_SENTIMIENTO_POR_DEFECTO
    batch_size: int = 64
    usar_cache: bool = True
    dir_cache: Path = Path("models/cache/embeddings_ft")

    # --- Segmentación en cláusulas ---
    min_tokens_clausula: int = 1  # "Excelente." es opinión válida
    max_clausulas_doc: int = 12
    #: Cláusulas con al menos estas palabras se cortan también por comas
    #: (en una versión previa el 24.8% de los outliers superaba 25 palabras). 0 = no cortar.
    cortar_clausulas_largas: int = 25
    min_palabras_fragmento: int = 6
    #: "dependencias" (producción) = SegmentadorDependencias, análisis sintáctico
    #: con spaCy es_core_news_lg. "conectores" = Segmentador por puntuación y
    #: conectores adversativos de negaciones.xlsx (alternativa, sin uso activo).
    metodo_segmentacion: str = "dependencias"
    #: Modelo de spaCy para SegmentadorDependencias. Se deja como
    #: parámetro (no hardcodeado) por si se decide evaluar "md" más adelante.
    modelo_spacy_dependencias: str = "es_core_news_lg"
    #: Reglas de reparación de fragmentos que aplica ``SegmentadorDependencias`` sobre la salida de spaCy
    #: (H29). Se ejecutan SIEMPRE en el orden R1, R2, R3, R4, R4B, R5. () = desactivadas (ft-r4.12).
    reglas_fragmentos: tuple[str, ...] = (
        "R1", "R2", "R3", "R4", "R4B", "R5", "R11", "R7", "R9", "R13", "R10", "R8", "R14", "R15", "R12",
    )
    #: R2: solo une la subordinada si tiene MENOS de este n.º de palabras de contenido (None = sin tope).
    r2_max_contenido: int | None = 6
    #: R3: True = prótasis + apódosis en UNA sola unidad (decisión del equipo); False = dos unidades.
    r3_unir_protasis: bool = True
    #: R4: no unir el adjunto si el fragmento anterior cierra con . ? !
    r4_estricto: bool = True
    #: R4B: palabras que, al final de un fragmento, lo dejan incompleto (se une al siguiente).
    r4b_cierres: tuple[str, ...] = (
        "incluso", "inclusive", "aun", "aún", "hasta", "ni siquiera", "sobre todo",
        "teniendo en cuenta", "tener en cuenta", "en cuenta", "debido a", "respecto a", "en cuanto a",
    )
    #: R12: una respuesta de <= este n.º de palabras (sin numerales) con varios fragmentos es UNA sola unidad.
    r12_max_palabras: int = 60
    #: R12: no unificar respuestas con `pero / sin embargo / no obstante` (protege dos opiniones contrapuestas).
    r12_guarda_adversativa: bool = False
    #: Cada unidad se devuelve como el tramo del texto ORIGINAL que cubre sus fragmentos (con su puntuación).
    #: Solo actúa si hay reglas activas; con ``reglas_fragmentos=()`` el texto es el de ft-r4.12.
    puntuacion_original: bool = True

    # --- Texto de aspecto (rama TÓPICOS) ---
    neutralizar_aspecto: bool = True
    quitar_nombres_propios: bool = True
    #: Palabras de contenido mínimas tras neutralizar para ser ASPECTO.
    min_contenido_aspecto: int = 1

    # --- UMAP ---
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.0

    # --- HDBSCAN ---
    min_topic_size: int = 60
    min_samples: int = 5
    cluster_selection_method: str = "leaf"
    cluster_selection_epsilon: float = 0.0

    # --- Post-proceso de la partición ---
    #: Coseno mínimo entre centroides para fusionar dos tópicos.
    umbral_fusion: float = 0.85
    #: Coseno mínimo entre una cláusula y el centroide para reasignarla.
    umbral_outliers: float = 0.45
    #: Crecimiento máximo de un tópico por reasignación (fracción).
    tope_crecimiento: float = 0.35
    #: Política de fusión de tópicos: "original" (solo coseno >= umbral_fusion,
    #: ``fusionar_redundantes``) o "raiz_coseno" (además fusiona a coseno >=
    #: ``fusion_umbral_raiz`` si comparten raíz en sus términos principales; sin
    #: cadenas y con tope de tamaño, ver ``crear_fusion_raiz_coseno``).
    politica_fusion: str = "original"
    #: Coseno mínimo si los dos tópicos comparten raíz (solo "raiz_coseno").
    fusion_umbral_raiz: float = 0.75
    #: Términos c-TF-IDF por tópico cuyas raíces se comparan.
    fusion_top_terminos: int = 3
    #: Prefijos de raíz que no cuentan como raíz compartida.
    fusion_raices_excluidas: tuple[str, ...] = ("univers", "estudi", "program")
    #: Tamaño máximo (% de las cláusulas con tópico) de un grupo fusionado (G4).
    fusion_tope_pct: float = 25.0

    # --- Estabilidad (ARI entre semillas UMAP) ---
    #: Semillas UMAP extra con las que se mide el ARI de cada finalista.
    ari_semillas: tuple[int, ...] = (7, 2024)
    #: Mejores filas de la rejilla a las que se mide el ARI.
    n_finalistas_estabilidad: int = 3

    # --- Representación (c-TF-IDF, después del clustering) ---
    ngram_range: tuple[int, int] = (1, 2)
    top_n_words: int = 10
    mmr_diversity: float = 0.15
    #: Fracción mínima de cláusulas del tópico que deben contener un término
    #: para mostrarlo (evita etiquetas de palabras raras o mal escritas).
    df_min_termino: float = 0.03

    # --- Sentimiento ---
    usar_preprocesamiento_tweet: bool = True
    max_length_sentimiento: int = 128
    #: NEU se reinterpreta como POS/NEG si |p_pos - p_neg| >= margen.
    margen_neu: float = 0.25

    # --- ABSA ---
    n_min_celda: int = 30
    z_wilson: float = 1.96
    #: Un tópico con cohesión baja (``revisar``) se considera NO
    #: confiable para priorizar si además su proporción de textos de aspecto
    #: de UNA palabra es al menos este múltiplo de la del corpus. Calibrado en
    #: obs19 (T2 = 3.25x; siguiente tópico con cohesión baja = 1.79x): todo
    #: umbral entre ~1.8 y ~3.2 da el mismo resultado; 2 = "el doble".
    umbral_veces_palabra_suelta: float = 2.0

    # --- Calibración del sentimiento (sección 11b) ---
    #: JSON de :class:`Calibrador`. Si existe, `sentimiento` sale calibrado.
    ruta_calibrador: Path | None = None

    taxonomia: dict[str, str] = field(default_factory=dict)

    # --- Poda de tópicos ---
    #: Tópicos con cohesión < media - k·sd se marcan `revisar`; con
    #: `podar_cohesion` True se disuelven en outliers antes de exportar.
    podar_cohesion: bool = False
    k_cohesion: float = 1.0

    # --- Auditoría ---
    muestra_pureza_por_topico: int = 10
    muestra_manual_n: int = 150
    silueta_muestra_max: int = 6000

    # --- Aspectos supervisados U4 (ft-r4.15) ---
    #: ``"bertopic"`` = UMAP/HDBSCAN (comportamiento exacto de ft-r4.14); ``"u4"`` = clasificador supervisado.
    modo_aspectos: str = "bertopic"
    #: Carpeta ``.../modelos/u4_aspectos/<clave>/<run_id>`` (o ``.../<clave>`` para tomar ``ultimo.json``).
    ruta_modelo_u4: Path | None = None
    #: Semilla de TODAS las fuentes de azar del entrenamiento de U4 (inicialización de la cabeza, dropout, lotes).
    semilla_u4: int = 42
    #: Respuesta corta = respuesta completa (sin limpiar) con <= este n.º de palabras (``split()``).
    max_palabras_respuesta_corta: int = 5
    #: Respuesta corta con probabilidad máxima de U4 < umbral -> «Comentario General». PROVISIONAL (ver diagnóstico).
    umbral_comentario_general: float = 0.50
    #: Plan v2, punto 7 (7.2): True = la cláusula SIN CONTENIDO (antes VALORACION_GLOBAL) va a «Opinión general» también en
    #: respuestas cortas, sin pasar por U4 (una cláusula sin contenido no tiene aspecto que predecir). False = como antes:
    #: en respuestas cortas pasa por U4 y puede terminar en un aspecto.
    sin_contenido_en_cortas: bool = True
    #: Cláusula larga (ASPECTO) con probabilidad máxima de U4 < umbral -> residual («Por descubrir»).
    umbral_residual_u4: float = 0.50
    #: Palabras mínimas de la cláusula (``split()`` sobre ``clausula``) para entrar al residual.
    min_palabras_residual: int = 6
    #: False = sin residual (toda cláusula ASPECTO conserva el aspecto de U4).
    residual_hdbscan_activo: bool = True
    #: Rejilla del modelado del residual ("rapida" / "completa").
    residual_rejilla: str = "rapida"
    #: Cláusulas mínimas del residual para ejecutar HDBSCAN (None = ``MINIMO_MODELABLES``).
    residual_minimo: int | None = None
    #: Sobrescribe campos de ``CriteriosNegocio`` para el modelado del residual (vacío = criterios relajados por defecto).
    residual_criterios: dict[str, Any] = field(default_factory=dict)

    # --- Salidas ---
    #: Salidas internas de modelado (una carpeta ``<encuesta>/<run_id>`` por corrida). Desde el plan v2 (punto 2) viven
    #: en ``data/processed/nlp/modelado`` (antes, la carpeta ``outputs`` de la raíz del proyecto).
    output_dir: Path = Path("data/processed/nlp/modelado")

    def __post_init__(self) -> None:
        if self.modo_aspectos not in (MODO_BERTOPIC, MODO_U4):
            raise ValueError(
                f"modo_aspectos debe ser '{MODO_BERTOPIC}' o '{MODO_U4}', no {self.modo_aspectos!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serializa a tipos JSON."""
        datos = asdict(self)
        for clave, valor in datos.items():
            if isinstance(valor, Path):
                datos[clave] = str(valor)
            elif isinstance(valor, tuple):
                datos[clave] = list(valor)
        return datos

    def huella(self) -> str:
        """Hash de los campos que afectan el resultado (no rutas ni cachés)."""
        irrelevantes = {"nombre", "usar_cache", "dir_cache", "output_dir", "batch_size"}
        datos = self.to_dict()
        # Opciones añadidas después de ft-r4.12: con su valor por defecto no
        # entran en la huella (la de una corrida Prisma previa no cambia).
        defecto = Config().to_dict()
        for clave in _CAMPOS_OPCIONALES_HUELLA:
            if datos.get(clave) == defecto[clave]:
                datos.pop(clave, None)
        # Reglas de fragmentos desactivadas = ft-r4.12: sus parámetros no cuentan.
        if not self.reglas_fragmentos:
            for clave in _CAMPOS_REGLAS_FRAGMENTOS:
                datos.pop(clave, None)
        # ft-r4.15: con modo_aspectos="bertopic" los campos U4 no cuentan (huella idéntica a ft-r4.14).
        if self.modo_aspectos != MODO_U4:
            for clave in _CAMPOS_U4 + _CAMPOS_RESIDUAL:
                datos.pop(clave, None)
        else:
            datos["ruta_modelo_u4"] = _huella_modelo_u4(self.ruta_modelo_u4)
            if not self.residual_hdbscan_activo:
                for clave in _CAMPOS_RESIDUAL:
                    datos.pop(clave, None)
        # El calibrador cambia el resultado: su contenido (no su ruta) va en la huella.
        datos["ruta_calibrador"] = (
            huella_texto(Path(self.ruta_calibrador).read_text(encoding="utf-8"))
            if self.ruta_calibrador and Path(self.ruta_calibrador).exists()
            else None
        )
        return huella_texto({k: v for k, v in datos.items() if k not in irrelevantes})


@dataclass
class CriteriosNegocio:
    """Puertas duras para aceptar una partición en el grid.

    Todas deben cumplirse. El puntaje sólo ordena lo que ya las cumple, y
    es monótono por construcción: más silueta, menos concentración, menos
    outliers y más estabilidad SIEMPRE suben el puntaje.
    """

    n_topicos_min: int = 8
    n_topicos_max: int = 30
    pct_outliers_max: float = 20.0  # tras reasignación
    pct_topico_mayor_max: float = 25.0
    pct_top2_max: float = 40.0
    silueta_min: float = 0.10
    ari_min: float = 0.70
    # Pesos del puntaje (suman 1).
    peso_silueta: float = 0.35
    peso_concentracion: float = 0.25
    peso_outliers: float = 0.20
    peso_estabilidad: float = 0.20


# ═══════════════════════════════════════════════════════════════════════════
# 5. CARGA DEL CORPUS
# ═══════════════════════════════════════════════════════════════════════════

ENCUESTAS: dict[str, str] = {
    "Evaluación Docente": (
        "Observaciones - 19. Aspectos para resaltar del docente "
        "y/o del espacio académico"
    ),
    "Autoevaluación Docente": (
        "Aspectos para resaltar de su labor como docente y/o del espacio académico"
    ),
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

#: Columnas adicionales por hoja. Cada columna es un modelo independiente.
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
        "Oportunidades de mejora de su labor como docente y/o del espacio académico",
    ],
}

#: Cláusulas con aspecto mínimas para admitir topic modeling.
MINIMO_MODELABLES: int = 300
#: Umbrales (Spearman documento, AUC nota baja) por tipo de pregunta, FIJADOS
#: ANTES de ver resultados de cada tipo. En preguntas de FORTALEZA ("aspectos
#: a resaltar") el texto es positivo por diseño y la nota se concentra arriba:
#: la varianza compartida posible es menor (efecto techo). El umbral estricto
#: se mantiene para MEJORA, donde el texto sí recorre toda la polaridad.
UMBRALES_VALIDACION: dict[str, tuple[float, float]] = {
    "FORTALEZA": (0.25, 0.65),
    "MEJORA": (0.35, 0.70),
    "ABIERTA": (0.30, 0.68),
}


def tipo_pregunta(columna: str) -> str:
    """FORTALEZA, MEJORA o ABIERTA según el enunciado de la columna."""
    texto = slug(columna)
    if "mejora" in texto or "sugerencia" in texto:
        return "MEJORA"
    if "resaltar" in texto or "destacar" in texto:
        return "FORTALEZA"
    return "ABIERTA"


def autor_texto(hoja: str, columna: str) -> str:
    """ESTUDIANTE o DOCENTE: quién escribió la respuesta.

    P24/P25 de Evaluación Docente y toda la Autoevaluación las escribe el
    propio docente sobre su labor. La nota de los estudiantes no puede
    validar ese texto (rho ≈ 0.04, AUC ≈ 0.50, como corresponde).
    """
    texto = slug(hoja) + " " + slug(columna)
    if "autoevaluacion" in texto or "su_labor" in texto:
        return "DOCENTE"
    return "ESTUDIANTE"


#: Calificación numérica usada como validación externa del sentimiento
#: (Paso 10 del notebook: Spearman, AUC, deciles contra esta nota).
#: ⚠️ MARCA PEDIDA POR EL EQUIPO — "¿de dónde sale la nota?" ⚠️
#: Esta es la respuesta: el módulo busca EXACTAMENTE una columna con este
#: nombre en la hoja/CSV de cada encuesta. Ver el método `Encuesta.notas()`
#: unas líneas más abajo: si esa columna no existe, `notas()` devuelve
#: `None` AHÍ MISMO, sin ningún error visible, y toda la validación externa
#: (Paso 10) queda saltada en silencio para esa encuesta. Confirmar con el
#: equipo si los 6 CSV finales institucionales traen una columna con este
#: nombre exacto (o cuál sería el nombre real, para ajustarlo aquí).
COLUMNA_VALIDACION: str = "Promedio Evaluación - General"


@dataclass
class Encuesta:
    """Una columna de texto abierto de una hoja."""

    nombre: str
    hoja: str
    columna: str
    marco: pd.DataFrame
    texto_original: list[str]
    #: Columnas de llave compuesta (hoja "llaves" de
    #: preguntas_abiertas.xlsx: Periodo, ID de respuesta, Código Cruce
    #: Autoevaluación, etc.). Vacío si se cargó con cargar_encuestas()
    #: (la ruta original, que no tenía este concepto). Se usa para que
    #: construir_unidades() pueda propagar la trazabilidad hasta cada fila
    #: de resultado (requisito institucional).
    columnas_llave: list[str] = field(default_factory=list)
    #: Corrida de la Etapa 1 de la que salen las respuestas (vacío si no vienen de la Etapa 1).
    id_corrida_preparacion: str = ""
    #: Conteos de la ingesta por pregunta (``total``, ``no_respuestas``, ``con_alerta``, ``procesadas``); vacío si no
    #: vienen de la Etapa 1.
    conteos_ingesta: dict[str, int] = field(default_factory=dict)

    @property
    def clave_stopwords(self) -> str:
        """Clave de la hoja ``particulares`` (p. ej. ``evaluaciondocente``)."""
        return slug(self.hoja).replace("_", "")

    def notas(self, columna: str = COLUMNA_VALIDACION) -> pd.Series | None:
        """Calificación numérica por fila, si la hoja la tiene.

        Este ``if`` es EXACTAMENTE el punto donde se salta la
        validación contra la nota (Paso 10) cuando la columna no existe.
        Ver el comentario junto a ``COLUMNA_VALIDACION`` más arriba.
        """
        if columna not in self.marco.columns:
            return None  # <- aquí se salta la validación del Paso 10, sin error
        return pd.to_numeric(self.marco[columna], errors="coerce").reset_index(
            drop=True
        )


def _etiqueta_columna(hoja: str, columna: str) -> str:
    """Nombre corto y estable ``Hoja :: P19``."""
    numero = re.match(r"Observaciones - (\d+)\.", columna)
    if numero:
        return f"{hoja} :: P{numero.group(1)}"
    return f"{hoja} :: {columna[:38].strip()}"


def cargar_encuestas(
    ruta_corpus: Path,
    encuestas: dict[str, str] | None = None,
    incluir_adicionales: bool = True,
) -> dict[str, Encuesta]:
    """Carga cada columna de texto abierto como unidad de modelado.

    Args:
        ruta_corpus: Libro Excel del corpus.
        encuestas: Mapa hoja -> columna principal (``ENCUESTAS`` si None).
        incluir_adicionales: Si también se cargan ``COLUMNAS_ADICIONALES``.

    Returns:
        Mapa ``"Hoja :: Pnn"`` -> :class:`Encuesta`.

    Raises:
        ValueError: Si no se pudo cargar ninguna columna.
    """
    mapa = encuestas or ENCUESTAS
    libro = pd.read_excel(ruta_corpus, sheet_name=None)
    logger.info("Hojas en el corpus: %s", list(libro))
    cargadas: dict[str, Encuesta] = {}
    for hoja, principal in mapa.items():
        if hoja not in libro:
            logger.warning("Hoja '%s' no está en el corpus; se omite", hoja)
            continue
        marco = libro[hoja].reset_index(drop=True)
        columnas = [principal] + (
            COLUMNAS_ADICIONALES.get(hoja, []) if incluir_adicionales else []
        )
        for columna in columnas:
            if columna not in marco.columns:
                logger.warning("Columna ausente en '%s': '%s...'", hoja, columna[:55])
                continue
            etiqueta = _etiqueta_columna(hoja, columna)
            cargadas[etiqueta] = Encuesta(
                nombre=etiqueta,
                hoja=hoja,
                columna=columna,
                marco=marco,
                texto_original=marco[columna].fillna("").astype(str).tolist(),
            )
            logger.info("  %-46s %7d filas", etiqueta[:46], len(marco))
    if not cargadas:
        raise ValueError("Ninguna columna pudo cargarse. Revise ENCUESTAS.")
    return cargadas


#: Encoding-fallback para CSV exportados desde Windows/Excel en
#: español. Mismo orden que se usó en el resto de scripts del proyecto.
_ENCODINGS_CSV: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

#: Sufijos de las columnas ya preprocesadas que TRAEN los 6 CSV
#: finales (_clean, _no_stopwords, _no_stopwords_no_adverbs). Se usan sólo
#: para EXCLUIRLAS al identificar la columna de texto original -- el equipo
#: decidió ignorarlas (ver CAMBIO 2 en la cabecera del módulo).
_SUFIJOS_PREPROCESADOS: tuple[str, ...] = (
    "_no_stopwords_no_adverbs",  # más específico primero
    "_no_stopwords",
    "_clean",
)


def _separador_del_encabezado(ruta: Path, por_defecto: str) -> str:
    """Separador que usa la PRIMERA línea del CSV (el encabezado).

    Solo elige entre separadores conocidos (";", ",", tabulador, "|") y solo si el
    encabezado contiene uno de ellos y NO contiene el separador por defecto. En
    cualquier duda devuelve ``por_defecto`` (comportamiento anterior).
    """
    for encoding in _ENCODINGS_CSV:
        try:
            with open(ruta, encoding=encoding) as archivo:
                encabezado = archivo.readline()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        return por_defecto
    if por_defecto in encabezado:
        return por_defecto
    candidatos = [s for s in (",", "\t", "|", ";") if s in encabezado]
    if len(candidatos) == 1:
        logger.info(
            "'%s': el encabezado usa separador %r (no %r); se lee con ese.",
            ruta.name,
            candidatos[0],
            por_defecto,
        )
        return candidatos[0]
    return por_defecto


def _leer_csv_institucion(ruta: Path, separador: str = ";") -> pd.DataFrame:
    """Lee un CSV institucional probando varios encodings.

    Args:
        ruta: Ruta del CSV.
        separador: Separador de campos esperado (";" por defecto, locale
            es-CO). Si con este separador el archivo revienta con
            ``ParserError`` (encabezado sin ese separador, p. ej. porque
            ese CSV en particular se exportó con "," u otro separador), se
            reintenta autodetectando el separador real del archivo
            (``sep=None, engine="python"``) y se deja constancia en el log
            de cuál se usó realmente, en vez de fallar en seco. Esto NO
            corrige una fila realmente corrupta (un "," o ";" suelto sin
            comillas dentro de una respuesta libre) -- si el archivo tiene
            eso, seguirá fallando y hay que revisar esa fila a mano.

    Returns:
        DataFrame con todas las columnas como texto (``dtype=str``); la
        conversión numérica de columnas específicas (p. ej. una nota) se
        hace después, donde se necesite.

    Raises:
        UnicodeDecodeError: Si ningún encoding de ``_ENCODINGS_CSV`` funciona.
        pd.errors.ParserError: Si ni siquiera autodetectando el separador
            se puede tokenizar el archivo de forma consistente (fila(s)
            realmente corrupta(s); requiere revisión manual).
    """
    ultimo: Exception | None = None
    separador = _separador_del_encabezado(ruta, separador)
    for encoding in _ENCODINGS_CSV:
        try:
            return pd.read_csv(ruta, sep=separador, encoding=encoding, dtype=str)
        except (UnicodeDecodeError, UnicodeError) as exc:
            ultimo = exc
            continue
        except pd.errors.ParserError as exc_parser:
            logger.warning(
                "'%s': falló con separador '%s' (%s). "
                "Reintentando con separador autodetectado...",
                ruta.name,
                separador,
                exc_parser,
            )
            try:
                marco = pd.read_csv(
                    ruta, sep=None, engine="python", encoding=encoding, dtype=str
                )
            except (UnicodeDecodeError, UnicodeError) as exc:
                ultimo = exc
                continue
            logger.warning(
                "'%s': leído con separador autodetectado en vez de '%s'. "
                "Si el resultado se ve mal (columnas de más/de menos), revise el "
                "archivo fuente -- puede tener un separador suelto sin comillas "
                "dentro de una respuesta de texto libre.",
                ruta.name,
                separador,
            )
            return marco
    raise UnicodeDecodeError(  # pragma: no cover - sólo si el archivo está muy corrupto
        "desconocido", b"", 0, 1, f"No se pudo leer {ruta} con {_ENCODINGS_CSV}: {ultimo}"
    )


def _columna_texto_original(
    columnas: Sequence[str],
    columnas_llave: Sequence[str],
    alias_redaccion: str | None = None,
) -> str:
    """Identifica la columna de texto crudo del CSV.

    CAMBIO (Opción A, pedida por el equipo): ahora SÍ se
    intenta primero por coincidencia EXACTA con ``alias_redaccion`` del
    catálogo -- se confía en que ``preguntas_abiertas.xlsx`` trae el nombre
    real de la columna, en vez de adivinar solo por descarte. Si
    ``alias_redaccion`` no coincide EXACTO con ninguna columna del CSV
    (catálogo desactualizado, o no se pasó), se registra una advertencia y
    se cae al método de eliminación original como respaldo -- nunca se
    rompe por esto, pero tampoco se "arregla" la diferencia en silencio con
    una comparación tolerante: si el catálogo y el CSV real no coinciden,
    eso debe verse y corregirse en la fuente, no disimularse aquí.

    Args:
        columnas: Encabezados del CSV, en orden.
        columnas_llave: Columnas de llave compuesta de esa encuesta.
        alias_redaccion: Nombre de columna que dice el catálogo
            (hoja "preguntas") para esta encuesta::pregunta, si lo trae.

    Returns:
        Nombre de la columna de texto original.

    Raises:
        ValueError: Si tampoco por eliminación queda exactamente una.
    """
    if alias_redaccion and str(alias_redaccion).strip():
        # .strip() aquí es solo higiene del VALOR leído de la celda
        # de Excel (espacio de más al escribir en el catálogo) -- no es una
        # tolerancia sobre el encabezado del CSV, que se compara tal cual.
        alias = str(alias_redaccion).strip()
        if alias in columnas:
            return alias
        logger.warning(
            "alias_redaccion='%s' no coincide EXACTO con ninguna "
            "columna del CSV; se cae al método de eliminación como respaldo. "
            "Revisar si el catálogo quedó desactualizado respecto al CSV real.",
            alias,
        )

    # Respaldo por eliminación. Se excluye también COLUMNA_VALIDACION
    # (la columna de la nota, CAMBIO del Paso 10) además de llaves y
    # preprocesadas -- si no, en las encuestas que ya traen la nota, quedarían
    # DOS columnas candidatas y esto fallaría igual que antes de agregarla.
    candidatas = [
        c
        for c in columnas
        if c not in columnas_llave
        and c != COLUMNA_VALIDACION
        and not any(c.endswith(sufijo) for sufijo in _SUFIJOS_PREPROCESADOS)
    ]
    if len(candidatas) != 1:
        raise ValueError(
            "No se identificó una única columna de texto original "
            f"(candidatas tras excluir llaves, '{COLUMNA_VALIDACION}' y "
            f"preprocesadas: {candidatas})"
        )
    return candidatas[0]


def cargar_encuestas_institucion(
    ruta_catalogo: Path,
    carpeta_datos: Path,
    *,
    separador_csv: str = ";",
) -> dict[str, Encuesta]:
    """Carga los 6 CSV finales reales de la institución.

    Reemplaza a :func:`cargar_encuestas` para este proyecto: los datos NO
    viven en un único Excel "corpus de ejemplo" con una hoja por encuesta

    sino en 6 CSV independientes -- uno por combinación encuesta::pregunta
    -- catalogados en ``preguntas_abiertas.xlsx`` (hojas "preguntas" y
    "llaves"), producto de un proceso previo sobre los 2 Excel crudos
    institucionales (Consolidado_Histórico_Evaluación_Docente_Anonimizado
    y Encuestas_Autoevaluación).

    Usa SOLO la columna de texto original de cada CSV (con tildes y
    puntuación intactas); las 3 columnas preprocesadas que trae cada CSV
    (_clean, _no_stopwords, _no_stopwords_no_adverbs) se ignoran a propósito
    -- ver el CAMBIO 2 documentado en la cabecera del módulo.

    Args:
        ruta_catalogo: Ruta a ``preguntas_abiertas.xlsx``.
        carpeta_datos: Carpeta donde están los 6 CSV finales, nombrados
            ``{encuesta}__{clave_logica}.csv``.
        separador_csv: Separador de los CSV (";" por defecto, locale es-CO).

    Returns:
        Mapa ``"{encuesta} :: {clave_logica}"`` -> :class:`Encuesta`, con
        ``columnas_llave`` poblado para trazabilidad end-to-end.

    Raises:
        ValueError: Si ninguna fila activa del catálogo pudo cargarse.
    """
    preguntas = pd.read_excel(ruta_catalogo, sheet_name="preguntas")
    llaves = pd.read_excel(ruta_catalogo, sheet_name="llaves")
    llaves_por_encuesta: dict[str, list[str]] = {
        str(enc): grupo["columna_llave"].astype(str).tolist()
        for enc, grupo in llaves.groupby("encuesta")
    }
    activas = preguntas[
        preguntas["activa"].astype(str).str.strip().str.lower() == "si"
    ]
    logger.info(
        "Catálogo %s: %d preguntas activas de %d",
        ruta_catalogo.name,
        len(activas),
        len(preguntas),
    )
    cargadas: dict[str, Encuesta] = {}
    for _, fila in activas.iterrows():
        encuesta_nombre = str(fila["encuesta"]).strip()
        clave_logica = str(fila["clave_logica"]).strip()
        nombre_csv = f"{encuesta_nombre}__{clave_logica}.csv"
        ruta_csv = carpeta_datos / nombre_csv
        if not ruta_csv.exists():
            logger.warning(
                "No existe '%s' en %s; se omite '%s'",
                nombre_csv,
                carpeta_datos,
                clave_logica,
            )
            continue
        marco = _leer_csv_institucion(ruta_csv, separador_csv).reset_index(drop=True)
        columnas_llave = [
            c for c in llaves_por_encuesta.get(encuesta_nombre, []) if c in marco.columns
        ]
        try:
            columna_texto = _columna_texto_original(
                list(marco.columns), columnas_llave, alias_redaccion=fila.get("alias_redaccion")
            )
        except ValueError as exc:
            logger.error("'%s': %s; se omite", clave_logica, exc)
            continue
        etiqueta = f"{encuesta_nombre} :: {clave_logica}"
        cargadas[etiqueta] = Encuesta(
            nombre=etiqueta,
            # OJO: `hoja` ya no es un nombre de hoja de Excel tipo
            # "Evaluación Docente" sino la clave `encuesta` del catálogo
            # (p. ej. "evadoc_evaluacion"). `clave_stopwords` (más abajo)
            # hace slug() de este valor para buscarlo en stopwords.xlsx --
            # HAY QUE VERIFICAR que la hoja "particulares" de stopwords.xlsx
            # institucional agrupe por esta misma convención de nombres; si
            # usa otra (p. ej. "Evaluación Docente"), stopwords_vectorizador()
            # no encontrará coincidencias particulares para esta encuesta
            # (seguiría funcionando con NLTK + globales, pero sin las
            # particulares por encuesta).
            hoja=encuesta_nombre,
            columna=columna_texto,
            marco=marco,
            texto_original=marco[columna_texto].fillna("").astype(str).tolist(),
            columnas_llave=columnas_llave,
        )
        logger.info(
            "  %-55s %7d filas | columna: '%s...' | llaves: %s",
            etiqueta[:55],
            len(marco),
            columna_texto[:40],
            columnas_llave,
        )
    if not cargadas:
        raise ValueError(
            "Ninguna encuesta pudo cargarse desde los CSV institucionales. "
            "Revise preguntas_abiertas.xlsx y la carpeta de datos."
        )
    return cargadas


# ═══════════════════════════════════════════════════════════════════════════
# 5b. INGESTA DESDE LA ETAPA 1 (contrato de datos v0.2, secciones 4.3 y 5.1)
# ═══════════════════════════════════════════════════════════════════════════

#: Columnas que el NLP exige de ``respuestas.csv`` (contrato v0.2, sección 4.3).
COLUMNAS_RESPUESTAS_ETAPA1: tuple[str, ...] = (
    "respuesta_id",
    "encuesta",
    "pregunta_codigo",
    "periodo",
    "rol_respondiente",
    "texto_original",
    "es_no_respuesta",
    "tiene_alerta",
    "id_corrida_preparacion",
)


def _leer_csv_etapa1(ruta: Path) -> pd.DataFrame:
    """CSV de la Etapa 1: todo texto (el periodo «2025-1» y valores como «NA» no se pierden)."""
    return pd.read_csv(ruta, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def leer_respuestas_etapa1(carpeta_etapa1: Path | str) -> tuple[str, Path, pd.DataFrame]:
    """Lee ``respuestas.csv`` de la corrida más reciente de la Etapa 1 (``preparacion.corridas.ultima_corrida``).

    Args:
        carpeta_etapa1: ``data/processed/etapa1`` (contiene una carpeta por ``id_corrida_preparacion``).

    Returns:
        ``(id_corrida_preparacion, carpeta de la corrida, respuestas)`` con TODAS las filas del archivo.

    Raises:
        FileNotFoundError: Si no hay ninguna corrida completa (lo da ``ultima_corrida``).
        ValueError: Si faltan columnas, si hay valores de ``es_no_respuesta`` distintos de True/False, si
            ``respuesta_id`` está vacío o repetido, o si las filas no traen todas el id de la corrida.
    """
    from preparacion.corridas import ultima_corrida  # pylint: disable=import-outside-toplevel

    id_corrida, carpeta = ultima_corrida(carpeta_etapa1)
    respuestas = _leer_csv_etapa1(carpeta / "respuestas.csv")
    faltan = [c for c in COLUMNAS_RESPUESTAS_ETAPA1 if c not in respuestas.columns]
    if faltan:
        sugerencia = (
            " La corrida es anterior a la Etapa 1 v1.10 (alertas): vuelva a correr `python run_etapa1.py`."
            if "tiene_alerta" in faltan
            else ""
        )
        raise ValueError(f"respuestas.csv de la corrida {id_corrida} no trae las columnas {faltan}.{sugerencia}")
    for columna in ("es_no_respuesta", "tiene_alerta"):
        raros = sorted(set(respuestas[columna]) - {"True", "False"})
        if raros:
            raise ValueError(f"{columna} solo admite True/False; se encontró {raros}.")
    if (respuestas["respuesta_id"].str.strip() == "").any():
        raise ValueError("respuestas.csv tiene respuesta_id vacío.")
    if respuestas["respuesta_id"].duplicated().any():
        raise ValueError(f"respuestas.csv tiene {int(respuestas['respuesta_id'].duplicated().sum())} respuesta_id repetidos.")
    ids = set(respuestas["id_corrida_preparacion"])
    if ids - {id_corrida}:
        raise ValueError(
            f"Las filas de respuestas.csv no traen todas el id de la corrida {id_corrida}: encontrado {sorted(ids)}."
        )
    return id_corrida, carpeta, respuestas


def cargar_encuestas_etapa1(
    carpeta_etapa1: Path | str,
    *,
    preguntas: Sequence[str] | None = None,
) -> dict[str, Encuesta]:
    """Adaptador: respuestas procesables de la Etapa 1 -> los mismos objetos :class:`Encuesta` de siempre.

    Reemplaza a :func:`cargar_encuestas_institucion` (que lee los 6 CSV ``{encuesta}__{clave}.csv``). Toma la
    corrida más reciente, conserva solo ``es_no_respuesta = "False"`` **y** ``tiene_alerta = "False"`` y deja
    ``texto_original`` SIN limpiar (es lo que se segmenta). Las respuestas con alerta NO pasan al NLP aunque el
    término esté negado (las alertas son de la Etapa 1, v1.10). ``respuesta_id`` va como llave
    (``columnas_llave``): :func:`construir_unidades` la propaga a ``documentos`` y ``clausulas``; ``doc_id``
    sigue siendo la posición interna dentro de la pregunta. ``Encuesta.conteos_ingesta`` guarda, por pregunta,
    cuántas filas había y cuántas se excluyeron por cada motivo (solo conteos, para la auditoría).

    Args:
        carpeta_etapa1: ``data/processed/etapa1``.
        preguntas: Códigos de pregunta a cargar (todas las que tengan comentarios si es None).

    Returns:
        ``"{encuesta} :: {pregunta_codigo}"`` -> :class:`Encuesta` (``hoja`` = código de encuesta,
        ``columna`` = ``"texto_original"``, ``id_corrida_preparacion`` = la corrida leída).

    Raises:
        FileNotFoundError, ValueError: ver :func:`leer_respuestas_etapa1`; además si una pregunta no está en
            ``catalogo_preguntas.csv``, si una pregunta aparece en dos encuestas o si no queda ninguna respuesta.
    """
    id_corrida, carpeta, respuestas = leer_respuestas_etapa1(carpeta_etapa1)
    ruta_catalogo = carpeta / "catalogo_preguntas.csv"
    if not ruta_catalogo.exists():  # la spec lo ubica junto a las corridas
        ruta_catalogo = Path(carpeta_etapa1) / "catalogo_preguntas.csv"
    if not ruta_catalogo.exists():
        raise FileNotFoundError(f"No existe catalogo_preguntas.csv en {carpeta} ni en {carpeta_etapa1}.")
    catalogo = _leer_csv_etapa1(ruta_catalogo)
    if preguntas is not None:
        respuestas = respuestas[respuestas["pregunta_codigo"].isin(list(preguntas))]
    sin_comentario = respuestas["es_no_respuesta"] == "True"
    con_alerta = (~sin_comentario) & (respuestas["tiene_alerta"] == "True")
    con_comentario = respuestas[~sin_comentario & ~con_alerta]
    conteos = {
        pregunta: {
            "total": int(len(g)),
            "no_respuestas": int((g["es_no_respuesta"] == "True").sum()),
            "con_alerta": int(((g["es_no_respuesta"] == "False") & (g["tiene_alerta"] == "True")).sum()),
            "procesadas": int(((g["es_no_respuesta"] == "False") & (g["tiene_alerta"] == "False")).sum()),
        }
        for pregunta, g in respuestas.groupby("pregunta_codigo", sort=True)
    }
    desconocidas = sorted(set(con_comentario["pregunta_codigo"]) - set(catalogo["pregunta_codigo"]))
    if desconocidas:
        raise ValueError(f"Preguntas de respuestas.csv que no están en catalogo_preguntas.csv: {desconocidas}")
    vacias = int((con_comentario["texto_original"].str.strip() == "").sum())
    if vacias:
        logger.warning("Corrida %s: %d respuestas con es_no_respuesta=False y texto vacío.", id_corrida, vacias)
    cargadas: dict[str, Encuesta] = {}
    for pregunta, grupo in con_comentario.groupby("pregunta_codigo", sort=True):
        encuestas = sorted(set(grupo["encuesta"]))
        if len(encuestas) != 1:
            raise ValueError(f"La pregunta {pregunta} aparece en varias encuestas: {encuestas}.")
        marco = grupo.reset_index(drop=True)
        etiqueta = f"{encuestas[0]} :: {pregunta}"
        cargadas[etiqueta] = Encuesta(
            nombre=etiqueta,
            hoja=encuestas[0],  # código de encuesta del catálogo; clave_stopwords = slug(hoja) sin "_"
            columna="texto_original",
            marco=marco,
            texto_original=marco["texto_original"].tolist(),
            columnas_llave=["respuesta_id"],
            id_corrida_preparacion=id_corrida,
            conteos_ingesta=conteos[pregunta],
        )
        logger.info("  %-55s %7d respuestas procesables (corrida %s)", etiqueta[:55], len(marco), id_corrida)
    if not cargadas:
        raise ValueError(
            f"La corrida {id_corrida} no tiene respuestas procesables (con comentario y sin alerta) "
            f"para {preguntas or 'ninguna pregunta'}."
        )
    return cargadas


def comparar_con_metadatos_etapa1(encuestas: dict[str, Encuesta], carpeta_etapa1: Path | str) -> pd.DataFrame:
    """Respuestas leídas vs ``metadatos_preparacion.csv`` de la misma corrida (solo conteos).

    Lo esperado es ``respuestas_con_comentario − respuestas_con_alerta``: las respuestas con alerta no pasan al NLP.

    Returns:
        Una fila por pregunta: ``total``, ``no_respuestas``, ``con_alerta`` (de la ingesta), ``leidas``,
        ``metadatos`` (con comentario − con alerta) y ``diferencia`` (``leidas - metadatos``).

    Raises:
        ValueError: Si ``metadatos_preparacion.csv`` no trae ``respuestas_con_alerta`` (corrida anterior a v1.10).
    """
    from preparacion.corridas import ultima_corrida  # pylint: disable=import-outside-toplevel

    _, carpeta = ultima_corrida(carpeta_etapa1)
    meta = _leer_csv_etapa1(carpeta / "metadatos_preparacion.csv")
    if "respuestas_con_alerta" not in meta.columns:
        raise ValueError(
            "metadatos_preparacion.csv no trae respuestas_con_alerta: la corrida es anterior a la Etapa 1 v1.10; "
            "vuelva a correr `python run_etapa1.py`."
        )
    esperado = pd.Series(
        (pd.to_numeric(meta["respuestas_con_comentario"]) - pd.to_numeric(meta["respuestas_con_alerta"])).to_numpy(),
        index=meta["pregunta_codigo"],
        dtype="int64",
    )
    filas = []
    for etiqueta, enc in encuestas.items():
        pregunta = etiqueta.split(" :: ")[-1]
        leidas = len(enc.texto_original)
        ref = int(esperado.get(pregunta, -1))
        c = enc.conteos_ingesta
        filas.append({
            "pregunta_codigo": pregunta,
            "total": c.get("total", -1),
            "no_respuestas": c.get("no_respuestas", -1),
            "con_alerta": c.get("con_alerta", -1),
            "leidas": leidas,
            "metadatos": ref,
            "diferencia": leidas - ref,
        })
    return pd.DataFrame(filas)


# ═══════════════════════════════════════════════════════════════════════════
# 6. CLÁUSULAS: SEGMENTACIÓN, NO OPINIÓN Y TEXTO DE ASPECTO
# ═══════════════════════════════════════════════════════════════════════════

_ABREVIATURAS = (
    "Dr",
    "Dra",
    "Drs",
    "Sr",
    "Sra",
    "Srta",
    "Ing",
    "Lic",
    "Prof",
    "Profa",
    "Mg",
    "Msc",
    "MSc",
    "PhD",
    "Esp",
    "Ud",
    "Uds",
    "Vd",
    "etc",
    "aprox",
    "Aprox",
    "pág",
    "Pág",
    "Nro",
    "núm",
    "Núm",
    "Cra",
    "Av",
    "Dpto",
    "Mtro",
    "Mtra",
    "Abog",
    "Adm",
    "Econ",
)
_SELLO = "\u2024"  # punto de una línea: nunca aparece en respuestas reales


def normalizar_clean(texto: str, recursos: RecursosLexicos) -> str:
    """Versión ``_clean``: minúsculas, sin tildes, sin símbolos ni puntuación.

    Args:
        texto: Texto original.
        recursos: Léxicos (mapa de acentos).

    Returns:
        Texto plano con espacios simples.
    """
    if not isinstance(texto, str):
        return ""
    plano = recursos.sin_acentos(texto.lower())
    plano = re.sub(r"[^a-z0-9ñ\s]", " ", plano)
    return re.sub(r"\s+", " ", plano).strip()


class Segmentador:
    """Parte una respuesta en cláusulas (unidades de opinión).

    Corta por puntuación fuerte, salto de línea y conectores ADVERSATIVOS
    de negaciones.xlsx. Protege abreviaturas ("Dr."), decimales ("4.5") e
    iniciales ("J. Pérez"), que el MVP 2 cortaba.
    """

    def __init__(self, cfg: Config, recursos: RecursosLexicos) -> None:
        self.cfg = cfg
        adversativos = sorted(
            recursos.terminos_por_tipo("ADVERSATIVO", con_tilde=True),
            key=len,
            reverse=True,  # "sin embargo" antes que "sin"
        )
        alternancia = "|".join(re.escape(a) for a in adversativos) or "pero"
        self._corte = re.compile(
            r"(?:[.;!?\n\r]+|\s*,?\s+(?:" + alternancia + r")\s+)",
            flags=re.IGNORECASE,
        )
        self._conector_inicial = re.compile(
            r"^\s*(?:" + alternancia + r")\b[\s,]*", flags=re.IGNORECASE
        )
        self._abreviatura = re.compile(
            r"\b(" + "|".join(re.escape(a) for a in _ABREVIATURAS) + r")\.",
        )
        self._decimal = re.compile(r"(?<=\d)\.(?=\d)")
        self._inicial = re.compile(r"\b([A-ZÁÉÍÓÚÑ])\.(?=\s*[A-ZÁÉÍÓÚÑ][a-záéíóúñ])")
        self._relleno = re.compile(
            r"^\s*(?:ningun[oa]?|ningún|nada|ninguno|ninguna)\s*[,.:;]\s*",
            flags=re.IGNORECASE,
        )

    def _proteger(self, texto: str) -> str:
        """Sustituye los puntos que no terminan oración por un sello."""
        texto = self._abreviatura.sub(lambda m: m.group(1) + _SELLO, texto)
        texto = self._decimal.sub(_SELLO, texto)
        return self._inicial.sub(lambda m: m.group(1) + _SELLO, texto)

    def segmentar(self, texto: str) -> list[str]:
        """Cláusulas de una respuesta, en orden de aparición.

        Args:
            texto: Respuesta original (con tildes).

        Returns:
            Lista de cláusulas limpias de conectores y puntuación de borde.
        """
        if not isinstance(texto, str) or not texto.strip():
            return []
        texto = self._relleno.sub("", self._proteger(texto.strip()))
        crudas: list[str] = []
        for fragmento in self._corte.split(texto):
            sin_conector = self._conector_inicial.sub("", fragmento)
            # Concesivo antepuesto: "Aunque llega tarde, sus clases..."
            if sin_conector != fragmento and "," in sin_conector:
                crudas.extend(sin_conector.split(",", 1))
            else:
                crudas.append(sin_conector)
        salida = []
        for cruda in crudas:
            limpia = self._relleno.sub("", cruda.replace(_SELLO, ".")).strip(" ,;:-")
            for parte in self._cortar_por_comas(limpia):
                if (
                    re.search(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]", parte)
                    and len(parte.split()) >= self.cfg.min_tokens_clausula
                ):
                    salida.append(parte)
        return salida[: self.cfg.max_clausulas_doc]

    def _cortar_por_comas(self, clausula: str) -> list[str]:
        """Parte cláusulas largas por comas en fragmentos de tamaño mínimo.

        "Clara, concisa e interesante" (corta) no se toca. Una cláusula de
        25+ palabras se corta por comas acumulando fragmentos hasta
        ``min_palabras_fragmento``; el sobrante corto se une al anterior,
        así una enumeración de adjetivos nunca queda como cláusulas sueltas.
        """
        limite = self.cfg.cortar_clausulas_largas
        if limite <= 0 or len(clausula.split()) < limite or "," not in clausula:
            return [clausula]
        partes: list[str] = []
        actual: list[str] = []
        for trozo in clausula.split(","):
            actual.append(trozo.strip())
            if len(" ".join(actual).split()) >= self.cfg.min_palabras_fragmento:
                partes.append(", ".join(a for a in actual if a))
                actual = []
        resto = ", ".join(a for a in actual if a)
        if resto:
            if partes and len(resto.split()) < self.cfg.min_palabras_fragmento:
                partes[-1] = f"{partes[-1]}, {resto}"
            else:
                partes.append(resto)
        return [p.strip(" ,") for p in partes if p.strip(" ,")]


_DEPS_CLAUSULA_SIEMPRE = frozenset({"ROOT", "advcl", "parataxis"})
#: Tokens de contenido mínimos (sin puntuación) por fragmento antes de
#: fusionarlo con el anterior en vez de dejarlo suelto.
_LONGITUD_MINIMA_TOKENS_DEP = 3


def _tiene_sujeto_propio_dep(token: Any) -> bool:
    """True si ``token`` tiene un hijo sujeto (nsubj/csubj, activa o pasiva)."""
    return any(
        hijo.dep_ in ("nsubj", "nsubj:pass", "csubj", "csubj:pass")
        for hijo in token.children
    )


def _tiene_copula_dep(token: Any) -> bool:
    """True si ``token`` es el predicado de una cópula ("X es/está Y")."""
    return any(hijo.dep_ == "cop" for hijo in token.children)


def _es_advcl_complemento_dep(token: Any) -> bool:
    """True si esta ``advcl`` es en realidad un complemento preposicional
    ("alineados CON lo que esperábamos", "se esfuerza EN hacer...") y no una
    cláusula subordinada genuina. Patrón: ``mark`` = preposición (ADP) + verbo
    en forma finita **o infinitivo** ().
    """
    marca = next((h for h in token.children if h.dep_ == "mark"), None)
    if marca is None or marca.pos_ != "ADP":
        return False
    return bool({"Fin", "Inf"} & set(token.morph.get("VerbForm")))


#: Terminaciones reales de gerundio. Se exigen además de la etiqueta
#: morfológica de spaCy porque el parser a veces marca como gerundio un verbo
#: conjugado ("aprendo" en "porque aprendo mucho"), y esa cláusula SÍ debe cortarse.
_TERMINACIONES_GERUNDIO = ("ando", "iendo", "yendo")


def _es_advcl_no_finita_adherida_dep(token: Any) -> bool:
    """True si esta ``advcl`` no finita depende de la oración
    principal y NO debe abrir una cláusula propia (complementa a
    ``_es_advcl_complemento_dep``, que solo mira ``mark``):

    - infinitivo con preposición como ``mark`` o ``case`` ("AL explicar los
      temas, usa ejemplos", "sin dejar de lado...");
    - gerundio real ("explica con paciencia, DANDO ejemplos").

    Un infinitivo SIN preposición sigue cortando: en las listas de
    sugerencias ("Realizar visitas..., Invitar expertos...") cada infinitivo
    es una idea distinta (obs20). Verificado: la primera
    versión de la regla (toda advcl no finita) unía esas listas.
    """
    forma = token.morph.get("VerbForm")
    if "Inf" in forma:
        return any(h.pos_ == "ADP" and h.dep_ in ("mark", "case") for h in token.children)
    if "Ger" in forma:
        return token.lower_.endswith(_TERMINACIONES_GERUNDIO)
    return False


def _es_perifrasis_modal_dep(token: Any) -> bool:
    """True si ``token`` es el
    infinitivo de una perífrasis modal ("tiene QUE volverse", "hay QUE
    cambiar") mal marcada como coordinada, no una cláusula independiente con
    sujeto propio genuino. Antes el segmentador devolvía ``['tiene .',
    'La universidad que volverse más ágil...']`` (Paso 3 del cuaderno).
    """
    if "Inf" not in token.morph.get("VerbForm"):
        return False
    return any(h.dep_ == "cc" and h.lower_ == "que" for h in token.children)


def _es_cabeza_de_clausula_dep(token: Any) -> bool:
    """True si ``token`` encabeza su propia cláusula (ver reglas abajo)."""
    es_copulativo = _tiene_copula_dep(token)
    if not es_copulativo and token.pos_ not in ("VERB", "AUX"):
        return False
    if token.dep_ == "advcl" and (
        _es_advcl_complemento_dep(token) or _es_advcl_no_finita_adherida_dep(token)
    ):
        return False
    if token.dep_ in _DEPS_CLAUSULA_SIEMPRE:
        return True
    if token.dep_ == "conj":
        # la perífrasis modal no es cláusula propia.
        if _es_perifrasis_modal_dep(token):
            return False
        return es_copulativo or _tiene_sujeto_propio_dep(token)
    return False


def _cabeza_de_clausula_ancestro_dep(token: Any, cabezas: set[Any]) -> Any:
    """Sube por el árbol de dependencias hasta la cabeza de cláusula más
    cercana (o hasta la raíz de la oración si ninguna aplica)."""
    actual = token
    while True:
        if actual in cabezas:
            return actual
        if actual.head == actual:  # raíz de la oración
            return actual
        actual = actual.head


def _reconstruir_dep(tokens: Sequence[Any]) -> str:
    """Reensambla una lista de tokens de spaCy preservando espacios."""
    return "".join(t.text_with_ws for t in tokens).strip()


# ═══════════════════════════════════════════════════════════════════════════
# 6b. REGLAS DE REPARACIÓN DE FRAGMENTOS (H29, ft-r4.13)
# ═══════════════════════════════════════════════════════════════════════════
# Funciones PURAS sobre la lista de fragmentos que devuelve spaCy. Cada una devuelve
# ``(fragmentos_nuevos, grupos)``: ``grupos[k]`` = índices de entrada que forman el fragmento nuevo k.
# Probadas contra la validación humana en Prisma_reglas_fragmentos.ipynb (R0 y R6 no se portan: no
# pasaron el criterio de controles).

#: Orden fijo de aplicación; ``Config.reglas_fragmentos`` elige cuáles se aplican.
REGLAS_FRAGMENTOS_ORDEN: tuple[str, ...] = (
    "R1", "R2", "R3", "R4", "R4B", "R5", "R11", "R7", "R9", "R13", "R10", "R8", "R14", "R15", "R12",
)
#: Paquete y cierres de R4B de ft-r4.13 (para reproducirlo, junto con ``puntuacion_original=False``).
REGLAS_FRAGMENTOS_FT413: tuple[str, ...] = ("R1", "R2", "R3", "R4", "R4B", "R5")
R4B_CIERRES_FT413: tuple[str, ...] = ("incluso", "inclusive", "aun", "aún", "hasta", "ni siquiera", "sobre todo")

_INI_PUNT_FRAG = re.compile(r"^\s*[,;:]+\s*")
_ADVERSATIVOS_R1: tuple[str, ...] = ("pero", "sin embargo", "no obstante", "empero")
_PROTASIS: tuple[str, ...] = (
    "a pesar de que", "a pesar de", "pese a que", "aun cuando", "aún cuando", "si bien",
    "así no", "asi no", "aunque", "si",
)
_SUBORDINANTES: tuple[str, ...] = (
    "porque", "ya que", "dado que", "puesto que", "pues", "para que", "a fin de que", "cuando",
    "mientras que", "mientras", "lo que", "lo cual", "donde", "de modo que", "de manera que",
    "por lo que", "por lo cual", "debido a que", "con lo que", "con lo cual", "que",
)


def _patron_inicio_frag(conectores: Sequence[str], con_conjuncion: bool = False) -> re.Pattern[str]:
    """Regex que detecta un conector al inicio (opcionalmente precedido de y/e/o/u/ni)."""
    alt = "|".join(re.escape(c) for c in sorted(conectores, key=len, reverse=True))
    prefijo = r"(?:(?:y|e|o|u|ni)\s+)?" if con_conjuncion else ""
    return re.compile(rf"^{prefijo}(?:{alt})\b", re.IGNORECASE)


def _unir_frag(a: str, b: str) -> str:
    """Une dos fragmentos conservando una coma/signo inicial de ``b``."""
    a, b = a.rstrip(), b.strip()
    if b[:1] in (",", ";", ":"):
        return a.rstrip(" .,;:") + b[0] + " " + b[1:].lstrip(" ,;:")
    return a.rstrip(" .") + " " + b


def r1_huerfano(
    fragmentos: Sequence[str],
    adversativos: Sequence[str] = _ADVERSATIVOS_R1,
    protasis: Sequence[str] = _PROTASIS,
) -> tuple[list[str], list[list[int]]]:
    """R1 · Fragmento huérfano: si empieza con ``,`` ``;`` ``:`` se une al anterior.

    Excepción: ``, pero …``, ``, sin embargo …``, ``, aunque …`` NO se unen (separan dos opiniones) y se
    dejan EXACTAMENTE como vienen (no se limpian comas ni conectores). Tampoco se toca el primer
    fragmento aunque empiece con coma. Si el anterior es una prótasis (aunque/si bien/si…), no se une:
    lo decide R3.
    """
    re_adv = _patron_inicio_frag(adversativos)
    re_prot = _patron_inicio_frag(protasis)
    re_aunque = re.compile(r"aunque\b", re.IGNORECASE)
    salida: list[str] = []
    grupos: list[list[int]] = []
    for k, frag in enumerate(fragmentos):
        m = _INI_PUNT_FRAG.match(frag)
        if not m:
            salida.append(frag); grupos.append([k]); continue
        resto = frag[m.end():].strip()
        previo = salida[-1] if salida else ""
        if re_adv.match(resto) or re_aunque.match(resto):
            salida.append(frag); grupos.append([k])          # adversativa: separada y SIN limpiar
        elif not salida:
            salida.append(frag); grupos.append([k])          # sin anterior a quien unirse
        elif re_prot.match(previo.lstrip(" ,;:")):
            salida.append(frag); grupos.append([k])          # la prótasis la resuelve R3
        else:
            salida[-1] = _unir_frag(previo, frag); grupos[-1].append(k)
    return salida, grupos


def r2_subordinada_adherida(
    fragmentos: Sequence[str],
    subordinantes: Sequence[str] = _SUBORDINANTES,
    respeta_punto: bool = True,
    max_contenido: int | None = None,
    contar_contenido: Callable[[str], int] | None = None,
) -> tuple[list[str], list[list[int]]]:
    """R2 · Subordinada adherida: un fragmento que empieza por causal/final/temporal/relativa/completiva
    (``porque, ya que, para que, cuando, que, lo que…``, o ese conector precedido de y/e/o) se une a la
    principal (el fragmento anterior). Las adversativas (``pero``, ``sin embargo``) NO se unen.

    Args:
        respeta_punto: No unir si el anterior cierra con . ? ! y el fragmento empieza en mayúscula.
        max_contenido: Si se pasa (con ``contar_contenido``), solo une la subordinada con MENOS de ese
            número de palabras de contenido; una causal con contenido propio queda separada.
    """
    re_sub = _patron_inicio_frag(subordinantes, con_conjuncion=True)
    salida: list[str] = []
    grupos: list[list[int]] = []
    for k, frag in enumerate(fragmentos):
        cuerpo = frag.lstrip(" ,;:")
        une = bool(salida) and bool(re_sub.match(cuerpo))
        if une and respeta_punto and salida[-1].rstrip()[-1:] in (".", "?", "!") and cuerpo[:1].isupper():
            une = False
        if une and max_contenido is not None and contar_contenido is not None and contar_contenido(cuerpo) >= max_contenido:
            une = False
        if une:
            salida[-1] = _unir_frag(salida[-1], frag); grupos[-1].append(k)
        else:
            salida.append(frag); grupos.append([k])
    return salida, grupos


def r3_protasis(
    fragmentos: Sequence[str],
    unir_protasis: bool = True,
    protasis: Sequence[str] = _PROTASIS,
) -> tuple[list[str], list[list[int]]]:
    """R3 · Prótasis (aunque, si bien, a pesar de que, así no, si…) + apódosis.

    Un fragmento que empieza con conector de prótasis y termina en coma (o cuyo siguiente empieza con
    coma/punto y coma) es la prótasis de la oración siguiente.

    Args:
        unir_protasis: True = una sola unidad (tópico/sentimiento de la apódosis); False = dos unidades,
            sin coma colgante.
    """
    re_prot = _patron_inicio_frag(protasis)
    salida: list[str] = []
    grupos: list[list[int]] = []
    k, n = 0, len(fragmentos)
    while k < n:
        frag = fragmentos[k]
        sigue = fragmentos[k + 1] if k + 1 < n else None
        es_prot = (
            bool(re_prot.match(frag.lstrip(" ,;:")))
            and sigue is not None
            and (frag.rstrip().endswith(",") or sigue.lstrip()[:1] in (",", ";"))
        )
        if es_prot and unir_protasis:
            salida.append(_unir_frag(frag, sigue)); grupos.append([k, k + 1]); k += 2; continue
        if es_prot:
            salida.append(frag.rstrip(" ,;:")); grupos.append([k])
            salida.append(sigue.lstrip(" ,;:")); grupos.append([k + 1]); k += 2; continue
        salida.append(frag); grupos.append([k]); k += 1
    return salida, grupos


def es_adjunto_sin_verbo(doc: Any) -> bool:
    """True si el doc (spaCy) es un adjunto: SIN verbo en forma finita y que inicia con preposición,
    gerundio o participio ("al momento de enseñar", "motivándolos a ir más allá", "por ser clara…").

    Un fragmento sin verbo que NO inicia así ("Muy buena clase") es una opinión válida y no se toca.
    """
    toks = [t for t in doc if not t.is_punct and not t.is_space]
    if not toks:
        return False
    finito = any(t.pos_ in ("VERB", "AUX") and "Fin" in t.morph.get("VerbForm") for t in toks)
    primero = toks[0]
    inicia = primero.pos_ == "ADP" or bool({"Ger", "Part"} & set(primero.morph.get("VerbForm")))
    return (not finito) and inicia


def r4_sin_verbo_finito(
    fragmentos: Sequence[str],
    analizar: Callable[[str], Any],
    respeta_punto: bool = True,
    estricto: bool = True,
) -> tuple[list[str], list[list[int]]]:
    """R4 · Un fragmento sin verbo finito (adjunto preposicional/gerundio/participio) no es cláusula:
    se une al anterior. Se llama a ``analizar`` solo para los fragmentos con anterior (k ≥ 1).

    Args:
        analizar: Función texto -> doc de spaCy (misma morfología que ``_es_advcl_no_finita_adherida_dep``).
        estricto: No unir si el anterior cierra con . ? ! (sin importar si el adjunto empieza en mayúscula).
    """
    salida: list[str] = []
    grupos: list[list[int]] = []
    for k, frag in enumerate(fragmentos):
        une = bool(salida) and es_adjunto_sin_verbo(analizar(frag))
        cierra = bool(salida) and salida[-1].rstrip()[-1:] in (".", "?", "!")
        if une and cierra and (estricto or (respeta_punto and frag.lstrip()[:1].isupper())):
            une = False
        if une:
            salida[-1] = _unir_frag(salida[-1], frag); grupos[-1].append(k)
        else:
            salida.append(frag); grupos.append([k])
    return salida, grupos


def r4b_fragmento_colgante(
    fragmentos: Sequence[str],
    cierres: Sequence[str],
) -> tuple[list[str], list[list[int]]]:
    """R4B · Un fragmento que TERMINA colgando en un adverbio de foco («incluso», «aun», «hasta», «ni siquiera»,
    «sobre todo») está incompleto: se une al SIGUIENTE, sea cual sea su inicio (R4 une hacia atrás; esta,
    hacia delante). No actúa si el fragmento cierra con . ? ! ni si es el último; si el resultado vuelve a
    terminar colgando, sigue uniendo.
    """
    alt = "|".join(re.escape(c) for c in sorted(cierres, key=len, reverse=True))
    re_fin = re.compile(rf"(?:^|\s)(?:{alt})[\s,]*$", re.IGNORECASE)
    salida: list[str] = []
    grupos: list[list[int]] = []
    colgando = False
    for k, frag in enumerate(fragmentos):
        if salida and colgando:
            salida[-1] = _unir_frag(salida[-1], frag); grupos[-1].append(k)
        else:
            salida.append(frag); grupos.append([k])
        colgando = bool(re_fin.search(salida[-1])) and not salida[-1].rstrip().endswith((".", "?", "!"))
    return salida, grupos


def _sin_marcadores_lista(texto: str) -> str:
    """Quita los marcadores de lista ``1)`` para no contarlos como paréntesis."""
    return re.sub(r"(?<![\w()])\d{1,2}\)", "", texto)


def r5_incisos(fragmentos: Sequence[str]) -> tuple[list[str], list[list[int]]]:
    """R5 · Un inciso ``(…)`` o ``-…-`` no abre cláusula: se une al fragmento que interrumpe.

    Se une al anterior cuando el fragmento (a) cierra un paréntesis que abrió el anterior o trae un ``)``
    huérfano, (b) está entero entre paréntesis/guiones, o (c) el anterior termina en un inciso y el
    fragmento sigue en minúscula (continuación de la cláusula interrumpida).
    """
    salida: list[str] = []
    grupos: list[list[int]] = []
    for k, frag in enumerate(fragmentos):
        txt = frag.strip()
        une = False
        if salida:
            previo = salida[-1].rstrip()
            neto = _sin_marcadores_lista(txt)
            previo_neto = _sin_marcadores_lista(previo)
            abierto_previo = previo_neto.count("(") > previo_neto.count(")")
            cierra_huerfano = neto.count(")") > neto.count("(")
            entero = bool(re.fullmatch(r"[(\[].*[)\]][\s.,;]*", txt, flags=re.S)
                          or re.fullmatch(r"[-–—][^-–—]+[-–—][\s.,;]*", txt))
            termina_inciso = bool(re.search(r"[(\[][^()\[\]]*[)\]]$", previo)
                                  or re.search(r"\s[-–—][^-–—]+[-–—]$", previo))
            une = abierto_previo or cierra_huerfano or entero or (termina_inciso and txt[:1].islower())
        if une:
            salida[-1] = _unir_frag(salida[-1], frag); grupos[-1].append(k)
        else:
            salida.append(frag); grupos.append([k])
    return salida, grupos


# ── ft-r4.14: reglas por los comentarios de Julián (11-nov-2026) ──────────────────────────
#: Anáforas tipo pronombre (sin demostrativos adjetivos como «ese/esa/estos/estas»).
_ANAFORAS: tuple[str, ...] = (
    "esto", "eso", "ello", "lo anterior", "lo mismo", "los mismos", "las mismas", "dicho", "dicha", "dichos",
    "dichas", "aquello",
)
#: R13 no actúa si el siguiente empieza con estas palabras (ya los cubren R1/R2/R3).
_NO_UNIR_R13: tuple[str, ...] = ("pero", "sin embargo", "aunque", "porque", "ya que", "que", "y", "e", "o", "si")
#: R7: abreviaturas sin punto; sin «etc», porque «etc.» sí suele cerrar la oración.
_ABREV_R7: tuple[str, ...] = tuple(
    a for a in (*_ABREVIATURAS, "ej", "vs", "pp", "cap", "art", "Ext", "ext") if a.lower() != "etc"
)


def _unir_si_frag(
    fragmentos: Sequence[str],
    condicion: Callable[[str, str], bool],
    unir: Callable[[str, str], str] | None = None,
) -> tuple[list[str], list[list[int]]]:
    """Une cada fragmento con el anterior cuando ``condicion(anterior, actual)``; devuelve ``(fragmentos, grupos)``.
    ``unir`` (por defecto :func:`_unir_frag`, que quita el punto final del anterior) permite conservarlo."""
    unir = unir or _unir_frag
    salida: list[str] = []
    grupos: list[list[int]] = []
    for k, frag in enumerate(fragmentos):
        if salida and condicion(salida[-1], frag):
            salida[-1] = unir(salida[-1], frag); grupos[-1].append(k)
        else:
            salida.append(frag); grupos.append([k])
    return salida, grupos


def r7_abreviatura_final(
    fragmentos: Sequence[str], abreviaturas: Sequence[str] = _ABREV_R7
) -> tuple[list[str], list[list[int]]]:
    """R7 · Un fragmento que TERMINA en abreviatura («…la Ext.», «Dr.», «Av.») y cuyo siguiente empieza en mayúscula
    se une con él: el punto de la abreviatura no cerró la oración («coordinación de la Ext. | Cali con los estudiantes»)."""
    alt = "|".join(re.escape(a.rstrip(".")) for a in sorted(set(abreviaturas), key=len, reverse=True))
    re_ab = re.compile(rf"(?:^|\s)(?:{alt})\.\s*$", re.IGNORECASE)
    return _unir_si_frag(
        fragmentos,
        lambda p, f: bool(re_ab.search(p)) and f.lstrip()[:1].isupper(),
        unir=lambda a, b: a.rstrip() + " " + b.lstrip(),  # conserva el punto de la abreviatura
    )


def r8_pero_tras_punto(fragmentos: Sequence[str]) -> tuple[list[str], list[list[int]]]:
    """R8 · Un fragmento que empieza con «pero» justo después de un cierre ``. ? !`` se une al anterior: ahí «pero» no
    abre otra opinión sino que matiza la frase previa. (Distinto de R1: ``, pero`` con coma sigue separado.)"""
    return _unir_si_frag(
        fragmentos,
        lambda p, f: bool(re.match(r"^\s*pero\b", f, re.IGNORECASE)) and p.rstrip()[-1:] in (".", "?", "!"),
    )


def r9_minuscula_tras_punto_suelto(fragmentos: Sequence[str]) -> tuple[list[str], list[list[int]]]:
    """R9 · Un fragmento que empieza en minúscula y cuyo anterior termina en « .» o «, .» (punto suelto: error de
    puntuación del encuestado) se une al anterior. Un punto normal («estudiar.») seguido de minúscula NO se toca."""
    return _unir_si_frag(
        fragmentos,
        lambda p, f: f.lstrip(" ,;:")[:1].islower() and re.search(r"(?:\s|,)\.\s*$", p.rstrip()) is not None,
    )


def r13_coma_minuscula(
    fragmentos: Sequence[str], excluir_inicio: Sequence[str] = _NO_UNIR_R13
) -> tuple[list[str], list[list[int]]]:
    """R13 · Un fragmento que TERMINA en coma y cuyo siguiente empieza en minúscula está incompleto: se une con él
    («…en el postgrado, | pueda participar , en horario nocturno,»). No actúa si el siguiente empieza con
    conjunción/subordinante/adversativa (``pero, porque, que, y, si…``), que ya cubren R1, R2 y R3."""
    re_ex = _patron_inicio_frag(excluir_inicio)
    return _unir_si_frag(
        fragmentos,
        lambda p, f: p.rstrip().endswith(",") and f.lstrip()[:1].islower() and not re_ex.match(f.lstrip()),
    )


def r10_anafora_inicial(
    fragmentos: Sequence[str], anaforas: Sequence[str] = _ANAFORAS, solo_tras_cierre: bool = True
) -> tuple[list[str], list[list[int]]]:
    """R10 · Un fragmento que empieza con una anáfora («Esto hace que…», «Lo anterior…», «los mismos…») no se entiende
    sin lo dicho antes: se une al anterior.

    Args:
        solo_tras_cierre: Solo une si el anterior termina en ``. ? !`` (frontera real de oración). Sin cierre, el corte
            cae en medio de una oración encadenada sin puntuar y la anáfora abre un pensamiento nuevo.
    """
    re_an = _patron_inicio_frag(anaforas)
    return _unir_si_frag(
        fragmentos,
        lambda p, f: bool(re_an.match(f.lstrip(" ,;:"))) and (not solo_tras_cierre or p.rstrip()[-1:] in (".", "?", "!")),
    )


def r11_numerales(
    fragmentos: Sequence[str], unir_preambulo: bool = False
) -> tuple[list[str], list[list[int]]]:
    """R11 · Una unidad por numeral: si la respuesta trae marcadores consecutivos ``1.``/``1)``, ``2.``/``2)``… (al menos
    2, empezando en 1), cada numeral completo es UNA unidad (se parten y se unen fragmentos para que coincidan con los
    numerales). El texto previo al ``1.`` queda como unidad aparte (o se une al primer numeral con ``unir_preambulo``).
    Ni los decimales (``4.5``) ni los años (``2021.``) cuentan como marcador.

    Returns:
        ``(unidades, grupos)``: ``grupos[k]`` = índices de los fragmentos de entrada que se solapan con la unidad k.
    """
    n = len(fragmentos)
    if n == 0:
        return [], []
    texto, inicios = "", []
    for f in fragmentos:
        if texto:
            texto += " "
        inicios.append(len(texto))
        texto += f
    marcas: list[re.Match[str]] = []
    for m in re.finditer(r"(?<![\w.,])(\d{1,2})\s?[.)](?=\s|$)", texto):
        if int(m.group(1)) == len(marcas) + 1:
            marcas.append(m)
    if len(marcas) < 2:
        return list(fragmentos), [[k] for k in range(n)]
    cortes = [m.start() for m in marcas]
    tramos = [(c, cortes[i + 1] if i + 1 < len(cortes) else len(texto)) for i, c in enumerate(cortes)]
    if texto[: cortes[0]].strip():
        if unir_preambulo:
            tramos[0] = (0, tramos[0][1])
        else:
            tramos.insert(0, (0, cortes[0]))
    fines = [inicios[k] + len(fragmentos[k]) for k in range(n)]
    unidades: list[str] = []
    grupos: list[list[int]] = []
    for a, b in tramos:
        t = texto[a:b].strip()
        if t:
            unidades.append(t)
            grupos.append([k for k in range(n) if inicios[k] < b and fines[k] > a] or [0])
    return unidades, grupos


def r12_unidad_unica(
    fragmentos: Sequence[str], max_palabras: int = 60, guarda_adversativa: bool = False
) -> tuple[list[str], list[list[int]]]:
    """R12 · Una respuesta corta (<= ``max_palabras`` palabras en total), con varios fragmentos y SIN numerales, es UNA
    sola unidad. Se aplica la última, sobre lo que dejaron las demás reglas.

    Args:
        guarda_adversativa: Si True, no unifica respuestas que contengan ``pero / sin embargo / no obstante``.
    """
    n = len(fragmentos)
    texto = " ".join(fragmentos)
    if n < 2 or len(texto.split()) > max_palabras or re.search(r"(?:^|\s)\d{1,2}\s?[.)]\s", texto):
        return list(fragmentos), [[k] for k in range(n)]
    if guarda_adversativa and re.search(r"\b(pero|sin embargo|no obstante)\b", texto, re.IGNORECASE):
        return list(fragmentos), [[k] for k in range(n)]
    unida = fragmentos[0]
    for f in fragmentos[1:]:
        unida = _unir_frag(unida, f)
    return [unida], [list(range(n))]


def r14_tal_como(fragmentos: Sequence[str]) -> tuple[list[str], list[list[int]]]:
    """R14 · Un fragmento que empieza con «tal como» se une al anterior, sin condiciones («…cubrir el programa | tal como
    está en la actualidad»): es un complemento comparativo de lo que acaba de decirse."""
    re_tc = re.compile(r"^\s*tal\s+como\b", re.IGNORECASE)
    return _unir_si_frag(fragmentos, lambda p, f: bool(re_tc.match(f)))


def r15_minuscula_sin_puntuacion(fragmentos: Sequence[str]) -> tuple[list[str], list[list[int]]]:
    """R15 · Un fragmento que empieza en minúscula y cuyo anterior NO termina en signo de puntuación (termina en letra o
    dígito) es la continuación de una oración encadenada sin puntuar: se une al anterior
    («…para ser publicado | realicé los ajustes…»)."""
    return _unir_si_frag(fragmentos, lambda p, f: f.lstrip()[:1].islower() and p.rstrip()[-1:].isalnum())


#: Signos de apertura que, justo antes de la primera palabra, se consideran parte de su unidad.
_APERTURAS_FRAG = ("(", "¿", "¡", "«", "[", "“")


def _es_apertura(token: Any) -> bool:
    """True si ``token`` abre la unidad que sigue: «(», «¿», «¡»…, o una raya pegada a la palabra siguiente («—es decir»)."""
    return token.text in _APERTURAS_FRAG or (token.text in ("—", "–") and token.whitespace_ == "")


def _restaurar_puntuacion_original(
    doc: Any,
    indices: Sequence[Sequence[int]],
    bases: Sequence[list[int] | None],
    textos: list[str],
) -> list[str]:
    """Devuelve cada unidad como su tramo del texto ORIGINAL, con su puntuación original.

    Las unidades se TESELAN: cada una empieza en su primera palabra (más los signos de apertura que la preceden) y termina
    donde empieza la siguiente; la primera arranca en el inicio del texto y la última llega al final. Así los tramos
    cubren todo el texto sin perder ni mover ningún signo, aunque el segmentador haya repartido los signos entre fragmentos.
    ``bases[k]`` = fragmentos base de la unidad k (``None`` si ya no es un tramo contiguo, p. ej. tras R11). Si alguna unidad
    no se puede ubicar, o dos unidades se entrelazan en el original (el segmentador movió texto entre fragmentos), la
    respuesta se deja con los textos de las reglas."""
    ext: list[tuple[int, int]] = []
    for b in bases:
        if b is None:
            return textos
        palabras = [i for k in b for i in indices[k] if not doc[i].is_punct and not doc[i].is_space]
        if not palabras:
            return textos
        a = min(palabras)
        while a > 0 and _es_apertura(doc[a - 1]):
            a -= 1
        ext.append((a, max(palabras)))
    orden = sorted(range(len(ext)), key=lambda k: ext[k])
    for x, y in zip(orden, orden[1:]):
        if ext[y][0] <= ext[x][1]:
            return textos
    original = doc.text
    salida = list(textos)
    for pos, k in enumerate(orden):
        ini_c = 0 if pos == 0 else doc[ext[k][0]].idx
        fin_c = len(original) if pos == len(orden) - 1 else doc[ext[orden[pos + 1]][0]].idx
        salida[k] = original[ini_c:fin_c].strip()
    return salida


def contador_contenido(neutralizador: NeutralizadorAspecto, recursos: RecursosLexicos) -> Callable[[str], int]:
    """Función fragmento -> n.º de palabras de contenido (la misma cuenta de ``NeutralizadorAspecto``),
    con caché. La usa R2 (``Config.r2_max_contenido``)."""
    cache: dict[str, int] = {}

    def contar(fragmento: str) -> int:
        if fragmento not in cache:
            limpio = normalizar_clean(fragmento, recursos)
            cache[fragmento] = neutralizador.neutralizar(fragmento, limpio)[1] if limpio else 0
        return cache[fragmento]

    return contar


def _contar_contenido_aprox(fragmento: str) -> int:
    """Cuenta de respaldo (sin léxicos): palabras alfabéticas de ≥ 3 letras. Solo si el segmentador se usa
    suelto, sin ``construir_unidades`` ni ``crear_segmentador(cfg, recursos)``."""
    return len(re.findall(r"[a-záéíóúñü]{3,}", fragmento.lower()))


class SegmentadorDependencias:
    """Parte una respuesta en cláusulas por análisis de dependencias
    sintácticas de spaCy, en vez de por puntuación + conectores adversativos.

    Misma interfaz que :class:`Segmentador` (``.segmentar(texto) -> list[str]``)
    para conectarse a :func:`construir_unidades` sin más cambios. No usa
    ``negaciones.xlsx`` para decidir dónde cortar: encuentra las cláusulas
    coordinadas/subordinadas directamente en la estructura gramatical de la
    oración (ROOT, advcl, parataxis, conj con sujeto propio o cópula).

    Requiere el modelo de spaCy con parser de dependencias cargado
    (``Config.modelo_spacy_dependencias``, por defecto ``es_core_news_lg``).
    """

    def __init__(
        self,
        cfg: Config,
        nlp_dependencias: Any,
        contar_contenido: Callable[[str], int] | None = None,
    ) -> None:
        """
        Args:
            cfg: Configuración (usa ``max_clausulas_doc`` y las reglas de fragmentos ``reglas_fragmentos``,
                ``r2_max_contenido``, ``r3_unir_protasis``, ``r4_estricto``, ``r4b_cierres``).
            nlp_dependencias: Pipeline de spaCy ya cargado, con parser
                (``spacy.load(cfg.modelo_spacy_dependencias)``).
            contar_contenido: Función fragmento -> n.º de palabras de contenido (para R2). Si es ``None``,
                ``construir_unidades`` / ``crear_segmentador(cfg, recursos)`` la asignan con los léxicos;
                si el segmentador se usa suelto, R2 usa una cuenta aproximada.
        """
        desconocidas = set(cfg.reglas_fragmentos) - set(REGLAS_FRAGMENTOS_ORDEN)
        if desconocidas:
            raise ValueError(
                f"Config.reglas_fragmentos: reglas desconocidas {sorted(desconocidas)}; "
                f"válidas: {list(REGLAS_FRAGMENTOS_ORDEN)}"
            )
        self.cfg = cfg
        self.nlp = nlp_dependencias
        self.contar_contenido = contar_contenido

    def _aplicar_reglas(
        self, doc: Any, fragmentos: list[str], indices: Sequence[Sequence[int]]
    ) -> list[str]:
        """Aplica ``cfg.reglas_fragmentos`` (orden fijo ``REGLAS_FRAGMENTOS_ORDEN``) sobre los fragmentos de spaCy de UNA
        respuesta y, con ``cfg.puntuacion_original``, devuelve cada unidad como su tramo del texto original."""
        activas = set(self.cfg.reglas_fragmentos)
        textos = list(fragmentos)
        if not activas:
            return textos
        cfg = self.cfg
        bases: list[list[int] | None] = [[k] for k in range(len(textos))]
        for nombre in REGLAS_FRAGMENTOS_ORDEN:
            if nombre not in activas or not textos:
                continue
            if nombre == "R1":
                nuevos, grupos = r1_huerfano(textos)
            elif nombre == "R2":
                nuevos, grupos = r2_subordinada_adherida(
                    textos,
                    max_contenido=cfg.r2_max_contenido,
                    contar_contenido=self.contar_contenido or _contar_contenido_aprox,
                )
            elif nombre == "R3":
                nuevos, grupos = r3_protasis(textos, unir_protasis=cfg.r3_unir_protasis)
            elif nombre == "R4":
                # R4 solo analiza los fragmentos con anterior (k ≥ 1): se parsean juntos, en lote.
                docs = dict(zip(textos[1:], self.nlp.pipe(textos[1:]))) if len(textos) > 1 else {}
                nuevos, grupos = r4_sin_verbo_finito(textos, docs.__getitem__, estricto=cfg.r4_estricto)
            elif nombre == "R4B":
                nuevos, grupos = r4b_fragmento_colgante(textos, cfg.r4b_cierres)
            elif nombre == "R5":
                nuevos, grupos = r5_incisos(textos)
            elif nombre == "R11":
                nuevos, grupos = r11_numerales(textos)
            elif nombre == "R7":
                nuevos, grupos = r7_abreviatura_final(textos)
            elif nombre == "R9":
                nuevos, grupos = r9_minuscula_tras_punto_suelto(textos)
            elif nombre == "R13":
                nuevos, grupos = r13_coma_minuscula(textos)
            elif nombre == "R10":
                nuevos, grupos = r10_anafora_inicial(textos)
            elif nombre == "R8":
                nuevos, grupos = r8_pero_tras_punto(textos)
            elif nombre == "R14":
                nuevos, grupos = r14_tal_como(textos)
            elif nombre == "R15":
                nuevos, grupos = r15_minuscula_sin_puntuacion(textos)
            else:  # "R12"
                nuevos, grupos = r12_unidad_unica(
                    textos, max_palabras=cfg.r12_max_palabras, guarda_adversativa=cfg.r12_guarda_adversativa
                )
            bases = [
                None if any(bases[j] is None for j in g) else sorted({b for j in g for b in bases[j]})
                for g in grupos
            ]
            if nombre == "R11" and nuevos != textos:  # las unidades de R11 parten fragmentos: no son un tramo de base
                bases = [None] * len(nuevos)
            textos = nuevos
        if cfg.puntuacion_original:
            textos = _restaurar_puntuacion_original(doc, indices, bases, textos)
        return textos

    def _segmentar_doc(self, doc: Any) -> list[str]:
        """Fragmentos de spaCy de UNA respuesta (sin reglas de reparación)."""
        return self._segmentar_doc_idx(doc)[0]

    def _segmentar_doc_idx(self, doc: Any) -> tuple[list[str], list[list[int]]]:
        """Igual que :meth:`_segmentar_doc` y, además, los índices de token (``token.i``) de cada fragmento: permiten
        recuperar después el tramo del texto ORIGINAL (con su puntuación) que cubre uno o varios fragmentos."""
        fragmentos: list[str] = []
        indices: list[list[int]] = []
        # Fragmento corto del inicio que espera al siguiente.
        prefijo_pendiente: str | None = None
        prefijo_idx: list[int] = []
        for sent in doc.sents:
            idx_sent = list(range(sent.start, sent.end))
            cabezas = {t for t in sent if _es_cabeza_de_clausula_dep(t)}
            if len(cabezas) <= 1:
                # Una sola proposición (aunque tenga perífrasis, relativas o
                # complementos con "que") -> no se parte.
                texto = sent.text.strip()
                tokens_contenido = [t for t in sent if not t.is_punct]
                if len(tokens_contenido) < _LONGITUD_MINIMA_TOKENS_DEP:
                    # Oración degenerada (p. ej. un marcador de lista "4.")
                    # -> se fusiona con el fragmento anterior en vez de
                    # quedar suelta y pasar por los clasificadores como si
                    # fuera una cláusula real. Si no hay anterior,
                    # queda pendiente para pegarse al siguiente.
                    if fragmentos:
                        fragmentos[-1] = f"{fragmentos[-1]} {texto}".strip()
                        indices[-1] = indices[-1] + idx_sent
                    elif texto:
                        prefijo_pendiente = (
                            f"{prefijo_pendiente} {texto}".strip()
                            if prefijo_pendiente else texto
                        )
                        prefijo_idx = prefijo_idx + idx_sent
                    continue
                elif texto:
                    idx = idx_sent
                    if prefijo_pendiente:
                        texto = f"{prefijo_pendiente} {texto}".strip()
                        idx = prefijo_idx + idx_sent
                        prefijo_pendiente, prefijo_idx = None, []
                    fragmentos.append(texto)
                    indices.append(idx)
                continue
            grupos: dict[int, list[Any]] = {cabeza.i: [] for cabeza in cabezas}
            for token in sent:
                ancestro = _cabeza_de_clausula_ancestro_dep(token, cabezas)
                grupos.setdefault(ancestro.i, []).append(token)
            for cabeza in sorted(cabezas, key=lambda c: c.i):
                tokens_grupo = sorted(grupos.get(cabeza.i, []), key=lambda t: t.i)
                idx_grupo = [t.i for t in tokens_grupo]
                tokens_contenido = [t for t in tokens_grupo if not t.is_punct]
                texto_fragmento = _reconstruir_dep(tokens_grupo)
                if len(tokens_contenido) < _LONGITUD_MINIMA_TOKENS_DEP:
                    if fragmentos:
                        fragmentos[-1] = f"{fragmentos[-1]} {texto_fragmento}".strip()
                        indices[-1] = indices[-1] + idx_grupo
                    elif texto_fragmento:
                        idx = idx_grupo
                        if prefijo_pendiente:
                            texto_fragmento = f"{prefijo_pendiente} {texto_fragmento}".strip()
                            idx = prefijo_idx + idx_grupo
                            prefijo_pendiente, prefijo_idx = None, []
                        fragmentos.append(texto_fragmento)
                        indices.append(idx)
                elif texto_fragmento:
                    idx = idx_grupo
                    if prefijo_pendiente:
                        texto_fragmento = f"{prefijo_pendiente} {texto_fragmento}".strip()
                        idx = prefijo_idx + idx_grupo
                        prefijo_pendiente, prefijo_idx = None, []
                    fragmentos.append(texto_fragmento)
                    indices.append(idx)
        # Si la respuesta entera era un fragmento corto, no se pierde.
        if prefijo_pendiente:
            if fragmentos:
                fragmentos[-1] = f"{fragmentos[-1]} {prefijo_pendiente}".strip()
                indices[-1] = indices[-1] + prefijo_idx
            else:
                fragmentos.append(prefijo_pendiente)
                indices.append(prefijo_idx)
        return fragmentos, indices

    def segmentar(self, texto: str) -> list[str]:
        """Cláusulas de una respuesta, en orden de aparición.

        Args:
            texto: Respuesta original (con tildes y puntuación).

        Returns:
            Lista de cláusulas, recortada a ``cfg.max_clausulas_doc``.
        """
        if not isinstance(texto, str) or not texto.strip():
            return []
        doc = self.nlp(texto)
        fragmentos, indices = self._segmentar_doc_idx(doc)
        return self._aplicar_reglas(doc, fragmentos, indices)[: self.cfg.max_clausulas_doc]

    def segmentar_lote(
        self, comentarios: Sequence[str], batch_size: int = 100, n_process: int = 1
    ) -> list[list[str]]:
        """Segmenta muchas respuestas de una vez con ``nlp.pipe`` (más
        rápido que llamar ``segmentar()`` una por una en corpus grandes).
        """
        salida: list[list[str]] = []
        for doc in self.nlp.pipe(comentarios, batch_size=batch_size, n_process=n_process):
            fragmentos, indices = self._segmentar_doc_idx(doc)
            salida.append(self._aplicar_reglas(doc, fragmentos, indices))
        return salida


def crear_segmentador(
    cfg: Config, recursos: RecursosLexicos | None = None
) -> Segmentador | SegmentadorDependencias:
    """Fábrica: instancia el segmentador según ``cfg.metodo_segmentacion``.

    Nota: ``Segmentador`` (conectores) también necesita ``recursos`` -- por
    eso esta fábrica sólo cubre el caso "dependencias" (sin recursos). Para
    "conectores" seguimos instanciando ``Segmentador(cfg, recursos)``
    directamente en :func:`construir_unidades`.

    Args:
        cfg: Configuración con ``metodo_segmentacion`` y
            ``modelo_spacy_dependencias``.
        recursos: Léxicos (opcional). Si se pasan, R2 cuenta las palabras de contenido con el
            ``NeutralizadorAspecto``; ``construir_unidades`` lo asigna igualmente si falta.

    Returns:
        Un :class:`SegmentadorDependencias` listo para usar.

    Raises:
        ValueError: Si ``cfg.metodo_segmentacion`` no es "dependencias".
    """
    if cfg.metodo_segmentacion != "dependencias":
        raise ValueError(
            f"crear_segmentador() sólo cubre 'dependencias'; "
            f"para 'conectores' use Segmentador(cfg, recursos) directamente."
        )
    import spacy  # pylint: disable=import-outside-toplevel

    logger.info(
        "Cargando modelo de dependencias '%s'...", cfg.modelo_spacy_dependencias
    )
    nlp_dependencias = spacy.load(cfg.modelo_spacy_dependencias)
    contar = (
        contador_contenido(NeutralizadorAspecto(cfg, recursos), recursos) if recursos is not None else None
    )
    return SegmentadorDependencias(cfg, nlp_dependencias, contar)


class NeutralizadorAspecto:
    """Construye el TEXTO DE ASPECTO de una cláusula.

    Retira valoración pura, persona, negadores/modificadores y nombres
    propios. Lo que queda es de qué se habla, no cómo se valora:

        "La profe Merly explica muy bien los temas"  ->  "explica los temas"
        "Excelente profesor"                          ->  ""  (global)
        "No es nada dinámico"                         ->  "dinamico"
    """

    def __init__(self, cfg: Config, recursos: RecursosLexicos) -> None:
        self.cfg = cfg
        self.recursos = recursos
        self._retirar = recursos.terminos_neutralizados()
        self._vacias = recursos.stopwords_nltk | recursos.genericos
        self._valorativos_largos = {v for v in recursos.valorativos if len(v) >= 6}
        self._borrados_valorativos = {
            b for v in self._valorativos_largos for b in self._borrados(v)
        }
        self._no_opinion = re.compile(
            "^(?:" + "|".join(PATRONES_NO_OPINION) + ")$", flags=re.IGNORECASE
        )

    @staticmethod
    def _borrados(palabra: str) -> set[str]:
        """Variantes con una letra borrada."""
        return {palabra[:i] + palabra[i + 1 :] for i in range(len(palabra))}

    def es_valorativo_aproximado(self, token: str) -> bool:
        """True si el token es un valorativo largo con una errata."""
        if len(token) < 6 or token in self._vacias:
            return False
        if token in self._borrados_valorativos:  # falta una letra
            return True
        borrados = self._borrados(token)
        return bool(
            borrados & self._valorativos_largos  # sobra una letra
            or borrados & self._borrados_valorativos  # letra cambiada
        )

    def es_no_opinion(self, clean: str) -> bool:
        """True si la cláusula no evalúa nada ("sin comentarios")."""
        return clean in RUIDO or bool(self._no_opinion.match(clean))

    def nombres_propios(self, original: str) -> set[str]:
        """Palabras capitalizadas que no inician la cláusula (forma clean).

        Se omite si la cláusula está casi toda en mayúsculas, donde la
        capitalización no informa nada.
        """
        if not self.cfg.quitar_nombres_propios:
            return set()
        letras = [c for c in original if c.isalpha()]
        if letras and sum(c.isupper() for c in letras) / len(letras) > 0.5:
            return set()
        palabras = re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñÜü]+", original)
        candidatas = {
            normalizar_clean(p, self.recursos)
            for i, p in enumerate(palabras)
            if i > 0 and len(p) >= 3 and p[0].isupper() and p[1:].islower()
        }
        return candidatas - self._vacias

    def neutralizar(self, original: str, clean: str) -> tuple[str, int]:
        """Texto de aspecto y número de palabras de contenido.

        Args:
            original: Cláusula con tildes (para detectar nombres propios).
            clean: Misma cláusula normalizada.

        Returns:
            (texto_aspecto, n_contenido). Texto vacío si no hay contenido.
        """
        tokens = clean.split()
        if not self.cfg.neutralizar_aspecto:
            contenido = [t for t in tokens if t not in self._vacias and len(t) >= 3]
            return clean, len(contenido)
        nombres = self.nombres_propios(original)
        conservados = [
            t
            for t in tokens
            if t not in self._retirar
            and t not in nombres
            and not self.es_valorativo_aproximado(t)
        ]
        # "conocimiento y conocimiento" -> tras neutralizar no se repite.
        conservados = [
            t for i, t in enumerate(conservados) if i == 0 or t != conservados[i - 1]
        ]
        contenido = [t for t in conservados if t not in self._vacias and len(t) >= 3]
        if len(contenido) < self.cfg.min_contenido_aspecto:
            return "", len(contenido)
        # Recorta palabras vacías en los bordes: "de los temas" -> "temas".
        while conservados and conservados[0] in self._vacias:
            conservados.pop(0)
        while conservados and conservados[-1] in self._vacias:
            conservados.pop()
        return " ".join(conservados), len(contenido)


def construir_unidades(
    encuesta: Encuesta,
    cfg: Config,
    recursos: RecursosLexicos,
    *,
    segmentador: Segmentador | SegmentadorDependencias | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabla de documentos y tabla de cláusulas con su motivo.

    Args:
        encuesta: Columna a procesar.
        cfg: Configuración.
        recursos: Léxicos.
        segmentador: Instancia ya creada (para reutilizarla entre
            encuestas, ya que cargar el modelo de spaCy de dependencias es
            costoso). Si es ``None``, se crea una según
            ``cfg.metodo_segmentacion``.

    Returns:
        ``(documentos, clausulas)``. ``documentos`` tiene una fila por fila
        del corpus con su estado y, si ``encuesta.columnas_llave`` no está
        vacío, las columnas de llave compuesta para trazabilidad.
        ``clausulas`` una por cláusula con ``clausula``, ``clausula_clean``,
        ``texto_aspecto``, ``n_contenido`` y ``motivo``.
    """
    if segmentador is None:
        segmentador = (
            crear_segmentador(cfg)
            if cfg.metodo_segmentacion == "dependencias"
            else Segmentador(cfg, recursos)
        )
    neutralizador = NeutralizadorAspecto(cfg, recursos)
    # R2 (reglas de fragmentos) cuenta palabras de contenido con ESTE neutralizador, si nadie le dio otro contador.
    if isinstance(segmentador, SegmentadorDependencias) and segmentador.contar_contenido is None:
        segmentador.contar_contenido = contador_contenido(neutralizador, recursos)
    filas_doc: list[dict[str, Any]] = []
    filas_cla: list[dict[str, Any]] = []
    columnas_llave = list(getattr(encuesta, "columnas_llave", []) or [])

    def _llaves_de(doc_id: int) -> dict[str, Any]:
        """Valores de llave compuesta de la fila `doc_id`, para trazabilidad."""
        if not columnas_llave:
            return {}
        fila = encuesta.marco.iloc[doc_id]
        return {col: fila[col] for col in columnas_llave if col in encuesta.marco.columns}

    for doc_id, texto in enumerate(encuesta.texto_original):
        llaves_fila = _llaves_de(doc_id)
        clean_doc = normalizar_clean(texto, recursos)
        if not clean_doc:
            filas_doc.append(
                {
                    "doc_id": doc_id,
                    "estado": DOC_VACIA,
                    "n_clausulas": 0,
                    **llaves_fila,
                }
            )
            continue
        if clean_doc in RUIDO:
            filas_doc.append(
                {
                    "doc_id": doc_id,
                    "estado": DOC_RUIDO,
                    "n_clausulas": 0,
                    **llaves_fila,
                }
            )
            continue

        motivos: list[str] = []
        for clausula_id, clausula in enumerate(segmentador.segmentar(texto)):
            clean = normalizar_clean(clausula, recursos)
            if not clean or clean in RUIDO:
                motivo, aspecto, contenido = MOTIVO_RUIDO, "", 0
            elif neutralizador.es_no_opinion(clean):
                motivo, aspecto, contenido = MOTIVO_NO_OPINION, "", 0
            else:
                aspecto, contenido = neutralizador.neutralizar(clausula, clean)
                motivo = MOTIVO_ASPECTO if aspecto else MOTIVO_GLOBAL
            motivos.append(motivo)
            filas_cla.append(
                {
                    "doc_id": doc_id,
                    "clausula_id": clausula_id,
                    "clausula": clausula,
                    "clausula_clean": clean,
                    "texto_aspecto": aspecto,
                    "n_contenido": contenido,
                    "motivo": motivo,
                }
            )
        opinion = any(m in (MOTIVO_ASPECTO, MOTIVO_GLOBAL) for m in motivos)
        filas_doc.append(
            {
                "doc_id": doc_id,
                "estado": DOC_CON_OPINION if opinion else DOC_NO_OPINION,
                "n_clausulas": len(motivos),
                **llaves_fila,
            }
        )
    documentos = pd.DataFrame(filas_doc)
    columnas = [
        "doc_id",
        "clausula_id",
        "clausula",
        "clausula_clean",
        "texto_aspecto",
        "n_contenido",
        "motivo",
    ]
    clausulas = pd.DataFrame(filas_cla, columns=columnas)
    resumen = clausulas["motivo"].value_counts().to_dict()
    logger.info(
        "  unidades: %d filas -> %s | %d cláusulas -> %s",
        len(documentos),
        documentos["estado"].value_counts().to_dict(),
        len(clausulas),
        resumen,
    )
    return documentos, clausulas


# ═══════════════════════════════════════════════════════════════════════════
# 7. ENCODER (inyectable)
# ═══════════════════════════════════════════════════════════════════════════


class Codificador(Protocol):
    """Cualquier función texto -> matriz L2-normalizada (n, d)."""

    def __call__(self, textos: Sequence[str]) -> np.ndarray:
        """Codifica una secuencia de textos."""


#: Desde cuántos textos una codificación se reporta en INFO con barra.
_LOTE_GRANDE_ENCODER = 1000


class EncoderSentenceTransformers:
    """Encoder real con caché en disco por hash COMPLETO del contenido.

    El MVP 2 hasheaba una muestra de 500 textos: dos listas distintas con
    la misma muestra podían recuperar embeddings ajenos.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._modelo: Any = None

    def _cargar(self) -> Any:
        """Carga perezosa del modelo (una sola vez por instancia)."""
        if self._modelo is None:
            # pylint: disable-next=import-outside-toplevel
            from sentence_transformers import SentenceTransformer

            logger.info("  cargando encoder '%s'...", self.cfg.encoder)
            self._modelo = SentenceTransformer(self.cfg.encoder)
        return self._modelo

    def __call__(self, textos: Sequence[str]) -> np.ndarray:
        """Codifica con caché; normaliza para que coseno == producto punto."""
        lista = [str(t) for t in textos]
        ruta = Path(self.cfg.dir_cache) / (
            f"emb_{huella_texto(self.cfg.encoder, lista)[:24]}.npy"
        )
        grande = len(lista) >= _LOTE_GRANDE_ENCODER
        nivel = logging.INFO if grande else logging.DEBUG
        if self.cfg.usar_cache and ruta.exists():
            logger.log(nivel, "  embeddings recuperados de caché (%d textos)", len(lista))
            return np.load(ruta)
        logger.log(nivel, "  codificando %d textos...", len(lista))
        matriz = np.asarray(
            self._cargar().encode(
                lista,
                batch_size=self.cfg.batch_size,
                show_progress_bar=grande,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )
        if self.cfg.usar_cache:
            ruta.parent.mkdir(parents=True, exist_ok=True)
            np.save(ruta, matriz)
        return matriz


def _adaptador_bertopic(codificador: Codificador) -> Any:
    """Envuelve un :class:`Codificador` como backend de BERTopic (para MMR)."""
    from bertopic.backend import BaseEmbedder  # pylint: disable=import-outside-toplevel

    class _Adaptador(BaseEmbedder):  # pylint: disable=too-few-public-methods
        """Backend mínimo: delega en el codificador del pipeline."""

        def embed(self, documents: list[str], verbose: bool = False) -> np.ndarray:
            del verbose
            return np.asarray(codificador(documents))

    return _Adaptador()


# ═══════════════════════════════════════════════════════════════════════════
# 8. PARTICIÓN: PIEZAS COMPARTIDAS POR GRID Y MODELO FINAL
# ═══════════════════════════════════════════════════════════════════════════


def proyectar_umap(
    embeddings: np.ndarray,
    n_neighbors: int,
    n_components: int,
    cfg: Config,
    semilla: int | None = None,
) -> np.ndarray:
    """Proyección UMAP determinista (semilla fija)."""
    from umap import UMAP  # pylint: disable=import-outside-toplevel

    vecinos = min(n_neighbors, max(2, len(embeddings) - 1))
    return UMAP(
        n_neighbors=vecinos,
        n_components=n_components,
        min_dist=cfg.umap_min_dist,
        metric="cosine",
        random_state=cfg.random_state if semilla is None else semilla,
    ).fit_transform(embeddings)


def agrupar_hdbscan(
    reducidos: np.ndarray,
    min_topic_size: int,
    min_samples: int,
    metodo: str,
    epsilon: float = 0.0,
) -> np.ndarray:
    """Etiquetas HDBSCAN respetando el método de selección pedido."""
    from hdbscan import HDBSCAN  # pylint: disable=import-outside-toplevel

    return HDBSCAN(
        min_cluster_size=int(min_topic_size),
        min_samples=int(min_samples),
        metric="euclidean",
        cluster_selection_method=metodo,
        cluster_selection_epsilon=float(epsilon),
    ).fit_predict(reducidos)


@dataclass
class ModeloResidual:
    """UMAP y HDBSCAN del residual, ajustados una vez con ``prediction_data=True`` (contrato 5.2, «Estabilidad»).

    Permite asignar respuestas NUEVAS a los grupos ya descubiertos con ``approximate_predict``, sin reajustar.
    ``mapa_final`` lleva cada grupo crudo de HDBSCAN al tópico final (tras la fusión y la reasignación de outliers).
    """

    umap: Any
    hdbscan: Any
    mapa_final: dict[int, int]
    config: dict[str, Any]
    semilla: int
    n_textos: int
    #: Pertenencia al grupo de cada texto único del ajuste (NaN en los que HDBSCAN dejó fuera).
    probabilidades: np.ndarray = field(default_factory=lambda: np.zeros(0))
    version: str = VERSION

    def asignar(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tópico final y probabilidad de pertenencia de embeddings nuevos (sin reajustar).

        Returns:
            ``(topicos, probabilidades)``; ``-1`` (outlier) con probabilidad 0 si HDBSCAN no los asigna.
        """
        from hdbscan import approximate_predict  # pylint: disable=import-outside-toplevel

        reducidos = self.umap.transform(np.asarray(embeddings))
        crudas, probs = approximate_predict(self.hdbscan, reducidos)
        finales = np.array([self.mapa_final.get(int(g), OUTLIER) if int(g) >= 0 else OUTLIER for g in crudas], dtype=int)
        return finales, np.asarray(probs, dtype=float)


def ajustar_residual_persistible(modelador: ModeladorAspectos, cfg: Config) -> ModeloResidual:
    """Reajusta UMAP y HDBSCAN del modelo final del residual con los MISMOS parámetros y semilla y ``prediction_data=True``.

    No cambia la partición ya calculada: comprueba que el reajuste la reproduce y avisa si no (UMAP no determinista).
    """
    from hdbscan import HDBSCAN  # pylint: disable=import-outside-toplevel
    from umap import UMAP  # pylint: disable=import-outside-toplevel

    emb = np.asarray(modelador.embeddings)
    vecinos = min(cfg.umap_n_neighbors, max(2, len(emb) - 1))
    reductor = UMAP(
        n_neighbors=vecinos, n_components=cfg.umap_n_components, min_dist=cfg.umap_min_dist,
        metric="cosine", random_state=cfg.random_state,
    ).fit(emb)
    reducidos = np.asarray(reductor.embedding_)
    agrupador = HDBSCAN(
        min_cluster_size=int(cfg.min_topic_size), min_samples=int(cfg.min_samples), metric="euclidean",
        cluster_selection_method=cfg.cluster_selection_method, cluster_selection_epsilon=float(cfg.cluster_selection_epsilon),
        prediction_data=True,
    ).fit(reducidos)
    crudas = np.asarray(agrupador.labels_, dtype=int)
    if not np.array_equal(renumerar(crudas), np.asarray(modelador.etiquetas_hdbscan)):
        logger.warning("  residual: el reajuste de UMAP/HDBSCAN no reproduce la partición original; revise la semilla.")
    finales = np.asarray(modelador.etiquetas, dtype=int)
    mapa = {int(g): int(pd.Series(finales[crudas == g]).mode().iloc[0]) for g in np.unique(crudas) if g >= 0}
    probs = np.where(crudas >= 0, np.asarray(agrupador.probabilities_, dtype=float), np.nan)
    return ModeloResidual(
        umap=reductor, hdbscan=agrupador, mapa_final=mapa,
        config={"umap_n_neighbors": cfg.umap_n_neighbors, "umap_n_components": cfg.umap_n_components,
                "umap_min_dist": cfg.umap_min_dist, "min_topic_size": cfg.min_topic_size, "min_samples": cfg.min_samples,
                "metodo": cfg.cluster_selection_method, "epsilon": cfg.cluster_selection_epsilon},
        semilla=int(cfg.random_state), n_textos=len(emb), probabilidades=probs,
    )


def guardar_modelo_residual(modelo: ModeloResidual, carpeta: Path | str) -> Path:
    """Guarda UMAP + HDBSCAN (``prediction_data=True``) con su versión en ``carpeta/residual_umap_hdbscan.joblib``."""
    import joblib  # pylint: disable=import-outside-toplevel

    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / "residual_umap_hdbscan.joblib"
    joblib.dump(modelo, ruta)
    return ruta


def cargar_modelo_residual(carpeta: Path | str) -> ModeloResidual:
    """Carga lo guardado por :func:`guardar_modelo_residual`."""
    import joblib  # pylint: disable=import-outside-toplevel

    return joblib.load(Path(carpeta) / "residual_umap_hdbscan.joblib")


def renumerar(etiquetas: np.ndarray) -> np.ndarray:
    """Tópicos 0..k-1 ordenados por tamaño descendente; -1 se conserva."""
    etiquetas = np.asarray(etiquetas, dtype=int)
    tamanos = pd.Series(etiquetas[etiquetas >= 0]).value_counts()
    mapa = {int(t): i for i, t in enumerate(tamanos.index)}
    return np.array([mapa.get(int(e), OUTLIER) for e in etiquetas], dtype=int)


def centroides(embeddings: np.ndarray, etiquetas: np.ndarray) -> dict[int, np.ndarray]:
    """Centroide L2-normalizado de cada tópico en el espacio del encoder."""
    salida: dict[int, np.ndarray] = {}
    for topico in sorted(set(etiquetas.tolist()) - {OUTLIER}):
        vector = embeddings[etiquetas == topico].mean(axis=0)
        salida[int(topico)] = vector / max(float(np.linalg.norm(vector)), 1e-12)
    return salida


def _fusionar_una_ronda(
    embeddings: np.ndarray, etiquetas: np.ndarray, umbral: float
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """UNA pasada de fusión (es la función original ``fusionar_redundantes``).

    Fusiona tópicos cuyos centroides superan el coseno ``umbral``.

    Returns:
        (etiquetas renumeradas, pares fusionados con su coseno).
    """
    cents = centroides(embeddings, etiquetas)
    ids = list(cents)
    if len(ids) < 2:
        return renumerar(etiquetas), []
    matriz = np.stack([cents[i] for i in ids])
    similitud = matriz @ matriz.T
    padre = {i: i for i in ids}

    def raiz(nodo: int) -> int:
        while padre[nodo] != nodo:
            padre[nodo] = padre[padre[nodo]]
            nodo = padre[nodo]
        return nodo

    pares: list[tuple[int, int, float]] = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            if similitud[a, b] >= umbral:
                pares.append((ids[a], ids[b], round(float(similitud[a, b]), 4)))
                padre[raiz(ids[b])] = raiz(ids[a])
    if not pares:
        return renumerar(etiquetas), []
    fusionadas = np.array(
        [raiz(int(e)) if e >= 0 else OUTLIER for e in etiquetas], dtype=int
    )
    return renumerar(fusionadas), pares


#: Máximo de pasadas de fusión (en la práctica converge en 2-3).
_MAX_RONDAS_FUSION = 10


def fusionar_redundantes(
    embeddings: np.ndarray, etiquetas: np.ndarray, umbral: float
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """Fusiona tópicos cuyos centroides superan el coseno ``umbral``, hasta
    que ningún par de tópicos RESULTANTES lo supere.

    Antes era una sola pasada sobre los grupos originales de
    HDBSCAN: si A se unía con B, el centroide de A∪B no se volvía a comparar.
    En obs19 eso dejó T0 y T4 con coseno 0,896 (> 0,85) sin
    fusionar, y tras la reasignación llegaron a 0,905 (falla G7). Ahora se
    repite la pasada hasta que no haya pares nuevos. El grid usa esta misma
    función, así que una fusión en cascada que concentre demasiado queda
    penalizada por la puerta G4 y esa configuración no se elige.

    Returns:
        (etiquetas renumeradas, pares fusionados con su coseno, de todas las
        pasadas; los ids de cada par son los de su pasada).
    """
    actuales = np.asarray(etiquetas)
    todos: list[tuple[int, int, float]] = []
    for _ in range(_MAX_RONDAS_FUSION):
        actuales, pares = _fusionar_una_ronda(embeddings, actuales, umbral)
        if not pares:
            break
        todos.extend(pares)
    return actuales, todos


def crear_fusion_raiz_coseno(
    textos: Sequence[str], stopwords: Iterable[str], cfg: Config
) -> Callable[
    [np.ndarray, np.ndarray, float], tuple[np.ndarray, list[tuple[int, int, float]]]
]:
    """Política de fusión "raiz_coseno": devuelve una función con la MISMA firma
    que :func:`fusionar_redundantes`.

    Dos tópicos son fusionables si su coseno supera ``umbral`` o si supera
    ``cfg.fusion_umbral_raiz`` y comparten raíz (SnowballStemmer) en sus
    ``cfg.fusion_top_terminos`` términos c-TF-IDF. Se fusiona de mayor a menor
    coseno, sin cadenas (todo par entre los dos grupos debe ser fusionable) y sin
    que el grupo supere ``cfg.fusion_tope_pct`` % de las cláusulas con tópico.
    Una sola pasada, a diferencia de la política original.

    Requiere ``nltk`` y ``scikit-learn`` (importados aquí, solo si se usa).

    Args:
        textos: Textos de aspecto únicos, alineados con los embeddings.
        stopwords: Stopwords del vectorizador de la encuesta.
        cfg: Configuración con los parámetros ``fusion_*``.
    """
    import warnings  # pylint: disable=import-outside-toplevel

    from nltk.stem.snowball import (  # pylint: disable=import-outside-toplevel
        SnowballStemmer,
    )
    from sklearn.feature_extraction.text import (  # pylint: disable=import-outside-toplevel
        CountVectorizer,
    )

    textos = [str(t) for t in textos]
    stopwords = list(stopwords)
    raices_excluidas = tuple(cfg.fusion_raices_excluidas)
    stem = SnowballStemmer("spanish")

    def raices_por_topico(etiquetas: np.ndarray, ids: list[int]) -> dict[int, set[str]]:
        docs = [" ".join(textos[i] for i in np.where(etiquetas == t)[0]) for t in ids]
        cv = CountVectorizer(ngram_range=(1, 2), stop_words=stopwords)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            conteo = cv.fit_transform(docs).toarray().astype(float)
        vocab = cv.get_feature_names_out()
        peso = (conteo / np.maximum(conteo.sum(1, keepdims=True), 1)) * np.log(
            1 + (conteo.sum() / len(ids)) / np.maximum(conteo.sum(0), 1)
        )
        salida = {}
        for fila, t in enumerate(ids):
            top = [
                vocab[j]
                for j in np.argsort(-peso[fila])[: cfg.fusion_top_terminos]
                if peso[fila, j] > 0
            ]
            raices = {stem.stem(w) for s in top for w in s.split()}
            salida[t] = {
                r for r in raices if len(r) >= 4 and not r.startswith(raices_excluidas)
            }
        return salida

    def comparten(ra: set[str], rb: set[str]) -> bool:
        return any(
            x == y or (min(len(x), len(y)) >= 5 and (x.startswith(y) or y.startswith(x)))
            for x in ra
            for y in rb
        )

    def fusionar(
        embeddings: np.ndarray, etiquetas: np.ndarray, umbral: float
    ) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
        etiquetas = np.asarray(etiquetas)
        assert len(textos) == len(
            etiquetas
        ), "Los textos de la fusión no coinciden con los embeddings"
        cents = centroides(embeddings, etiquetas)
        ids = sorted(cents)
        if len(ids) < 2:
            return renumerar(etiquetas), []
        raices = raices_por_topico(etiquetas, ids)
        tam = {t: int((etiquetas == t).sum()) for t in ids}
        total = sum(tam.values())
        sim = {
            (a, b): float(cents[a] @ cents[b])
            for i, a in enumerate(ids)
            for b in ids[i + 1 :]
        }
        apto = {
            p: s >= umbral
            or (s >= cfg.fusion_umbral_raiz and comparten(raices[p[0]], raices[p[1]]))
            for p, s in sim.items()
        }
        grupos = {t: {t} for t in ids}
        dueno = {t: t for t in ids}
        pares: list[tuple[int, int, float]] = []
        for (a, b), s in sorted(sim.items(), key=lambda kv: -kv[1]):
            ga, gb = dueno[a], dueno[b]
            if ga == gb or not apto[(a, b)]:
                continue
            union = grupos[ga] | grupos[gb]
            sin_cadena = all(
                apto[(min(x, y), max(x, y))] for x in grupos[ga] for y in grupos[gb]
            )
            if (
                sin_cadena
                and 100 * sum(tam[t] for t in union) / total <= cfg.fusion_tope_pct
            ):
                grupos[ga] = union
                for t in grupos.pop(gb):
                    dueno[t] = ga
                pares.append((int(a), int(b), round(s, 4)))
        nuevas = etiquetas.copy()
        for t in ids:
            nuevas[etiquetas == t] = dueno[t]
        return renumerar(nuevas), pares

    return fusionar


def fusionar_segun_politica(
    cfg: Config,
    recursos: RecursosLexicos,
    clave_stopwords: str | None,
    textos: Sequence[str],
    embeddings: np.ndarray,
    etiquetas: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """Fusiona tópicos según ``cfg.politica_fusion`` (grid y modelo final).

    Args:
        cfg: Configuración (``politica_fusion``, ``umbral_fusion``, ``fusion_*``).
        recursos: Léxicos (stopwords del vectorizador, solo "raiz_coseno").
        clave_stopwords: Clave de encuesta para esas stopwords.
        textos: Textos de aspecto únicos alineados con ``embeddings``.
        embeddings: Matriz normalizada (n, d).
        etiquetas: Partición con -1.

    Returns:
        (etiquetas renumeradas, pares fusionados con su coseno).

    Raises:
        ValueError: Si ``cfg.politica_fusion`` no es una política conocida.
    """
    if cfg.politica_fusion == "original":
        return fusionar_redundantes(embeddings, etiquetas, cfg.umbral_fusion)
    if cfg.politica_fusion == "raiz_coseno":
        fusion = crear_fusion_raiz_coseno(
            textos, recursos.stopwords_vectorizador(clave_stopwords), cfg
        )
        return fusion(embeddings, etiquetas, cfg.umbral_fusion)
    raise ValueError(
        f"politica_fusion desconocida: {cfg.politica_fusion!r} "
        "(use 'original' o 'raiz_coseno')."
    )


def reasignar_outliers(
    embeddings: np.ndarray, etiquetas: np.ndarray, umbral: float, tope: float
) -> tuple[np.ndarray, dict[int, float]]:
    """Reasigna outliers al centroide más similar, con umbral y cupo.

    Los candidatos se procesan de MAYOR a menor similitud, de modo que el
    cupo de cada tópico lo ocupan las cláusulas que mejor encajan (el MVP 2
    los aceptaba en orden de aparición). Es la MISMA función en el grid y
    en el modelo final.

    Args:
        embeddings: Matriz normalizada (n, d).
        etiquetas: Partición con -1.
        umbral: Coseno mínimo para aceptar.
        tope: Crecimiento máximo por tópico (fracción de su tamaño).

    Returns:
        (nuevas etiquetas, crecimiento % por tópico).
    """
    etiquetas = np.asarray(etiquetas, dtype=int).copy()
    cents = centroides(embeddings, etiquetas)
    if not cents or not (etiquetas == OUTLIER).any():
        return etiquetas, dict.fromkeys(cents, 0.0)
    ids = np.array(list(cents))
    matriz = np.stack([cents[i] for i in ids])
    idx_out = np.where(etiquetas == OUTLIER)[0]
    similitud = embeddings[idx_out] @ matriz.T
    mejor = similitud.argmax(axis=1)
    valor = similitud[np.arange(len(idx_out)), mejor]
    tamano = pd.Series(etiquetas[etiquetas >= 0]).value_counts().to_dict()
    cupo = {int(t): int(round(n * tope)) for t, n in tamano.items()}
    usado = dict.fromkeys(cupo, 0)
    for posicion in np.argsort(-valor):
        if valor[posicion] < umbral:
            break
        destino = int(ids[mejor[posicion]])
        if usado[destino] < cupo[destino]:
            usado[destino] += 1
            etiquetas[idx_out[posicion]] = destino
    crecimiento = {
        int(t): round(100 * usado[int(t)] / max(n, 1), 1) for t, n in tamano.items()
    }
    return etiquetas, crecimiento


def concentracion(etiquetas: np.ndarray) -> dict[str, float]:
    """Masa en los tópicos mayores y Gini de tamaños (ignora códigos < 0)."""
    asignados = np.asarray(etiquetas)
    asignados = asignados[asignados >= 0]
    if asignados.size == 0:
        return {"pct_top1": 100.0, "pct_top2": 100.0, "pct_top3": 100.0, "gini": 1.0}
    tam = np.sort(pd.Series(asignados).value_counts().to_numpy())[::-1]
    total = float(tam.sum())
    orden = np.sort(tam)
    n = orden.size
    gini = float(
        (2 * (np.arange(1, n + 1) * orden).sum()) / (n * orden.sum()) - (n + 1) / n
    )
    return {
        "pct_top1": round(100 * float(tam[:1].sum()) / total, 2),
        "pct_top2": round(100 * float(tam[:2].sum()) / total, 2),
        "pct_top3": round(100 * float(tam[:3].sum()) / total, 2),
        "gini": round(gini, 3),
    }


def silueta(
    reducidos: np.ndarray, etiquetas: np.ndarray, maximo: int, semilla: int
) -> float:
    """Silueta en el espacio UMAP sobre una muestra de puntos asignados."""
    from sklearn.metrics import (
        silhouette_score,
    )  # pylint: disable=import-outside-toplevel

    mascara = np.asarray(etiquetas) >= 0
    if len(set(np.asarray(etiquetas)[mascara].tolist())) < 2:
        return float("nan")
    x, y = reducidos[mascara], np.asarray(etiquetas)[mascara]
    if len(y) > maximo:
        elegidos = np.random.default_rng(semilla).choice(len(y), maximo, replace=False)
        x, y = x[elegidos], y[elegidos]
    return round(float(silhouette_score(x, y, metric="euclidean")), 4)


def max_coseno_centroides(embeddings: np.ndarray, etiquetas: np.ndarray) -> float:
    """Mayor coseno entre centroides distintos (redundancia residual)."""
    cents = centroides(embeddings, etiquetas)
    if len(cents) < 2:
        return float("nan")
    matriz = np.stack(list(cents.values()))
    similitud = matriz @ matriz.T
    np.fill_diagonal(similitud, -1.0)
    return round(float(similitud.max()), 4)


# ═══════════════════════════════════════════════════════════════════════════
# 9. RAMA A — MODELADO DE ASPECTOS SOBRE CLÁUSULAS
# ═══════════════════════════════════════════════════════════════════════════


class ModeladorAspectos:  # pylint: disable=too-many-instance-attributes
    """UMAP -> HDBSCAN -> fusión -> reasignación -> c-TF-IDF sobre cláusulas.

    La partición se calcula con las piezas de la sección 8 (idénticas al
    grid) y BERTopic se usa en modo MANUAL sólo para representar: recibe
    las etiquetas ya decididas y produce términos c-TF-IDF + MMR.
    """

    def __init__(
        self,
        cfg: Config,
        recursos: RecursosLexicos,
        codificador: Codificador | None = None,
        clave_stopwords: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.recursos = recursos
        self.codificador: Codificador = codificador or EncoderSentenceTransformers(cfg)
        self.clave_stopwords = clave_stopwords
        self.modelo: Any = None
        self.textos: list[str] = []  # textos de aspecto ÚNICOS
        self.embeddings: np.ndarray | None = None
        self.reducidos: np.ndarray | None = None
        self.etiquetas_hdbscan: np.ndarray | None = None
        self.etiquetas: np.ndarray | None = None  # partición final (únicos)
        self.pares_fusionados: list[tuple[int, int, float]] = []
        self.crecimiento: dict[int, float] = {}
        self.diagnostico: dict[str, Any] = {}

    # ── Preparación ────────────────────────────────────────────────────────

    def preparar(self, textos_aspecto: Sequence[str]) -> np.ndarray:
        """Deduplica y codifica. Devuelve el mapa cláusula -> único."""
        codigos, unicos = pd.factorize(pd.Series(list(textos_aspecto)), sort=False)
        self.textos = [str(t) for t in unicos]
        self.embeddings = np.asarray(self.codificador(self.textos), dtype=np.float32)
        logger.info(
            "  aspectos: %d cláusulas -> %d textos únicos (%.1f%% duplicados)",
            len(codigos),
            len(self.textos),
            100 * (1 - len(self.textos) / max(len(codigos), 1)),
        )
        return np.asarray(codigos, dtype=int)

    # ── Partición ──────────────────────────────────────────────────────────

    def particionar(self, reducidos: np.ndarray | None = None) -> np.ndarray:
        """Calcula la partición final sobre los textos únicos.

        Args:
            reducidos: Proyección UMAP ya calculada (p. ej. por el grid)
                para garantizar identidad exacta con la fila elegida.

        Returns:
            Etiquetas finales de los textos únicos.
        """
        if self.embeddings is None:
            raise RuntimeError("Llame a preparar() antes de particionar().")
        cfg = self.cfg
        self.reducidos = (
            reducidos
            if reducidos is not None
            else proyectar_umap(
                self.embeddings, cfg.umap_n_neighbors, cfg.umap_n_components, cfg
            )
        )
        crudas = agrupar_hdbscan(
            self.reducidos,
            cfg.min_topic_size,
            cfg.min_samples,
            cfg.cluster_selection_method,
            cfg.cluster_selection_epsilon,
        )
        self.etiquetas_hdbscan = renumerar(crudas)
        fusionadas, self.pares_fusionados = fusionar_segun_politica(
            cfg,
            self.recursos,
            self.clave_stopwords,
            self.textos,
            self.embeddings,
            self.etiquetas_hdbscan,
        )
        self.etiquetas, self.crecimiento = reasignar_outliers(
            self.embeddings, fusionadas, cfg.umbral_outliers, cfg.tope_crecimiento
        )
        self.diagnostico = {
            "n_topicos_hdbscan": int(self.etiquetas_hdbscan.max() + 1),
            "pct_outliers_hdbscan": round(
                100 * float((self.etiquetas_hdbscan == OUTLIER).mean()), 2
            ),
            "pares_fusionados": len(self.pares_fusionados),
            "silueta": silueta(
                self.reducidos,
                self.etiquetas_hdbscan,
                cfg.silueta_muestra_max,
                cfg.random_state,
            ),
            "huella_particion": huella_texto(self.etiquetas_hdbscan.tolist())[:16],
        }
        logger.info(
            "  partición: HDBSCAN %d tópicos / %.1f%% outliers -> %d fusiones -> "
            "final %d tópicos / %.1f%% outliers | silueta %.3f",
            self.diagnostico["n_topicos_hdbscan"],
            self.diagnostico["pct_outliers_hdbscan"],
            len(self.pares_fusionados),
            int(self.etiquetas.max() + 1) if (self.etiquetas >= 0).any() else 0,
            100 * float((self.etiquetas == OUTLIER).mean()),
            self.diagnostico["silueta"],
        )
        return self.etiquetas

    # ── Representación ─────────────────────────────────────────────────────

    def representar(self) -> Any:
        """BERTopic manual: c-TF-IDF + MMR sobre la partición ya fijada."""
        # pylint: disable=import-outside-toplevel
        from bertopic import BERTopic
        from bertopic.cluster import BaseCluster
        from bertopic.dimensionality import BaseDimensionalityReduction
        from bertopic.representation import MaximalMarginalRelevance
        from bertopic.vectorizers import ClassTfidfTransformer
        from sklearn.feature_extraction.text import CountVectorizer

        if self.etiquetas is None or self.embeddings is None:
            raise RuntimeError("Llame a particionar() antes de representar().")
        self.modelo = BERTopic(
            embedding_model=_adaptador_bertopic(self.codificador),
            umap_model=BaseDimensionalityReduction(),
            hdbscan_model=BaseCluster(),
            vectorizer_model=CountVectorizer(
                stop_words=self.recursos.stopwords_vectorizador(self.clave_stopwords),
                ngram_range=self.cfg.ngram_range,
                token_pattern=r"(?u)\b[a-zñ]{3,}\b",
                min_df=1,
                max_df=1.0,
            ),
            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
            representation_model=MaximalMarginalRelevance(
                diversity=self.cfg.mmr_diversity
            ),
            top_n_words=max(30, self.cfg.top_n_words),
            calculate_probabilities=False,
            verbose=False,
        )
        self.modelo.fit_transform(
            self.textos, embeddings=self.embeddings, y=self.etiquetas.tolist()
        )
        cruce = pd.crosstab(self.etiquetas, np.asarray(self.modelo.topics_))
        self._mapa_modelo = {
            int(t): int(cruce.loc[t].idxmax()) for t in cruce.index if int(t) >= 0
        }
        puro = all(
            int(cruce.loc[t].max()) == int(cruce.loc[t].sum()) for t in cruce.index
        )
        self.diagnostico["representacion_alineada"] = bool(
            puro and len(set(self._mapa_modelo.values())) == len(self._mapa_modelo)
        )
        return self.modelo

    def terminos(self, topico: int, n: int = 8) -> list[str]:
        """Términos representativos de NUESTRO tópico ``topico``.

        Traduce el id al de BERTopic y filtra candidatos c-TF-IDF que no
        aparecen en al menos ``df_min_termino`` de las cláusulas del tópico
        o que repiten palabra ("interesante interesante").
        """
        if self.modelo is None or topico < 0 or self.etiquetas is None:
            return []
        id_modelo = getattr(self, "_mapa_modelo", {}).get(int(topico))
        if id_modelo is None:
            return []
        candidatos = [w for w, _ in (self.modelo.get_topic(id_modelo) or [])]
        textos = [
            set(t.split())
            for t, e in zip(self.textos, self.etiquetas, strict=True)
            if e == topico
        ]
        minimo = max(1, int(np.ceil(self.cfg.df_min_termino * len(textos))))
        elegidos = []
        for termino in candidatos:
            palabras = termino.split()
            if len(set(palabras)) < len(palabras):
                continue
            presentes = sum(all(p in t for p in palabras) for t in textos)
            if presentes >= minimo:
                elegidos.append(termino)
        return (elegidos or candidatos)[:n]

    def cohesion(self) -> dict[int, float]:
        """Coseno medio cláusula -> centroide de su tópico (únicos)."""
        if self.etiquetas is None or self.embeddings is None:
            return {}
        cents = centroides(self.embeddings, self.etiquetas)
        return {
            t: round(float((self.embeddings[self.etiquetas == t] @ c).mean()), 4)
            for t, c in cents.items()
        }

    def podar_por_cohesion(self, k: float) -> np.ndarray:
        """Disuelve en outliers los tópicos con cohesión < media - k·sd.

        Los tópicos de baja cohesión fueron los de menor pureza humana en
        una versión previa (T2 "decir·queja" 0/10). Renumera y actualiza ``etiquetas``.
        """
        if self.etiquetas is None:
            raise RuntimeError("Llame a particionar() antes de podar.")
        coh = pd.Series(self.cohesion())
        if len(coh) < 3:
            return self.etiquetas
        umbral = coh.mean() - k * coh.std()
        podados = [int(t) for t, c in coh.items() if c < umbral]
        if podados:
            etiquetas = self.etiquetas.copy()
            etiquetas[np.isin(etiquetas, podados)] = OUTLIER
            self.etiquetas = renumerar(etiquetas)
            self.diagnostico["topicos_podados"] = len(podados)
            logger.info("  poda por cohesión: %d tópicos disueltos", len(podados))
        return self.etiquetas

    def diversidad(self, top_n: int = 10) -> float:
        """Proporción de términos únicos entre los top-N de todos los tópicos."""
        if self.modelo is None:
            return float("nan")
        grupos = [
            self.terminos(t, top_n)
            for t in sorted(k for k in self.modelo.get_topics() if k >= 0)
        ]
        todos = [w for g in grupos for w in g]
        return round(len(set(todos)) / len(todos), 4) if todos else float("nan")


# ═══════════════════════════════════════════════════════════════════════════
# 10. BÚSQUEDA DE HIPERPARÁMETROS
# ═══════════════════════════════════════════════════════════════════════════


class SinConfiguracionValida(RuntimeError):
    """Ninguna combinación del grid cumple las puertas duras."""


def _normalizado(valor: float, bajo: float, alto: float) -> float:
    """Escala lineal a [0, 1] con recorte."""
    if not np.isfinite(valor) or alto <= bajo:
        return 0.0
    return float(min(1.0, max(0.0, (valor - bajo) / (alto - bajo))))


def puntuar(fila: dict[str, Any], crit: CriteriosNegocio) -> dict[str, Any]:
    """Puertas y puntaje monótono de una combinación.

    Returns:
        ``cumple`` (bool), ``puertas_fallidas`` (texto), ``distancia`` a las
        puertas (0 si cumple) y ``puntaje`` en [0, 1].
    """
    fallos: dict[str, float] = {}
    n = fila["n_topicos"]
    if n < crit.n_topicos_min:
        fallos["n_topicos"] = (crit.n_topicos_min - n) / crit.n_topicos_min
    elif n > crit.n_topicos_max:
        fallos["n_topicos"] = (n - crit.n_topicos_max) / crit.n_topicos_max
    for clave, limite in (
        ("pct_outliers", crit.pct_outliers_max),
        ("pct_top1", crit.pct_topico_mayor_max),
        ("pct_top2", crit.pct_top2_max),
    ):
        if fila[clave] > limite:
            fallos[clave] = (fila[clave] - limite) / limite
    sil = fila["silueta"]
    if not np.isfinite(sil) or sil < crit.silueta_min:
        fallos["silueta"] = 1.0 if not np.isfinite(sil) else (crit.silueta_min - sil)
    ari = fila.get("ari_medio", float("nan"))
    if np.isfinite(ari) and ari < crit.ari_min:
        fallos["ari"] = crit.ari_min - ari
    estabilidad = _normalizado(ari, 0.4, 1.0) if np.isfinite(ari) else 0.5
    puntaje = (
        crit.peso_silueta * _normalizado(sil, 0.0, 0.6)
        + crit.peso_concentracion * (1 - _normalizado(fila["pct_top1"], 5.0, 50.0))
        + crit.peso_outliers * (1 - _normalizado(fila["pct_outliers"], 0.0, 40.0))
        + crit.peso_estabilidad * estabilidad
    )
    return {
        "cumple": not fallos,
        "puertas_fallidas": ", ".join(sorted(fallos)),
        "distancia": round(float(sum(fallos.values())), 4),
        "puntaje": round(float(puntaje), 4),
    }


def frente_pareto(tabla: pd.DataFrame) -> pd.Series:
    """Marca filas no dominadas en (silueta max, pct_top1 min, outliers min)."""
    valores = np.column_stack(
        [
            -tabla["silueta"].fillna(-1).to_numpy(),
            tabla["pct_top1"].to_numpy(),
            tabla["pct_outliers"].to_numpy(),
        ]
    )
    dominada = np.zeros(len(valores), dtype=bool)
    for i, fila in enumerate(valores):
        mejores = np.all(valores <= fila, axis=1) & np.any(valores < fila, axis=1)
        dominada[i] = bool(mejores.any())
    return pd.Series(~dominada, index=tabla.index)


class GridSearchAspectos:
    """Grid sobre UMAP y HDBSCAN con las MISMAS piezas del modelo final.

    Cada combinación se evalúa tras fusión y reasignación reales, así que
    la cifra del grid es la que tendrá el modelo. Las proyecciones UMAP se
    cachean por (vecinos, componentes, semilla).

    Uso:
        grid = GridSearchAspectos(cfg, recursos, codificador).preparar(textos)
        tabla = grid.ejecutar()
        cfg_final = grid.mejor_config()
    """

    REJILLAS: dict[str, dict[str, tuple[Any, ...]]] = {
        "rapida": {
            "umap": ((15, 5), (30, 5)),
            "ratios": (0.03, 0.10),
            "metodos": ("eom", "leaf"),
            "n_tamanos": (4,),
        },
        "completa": {
            "umap": ((10, 5), (15, 5), (30, 5), (15, 8), (30, 8)),
            "ratios": (0.02, 0.05, 0.10),
            "metodos": ("eom", "leaf"),
            "n_tamanos": (6,),
        },
    }

    def __init__(
        self,
        cfg: Config,
        recursos: RecursosLexicos,
        codificador: Codificador | None = None,
        criterios: CriteriosNegocio | None = None,
    ) -> None:
        self.cfg = cfg
        self.criterios = criterios or CriteriosNegocio()
        self.modelador = ModeladorAspectos(cfg, recursos, codificador)
        self.resultados: pd.DataFrame = pd.DataFrame()
        self._proyecciones: dict[tuple[int, int, int], np.ndarray] = {}
        self._particiones: dict[int, np.ndarray] = {}
        self._finales: dict[int, np.ndarray] = {}

    def preparar(self, textos_aspecto: Sequence[str]) -> GridSearchAspectos:
        """Deduplica y codifica una sola vez para todo el grid."""
        self.modelador.preparar(textos_aspecto)
        return self

    @property
    def n_unicos(self) -> int:
        """Textos de aspecto únicos."""
        return len(self.modelador.textos)

    def proyeccion(self, vecinos: int, componentes: int, semilla: int) -> np.ndarray:
        """Proyección UMAP cacheada."""
        clave = (vecinos, componentes, semilla)
        if clave not in self._proyecciones:
            logger.info(
                "  UMAP n_neighbors=%d n_components=%d semilla=%d (%d textos)...",
                vecinos,
                componentes,
                semilla,
                self.n_unicos,
            )
            assert self.modelador.embeddings is not None
            self._proyecciones[clave] = proyectar_umap(
                self.modelador.embeddings, vecinos, componentes, self.cfg, semilla
            )
        return self._proyecciones[clave]

    def tamanos_sugeridos(self, n_valores: int) -> list[int]:
        """``min_topic_size`` geométricos derivados del rango de tópicos."""
        crit = self.criterios
        bajo = max(15, int(self.n_unicos / (8 * crit.n_topicos_max)))
        alto = max(bajo + 5, int(self.n_unicos / (1.5 * crit.n_topicos_max)))
        return sorted({int(round(v)) for v in np.geomspace(bajo, alto, n_valores)})

    def _evaluar(
        self, reducidos: np.ndarray, tamano: int, muestras: int, metodo: str
    ) -> tuple[dict[str, Any], np.ndarray]:
        """Mide una combinación con fusión y reasignación reales."""
        cfg = self.cfg
        emb = self.modelador.embeddings
        assert emb is not None
        crudas = renumerar(agrupar_hdbscan(reducidos, tamano, muestras, metodo))
        fusionadas, pares = fusionar_segun_politica(
            cfg,
            self.modelador.recursos,
            self.modelador.clave_stopwords,
            self.modelador.textos,
            emb,
            crudas,
        )
        finales, crecimiento = reasignar_outliers(
            emb, fusionadas, cfg.umbral_outliers, cfg.tope_crecimiento
        )
        medidas = {
            "n_topicos": int((finales.max() + 1) if (finales >= 0).any() else 0),
            "pct_outliers_hdbscan": round(100 * float((crudas == OUTLIER).mean()), 2),
            "pct_outliers": round(100 * float((finales == OUTLIER).mean()), 2),
            **concentracion(finales),
            "silueta": silueta(
                reducidos, crudas, cfg.silueta_muestra_max, cfg.random_state
            ),
            "max_coseno": max_coseno_centroides(emb, finales),
            "pares_fusionados": len(pares),
            "crecimiento_max": max(crecimiento.values(), default=0.0),
            "huella_particion": huella_texto(crudas.tolist())[:16],
        }
        return medidas, crudas

    def ejecutar(
        self,
        rejilla: str = "rapida",
        n_finalistas_estabilidad: int | None = None,
        semillas_extra: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """Recorre la rejilla, deduplica particiones y mide estabilidad.

        Args:
            rejilla: ``"rapida"`` o ``"completa"``.
            n_finalistas_estabilidad: Mejores filas a las que se mide ARI.
                ``None`` = ``cfg.n_finalistas_estabilidad``.
            semillas_extra: Semillas UMAP para el ARI. ``None`` =
                ``cfg.ari_semillas`` (las mismas que usan ``mejor_config`` y
                ``_config_de_fila``, para que los ARI de la tabla sean comparables).

        Returns:
            Tabla ordenada: cumple, frente de Pareto, puntaje.
        """
        if n_finalistas_estabilidad is None:
            n_finalistas_estabilidad = self.cfg.n_finalistas_estabilidad
        if semillas_extra is None:
            semillas_extra = self.cfg.ari_semillas
        espec = self.REJILLAS[rejilla]
        tamanos = self.tamanos_sugeridos(espec["n_tamanos"][0])
        combinaciones = [
            (v, c, t, r, m)
            for v, c in espec["umap"]
            for t in tamanos
            for r in espec["ratios"]
            for m in espec["metodos"]
        ]
        logger.info(
            "Grid '%s': %d combinaciones | min_topic_size %s | %d únicos",
            rejilla,
            len(combinaciones),
            tamanos,
            self.n_unicos,
        )
        filas: list[dict[str, Any]] = []
        vistas: set[str] = set()
        for i, (vecinos, comp, tamano, ratio, metodo) in enumerate(combinaciones, 1):
            if vecinos >= self.n_unicos:
                continue
            reducidos = self.proyeccion(vecinos, comp, self.cfg.random_state)
            muestras = max(2, int(round(tamano * ratio)))
            medidas, crudas = self._evaluar(reducidos, tamano, muestras, metodo)
            duplicada = medidas["huella_particion"] in vistas
            vistas.add(medidas["huella_particion"])
            fila = {
                "umap_n_neighbors": vecinos,
                "umap_n_components": comp,
                "min_topic_size": tamano,
                "min_samples": muestras,
                "metodo": metodo,
                "duplicada": duplicada,
                **medidas,
            }
            fila["id"] = len(filas)
            self._particiones[fila["id"]] = crudas
            self._finales[fila["id"]] = self._finalizar(crudas)
            filas.append(fila)
            logger.info(
                "  [%3d/%3d] nn=%2d mts=%4d ms=%3d %-4s -> %2d tóp | out %5.1f%% | "
                "top1 %4.1f%% | sil %.3f%s",
                i,
                len(combinaciones),
                vecinos,
                tamano,
                muestras,
                metodo,
                medidas["n_topicos"],
                medidas["pct_outliers"],
                medidas["pct_top1"],
                medidas["silueta"],
                " (duplicada)" if duplicada else "",
            )
        if not filas:
            raise SinConfiguracionValida("Ninguna combinación fue evaluable.")
        tabla = pd.DataFrame(filas)
        tabla["ari_medio"] = float("nan")
        tabla = self._puntuar_tabla(tabla)
        # Se mide ARI a los finalistas; si tras reordenar la primera fila aún
        # no tiene ARI medido, se mide también (nunca se elige sin medirlo).
        pendientes = tabla[~tabla["duplicada"]].head(n_finalistas_estabilidad)
        self._medir_estabilidad(tabla, pendientes.index, semillas_extra)
        for _ in range(2 * n_finalistas_estabilidad):
            tabla = self._puntuar_tabla(tabla)
            primera = tabla[~tabla["duplicada"]].head(1)
            if primera.empty or np.isfinite(primera["ari_medio"].iloc[0]):
                break
            self._medir_estabilidad(tabla, primera.index, semillas_extra)
        self.resultados = self._puntuar_tabla(tabla)
        logger.info(
            "Grid terminado: %d evaluadas | %d únicas | %d cumplen",
            len(tabla),
            int((~tabla["duplicada"]).sum()),
            int(self.resultados["cumple"].sum()),
        )
        return self.resultados

    def _puntuar_tabla(self, tabla: pd.DataFrame) -> pd.DataFrame:
        """Aplica puertas, puntaje y Pareto; ordena."""
        tabla = tabla.drop(
            columns=["cumple", "puertas_fallidas", "distancia", "puntaje", "pareto"],
            errors="ignore",
        )
        puntos = pd.DataFrame(
            [puntuar(f, self.criterios) for f in tabla.to_dict("records")],
            index=tabla.index,
        )
        tabla = pd.concat([tabla, puntos], axis=1)
        tabla["pareto"] = frente_pareto(tabla)
        # Entre las que cumplen: Pareto primero y luego puntaje. Entre las
        # que no: la más cercana a las puertas (nunca la de mejor puntaje).
        tabla["_orden"] = np.where(
            tabla["cumple"],
            -(tabla["puntaje"] + tabla["pareto"].astype(float)),
            tabla["distancia"],
        )
        return (
            tabla.sort_values(
                ["duplicada", "cumple", "_orden"], ascending=[True, False, True]
            )
            .drop(columns="_orden")
            .reset_index(drop=True)
        )

    def _finalizar(self, crudas: np.ndarray) -> np.ndarray:
        """Aplica fusión y reasignación: la partición que se ENTREGA."""
        emb = self.modelador.embeddings
        assert emb is not None
        fusionadas, _ = fusionar_segun_politica(
            self.cfg,
            self.modelador.recursos,
            self.modelador.clave_stopwords,
            self.modelador.textos,
            emb,
            crudas,
        )
        finales, _ = reasignar_outliers(
            emb, fusionadas, self.cfg.umbral_outliers, self.cfg.tope_crecimiento
        )
        return finales

    def _medir_estabilidad(
        self, tabla: pd.DataFrame, indices: Iterable[Any], semillas: Sequence[int]
    ) -> None:
        """ARI entre particiones FINALES de distintas semillas UMAP.

        En una versión previa se comparaban las particiones crudas de HDBSCAN, con 43% de
        outliers etiquetados -1: el ARI medía sobre todo la inestabilidad de
        la frontera de ruido, no la de los aspectos entregados.
        """
        from sklearn.metrics import (
            adjusted_rand_score,
        )  # pylint: disable=import-outside-toplevel

        for indice in indices:
            fila = tabla.loc[indice]
            aris = []
            for semilla in semillas:
                reducidos = self.proyeccion(
                    int(fila["umap_n_neighbors"]),
                    int(fila["umap_n_components"]),
                    semilla,
                )
                otra = agrupar_hdbscan(
                    reducidos,
                    int(fila["min_topic_size"]),
                    int(fila["min_samples"]),
                    str(fila["metodo"]),
                )
                aris.append(
                    adjusted_rand_score(
                        self._finales[int(fila["id"])],
                        self._finalizar(renumerar(otra)),
                    )
                )
            tabla.loc[indice, "ari_medio"] = round(float(np.mean(aris)), 4)
            logger.info(
                "  estabilidad fila %d: ARI medio %.3f entre %d semillas",
                int(fila["id"]),
                tabla.loc[indice, "ari_medio"],
                len(semillas) + 1,
            )

    def mejor_config(
        self,
        indice: int | None = None,
        permitir_no_cumple: bool = False,
        max_intentos: int = 5,
    ) -> Config:
        """Config de la fila elegida, lista para el modelo final.

        Antes, ``indice=0`` por defecto: se tomaba la primera fila y
        su ARI se medía DESPUÉS de elegirla, sin volver a aplicar la puerta de
        estabilidad (en obs19 se llegó a entregar una fila con ARI 0,608 <
        0,70 que figuraba como ``cumple=True``). Ahora, con ``indice=None``
        (por defecto), se recorren en orden las filas que cumplen, se mide el
        ARI de las que no lo tienen (hasta ``max_intentos`` mediciones) y se
        entrega la PRIMERA que pasa también la puerta de estabilidad. Si
        ninguna pasa: con ``permitir_no_cumple`` se entrega la más estable de
        las medidas (queda en el log y el scorecard marcará G2); sin él, falla.

        Args:
            indice: Posición en la tabla ordenada para forzar una fila
                (decisión editorial). ``None`` = selección automática estable.
            permitir_no_cumple: Si False y no hay fila válida, falla.
            max_intentos: Mediciones de ARI adicionales como máximo (cada una
                recalcula UMAP con otras semillas: minutos).

        Raises:
            SinConfiguracionValida: Si no hay fila válida y no se autorizó.
        """
        if self.resultados.empty:
            raise RuntimeError("Ejecute ejecutar() primero.")
        if indice is not None:
            return self._config_de_fila(indice, permitir_no_cumple)

        minimo = self.criterios.ari_min
        for _ in range(max_intentos + 1):
            self.resultados = self._puntuar_tabla(self.resultados)
            candidatas = self.resultados[
                ~self.resultados["duplicada"] & self.resultados["cumple"]
            ]
            if candidatas.empty:
                break
            primera = candidatas.index[0]
            ari = float(self.resultados.loc[primera, "ari_medio"])
            if np.isfinite(ari):
                # Sigue cumpliendo con ARI medido => pasó también la estabilidad.
                return self._config_de_fila(int(primera), permitir_no_cumple)
            self._medir_estabilidad(self.resultados, [primera], self.cfg.ari_semillas)
        # Ninguna fila cumple con estabilidad medida dentro del presupuesto.
        medidas = self.resultados[
            ~self.resultados["duplicada"] & np.isfinite(self.resultados["ari_medio"])
        ]
        logger.warning(
            "Ninguna configuración alcanzó ARI >= %.2f (medidas: %d; mejor "
            "ARI %.3f). %s",
            minimo,
            len(medidas),
            float(medidas["ari_medio"].max()) if not medidas.empty else float("nan"),
            "Se usa la más estable de las medidas (G2 fallará en el scorecard)."
            if permitir_no_cumple
            else "Amplíe la rejilla, revise CriteriosNegocio o use permitir_no_cumple.",
        )
        if not permitir_no_cumple or medidas.empty:
            if medidas.empty and permitir_no_cumple:
                return self._config_de_fila(0, permitir_no_cumple)
            raise SinConfiguracionValida(
                f"Ninguna configuración estable (ARI >= {minimo}) tras "
                f"{len(medidas)} mediciones."
            )
        return self._config_de_fila(
            int(medidas["ari_medio"].idxmax()), permitir_no_cumple
        )

    def _config_de_fila(self, indice: int, permitir_no_cumple: bool) -> Config:
        """Config de una fila concreta (lógica de la versión anterior)."""
        fila = self.resultados.iloc[indice]
        if not bool(fila["cumple"]) and not permitir_no_cumple:
            raise SinConfiguracionValida(
                f"La fila {indice} no cumple: {fila['puertas_fallidas']}. Revise la "
                "tabla, amplíe la rejilla ('completa') o ajuste CriteriosNegocio. "
                "Para usarla igualmente: permitir_no_cumple=True (quedará registrado)."
            )
        if not np.isfinite(float(fila["ari_medio"])) and not bool(fila["duplicada"]):
            self._medir_estabilidad(self.resultados, [indice], self.cfg.ari_semillas)
            fila = self.resultados.iloc[indice]
        ari = float(fila["ari_medio"])
        if np.isfinite(ari) and ari < self.criterios.ari_min:
            logger.warning(
                "La fila %d tiene ARI %.3f < %.2f: no es estable (G2 fallará).",
                indice,
                ari,
                self.criterios.ari_min,
            )
        nueva = replace(
            self.cfg,
            umap_n_neighbors=int(fila["umap_n_neighbors"]),
            umap_n_components=int(fila["umap_n_components"]),
            min_topic_size=int(fila["min_topic_size"]),
            min_samples=int(fila["min_samples"]),
            cluster_selection_method=str(fila["metodo"]),
        )
        logger.info(
            "Configuración elegida (fila %d, cumple=%s, ARI=%.3f): nn=%d, mts=%d, ms=%d, %s",
            indice,
            bool(fila["cumple"]) and not (np.isfinite(ari) and ari < self.criterios.ari_min),
            ari,
            nueva.umap_n_neighbors,
            nueva.min_topic_size,
            nueva.min_samples,
            nueva.cluster_selection_method,
        )
        return nueva


    def proyeccion_de(self, cfg: Config) -> np.ndarray | None:
        """Proyección ya calculada para una config (garantiza identidad)."""
        return self._proyecciones.get(
            (cfg.umap_n_neighbors, cfg.umap_n_components, cfg.random_state)
        )


# ═══════════════════════════════════════════════════════════════════════════
# 11. RAMA B — SENTIMIENTO (inyectable)
# ═══════════════════════════════════════════════════════════════════════════


class ClasificadorPolaridad(Protocol):
    """Cualquier función textos -> lista de {POS: p, NEU: p, NEG: p}."""

    def __call__(self, textos: Sequence[str]) -> list[dict[str, float]]:
        """Probabilidades por clase para cada texto."""


class ClasificadorRobertuito:
    """RoBERTuito vía transformers, con probabilidades de las tres clases."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._pipeline: Any = None
        self._preprocesar: Callable[[str], str] = lambda t: t

    def _cargar(self) -> Any:
        """Carga perezosa del pipeline y del preprocesado de pysentimiento."""
        if self._pipeline is not None:
            return self._pipeline
        from transformers import pipeline  # pylint: disable=import-outside-toplevel

        if self.cfg.usar_preprocesamiento_tweet:
            try:
                # pylint: disable-next=import-outside-toplevel
                from pysentimiento.preprocessing import preprocess_tweet

                self._preprocesar = preprocess_tweet
            except ImportError:
                logger.warning("pysentimiento no instalado: sin preprocesado de tweet")
        logger.info("  cargando '%s'...", self.cfg.modelo_sentimiento)
        self._pipeline = pipeline(  # type: ignore[call-overload]
            task="text-classification",
            model=self.cfg.modelo_sentimiento,
            top_k=None,
            truncation=True,
            max_length=self.cfg.max_length_sentimiento,
        )
        return self._pipeline

    def __call__(self, textos: Sequence[str]) -> list[dict[str, float]]:
        """Clasifica por lotes."""
        clasificador = self._cargar()
        salida: list[dict[str, float]] = []
        for inicio in range(0, len(textos), self.cfg.batch_size):
            lote = [
                self._preprocesar(str(t))
                for t in textos[inicio : inicio + self.cfg.batch_size]
            ]
            for resultado in clasificador(lote):
                salida.append(
                    {str(r["label"]).upper(): float(r["score"]) for r in resultado}
                )
        return salida


class AnalizadorSentimiento:
    """Polaridad de cláusulas sobre texto ORIGINAL (con tildes y negadores)."""

    def __init__(
        self,
        cfg: Config,
        recursos: RecursosLexicos,
        clasificador: ClasificadorPolaridad | None = None,
    ) -> None:
        self.cfg = cfg
        self.recursos = recursos
        self.clasificador: ClasificadorPolaridad = (
            clasificador or ClasificadorRobertuito(cfg)
        )
        self.calibrador: Calibrador | None = None
        if cfg.ruta_calibrador and Path(cfg.ruta_calibrador).exists():
            self.calibrador = Calibrador.cargar(cfg.ruta_calibrador)
            logger.info(
                "  calibrador cargado (%s): F1 macro CV %s",
                Path(cfg.ruta_calibrador).name,
                self.calibrador.metricas.get("f1_macro_cv"),
            )

    def clasificar(self, textos: Sequence[str], tipo: str = "ABIERTA") -> pd.DataFrame:
        """Probabilidades, etiqueta del modelo, polaridad y etiqueta final.

        ``sentimiento_modelo`` es el argmax de RoBERTuito y siempre se
        conserva. ``sentimiento`` es (a) la etiqueta del :class:`Calibrador`
        si hay uno cargado, o (b) el argmax con NEU reinterpretada por
        ``margen_neu`` si no lo hay. ``neu_ajustado`` marca las cláusulas
        cuya etiqueta final difiere de la del modelo.

        Args:
            textos: Cláusulas originales.
            tipo: Tipo de pregunta (rasgo del calibrador).

        Returns:
            DataFrame alineado con ``textos``.
        """
        serie = pd.Series([str(t) for t in textos])
        codigos, unicos = pd.factorize(serie, sort=False)
        logger.info("  sentimiento: %d cláusulas (%d únicas)", len(serie), len(unicos))
        probabilidades = pd.DataFrame(self.clasificador(list(unicos))).reindex(
            columns=[POS, NEU, NEG], fill_value=0.0
        )
        marco = probabilidades.iloc[codigos].reset_index(drop=True)
        marco.columns = ["p_pos", "p_neu", "p_neg"]
        marco["sentimiento_modelo"] = (
            marco[["p_pos", "p_neu", "p_neg"]].to_numpy().argmax(axis=1)
        )
        marco["sentimiento_modelo"] = marco["sentimiento_modelo"].map(
            {0: POS, 1: NEU, 2: NEG}
        )
        marco["confianza"] = marco[["p_pos", "p_neu", "p_neg"]].max(axis=1).round(4)
        marco["polaridad"] = (marco["p_pos"] - marco["p_neg"]).round(4)
        if self.calibrador is not None:
            probs, etiquetas = self.calibrador.aplicar(marco.assign(tipo_pregunta=tipo))
            marco["sentimiento"] = etiquetas
            marco["polaridad"] = np.round(probs[:, 0] - probs[:, 2], 4)
            marco["confianza"] = np.round(probs.max(axis=1), 4)
        else:
            ajustable = (marco["sentimiento_modelo"] == NEU) & (
                marco["polaridad"].abs() >= self.cfg.margen_neu
            )
            marco["sentimiento"] = marco["sentimiento_modelo"]
            marco.loc[ajustable, "sentimiento"] = np.where(
                marco.loc[ajustable, "polaridad"] > 0, POS, NEG
            )
        marco["neu_ajustado"] = marco["sentimiento"] != marco["sentimiento_modelo"]
        for columna in ("p_pos", "p_neu", "p_neg"):
            marco[columna] = marco[columna].round(4)
        logger.info(
            "  etiquetas corregidas respecto al modelo (%s): %d (%.1f%%)",
            "calibrador" if self.calibrador else f"margen {self.cfg.margen_neu:.2f}",
            int(marco["neu_ajustado"].sum()),
            100 * float(marco["neu_ajustado"].mean()),
        )
        return marco

    def marcar_modificadores(self, textos: Sequence[str]) -> pd.DataFrame:
        """Presencia de negadores, cuantificadores bajos y modales (auditoría)."""
        columnas: dict[str, list[bool]] = {}
        for tipo in ("NEGADOR", "CUANTIFICADOR_BAJO", "MODAL_ATENUANTE"):
            terminos = self.recursos.terminos_por_tipo(tipo, con_tilde=True)
            if not terminos:
                columnas[f"tiene_{tipo.lower()}"] = [False] * len(textos)
                continue
            patron = re.compile(
                r"\b(?:" + "|".join(re.escape(t) for t in terminos) + r")\b",
                flags=re.IGNORECASE,
            )
            columnas[f"tiene_{tipo.lower()}"] = [
                bool(patron.search(str(t))) for t in textos
            ]
        return pd.DataFrame(columnas)


# ═══════════════════════════════════════════════════════════════════════════
# 11b. CALIBRACIÓN DEL SENTIMIENTO CON ANOTACIONES HUMANAS
# ═══════════════════════════════════════════════════════════════════════════
# RoBERTuito se entrenó con tuits. En evaluación docente subestima elogios
# sin adjetivo ("está pendiente de nosotros" -> NEG) y lee sugerencias de
# mejora como neutras. La auditoría de una versión previa (≈1.000 cláusulas anotadas en 9
# encuestas) mostró además que el error cambia de dirección según el tipo de
# pregunta. Una regresión logística sobre las probabilidades del modelo y el
# tipo de pregunta corrige ese sesgo; se evalúa con validación cruzada para
# no medirse con los datos con que se ajusta.

TIPOS_PREGUNTA: tuple[str, ...] = ("FORTALEZA", "MEJORA", "ABIERTA")
#: Anotaciones mínimas de un tipo de pregunta para darle calibrador PROPIO.
#: Con menos, ese tipo usa el calibrador compartido (hoy ABIERTA: 0 anotaciones).
MIN_ANOTACIONES_CALIBRADOR_TIPO: int = 100


def _probs_logisticas(
    x: np.ndarray, coef: Sequence[Sequence[float]], intercepto: Sequence[float], clases: Sequence[str]
) -> np.ndarray:
    """Softmax de una regresión logística guardada, en orden (POS, NEU, NEG)."""
    logits = x @ np.asarray(coef).T + np.asarray(intercepto)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs[:, [list(clases).index(c) for c in (POS, NEU, NEG)]]


def _log_probs_calibracion(marco: pd.DataFrame) -> np.ndarray:
    """Rasgos del calibrador por tipo: solo las 3 log-probabilidades."""
    return np.log(np.clip(marco[["p_pos", "p_neu", "p_neg"]].to_numpy(float), 1e-4, 1))


def _rasgos_calibracion(marco: pd.DataFrame) -> np.ndarray:
    """log-probabilidades + indicadores del tipo de pregunta."""
    probs = np.clip(marco[["p_pos", "p_neu", "p_neg"]].to_numpy(float), 1e-4, 1)
    tipos = marco["tipo_pregunta"].astype(str).to_numpy()
    dummies = np.column_stack([(tipos == t).astype(float) for t in TIPOS_PREGUNTA])
    return np.column_stack([np.log(probs), dummies])


@dataclass
class Calibrador:
    """Corrección aprendida de las probabilidades de RoBERTuito.

    Además del modelo COMPARTIDO (log-probabilidades + indicador
    del tipo de pregunta), guarda un modelo PROPIO por cada tipo con al menos
    ``MIN_ANOTACIONES_CALIBRADOR_TIPO`` anotaciones. Ronda 7 (431 anotaciones,
    10 repeticiones de CV 5 pliegues con los mismos pliegues): FORTALEZA
    0,645 -> 0,669 (+0,024, ~3 desviaciones); MEJORA 0,712 -> 0,714 (igual).
    Un tipo sin modelo propio usa el compartido; también un tipo cuyo modelo
    propio rinde PEOR en CV que el compartido para ese tipo. Un JSON viejo (sin
    ``por_tipo``) se carga y funciona como antes.

    Attributes:
        coef: Matriz (3 clases x 6 rasgos) del modelo compartido.
        intercepto: Vector (3,).
        clases: Orden de las clases.
        metricas: F1 macro y acierto en validación cruzada, n de anotaciones.
        por_tipo: ``{tipo: {coef, intercepto, clases, n}}`` (3 rasgos).
    """

    coef: list[list[float]]
    intercepto: list[float]
    clases: list[str]
    metricas: dict[str, Any] = field(default_factory=dict)
    por_tipo: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def ajustar(cls, anotadas: pd.DataFrame, semilla: int = 42) -> Calibrador:
        """Ajusta y evalúa con validación cruzada estratificada (5 pliegues).

        Args:
            anotadas: Columnas ``p_pos``, ``p_neu``, ``p_neg``,
                ``tipo_pregunta`` y ``etiqueta_humana``.
            semilla: Para los pliegues.

        Returns:
            Calibrador ajustado sobre TODAS las anotaciones, con métricas de
            validación cruzada (las honestas) en ``metricas``.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        marco = anotadas.copy()
        marco["etiqueta_humana"] = marco["etiqueta_humana"].astype(str).str.strip().str.upper()
        marco = marco[marco["etiqueta_humana"].isin([POS, NEU, NEG])].reset_index(
            drop=True
        )
        if len(marco) < 100:
            raise ValueError(f"Se requieren >= 100 anotaciones; hay {len(marco)}")
        x, y = _rasgos_calibracion(marco), marco["etiqueta_humana"].to_numpy()
        modelo = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        pliegues = StratifiedKFold(5, shuffle=True, random_state=semilla)
        pred_cv = cross_val_predict(modelo, x, y, cv=pliegues)
        base = (
            marco["sentimiento_modelo"].to_numpy()
            if "sentimiento_modelo" in marco
            else None
        )
        metricas = {
            "n_anotadas": int(len(marco)),
            "f1_macro_cv": round(float(f1_score(y, pred_cv, average="macro")), 4),
            "acierto_cv": round(float((pred_cv == y).mean()), 4),
            "f1_macro_sin_calibrar": (
                round(float(f1_score(y, base, average="macro")), 4)
                if base is not None
                else None
            ),
            "por_tipo": {
                t: {
                    "n": int((marco["tipo_pregunta"] == t).sum()),
                    "f1_macro_cv": round(
                        float(
                            f1_score(
                                y[marco["tipo_pregunta"] == t],
                                pred_cv[marco["tipo_pregunta"] == t],
                                average="macro",
                            )
                        ),
                        4,
                    ),
                }
                for t in TIPOS_PREGUNTA
                if (marco["tipo_pregunta"] == t).sum() >= 20
            },
        }
        modelo.fit(x, y)

        # Modelo PROPIO por tipo de pregunta (si hay anotaciones suficientes).
        # Se evalúa con los MISMOS pliegues que el compartido: la predicción CV "final"
        # usa el modelo propio donde existe y el compartido donde no.
        tipos = marco["tipo_pregunta"].astype(str).to_numpy()
        x_tipo = _log_probs_calibracion(marco)
        propios = [
            t for t in TIPOS_PREGUNTA
            if (tipos == t).sum() >= MIN_ANOTACIONES_CALIBRADOR_TIPO
            and len(set(y[tipos == t])) == 3
        ]
        pred_final = pred_cv.copy()
        for entreno, prueba in pliegues.split(x, y):
            for t in propios:
                tr, te = entreno[tipos[entreno] == t], prueba[tipos[prueba] == t]
                if len(te) and len(set(y[tr])) == 3:
                    m_t = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
                    pred_final[te] = m_t.fit(x_tipo[tr], y[tr]).predict(x_tipo[te])
        por_tipo: dict[str, dict[str, Any]] = {}
        for t in propios:
            m_t = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
            m_t.fit(x_tipo[tipos == t], y[tipos == t])
            por_tipo[t] = {
                "coef": m_t.coef_.tolist(),
                "intercepto": m_t.intercept_.tolist(),
                "clases": [str(c) for c in m_t.classes_],
                "n": int((tipos == t).sum()),
            }
        # Un tipo conserva su modelo propio solo si en CV no rinde peor que el compartido
        # para ESE tipo; si rinde peor, vuelve al compartido (sus filas recuperan pred_cv).
        for t in list(por_tipo):
            k = tipos == t
            if f1_score(y[k], pred_final[k], average="macro") < f1_score(y[k], pred_cv[k], average="macro"):
                pred_final[k] = pred_cv[k]
                del por_tipo[t]
        metricas["f1_macro_cv_compartido"] = metricas["f1_macro_cv"]
        metricas["f1_macro_cv"] = round(float(f1_score(y, pred_final, average="macro")), 4)
        metricas["acierto_cv"] = round(float((pred_final == y).mean()), 4)
        for t, detalle in metricas["por_tipo"].items():
            k = tipos == t
            detalle["f1_macro_cv_compartido"] = detalle["f1_macro_cv"]
            detalle["f1_macro_cv"] = round(float(f1_score(y[k], pred_final[k], average="macro")), 4)
            detalle["modelo"] = "propio" if t in por_tipo else "compartido"
        logger.info(
            "Calibrador: %d anotaciones | F1 macro CV %.3f (compartido %.3f, sin calibrar %s) | "
            "acierto CV %.3f | modelo propio: %s",
            len(marco),
            metricas["f1_macro_cv"],
            metricas["f1_macro_cv_compartido"],
            metricas["f1_macro_sin_calibrar"],
            metricas["acierto_cv"],
            ", ".join(f"{t} (n={d['n']})" for t, d in por_tipo.items()) or "ninguno",
        )
        return cls(
            coef=modelo.coef_.tolist(),
            intercepto=modelo.intercept_.tolist(),
            clases=[str(c) for c in modelo.classes_],
            metricas=metricas,
            por_tipo=por_tipo,
        )


    def aplicar(self, marco: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Probabilidades calibradas (n, 3 en orden POS, NEU, NEG) y etiquetas.

        Cada fila usa el modelo propio de su ``tipo_pregunta`` si
        existe; si no, el compartido.
        """
        probs = _probs_logisticas(
            _rasgos_calibracion(marco), self.coef, self.intercepto, self.clases
        )
        if self.por_tipo:
            tipos = marco["tipo_pregunta"].astype(str).to_numpy()
            x_tipo = _log_probs_calibracion(marco)
            for t, m in self.por_tipo.items():
                k = tipos == t
                if k.any():
                    probs[k] = _probs_logisticas(x_tipo[k], m["coef"], m["intercepto"], m["clases"])
        etiquetas = np.array([POS, NEU, NEG])[probs.argmax(axis=1)]
        return probs, etiquetas

    def guardar(self, ruta: Path | str) -> Path:
        """Escribe el calibrador en JSON."""
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(
            json.dumps(_a_json(asdict(self)), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return ruta

    @classmethod
    def cargar(cls, ruta: Path | str) -> Calibrador:
        """Lee un calibrador guardado con :meth:`guardar`."""
        datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
        return cls(**datos)


def reunir_anotaciones(salida: Path | str) -> pd.DataFrame:
    """Une las ``muestra_manual`` anotadas de todas las corridas exportadas.

    Cada anotación se cruza por texto de cláusula con ``clausulas.parquet``
    de su corrida para recuperar las probabilidades crudas de RoBERTuito, y
    con ``manifest.json`` para el tipo de pregunta. Es el insumo de
    :meth:`Calibrador.ajustar`.
    """
    partes = []
    for manifiesto in sorted(Path(salida).glob("*/*/manifest.json")):
        carpeta = manifiesto.parent
        ruta = carpeta / "auditoria.xlsx"
        if not ruta.exists():
            continue
        manual = pd.read_excel(ruta, sheet_name="muestra_manual")
        manual["etiqueta_humana"] = manual["etiqueta_humana"].astype(str).str.strip().str.upper()
        manual = manual[manual["etiqueta_humana"].isin([POS, NEU, NEG])]
        if manual.empty:
            continue
        datos = json.loads(manifiesto.read_text(encoding="utf-8"))
        tipo = datos.get("metricas", {}).get("tipo_pregunta", "ABIERTA")
        try:
            crudas = pd.read_parquet(
                carpeta / "clausulas.parquet",
                columns=["clausula", "p_pos", "p_neu", "p_neg", "sentimiento_modelo"],
            )
        except (ImportError, FileNotFoundError):
            crudas = pd.read_csv(
                carpeta / "clausulas.csv",
                usecols=["clausula", "p_pos", "p_neu", "p_neg", "sentimiento_modelo"],
            )
        cruce = manual[["clausula", "etiqueta_humana"]].merge(
            crudas.dropna(subset=["p_pos"]).drop_duplicates("clausula"),
            on="clausula",
            how="inner",
        )
        cruce["tipo_pregunta"] = tipo
        cruce["encuesta"] = carpeta.parent.name
        partes.append(cruce)
    if not partes:
        return pd.DataFrame()
    todas = pd.concat(partes, ignore_index=True)
    logger.info(
        "Anotaciones reunidas: %d cláusulas de %d corridas", len(todas), len(partes)
    )
    return todas


def calibrar_sentimiento(
    salida: Path | str, ruta_calibrador: Path | str, semilla: int = 42
) -> Calibrador:
    """Reúne anotaciones, ajusta el calibrador y lo guarda."""
    anotadas = reunir_anotaciones(salida)
    calibrador = Calibrador.ajustar(anotadas, semilla)
    calibrador.guardar(ruta_calibrador)
    return calibrador


def sugerir_aspectos(
    modelador: ModeladorAspectos | None, topicos: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Aspecto de la taxonomía más afín a cada tópico (INFORMATIVO).

    Compara el centroide del tópico con los embeddings de las descripciones,
    CENTRADOS: se resta la media de las descripciones para que la componente
    común ("la ... de los estudiantes") no atraiga todo hacia un mismo
    aspecto, como pasó con CLARIDAD en una versión previa. No es una puerta ni reemplaza
    la lectura humana; es una ayuda para nombrar.
    """
    if (
        topicos.empty
        or not cfg.taxonomia
        or modelador is None
        or modelador.embeddings is None
        or modelador.etiquetas is None
    ):
        return topicos.assign(aspecto_sugerido="", afinidad_aspecto=np.nan)
    claves = list(cfg.taxonomia)
    textos = [f"{k.replace('_', ' ').lower()}: {v}" for k, v in cfg.taxonomia.items()]
    desc = np.asarray(modelador.codificador(textos), dtype=float)
    desc = desc - desc.mean(axis=0, keepdims=True)
    desc /= np.maximum(np.linalg.norm(desc, axis=1, keepdims=True), 1e-12)
    cents = centroides(modelador.embeddings, modelador.etiquetas)
    matriz = np.stack([cents[int(t)] for t in topicos["topico"]])
    matriz = matriz - np.asarray(modelador.codificador(textos), dtype=float).mean(
        axis=0, keepdims=True
    )
    matriz /= np.maximum(np.linalg.norm(matriz, axis=1, keepdims=True), 1e-12)
    sim = matriz @ desc.T
    mejor = sim.argmax(axis=1)
    return topicos.assign(
        aspecto_sugerido=[claves[i] for i in mejor],
        afinidad_aspecto=np.round(sim.max(axis=1), 4),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 12. VALIDACIÓN DEL SENTIMIENTO
# ═══════════════════════════════════════════════════════════════════════════


def validar_contra_nota(
    clausulas: pd.DataFrame, notas: pd.Series | None, autor: str = "ESTUDIANTE"
) -> dict[str, Any]:
    """Contrasta la polaridad contra la calificación numérica.

    Mide a dos niveles: cláusula (cada una con la nota de su documento) y
    documento (polaridad media de sus cláusulas), que es la unidad donde
    la nota realmente vive.

    Advertencia de diseño: en preguntas de "aspectos a resaltar" el texto
    es positivo casi por construcción y la nota se concentra arriba; una
    correlación moderada ahí es esperable y no implica un mal modelo.

    Args:
        clausulas: Con ``doc_id``, ``polaridad`` y ``sentimiento`` (NaN en
            las cláusulas sin opinión).
        notas: Nota por fila del corpus (índice = doc_id).

    Returns:
        Métricas y tablas de detalle (``por_polaridad``, ``deciles``).
    """
    # pylint: disable=import-outside-toplevel
    from scipy.stats import kruskal, spearmanr
    from sklearn.metrics import roc_auc_score

    if notas is None:
        return {"disponible": False, "motivo": f"sin columna {COLUMNA_VALIDACION}"}
    if autor == "DOCENTE":
        return {
            "disponible": False,
            "motivo": "texto escrito por el docente: la nota de los estudiantes no lo valida",
        }
    marco = clausulas.dropna(subset=["sentimiento"]).copy()
    marco["nota"] = marco["doc_id"].map(notas)
    marco = marco.dropna(subset=["nota"])
    if len(marco) < 30 or marco["nota"].nunique() < 2:
        return {"disponible": False, "motivo": "notas insuficientes o constantes"}

    rho_c, p_c = spearmanr(marco["polaridad"], marco["nota"])
    docs = marco.groupby("doc_id").agg(
        polaridad=("polaridad", "mean"), nota=("nota", "first")
    )
    rho_d, p_d = spearmanr(docs["polaridad"], docs["nota"])

    corte = float(docs["nota"].quantile(0.20))
    baja = docs["nota"] <= corte
    if baja.mean() > 0.5:  # notas muy concentradas: se usa "bajo la mediana"
        baja = docs["nota"] < docs["nota"].median()
    auc = (
        float(roc_auc_score(baja.astype(int), -docs["polaridad"]))
        if 0 < baja.sum() < len(baja)
        else float("nan")
    )
    grupos = [
        g["nota"].to_numpy() for _, g in marco.groupby("sentimiento") if len(g) > 1
    ]
    h_stat, p_h = kruskal(*grupos) if len(grupos) >= 2 else (np.nan, np.nan)

    por_polaridad = marco.groupby("sentimiento")["nota"].agg(
        ["mean", "median", "count"]
    )
    docs["decil_nota"] = (
        pd.qcut(docs["nota"].rank(method="first"), 10, labels=False) + 1
    )
    deciles = docs.groupby("decil_nota").agg(
        nota_media=("nota", "mean"),
        polaridad_media=("polaridad", "mean"),
        n=("nota", "size"),
    )
    monotonia, _ = spearmanr(deciles.index, deciles["polaridad_media"])
    medias = por_polaridad["mean"].to_dict()
    return {
        "disponible": True,
        "n_clausulas": int(len(marco)),
        "n_documentos": int(len(docs)),
        "spearman_clausula": round(float(rho_c), 4),
        "spearman_clausula_p": float(p_c),
        "spearman_documento": round(float(rho_d), 4),
        "spearman_documento_p": float(p_d),
        "auc_nota_baja": round(auc, 4),
        "monotonia_deciles": round(float(monotonia), 4),
        "nota_media_POS": round(float(medias.get(POS, np.nan)), 3),
        "nota_media_NEU": round(float(medias.get(NEU, np.nan)), 3),
        "nota_media_NEG": round(float(medias.get(NEG, np.nan)), 3),
        "kruskal_H": round(float(h_stat), 2),
        "kruskal_p": float(p_h),
        "por_polaridad": por_polaridad.round(3),
        "deciles": deciles.round(4),
    }


def muestra_manual(clausulas: pd.DataFrame, n: int, semilla: int) -> pd.DataFrame:
    """Muestra estratificada (etiqueta x confianza) para anotar a mano.

    Sobrerrepresenta NEG y NEU y la confianza baja, que es donde el modelo
    falla; así 150 cláusulas informan más que 150 al azar.

    """
    base = clausulas.dropna(subset=["sentimiento"]).copy()
    if base.empty:
        return pd.DataFrame()
    base["banda"] = pd.cut(
        base["confianza"], [0, 0.6, 0.85, 1.0], labels=["baja", "media", "alta"]
    )
    estratos = list(base.groupby(["sentimiento", "banda"], observed=True))
    cuota = max(1, n // max(len(estratos), 1))
    partes = [g.sample(min(len(g), cuota), random_state=semilla) for _, g in estratos]
    salida = pd.concat(partes).head(n)
    columnas = [
        "doc_id",
        "clausula_id",
        "clausula",
        "sentimiento",
        "confianza",
        "topico",
    ]
    salida = salida[[c for c in columnas if c in salida.columns]].copy()
    salida["etiqueta_humana"] = ""  # POS / NEU / NEG
    return salida.reset_index(drop=True)


def evaluar_muestra_manual(
    ruta_o_marco: Path | str | pd.DataFrame,
    distribucion_poblacional: dict[str, float] | None = None,
) -> dict[str, Any]:
    """F1 macro y matriz de confusión de una muestra anotada.

    La muestra se estratifica 1/3 por etiqueta del modelo para anotar con
    eficiencia, pero la población real es muy distinta (P19: 80/13/6). Un
    F1 sin corregir sobre la muestra sobrepesa la clase minoritaria (error
    de diseño de una versión previa). Con ``distribucion_poblacional`` cada cláusula se
    pondera por población/muestra según su etiqueta de modelo y el F1
    resultante estima el de la población.

    Args:
        ruta_o_marco: Excel devuelto por el anotador o DataFrame.
        distribucion_poblacional: ``{POS: p, NEU: p, NEG: p}`` de la corrida.

    Returns:
        ``f1_macro`` (reponderado si hay distribución), ``f1_macro_muestra``,
        ``acierto_poblacional``, reporte por clase y matriz de confusión.
    """
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
    )  # pylint: disable=import-outside-toplevel

    marco = (
        ruta_o_marco
        if isinstance(ruta_o_marco, pd.DataFrame)
        else pd.read_excel(ruta_o_marco, sheet_name="muestra_manual")
    )
    marco = marco.assign(
        etiqueta_humana=marco["etiqueta_humana"].astype(str).str.strip().str.upper()
    )
    marco = marco[marco["etiqueta_humana"].isin([POS, NEU, NEG])]
    if len(marco) < 30:
        return {"disponible": False, "n_anotadas": int(len(marco))}
    # Sólo clases presentes: una clase ausente de la muestra no es un error
    # del modelo, y promediarla como F1=0 hundía el macro-F1 artificialmente.
    presentes = set(marco["etiqueta_humana"]) | set(marco["sentimiento"])
    etiquetas = [e for e in (POS, NEU, NEG) if e in presentes]
    pesos = None
    if distribucion_poblacional:
        cuota = marco["sentimiento"].value_counts(normalize=True)
        pesos = (
            marco["sentimiento"]
            .map(
                lambda e: distribucion_poblacional.get(e, 0.0)
                / max(cuota.get(e, 1e-9), 1e-9)
            )
            .to_numpy(float)
        )
    aciertos = (marco["etiqueta_humana"] == marco["sentimiento"]).to_numpy(float)
    return {
        "disponible": True,
        "n_anotadas": int(len(marco)),
        "reponderado": pesos is not None,
        "f1_macro_muestra": round(
            float(
                f1_score(
                    marco["etiqueta_humana"],
                    marco["sentimiento"],
                    labels=etiquetas,
                    average="macro",
                    zero_division=0,
                )
            ),
            4,
        ),
        "acierto_poblacional": round(float(np.average(aciertos, weights=pesos)), 4),
        "f1_macro": round(
            float(
                f1_score(
                    marco["etiqueta_humana"],
                    marco["sentimiento"],
                    labels=etiquetas,
                    average="macro",
                    zero_division=0,
                    sample_weight=pesos,
                )
            ),
            4,
        ),
        "reporte": classification_report(
            marco["etiqueta_humana"],
            marco["sentimiento"],
            labels=etiquetas,
            zero_division=0,
            output_dict=True,
        ),
        "confusion": pd.DataFrame(
            confusion_matrix(
                marco["etiqueta_humana"], marco["sentimiento"], labels=etiquetas
            ),
            index=[f"real_{e}" for e in etiquetas],
            columns=[f"pred_{e}" for e in etiquetas],
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 13. FUSIÓN — ABSA
# ═══════════════════════════════════════════════════════════════════════════


def wilson(
    exitos: np.ndarray, total: np.ndarray, z: float
) -> tuple[np.ndarray, np.ndarray]:
    """Intervalo de Wilson para proporciones (vectorizado)."""
    total = np.maximum(np.asarray(total, dtype=float), 1.0)
    p = np.asarray(exitos, dtype=float) / total
    denominador = 1 + z**2 / total
    centro = (p + z**2 / (2 * total)) / denominador
    radio = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominador
    return np.clip(centro - radio, 0, 1), np.clip(centro + radio, 0, 1)


def construir_absa(
    clausulas: pd.DataFrame, tabla_topicos: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Matriz aspecto x polaridad con intervalos y prioridad robusta.

    ``prioridad = Wilson_inferior(%NEG) x log(1 + volumen)``: un aspecto
    sube por negatividad CIERTA, no por tamaño. Las filas especiales
    (valoración global, sin tópico) se reportan pero no se priorizan.
    """
    base = clausulas.dropna(subset=["sentimiento"])
    base = base[base["topico"] != EXCLUIDO]
    if base.empty:
        return pd.DataFrame()
    conteos = pd.crosstab(base["topico"], base["sentimiento"]).reindex(
        columns=[POS, NEU, NEG], fill_value=0
    )
    absa = conteos.copy()
    absa["total_clausulas"] = absa.sum(axis=1)
    absa["documentos"] = base.groupby("topico")["doc_id"].nunique()
    for etiqueta, columna in (
        (POS, "pct_positivo"),
        (NEU, "pct_neutro"),
        (NEG, "pct_negativo"),
    ):
        absa[columna] = (100 * absa[etiqueta] / absa["total_clausulas"]).round(1)
    li, ls = wilson(
        absa[NEG].to_numpy(), absa["total_clausulas"].to_numpy(), cfg.z_wilson
    )
    absa["neg_ic_inf"] = (100 * li).round(1)
    absa["neg_ic_sup"] = (100 * ls).round(1)
    absa["polaridad_neta"] = ((absa[POS] - absa[NEG]) / absa["total_clausulas"]).round(
        3
    )
    absa["polaridad_media"] = base.groupby("topico")["polaridad"].mean().round(3)
    absa["es_aspecto"] = absa.index >= 0
    # ft-r4.15 (U4): las clases «sin aspecto» del modelo U4 tienen código >= 0 pero son filas especiales:
    # se reportan, no se priorizan. Sin columna ``especial`` (modo bertopic) no cambia nada.
    if (
        not tabla_topicos.empty
        and "especial" in tabla_topicos.columns
        and tabla_topicos["especial"].astype(bool).any()
    ):
        especiales = set(
            tabla_topicos.loc[tabla_topicos["especial"].astype(bool), "topico"].astype(int)
        )
        absa["es_aspecto"] = absa["es_aspecto"] & ~absa.index.isin(list(especiales))
    absa["n_suficiente"] = absa["total_clausulas"] >= cfg.n_min_celda
    absa["prioridad"] = np.where(
        absa["es_aspecto"] & absa["n_suficiente"],
        (li * np.log1p(absa["total_clausulas"].to_numpy())).round(4),
        np.nan,
    )
    revisar = (
        tabla_topicos.set_index("topico")["revisar"].astype(bool)
        if not tabla_topicos.empty and "revisar" in tabla_topicos.columns
        else pd.Series(dtype=bool)
    )
    if {"motivo", "texto_aspecto"} <= set(clausulas.columns):
        con_aspecto = clausulas[
            (clausulas["motivo"] == MOTIVO_ASPECTO) & (clausulas["topico"] >= 0)
        ]
        es_suelta = con_aspecto["texto_aspecto"].fillna("").str.split().str.len() == 1
        tasa_corpus = float(
            (
                clausulas.loc[clausulas["motivo"] == MOTIVO_ASPECTO, "texto_aspecto"]
                .fillna("")
                .str.split()
                .str.len()
                == 1
            ).mean()
        )
        pct_suelta = es_suelta.groupby(con_aspecto["topico"]).mean()
    else:
        tasa_corpus, pct_suelta = float("nan"), pd.Series(dtype=float)
    absa["pct_palabra_suelta"] = (100 * pct_suelta.reindex(absa.index)).round(1)
    absa["veces_base_suelta"] = (
        (pct_suelta / tasa_corpus).reindex(absa.index).round(2)
        if tasa_corpus and tasa_corpus == tasa_corpus  # evita /0 y NaN
        else np.nan
    )
    marcado = pd.Series(
        [bool(revisar.get(int(t), False)) for t in absa.index], index=absa.index
    )
    absa["confiable"] = ~(
        marcado
        & (absa["veces_base_suelta"].fillna(0) >= cfg.umbral_veces_palabra_suelta)
    )
    priorizables = absa["es_aspecto"] & absa["n_suficiente"]
    orden = (
        absa[priorizables]
        .sort_values(["confiable", "prioridad"], ascending=[False, False])
        .index
    )
    absa["rango_prioridad"] = pd.Series(
        np.arange(1, len(orden) + 1), index=orden, dtype="Int64"
    )
    nombres = (
        tabla_topicos.set_index("topico")["etiqueta"]
        if not tabla_topicos.empty
        else pd.Series(dtype=str)
    )
    absa["etiqueta"] = [
        ETIQUETAS_ESPECIALES.get(int(t), nombres.get(t, str(t))) for t in absa.index
    ]
    absa.index.name = "topico"
    return absa.sort_values(
        ["es_aspecto", "rango_prioridad", "prioridad"],
        ascending=[False, True, False],
        na_position="last",
    )


def aspectos_por_documento(clausulas: pd.DataFrame, maximo: int = 3) -> pd.DataFrame:
    """Hasta ``maximo`` aspectos por documento, por frecuencia (esquema TOPICO)."""
    asignadas = clausulas[clausulas["topico"] >= 0]
    if asignadas.empty:
        return pd.DataFrame(columns=["doc_id", "topicos", "polaridad_media"])
    return (
        asignadas.groupby("doc_id")
        .agg(
            topicos=("topico", lambda s: list(s.value_counts().index[:maximo])),
            polaridad_media=("polaridad", "mean"),
        )
        .reset_index()
    )


# ═══════════════════════════════════════════════════════════════════════════
# 14. RESULTADO, AUDITORÍA Y EXPORTACIÓN
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Resultado:  # pylint: disable=too-many-instance-attributes
    """Todo lo que produce UNA corrida sobre UNA encuesta."""

    encuesta: str
    cfg: Config
    run_id: str
    huellas: dict[str, str]
    documentos: pd.DataFrame
    clausulas: pd.DataFrame
    topicos: pd.DataFrame
    absa: pd.DataFrame
    metricas: dict[str, Any]
    validacion: dict[str, Any]
    conciliacion: pd.DataFrame
    scorecard: dict[str, Any]
    muestra_pureza: pd.DataFrame
    muestra_manual: pd.DataFrame
    grid: pd.DataFrame = field(default_factory=pd.DataFrame)
    modelador: ModeladorAspectos | None = None
    modo: str = MODO_BERTOPIC
    #: ft-r4.15 (modo u4): tópicos del residual (una fila por tópico), vacío en modo bertopic.
    topicos_residuales: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: ft-r4.15 (modo u4): modelo U4, parámetros y métricas del residual, huella del archivo auditado.
    u4: dict[str, Any] = field(default_factory=dict)


def tabla_topicos(
    modelador: ModeladorAspectos | None, clausulas: pd.DataFrame
) -> pd.DataFrame:
    """Una fila por tópico con tamaño, términos, cohesión y ejemplos."""
    columnas = [
        "topico",
        "etiqueta",
        "terminos",
        "clausulas",
        "documentos",
        "textos_unicos",
        "cohesion",
        "crecimiento_reasignacion_pct",
        "ejemplos",
    ]
    asignadas = clausulas[clausulas["topico"] >= 0]
    if modelador is None or asignadas.empty or modelador.etiquetas is None:
        return pd.DataFrame(columns=columnas)
    cohesion = modelador.cohesion()
    unicos = pd.Series(modelador.etiquetas).value_counts()
    filas = []
    for topico, grupo in asignadas.groupby("topico"):
        terminos = modelador.terminos(int(topico), 8)
        ejemplos = (
            grupo["clausula"]
            .drop_duplicates()
            .sample(
                min(3, grupo["clausula"].nunique()),
                random_state=modelador.cfg.random_state,
            )
            .tolist()
        )
        filas.append(
            {
                "topico": int(topico),
                "etiqueta": f"T{int(topico)} · " + " · ".join(terminos[:3]),
                "terminos": " | ".join(terminos),
                "clausulas": int(len(grupo)),
                "documentos": int(grupo["doc_id"].nunique()),
                "textos_unicos": int(unicos.get(topico, 0)),
                "cohesion": cohesion.get(int(topico), float("nan")),
                "crecimiento_reasignacion_pct": modelador.crecimiento.get(
                    int(topico), 0.0
                ),
                "ejemplos": " || ".join(e[:120] for e in ejemplos),
            }
        )
    tabla = pd.DataFrame(filas, columns=columnas)
    umbral = (
        tabla["cohesion"].mean() - modelador.cfg.k_cohesion * tabla["cohesion"].std()
    )
    tabla["revisar"] = tabla["cohesion"] < umbral  # candidatos a baja pureza
    return tabla.sort_values("clausulas", ascending=False).reset_index(drop=True)


def conciliar(
    documentos: pd.DataFrame,
    clausulas: pd.DataFrame,
    absa: pd.DataFrame,
    chequeos_extra: dict[str, bool] | None = None,
) -> pd.DataFrame:
    """Cada denominador del informe y su comprobación aritmética.

    Args:
        documentos: Tabla de documentos con ``estado``.
        clausulas: Tabla de cláusulas con ``motivo``, ``topico`` y
            ``sentimiento``.
        absa: Matriz ABSA.
        chequeos_extra: Comprobaciones adicionales (p. ej. identidad con
            la fila del grid).

    Returns:
        Tabla nivel | concepto | valor; las filas ``chequeo`` valen 1 o 0.
    """
    estados = documentos["estado"].value_counts()
    motivos = clausulas["motivo"].value_counts()
    topico = clausulas["topico"]
    con_sentimiento = int(clausulas["sentimiento"].notna().sum())
    # ft-r4.15: en modo u4 las VALORACION_GLOBAL de respuestas cortas pasaban por U4 (regla 4c) y salían del código -3. Con
    # ``sin_contenido_en_cortas`` (punto 7) ya no pasan: solo cuentan las que SÍ salieron del código -3.
    es_u4 = "aspecto_u4" in clausulas.columns
    n_global_corta = (
        int(
            (
                (clausulas["motivo"] == MOTIVO_GLOBAL)
                & clausulas["respuesta_corta"].fillna(False).astype(bool)
                & (clausulas["topico"] != VALORACION_GLOBAL)
            ).sum()
        )
        if es_u4 and "respuesta_corta" in clausulas.columns
        else 0
    )
    filas = [
        ("documentos", "filas del corpus", len(documentos)),
        *[
            ("documentos", f"estado {e}", int(estados.get(e, 0)))
            for e in (DOC_VACIA, DOC_RUIDO, DOC_NO_OPINION, DOC_CON_OPINION)
        ],
        ("clausulas", "total", len(clausulas)),
        *[
            ("clausulas", f"motivo {m}", int(motivos.get(m, 0)))
            for m in (
                MOTIVO_RUIDO,
                MOTIVO_NO_OPINION,
                MOTIVO_GLOBAL,
                MOTIVO_ASPECTO,
            )
        ],
        ("clausulas", "asignadas a tópico", int((topico >= 0).sum())),
        ("clausulas", "outliers (-1)", int((topico == OUTLIER).sum())),
        (
            "clausulas",
            "valoración global (-3)",
            int((topico == VALORACION_GLOBAL).sum()),
        ),
        ("clausulas", "excluidas (-2)", int((topico == EXCLUIDO).sum())),
        *(
            [
                ("clausulas", "comentario general (-5)", int((topico == COMENTARIO_GENERAL).sum())),
                ("clausulas", "por descubrir (-6)", int((topico == POR_DESCUBRIR).sum())),
                ("clausulas", "valoración global en respuesta corta (pasan por U4)", int(n_global_corta)),
            ]
            if es_u4
            else []
        ),
        ("sentimiento", "cláusulas clasificadas", con_sentimiento),
        (
            "absa",
            "cláusulas en la matriz",
            int(absa["total_clausulas"].sum()) if not absa.empty else 0,
        ),
    ]
    tabla = pd.DataFrame(filas, columns=["nivel", "concepto", "valor"])
    chequeos = {
        "docs: estados suman filas": sum(
            int(estados.get(e, 0))
            for e in (DOC_VACIA, DOC_RUIDO, DOC_NO_OPINION, DOC_CON_OPINION)
        )
        == len(documentos),
        "cláusulas: motivos suman total": int(motivos.sum()) == len(clausulas),
        **(
            {
                # u4: ASPECTO + VALORACION_GLOBAL de respuesta corta = asignadas + comentario general + por descubrir
                "aspecto (+ global corto) = asignadas + comentario general + por descubrir + outliers": int(
                    motivos.get(MOTIVO_ASPECTO, 0)
                )
                + n_global_corta
                == int(
                    (topico >= 0).sum()
                    + (topico == OUTLIER).sum()
                    + (topico == COMENTARIO_GENERAL).sum()
                    + (topico == POR_DESCUBRIR).sum()
                ),
                "global = código -3 + global de respuesta corta": int(motivos.get(MOTIVO_GLOBAL, 0))
                == int((topico == VALORACION_GLOBAL).sum()) + n_global_corta,
            }
            if es_u4
            else {
                "aspecto = asignadas + outliers": int(motivos.get(MOTIVO_ASPECTO, 0))
                == int((topico >= 0).sum() + (topico == OUTLIER).sum()),
                "global = código -3": int(motivos.get(MOTIVO_GLOBAL, 0))
                == int((topico == VALORACION_GLOBAL).sum()),
            }
        ),
        "sentimiento = aspecto + global": con_sentimiento
        == int(motivos.get(MOTIVO_ASPECTO, 0) + motivos.get(MOTIVO_GLOBAL, 0)),
        "absa = sentimiento": absa.empty
        or int(absa["total_clausulas"].sum()) == con_sentimiento,
        **(chequeos_extra or {}),
    }
    cheq = pd.DataFrame(
        [("chequeo", k, int(v)) for k, v in chequeos.items()],
        columns=["nivel", "concepto", "valor"],
    )
    return pd.concat([tabla, cheq], ignore_index=True)


@dataclass
class Puerta:
    """Una condición verificable del scorecard."""

    clave: str
    descripcion: str
    valor: float | None
    umbral: str
    estado: str  # PASA | FALLA | PENDIENTE | NO_APLICA
    peso: float


def _puerta(
    clave: str,
    descripcion: str,
    valor: float | None,
    cumple: Callable[[float], bool],
    umbral: str,
    peso: float,
    *,
    aplica: bool = True,
    motivo: str = "",
) -> Puerta:
    """Construye una puerta; PENDIENTE si no es medible, NO_APLICA si no rige.

    ``motivo`` (ft-r4.15): por qué no rige; se escribe en ``umbral`` (solo si se da).
    """
    if not aplica:
        return Puerta(
            clave,
            descripcion,
            None,
            f"NO_APLICA: {motivo}" if motivo else umbral,
            "NO_APLICA",
            peso,
        )
    if valor is None or (isinstance(valor, float) and not np.isfinite(valor)):
        return Puerta(clave, descripcion, None, umbral, "PENDIENTE", peso)
    return Puerta(
        clave,
        descripcion,
        round(float(valor), 4),
        umbral,
        "PASA" if cumple(float(valor)) else "FALLA",
        peso,
    )


def _puertas_u4(metricas: dict[str, Any]) -> list[Puerta]:
    """Puertas informativas del modo u4: siempre PENDIENTE (el equipo aún no fija umbrales); muestran el valor."""

    def _p(clave: str, descripcion: str, valor: Any) -> Puerta:
        ok = isinstance(valor, (int, float, np.floating)) and np.isfinite(valor)
        return Puerta(
            clave,
            descripcion,
            round(float(valor), 4) if ok else None,
            "sin umbral definido (informativa; lo fija el equipo)",
            "PENDIENTE",
            0.0,
        )

    return [
        _p("U1_acierto_holdout", "Acierto de U4 en el hold-out (una sola evaluación)", metricas.get("acierto_holdout")),
        _p("U2_f1_macro_holdout", "F1 macro de U4 en el hold-out", metricas.get("f1_macro_holdout")),
        _p("U3_cobertura_u4", "% de cláusulas sujetas a aspecto con un aspecto (código >= 0)", metricas.get("cobertura_u4")),
        _p("U4_pct_comentario_general", "% de cláusulas sujetas a aspecto en «Comentario General»", metricas.get("pct_comentario_general")),
        _p(
            "U5_pct_aspecto_lista",
            "% de cláusulas con aspecto DE LA LISTA asignado (excluye «Por descubrir», «Comentario General» y las clases sin aspecto)",
            metricas.get("pct_aspecto_lista"),
        ),
        _p("U6_pct_por_descubrir", "% de cláusulas sujetas a aspecto en «Por descubrir» (residual)", metricas.get("pct_por_descubrir")),
    ]


def construir_scorecard(
    metricas: dict[str, Any],
    validacion: dict[str, Any],
    conciliacion: pd.DataFrame,
    pureza: dict[str, Any] | None = None,
    f1_manual: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Puertas numéricas y nota 1-10 derivada de ellas.

    La nota es la fracción ponderada de puertas que pasan entre las
    MEDIDAS. Dos topes explícitos: sin integridad (conciliación) la nota no
    supera 6; con validación humana pendiente (pureza o F1) no supera 8.

    ft-r4.15 (modo u4): la fórmula de la nota NO cambia (fracción ponderada de las puertas PASA/FALLA). Las puertas
    geométricas quedan NO_APLICA y se añaden seis puertas informativas ``U1``–``U6`` en estado PENDIENTE (sin
    umbral definido por el equipo: muestran el valor pero NO cuentan en la nota; así la nota no se infla).
    """
    chequeos = conciliacion[conciliacion["nivel"] == "chequeo"]["valor"]
    pureza = pureza or {}
    f1_manual = f1_manual or {}
    # Sin modelo de tópicos (corpus pequeño) las puertas de aspectos no
    # rigen; sin nota válida (texto del docente, o sin columna) tampoco las
    # de validación externa. NO_APLICA no cuenta en la nota.
    geometria = metricas.get("modo_aspectos") == MODO_BERTOPIC
    # ft-r4.15: en modo u4 no hay partición UMAP/HDBSCAN que medir (silueta, ARI, outliers, concentración,
    # redundancia, pureza de tópicos): las puertas G2–G7 y G10 quedan NO_APLICA con su motivo escrito.
    modo_u4 = metricas.get("modo_aspectos") == MODO_U4
    motivo_geo = (
        "modo u4: los aspectos los asigna un clasificador supervisado (sin UMAP/HDBSCAN); "
        "el residual «Por descubrir» se mide aparte (topicos_residuales.csv)"
        if modo_u4
        else ""
    )
    valida_nota = bool(validacion.get("disponible"))
    tipo_pregunta = str(metricas.get("tipo_pregunta", "ABIERTA"))
    umbral_rho, umbral_auc = UMBRALES_VALIDACION.get(
        tipo_pregunta, UMBRALES_VALIDACION["ABIERTA"]
    )
    puertas = [
        _puerta(
            "G1_conciliacion",
            "Todos los denominadores cuadran",
            float(chequeos.min()) if len(chequeos) else None,
            lambda v: v == 1,
            "= 1",
            1.0,
        ),
        _puerta(
            "G2_estabilidad",
            "ARI medio entre semillas UMAP",
            metricas.get("ari_medio"),
            lambda v: v >= 0.70,
            ">= 0.70",
            1.0,
            aplica=geometria,
            motivo=motivo_geo,
        ),
        _puerta(
            "G3_cobertura_aspectual",
            "% cláusulas con aspecto asignadas a tópico",
            metricas.get("cobertura_aspectual"),
            lambda v: v >= 75,
            ">= 75",
            1.5,
            aplica=geometria,
            motivo=motivo_geo,
        ),
        _puerta(
            "G4_concentracion",
            "% del tópico mayor",
            metricas.get("pct_top1"),
            lambda v: v <= 25,
            "<= 25",
            1.0,
            aplica=geometria,
            motivo=motivo_geo,
        ),
        _puerta(
            "G5_silueta",
            "Silueta en espacio UMAP",
            metricas.get("silueta"),
            lambda v: v >= 0.10,
            ">= 0.10",
            1.0,
            aplica=geometria,
            motivo=motivo_geo,
        ),
        _puerta(
            "G6_n_topicos",
            "Número de tópicos legible",
            metricas.get("n_topicos"),
            lambda v: 8 <= v <= 30,
            "8-30",
            0.5,
            aplica=geometria,
            motivo=motivo_geo,
        ),
        _puerta(
            "G7_redundancia",
            "Máximo coseno entre centroides",
            metricas.get("max_coseno"),
            lambda v: v < 0.90,
            "< 0.90",
            0.5,
            aplica=geometria,
            motivo=motivo_geo,
        ),
        _puerta(
            "G8_spearman_doc",
            "Spearman polaridad-nota (documento)",
            validacion.get("spearman_documento"),
            lambda v: v >= umbral_rho,
            f">= {umbral_rho} ({tipo_pregunta})",
            1.0,
            aplica=valida_nota,
        ),
        _puerta(
            "G9_auc_nota_baja",
            "AUC detectando nota baja",
            validacion.get("auc_nota_baja"),
            lambda v: v >= umbral_auc,
            f">= {umbral_auc} ({tipo_pregunta})",
            1.0,
            aplica=valida_nota,
        ),
        _puerta(
            "G10_pureza_humana",
            "% tópicos con >=7/10 cláusulas pertinentes",
            pureza.get("pct_topicos_puros"),
            lambda v: v >= 85,
            ">= 85",
            1.5,
            aplica=geometria,
            motivo=motivo_geo,
        ),
        _puerta(
            "G11_f1_manual",
            "F1 macro vs. muestra anotada (reponderado a población)",
            f1_manual.get("f1_macro"),
            lambda v: v >= 0.75,
            ">= 0.75",
            1.0,
        ),
    ]
    if modo_u4:
        puertas += _puertas_u4(metricas)
    medidas = [p for p in puertas if p.estado in ("PASA", "FALLA")]
    total = sum(p.peso for p in medidas)
    nota = (
        10 * sum(p.peso for p in medidas if p.estado == "PASA") / total
        if total
        else 0.0
    )
    topes: list[str] = []
    estado = {p.clave: p.estado for p in puertas}
    if estado["G1_conciliacion"] != "PASA":
        nota, topes = min(nota, 6.0), topes + ["integridad: nota <= 6"]
    if "PENDIENTE" in (estado["G10_pureza_humana"], estado["G11_f1_manual"]):
        nota, topes = min(nota, 8.0), topes + ["validación humana pendiente: nota <= 8"]
    return {
        "nota": round(max(1.0, nota), 1),
        "topes_aplicados": topes,
        "puertas": [asdict(p) for p in puertas],
        "pasan": sum(p.estado == "PASA" for p in puertas),
        "fallan": sum(p.estado == "FALLA" for p in puertas),
        "pendientes": sum(p.estado == "PENDIENTE" for p in puertas),
        "no_aplican": sum(p.estado == "NO_APLICA" for p in puertas),
    }


def muestra_pureza(
    clausulas: pd.DataFrame, topicos: pd.DataFrame, n: int, semilla: int
) -> pd.DataFrame:
    """N cláusulas ALEATORIAS por tópico para juicio humano de pertinencia.

    Aleatorias y no representativas: las representativas de BERTopic se
    ven bien por construcción y no auditan nada.
    """
    asignadas = clausulas[clausulas["topico"] >= 0]
    if asignadas.empty:
        return pd.DataFrame(columns=["topico", "etiqueta", "clausula", "pertenece_s_n"])
    nombres = (
        topicos.set_index("topico")["etiqueta"]
        if not topicos.empty
        else pd.Series(dtype=str)
    )
    partes = [
        g.sample(min(n, len(g)), random_state=semilla).assign(
            etiqueta=nombres.get(t, str(t))
        )
        for t, g in asignadas.groupby("topico")
    ]
    salida = pd.concat(partes)[["topico", "etiqueta", "clausula"]].reset_index(
        drop=True
    )
    salida["pertenece_s_n"] = ""
    return salida


def evaluar_pureza(
    ruta_o_marco: Path | str | pd.DataFrame, minimo: int = 7
) -> dict[str, Any]:
    """% de tópicos con al menos ``minimo`` cláusulas marcadas "S" de 10."""
    marco = (
        ruta_o_marco
        if isinstance(ruta_o_marco, pd.DataFrame)
        else pd.read_excel(ruta_o_marco, sheet_name="muestra_pureza")
    )
    marco = marco.assign(
        pertenece_s_n=marco["pertenece_s_n"].astype(str).str.strip().str.upper()
    )
    juzgadas = marco[marco["pertenece_s_n"].isin(["S", "N"])]
    if juzgadas.empty:
        return {"disponible": False}
    por_topico = juzgadas.groupby("topico")["pertenece_s_n"].apply(
        lambda s: (s == "S").sum() / len(s) * 10
    )
    return {
        "disponible": True,
        "topicos_juzgados": int(len(por_topico)),
        "pct_topicos_puros": round(100 * float((por_topico >= minimo).mean()), 1),
        "pureza_media_sobre_10": round(float(por_topico.mean()), 2),
        "detalle": por_topico.round(1).rename("pertinentes_de_10"),
    }


def _a_json(valor: Any) -> Any:
    """Convierte tipos numpy/pandas a JSON (tablas de detalle se omiten)."""
    if isinstance(valor, dict):
        return {
            k: _a_json(v)
            for k, v in valor.items()
            if not isinstance(v, (pd.DataFrame, pd.Series))
        }
    if isinstance(valor, (list, tuple)):
        return [_a_json(v) for v in valor]
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.floating, float)):
        return None if not np.isfinite(valor) else float(valor)
    if isinstance(valor, (np.bool_,)):
        return bool(valor)
    if isinstance(valor, Path):
        return str(valor)
    return valor


def _verificar_union_clausulas_documentos(clausulas: pd.DataFrame, documentos: pd.DataFrame) -> None:
    """Todo ``doc_id`` de ``clausulas`` existe en ``documentos`` (llave única) y el merge entrega ``respuesta_completa``.

    No exige que ``clausula`` sea subcadena literal de ``respuesta_completa`` (las reglas R1–R5 pueden reconstruir
    cláusulas); solo reporta en el log el % que sí lo es.
    """
    if documentos["doc_id"].duplicated().any():
        raise RuntimeError("documentos: doc_id no es único (no sirve de llave de unión con clausulas).")
    faltan = set(clausulas["doc_id"]) - set(documentos["doc_id"])
    if faltan:
        raise RuntimeError(f"{len(faltan)} doc_id de clausulas no existen en documentos (p. ej. {sorted(faltan)[:5]}).")
    if "respuesta_completa" not in documentos.columns:
        raise RuntimeError("documentos no trae respuesta_completa (modo u4).")
    unido = clausulas[["doc_id", "clausula"]].merge(
        documentos[["doc_id", "respuesta_completa"]], on="doc_id", how="left", validate="many_to_one"
    )
    if len(unido) != len(clausulas) or unido["respuesta_completa"].isna().any():
        raise RuntimeError("clausulas ⟕ documentos por doc_id no entrega respuesta_completa en todas las filas.")
    es_sub = [str(c) in str(r) for c, r in zip(unido["clausula"], unido["respuesta_completa"])]
    logger.info(
        "  unión clausulas ⟕ documentos: OK (%d filas). La cláusula es subcadena LITERAL de la respuesta completa en %.1f%% "
        "(el resto fue reconstruido por las reglas R1–R5/R7–R15 o por la normalización).",
        len(unido),
        100 * float(np.mean(es_sub)) if es_sub else float("nan"),
    )


def exportar_resultado(resultado: Resultado, raiz_salida: Path | None = None) -> Path:
    """Escribe TODOS los artefactos de una corrida desde un único estado.

    Tras escribir, relee parquet y CSV y verifica que el tópico y el
    sentimiento de cada cláusula coinciden: el fallo H1 de la auditoría
    (17% de coincidencia) queda imposible sin que la exportación lo detecte.

    Returns:
        Carpeta de la corrida.

    Raises:
        RuntimeError: Si la relectura no coincide con lo exportado.
    """
    base = Path(raiz_salida or resultado.cfg.output_dir)
    destino = base / slug(resultado.encuesta) / resultado.run_id
    destino.mkdir(parents=True, exist_ok=True)
    clausulas = resultado.clausulas
    es_u4 = resultado.modo == MODO_U4
    if es_u4:  # ft-r4.15: clausulas ⟕ documentos por doc_id debe entregar respuesta_completa en TODAS las filas
        _verificar_union_clausulas_documentos(clausulas, resultado.documentos)
    clausulas.to_csv(destino / "clausulas.csv", index=False, encoding="utf-8")
    resultado.documentos.to_csv(destino / "documentos.csv", index=False)
    resultado.topicos.to_csv(destino / "topicos.csv", index=False, encoding="utf-8")
    resultado.absa.to_csv(destino / "absa.csv", encoding="utf-8")
    if not resultado.grid.empty:
        resultado.grid.to_csv(destino / "grid.csv", index=False)
    if es_u4:
        resultado.topicos_residuales.to_csv(destino / "topicos_residuales.csv", index=False, encoding="utf-8")

    claves = ["doc_id", "clausula_id", "topico"]
    extra_relectura = ["aspecto"] if "aspecto" in clausulas.columns else []
    releido_csv = pd.read_csv(
        destino / "clausulas.csv",
        usecols=claves + ["sentimiento"] + extra_relectura,
        keep_default_na=False if extra_relectura else True,
        na_values=[""] if extra_relectura else None,
    )
    try:
        clausulas.to_parquet(destino / "clausulas.parquet", index=False)
        releido_pq = pd.read_parquet(
            destino / "clausulas.parquet", columns=claves + ["sentimiento"] + extra_relectura
        )
        coincide = (
            releido_pq[claves].equals(releido_csv[claves])
            and (
                releido_pq["sentimiento"].fillna("").to_numpy()
                == releido_csv["sentimiento"].fillna("").to_numpy()
            ).all()
            and all(
                (
                    releido_pq[c].fillna("").astype(str).to_numpy()
                    == releido_csv[c].fillna("").astype(str).to_numpy()
                ).all()
                for c in extra_relectura
            )
        )
        if not coincide:
            raise RuntimeError(f"parquet y CSV no coinciden en {destino}")
    except ImportError:
        logger.warning("pyarrow no disponible: se omite clausulas.parquet")
    if not releido_csv[claves].equals(clausulas[claves].reset_index(drop=True)):
        raise RuntimeError(
            f"El CSV releído no coincide con el estado en memoria ({destino})"
        )

    manifiesto = {
        "version": VERSION,
        "run_id": resultado.run_id,
        "encuesta": resultado.encuesta,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "huellas": resultado.huellas,
        "config": resultado.cfg.to_dict(),
        "metricas": resultado.metricas,
        "modo_aspectos": resultado.modo,
        "validacion_sentimiento": resultado.validacion,
        **(
            {"modelo_u4_id": resultado.u4.get("modelo_u4_id"), "u4": resultado.u4}
            if es_u4 and resultado.u4
            else {}
        ),
    }
    (destino / "manifest.json").write_text(
        json.dumps(_a_json(manifiesto), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destino / "scorecard.json").write_text(
        json.dumps(_a_json(resultado.scorecard), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with pd.ExcelWriter(destino / "auditoria.xlsx") as libro:
        pd.DataFrame(
            {
                "campo": ["run_id", "encuesta", "version", "instruccion"],
                "valor": [
                    resultado.run_id,
                    resultado.encuesta,
                    VERSION,
                    "No edite esta hoja: identifica la corrida al recalificar.",
                ],
            }
        ).to_excel(libro, sheet_name="corrida", index=False)
        pd.DataFrame(resultado.scorecard["puertas"]).to_excel(
            libro, sheet_name="scorecard", index=False
        )
        resultado.conciliacion.to_excel(libro, sheet_name="conciliacion", index=False)
        resultado.topicos.to_excel(libro, sheet_name="topicos", index=False)
        if es_u4:
            resultado.topicos_residuales.to_excel(libro, sheet_name="topicos_residuales", index=False)
        resultado.absa.to_excel(libro, sheet_name="absa")
        resultado.muestra_pureza.to_excel(
            libro, sheet_name="muestra_pureza", index=False
        )
        resultado.muestra_manual.to_excel(
            libro, sheet_name="muestra_manual", index=False
        )
        for clave in ("por_polaridad", "deciles"):
            tabla = resultado.validacion.get(clave)
            if isinstance(tabla, pd.DataFrame):
                tabla.to_excel(libro, sheet_name=f"nota_{clave}")
        if not resultado.grid.empty:
            resultado.grid.to_excel(libro, sheet_name="grid", index=False)
    logger.info(
        "Artefactos de '%s' en %s (nota %.1f)",
        resultado.encuesta,
        destino,
        resultado.scorecard["nota"],
    )
    return destino


def comprimir_carpeta(carpeta: Path | str, destino_zip: Path | str) -> Path:
    """Comprime ``carpeta`` en ``destino_zip`` (puede quedar dentro de ella) sin meter ningún ``.zip`` dentro del zip.

    Se usa para empaquetar ``data/processed/nlp/modelado`` (Pasos 15 y 16 del cuaderno): el zip se guarda junto a las
    corridas y no se incluye a sí mismo ni a zips anteriores. Las rutas dentro del zip son relativas a ``carpeta``.

    Returns:
        Ruta del zip escrito.
    """
    import zipfile  # pylint: disable=import-outside-toplevel

    carpeta, destino_zip = Path(carpeta), Path(destino_zip)
    destino_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destino_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(carpeta.rglob("*")):
            if f.is_file() and f.suffix.lower() != ".zip":
                z.write(f, f.relative_to(carpeta).as_posix())
    return destino_zip


# ═══════════════════════════════════════════════════════════════════════════
# 14b. ASPECTOS SUPERVISADOS · U4 (ft-r4.15)
# ═══════════════════════════════════════════════════════════════════════════
# U4 = encoder mpnet (paraphrase-multilingual-mpnet-base-v2) con las ÚLTIMAS 4 capas descongeladas + mean pooling +
# dropout + cabeza lineal. Se entrena POR PREGUNTA, SOLO con etiquetas auditadas a mano (un Excel por pregunta con
# el mismo formato que el de obs19). Sustituye a UMAP/HDBSCAN como mecanismo de asignación de aspectos
# (``Config.modo_aspectos="u4"``); el código de UMAP/HDBSCAN sigue disponible (modo "bertopic") y se usa, sobre el
# «residual», para descubrir aspectos nuevos.
#
# ETAPAS
#   A. ``entrenar_u4``          -> entrena, evalúa UNA vez en el hold-out, guarda el modelo y lo verifica (ida y vuelta).
#   B. ``cargar_u4``            -> reconstruye el modelo guardado (sirve para etiquetar sin reentrenar).
#      ``etiquetar_corpus_u4``  -> etiqueta TODO el corpus de la pregunta + reglas (Comentario General).
#   B2.``modelar_residual_u4``  -> UMAP/HDBSCAN sobre el residual («Por descubrir»).
#   C. ``procesar_encuesta``    -> las usa cuando ``Config.modo_aspectos == "u4"``.
#
# ``torch``/``transformers`` se importan DENTRO de cada función: importar el módulo no los exige.

#: Hiperparámetros de U4 (los del experimento de ``regresion logistica/prisma_finetuning_pruebas-regresionlogistica.ipynb``).
HIPERPARAMETROS_U4: dict[str, Any] = {
    "k": 4,  # capas finales del encoder que se descongelan
    "lr_enc": 2e-5,
    "lr_cabeza": 1e-3,
    "max_length": 64,
    "batch": 32,
    "epocas_max": 10,
    "paciencia": 3,
    "weight_decay": 0.01,
    "warmup": 0.10,
    "label_smoothing": 0.1,
    "dropout_cabeza": 0.1,
    "fp16": True,
    "n_interno": 8,  # validación interna agrupada ≈ 1/8 del entrenamiento
    "frac_holdout": 0.15,
    # Semillas de los CORTES de datos: las del cuaderno de regresión logística (mismos textos en los mismos conjuntos).
    "semilla_cv": 7,
    "semilla_holdout": 2024,
    "n_min_concluyente": 10,  # clases con n < 10 en el hold-out: «no concluyente»
    "lote_etiquetado": 128,
}
#: Clases de U4 que NO son un aspecto (opinión sin aspecto concreto): se reportan, no se priorizan en el ABSA.
CLASES_SIN_ASPECTO_U4: tuple[str, ...] = ("Opinión general", "Sin aspecto", "Ninguno de la lista")
#: Cambia solo si cambia el formato de los archivos del modelo guardado (entra en la huella de entrenamiento).
_FORMATO_MODELO_U4 = 1
_UMBRALES_DIAGNOSTICO = (0.3, 0.4, 0.5, 0.6, 0.7)
#: Criterios RELAJADOS del modelado del residual. Razón: el residual es pequeño y heterogéneo por construcción
#: (son las cláusulas largas que U4 no reconoce con seguridad); exigirle 8–30 tópicos, silueta ≥ 0,10 y ARI ≥ 0,70
#: como al corpus completo lo haría fallar siempre. Se mantienen las puertas, con rangos acordes a su tamaño.
CRITERIOS_RESIDUAL_POR_DEFECTO: dict[str, Any] = {
    "n_topicos_min": 2,
    "n_topicos_max": 12,
    "pct_outliers_max": 40.0,
    "pct_topico_mayor_max": 60.0,
    "pct_top2_max": 85.0,
    "silueta_min": 0.0,
    "ari_min": 0.50,
}


class ErrorEtiquetasU4(ValueError):
    """Archivo de etiquetas auditado ausente o incompleto (hoja o columnas que faltan)."""


def norm_txt_u4(texto: Any) -> str:
    """Minúsculas, sin tildes y con espacios colapsados (comparación aproximada de textos)."""
    t = unicodedata.normalize("NFKD", str(texto).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", t).strip()


def clave_logica(nombre_encuesta: str) -> str:
    """``"Evaluación Docente :: obs19_aspectos_resaltar"`` -> ``"obs19_aspectos_resaltar"``."""
    return str(nombre_encuesta).split(" :: ")[-1].strip()


@dataclass
class ParametrosU4:  # pylint: disable=too-many-instance-attributes
    """``PERFILES[clave]['aspectos_u4']``: de dónde salen las etiquetas y cómo se entrena. Sin lógica de obs19."""

    #: Nombre de la encuesta en el Excel de validación y en el nombre del archivo auditado ("obs19", "acreditacion"...).
    encuesta_etiquetas: str = ""
    #: Ruta explícita (absoluta o relativa a RAIZ). None = patrón ``--Prisma_Validacion_Humana-<encuesta>*auditado*.xlsx``
    #: en ``carpeta_etiquetas`` (si hay varios, el de fecha de modificación más reciente; se imprime cuál).
    archivo_etiquetas: str | None = None
    carpeta_etiquetas: str = "data/etiquetas_manuales"
    hoja_etiquetas: str = "obs19"
    col_fragmento: str = "fragmento_final_entrenamiento"
    col_aspecto: str = "aspecto_final_entrenamiento"
    col_respuesta: str = "respuesta_completa_original"
    clases_sin_aspecto: tuple[str, ...] = CLASES_SIN_ASPECTO_U4
    #: Sobrescribe claves de ``HIPERPARAMETROS_U4`` (p. ej. ``{"k": 4}``). Vacío = los de U4.
    hiperparametros: dict[str, Any] = field(default_factory=dict)
    encoder: str = ENCODER_POR_DEFECTO
    semilla: int = 42
    umbral_comentario_general: float = 0.50
    max_palabras_respuesta_corta: int = 5
    reutilizar_modelo: bool = True
    permitir_cpu: bool = False
    #: Solo pruebas de humo: submuestra de filas del archivo auditado.
    n_filas_prueba: int | None = None
    activo: bool = True

    @classmethod
    def desde(cls, valor: ParametrosU4 | dict[str, Any] | None, clave: str = "") -> ParametrosU4:
        """Acepta un dict de ``PERFILES`` (ignora llaves ajenas, p. ej. 'activo' o comentarios) o una instancia."""
        if isinstance(valor, ParametrosU4):
            return valor
        datos = dict(valor or {})
        conocidos = {f for f in cls.__dataclass_fields__}  # pylint: disable=no-member
        extra = sorted(set(datos) - conocidos)
        if extra:
            logger.warning("aspectos_u4[%s]: llaves desconocidas ignoradas: %s", clave, extra)
        parametros = cls(**{k: v for k, v in datos.items() if k in conocidos})
        parametros.clases_sin_aspecto = tuple(parametros.clases_sin_aspecto)
        return parametros

    def hip(self) -> dict[str, Any]:
        """Hiperparámetros efectivos (defaults de U4 + sobrescritura del perfil)."""
        return {**HIPERPARAMETROS_U4, **dict(self.hiperparametros)}


# ── Rutas y comprobaciones de escritura ────────────────────────────────────


def asegurar_carpeta_escribible(carpeta: Path | str) -> Path:
    """Crea la carpeta (con padres) y comprueba que se puede escribir en ella. Falla DE INMEDIATO si no.

    En Colab, si la ruta cuelga de ``/content/drive`` y Drive no está montado, falla antes de crear nada
    (de lo contrario se crearía una carpeta local efímera que se pierde al cerrar la sesión).
    """
    carpeta = Path(carpeta)
    posix = carpeta.as_posix()
    if posix.startswith("/content/drive") and not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError(
            f"La ruta {carpeta} está en Google Drive pero Drive NO está montado "
            "(falta /content/drive/MyDrive). Ejecute el Paso 0 (drive.mount) antes de entrenar."
        )
    try:
        carpeta.mkdir(parents=True, exist_ok=True)
        prueba = carpeta / f".prueba_escritura_{os.getpid()}"
        prueba.write_text("ok", encoding="utf-8")
        prueba.unlink()
    except OSError as exc:
        raise RuntimeError(
            f"La carpeta de salida {carpeta} no existe o no es escribible ({type(exc).__name__}: {exc}). "
            "Se aborta ANTES de entrenar para no gastar tiempo de GPU."
        ) from exc
    return carpeta


def carpeta_modelos_u4(raiz: Path | str, clave: str) -> Path:
    """``<raiz>/modelos/u4_aspectos/<slug(clave)>`` (sin crearla). En el repo, ``raiz`` = ``models/``."""
    return Path(raiz) / "modelos" / "u4_aspectos" / slug(clave)


def _huella_modelo_u4(ruta: Path | str | None) -> str | None:
    """Huella del modelo U4 que usa una corrida (para ``Config.huella``): su configuración y sus clases, no su ruta."""
    if not ruta:
        return None
    try:
        carpeta = _resolver_carpeta_modelo(Path(ruta))
        cfg = (carpeta / "config_u4.json").read_text(encoding="utf-8")
        cls = (carpeta / "clases.json").read_text(encoding="utf-8")
        return huella_texto(cfg, cls, carpeta.name)
    except (OSError, FileNotFoundError, ValueError):
        return huella_texto(str(ruta))


def _resolver_carpeta_modelo(ruta: Path) -> Path:
    """Acepta la carpeta de una corrida (``.../<run_id>``) o la de la pregunta (con ``ultimo.json``)."""
    if (ruta / "ultimo.json").exists() and not (ruta / "config_u4.json").exists():
        ultimo = json.loads((ruta / "ultimo.json").read_text(encoding="utf-8"))
        destino = ruta / ultimo["run_id"]
        if not destino.exists():
            raise FileNotFoundError(f"ultimo.json apunta a {destino}, que no existe.")
        return destino
    if not (ruta / "config_u4.json").exists():
        raise FileNotFoundError(
            f"{ruta} no es un modelo U4 (falta config_u4.json) ni una carpeta con ultimo.json."
        )
    return ruta


# ── Datos de entrenamiento: SOLO archivos auditados ────────────────────────


def resolver_archivo_etiquetas(raiz: Path | str, params: ParametrosU4, clave: str = "") -> Path:
    """Ruta del archivo auditado de la pregunta. NUNCA cae a hojas no auditadas: si no existe, falla."""
    raiz = Path(raiz)
    if params.archivo_etiquetas:
        ruta = Path(params.archivo_etiquetas)
        ruta = ruta if ruta.is_absolute() else raiz / ruta
        if not ruta.exists():
            raise ErrorEtiquetasU4(
                f"[{clave}] No existe el archivo de etiquetas auditado indicado en PERFILES: {ruta}. "
                f"Se esperaba un Excel con la hoja '{params.hoja_etiquetas}' y las columnas "
                f"'{params.col_respuesta}', '{params.col_fragmento}', '{params.col_aspecto}'. "
                "NO se entrena con datos sin auditar."
            )
        return ruta
    carpeta = raiz / params.carpeta_etiquetas
    patron = re.compile(
        rf"^--Prisma_Validacion_Humana-{re.escape(params.encuesta_etiquetas)}(?![0-9A-Za-z]).*auditado.*\.xlsx$",
        re.IGNORECASE,
    )
    candidatos = (
        [carpeta / n for n in os.listdir(carpeta) if patron.match(n) and not n.startswith("~$")]
        if carpeta.is_dir()
        else []
    )
    if not candidatos:
        raise ErrorEtiquetasU4(
            f"[{clave}] No hay archivo de etiquetas auditado para '{params.encuesta_etiquetas}'. Se buscó "
            f"'--Prisma_Validacion_Humana-{params.encuesta_etiquetas}*auditado*.xlsx' en {carpeta}"
            f"{'' if carpeta.is_dir() else ' (la carpeta no existe)'}. Súbalo allí o indique "
            "PERFILES[clave]['aspectos_u4']['archivo_etiquetas']. NO se entrena con datos sin auditar "
            "(p. ej. las hojas sin auditar del Excel de validación)."
        )
    elegido = max(candidatos, key=lambda p: (p.stat().st_mtime, p.name))
    if len(candidatos) > 1:
        logger.warning(
            "[%s] %d archivos auditados coinciden; se usa el más reciente: %s (otros: %s)",
            clave,
            len(candidatos),
            elegido.name,
            sorted(c.name for c in candidatos if c != elegido),
        )
    print(f"[{clave}] archivo de etiquetas auditado elegido: {elegido}")
    return elegido


@dataclass
class DatosU4:
    """Datos auditados listos para entrenar: tabla por fragmento, clases y un informe de integridad."""

    tabla: pd.DataFrame
    clases: list[str]
    ruta: Path
    md5: str
    informe: dict[str, Any]


def _componentes_conexas(resp: Sequence[str], frag: Sequence[str]) -> np.ndarray:
    """Grupo = componente conexa del grafo respuesta–fragmento (misma respuesta o mismo texto ⇒ mismo grupo)."""
    padre: dict[Any, Any] = {}

    def raiz_de(x: Any) -> Any:
        padre.setdefault(x, x)
        r = x
        while padre[r] != r:
            r = padre[r]
        while padre[x] != r:
            padre[x], x = r, padre[x]
        return r

    for r_, f_ in zip(resp, frag):
        a, b = raiz_de(("r", r_)), raiz_de(("f", f_))
        if a != b:
            padre[a] = b
    ids: dict[Any, int] = {}
    return np.array([ids.setdefault(raiz_de(("r", r_)), len(ids)) for r_ in resp])


def cargar_datos_u4(raiz: Path | str, params: ParametrosU4, clave: str = "") -> DatosU4:
    """Carga el archivo AUDITADO de una pregunta. No limpia nada: el archivo ya trae texto y aspecto finales.

    Solo aplica guardas de integridad (descarta filas con fragmento o aspecto vacío y lo reporta; avisa de
    fragmentos repetidos con etiquetas distintas y los excluye de la evaluación; filas sin ``id`` se aceptan).
    Replica la lectura del cuaderno de regresión logística para que los cortes caigan sobre los mismos textos.

    Raises:
        ErrorEtiquetasU4: archivo ausente, sin la hoja/columnas esperadas o con las columnas que faltan.
    """
    raiz = Path(raiz)
    ruta = resolver_archivo_etiquetas(raiz, params, clave)
    try:
        t = pd.read_excel(ruta, sheet_name=params.hoja_etiquetas)
    except ValueError as exc:
        hojas = pd.ExcelFile(ruta).sheet_names
        raise ErrorEtiquetasU4(
            f"[{clave}] {ruta.name} no tiene la hoja '{params.hoja_etiquetas}'. Hojas: {hojas}."
        ) from exc
    esperadas = [params.col_respuesta, params.col_fragmento, params.col_aspecto]
    faltan = [c for c in esperadas if c not in t.columns]
    if faltan:
        raise ErrorEtiquetasU4(
            f"[{clave}] En {ruta.name}[{params.hoja_etiquetas}] faltan las columnas {faltan}. "
            f"Se esperaban (además de 'id' y 'revisor', opcionales): {esperadas}. Hay: {list(t.columns)}."
        )
    n_leidas = len(t)
    vacias = t[params.col_fragmento].isna() | t[params.col_aspecto].isna()
    t = t[~vacias]
    t = t[t[params.col_fragmento].astype(str).str.strip() != ""]
    n_descartadas = n_leidas - len(t)
    if params.n_filas_prueba:
        t = t.sample(min(params.n_filas_prueba, len(t)), random_state=0)
    if n_descartadas:
        logger.warning(
            "[%s] se descartaron %d de %d filas del auditado con fragmento o aspecto vacío.",
            clave,
            n_descartadas,
            n_leidas,
        )
    t = t.reset_index(drop=True)
    if t.empty:
        raise ErrorEtiquetasU4(f"[{clave}] {ruta.name}[{params.hoja_etiquetas}] no tiene filas con fragmento y aspecto.")

    # Vocabulario: SOLO las clases del archivo auditado, normalizando mayúsculas, tildes y espacios. Nombre canónico =
    # la grafía más frecuente del archivo (empate: la primera que aparece); las clases «sin aspecto» conservan la
    # grafía de ``clases_sin_aspecto``. Ya no se contrasta con la hoja «aspectos»: una errata se vuelve una clase propia.
    crudo = t[params.col_aspecto].astype(str).str.strip().map(lambda s_: re.sub(r"\s+", " ", s_))
    clave_norm = crudo.map(norm_txt_u4)
    cuentas = pd.DataFrame({"k": clave_norm, "g": crudo}).groupby(["k", "g"], sort=False).size().reset_index(name="n")
    sin_asp_norm = {norm_txt_u4(c): c for c in params.clases_sin_aspecto}
    canon: dict[str, str] = {}
    variantes: dict[str, dict[str, int]] = {}
    for k_, g_ in cuentas.groupby("k", sort=False):
        canon[k_] = sin_asp_norm.get(k_) or str(g_.sort_values("n", ascending=False, kind="stable").iloc[0]["g"])
        if len(g_) > 1:
            variantes[canon[k_]] = {str(a_): int(n_) for a_, n_ in zip(g_["g"], g_["n"])}
    etiqueta = clave_norm.map(canon)
    if variantes:
        logger.info("[%s] grafías normalizadas (mayúsculas/tildes) fusionadas en una sola clase: %s", clave, variantes)

    tiene_id = t["id"].notna() if "id" in t.columns else pd.Series(False, index=t.index)

    def _id(i: Any, n: int, ok: bool) -> str:
        if not ok:
            return f"aud-{n + 1:04d}"
        if isinstance(i, float) and float(i).is_integer():
            return str(int(i))
        return str(i)

    ids = [_id(i, n, ok) for n, (i, ok) in enumerate(zip(t["id"] if "id" in t.columns else [None] * len(t), tiene_id))]
    resp = t[params.col_respuesta]
    d = pd.DataFrame({"id": ids})
    d["revisor"] = t["revisor"].values if "revisor" in t.columns else np.nan
    d["respuesta_completa"] = np.where(tiene_id, resp, None)
    d["texto"] = t[params.col_fragmento].astype(str).str.strip().values
    d["etiqueta"] = etiqueta.values
    d["tipo_fila"] = np.where(tiene_id, "limpia", "anadida")
    d["motivo"] = np.where(tiene_id, "", "anadida_en_auditoria_sin_id_original")
    if "peso" in t.columns:
        d["peso"] = pd.to_numeric(t["peso"], errors="coerce").fillna(1.0).astype(float).values
    else:
        d["peso"] = 1.0
    # Mismo fragmento con etiquetas distintas tras la auditoría: AVISA, entra a entrenamiento con peso x0,5, FUERA de la evaluación.
    n_et = d.groupby("texto")["etiqueta"].nunique()
    conflicto = d["texto"].isin(n_et[n_et > 1].index)
    d.loc[conflicto, ["tipo_fila", "motivo"]] = ["conflicto", "etiquetas_contradictorias"]
    d.loc[conflicto, "peso"] = d.loc[conflicto, "peso"] * 0.5
    n_frag_conf = int(d.loc[conflicto, "texto"].nunique())
    if n_frag_conf:
        ejemplos = d.loc[conflicto].sort_values("texto").groupby("texto")["etiqueta"].agg(lambda s: sorted(set(s))).head(8)
        logger.warning(
            "[%s] %d fragmentos aparecen con etiquetas DISTINTAS tras la auditoría (%d filas): entran a entrenamiento "
            "con peso x0,5 y quedan FUERA de la evaluación. Ejemplos: %s",
            clave,
            n_frag_conf,
            int(conflicto.sum()),
            ejemplos.to_dict(),
        )

    aspectos_orden = sorted((k_ for k_ in canon if k_ not in sin_asp_norm), key=lambda k_: k_)
    sin_asp = sorted(k_ for k_ in canon if k_ in sin_asp_norm)
    # Orden determinista (= código de aspecto): aspectos alfabéticos por clave normalizada y luego las clases sin aspecto.
    # NO coincide con el orden del experimento de regresión logística (que seguía la hoja «aspectos»).
    clases = [canon[k_] for k_ in aspectos_orden + sin_asp]

    resp_clave = d["respuesta_completa"].fillna(d["id"]).astype(str)
    d["grupo"] = _componentes_conexas(resp_clave.tolist(), d["texto"].tolist())
    d["y"] = d["etiqueta"].map({c: i for i, c in enumerate(clases)}).astype(int)
    d["usa_evaluacion"] = d["tipo_fila"].isin(("limpia", "anadida"))
    md5 = hashlib.md5(ruta.read_bytes()).hexdigest()
    informe = {
        "archivo": str(ruta),
        "md5": md5,
        "hoja": params.hoja_etiquetas,
        "filas_leidas": int(n_leidas),
        "filas_descartadas_vacias": int(n_descartadas),
        "filas_usadas": int(len(d)),
        "filas_sin_id": int((d["tipo_fila"] == "anadida").sum()),
        "fragmentos_con_etiquetas_contradictorias": n_frag_conf,
        "filas_conflicto": int(conflicto.sum()),
        "n_por_clase": {c: int((d["etiqueta"] == c).sum()) for c in clases},
        "variantes_normalizadas": variantes,
        "clases_sin_aspecto_presentes": [canon[k_] for k_ in sin_asp],
    }
    return DatosU4(tabla=d, clases=clases, ruta=ruta, md5=md5, informe=informe)


def particionar_datos_u4(datos: DatosU4, hip: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hold-out agrupado (``frac_holdout``) y, dentro del resto, validación interna agrupada (``n_interno``).

    Replica ``correr_holdout``/``split_interno`` del cuaderno de regresión logística (rep=0, pliegue=-1):
    mismas semillas, así los mismos textos caen en los mismos conjuntos.

    Returns:
        Posiciones (en ``datos.tabla``) de ``(entrenamiento, validacion_interna, holdout)``.
    """
    from sklearn.model_selection import StratifiedGroupKFold  # pylint: disable=import-outside-toplevel

    d = datos.tabla
    y, g = d["y"].to_numpy(), d["grupo"].to_numpy()
    sg = StratifiedGroupKFold(
        n_splits=max(2, round(1 / float(hip["frac_holdout"]))), shuffle=True, random_state=int(hip["semilla_holdout"])
    )
    _, ih = next(sg.split(np.zeros(len(d)), y, g))
    hold = np.sort(ih)
    dev = np.setdiff1d(np.arange(len(d)), hold)
    if set(g[dev]) & set(g[hold]):
        raise RuntimeError("fuga de grupos al hold-out")
    ev = d["usa_evaluacion"].to_numpy()
    cand = dev[ev[dev]]
    sgi = StratifiedGroupKFold(
        n_splits=int(hip["n_interno"]),
        shuffle=True,
        random_state=int(hip["semilla_cv"]) * 1000 + 0 * 100 + (-1) + 50,  # = 7049 con la semilla del cuaderno
    )
    _, iv = next(sgi.split(np.zeros(len(cand)), y[cand], g[cand]))
    val = np.sort(cand[iv])
    tr = np.setdiff1d(dev, val)
    return tr, val, hold


# ── Modelo (torch se importa aquí, no al importar el módulo) ───────────────

_CLASE_CLASIFICADOR: Any = None


def _clase_clasificador() -> Any:
    """Define (una vez) ``ClasificadorU4``: backbone + mean pooling + dropout + capa lineal."""
    global _CLASE_CLASIFICADOR  # pylint: disable=global-statement
    if _CLASE_CLASIFICADOR is None:
        import torch.nn as nn  # pylint: disable=import-outside-toplevel

        class ClasificadorU4(nn.Module):  # type: ignore[misc]
            """Mean pooling de ``last_hidden_state`` -> dropout -> capa lineal."""

            def __init__(self, backbone: Any, n_clases: int, hidden: int, dropout: float) -> None:
                super().__init__()
                self.backbone = backbone
                self.drop = nn.Dropout(dropout)
                self.cabeza = nn.Linear(hidden, n_clases)

            def forward(self, ids: Any, mask: Any) -> Any:  # noqa: D102
                h = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
                m = mask.unsqueeze(-1).to(h.dtype)
                return self.cabeza(self.drop((h * m).sum(1) / m.sum(1).clamp(min=1e-9)))

        _CLASE_CLASIFICADOR = ClasificadorU4
    return _CLASE_CLASIFICADOR


def fabrica_encoder_hf(nombre: str) -> tuple[Any, Any]:
    """``(tokenizador, backbone)`` de Hugging Face. Inyectable en las pruebas (backbone de juguete)."""
    from transformers import AutoModel, AutoTokenizer  # pylint: disable=import-outside-toplevel

    return AutoTokenizer.from_pretrained(nombre), AutoModel.from_pretrained(nombre)


def _hidden_de(backbone: Any) -> int:
    return int(backbone.config.hidden_size)


def _construir_modelo_u4(backbone: Any, n_clases: int, k: int, dropout: float) -> Any:
    """Congela todo el backbone, descongela sus ``k`` últimas capas y añade la cabeza. (El llamador lo pasa a device.)"""
    capas = backbone.encoder.layer
    if not 1 <= k <= len(capas):
        raise ValueError(f"k={k} fuera de rango: el encoder tiene {len(capas)} capas.")
    for p in backbone.parameters():
        p.requires_grad = False
    for capa in capas[-k:]:
        for p in capa.parameters():
            p.requires_grad = True
    return _clase_clasificador()(backbone, n_clases, _hidden_de(backbone), dropout)


def _nombres_entrenables(modelo: Any) -> list[str]:
    return [n for n, p in modelo.named_parameters() if p.requires_grad]


def fijar_semillas_u4(semilla: int) -> None:
    """``random``, ``numpy``, ``torch`` y ``torch.cuda`` con la misma semilla (cabeza, dropout y orden de lotes)."""
    import random  # pylint: disable=import-outside-toplevel

    import torch  # pylint: disable=import-outside-toplevel

    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def _resolver_dispositivo(permitir_cpu: bool, para_entrenar: bool) -> str:
    import torch  # pylint: disable=import-outside-toplevel

    if torch.cuda.is_available():
        return "cuda"
    if para_entrenar and not permitir_cpu:
        raise RuntimeError(
            "NO HAY GPU y PERMITIR_CPU=False: entrenar U4 en CPU tardaría horas. En Colab: Entorno de ejecución → "
            "Cambiar tipo de entorno → T4 GPU. (Para una prueba de humo pequeña ponga permitir_cpu=True.)"
        )
    logger.warning(
        "Sin GPU: %s correrá en CPU (lento).", "el entrenamiento" if para_entrenar else "el etiquetado"
    )
    return "cpu"


def _tokenizar(tokenizador: Any, textos: Sequence[str], max_length: int) -> tuple[Any, Any]:
    t = tokenizador(
        [str(x) for x in textos],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    return t["input_ids"], t["attention_mask"]


def _softmax_lotes(modelo: Any, ids: Any, mask: Any, dispositivo: str, fp16: bool, lote: int = 128) -> np.ndarray:
    import torch  # pylint: disable=import-outside-toplevel
    import torch.nn.functional as F  # pylint: disable=import-outside-toplevel,invalid-name

    modelo.eval()
    salida = []
    with torch.no_grad():
        for i in range(0, len(ids), lote):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(fp16 and dispositivo == "cuda")):
                lg = modelo(ids[i : i + lote].to(dispositivo), mask[i : i + lote].to(dispositivo))
            salida.append(F.softmax(lg.float(), dim=-1).cpu())
    return torch.cat(salida).numpy() if salida else np.zeros((0, 0), np.float32)


@dataclass
class ModeloU4:  # pylint: disable=too-many-instance-attributes
    """Modelo U4 cargado en memoria (o recién entrenado) y sus metadatos."""

    carpeta: Path
    run_id: str
    clave: str
    clases: list[str]
    config: dict[str, Any]
    metricas: dict[str, Any]
    modelo: Any
    tokenizador: Any
    dispositivo: str
    textos_vistos: frozenset[str] = frozenset()
    holdout: pd.DataFrame = field(default_factory=pd.DataFrame)
    reutilizado: bool = False

    @property
    def clases_especiales(self) -> list[str]:
        """Clases de U4 que son «sin aspecto» (opinión general, ninguno de la lista...)."""
        sin = {norm_txt_u4(c) for c in self.config.get("clases_sin_aspecto", CLASES_SIN_ASPECTO_U4)}
        return [c for c in self.clases if norm_txt_u4(c) in sin]

    def predecir_proba(self, textos: Sequence[str], lote: int | None = None, fp16: bool | None = None) -> np.ndarray:
        """Probabilidades (n × K) de TODAS las clases, con el texto ORIGINAL (MAX_LENGTH del entrenamiento)."""
        lote = int(lote or self.config.get("lote_etiquetado", 128))
        fp16 = bool(self.config.get("fp16", True)) if fp16 is None else fp16
        ml = int(self.config["max_length"])
        salida = []
        for i in range(0, len(textos), lote):
            ids, mask = _tokenizar(self.tokenizador, textos[i : i + lote], ml)
            salida.append(_softmax_lotes(self.modelo, ids, mask, self.dispositivo, fp16, lote))
        return np.vstack(salida) if salida else np.zeros((0, len(self.clases)), np.float32)


def etiquetar_clausulas(modelo: ModeloU4, clausulas: Sequence[str], lote: int | None = None) -> np.ndarray:
    """Probabilidades (n × K) de cada texto, deduplicando textos idénticos y reexpandiendo."""
    textos = [str(c) for c in clausulas]
    if not textos:
        return np.zeros((0, len(modelo.clases)), np.float32)
    codigos, unicos = pd.factorize(pd.Series(textos), sort=False)
    P = modelo.predecir_proba([str(u) for u in unicos], lote=lote)
    return P[np.asarray(codigos)]


def _guardar_pesos_entrenables(modelo: Any, ruta: Path) -> list[str]:
    import torch  # pylint: disable=import-outside-toplevel

    nombres = _nombres_entrenables(modelo)
    estado = dict(modelo.named_parameters())
    torch.save({n: estado[n].detach().cpu().clone() for n in nombres}, ruta)
    return nombres


def cargar_u4(
    ruta: Path | str,
    *,
    fabrica_encoder: Callable[[str], tuple[Any, Any]] | None = None,
    dispositivo: str | None = None,
    permitir_cpu: bool = True,
) -> ModeloU4:
    """Reconstruye un modelo U4 guardado: encoder base + SOLO los parámetros entrenables, comprobando claves.

    Sirve para etiquetar sin reentrenar. ``ruta`` = carpeta de la corrida (``.../<run_id>``) o de la pregunta
    (``.../u4_aspectos/<clave>``, que se resuelve por ``ultimo.json``).
    """
    import torch  # pylint: disable=import-outside-toplevel

    carpeta = _resolver_carpeta_modelo(Path(ruta))
    cfg = json.loads((carpeta / "config_u4.json").read_text(encoding="utf-8"))
    clases = json.loads((carpeta / "clases.json").read_text(encoding="utf-8"))
    base = (carpeta / "modelo_base.txt").read_text(encoding="utf-8").strip()
    metricas = json.loads((carpeta / "metricas.json").read_text(encoding="utf-8"))
    dispositivo = dispositivo or _resolver_dispositivo(permitir_cpu, para_entrenar=False)
    tok, backbone = (fabrica_encoder or fabrica_encoder_hf)(base)
    modelo = _construir_modelo_u4(backbone, len(clases), int(cfg["k"]), float(cfg["dropout_cabeza"]))
    estado = torch.load(carpeta / "pesos_u4.pt", map_location="cpu", weights_only=True)
    esperados = set(_nombres_entrenables(modelo))
    if set(estado) != esperados:
        raise RuntimeError(
            f"Los pesos de {carpeta.name} no corresponden a este encoder/k: "
            f"faltan {sorted(esperados - set(estado))[:6]}, sobran {sorted(set(estado) - esperados)[:6]}."
        )
    propios = dict(modelo.named_parameters())
    for n, v in estado.items():
        if tuple(propios[n].shape) != tuple(v.shape):
            raise RuntimeError(f"Forma distinta en '{n}': guardado {tuple(v.shape)} vs modelo {tuple(propios[n].shape)}.")
    with torch.no_grad():
        for n, v in estado.items():
            propios[n].copy_(v)
    modelo.to(dispositivo).eval()
    vistos: frozenset[str] = frozenset()
    ruta_datos = carpeta / "datos_usados.csv"
    if ruta_datos.exists():
        du = pd.read_csv(ruta_datos, usecols=["texto_norm", "rol"], dtype=str, keep_default_na=False)
        vistos = frozenset(du.loc[du["rol"] != "holdout", "texto_norm"])
    hold = pd.DataFrame()
    if (carpeta / "holdout_predicciones.csv").exists():
        hold = pd.read_csv(carpeta / "holdout_predicciones.csv")
    return ModeloU4(
        carpeta=carpeta,
        run_id=carpeta.name,
        clave=str(cfg.get("clave", carpeta.parent.name)),
        clases=list(clases),
        config=cfg,
        metricas=metricas,
        modelo=modelo,
        tokenizador=tok,
        dispositivo=dispositivo,
        textos_vistos=vistos,
        holdout=hold,
    )


def _modelo_completo(carpeta: Path) -> bool:
    """¿Es una corrida terminada y verificada? (sirve para ``reutilizar_modelo``)."""
    requeridos = ("config_u4.json", "clases.json", "metricas.json", "pesos_u4.pt", "modelo_base.txt", "datos_usados.csv")
    if not all((carpeta / r).exists() for r in requeridos):
        return False
    try:
        return bool(json.loads((carpeta / "config_u4.json").read_text(encoding="utf-8")).get("verificacion_ida_y_vuelta"))
    except (OSError, ValueError):
        return False


# ── A. ENTRENAMIENTO ───────────────────────────────────────────────────────


def _entrenar_clasificador(  # pylint: disable=too-many-locals,too-many-arguments,too-many-statements
    base: Any,
    n_clases: int,
    hip: dict[str, Any],
    ids_tr: Any,
    mask_tr: Any,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    ids_va: Any,
    mask_va: Any,
    y_va: np.ndarray,
    semilla: int,
    dispositivo: str,
) -> tuple[Any, dict[str, Any]]:
    """Entrena U4 (réplica de ``entrenar_predecir`` del cuaderno de regresión logística) con parada temprana."""
    import torch  # pylint: disable=import-outside-toplevel
    import torch.nn.functional as F  # pylint: disable=import-outside-toplevel,invalid-name
    from sklearn.metrics import f1_score  # pylint: disable=import-outside-toplevel

    cuda = dispositivo == "cuda"
    fp16 = bool(hip["fp16"]) and cuda
    # El backbone se carga ANTES de fijar la semilla (como BASE en el cuaderno): así la cabeza y los lotes
    # consumen el mismo azar que en el experimento ya realizado.
    fijar_semillas_u4(semilla)
    modelo = _construir_modelo_u4(copy.deepcopy(base), n_clases, int(hip["k"]), float(hip["dropout_cabeza"])).to(dispositivo)
    n = len(y_tr)
    cnt = np.bincount(y_tr, minlength=n_clases).astype(np.float64)
    cw = np.where(cnt > 0, 1.0 / np.sqrt(np.maximum(cnt, 1)), 0.0)  # pesos de clase ∝ 1/√frecuencia
    cw = cw / (cw[y_tr] * 1.0).mean()  # peso medio por fila = 1
    cw_t = torch.tensor(cw, dtype=torch.float32, device=dispositivo)
    enc_params = [p for p in modelo.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": float(hip["lr_enc"]), "weight_decay": float(hip["weight_decay"])},
            {"params": list(modelo.cabeza.parameters()), "lr": float(hip["lr_cabeza"]), "weight_decay": float(hip["weight_decay"])},
        ]
    )
    batch, epocas = int(hip["batch"]), int(hip["epocas_max"])
    pasos_ep = math.ceil(n / batch)
    total, calentamiento = pasos_ep * epocas, int(float(hip["warmup"]) * pasos_ep * epocas)

    def _lr(paso: int) -> float:  # mismo calendario que get_linear_schedule_with_warmup
        if paso < calentamiento:
            return paso / max(1, calentamiento)
        return max(0.0, (total - paso) / max(1, total - calentamiento))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    y_t, w_t = torch.tensor(y_tr), torch.tensor(w_tr, dtype=torch.float32)
    gen = torch.Generator().manual_seed(semilla)
    entrenables = [(n_, p) for n_, p in modelo.named_parameters() if p.requires_grad]
    mejor_f, mejor_estado, sin_mejora, mejor_ep, ep_ejec = -1.0, None, 0, 0, 0
    for ep in range(epocas):
        ep_ejec = ep + 1
        modelo.train()
        orden = torch.randperm(n, generator=gen)
        for i in range(0, n, batch):
            b = orden[i : i + batch]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
                lg = modelo(ids_tr[b].to(dispositivo), mask_tr[b].to(dispositivo))
            yb, wb = y_t[b].to(dispositivo), w_t[b].to(dispositivo)
            perd = F.cross_entropy(lg.float(), yb, weight=cw_t, label_smoothing=float(hip["label_smoothing"]), reduction="none")
            perd = (perd * wb).sum() / (cw_t[yb] * wb).sum().clamp(min=1e-9)
            opt.zero_grad(set_to_none=True)
            scaler.scale(perd).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for _, p in entrenables], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        if len(y_va):
            pred = _softmax_lotes(modelo, ids_va, mask_va, dispositivo, fp16).argmax(1)
            f = f1_score(y_va, pred, labels=np.unique(y_va), average="macro", zero_division=0)
            if f > mejor_f + 1e-4:
                mejor_f, sin_mejora, mejor_ep = f, 0, ep + 1
                mejor_estado = {n_: p.detach().cpu().clone() for n_, p in entrenables}
            else:
                sin_mejora += 1
                if sin_mejora >= int(hip["paciencia"]):
                    break
        logger.info("  época %d/%d | mejor F1 macro de validación interna %.4f (época %d)", ep_ejec, epocas, mejor_f, mejor_ep)
    if mejor_estado is not None:
        with torch.no_grad():
            for n_, p in entrenables:
                p.copy_(mejor_estado[n_].to(p.device))
    modelo.eval()
    return modelo, {"mejor_epoca": mejor_ep, "epocas_ejecutadas": ep_ejec, "f1_macro_validacion_interna": float(mejor_f)}


#: Plan v2, punto 8: Excel de métricas de U4 sobre el HOLD-OUT (antes ``entrenamiento_u4.xlsx``; ``etiquetar_corpus_u4`` le añade
#: las hojas de diagnóstico del corpus) y sobre el conjunto de ENTRENAMIENTO (filas ``rol = entrenamiento``).
ARCHIVO_ENTRENAMIENTO_TEST = "entrenamiento_u4_test.xlsx"
ARCHIVO_ENTRENAMIENTO_TRAIN = "entrenamiento_u4_train.xlsx"


def _tabla_predicciones_u4(d: pd.DataFrame, idx: np.ndarray, y: np.ndarray, P: np.ndarray, clases: list[str]) -> pd.DataFrame:
    """Predicciones por fragmento de las filas ``idx`` de ``d``: ``id``, ``texto``, ``y_true``, ``y_pred``, ``p_max``, ``correcto`` y ``p_<clase>``."""
    pred = pd.DataFrame(
        {
            "id": d["id"].to_numpy()[idx],
            "texto": d["texto"].to_numpy()[idx],
            "y_true": [clases[j] for j in y[idx]],
            "y_pred": [clases[j] for j in P.argmax(1)] if len(P) else [],
            "p_max": P.max(1) if len(P) else [],
        }
    )
    pred["correcto"] = pred["y_true"] == pred["y_pred"]
    for j, c in enumerate(clases):
        pred[f"p_{c}"] = np.round(P[:, j], 6) if len(P) else []
    return pred


def _metricas_holdout(y_true: np.ndarray, P: np.ndarray, clases: list[str], n_min: int) -> dict[str, Any]:
    """Acierto, F1 macro, precisión/recall/F1 por clase con su n, y matriz de confusión (conteos y por fila).

    Es genérica: la usan la evaluación del hold-out y la del conjunto de entrenamiento (punto 8).
    """
    from sklearn.metrics import (  # pylint: disable=import-outside-toplevel
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    k = len(clases)
    if len(y_true) == 0:
        vacio = pd.DataFrame(0, index=clases, columns=clases)
        return {
            "accuracy": float("nan"),
            "f1_macro": float("nan"),
            "f1_macro_todas_las_clases": float("nan"),
            "n_holdout_evaluable": 0,
            "por_clase": pd.DataFrame(columns=["clase", "n", "concluyente", "precision", "recall", "f1"]),
            "confusion_conteos": vacio,
            "confusion_normalizada": vacio.astype(float),
        }
    y_pred = P.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(k), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(k))
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    por_clase = pd.DataFrame(
        {
            "clase": clases,
            "n": s.astype(int),
            "concluyente": np.where(s >= n_min, "sí", "NO CONCLUYENTE"),
            "precision": p,
            "recall": r,
            "f1": f,
        }
    )
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "f1_macro": float(f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro", zero_division=0)),
        "f1_macro_todas_las_clases": float(f1_score(y_true, y_pred, labels=range(k), average="macro", zero_division=0)),
        "n_holdout_evaluable": int(len(y_true)),
        "por_clase": por_clase,
        "confusion_conteos": pd.DataFrame(cm, index=clases, columns=clases),
        "confusion_normalizada": pd.DataFrame(norm, index=clases, columns=clases),
    }


def _reemplazar_hojas_excel(ruta: Path, hojas: dict[str, pd.DataFrame], estricto: bool = False) -> None:
    """Escribe/reemplaza hojas de un .xlsx (lo crea si no existe).

    ``estricto=False`` (diagnósticos): un fallo de escritura se avisa y no rompe el flujo.
    ``estricto=True`` (Excel de entrenamiento): el fallo se propaga.
    """
    try:
        existe = ruta.exists()
        extra = {"if_sheet_exists": "replace"} if existe else {}
        with pd.ExcelWriter(ruta, engine="openpyxl", mode="a" if existe else "w", **extra) as libro:
            for nombre, tabla in hojas.items():
                tabla.to_excel(libro, sheet_name=nombre[:31], index=aplanar_indice(tabla))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        if estricto:
            raise
        logger.warning("No se pudo escribir %s (%s: %s)", ruta, type(exc).__name__, exc)


def aplanar_indice(tabla: pd.DataFrame) -> bool:
    """True si el índice del DataFrame lleva información (nombres de clase en una matriz de confusión)."""
    return not isinstance(tabla.index, pd.RangeIndex)


def _escribir_excel_entrenamiento(  # pylint: disable=too-many-arguments
    carpeta: Path,
    datos: DatosU4,
    m: dict[str, Any],
    pred: pd.DataFrame,
    config: dict[str, Any],
    info_entrenamiento: dict[str, Any],
    *,
    parte: str = "test",
) -> None:
    """Excel de métricas de U4: ``parte="test"`` (hold-out, ``entrenamiento_u4_test.xlsx``) o ``"train"`` (entrenamiento).

    Mismas hojas y columnas en ambos; el de entrenamiento usa los campos ``*_train``, lleva otra nota y no lleva el marcador
    ``diagnostico_comentario_general`` (los diagnósticos del corpus se añaden solo al archivo ``_test``).
    """
    if parte not in ("test", "train"):
        raise ValueError(f"parte debe ser 'test' o 'train', no {parte!r}")
    es_test = parte == "test"
    suf = "holdout" if es_test else "train"
    errores = pred[~pred["correcto"]]
    if errores.empty:
        frecuentes = pd.DataFrame(columns=["y_true", "y_pred", "n", "ejemplos"])
    else:
        frecuentes = (
            errores.groupby(["y_true", "y_pred"])
            .agg(n=("texto", "size"), ejemplos=("texto", lambda s: " || ".join(str(x)[:100] for x in list(s)[:3])))
            .reset_index()
            .sort_values("n", ascending=False)
        )
    resumen = pd.DataFrame(
        [
            ("clave", config.get("clave")),
            ("run_id", config.get("run_id")),
            ("modelo_base", config.get("modelo_base")),
            ("semilla_entrenamiento", config.get("semilla")),
            (f"acierto_{suf}", m["accuracy"]),
            (f"f1_macro_{suf}", m["f1_macro"]),
            (f"f1_macro_{suf}_todas_las_clases", m["f1_macro_todas_las_clases"]),
            (f"n_{suf}_evaluable", m["n_holdout_evaluable"]),
            ("mejor_epoca", info_entrenamiento.get("mejor_epoca")),
            ("epocas_ejecutadas", info_entrenamiento.get("epocas_ejecutadas")),
            ("f1_macro_validacion_interna", info_entrenamiento.get("f1_macro_validacion_interna")),
            ("archivo_auditado", datos.informe["archivo"]),
            ("md5_auditado", datos.md5),
            ("filas_usadas", datos.informe["filas_usadas"]),
            ("filas_descartadas_vacias", datos.informe["filas_descartadas_vacias"]),
            ("fragmentos_contradictorios", datos.informe["fragmentos_con_etiquetas_contradictorias"]),
            ("variantes_normalizadas", json.dumps(datos.informe["variantes_normalizadas"], ensure_ascii=False)),
            (
                "nota",
                "Métricas de UNA sola evaluación en el hold-out (15 % de los grupos, nunca usado para entrenar ni elegir)."
                if es_test
                else "Métricas sobre el conjunto de ENTRENAMIENTO (filas rol = entrenamiento, evaluables): NO miden generalización "
                "(el modelo ajustó sus pesos con estas filas); sirven para compararlas con las del hold-out (entrenamiento_u4_test.xlsx).",
            ),
        ],
        columns=["campo", "valor"],
    )
    usados = datos.tabla[
        ["id", "revisor", "respuesta_completa", "texto", "etiqueta", "peso", "tipo_fila", "motivo", "grupo", "rol"]
    ]
    cfg_tabla = pd.DataFrame(
        [(k, json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list)) else v) for k, v in config.items()],
        columns=["parametro", "valor"],
    )
    pendiente = pd.DataFrame(
        {"estado": ["pendiente"], "detalle": ["Cada etiquetado del corpus (etiquetar_corpus_u4) añade las hojas diag_cg_<fecha-hora> y diag_res_<fecha-hora>; hace falta la distribución de p_aspecto_u4 en las cláusulas de respuestas cortas."]}
    )
    hojas = {
        "resumen": resumen,
        "recall_por_clase": m["por_clase"],
        "confusion_conteos": m["confusion_conteos"],
        "confusion_normalizada": m["confusion_normalizada"].round(4),
        "predicciones": pred,
        "datos_usados": usados,
        "errores_frecuentes": frecuentes,
        "config": cfg_tabla,
    }
    if es_test:
        hojas["diagnostico_comentario_general"] = pendiente
    _reemplazar_hojas_excel(
        carpeta / (ARCHIVO_ENTRENAMIENTO_TEST if es_test else ARCHIVO_ENTRENAMIENTO_TRAIN), hojas, estricto=True
    )


def entrenar_u4(  # pylint: disable=too-many-locals,too-many-arguments,too-many-statements
    clave_encuesta: str,
    cfg: ParametrosU4 | dict[str, Any],
    raiz: Path | str,
    *,
    reutilizar: bool | None = None,
    permitir_cpu: bool | None = None,
    fabrica_encoder: Callable[[str], tuple[Any, Any]] | None = None,
) -> ModeloU4:
    """Entrena, evalúa UNA vez en el hold-out, guarda y verifica el modelo U4 de UNA pregunta.

    Parametrizada por ``PERFILES[clave]['aspectos_u4']`` (``cfg``); sin lógica específica de obs19.
    Orden: carga del archivo AUDITADO (falla con un error claro si no existe o le faltan columnas; nunca cae a
    datos sin auditar) → crea y comprueba las carpetas de salida (ANTES de gastar GPU) → huella (si ya existe un
    modelo con la misma huella de datos + configuración y ``reutilizar``, lo carga y no reentrena) → GPU →
    entrenamiento con parada temprana → evaluación en el hold-out → guardado → verificación ida y vuelta
    (32 textos, ``atol=1e-4``) → ``ultimo.json``.

    Returns:
        :class:`ModeloU4` (el entrenado, o el reutilizado con ``reutilizado=True``).
    """
    import torch  # pylint: disable=import-outside-toplevel

    params = ParametrosU4.desde(cfg, clave_encuesta)
    raiz = Path(raiz)
    hip = params.hip()
    reutilizar = params.reutilizar_modelo if reutilizar is None else reutilizar
    permitir_cpu = params.permitir_cpu if permitir_cpu is None else permitir_cpu
    fabrica = fabrica_encoder or fabrica_encoder_hf

    datos = cargar_datos_u4(raiz, params, clave_encuesta)  # 1) datos AUDITADOS (error claro si faltan)
    carpeta_clave = carpeta_modelos_u4(raiz, clave_encuesta)  # 2) carpetas, ANTES de entrenar
    asegurar_carpeta_escribible(raiz / "modelos")
    asegurar_carpeta_escribible(raiz / "modelos" / "u4_aspectos")
    asegurar_carpeta_escribible(carpeta_clave)
    huella = hashlib.md5(
        json.dumps(
            {
                "formato": _FORMATO_MODELO_U4,
                "md5_datos": datos.md5,
                "hoja": params.hoja_etiquetas,
                "cols": [params.col_respuesta, params.col_fragmento, params.col_aspecto],
                "clases": datos.clases,
                "sin_aspecto": list(params.clases_sin_aspecto),
                "hip": hip,
                "encoder": params.encoder,
                "semilla": params.semilla,
                "n_filas_prueba": params.n_filas_prueba,
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:16]

    if reutilizar:  # 3) ¿ya existe un modelo con la misma huella?
        for c in sorted((p for p in carpeta_clave.iterdir() if p.is_dir()), reverse=True):
            if _modelo_completo(c):
                try:
                    h = json.loads((c / "config_u4.json").read_text(encoding="utf-8")).get("huella_entrenamiento")
                except (OSError, ValueError):
                    continue
                if h == huella:
                    logger.info("[%s] REUTILIZAR_MODELO: ya existe %s con la misma huella (%s); no se reentrena.", clave_encuesta, c.name, huella)
                    modelo = cargar_u4(c, fabrica_encoder=fabrica, permitir_cpu=True)
                    modelo.reutilizado = True
                    _escribir_ultimo(carpeta_clave, c.name, huella)
                    return modelo

    dispositivo = _resolver_dispositivo(permitir_cpu, para_entrenar=True)  # 4) GPU
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{huella[:8]}"
    carpeta = asegurar_carpeta_escribible(carpeta_clave / run_id)

    d = datos.tabla
    tr, va, ho = particionar_datos_u4(datos, hip)
    d["rol"] = "entrenamiento"
    d.loc[d.index[va], "rol"] = "validacion_interna"
    d.loc[d.index[ho], "rol"] = "holdout"
    logger.info(
        "[%s] %d clases | entrenamiento %d · validación interna %d · hold-out %d (evaluables %d) | dispositivo %s",
        clave_encuesta,
        len(datos.clases),
        len(tr),
        len(va),
        len(ho),
        int(d["usa_evaluacion"].to_numpy()[ho].sum()),
        dispositivo,
    )
    tokenizador, base = fabrica(params.encoder)
    ids_all, mask_all = _tokenizar(tokenizador, d["texto"].tolist(), int(hip["max_length"]))
    y, w = d["y"].to_numpy(), d["peso"].to_numpy().astype(np.float32)
    modelo_t, info = _entrenar_clasificador(
        base, len(datos.clases), hip, ids_all[tr], mask_all[tr], y[tr], w[tr], ids_all[va], mask_all[va], y[va], params.semilla, dispositivo
    )

    # 5) evaluación UNA vez en el hold-out (solo filas evaluables: sin fragmentos contradictorios)
    ho_ev = ho[d["usa_evaluacion"].to_numpy()[ho]]
    fp16 = bool(hip["fp16"])
    P = _softmax_lotes(modelo_t, ids_all[ho_ev], mask_all[ho_ev], dispositivo, fp16) if len(ho_ev) else np.zeros((0, len(datos.clases)))
    m = _metricas_holdout(y[ho_ev], P, datos.clases, int(hip["n_min_concluyente"]))
    pred = _tabla_predicciones_u4(d, ho_ev, y, P, datos.clases)
    logger.info(
        "[%s] HOLD-OUT (una sola evaluación): acierto %.3f · F1 macro %.3f · n=%d",
        clave_encuesta,
        m["accuracy"],
        m["f1_macro"],
        m["n_holdout_evaluable"],
    )
    # 5b) plan v2, punto 8: métricas sobre el conjunto de ENTRENAMIENTO (rol = entrenamiento, evaluables) para compararlas con las
    # del hold-out; no cambia el modelo, sus pesos ni nada de lo que se guarda como modelo. No mide generalización.
    tr_ev = tr[d["usa_evaluacion"].to_numpy()[tr]]
    P_tr = _softmax_lotes(modelo_t, ids_all[tr_ev], mask_all[tr_ev], dispositivo, fp16) if len(tr_ev) else np.zeros((0, len(datos.clases)))
    m_tr = _metricas_holdout(y[tr_ev], P_tr, datos.clases, int(hip["n_min_concluyente"]))
    pred_tr = _tabla_predicciones_u4(d, tr_ev, y, P_tr, datos.clases)
    logger.info(
        "[%s] ENTRENAMIENTO (no mide generalización): acierto %.3f · F1 macro %.3f · n=%d",
        clave_encuesta,
        m_tr["accuracy"],
        m_tr["f1_macro"],
        m_tr["n_holdout_evaluable"],
    )

    # 6) guardado (solo parámetros entrenables) + metadatos
    nombres = _guardar_pesos_entrenables(modelo_t, carpeta / "pesos_u4.pt")
    (carpeta / "modelo_base.txt").write_text(params.encoder, encoding="utf-8")
    (carpeta / "clases.json").write_text(json.dumps(datos.clases, ensure_ascii=False, indent=2), encoding="utf-8")
    config = {
        **hip,
        "clave": clave_encuesta,
        "run_id": run_id,
        "modelo_base": params.encoder,
        "semilla": params.semilla,
        "umbral_comentario_general": params.umbral_comentario_general,
        "max_palabras_respuesta_corta": params.max_palabras_respuesta_corta,
        "clases_sin_aspecto": list(params.clases_sin_aspecto),
        "version_modulo": VERSION,
        "huella_entrenamiento": huella,
        "n_parametros_entrenables_tensores": len(nombres),
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "dispositivo": dispositivo,
        "verificacion_ida_y_vuelta": False,
        "torch": torch.__version__,
    }
    (carpeta / "config_u4.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    metricas_json = {
        "acierto_holdout": m["accuracy"],
        "f1_macro_holdout": m["f1_macro"],
        "f1_macro_holdout_todas_las_clases": m["f1_macro_todas_las_clases"],
        "n_holdout_evaluable": m["n_holdout_evaluable"],
        "n_entrenamiento": int(len(tr)),
        "n_validacion_interna": int(len(va)),
        "n_holdout": int(len(ho)),
        **info,
        "por_clase": m["por_clase"].to_dict(orient="records"),
        "nota": "Métricas de una sola evaluación en el hold-out (grupos separados del entrenamiento).",
    }
    (carpeta / "metricas.json").write_text(json.dumps(_a_json(metricas_json), ensure_ascii=False, indent=2), encoding="utf-8")
    (carpeta / "huella_datos.json").write_text(json.dumps(_a_json(datos.informe), ensure_ascii=False, indent=2), encoding="utf-8")
    d_usados = pd.DataFrame({"id": d["id"], "texto_norm": d["texto"].map(norm_txt_u4), "rol": d["rol"], "etiqueta": d["etiqueta"]})
    d_usados.to_csv(carpeta / "datos_usados.csv", index=False, encoding="utf-8")
    pred[["id", "texto", "y_true", "y_pred", "p_max", "correcto"]].to_csv(carpeta / "holdout_predicciones.csv", index=False, encoding="utf-8")
    _escribir_excel_entrenamiento(carpeta, datos, m, pred, config, info)  # entrenamiento_u4_test.xlsx (hold-out)
    _escribir_excel_entrenamiento(carpeta, datos, m_tr, pred_tr, config, info, parte="train")  # entrenamiento_u4_train.xlsx

    modelo = ModeloU4(
        carpeta=carpeta,
        run_id=run_id,
        clave=clave_encuesta,
        clases=list(datos.clases),
        config=config,
        metricas=_a_json(metricas_json),
        modelo=modelo_t,
        tokenizador=tokenizador,
        dispositivo=dispositivo,
        textos_vistos=frozenset(d_usados.loc[d_usados["rol"] != "holdout", "texto_norm"]),
        holdout=pred[["id", "texto", "y_true", "y_pred", "p_max", "correcto"]].copy(),
    )

    # 7) verificación de ida y vuelta: recarga y compara 32 textos (fp32, atol=1e-4)
    muestra = d["texto"].tolist()[:32]
    p_mem = modelo.predecir_proba(muestra, fp16=False)
    recargado = cargar_u4(carpeta, fabrica_encoder=fabrica, dispositivo=dispositivo)
    p_cargado = recargado.predecir_proba(muestra, fp16=False)
    dif = float(np.abs(p_mem - p_cargado).max()) if len(muestra) else 0.0
    del recargado
    if dispositivo == "cuda":
        torch.cuda.empty_cache()
    if not np.allclose(p_mem, p_cargado, atol=1e-4):
        raise RuntimeError(
            f"Verificación de ida y vuelta FALLÓ: el modelo recargado de {carpeta} difiere del de memoria "
            f"(diferencia máxima {dif:.2e} > 1e-4). No se actualiza ultimo.json."
        )
    config["verificacion_ida_y_vuelta"] = True
    config["verificacion_diferencia_max"] = dif
    modelo.config = config
    (carpeta / "config_u4.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    _escribir_ultimo(carpeta_clave, run_id, huella)
    logger.info("[%s] modelo guardado y verificado (dif. máx. %.1e): %s", clave_encuesta, dif, carpeta)
    return modelo


def _escribir_ultimo(carpeta_clave: Path, run_id: str, huella: str) -> None:
    tmp = carpeta_clave / "ultimo.json.tmp"
    tmp.write_text(
        json.dumps({"run_id": run_id, "huella_entrenamiento": huella, "fecha": datetime.now().isoformat(timespec="seconds")}, indent=2),
        encoding="utf-8",
    )
    tmp.replace(carpeta_clave / "ultimo.json")


def modelo_u4_de_clave(raiz: Path | str, clave: str, **kwargs: Any) -> ModeloU4:
    """Carga el modelo más reciente (``ultimo.json``) de una pregunta, sin reentrenar."""
    return cargar_u4(carpeta_modelos_u4(raiz, clave), **kwargs)


# ── B. ETIQUETADO DEL CORPUS Y REGLAS ──────────────────────────────────────


@dataclass
class EtapasU4:
    """Resultado de las etapas B/B2: tablas ya etiquetadas, residual modelado y diagnósticos."""

    documentos: pd.DataFrame
    clausulas: pd.DataFrame
    modelo: ModeloU4 | None = None
    tabla_residual: pd.DataFrame = field(default_factory=pd.DataFrame)
    metricas_residual: dict[str, Any] = field(default_factory=dict)
    diagnostico_cg: dict[str, pd.DataFrame] = field(default_factory=dict)
    diagnostico_residual: pd.DataFrame = field(default_factory=pd.DataFrame)
    residual_listo: bool = False
    modelador_residual: ModeladorAspectos | None = None
    #: UMAP/HDBSCAN ajustados con prediction_data=True (None si no se pudo ajustar o no se ejecutó el residual).
    residual_persistible: ModeloResidual | None = None


def nombres_columnas_p(clases: Sequence[str]) -> list[str]:
    """``p_u4_<slug_aspecto>`` por clase (con sufijo numérico si dos clases colapsan al mismo slug)."""
    vistos: dict[str, int] = {}
    salida = []
    for c in clases:
        s = f"p_u4_{slug(c)}"
        vistos[s] = vistos.get(s, 0) + 1
        salida.append(s if vistos[s] == 1 else f"{s}_{vistos[s]}")
    return salida


def _validar_alineacion_documentos(documentos: pd.DataFrame, encuesta: Encuesta) -> None:
    """``documentos`` debe ser una fila por fila del corpus, en el mismo orden (doc_id = posición) y con las mismas llaves."""
    n = len(encuesta.texto_original)
    if len(documentos) != n:
        raise ValueError(f"documentos tiene {len(documentos)} filas y la encuesta {n}: no están alineados.")
    if not (documentos["doc_id"].to_numpy() == np.arange(n)).all():
        raise ValueError("documentos['doc_id'] no es 0..n-1 en orden: no se puede alinear con texto_original.")
    marco = getattr(encuesta, "marco", None)
    for col in getattr(encuesta, "columnas_llave", []) or []:
        if col in documentos.columns and marco is not None and col in marco.columns:
            a = documentos[col].astype(str).to_numpy()
            b = marco[col].reset_index(drop=True).astype(str).to_numpy()
            if len(a) != len(b) or not (a == b).all():
                raise ValueError(f"La llave '{col}' de documentos no coincide con la de la encuesta: desalineación.")


def diagnostico_comentario_general(p_cortas: np.ndarray, umbral: float) -> dict[str, pd.DataFrame]:
    """Distribución de ``p_aspecto_u4`` en cláusulas de respuestas cortas y % que pasaría a «Comentario General»."""
    p = np.asarray(p_cortas, dtype=float)
    p = p[np.isfinite(p)]
    umbrales = sorted({*_UMBRALES_DIAGNOSTICO, round(float(umbral), 4)})
    por_umbral = pd.DataFrame(
        {
            "umbral": umbrales,
            "n_clausulas_respuesta_corta": len(p),
            "n_pasarian_a_comentario_general": [int((p < u).sum()) for u in umbrales],
            "pct_comentario_general": [round(100 * float((p < u).mean()), 2) if len(p) else float("nan") for u in umbrales],
            "es_el_umbral_actual": [bool(abs(u - umbral) < 1e-9) for u in umbrales],
        }
    )
    cuantiles = (
        pd.Series(p).describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9]).rename("p_aspecto_u4").round(4).reset_index()
        if len(p)
        else pd.DataFrame({"index": [], "p_aspecto_u4": []})
    ).rename(columns={"index": "estadistico"})
    hist = np.histogram(p, bins=np.linspace(0, 1, 11))[0] if len(p) else np.zeros(10, int)
    distribucion = pd.DataFrame({"p_desde": np.round(np.linspace(0, 0.9, 10), 1), "p_hasta": np.round(np.linspace(0.1, 1.0, 10), 1), "n": hist})
    return {"umbrales": por_umbral, "cuantiles": cuantiles, "distribucion": distribucion}


def diagnostico_residual(
    clausulas: pd.DataFrame, holdout: pd.DataFrame | None, min_palabras: int, umbral_actual: float
) -> pd.DataFrame:
    """Para cada umbral: % del corpus (cláusulas ASPECTO) que sería residual y acierto de U4 bajo/sobre el umbral en el hold-out."""
    aspecto = clausulas[clausulas["motivo"] == MOTIVO_ASPECTO]
    n_pal = aspecto["clausula"].astype(str).str.split().str.len()
    elegible = (n_pal >= min_palabras) & ~aspecto["respuesta_corta"].astype(bool)
    p = aspecto["p_aspecto_u4"].to_numpy(dtype=float)
    tiene_holdout = holdout is not None and len(holdout) > 0
    filas = []
    for u in sorted({*_UMBRALES_DIAGNOSTICO, round(float(umbral_actual), 4)}):
        res = elegible.to_numpy() & (p < u)
        fila: dict[str, Any] = {
            "umbral": u,
            "n_residual": int(res.sum()),
            "pct_del_corpus_aspecto": round(100 * float(res.mean()), 2) if len(res) else float("nan"),
            "pct_de_elegibles": round(100 * float(res.sum() / max(int(elegible.sum()), 1)), 2),
            "es_el_umbral_actual": bool(abs(u - umbral_actual) < 1e-9),
        }
        if tiene_holdout:
            ph = holdout["p_max"].to_numpy(dtype=float)
            ok = holdout["correcto"].to_numpy(dtype=bool)
            ab, ar = ph < u, ph >= u
            fila.update(
                holdout_n_bajo=int(ab.sum()),
                holdout_acierto_bajo=round(float(ok[ab].mean()), 4) if ab.any() else float("nan"),
                holdout_n_sobre=int(ar.sum()),
                holdout_acierto_sobre=round(float(ok[ar].mean()), 4) if ar.any() else float("nan"),
            )
        filas.append(fila)
    return pd.DataFrame(filas)


def etiquetar_corpus_u4(  # pylint: disable=too-many-locals,too-many-statements
    modelo: ModeloU4,
    encuesta: Encuesta,
    documentos: pd.DataFrame,
    clausulas: pd.DataFrame,
    cfg: Config,
    *,
    escribir_diagnosticos: bool = True,
) -> EtapasU4:
    """Etapa B: etiqueta TODAS las cláusulas ASPECTO (y, si ``cfg.sin_contenido_en_cortas`` es False, las VALORACION_GLOBAL de
    respuestas cortas) y aplica las reglas.

    Precedencia por cláusula:
      a. ``NO_OPINION`` y ``RUIDO`` conservan su tratamiento.
      b. Respuesta corta (``<= cfg.max_palabras_respuesta_corta`` palabras en ``encuesta.texto_original``, sin limpiar):
         a toda cláusula ASPECTO (y a las VALORACION_GLOBAL solo si ``cfg.sin_contenido_en_cortas`` es False) se le pasa U4; si ``p_aspecto_u4`` <
         ``cfg.umbral_comentario_general`` el aspecto final es «Comentario General»; si no, el de U4.
      c. ``respuesta_corta`` (bool) se marca en TODA cláusula y documento de respuestas cortas.
      d. Residual: ASPECTO, ``>= cfg.min_palabras_residual`` palabras, ``p < cfg.umbral_residual_u4``, de respuesta no
         corta -> «Por descubrir» (el modelado con UMAP/HDBSCAN lo hace :func:`modelar_residual_u4`).
      e. Cláusula de ≤5 palabras con ``p < umbral_residual_u4`` que no es «Comentario General»: conserva el aspecto
         de U4 y se marca ``u4_baja_confianza``. **Caso sin regla**: cláusula corta (≤5 palabras) de una respuesta
         LARGA: no es residual (le faltan palabras para el modelado) ni «Comentario General» (la respuesta no es
         corta); conserva el aspecto de U4, con ``u4_baja_confianza`` si la probabilidad es baja.
      f. VALORACION_GLOBAL («sin contenido») sigue como VALORACION_GLOBAL, sin pasar por U4, en respuestas largas y, con
         ``cfg.sin_contenido_en_cortas`` (por defecto), también en las cortas.

    Plan v2, punto 7: las salidas con código -3 (sin contenido), -5 (Comentario General) o una clase «sin aspecto» predicha por U4
    son UNA sola categoría («Opinión general»); la columna ``motivo_opinion_general`` (``sin_contenido``, ``comentario_corto``,
    ``clasificador`` o vacío) deja constancia de cuál fue el origen.

    Escribe los diagnósticos (umbral de «Comentario General» y del residual) en el log y en el Excel de entrenamiento.
    """
    documentos = documentos.copy()
    clausulas = clausulas.copy().reset_index(drop=True)
    _validar_alineacion_documentos(documentos, encuesta)
    textos = [str(t) for t in encuesta.texto_original]
    n_pal = np.fromiter((len(t.split()) for t in textos), dtype=int, count=len(textos))
    corta_doc = n_pal <= int(cfg.max_palabras_respuesta_corta)
    documentos["respuesta_completa"] = textos  # texto EXACTO, sin limpiar
    documentos["n_palabras_respuesta"] = n_pal
    documentos["respuesta_corta"] = corta_doc
    doc_ids = clausulas["doc_id"].to_numpy(dtype=int)
    corta = corta_doc[doc_ids] if len(clausulas) else np.zeros(0, bool)
    clausulas["respuesta_corta"] = corta
    motivo = clausulas["motivo"].to_numpy()
    es_aspecto = motivo == MOTIVO_ASPECTO
    es_global = motivo == MOTIVO_GLOBAL
    # NO_OPINION / RUIDO no pasan; la cláusula sin contenido solo pasa si la regla de respuestas cortas está apagada
    pasa = es_aspecto | (es_global & corta & (not bool(cfg.sin_contenido_en_cortas)))

    clases = list(modelo.clases)
    k = len(clases)
    cols_p = nombres_columnas_p(clases)
    mat = np.full((len(clausulas), k), np.nan)
    idx = np.flatnonzero(pasa)
    logger.info("  U4 «%s»: etiquetando %d cláusulas (%d únicas)...", modelo.run_id, len(idx), clausulas.loc[pasa, "clausula"].nunique())
    if len(idx):
        mat[idx] = etiquetar_clausulas(modelo, clausulas.loc[pasa, "clausula"].tolist())
    p_max = np.full(len(clausulas), np.nan)
    cod_u4 = np.full(len(clausulas), -999, dtype=int)
    if len(idx):
        p_max[idx] = mat[idx].max(1)
        cod_u4[idx] = mat[idx].argmax(1)
    clausulas = pd.concat([clausulas, pd.DataFrame(mat, columns=cols_p, index=clausulas.index)], axis=1)
    clausulas["aspecto_u4"] = [clases[c] if c >= 0 else None for c in cod_u4]
    clausulas["p_aspecto_u4"] = p_max

    # ── asignación ──
    topico = np.full(len(clausulas), EXCLUIDO, dtype=int)
    topico[es_global] = VALORACION_GLOBAL
    asignacion = np.array(
        [{MOTIVO_NO_OPINION: "no_opinion", MOTIVO_RUIDO: "ruido", MOTIVO_GLOBAL: "valoracion_global"}.get(m)
         for m in motivo],
        dtype=object,
    )
    topico[pasa] = cod_u4[pasa]
    asignacion[pasa] = "u4"
    es_cg = pasa & corta & (p_max < float(cfg.umbral_comentario_general))
    topico[es_cg] = COMENTARIO_GENERAL
    asignacion[es_cg] = "regla_comentario_general"
    n_pal_cl = clausulas["clausula"].astype(str).str.split().str.len().to_numpy()
    es_residual = np.zeros(len(clausulas), bool)
    if cfg.residual_hdbscan_activo:
        es_residual = es_aspecto & ~corta & (n_pal_cl >= int(cfg.min_palabras_residual)) & (p_max < float(cfg.umbral_residual_u4))
    topico[es_residual] = POR_DESCUBRIR
    asignacion[es_residual] = "residual_hdbscan"
    baja = pasa & ~es_cg & ~es_residual & (n_pal_cl < int(cfg.min_palabras_residual)) & (p_max < float(cfg.umbral_residual_u4))
    nombres_especiales = dict(ETIQUETAS_ESPECIALES)
    clausulas["topico"] = topico
    clausulas["aspecto"] = [clases[t] if t >= 0 else nombres_especiales.get(int(t), "") for t in topico]
    clausulas["asignacion"] = asignacion
    # Plan v2, punto 7: origen de la categoría «Opinión general» (columna interna; no va a las tablas del contrato)
    sin_aspecto_u4 = {norm_txt_u4(c) for c in modelo.config.get("clases_sin_aspecto", CLASES_SIN_ASPECTO_U4)}
    es_clase_og = np.array([bool(t >= 0 and norm_txt_u4(clases[t]) in sin_aspecto_u4) for t in topico], dtype=bool)
    motivo_og = np.full(len(clausulas), "", dtype=object)
    motivo_og[es_clase_og] = MOTIVO_OG_CLASIFICADOR
    motivo_og[topico == COMENTARIO_GENERAL] = MOTIVO_OG_COMENTARIO_CORTO
    motivo_og[topico == VALORACION_GLOBAL] = MOTIVO_OG_SIN_CONTENIDO
    clausulas["motivo_opinion_general"] = motivo_og
    clausulas["es_residual"] = es_residual
    # Nulo (<NA>) en las cláusulas que NO son residual; -1 solo en las residuales que HDBSCAN dejó sin grupo.
    clausulas["topico_residual"] = pd.array([pd.NA] * len(clausulas), dtype="Int64")
    clausulas["etiqueta_residual"] = ""
    clausulas["u4_baja_confianza"] = baja
    vistos = modelo.textos_vistos
    clausulas["u4_visto_en_entrenamiento"] = [
        bool(p and norm_txt_u4(t) in vistos) for t, p in zip(clausulas["clausula"].astype(str), pasa)
    ]  # coincidencia APROXIMADA de texto (minúsculas, sin tildes, espacios colapsados); no es identidad de respuesta
    clausulas["modelo_u4_id"] = modelo.run_id

    # ── diagnósticos ──
    diag_cg = diagnostico_comentario_general(p_max[pasa & corta], float(cfg.umbral_comentario_general))
    diag_res = diagnostico_residual(clausulas, modelo.holdout, int(cfg.min_palabras_residual), float(cfg.umbral_residual_u4))
    logger.info(
        "  U4: %d cláusulas con U4 · Comentario General %d · residual (Por descubrir) %d · baja confianza %d · respuesta corta %d docs",
        int(pasa.sum()),
        int(es_cg.sum()),
        int(es_residual.sum()),
        int(baja.sum()),
        int(corta_doc.sum()),
    )
    if len(diag_cg["umbrales"]):
        logger.info(
            "  DIAGNÓSTICO «Comentario General» (umbral actual %.2f = PROVISIONAL): %%%% de cláusulas de respuestas cortas que "
            "pasarían a Comentario General por umbral -> %s",
            float(cfg.umbral_comentario_general),
            {float(u): float(v) for u, v in zip(diag_cg["umbrales"]["umbral"], diag_cg["umbrales"]["pct_comentario_general"])},
        )
    if len(diag_res):
        logger.info(
            "  DIAGNÓSTICO residual: %% del corpus ASPECTO que sería residual por umbral -> %s",
            {float(u): float(v) for u, v in zip(diag_res["umbral"], diag_res["pct_del_corpus_aspecto"])},
        )
    if escribir_diagnosticos and modelo.carpeta is not None and (Path(modelo.carpeta) / ARCHIVO_ENTRENAMIENTO_TEST).exists():
        sello = f"{datetime.now():%Y%m%d-%H%M%S}"
        hoja_cg = pd.concat(
            [
                diag_cg["umbrales"].assign(seccion="umbrales"),
                diag_cg["cuantiles"].assign(seccion="cuantiles"),
                diag_cg["distribucion"].assign(seccion="distribucion"),
            ],
            ignore_index=True,
        )
        _reemplazar_hojas_excel(
            Path(modelo.carpeta) / ARCHIVO_ENTRENAMIENTO_TEST,
            {f"diag_cg_{sello}": hoja_cg, f"diag_res_{sello}": diag_res},
        )  # hojas con sello de fecha-hora: cada etiquetado AÑADE las suyas y no pisa las de corridas anteriores
    return EtapasU4(
        documentos=documentos,
        clausulas=clausulas,
        modelo=modelo,
        diagnostico_cg=diag_cg,
        diagnostico_residual=diag_res,
    )


# ── B2. RESIDUAL: UMAP/HDBSCAN SOLO SOBRE LAS CLÁUSULAS QUE U4 NO RECONOCE ─


def _tabla_residual(
    res: pd.DataFrame, tr: np.ndarray, terminos: dict[int, list[str]], cohesion: dict[int, float], metricas: dict[str, Any]
) -> pd.DataFrame:
    """Una fila por tópico residual (y una fila -1 = sin grupo) con términos c-TF-IDF y aspecto U4 dominante."""
    columnas = [
        "topico_residual", "etiqueta", "terminos", "clausulas", "documentos", "cohesion", "ejemplos",
        "aspecto_u4_dominante", "pureza_u4", "residual_n_clausulas", "residual_n_topicos", "residual_pct_outliers", "residual_silueta",
    ]
    filas = []
    for t in sorted(set(tr.tolist())):
        g = res.loc[tr == t]
        if g.empty:
            continue
        cuenta = g["aspecto_u4"].value_counts()
        term = terminos.get(int(t), [])
        filas.append(
            {
                "topico_residual": int(t),
                "etiqueta": ("SIN GRUPO (outliers del residual)" if t < 0 else f"R{int(t)} · " + " · ".join(term[:3])),
                "terminos": " | ".join(term),
                "clausulas": int(len(g)),
                "documentos": int(g["doc_id"].nunique()),
                "cohesion": cohesion.get(int(t), float("nan")),
                "ejemplos": " || ".join(e[:120] for e in g["clausula"].drop_duplicates().head(3)),
                "aspecto_u4_dominante": str(cuenta.index[0]) if len(cuenta) else "",
                "pureza_u4": round(100 * float(cuenta.iloc[0] / len(g)), 1) if len(cuenta) else float("nan"),
                "residual_n_clausulas": metricas.get("n_clausulas"),
                "residual_n_topicos": metricas.get("n_topicos"),
                "residual_pct_outliers": metricas.get("pct_outliers"),
                "residual_silueta": metricas.get("silueta"),
            }
        )
    return pd.DataFrame(filas, columns=columnas)


def modelar_residual_u4(  # pylint: disable=too-many-locals,too-many-arguments
    etapas: EtapasU4,
    cfg: Config,
    recursos: RecursosLexicos,
    *,
    codificador: Codificador | None = None,
    clave_stopwords: str | None = None,
) -> EtapasU4:
    """Etapa B2: descubre aspectos nuevos con UMAP → HDBSCAN → fusión → reasignación → c-TF-IDF sobre el residual.

    Usa los embeddings mpnet ORIGINALES (``codificador``) del ``texto_aspecto`` neutralizado, NO los de U4. Con menos
    cláusulas que ``cfg.residual_minimo`` (por defecto ``MINIMO_MODELABLES``) no se ejecuta HDBSCAN: todas quedan
    «Por descubrir» con ``topico_residual = -1`` (sin grupo; en las cláusulas no residuales es nulo). Criterios de negocio RELAJADOS (``CRITERIOS_RESIDUAL_POR_DEFECTO``,
    sobrescribibles con ``cfg.residual_criterios``); rejilla ``cfg.residual_rejilla``; ``permitir_no_cumple=True``.
    El código de aspecto de estas cláusulas sigue siendo «Por descubrir» (``topico = POR_DESCUBRIR``).
    """
    cl = etapas.clausulas
    mask = cl["es_residual"].to_numpy(dtype=bool)
    res = cl.loc[mask]
    minimo = int(cfg.residual_minimo) if cfg.residual_minimo is not None else MINIMO_MODELABLES
    metricas: dict[str, Any] = {
        "activo": bool(cfg.residual_hdbscan_activo),
        "ejecutado": False,
        "motivo_no_ejecutado": "",
        "n_clausulas": int(mask.sum()),
        "n_textos_unicos": int(res["texto_aspecto"].nunique()) if len(res) else 0,
        "minimo_residual": minimo,
        "umbral_residual_u4": float(cfg.umbral_residual_u4),
        "min_palabras_residual": int(cfg.min_palabras_residual),
        "rejilla": cfg.residual_rejilla,
        "n_topicos": 0,
        "pct_outliers": float("nan"),
        "silueta": float("nan"),
        "ari_medio": float("nan"),
    }
    tr = np.full(int(mask.sum()), -1, dtype=int)
    terminos: dict[int, list[str]] = {}
    cohesion: dict[int, float] = {}
    p_residual: np.ndarray | None = None
    if not cfg.residual_hdbscan_activo:
        metricas["motivo_no_ejecutado"] = "residual_hdbscan_activo=False"
    elif not mask.any():
        metricas["motivo_no_ejecutado"] = "no hay cláusulas residuales"
    elif mask.sum() < minimo:
        metricas["motivo_no_ejecutado"] = f"{int(mask.sum())} cláusulas < mínimo modelable {minimo}: no se ejecuta HDBSCAN"
        logger.warning("  residual: %s; todas quedan «Por descubrir» con topico_residual=-1.", metricas["motivo_no_ejecutado"])
    else:
        criterios = CriteriosNegocio(**{**CRITERIOS_RESIDUAL_POR_DEFECTO, **dict(cfg.residual_criterios)})
        metricas["criterios"] = asdict(criterios)
        textos = res["texto_aspecto"].tolist()
        grid = GridSearchAspectos(cfg, recursos, codificador, criterios).preparar(textos)
        grid.modelador.clave_stopwords = clave_stopwords
        try:
            grid.ejecutar(cfg.residual_rejilla)
            cfg_fin = grid.mejor_config(permitir_no_cumple=True)
        except SinConfiguracionValida as exc:
            metricas["motivo_no_ejecutado"] = f"sin configuración evaluable: {exc}"
            logger.warning("  residual: %s", metricas["motivo_no_ejecutado"])
        else:
            modelador = grid.modelador
            modelador.cfg = cfg_fin
            codigos = pd.Series(textos).map({t: i for i, t in enumerate(modelador.textos)}).to_numpy()
            etiquetas = modelador.particionar(grid.proyeccion_de(cfg_fin))
            try:
                modelador.representar()
                n_t = int(etiquetas.max() + 1) if (etiquetas >= 0).any() else 0
                terminos = {t: modelador.terminos(t, 8) for t in range(n_t)}
            except ImportError as exc:  # bertopic ausente: se conservan los grupos, sin términos
                logger.warning("  residual: sin términos c-TF-IDF (%s)", exc)
            cohesion = modelador.cohesion()
            tr = np.asarray(etiquetas[codigos], dtype=int)
            try:  # estabilidad: UMAP/HDBSCAN reajustados con prediction_data=True; un fallo aquí no detiene la corrida
                etapas.residual_persistible = ajustar_residual_persistible(modelador, cfg_fin)
                p_residual = etapas.residual_persistible.probabilidades[codigos]
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("  residual: no se pudo ajustar el modelo persistible de UMAP/HDBSCAN (%s)", exc)
            coincide = grid.resultados[
                (grid.resultados["umap_n_neighbors"] == cfg_fin.umap_n_neighbors)
                & (grid.resultados["min_topic_size"] == cfg_fin.min_topic_size)
                & (grid.resultados["min_samples"] == cfg_fin.min_samples)
                & (grid.resultados["metodo"] == cfg_fin.cluster_selection_method)
            ]
            metricas.update(
                ejecutado=True,
                n_topicos=int((tr >= 0).any() and tr.max() + 1),
                pct_outliers=round(100 * float((tr == OUTLIER).mean()), 2),
                silueta=modelador.diagnostico.get("silueta", float("nan")),
                ari_medio=float(coincide["ari_medio"].iloc[0]) if not coincide.empty else float("nan"),
                config_elegida={
                    "umap_n_neighbors": cfg_fin.umap_n_neighbors,
                    "umap_n_components": cfg_fin.umap_n_components,
                    "min_topic_size": cfg_fin.min_topic_size,
                    "min_samples": cfg_fin.min_samples,
                    "metodo": cfg_fin.cluster_selection_method,
                },
            )
            etapas.modelador_residual = modelador
    cl = cl.copy()
    cl["topico_residual"] = cl["topico_residual"].astype("Int64")
    cl.loc[mask, "topico_residual"] = tr
    # Pertenencia al grupo de HDBSCAN (contrato: confianza_aspecto de las cláusulas hdbscan). Vacía (NaN) si no hay modelo
    # persistible o si el texto era outlier de HDBSCAN y se reasignó por centroide.
    cl["p_residual"] = np.nan
    if p_residual is not None and mask.any():
        cl.loc[mask, "p_residual"] = p_residual
    etiq = {t: f"R{t} · " + " · ".join(terminos.get(t, [])[:3]) for t in set(tr.tolist()) if t >= 0}
    cl.loc[mask, "etiqueta_residual"] = [etiq.get(int(t), "") for t in tr]
    etapas.clausulas = cl
    etapas.metricas_residual = metricas
    etapas.tabla_residual = (
        _tabla_residual(cl.loc[mask], tr, terminos, cohesion, metricas) if mask.any() else pd.DataFrame()
    )
    etapas.residual_listo = True
    logger.info(
        "  residual: %d cláusulas -> %d tópicos | %% outliers %s | silueta %s | ejecutado=%s",
        metricas["n_clausulas"],
        metricas["n_topicos"],
        metricas["pct_outliers"],
        metricas["silueta"],
        metricas["ejecutado"],
    )
    return etapas


def asignar_aspectos_u4(
    modelo: ModeloU4,
    encuesta: Encuesta,
    documentos: pd.DataFrame,
    clausulas: pd.DataFrame,
    cfg: Config,
    recursos: RecursosLexicos,
    *,
    codificador: Codificador | None = None,
) -> EtapasU4:
    """Etapas B + B2 de una vez: etiquetado del corpus, reglas y modelado del residual."""
    etapas = etiquetar_corpus_u4(modelo, encuesta, documentos, clausulas, cfg)
    return modelar_residual_u4(etapas, cfg, recursos, codificador=codificador, clave_stopwords=encuesta.clave_stopwords)


def tabla_aspectos_u4(modelo: ModeloU4, clausulas: pd.DataFrame, semilla: int = 42) -> pd.DataFrame:
    """Tabla de aspectos del modo u4 (reemplaza a ``tabla_topicos``): una fila por clase de U4 y por código especial."""
    columnas = [
        "topico", "etiqueta", "aspecto", "terminos", "clausulas", "documentos", "textos_unicos", "cohesion",
        "crecimiento_reasignacion_pct", "ejemplos", "p_media", "p_mediana", "especial", "revisar",
    ]
    especiales = set(modelo.clases_especiales)
    filas = []
    codigos = [(i, c, c in especiales) for i, c in enumerate(modelo.clases)] + [
        (COMENTARIO_GENERAL, NOMBRE_COMENTARIO_GENERAL, True),
        (POR_DESCUBRIR, NOMBRE_POR_DESCUBRIR, True),
    ]
    for cod, nombre, especial in codigos:
        g = clausulas[clausulas["topico"] == cod]
        if cod < 0 and g.empty:
            continue
        ejemplos = (
            g["clausula"].drop_duplicates().sample(min(3, g["clausula"].nunique()), random_state=semilla).tolist() if len(g) else []
        )
        filas.append(
            {
                "topico": int(cod),
                "etiqueta": nombre,
                "aspecto": nombre,
                "terminos": "",
                "clausulas": int(len(g)),
                "documentos": int(g["doc_id"].nunique()),
                "textos_unicos": int(g["clausula"].nunique()),
                "cohesion": float("nan"),
                "crecimiento_reasignacion_pct": 0.0,
                "ejemplos": " || ".join(e[:120] for e in ejemplos),
                "p_media": round(float(g["p_aspecto_u4"].mean()), 4) if len(g) else float("nan"),
                "p_mediana": round(float(g["p_aspecto_u4"].median()), 4) if len(g) else float("nan"),
                "especial": bool(especial),
                "revisar": False,
            }
        )
    return pd.DataFrame(filas, columns=columnas)


def metricas_aspectos_u4(clausulas: pd.DataFrame, modelo: ModeloU4, metricas_residual: dict[str, Any]) -> dict[str, Any]:
    """Métricas del modo u4: cobertura, % «Comentario General», % «Por descubrir», % de aspecto de la lista y hold-out.

    Denominador = cláusulas SUJETAS a asignación de aspecto: motivo ASPECTO, más las VALORACION_GLOBAL de respuestas cortas.
    """
    sujetas = clausulas["asignacion"].isin(["u4", "regla_comentario_general", "residual_hdbscan"])
    n = max(int(sujetas.sum()), 1)
    t = clausulas.loc[sujetas, "topico"]
    especiales = {i for i, c in enumerate(modelo.clases) if c in set(modelo.clases_especiales)}
    lista = (t >= 0) & ~t.isin(list(especiales))
    return {
        "modo_aspectos": MODO_U4,
        "modelo_u4_id": modelo.run_id,
        "n_clases_u4": len(modelo.clases),
        "n_clausulas_sujetas_aspecto": int(sujetas.sum()),
        "cobertura_u4": round(100 * float((t >= 0).sum()) / n, 2),
        "pct_comentario_general": round(100 * float((t == COMENTARIO_GENERAL).sum()) / n, 2),
        "pct_por_descubrir": round(100 * float((t == POR_DESCUBRIR).sum()) / n, 2),
        "pct_aspecto_lista": round(100 * float(lista.sum()) / n, 2),
        "pct_clausulas_respuesta_corta": round(100 * float(clausulas["respuesta_corta"].mean()), 2) if len(clausulas) else float("nan"),
        "pct_baja_confianza": round(100 * float(clausulas["u4_baja_confianza"].mean()), 2) if len(clausulas) else float("nan"),
        "acierto_holdout": modelo.metricas.get("acierto_holdout", float("nan")),
        "f1_macro_holdout": modelo.metricas.get("f1_macro_holdout", float("nan")),
        "n_holdout_evaluable": modelo.metricas.get("n_holdout_evaluable"),
        "umbral_comentario_general": modelo.config.get("umbral_comentario_general"),
        "residual_n_clausulas": metricas_residual.get("n_clausulas"),
        "residual_n_topicos": metricas_residual.get("n_topicos"),
        "residual_pct_outliers": metricas_residual.get("pct_outliers"),
        "residual_silueta": metricas_residual.get("silueta"),
        "residual_ejecutado": metricas_residual.get("ejecutado"),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 15. ORQUESTADOR
# ═══════════════════════════════════════════════════════════════════════════


def procesar_encuesta(  # pylint: disable=too-many-arguments,too-many-locals
    encuesta: Encuesta,
    cfg: Config,
    recursos: RecursosLexicos,
    *,
    codificador: Codificador | None = None,
    clasificador: ClasificadorPolaridad | None = None,
    unidades: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    grid: GridSearchAspectos | None = None,
    modelo_u4: ModeloU4 | None = None,
    etapas_u4: EtapasU4 | None = None,
) -> Resultado:
    """Ejecuta ambas ramas sobre una encuesta y audita el resultado.

    Args:
        encuesta: Columna a procesar.
        cfg: Configuración final (normalmente de ``grid.mejor_config()``).
        recursos: Léxicos.
        codificador: Encoder inyectable (por defecto SentenceTransformers).
        clasificador: Polaridad inyectable (por defecto RoBERTuito).
        unidades: ``construir_unidades`` ya calculado.
        grid: Grid ejecutado; se reutilizan sus embeddings y proyección
            para que el modelo sea idéntico a la fila elegida.
        modelo_u4: (solo ``cfg.modo_aspectos="u4"``) modelo U4 de esta pregunta; si es ``None`` se carga de
            ``cfg.ruta_modelo_u4``.
        etapas_u4: (solo modo u4) etapas B/B2 ya calculadas (``etiquetar_corpus_u4`` / ``modelar_residual_u4``);
            si se pasan, no se recalculan.

    Returns:
        :class:`Resultado` completo.
    """
    logger.info("=" * 74)
    logger.info("ENCUESTA: %s", encuesta.nombre)
    logger.info("=" * 74)
    documentos, clausulas = unidades or construir_unidades(encuesta, cfg, recursos)
    clausulas = clausulas.copy()
    es_aspecto = clausulas["motivo"] == MOTIVO_ASPECTO

    # ── RAMA A: aspectos ───────────────────────────────────────────────────
    clausulas["topico"] = EXCLUIDO
    clausulas.loc[clausulas["motivo"] == MOTIVO_GLOBAL, "topico"] = VALORACION_GLOBAL
    modelador: ModeladorAspectos | None = None
    ari = float("nan")
    huella_grid: str | None = None
    n_unicos = int(clausulas.loc[es_aspecto, "texto_aspecto"].nunique())
    clausulas["asignacion"] = None  # se llena abajo solo para ASPECTO
    modo = (
        MODO_U4
        if cfg.modo_aspectos == MODO_U4
        else (
            MODO_BERTOPIC
            if grid is not None or int(es_aspecto.sum()) >= MINIMO_MODELABLES
            else MODO_NINGUNO
        )
    )
    logger.info("  modo de aspectos: %s (%d textos de aspecto únicos)", modo, n_unicos)
    etapas: EtapasU4 | None = None
    if modo == MODO_U4:
        # ft-r4.15: el aspecto lo decide U4; UMAP/HDBSCAN solo actúan sobre el residual (modelar_residual_u4).
        if modelo_u4 is None:
            if cfg.ruta_modelo_u4 is None:
                raise ValueError(
                    "modo_aspectos='u4' requiere un modelo: pase modelo_u4=... o fije Config.ruta_modelo_u4 "
                    "(carpeta de la corrida o de la pregunta con ultimo.json). Entrénelo con entrenar_u4()."
                )
            modelo_u4 = cargar_u4(cfg.ruta_modelo_u4)
        etapas = etapas_u4 or etiquetar_corpus_u4(modelo_u4, encuesta, documentos, clausulas, cfg)
        if not etapas.residual_listo:
            etapas = modelar_residual_u4(
                etapas, cfg, recursos, codificador=codificador, clave_stopwords=encuesta.clave_stopwords
            )
        documentos, clausulas = etapas.documentos, etapas.clausulas.copy()
        es_aspecto = clausulas["motivo"] == MOTIVO_ASPECTO
    elif modo == MODO_BERTOPIC:
        if grid is not None:
            modelador = grid.modelador
            modelador.cfg = cfg
            codigos = (
                pd.Series(clausulas.loc[es_aspecto, "texto_aspecto"])
                .map({t: i for i, t in enumerate(modelador.textos)})
                .to_numpy()
            )
            reducidos = grid.proyeccion_de(cfg)
            coincide = grid.resultados[
                (grid.resultados["umap_n_neighbors"] == cfg.umap_n_neighbors)
                & (grid.resultados["umap_n_components"] == cfg.umap_n_components)
                & (grid.resultados["min_topic_size"] == cfg.min_topic_size)
                & (grid.resultados["min_samples"] == cfg.min_samples)
                & (grid.resultados["metodo"] == cfg.cluster_selection_method)
            ]
            if not coincide.empty:
                ari = float(coincide["ari_medio"].iloc[0])
                huella_grid = str(coincide["huella_particion"].iloc[0])
        else:
            modelador = ModeladorAspectos(
                cfg, recursos, codificador, encuesta.clave_stopwords
            )
            codigos = modelador.preparar(
                clausulas.loc[es_aspecto, "texto_aspecto"].tolist()
            )
            reducidos = None
        modelador.clave_stopwords = encuesta.clave_stopwords
        etiquetas = modelador.particionar(reducidos)
        if cfg.podar_cohesion:
            etiquetas = modelador.podar_por_cohesion(cfg.k_cohesion)
        modelador.representar()
        clausulas.loc[es_aspecto, "topico"] = etiquetas[codigos]
        # Trazabilidad de la asignación: "directa" = HDBSCAN la puso
        # en un tópico; "reasignada" = HDBSCAN la dejó como outlier y
        # reasignar_outliers la acercó al centroide más cercano; "sin_topico" =
        # quedó outlier. La cobertura reportada antes mezclaba las dos primeras.
        finales = etiquetas[codigos]
        crudas = modelador.etiquetas_hdbscan[codigos]
        clausulas.loc[es_aspecto, "asignacion"] = np.where(
            finales == OUTLIER,
            "sin_topico",
            np.where(crudas == OUTLIER, "reasignada", "directa"),
        )
    else:
        logger.warning(
            "  %d cláusulas con aspecto (< %d): sin modelo de tópicos",
            int(es_aspecto.sum()),
            MINIMO_MODELABLES,
        )
        clausulas.loc[es_aspecto, "topico"] = OUTLIER
        clausulas.loc[es_aspecto, "asignacion"] = "sin_topico"
    clausulas["topico"] = clausulas["topico"].astype(int)

    # ── RAMA B: sentimiento (sólo opiniones) ───────────────────────────────
    analizador = AnalizadorSentimiento(cfg, recursos, clasificador)
    opinion = clausulas["motivo"].isin([MOTIVO_ASPECTO, MOTIVO_GLOBAL]).to_numpy()
    columnas_sent = [
        "p_pos",
        "p_neu",
        "p_neg",
        "sentimiento_modelo",
        "confianza",
        "polaridad",
        "sentimiento",
        "neu_ajustado",
    ]
    clausulas = clausulas.reset_index(drop=True)
    sentimiento = pd.DataFrame(
        index=clausulas.index, columns=columnas_sent, dtype=object
    )
    tipo = tipo_pregunta(encuesta.columna)
    if opinion.any():
        clasificadas = analizador.clasificar(
            clausulas.loc[opinion, "clausula"].tolist(), tipo
        )
        clasificadas.index = clausulas.index[opinion]
        sentimiento.loc[clasificadas.index, columnas_sent] = clasificadas[columnas_sent]
    for columna in ("p_pos", "p_neu", "p_neg", "confianza", "polaridad"):
        sentimiento[columna] = pd.to_numeric(sentimiento[columna])
    sentimiento["neu_ajustado"] = sentimiento["neu_ajustado"].astype("boolean")
    clausulas = pd.concat(
        [
            clausulas,
            sentimiento,
            analizador.marcar_modificadores(clausulas["clausula"].tolist()),
        ],
        axis=1,
    )

    # ── FUSIÓN Y AUDITORÍA ─────────────────────────────────────────────────
    if modo == MODO_U4:
        topicos = sugerir_aspectos(None, tabla_aspectos_u4(modelo_u4, clausulas, cfg.random_state), cfg)
    else:
        topicos = sugerir_aspectos(modelador, tabla_topicos(modelador, clausulas), cfg)
    absa = construir_absa(clausulas, topicos, cfg)
    autor = autor_texto(encuesta.hoja, encuesta.columna)
    validacion = validar_contra_nota(clausulas, encuesta.notas(), autor)
    extra: dict[str, bool] = {}
    if modelador is not None and "representacion_alineada" in modelador.diagnostico:
        extra["términos c-TF-IDF alineados con su tópico"] = bool(
            modelador.diagnostico["representacion_alineada"]
        )
    if huella_grid is not None and modelador is not None:
        extra["partición final = fila del grid"] = (
            modelador.diagnostico.get("huella_particion") == huella_grid
        )
    conciliacion = conciliar(documentos, clausulas, absa, extra)
    asignadas = int((clausulas["topico"] >= 0).sum())
    con_aspecto = int(es_aspecto.sum())
    arr = clausulas.loc[es_aspecto, "topico"].to_numpy()
    distribucion = clausulas["sentimiento"].value_counts(normalize=True)
    metricas: dict[str, Any] = {
        "encuesta": encuesta.nombre,
        "modo_aspectos": modo,
        "tipo_pregunta": tipo,
        "autor_texto": autor,
        "encoder": cfg.encoder,
        "sentimiento_calibrado": analizador.calibrador is not None,
        "filas": len(documentos),
        "documentos_con_opinion": int((documentos["estado"] == DOC_CON_OPINION).sum()),
        "n_clausulas": len(clausulas),
        "clausulas_por_documento_con_opinion": round(
            len(clausulas)
            / max(
                int(
                    (documentos["estado"] != DOC_VACIA).sum()
                    - (documentos["estado"] == DOC_RUIDO).sum()
                ),
                1,
            ),
            3,
        ),
        "clausulas_con_aspecto": con_aspecto,
        "pct_valoracion_global": round(
            100
            * float((clausulas["topico"] == VALORACION_GLOBAL).sum())
            / max(int(opinion.sum()), 1),
            2,
        ),
        "n_topicos": int(topicos["topico"].nunique()) if not topicos.empty else 0,
        "pct_outliers": (
            round(100 * float((arr == OUTLIER).mean()), 2)
            if con_aspecto
            else float("nan")
        ),
        "cobertura_aspectual": round(100 * asignadas / max(con_aspecto, 1), 2),
        # Parte de la cobertura que HDBSCAN asignó por sí mismo (sin
        # reasignación por cercanía). La puerta G3 sigue usando la total.
        "cobertura_directa": round(
            100 * int((clausulas["asignacion"] == "directa").sum()) / max(con_aspecto, 1),
            2,
        ),
        **concentracion(arr),
        "silueta": (
            modelador.diagnostico.get("silueta") if modelador else float("nan")
        ),
        "max_coseno": (
            max_coseno_centroides(modelador.embeddings, modelador.etiquetas)
            if modelador
            and modelador.embeddings is not None
            and modelador.etiquetas is not None
            else float("nan")
        ),
        "diversidad": modelador.diversidad() if modelador else float("nan"),
        "pares_fusionados": len(modelador.pares_fusionados) if modelador else 0,
        "ari_medio": ari,
        "pct_positivo": round(100 * float(distribucion.get(POS, 0)), 2),
        "pct_neutro": round(100 * float(distribucion.get(NEU, 0)), 2),
        "pct_negativo": round(100 * float(distribucion.get(NEG, 0)), 2),
        "pct_neu_ajustado": round(
            100
            * float(clausulas["neu_ajustado"].fillna(False).astype(bool).sum())
            / max(int(opinion.sum()), 1),
            2,
        ),
        "confianza_media": round(float(clausulas["confianza"].mean()), 4),
        **(
            {k: v for k, v in modelador.diagnostico.items() if k != "silueta"}
            if modelador
            else {}
        ),
    }
    if modo == MODO_U4:
        metricas.update(metricas_aspectos_u4(clausulas, modelo_u4, etapas.metricas_residual))
        # Las claves de la ruta bertopic se reinterpretan: cobertura = ASPECTO con aspecto de U4 (código >= 0).
        metricas["cobertura_aspectual"] = round(
            100 * float((clausulas.loc[es_aspecto, "topico"] >= 0).sum()) / max(con_aspecto, 1), 2
        )
        metricas["cobertura_directa"] = float("nan")  # no hay asignación «directa» de HDBSCAN
        metricas["n_topicos"] = int(topicos.loc[(topicos["topico"] >= 0) & ~topicos["especial"] & (topicos["clausulas"] > 0), "topico"].nunique())
        metricas["pct_documentos_respuesta_corta"] = round(100 * float(documentos["respuesta_corta"].mean()), 2)
    scorecard = construir_scorecard(metricas, validacion, conciliacion)
    huellas = {
        "config": cfg.huella(),
        "lexicos": recursos.huella(),
        "corpus": huella_texto(encuesta.columna, encuesta.texto_original),
        "version": VERSION,
        **({"modelo_u4": huella_texto(modelo_u4.run_id, modelo_u4.config.get("huella_entrenamiento"))} if modo == MODO_U4 else {}),
    }
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{huella_texto(huellas)[:10]}"
    logger.info(
        "[OK] %s (%s): %d tópicos | cobertura %.1f%% (directa %.1f%%) | top1 %.1f%% | "
        "global %.1f%% | NEG %.1f%% | nota %.1f",  # + cobertura directa
        encuesta.nombre,
        modo,
        metricas["n_topicos"],
        metricas["cobertura_aspectual"],
        metricas["cobertura_directa"],
        metricas["pct_top1"],
        metricas["pct_valoracion_global"],
        metricas["pct_negativo"],
        scorecard["nota"],
    )
    return Resultado(
        encuesta=encuesta.nombre,
        cfg=cfg,
        run_id=run_id,
        huellas=huellas,
        documentos=documentos,
        clausulas=clausulas,
        topicos=topicos,
        absa=absa,
        metricas=metricas,
        validacion=validacion,
        conciliacion=conciliacion,
        scorecard=scorecard,
        muestra_pureza=muestra_pureza(
            clausulas, topicos, cfg.muestra_pureza_por_topico, cfg.random_state
        ),
        muestra_manual=muestra_manual(
            clausulas, cfg.muestra_manual_n, cfg.random_state
        ),
        grid=grid.resultados if grid is not None else pd.DataFrame(),
        modelador=modelador,
        modo=modo,
        topicos_residuales=etapas.tabla_residual if etapas is not None else pd.DataFrame(),
        u4=(
            {
                "modelo_u4_id": modelo_u4.run_id,
                "modelo_u4_carpeta": str(modelo_u4.carpeta) if modelo_u4.carpeta else None,
                "huella_datos_auditados": (
                    _leer_json_opcional(Path(modelo_u4.carpeta) / "huella_datos.json") if modelo_u4.carpeta else None
                ),
                "metricas_holdout": {
                    k: modelo_u4.metricas.get(k) for k in ("acierto_holdout", "f1_macro_holdout", "n_holdout_evaluable")
                },
                "umbral_comentario_general": cfg.umbral_comentario_general,
                "max_palabras_respuesta_corta": cfg.max_palabras_respuesta_corta,
                "sin_contenido_en_cortas": cfg.sin_contenido_en_cortas,
                "residual": {
                    "umbral_residual_u4": cfg.umbral_residual_u4,
                    "min_palabras_residual": cfg.min_palabras_residual,
                    "activo": cfg.residual_hdbscan_activo,
                    "rejilla": cfg.residual_rejilla,
                    "criterios": {**CRITERIOS_RESIDUAL_POR_DEFECTO, **dict(cfg.residual_criterios)},
                    "metricas": etapas.metricas_residual,
                },
            }
            if etapas is not None and modelo_u4 is not None
            else {}
        ),
    )


def _leer_json_opcional(ruta: Path) -> Any:
    """JSON de un archivo, o ``None`` si no existe o está dañado."""
    try:
        return json.loads(Path(ruta).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _f1_con_predicciones(
    manual: pd.DataFrame, clausulas: pd.DataFrame
) -> dict[str, Any]:
    """F1 de una muestra anotada usando las predicciones de OTRA tabla.

    Cruza por texto de cláusula; si una cláusula no existe en ``clausulas``
    conserva la predicción que traía la hoja.
    """
    actuales = (
        clausulas.dropna(subset=["sentimiento"])
        .drop_duplicates("clausula")
        .set_index("clausula")["sentimiento"]
    )
    coincidentes = manual["clausula"].map(actuales)
    logger.info(
        "  muestra manual: %d/%d cláusulas con predicción de esta corrida",
        int(coincidentes.notna().sum()),
        len(manual),
    )
    manual = manual.assign(sentimiento=coincidentes.fillna(manual["sentimiento"]))
    poblacion = clausulas["sentimiento"].dropna().value_counts(normalize=True).to_dict()
    return evaluar_muestra_manual(manual, poblacion)


def recalificar(
    resultado: Resultado,
    carpeta: Path,
    ruta_muestra_manual: Path | str | None = None,
) -> dict[str, Any]:
    """Recalcula el scorecard con la auditoría humana, con la sesión VIVA.

    Args:
        resultado: Resultado en memoria.
        carpeta: Carpeta de la corrida (su ``auditoria.xlsx``).
        ruta_muestra_manual: Otro ``auditoria.xlsx`` con ``muestra_manual``
            ya anotada (se cruza por texto de cláusula).

    Returns:
        Scorecard nuevo (también reescribe ``scorecard.json``).
    """
    ruta = Path(carpeta) / "auditoria.xlsx"
    manual = pd.read_excel(ruta_muestra_manual or ruta, sheet_name="muestra_manual")
    resultado.scorecard = construir_scorecard(
        resultado.metricas,
        resultado.validacion,
        resultado.conciliacion,
        evaluar_pureza(ruta),
        _f1_con_predicciones(manual, resultado.clausulas),
    )
    _guardar_scorecard(Path(carpeta), resultado.scorecard)
    return resultado.scorecard


def recalificar_desde_carpeta(
    carpeta: Path | str,
    ruta_muestra_manual: Path | str | None = None,
) -> dict[str, Any]:
    """Recalcula el scorecard SÓLO con los archivos de la corrida.

    No necesita el objeto ``Resultado`` ni volver a correr modelos: sirve
    cuando la sesión de Colab ya se cerró. Lee ``manifest.json`` (métricas
    y validación), la hoja ``conciliacion`` y las hojas humanas de
    ``auditoria.xlsx``, y las predicciones de ``clausulas.parquet``.

    Args:
        carpeta: Carpeta de la corrida (``data/processed/nlp/modelado/<encuesta>/<run_id>``).
        ruta_muestra_manual: Otro ``auditoria.xlsx`` con ``muestra_manual``
            anotada (por defecto la de esta misma carpeta).

    Returns:
        Scorecard recalculado. El original se preserva como
        ``scorecard_original.json`` la primera vez.
    """
    carpeta = Path(carpeta)
    manifiesto = json.loads((carpeta / "manifest.json").read_text(encoding="utf-8"))
    ruta = carpeta / "auditoria.xlsx"
    conciliacion = pd.read_excel(ruta, sheet_name="conciliacion")
    try:
        clausulas = pd.read_parquet(
            carpeta / "clausulas.parquet", columns=["clausula", "sentimiento"]
        )
    except (ImportError, FileNotFoundError):
        clausulas = pd.read_csv(
            carpeta / "clausulas.csv", usecols=["clausula", "sentimiento"]
        )
    manual = pd.read_excel(ruta_muestra_manual or ruta, sheet_name="muestra_manual")
    tarjeta = construir_scorecard(
        manifiesto["metricas"],
        manifiesto.get("validacion_sentimiento", {}),
        conciliacion,
        evaluar_pureza(ruta),
        _f1_con_predicciones(manual, clausulas),
    )
    _guardar_scorecard(carpeta, tarjeta)
    logger.info(
        "Recalificado %s: nota %.1f | topes %s",
        manifiesto.get("encuesta"),
        tarjeta["nota"],
        tarjeta["topes_aplicados"],
    )
    return tarjeta


def _guardar_scorecard(carpeta: Path, tarjeta: dict[str, Any]) -> None:
    """Reescribe scorecard.json preservando el original una sola vez."""
    actual, original = carpeta / "scorecard.json", carpeta / "scorecard_original.json"
    if actual.exists() and not original.exists():
        original.write_text(actual.read_text(encoding="utf-8"), encoding="utf-8")
    actual.write_text(
        json.dumps(_a_json(tarjeta), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _clausulas_de_muestra(ruta: Path) -> set[str]:
    """Textos de cláusula de las hojas humanas de un auditoria.xlsx."""
    textos: set[str] = set()
    for hoja in ("muestra_manual", "muestra_pureza"):
        try:
            marco = pd.read_excel(ruta, sheet_name=hoja, usecols=["clausula"])
        except (ValueError, KeyError):
            continue
        textos |= set(marco["clausula"].dropna().astype(str))
    return textos


def instalar_auditorias(
    subidos: Iterable[Path | str], salida: Path | str
) -> list[Path]:
    """Coloca cada auditoria.xlsx diligenciado en la carpeta de SU corrida.

    El nombre del archivo subido no importa: la corrida se identifica por
    CONTENIDO (las cláusulas de sus muestras), así un Excel renombrado o
    subido a la carpeta equivocada no se mezcla con otra encuesta. La
    primera vez se preserva el original como ``auditoria_original.xlsx``.

    Args:
        subidos: Rutas de los Excel diligenciados.
        salida: Raíz ``data/processed/nlp/modelado`` con una carpeta por encuesta y corrida.

    Returns:
        Carpeta de destino de cada archivo, en el mismo orden.

    Raises:
        FileNotFoundError: Si un Excel no coincide con ninguna corrida.
    """
    corridas = {
        manifiesto.parent: _clausulas_de_muestra(manifiesto.parent / "auditoria.xlsx")
        for manifiesto in sorted(Path(salida).glob("*/*/manifest.json"))
        if (manifiesto.parent / "auditoria.xlsx").exists()
    }
    destinos: list[Path] = []
    for subido in map(Path, subidos):
        propias = _clausulas_de_muestra(subido)
        puntajes = {
            carpeta: len(propias & textos) / max(len(propias | textos), 1)
            for carpeta, textos in corridas.items()
        }
        mejor = max(puntajes, key=lambda c: (puntajes[c], c.name), default=None)
        if mejor is None or puntajes[mejor] < 0.9:
            raise FileNotFoundError(
                f"'{subido.name}' no coincide con ninguna corrida en {salida} "
                f"(mejor coincidencia {max(puntajes.values(), default=0):.0%})."
            )
        original = mejor / "auditoria_original.xlsx"
        if not original.exists():
            original.write_bytes((mejor / "auditoria.xlsx").read_bytes())
        if subido.resolve() != (mejor / "auditoria.xlsx").resolve():
            (mejor / "auditoria.xlsx").write_bytes(subido.read_bytes())
        logger.info("  %s -> %s", subido.name, mejor)
        destinos.append(mejor)
    return destinos


def recalificar_todas(
    salida: Path | str,
    muestras_manuales: dict[str, Path | str] | None = None,
    solo: Iterable[str] | None = None,
    solo_ultima_corrida: bool = True,
) -> pd.DataFrame:
    """Recalifica corridas exportadas bajo ``salida`` (CAMBIO ).

    Args:
        salida: Raíz ``data/processed/nlp/modelado``.
        muestras_manuales: Opcional, slug de encuesta -> auditoria.xlsx con
            ``muestra_manual`` ya anotada en otra corrida (p. ej.
            ``{"evaluacion_docente_p19": ".../<run_r11>/auditoria.xlsx"}``).
        solo: Opcional, nombres de encuesta (los mismos que usa
            ``ejecutar_pipeline(solo=...)``, p. ej. "Evaluación Docente ::
            P19", tal como aparecen en el campo ``encuesta`` de
            ``manifest.json``) a recalificar. ``None`` = todas las que
            tengan corridas.
        solo_ultima_corrida: Si es True (por defecto), cuando una encuesta
            tiene varias carpetas de corrida (varias fechas), sólo se
            recalifica la más reciente por ``run_id`` -- evita mezclar una
            corrida vigente con una vieja que quedó atrás con un
            ``auditoria.xlsx`` sin diligenciar. Póngalo en False para
            recuperar el comportamiento original (todas las corridas).

    Returns:
        Una fila por corrida con nota antes y después y estado de G10/G11.

    Raises:
        ValueError: Si ``solo`` pide una encuesta sin corridas en ``salida``.
    """
    muestras_manuales = muestras_manuales or {}
    manifiestos = sorted(Path(salida).glob("*/*/manifest.json"))

    # Agrupa por el nombre REAL de la encuesta (con "::", leído del propio
    # manifest -- el nombre de carpeta es sólo su slug) para poder filtrar
    # por `solo` con la misma clave que ya usa ejecutar_pipeline, y para
    # poder quedarnos con la corrida más reciente de cada una.
    por_encuesta: dict[str, list[Path]] = {}
    for manifiesto in manifiestos:
        nombre = json.loads(manifiesto.read_text(encoding="utf-8")).get(
            "encuesta", manifiesto.parent.parent.name
        )
        por_encuesta.setdefault(nombre, []).append(manifiesto.parent)

    if solo is not None:
        pedidas = set(solo)
        faltantes = pedidas - set(por_encuesta)
        if faltantes:
            raise ValueError(
                f"No hay corridas para: {sorted(faltantes)}. "
                f"Disponibles: {sorted(por_encuesta)}"
            )
        por_encuesta = {k: v for k, v in por_encuesta.items() if k in pedidas}

    filas = []
    for carpetas in por_encuesta.values():
        # run_id empieza con AAAAMMDD-HHMMSS: orden alfabético == cronológico.
        carpetas = sorted(carpetas)
        seleccionadas = carpetas[-1:] if solo_ultima_corrida else carpetas
        for carpeta in seleccionadas:
            if not (carpeta / "auditoria.xlsx").exists():
                continue
            original = carpeta / "scorecard_original.json"
            previa = json.loads(
                (original if original.exists() else carpeta / "scorecard.json").read_text(
                    encoding="utf-8"
                )
            )
            tarjeta = recalificar_desde_carpeta(
                carpeta, muestras_manuales.get(carpeta.parent.name)
            )
            estados = {p["clave"]: p["estado"] for p in tarjeta["puertas"]}
            filas.append(
                {
                    "encuesta": carpeta.parent.name,
                    "run_id": carpeta.name,
                    "nota_original": previa.get("nota"),
                    "nota": tarjeta["nota"],
                    "G10_pureza": estados.get("G10_pureza_humana"),
                    "G11_f1": estados.get("G11_f1_manual"),
                    "pasan": tarjeta["pasan"],
                    "fallan": tarjeta["fallan"],
                    "topes": "; ".join(tarjeta["topes_aplicados"]),
                }
            )
    return pd.DataFrame(filas)


def ejecutar_pipeline(  # pylint: disable=too-many-arguments,too-many-locals
    cfg: Config | None = None,
    *,
    raiz: Path | None = None,
    dir_lexicos: Path | str | None = None,
    raiz_modelos: Path | str | None = None,
    encuestas: dict[str, str] | None = None,
    solo: Sequence[str] | None = None,
    usar_grid: bool = True,
    rejilla: str = "rapida",
    criterios: CriteriosNegocio | None = None,
    permitir_no_cumple: bool = False,
    codificador: Codificador | None = None,
    clasificador: ClasificadorPolaridad | None = None,
    exportar: bool = True,
    ruta_corpus: Path | str | None = None,
    ruta_catalogo: Path | str | None = None,
    carpeta_datos: Path | str | None = None,
    carpeta_etapa1: Path | str | None = None,
    separador_csv: str = ";",
    modelos_u4: dict[str, ModeloU4 | Path | str] | None = None,
    config_por_encuesta: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Resultado]:
    """Corre el pipeline Ft sobre las encuestas del corpus.

    Args:
        cfg: Configuración base (la del grid la sobrescribe por encuesta).
        raiz: Raíz del proyecto (autodetectada si None).
        dir_lexicos: Carpeta de los léxicos y de ``taxonomia.xlsx`` (``data/utilities``). None = ``raiz``.
        raiz_modelos: Carpeta que contiene ``modelos/u4_aspectos/`` (``models``). None = ``raiz``.
        encuestas: Mapa hoja -> columna (``ENCUESTAS`` si None). Sólo aplica
            a la ruta de ingesta original (``ruta_catalogo`` es None).
        solo: Nombres ``"Hoja :: Pnn"`` a procesar (todas si None).
        usar_grid: Si se busca la configuración por encuesta.
        rejilla: ``"rapida"`` o ``"completa"``.
        criterios: Puertas del grid.
        permitir_no_cumple: Usar la mejor fila aunque no cumpla.
        codificador: Encoder compartido (se crea uno si None).
        clasificador: Clasificador compartido (se crea uno si None).
        exportar: Escribir artefactos en ``cfg.output_dir``.
        ruta_corpus: Excel del corpus "de ejemplo"; sólo se usa si
            ``ruta_catalogo`` es None (ingesta original).
        ruta_catalogo: Ruta a ``preguntas_abiertas.xlsx``. Si se da,
            se activa la ingesta institucional real (CAMBIO 2).
        carpeta_datos: Carpeta con los 6 CSV finales institucionales.
            Requerido si se pasa ``ruta_catalogo``.
        carpeta_etapa1: ``data/processed/etapa1``. Si se da, la ingesta lee ``respuestas.csv`` de la corrida más
            reciente de la Etapa 1 (tiene prioridad sobre ``ruta_catalogo``/``carpeta_datos``).
        separador_csv: Separador de los CSV institucionales.
        modelos_u4: (modo u4) ``clave lógica -> ModeloU4 | ruta``. Sin entrada para una pregunta se usa su
            ``modelos/u4_aspectos/<slug(clave)>/ultimo.json`` (sin reentrenar).
        config_por_encuesta: ``clave lógica -> {campo de Config: valor}`` (p. ej. el umbral del residual de cada
            pregunta, tomado de ``PERFILES``).

    Returns:
        Un :class:`Resultado` por encuesta procesada.
    """
    cfg = cfg or Config()
    raiz = raiz or encontrar_raiz()
    dir_lexicos = Path(dir_lexicos) if dir_lexicos else raiz
    raiz_modelos = Path(raiz_modelos) if raiz_modelos else raiz
    recursos = RecursosLexicos.cargar(dir_lexicos)
    if carpeta_etapa1 is not None:
        cargadas = cargar_encuestas_etapa1(carpeta_etapa1)
    elif ruta_catalogo is not None:
        if carpeta_datos is None:
            raise ValueError(
                "Si pasa ruta_catalogo debe pasar también carpeta_datos "
                "(la carpeta con los 6 CSV finales institucionales)."
            )
        cargadas = cargar_encuestas_institucion(
            Path(ruta_catalogo), Path(carpeta_datos), separador_csv=separador_csv
        )
    else:
        cargadas = cargar_encuestas(
            Path(ruta_corpus) if ruta_corpus else buscar_archivo(raiz, "corpus"),
            encuestas,
        )
    codificador = codificador or EncoderSentenceTransformers(cfg)
    clasificador = clasificador or ClasificadorRobertuito(cfg)
    segmentador = (
        crear_segmentador(cfg) if cfg.metodo_segmentacion == "dependencias" else None
    )
    taxonomias = cargar_taxonomias(dir_lexicos)
    resultados: dict[str, Resultado] = {}
    for nombre, encuesta in cargadas.items():
        if solo is not None and nombre not in solo:
            continue
        # `encuesta.hoja` es el nombre de encuesta tal como lo usa
        # el catálogo (p. ej. "Evaluación Docente"), la misma clave que debe
        # usarse en la columna ENCUESTA de taxonomia.xlsx.
        # Taxonomía por pregunta (hoja aspectos: clave lógica); si no, la clave antigua (ENCUESTA = hoja).
        clave_tax = clave_logica(nombre) if clave_logica(nombre) in taxonomias else encuesta.hoja
        if cfg.modo_aspectos != MODO_U4 and clave_tax not in taxonomias:
            raise ValueError(
                f"No hay taxonomía diseñada para '{clave_logica(nombre)}' en "
                "taxonomia.xlsx (hoja aspectos, columna pregunta_codigo). Agréguela antes de procesar "
                "esta encuesta -- ya no hay una taxonomía genérica de respaldo."
            )
        cfg_encuesta = replace(
            cfg,
            nombre=nombre,
            taxonomia=taxonomias.get(clave_tax, {}),
            **(config_por_encuesta or {}).get(clave_logica(nombre), {}),
        )
        unidades = construir_unidades(
            encuesta, cfg_encuesta, recursos, segmentador=segmentador
        )
        textos = (
            unidades[1]
            .loc[unidades[1]["motivo"] == MOTIVO_ASPECTO, "texto_aspecto"]
            .tolist()
        )
        grid = None
        modelo_u4: ModeloU4 | None = None
        if cfg_encuesta.modo_aspectos == MODO_U4:
            fuente = (modelos_u4 or {}).get(clave_logica(nombre))
            if isinstance(fuente, ModeloU4):
                modelo_u4 = fuente
            else:
                modelo_u4 = cargar_u4(fuente or carpeta_modelos_u4(raiz_modelos, clave_logica(nombre)))
        elif usar_grid and len(textos) >= MINIMO_MODELABLES:
            grid = GridSearchAspectos(
                cfg_encuesta, recursos, codificador, criterios
            ).preparar(textos)
            grid.modelador.clave_stopwords = encuesta.clave_stopwords
            grid.ejecutar(rejilla)
            try:
                cfg_encuesta = grid.mejor_config(permitir_no_cumple=permitir_no_cumple)
            except SinConfiguracionValida as exc:
                logger.error("'%s': %s", nombre, exc)
                if exportar:
                    carpeta = (
                        Path(cfg.output_dir) / slug(nombre) / "sin_configuracion_valida"
                    )
                    carpeta.mkdir(parents=True, exist_ok=True)
                    grid.resultados.to_csv(carpeta / "grid.csv", index=False)
                continue
        resultado = procesar_encuesta(
            encuesta,
            cfg_encuesta,
            recursos,
            codificador=codificador,
            clasificador=clasificador,
            unidades=unidades,
            grid=grid,
            modelo_u4=modelo_u4,
        )
        if exportar:
            exportar_resultado(resultado)
        resultados[nombre] = resultado
    return resultados


def resumen_comparativo(resultados: dict[str, Resultado]) -> pd.DataFrame:
    """Una fila por encuesta con métricas clave y nota."""
    filas = [
        {
            **{
                k: r.metricas.get(k)
                for k in (
                    "modo_aspectos",
                    "encoder",
                    "sentimiento_calibrado",
                    "filas",
                    "documentos_con_opinion",
                    "n_clausulas",
                    "clausulas_con_aspecto",
                    "pct_valoracion_global",
                    "n_topicos",
                    "cobertura_aspectual",
                    "cobertura_directa",
                    "pct_top1",
                    "silueta",
                    "ari_medio",
                    "pct_positivo",
                    "pct_neutro",
                    "pct_negativo",
                    *(
                        (
                            "modelo_u4_id",
                            "pct_comentario_general",
                            "pct_aspecto_lista",
                            "pct_por_descubrir",
                            "acierto_holdout",
                            "f1_macro_holdout",
                        )
                        if any(x.modo == MODO_U4 for x in resultados.values())
                        else ()
                    ),
                )
            },
            "spearman_doc": r.validacion.get("spearman_documento"),
            "nota": r.scorecard["nota"],
            "encuesta": nombre,
        }
        for nombre, r in resultados.items()
    ]
    return pd.DataFrame(filas).set_index("encuesta") if filas else pd.DataFrame()


if __name__ == "__main__":
    SALIDA = ejecutar_pipeline()
    print(resumen_comparativo(SALIDA).to_string())
