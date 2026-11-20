"""
Salidas de la Etapa 2 (NLP) según el contrato de datos v0.2 (sección 5.3), con los ajustes del plan v2.

Convierte lo que produce el pipeline (``Resultado``: tablas ``documentos`` y ``clausulas`` con columnas
internas como ``doc_id``, ``aspecto``, ``asignacion``, ``p_u4_*``) en cuatro CSV:

    clausulas.csv, auditoria_clasificacion.csv, catalogo_aspectos.csv, metadatos_nlp.csv

Las alertas ya NO son del NLP (Etapa 1 v1.10): ``alertas.csv`` y los conteos y huellas de alertas salen de estas
salidas, y las respuestas con ``tiene_alerta = True`` no llegan al pipeline.

Principios:

* Una sola función de construcción (:func:`construir_salidas_contrato`); no renombra columnas internas del modelo.
* Pandas puro: no importa torch, spaCy ni UMAP; se prueba con datos sintéticos.
* Las decisiones abiertas del equipo (spec §5) son parámetros (:class:`ParametrosSalidas`) con el valor de hoy.
* :func:`validar_salidas_nlp` implementa los criterios de aceptación (contrato 5.4 / spec §6) y se detiene con un
  mensaje claro. Solo informa conteos, nunca textos de clientes.

Formato de archivo (contrato 3.1): utf-8-sig, coma, punto decimal, ``True``/``False``, celda vacía para el dato
ausente (nunca «NA»).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("salidas_contrato")

VERSION_CONTRATO = "0.2"
VERSION_SALIDAS = "1.0.0"

SIN_ASPECTO = "SIN_ASPECTO"
OPINION_GENERAL = "OPINION_GENERAL"
GRUPO_DEFINIDOS = "Aspectos definidos"
GRUPO_EMERGENTES = "Temas emergentes"
GRUPO_SIN_GRUPO = "Sin grupo visual"   # plan v2, punto 4: SIN_ASPECTO (residual que HDBSCAN dejó sin grupo)
#: Formato de todo ``aspecto_id`` (mismo patrón que ``nlp_utilidades.taxonomia.PATRON_ID``).
PATRON_ID = re.compile(r"^[A-Z0-9]+(_[A-Z0-9]+)*$")
ORIGENES = ("clasificador", "hdbscan", "regla", "sistema", "manual")
SENTIMIENTOS = ("POS", "NEU", "NEG")

#: Nombre de cada archivo de salida.
ARCHIVOS = {
    "clausulas": "clausulas.csv",
    "auditoria": "auditoria_clasificacion.csv",
    "catalogo": "catalogo_aspectos.csv",
    "metadatos": "metadatos_nlp.csv",
}

COLUMNAS = {
    "clausulas": ["clausula_id", "respuesta_id", "orden", "texto_clausula", "es_opinion", "aspecto_id",
                  "aspecto_origen", "confianza_aspecto", "sentimiento", "confianza_sentimiento", "version_modelo"],
    "auditoria": ["clausula_id", "id_corrida_nlp", "version_modelo", "confianza_pos", "confianza_neu", "confianza_neg",
                  "aspecto_1_id", "confianza_aspecto_1", "aspecto_2_id", "confianza_aspecto_2", "aspecto_3_id",
                  "confianza_aspecto_3", "visto_en_entrenamiento"],
    "catalogo": ["aspecto_id", "aspecto_nombre", "aspecto_origen", "grupo_visual", "palabras_clave", "encuesta", "orden"],
    "metadatos": ["id_corrida_nlp", "id_corrida_preparacion", "encuesta", "pregunta_codigo", "fecha_corrida",
                  "version_modelo", "version_taxonomia", "umbral_respaldo_hdbscan", "accuracy_aspecto",
                  "metricas_sentimiento", "regla_ajuste_sentimiento", "n_respuestas", "n_clausulas",
                  "pct_aspecto_clasificador", "pct_aspecto_hdbscan", "pct_aspecto_regla", "pct_sin_aspecto"],
}

#: ``asignacion`` interna del pipeline -> (aspecto_id fijo o None si sale del clasificador/HDBSCAN, aspecto_origen).
#: «Comentario General» y «VALORACIÓN GLOBAL» van a OPINION_GENERAL (decisión Fase 4; existe en las 6 taxonomías).
#: Se puede cambiar sin tocar la lógica (parámetro ``mapeo_asignacion`` de :class:`ParametrosSalidas`).
MAPEO_ASIGNACION_DEFECTO: dict[str, tuple[str | None, str]] = {
    "u4": (None, "clasificador"),
    "residual_hdbscan": (None, "hdbscan"),
    "regla_comentario_general": (OPINION_GENERAL, "regla"),
    "valoracion_global": (OPINION_GENERAL, "regla"),
    # Punto 4: todo SIN_ASPECTO sale con origen «hdbscan» (el origen «sistema» ya no se emite). Estas dos filas (cláusulas
    # sin opinión) desaparecen de clausulas.csv con el Punto 5.
    "no_opinion": (SIN_ASPECTO, "hdbscan"),
    "ruido": (SIN_ASPECTO, "hdbscan"),
}
#: Asignaciones que son opinión (llevan sentimiento). El resto: es_opinion = False.
ASIGNACIONES_OPINION = frozenset({"u4", "residual_hdbscan", "regla_comentario_general", "valoracion_global"})


class ErrorSalidasNLP(ValueError):
    """Alguna de las validaciones de las salidas del NLP falló (el mensaje lista todas las fallas, con conteos)."""


# ═══════════════════════════════════════════════════════════════════════════
# Utilidades de texto
# ═══════════════════════════════════════════════════════════════════════════


def _plano(texto: str) -> tuple[str, list[int]]:
    """Minúsculas y sin tildes, conservando de dónde viene cada carácter (para devolver el fragmento ORIGINAL)."""
    salida: list[str] = []
    origen: list[int] = []
    for i, ch in enumerate(str(texto)):
        for c in unicodedata.normalize("NFD", ch.lower()):
            if unicodedata.category(c) != "Mn":
                salida.append(c)
                origen.append(i)
    return "".join(salida), origen


def _norm(texto: str) -> str:
    """Comparación sin tildes ni mayúsculas, con espacios colapsados (como el entrenamiento de U4)."""
    return " ".join(_plano(texto)[0].split())


# ═══════════════════════════════════════════════════════════════════════════
# Temas de HDBSCAN: aspecto_id = primer sustantivo de palabras_clave (plan v2, punto 4)
# ═══════════════════════════════════════════════════════════════════════════

#: Etiquetador de categorías gramaticales: recibe las palabras del grupo EN ORDEN (una frase de contexto) y devuelve
#: una etiqueta universal por palabra (``"NOUN"``, ``"PROPN"``, ``"ADJ"``...). Inyectable para probar sin spaCy.
FuncionPOS = Callable[[Sequence[str]], Sequence[str]]

_POS_SPACY: dict[str, FuncionPOS] = {}


def crear_pos_spacy(modelo: str = "es_core_news_lg") -> FuncionPOS:
    """Etiquetador con spaCy: etiqueta las palabras del grupo juntas, como una frase corta (en contexto).

    El POS de una palabra suelta es poco fiable («dinámicas» puede salir adjetivo); en contexto es más estable. Se
    construye un ``Doc`` con las mismas palabras (sin retokenizar) y se aplican el etiquetador y el analizador.

    Raises:
        ImportError: Si spaCy o el modelo no están instalados (en ese caso pase ``pos_fn`` a
            :func:`construir_salidas_contrato`).
    """
    if modelo in _POS_SPACY:
        return _POS_SPACY[modelo]
    try:
        import spacy  # pylint: disable=import-outside-toplevel
        from spacy.tokens import Doc  # pylint: disable=import-outside-toplevel

        nlp = spacy.load(modelo, exclude=["ner", "lemmatizer"])
    except (ImportError, OSError) as exc:
        raise ImportError(f"Se necesita spaCy con el modelo {modelo} para elegir el sustantivo de cada tema: {exc}") from exc

    def pos(palabras: Sequence[str]) -> list[str]:
        doc = Doc(nlp.vocab, words=list(palabras))
        for _, proc in nlp.pipeline:
            doc = proc(doc)
        return [t.pos_ for t in doc]

    _POS_SPACY[modelo] = pos
    return pos


def parsear_palabras_clave(valor: Any) -> list[str]:
    """Términos de ``palabras_clave`` (lista, o texto separado por «|» o «,»), en orden de relevancia y sin vacíos."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return []
    partes = valor if isinstance(valor, (list, tuple)) else re.split(r"[|,]", str(valor))
    return [str(t).strip() for t in partes if str(t).strip()]


def normalizar_id(texto: str) -> str:
    """Texto -> ``aspecto_id``: MAYÚSCULAS, sin tildes (Ñ -> N), lo no alfanumérico -> «_» y sin «_» repetidos ni en los bordes."""
    plano = "".join(c for c in unicodedata.normalize("NFD", str(texto).upper().replace("Ñ", "N")) if unicodedata.category(c) != "Mn")
    return re.sub(r"_+", "_", re.sub(r"[^A-Z0-9]", "_", plano)).strip("_")


def primer_sustantivo(terminos: Sequence[str], pos_fn: FuncionPOS) -> tuple[str | None, bool]:
    """Primer sustantivo (``NOUN``, nunca ``PROPN``) de ``terminos``, etiquetándolos juntos como frase de contexto.

    Los términos con varias palabras se parten en palabras: el sustantivo es la primera palabra con etiqueta ``NOUN``
    siguiendo el orden de relevancia de ``terminos``.

    Returns:
        ``(palabra, encontrado)``. Si ninguna palabra es sustantivo: ``(primera palabra, False)``; si no hay términos,
        ``(None, False)``.
    """
    palabras = [w for t in terminos for w in str(t).split()]
    if not palabras:
        return None, False
    etiquetas = list(pos_fn(palabras))
    if len(etiquetas) != len(palabras):
        raise ValueError(f"pos_fn devolvió {len(etiquetas)} etiquetas para {len(palabras)} palabras.")
    for palabra, etiqueta in zip(palabras, etiquetas):
        if etiqueta == "NOUN" and normalizar_id(palabra):
            return palabra, True
    for palabra in palabras:
        if normalizar_id(palabra):
            return palabra, False
    return None, False


def id_de_tema(
    sustantivo: str | None,
    *,
    fijos: set[str],
    usados_pregunta: set[str],
    n_respaldo: int,
) -> str:
    """``aspecto_id`` de un tema a partir de su sustantivo, resolviendo choques.

    * Choque con un aspecto fijo de la taxonomía (de cualquier pregunta de la encuesta): prefijo ``EMERGENTE_``.
    * Choque con otro tema de la MISMA pregunta: sufijo numérico (``_2``, ``_3``...).
    * Un tema de OTRA pregunta de la encuesta con el mismo sustantivo comparte el id (lo resuelve el llamador).
    * Sin sustantivo utilizable: ``TEMA_<n_respaldo>``.
    """
    base = normalizar_id(sustantivo or "") or f"TEMA_{n_respaldo}"
    candidato = f"EMERGENTE_{base}" if (base in fijos or base == SIN_ASPECTO) else base
    k, final = 2, candidato
    while final in usados_pregunta:
        final = f"{candidato}_{k}"
        k += 1
    return final


# ═══════════════════════════════════════════════════════════════════════════
# Parámetros y entradas
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ParametrosSalidas:
    """Decisiones abiertas (spec §5) como parámetros, con el valor de hoy.

    Solo se implementa el mapeo de reglas. ``regla_opinion_general = "contenido"`` cambia cómo se ASIGNA el aspecto
    dentro del pipeline (no en esta tabla): queda declarada y falla con un mensaje claro.
    """

    #: Confianza del clasificador por debajo de la cual se usa HDBSCAN (valor de hoy: ``Config.umbral_residual_u4``).
    umbral_respaldo_hdbscan: float | None = None
    #: "palabras" (hoy: respuesta de ≤ 5 palabras con baja confianza → Comentario General) | "contenido" (pendiente).
    regla_opinion_general: str = "palabras"
    #: Texto para ``metadatos_nlp.regla_ajuste_sentimiento`` (vacío si no hay regla).
    regla_ajuste_sentimiento: str = ""
    #: ``asignacion`` interna -> (aspecto_id fijo | None, aspecto_origen).
    mapeo_asignacion: Mapping[str, tuple[str | None, str]] = field(default_factory=lambda: dict(MAPEO_ASIGNACION_DEFECTO))

    def validar(self) -> None:
        if self.regla_opinion_general != "palabras":
            raise NotImplementedError("regla_opinion_general = 'contenido' aún no está implementada: se aplica dentro de etiquetar_corpus_u4.")
        malos = {k: v for k, v in self.mapeo_asignacion.items() if v[1] not in ORIGENES}
        if malos:
            raise ValueError(f"mapeo_asignacion con aspecto_origen fuera de {ORIGENES}: {malos}")


@dataclass
class EntradaPregunta:
    """Lo que produce el pipeline para UNA pregunta (en modo u4)."""

    pregunta_codigo: str
    documentos: pd.DataFrame      # doc_id, respuesta_id, respuesta_completa
    clausulas: pd.DataFrame       # doc_id, clausula_id (interno), clausula, aspecto, asignacion, p_aspecto_u4, p_u4_*, ...
    clases: Sequence[str]         # clases del clasificador, en el orden de ``columnas_p``
    columnas_p: Sequence[str]     # columnas p_u4_* de ``clausulas`` (una por clase)
    modelo_u4_id: str
    modelo_sentimiento: str
    accuracy_aspecto: float | None = None
    metricas_sentimiento: Mapping[str, Any] | None = None
    topicos_residuales: pd.DataFrame | None = None   # topico, etiqueta, terminos


@dataclass
class SalidasContrato:
    """Las cuatro tablas del contrato más los datos de la corrida."""

    clausulas: pd.DataFrame
    auditoria: pd.DataFrame
    catalogo: pd.DataFrame
    metadatos: pd.DataFrame
    id_corrida_nlp: str
    fecha_corrida: str
    avisos: list[str] = field(default_factory=list)

    def tablas(self) -> dict[str, pd.DataFrame]:
        return {k: getattr(self, k) for k in ARCHIVOS}


# ═══════════════════════════════════════════════════════════════════════════
# Construcción (una sola función)
# ═══════════════════════════════════════════════════════════════════════════


def _partes_id(respuesta_id: str) -> tuple[str, str, str]:
    """(encuesta, periodo, pregunta) de ``encuesta|periodo|id_origen|pregunta`` (el «|» está reservado)."""
    p = respuesta_id.split("|")
    return p[0], p[1], p[-1]


def version_modelo(modelo_u4_id: str, modelo_sentimiento: str) -> str:
    """Versión del clasificador de aspectos y del modelo de sentimiento que produjeron la cláusula."""
    return f"{modelo_u4_id}|{modelo_sentimiento}"


def describir_regla_ajuste_sentimiento(margen_neu: float | None, calibrador: bool, f1_macro_cv: float | None = None) -> str:
    """Texto de ``regla_ajuste_sentimiento`` para el ajuste que hace hoy ``ClasificadorPolaridad.clasificar``."""
    if calibrador:
        extra = f" (F1 macro CV {f1_macro_cv})" if f1_macro_cv is not None else ""
        return f"calibrador: la etiqueta final sale del calibrador entrenado con anotaciones humanas{extra}; neu_ajustado marca las que difieren del modelo"
    return (f"margen_neu={margen_neu}: si el modelo dice NEU y |p_pos - p_neg| >= margen_neu, la etiqueta pasa a POS "
            "o NEG según el signo; neu_ajustado marca las cláusulas ajustadas")


def _top3(fila: np.ndarray, ids: Sequence[str]) -> list[tuple[str, float]]:
    orden = sorted(range(len(fila)), key=lambda i: (-fila[i], i))[:3]
    return [(ids[i], float(fila[i])) for i in orden]


def construir_salidas_contrato(  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    entradas: Sequence[EntradaPregunta],
    *,
    taxonomia: pd.DataFrame,
    id_corrida_preparacion: str,
    version_taxonomia: str,
    parametros: ParametrosSalidas | None = None,
    fecha: datetime | None = None,
    pos_fn: FuncionPOS | None = None,
) -> SalidasContrato:
    """Construye las cuatro tablas del contrato v0.2 a partir de las preguntas procesadas.

    Args:
        entradas: Una por pregunta procesada (modo u4).
        taxonomia: Hoja ``aspectos`` de ``taxonomia.xlsx`` (pregunta_codigo, aspecto_id, aspecto_nombre, descripcion).
        id_corrida_preparacion: Corrida de la Etapa 1 que se procesó.
        version_taxonomia: Huella de ``taxonomia.xlsx`` (``nlp_utilidades.taxonomia.version_taxonomia``).
        parametros: Decisiones abiertas.
        fecha: Fecha UTC de la corrida (se fija en las pruebas para comparar dos corridas).
        pos_fn: Etiquetador de categorías gramaticales para elegir el sustantivo de cada tema de HDBSCAN (por defecto,
            :func:`crear_pos_spacy`, solo si hay temas). Inyectable para pruebas.

    Raises:
        ValueError, KeyError, NotImplementedError: datos inconsistentes o alternativa no implementada.
    """
    p = parametros or ParametrosSalidas()
    p.validar()
    if not entradas:
        raise ValueError("No hay preguntas procesadas.")
    fecha = fecha or datetime.now(timezone.utc)
    fecha_txt = fecha.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entradas = sorted(entradas, key=lambda e: e.pregunta_codigo)

    # ── id de la corrida: fecha UTC + huella de insumos, taxonomía y versión del modelo ──
    ids_resp = sorted(i for e in entradas for i in e.documentos["respuesta_id"])
    huella = hashlib.sha256(json.dumps({
        "preparacion": id_corrida_preparacion, "taxonomia": version_taxonomia,
        "modelos": sorted({version_modelo(e.modelo_u4_id, e.modelo_sentimiento) for e in entradas}),
        "respuestas": hashlib.sha256("\n".join(ids_resp).encode("utf-8")).hexdigest(),
        "parametros": {k: v for k, v in sorted(vars(p).items()) if k != "mapeo_asignacion"},
        "mapeo": {k: list(v) for k, v in sorted(p.mapeo_asignacion.items())},
    }, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:10]
    id_corrida_nlp = f"{fecha.astimezone(timezone.utc):%Y%m%d-%H%M%S}_{huella}"

    filas_cla: list[pd.DataFrame] = []
    filas_aud: list[pd.DataFrame] = []
    filas_meta: list[dict[str, Any]] = []
    cat_def: dict[tuple[str, str], dict[str, Any]] = {}
    cat_emg: dict[tuple[str, str], dict[str, Any]] = {}   # (encuesta, aspecto_id) -> fila; el primero gana
    # aspecto_id fijos de TODAS las preguntas procesadas de cada encuesta (un tema con ese nombre lleva prefijo EMERGENTE_)
    fijos_por_encuesta: dict[str, set[str]] = {}
    for e0 in entradas:
        enc0 = _partes_id(str(e0.documentos["respuesta_id"].iloc[0]))[0]
        fijos_por_encuesta.setdefault(enc0, set()).update(taxonomia.loc[taxonomia["pregunta_codigo"] == e0.pregunta_codigo, "aspecto_id"])
    n_sin_sustantivo = 0
    avisos: list[str] = []

    for e in entradas:
        pregunta = e.pregunta_codigo
        tax = taxonomia[taxonomia["pregunta_codigo"] == pregunta].reset_index(drop=True)
        if tax.empty:
            raise KeyError(f"{pregunta}: no hay aspectos en la taxonomía.")
        id_por_nombre = {_norm(n): i for n, i in zip(tax["aspecto_nombre"], tax["aspecto_id"])}
        ids_tax = set(tax["aspecto_id"])
        for asig, (fijo, _) in p.mapeo_asignacion.items():
            if fijo not in (None, SIN_ASPECTO) and fijo not in ids_tax:
                raise KeyError(f"{pregunta}: el mapeo de '{asig}' apunta a {fijo}, que no está en la taxonomía de la pregunta.")
        sin_id = sorted(c for c in e.clases if _norm(c) not in id_por_nombre)
        if sin_id:
            raise KeyError(f"{pregunta}: clases del clasificador sin aspecto_id en la taxonomía: {sin_id}")
        ids_clases = [id_por_nombre[_norm(c)] for c in e.clases]
        vm = version_modelo(e.modelo_u4_id, e.modelo_sentimiento)

        # ── cláusulas: respuesta_id por doc_id, orden desde 1 sin saltos ──
        doc = e.documentos[["doc_id", "respuesta_id"]]
        if doc["doc_id"].duplicated().any():
            raise ValueError(f"{pregunta}: documentos.doc_id no es único.")
        c = e.clausulas.merge(doc, on="doc_id", how="left", validate="many_to_one")
        if c["respuesta_id"].isna().any():
            raise ValueError(f"{pregunta}: {int(c['respuesta_id'].isna().sum())} cláusulas sin respuesta_id.")
        c = c.sort_values(["doc_id", "clausula_id"], kind="mergesort").reset_index(drop=True)
        c["orden"] = c.groupby("doc_id").cumcount() + 1
        desconocidas = sorted(set(c["asignacion"].dropna()) - set(p.mapeo_asignacion))
        if c["asignacion"].isna().any() or desconocidas:
            raise ValueError(f"{pregunta}: asignaciones sin mapeo al contrato: {desconocidas or 'nulas'} (¿modo bertopic?).")

        # ── aspecto_id y aspecto_origen ──
        encuesta = _partes_id(c["respuesta_id"].iloc[0])[0] if len(c) else ""
        grupos = sorted(int(g) for g in c.loc[(c["asignacion"] == "residual_hdbscan") & c["topico_residual"].notna(), "topico_residual"].unique() if int(g) >= 0) \
            if "topico_residual" in c.columns else []
        id_emergente: dict[int, str] = {}
        usados_pregunta: set[str] = set()
        sin_sust_pregunta = 0
        for g in grupos:
            claves_l: list[str] = []
            if e.topicos_residuales is not None and len(e.topicos_residuales):
                t = e.topicos_residuales
                fila = t[t["topico"] == g]
                if len(fila):
                    claves_l = parsear_palabras_clave(fila["terminos"].iloc[0])
            if pos_fn is None:
                pos_fn = crear_pos_spacy()
            sust, hallado = primer_sustantivo(claves_l, pos_fn)
            if not hallado:
                sin_sust_pregunta += 1
            id_g = id_de_tema(sust, fijos=fijos_por_encuesta[encuesta], usados_pregunta=usados_pregunta, n_respaldo=g)
            usados_pregunta.add(id_g)
            id_emergente[g] = id_g
            if (encuesta, id_g) not in cat_emg:   # mismo id en otra pregunta de la encuesta = mismo aspecto (una sola fila)
                cat_emg[(encuesta, id_g)] = {"aspecto_id": id_g, "aspecto_nombre": str(sust or id_g).lower(),
                                             "aspecto_origen": "hdbscan", "grupo_visual": GRUPO_EMERGENTES,
                                             "palabras_clave": ", ".join(claves_l), "encuesta": encuesta}
        if sin_sust_pregunta:
            n_sin_sustantivo += sin_sust_pregunta
            avisos.append(f"{pregunta}: {sin_sust_pregunta} temas sin ningún sustantivo en palabras_clave; su aspecto_id sale del primer término.")

        aspecto_id: list[str] = []
        origen: list[str] = []
        for asig, nombre, grupo in zip(c["asignacion"], c["aspecto"], c.get("topico_residual", pd.Series([pd.NA] * len(c)))):
            fijo, org = p.mapeo_asignacion[asig]
            if asig == "residual_hdbscan":
                if pd.notna(grupo) and int(grupo) in id_emergente:
                    aspecto_id.append(id_emergente[int(grupo)]); origen.append(org)
                else:  # residual que HDBSCAN dejó sin grupo
                    aspecto_id.append(SIN_ASPECTO); origen.append("hdbscan")
            elif fijo is not None:
                aspecto_id.append(fijo); origen.append(org)
            else:
                if _norm(str(nombre)) not in id_por_nombre:
                    raise KeyError(f"{pregunta}: el aspecto asignado no está en la taxonomía ({int(len(str(nombre)))} caracteres).")
                aspecto_id.append(id_por_nombre[_norm(str(nombre))]); origen.append(org)
        c["aspecto_id"] = aspecto_id
        c["aspecto_origen"] = origen
        es_op = c["asignacion"].isin(ASIGNACIONES_OPINION)

        # ── confianza del aspecto: clasificador = p de la clase ganadora; hdbscan = pertenencia; regla/sistema = vacío ──
        conf_asp = np.full(len(c), np.nan)
        m_clas = (c["aspecto_origen"] == "clasificador").to_numpy()
        conf_asp[m_clas] = pd.to_numeric(c.loc[m_clas, "p_aspecto_u4"], errors="coerce").to_numpy()
        if "p_residual" in c.columns:
            m_h = (c["aspecto_origen"] == "hdbscan").to_numpy()
            conf_asp[m_h] = pd.to_numeric(c.loc[m_h, "p_residual"], errors="coerce").to_numpy()
        elif (c["aspecto_origen"] == "hdbscan").any():
            avisos.append(f"{pregunta}: no hay columna p_residual; confianza_aspecto queda vacía en las cláusulas de HDBSCAN.")

        # ── sentimiento: vacío si no es opinión ──
        sent = c["sentimiento"].where(es_op, "").fillna("").astype(str)
        malos = sorted(set(sent[es_op]) - set(SENTIMIENTOS))
        if malos:
            raise ValueError(f"{pregunta}: sentimiento fuera de {SENTIMIENTOS}: {malos}")
        conf_sent = np.where(es_op, pd.to_numeric(c["confianza"], errors="coerce"), np.nan)

        cid = c["respuesta_id"] + "|" + c["orden"].astype(str)
        filas_cla.append(pd.DataFrame({
            "clausula_id": cid, "respuesta_id": c["respuesta_id"], "orden": c["orden"].astype("Int64"),
            "texto_clausula": c["clausula"].astype(str),
            "es_opinion": np.where(es_op, "True", "False"),
            "aspecto_id": c["aspecto_id"], "aspecto_origen": c["aspecto_origen"],
            "confianza_aspecto": np.round(conf_asp, 6), "sentimiento": sent,
            "confianza_sentimiento": np.round(conf_sent.astype(float), 6), "version_modelo": vm,
        }))

        # ── auditoría: una fila por cláusula ──
        falta = [col for col in e.columnas_p if col not in c.columns]
        if falta:
            raise KeyError(f"{pregunta}: faltan columnas de probabilidad del clasificador: {falta[:3]}...")
        pm = c[list(e.columnas_p)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        a1 = [("", np.nan)] * 3
        tops = [_top3(fila, ids_clases) if not np.isnan(fila).all() else a1 for fila in pm]
        aud = pd.DataFrame({"clausula_id": cid, "id_corrida_nlp": id_corrida_nlp, "version_modelo": vm})
        for col, src in (("confianza_pos", "p_pos"), ("confianza_neu", "p_neu"), ("confianza_neg", "p_neg")):
            aud[col] = np.round(np.where(es_op, pd.to_numeric(c[src], errors="coerce"), np.nan).astype(float), 6)
        for k in range(3):
            aud[f"aspecto_{k + 1}_id"] = [t[k][0] if len(t) > k else "" for t in tops]
            aud[f"confianza_aspecto_{k + 1}"] = np.round([t[k][1] if len(t) > k else np.nan for t in tops], 6)
        aud["visto_en_entrenamiento"] = np.where(
            c.get("u4_visto_en_entrenamiento", pd.Series(False, index=c.index)).fillna(False).astype(bool), "True", "False")
        filas_aud.append(aud)

        # ── catálogo (lista fija de la pregunta) ──
        for orden_tax, (aid, nom) in enumerate(zip(tax["aspecto_id"], tax["aspecto_nombre"]), start=1):
            clave = (aid, encuesta)
            if clave not in cat_def or orden_tax < cat_def[clave]["orden"]:
                cat_def[clave] = {"aspecto_id": aid, "aspecto_nombre": nom, "aspecto_origen": "clasificador",
                                  "grupo_visual": GRUPO_DEFINIDOS, "palabras_clave": "", "encuesta": encuesta, "orden": orden_tax}

        # ── metadatos de la pregunta ──
        n_cl = len(c)
        es_sin = (c["aspecto_id"] == SIN_ASPECTO)
        # sin solapes (plan v2, 4.3): hdbscan = solo temas emergentes; sin_aspecto = SIN_ASPECTO (también origen hdbscan)
        pct_f = lambda m: round(100.0 * float(m.sum()) / n_cl, 2) if n_cl else float("nan")  # noqa: E731
        pct = lambda org: pct_f((c["aspecto_origen"] == org) & ~es_sin)  # noqa: E731
        ms = json.dumps(e.metricas_sentimiento, ensure_ascii=False, sort_keys=True, default=float) if e.metricas_sentimiento else ""
        filas_meta.append({
            "id_corrida_nlp": id_corrida_nlp, "id_corrida_preparacion": id_corrida_preparacion, "encuesta": encuesta,
            "pregunta_codigo": pregunta, "fecha_corrida": fecha_txt, "version_modelo": vm, "version_taxonomia": version_taxonomia,
            "umbral_respaldo_hdbscan": p.umbral_respaldo_hdbscan, "accuracy_aspecto": e.accuracy_aspecto,
            "metricas_sentimiento": ms, "regla_ajuste_sentimiento": p.regla_ajuste_sentimiento,
            "n_respuestas": int(e.documentos["respuesta_id"].nunique()), "n_clausulas": n_cl,
            "pct_aspecto_clasificador": pct("clasificador"), "pct_aspecto_hdbscan": pct("hdbscan"),
            "pct_aspecto_regla": pct("regla"), "pct_sin_aspecto": pct_f(es_sin),
        })

    # ── catálogo: temas emergentes al final de cada encuesta y SIN_ASPECTO de último ──
    filas_cat = list(cat_def.values())
    for enc in sorted({f["encuesta"] for f in filas_cat}):
        base = max(f["orden"] for f in filas_cat if f["encuesta"] == enc)
        emg = [f for (en, _), f in cat_emg.items() if en == enc]
        for k, f in enumerate(emg, start=1):
            filas_cat.append({**f, "orden": base + k})
        filas_cat.append({"aspecto_id": SIN_ASPECTO, "aspecto_nombre": "Sin aspecto", "aspecto_origen": "hdbscan",
                          "grupo_visual": GRUPO_SIN_GRUPO, "palabras_clave": "", "encuesta": enc, "orden": base + len(emg) + 1})

    if n_sin_sustantivo:
        logger.warning("%d temas de HDBSCAN sin sustantivo en palabras_clave: se usó el primer término como aspecto_id.", n_sin_sustantivo)
    cla = pd.concat(filas_cla, ignore_index=True)
    aud = pd.concat(filas_aud, ignore_index=True)
    cat = pd.DataFrame(filas_cat, columns=COLUMNAS["catalogo"]).sort_values(["encuesta", "orden", "aspecto_id"], kind="mergesort").reset_index(drop=True)
    cat["orden"] = cat["orden"].astype("Int64")
    meta = pd.DataFrame(filas_meta, columns=COLUMNAS["metadatos"])
    for col in ("n_respuestas", "n_clausulas"):
        meta[col] = meta[col].astype("Int64")
    return SalidasContrato(cla[COLUMNAS["clausulas"]], aud[COLUMNAS["auditoria"]], cat, meta, id_corrida_nlp, fecha_txt, avisos)


def entrada_desde_resultado(
    resultado: Any,
    *,
    clases: Sequence[str],
    columnas_p: Sequence[str],
    modelo_u4_id: str,
    modelo_sentimiento: str,
    metricas_sentimiento: Mapping[str, Any] | None = None,
) -> EntradaPregunta:
    """Adapta un ``Resultado`` del pipeline (modo u4) a :class:`EntradaPregunta`.

    ``clases`` y ``columnas_p`` salen del modelo (``modelo.clases`` y ``nombres_columnas_p(clases)``).
    """
    metricas = {**getattr(resultado, "u4", {}), **getattr(resultado, "metricas", {})}
    acc = metricas.get("acierto_holdout")
    return EntradaPregunta(
        pregunta_codigo=str(resultado.encuesta).split(" :: ")[-1],
        documentos=resultado.documentos,
        clausulas=resultado.clausulas,
        clases=list(clases),
        columnas_p=list(columnas_p),
        modelo_u4_id=modelo_u4_id,
        modelo_sentimiento=modelo_sentimiento,
        accuracy_aspecto=None if acc is None or (isinstance(acc, float) and np.isnan(acc)) else float(acc),
        metricas_sentimiento=metricas_sentimiento,
        topicos_residuales=getattr(resultado, "topicos_residuales", None),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Escritura y relectura
# ═══════════════════════════════════════════════════════════════════════════


def escribir_salidas_contrato(salidas: SalidasContrato, carpeta: Path | str) -> dict[str, Path]:
    """Escribe los 4 CSV (utf-8-sig, coma, punto decimal, vacío para ausente)."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = {}
    for nombre, df in salidas.tablas().items():
        ruta = carpeta / ARCHIVOS[nombre]
        df.to_csv(ruta, index=False, encoding="utf-8-sig", sep=",", decimal=".", na_rep="", lineterminator="\n")
        rutas[nombre] = ruta
    return rutas


def leer_salidas_contrato(carpeta: Path | str) -> dict[str, pd.DataFrame]:
    """Relee los 4 CSV como texto (``dtype=str``, ``utf-8-sig``, sin convertir «NA»)."""
    carpeta = Path(carpeta)
    return {n: pd.read_csv(carpeta / f, dtype=str, keep_default_na=False, encoding="utf-8-sig") for n, f in ARCHIVOS.items()}


def huella_archivos(carpeta: Path | str) -> str:
    """Hash de los 4 archivos juntos (para comprobar que dos corridas dan lo mismo)."""
    h = hashlib.sha256()
    for f in ARCHIVOS.values():
        h.update(Path(carpeta, f).read_bytes())
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# Archivos exportados sin datos de respuestas excluidas (plan v2, punto 3)
# ═══════════════════════════════════════════════════════════════════════════


def ids_respuestas_excluidas(respuestas: pd.DataFrame, preguntas: Sequence[str] | None = None) -> set[str]:
    """``respuesta_id`` de las respuestas que NO entran al NLP: no-respuestas y respuestas con alerta.

    Args:
        respuestas: ``respuestas.csv`` de la Etapa 1 (como texto), con ``es_no_respuesta`` y ``tiene_alerta``.
        preguntas: Si se da, solo estas preguntas.
    """
    r = respuestas if preguntas is None else respuestas[respuestas["pregunta_codigo"].isin(list(preguntas))]
    return set(r.loc[(r["es_no_respuesta"] == "True") | (r["tiene_alerta"] == "True"), "respuesta_id"])


def verificar_archivos_sin_excluidas(
    carpeta: Path | str,
    respuestas: pd.DataFrame,
    *,
    preguntas: Sequence[str] | None = None,
    incluir_subcarpetas: bool = True,
) -> pd.DataFrame:
    """Comprueba que ningún CSV/XLSX de ``carpeta`` trae el ``respuesta_id`` de una respuesta excluida.

    Revisa toda columna ``respuesta_id`` y toda columna ``clausula_id``/``alerta_id`` cuyos valores tengan la forma
    ``respuesta_id|n`` (contrato); en los XLSX revisa todas las hojas. Los archivos internos que solo traen ``doc_id``
    (p. ej. ``clausulas.csv`` de modelado) no se pueden revisar por id: salen con ``columnas_id = 0`` y su contenido se
    cubre porque nacen del mismo corpus filtrado que ``documentos.csv`` (que sí trae ``respuesta_id``).

    Args:
        carpeta: Carpeta de una corrida (modelado o contrato).
        respuestas: ``respuestas.csv`` de la Etapa 1 (como texto).
        preguntas: Limita las excluidas a estas preguntas (por defecto, todas).

    Returns:
        Una fila por archivo y hoja: ``archivo``, ``hoja``, ``columnas_id``, ``filas_revisadas`` y ``excluidas``.

    Raises:
        ErrorSalidasNLP: Si algún archivo contiene respuestas excluidas (solo conteos, nunca textos).
    """
    carpeta = Path(carpeta)
    excluidas = ids_respuestas_excluidas(respuestas, preguntas)
    archivos = sorted(f for f in (carpeta.rglob("*") if incluir_subcarpetas else carpeta.glob("*"))
                      if f.is_file() and f.suffix.lower() in (".csv", ".xlsx"))
    filas, fallas = [], []
    for f in archivos:
        if f.suffix.lower() == ".csv":
            hojas = {"": pd.read_csv(f, dtype=str, keep_default_na=False, encoding="utf-8-sig")}
        else:
            hojas = pd.read_excel(f, sheet_name=None, dtype=str, keep_default_na=False)
        for hoja, df in hojas.items():
            n_cols, revisadas, malas = 0, 0, 0
            for col in ("respuesta_id", "clausula_id", "alerta_id"):
                if col not in df.columns:
                    continue
                valores = df[col]
                if col != "respuesta_id":
                    valores = valores[valores.str.contains("|", regex=False)].str.rsplit("|", n=1).str[0]
                    if valores.empty:
                        continue
                n_cols += 1
                revisadas += int(len(valores))
                malas += int(valores.isin(excluidas).sum())
            filas.append({"archivo": f.relative_to(carpeta).as_posix(), "hoja": hoja, "columnas_id": n_cols,
                          "filas_revisadas": revisadas, "excluidas": malas})
            if malas:
                fallas.append(f"{f.name}{'/' + hoja if hoja else ''}: {malas} filas de respuestas excluidas (no-respuesta o con alerta)")
    if fallas:
        detalle = "\n  - ".join(fallas)
        raise ErrorSalidasNLP(f"Hay datos de respuestas excluidas en archivos exportados:\n  - {detalle}")
    return pd.DataFrame(filas, columns=["archivo", "hoja", "columnas_id", "filas_revisadas", "excluidas"])


# ═══════════════════════════════════════════════════════════════════════════
# Validaciones (contrato 5.4 / spec §6)
# ═══════════════════════════════════════════════════════════════════════════


def validar_salidas_nlp(  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    tablas: Mapping[str, pd.DataFrame],
    respuestas: pd.DataFrame,
    *,
    id_corrida_preparacion: str,
    preguntas_procesadas: Sequence[str],
    clases_por_pregunta: Mapping[str, Sequence[str]] | None = None,
    taxonomia: pd.DataFrame | None = None,
) -> None:
    """Comprueba las salidas RELEÍDAS como texto contra ``respuestas.csv`` de la Etapa 1.

    Args:
        tablas: ``clausulas``, ``auditoria``, ``catalogo`` y ``metadatos`` como texto
            (:func:`leer_salidas_contrato`).
        respuestas: ``respuestas.csv`` completo de la corrida procesada (como texto; con ``es_no_respuesta`` y
            ``tiene_alerta``).
        id_corrida_preparacion: Corrida que se procesó.
        preguntas_procesadas: Preguntas incluidas en la corrida.
        clases_por_pregunta, taxonomia: Si se dan, se exige que la taxonomía coincida con las clases del modelo.

    Raises:
        ErrorSalidasNLP: con todas las fallas encontradas (solo conteos y nombres de columnas).
    """
    c, a, cat, m = (tablas[k] for k in ("clausulas", "auditoria", "catalogo", "metadatos"))
    fallas: list[str] = []

    def falla(cond: bool, msg: str) -> None:
        if cond:
            fallas.append(msg)

    for nombre, df in tablas.items():
        faltan = [col for col in COLUMNAS[nombre] if col not in df.columns]
        falla(bool(faltan), f"{nombre}: faltan columnas {faltan}")
    faltan_r = [col for col in ("respuesta_id", "pregunta_codigo", "es_no_respuesta", "tiene_alerta") if col not in respuestas.columns]
    falla(bool(faltan_r), f"respuestas: faltan columnas {faltan_r}")
    if fallas:
        raise ErrorSalidasNLP("\n".join(fallas))

    pregs = set(preguntas_procesadas)
    resp = respuestas[respuestas["pregunta_codigo"].isin(pregs)]
    # entrada del NLP (plan v2, punto 1): es_no_respuesta = False y tiene_alerta = False
    procesables = set(resp.loc[(resp["es_no_respuesta"] == "False") & (resp["tiene_alerta"] == "False"), "respuesta_id"])
    no_resp = set(resp.loc[resp["es_no_respuesta"] == "True", "respuesta_id"])
    con_alerta = set(resp.loc[(resp["es_no_respuesta"] == "False") & (resp["tiene_alerta"] == "True"), "respuesta_id"])
    con_clausula = set(c["respuesta_id"])
    # cobertura
    falla(bool(procesables - con_clausula), f"{len(procesables - con_clausula)} respuestas procesables sin ninguna cláusula")
    falla(bool(no_resp & con_clausula), f"{len(no_resp & con_clausula)} no-respuestas tienen cláusulas")
    falla(bool(con_alerta & con_clausula), f"{len(con_alerta & con_clausula)} respuestas con alerta tienen cláusulas (las alertas no pasan al NLP)")
    # unicidad y orden
    falla(bool(c["clausula_id"].duplicated().any()), f"clausula_id repetido en {int(c['clausula_id'].duplicated().sum())} filas")
    orden = pd.to_numeric(c["orden"], errors="coerce")
    falla(bool(orden.isna().any()), f"orden no numérico en {int(orden.isna().sum())} filas")
    g = c.assign(_o=orden).groupby("respuesta_id")["_o"]
    malos = int(((g.min() != 1) | (g.max() != g.count())).sum())
    falla(malos > 0, f"orden no empieza en 1 o tiene saltos en {malos} respuestas")
    falla(bool((c["clausula_id"] != c["respuesta_id"] + "|" + c["orden"]).any()), "clausula_id no es respuesta_id|orden")
    # integridad referencial
    todas = set(respuestas["respuesta_id"])
    falla(bool(set(c["respuesta_id"]) - todas), f"{len(set(c['respuesta_id']) - todas)} respuesta_id de clausulas no existen en respuestas.csv")
    ids_cat = set(cat["aspecto_id"])
    falla(bool(set(c["aspecto_id"]) - ids_cat), f"{len(set(c['aspecto_id']) - ids_cat)} aspecto_id de clausulas no están en catalogo_aspectos")
    aud_ids = set(a[["aspecto_1_id", "aspecto_2_id", "aspecto_3_id"]].to_numpy().ravel()) - {""}
    falla(bool(aud_ids - ids_cat), f"{len(aud_ids - ids_cat)} aspecto_id de la auditoría no están en catalogo_aspectos")
    falla(cat.duplicated(["aspecto_id", "encuesta"]).any(), "catalogo_aspectos repite (aspecto_id, encuesta)")
    # taxonomía vs clases del modelo
    if clases_por_pregunta is not None and taxonomia is not None:
        from nlp_utilidades.taxonomia import comparar_con_clases  # pylint: disable=import-outside-toplevel

        for preg, clases in clases_por_pregunta.items():
            dif = comparar_con_clases(taxonomia, preg, list(clases))
            falla(bool(dif["solo_en_taxonomia"] or dif["solo_en_modelo"]),
                  f"{preg}: la taxonomía no coincide con las clases del modelo ({len(dif['solo_en_taxonomia'])} solo en taxonomía, {len(dif['solo_en_modelo'])} solo en modelo)")
    # dominios
    falla(bool(set(c["es_opinion"]) - {"True", "False"}), "es_opinion fuera de True/False")
    falla(bool(set(c["sentimiento"]) - {*SENTIMIENTOS, ""}), "sentimiento fuera de POS/NEU/NEG/vacío")
    falla(bool(((c["sentimiento"] == "") != (c["es_opinion"] == "False")).any()), "sentimiento vacío no coincide con es_opinion = False")
    falla(bool(set(c["aspecto_origen"]) - set(ORIGENES)), f"aspecto_origen fuera de {ORIGENES}")
    # plan v2, punto 4: SIN_ASPECTO = residual sin grupo de HDBSCAN; los temas llevan id derivado del sustantivo
    sin_c = c["aspecto_id"] == SIN_ASPECTO
    falla(bool((c.loc[sin_c, "aspecto_origen"] != "hdbscan").any()), f"clausulas: {int((c.loc[sin_c, 'aspecto_origen'] != 'hdbscan').sum())} SIN_ASPECTO con aspecto_origen distinto de hdbscan")
    falla(bool((~c["aspecto_id"].map(lambda x: bool(PATRON_ID.match(x)))).any()), "clausulas: aspecto_id con formato inválido (MAYÚSCULAS, dígitos y «_»)")
    falla(bool((~cat["aspecto_id"].map(lambda x: bool(PATRON_ID.match(x)))).any()), "catalogo_aspectos: aspecto_id con formato inválido")
    falla(bool(cat["aspecto_id"].str.match(r"^EMERGENTE_\d{4}-\d+_\d+$").any()), "catalogo_aspectos: quedan ids EMERGENTE_<periodo>_<n> (deben derivar del sustantivo)")
    falla(bool(set(cat["grupo_visual"]) - {GRUPO_DEFINIDOS, GRUPO_EMERGENTES, GRUPO_SIN_GRUPO}), "grupo_visual fuera del dominio")
    sin_k = cat[cat["aspecto_id"] == SIN_ASPECTO]
    falla(bool(((sin_k["aspecto_origen"] != "hdbscan") | (sin_k["grupo_visual"] != GRUPO_SIN_GRUPO)).any()), "catalogo_aspectos: SIN_ASPECTO debe ser hdbscan / Sin grupo visual")
    emg_k = cat[cat["grupo_visual"] == GRUPO_EMERGENTES]
    falla(bool((emg_k["aspecto_origen"] != "hdbscan").any()), "catalogo_aspectos: los temas emergentes deben tener aspecto_origen hdbscan")
    falla(bool((cat.loc[cat["grupo_visual"] == GRUPO_DEFINIDOS, "aspecto_origen"] == "hdbscan").any()), "catalogo_aspectos: un aspecto definido con aspecto_origen hdbscan")
    falla(bool(set(a["visto_en_entrenamiento"]) - {"True", "False"}), "visto_en_entrenamiento fuera de True/False")
    # confianzas
    def numeros(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
        return df[list(cols)].apply(lambda s: pd.to_numeric(s.replace("", np.nan), errors="coerce"))

    for nombre, df, cols in (("clausulas", c, ["confianza_aspecto", "confianza_sentimiento"]),
                              ("auditoria", a, ["confianza_pos", "confianza_neu", "confianza_neg", "confianza_aspecto_1", "confianza_aspecto_2", "confianza_aspecto_3"])):
        num = numeros(df, cols)
        invalidas = sum(int(((num[k] < 0) | (num[k] > 1)).sum()) for k in cols)
        no_num = sum(int((df[k] != "") .sum() - num[k].notna().sum()) for k in cols)
        falla(invalidas > 0, f"{nombre}: {invalidas} confianzas fuera de [0, 1]")
        falla(no_num > 0, f"{nombre}: {no_num} confianzas no numéricas")
    falla(bool(((c["confianza_aspecto"] == "") & c["aspecto_origen"].isin(["clasificador"])).any()), "hay cláusulas del clasificador sin confianza_aspecto")
    falla(bool(((c["sentimiento"] != "") & (c["confianza_sentimiento"] == "")).any()), "hay sentimiento sin confianza_sentimiento")
    # suma de las tres confianzas de sentimiento
    sent_ids = set(c.loc[c["sentimiento"] != "", "clausula_id"])
    ap = a[a["clausula_id"].isin(sent_ids)]
    suma = numeros(ap, ["confianza_pos", "confianza_neu", "confianza_neg"]).sum(axis=1, min_count=3)
    falla(bool(((suma - 1).abs() > 0.001).any() or suma.isna().any()), f"auditoria: {int((((suma - 1).abs() > 0.001) | suma.isna()).sum())} cláusulas con confianza_pos+neu+neg distinta de 1 (±0,001)")
    # auditoría: una fila por cláusula y aspectos ordenados
    falla(bool(set(a["clausula_id"]) != set(c["clausula_id"]) or len(a) != len(c)), "auditoria no tiene exactamente una fila por cláusula de clausulas")
    conf = numeros(a, ["confianza_aspecto_1", "confianza_aspecto_2", "confianza_aspecto_3"])
    desorden = int(((conf["confianza_aspecto_1"] < conf["confianza_aspecto_2"]) | (conf["confianza_aspecto_2"] < conf["confianza_aspecto_3"])).sum())
    falla(desorden > 0, f"auditoria: {desorden} filas con aspectos no ordenados de mayor a menor confianza")
    # metadatos
    falla(set(m["id_corrida_preparacion"]) != {id_corrida_preparacion}, "metadatos_nlp no registra el id_corrida_preparacion procesado")
    falla(bool(set(m["pregunta_codigo"]) != pregs), "metadatos_nlp no tiene una fila por pregunta procesada")
    falla(len(set(m["id_corrida_nlp"])) != 1 or set(a["id_corrida_nlp"]) != set(m["id_corrida_nlp"]), "id_corrida_nlp distinto entre auditoria y metadatos")
    por_preg_cl = c.assign(p=c["respuesta_id"].str.split("|").str[-1]).groupby("p")
    for _, f in m.iterrows():
        pr = f["pregunta_codigo"]
        esp_cl = int(len(por_preg_cl.get_group(pr))) if pr in por_preg_cl.groups else 0
        esp_re = int(por_preg_cl.get_group(pr)["respuesta_id"].nunique()) if pr in por_preg_cl.groups else 0
        falla(f["n_clausulas"] != str(esp_cl) or f["n_respuestas"] != str(esp_re),
              f"metadatos_nlp ({pr}): los conteos no coinciden con los archivos")
    # los cuatro porcentajes de origen no se solapan: suman 100
    pcts = numeros(m, ["pct_aspecto_clasificador", "pct_aspecto_hdbscan", "pct_aspecto_regla", "pct_sin_aspecto"]).sum(axis=1, min_count=1)
    falla(bool(((pcts - 100).abs() > 0.05).any()), "metadatos_nlp: pct_aspecto_* y pct_sin_aspecto no suman 100")
    # periodo como texto
    falla(bool(c["respuesta_id"].str.split("|").str[1].isin(["", "nan", "NaN"]).any()), "periodo vacío en respuesta_id")
    # «NA» solo es válido como texto de un comentario real; en cualquier otra columna delata un dato ausente mal escrito
    sin_texto = c.drop(columns=["texto_clausula"])
    falla(bool(sin_texto.isin(["NA", "nan", "None", "null"]).to_numpy().any()), "hay valores «NA»/«nan»/«None» en clausulas")
    if fallas:
        raise ErrorSalidasNLP(f"{len(fallas)} validaciones del contrato fallaron:\n  - " + "\n  - ".join(fallas))
