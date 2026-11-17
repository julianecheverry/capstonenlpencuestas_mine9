"""
Informe de inconsistencias de la Etapa 1 (para el cliente).

Revisa las bases crudas tal como llegan del cliente y escribe un Excel con una
hoja de resumen y una hoja de detalle por inconsistencia:

  I01–I14  evaluación y autoevaluación docente (EvaDoc Pregrado)
  I15–I17  autoevaluación de acreditación
  I18–I21  calidad de servicios

Lo llama `ejecutar_etapa1` al final de cada corrida y lo guarda en
`reportes/etapa 1 preparación/Informe_inconsistencias_corrida_<id_corrida>.xlsx`.
La carpeta `reportes/` está fuera de git porque el informe trae ID de respuesta
y otros datos del cliente.

Reglas que se respetan siempre:
- Ningún comentario sale en el informe: un texto real se muestra como [TEXTO]
  (con su largo); solo se muestran tal cual los valores de las listas de
  no-respuesta y de valores cerrados («NR», «Ninguna», «De acuerdo»…), que son
  justamente lo que el cliente tiene que revisar.
- Las reglas que antes se escribían a mano (la «receta» del Código Cruce, la
  regla del Código Conteo, el Id Estudiante de relleno) se calculan en cada
  corrida, para que el informe siga siendo correcto con periodos nuevos.
- Si una revisión falla (p. ej. cambió el nombre de una columna), el resumen lo
  dice en su fila y las demás revisiones siguen.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from preparacion.configuracion import asignar_preguntas, preguntas_activas, resolver_columna
from preparacion.normalizacion import normalizar

RAIZ = Path(__file__).resolve().parents[2]
CARPETA_INFORME = RAIZ / "reportes" / "etapa 1 preparación"
OCULTO = "[TEXTO]"

# Encuestas de la hoja `etapa1` que cubre cada bloque de revisiones.
EVADOC = ("evadoc_evaluacion", "evadoc_autoevaluacion")
ACREDITACION = "autoevaluacion_acreditacion"
CALIDAD = "calidad_servicios"

# Columnas de la base de evaluación docente (nombres del cliente).
PERIODO, ID = "Periodo", "ID de respuesta"
CRUCE, CONTEO = "Código Cruce Autoevaluación", "Código Conteo"
DOCENTE, ESTUDIANTE = "Id Docente", "Id Estudiante"
MARCA_TIEMPO = "Marca de tiempo (mm/dd/yyyy)"
BASE = [PERIODO, ID, "Codigo del Grupo", "Grupo", "Codigo Materia", "Materia", "Facultad de oferta"]
# Columnas con las que se prueba la «receta» del Código Cruce y del Código Conteo.
CANDIDATAS_RECETA = ["Codigo del Grupo", "Grupo", "Materia", "Codigo Materia"]

NOMBRES = {
    DOCENTE: "Id Docente (codificado)",
    ESTUDIANTE: "Id Estudiante (codificado)",
    CRUCE: "Código Cruce (codificado)",
    CONTEO: "Código Conteo (codificado)",
    MARCA_TIEMPO: "Marca de tiempo",
}
NOMBRES_CORTOS = {"Codigo del Grupo": "Código del Grupo", "Codigo Materia": "Código Materia"}

VALORES_RELLENO = {"sin informacion", "n a", "na", "no aplica", "no registra", "otro", "otros"}


@dataclass
class Hoja:
    """Una inconsistencia: su fila del resumen y su hoja de detalle."""
    codigo: str
    titulo: str
    observado: str
    impacto: str
    tratamiento: str
    detalle: pd.DataFrame | None = None
    periodos: str = "—"
    error: str = ""
    notas: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- utilidades

def _vacio(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip() == ""


def _si(s: pd.Series) -> pd.Series:
    return s.map(normalizar) == "si"


def _grupo(d: pd.DataFrame, llave: list[str]) -> pd.Series:
    return d.groupby(llave, sort=True).ngroup() + 1


def _periodos(d: pd.DataFrame | None, col: str = PERIODO) -> str:
    if d is None or col not in d or d.empty:
        return "—"
    valores = {v.strip() for x in d[col].astype(str) for v in x.split(",") if v.strip()}
    return ", ".join(sorted(valores))


def _sino(cond: pd.Series) -> pd.Series:
    return cond.map({True: "Sí", False: "No"})


class Textos:
    """Oculta los comentarios: muestra solo valores de no-respuesta o cerrados."""

    def __init__(self, no_respuesta: frozenset[str], valores_cerrados: frozenset[str]):
        self.no_respuesta = no_respuesta
        self.cerrados = valores_cerrados - no_respuesta
        self.visibles = no_respuesta | valores_cerrados

    def es_comentario(self, s: pd.Series) -> pd.Series:
        return ~s.map(normalizar).isin(self.no_respuesta)

    def ocultar(self, s: pd.Series) -> pd.Series:
        s = s.fillna("").astype(str)
        return s.where(s.map(normalizar).isin(self.visibles) | (s.str.strip() == ""), OCULTO)

    def largo(self, s: pd.Series) -> pd.Series:
        """Largo del comentario; 0 si es no-respuesta (como en el informe original)."""
        s = s.fillna("").astype(str)
        return s.str.strip().str.len().where(self.es_comentario(s), 0)


def _columna(df: pd.DataFrame, prefijo: str) -> str:
    for c in df.columns:
        if str(c).startswith(prefijo):
            return c
    raise KeyError(f"No se encontró la columna que empieza por «{prefijo}».")


def _receta(d: pd.DataFrame, codigo: str, exacta: bool) -> tuple[list[str] | None, pd.Series]:
    """Combinación de columnas (siempre con Id Docente) que mejor reproduce un código.

    Devuelve la receta y la máscara de filas que no la cumplen. Con `exacta`,
    solo acepta una receta sin ninguna fila fuera (relación uno a uno).
    """
    mejor, mejor_mascara = None, None
    for n in range(1, len(CANDIDATAS_RECETA) + 1):
        for combo in itertools.combinations(CANDIDATAS_RECETA, n):
            receta = list(combo) + [DOCENTE]
            a, b = _choques(d, receta, codigo)
            mascara = a | b
            if mejor_mascara is None or mascara.sum() < mejor_mascara.sum():
                mejor, mejor_mascara = receta, mascara
    if exacta and mejor_mascara is not None and mejor_mascara.any():
        return None, mejor_mascara
    return mejor, mejor_mascara


def _choques(d: pd.DataFrame, receta: list[str], codigo: str) -> tuple[pd.Series, pd.Series]:
    """(combinación con más de un código, código con más de una combinación), por fila."""
    clave = d[receta[0]].astype(str)
    for c in receta[1:]:
        clave = clave + "|" + d[c].astype(str)
    t = pd.DataFrame({"clave": clave, "codigo": d[codigo]}, index=d.index)
    a = t.groupby("clave")["codigo"].transform("nunique") > 1
    b = t.groupby("codigo")["clave"].transform("nunique") > 1
    return a, b


def _receta_texto(receta: list[str] | None) -> str:
    return " + ".join(NOMBRES_CORTOS.get(c, c) for c in receta) if receta else "—"


# --------------------------------------------------------------------------- evaluación docente

class RevisionEvaDoc:
    def __init__(self, df: pd.DataFrame, textos: Textos):
        self.df = df.copy()
        self.t = textos
        c = list(df.columns)
        self.p19, self.p20 = _columna(df, "Observaciones - 19"), _columna(df, "Observaciones - 20")
        self.p23, self.p24, self.p25 = (_columna(df, f"Observaciones - {n}") for n in (23, 24, 25))
        self.items = c[c.index("Facultad Estudiante") + 1:c.index(self.p19)]
        self.etiquetas = {self.p19: "P19", self.p20: "P20", self.p23: "P23", self.p24: "P24", self.p25: "P25"}

    def _comentarios(self, d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        out = d.copy()
        for col in cols:
            et = self.etiquetas[col]
            out[f"{et} (comentario oculto)"] = self.t.ocultar(d[col])
            out[f"Largo {et} (caracteres)"] = self.t.largo(d[col])
        return out

    def hojas(self) -> list[Callable[[], Hoja]]:
        return [self.i01, self.i02, self.i03, self.i04, self.i05, self.i06, self.i07,
                self.i08, self.i09, self.i10, self.i11, self.i12, self.i13, self.i14]

    def _copias(self) -> pd.DataFrame:
        k = [PERIODO, ID, DOCENTE]
        d = self.df[self.df.duplicated(k, keep=False)].copy()
        d["Grupo de copias"] = _grupo(d, k)
        return d

    def i01(self) -> Hoja:
        d = self._copias()
        g = d.groupby([PERIODO, ID, DOCENTE])
        d["¿Facultad de oferta distinta entre copias?"] = _sino(g["Facultad de oferta"].transform("nunique") > 1)
        d["¿Id Estudiante igual en todas las copias?"] = _sino(g[ESTUDIANTE].transform("nunique") == 1)
        cambia = (d["¿Id Estudiante igual en todas las copias?"] == "No").groupby(d["Grupo de copias"]).any().sum()
        det = d.sort_values(["Grupo de copias", "Facultad de oferta"])[
            ["Grupo de copias"] + BASE + [ESTUDIANTE, DOCENTE, CRUCE,
                                          "¿Facultad de oferta distinta entre copias?",
                                          "¿Id Estudiante igual en todas las copias?"]]
        return Hoja("I01", "Respuestas copiadas por facultad de oferta",
                    f"La misma respuesta (mismo periodo, ID de respuesta y docente) aparece en varias filas, "
                    f"una por cada facultad que ofrece el curso: {d['Grupo de copias'].nunique():,} respuestas en "
                    f"{len(d):,} filas. En {cambia:,} de ellas el Id Estudiante cambia entre copias.",
                    "Si no se deduplica, el comentario se cuenta una vez por facultad y el Id Estudiante no sirve "
                    "para identificar la respuesta.",
                    "Se conserva una sola respuesta y se registra su relación con cada facultad de oferta (tabla "
                    "respuestas_oferta). No se usa Id Estudiante como llave.",
                    det, _periodos(d))

    def i02(self) -> Hoja:
        d = self._copias()
        g = d.groupby("Grupo de copias")
        distinto = (g[self.p19].transform("nunique") > 1) | (g[self.p20].transform("nunique") > 1)
        e = self._comentarios(d[distinto], [self.p19, self.p20])
        cols = ["Grupo de copias"] + BASE + ["P19 (comentario oculto)", "Largo P19 (caracteres)",
                                            "P20 (comentario oculto)", "Largo P20 (caracteres)"]
        return Hoja("I02", "Pregunta abierta registrada distinto entre copias de la misma respuesta",
                    f"En {e['Grupo de copias'].nunique():,} respuestas copiadas por facultad, P19 o P20 cambia entre "
                    "copias: «NR» en una y «Ninguna», «Nada» o vacío en otra; en algunos casos una copia trae un "
                    "texto y la otra no.",
                    "Según la copia que se tome, el comentario existe o no.",
                    "Se elige la copia que tiene comentario y, entre esas, la más larga.",
                    e.sort_values(["Grupo de copias", "Facultad de oferta"])[cols], _periodos(e))

    def i03(self) -> Hoja:
        k = [PERIODO, ID]
        m = self.df[self.df.groupby(k)[DOCENTE].transform("nunique") > 1].copy()
        items_nr = m[self.items].apply(pd.to_numeric, errors="coerce").isna().all(axis=1)
        real = _si(m["Evaluación"])
        m["Tipo de fila"] = "Evaluación = No con datos"
        m.loc[real, "Tipo de fila"] = "Evaluación real"
        m.loc[~real & items_nr & _vacio(m[MARCA_TIEMPO]), "Tipo de fila"] = \
            "Registro vacío (Evaluación = No, sin marca de tiempo, ítems en NR)"
        m["Grupo de ID"] = _grupo(m, k)
        m["¿Tiene marca de tiempo?"] = _sino(~_vacio(m[MARCA_TIEMPO]))
        n_grupos = m["Grupo de ID"].nunique()
        una_real = (real.groupby(m["Grupo de ID"]).sum() == 1).sum()
        det = m.sort_values(["Grupo de ID", "Tipo de fila"])[
            ["Grupo de ID", "Tipo de fila"] + BASE + [DOCENTE, "Evaluación", "¿Tiene marca de tiempo?"]]
        return Hoja("I03", "ID de respuesta reutilizado con varios docentes",
                    f"En {n_grupos:,} casos un mismo ID de respuesta aparece con varios docentes ({len(m):,} filas). "
                    f"En {una_real:,} de ellos hay una sola evaluación real y las demás filas son registros vacíos "
                    "(Evaluación = No, sin marca de tiempo, ítems en «NR»); la columna «Tipo de fila» clasifica cada una.",
                    "Ninguno sobre los comentarios si las filas sobrantes son registros vacíos: ningún comentario "
                    "queda atribuido al docente equivocado.",
                    "Se incluyen solo las filas con Evaluación = Sí (filtro de respondientes efectivos). Se informa "
                    "para que el cliente revise por qué esos registros reutilizan el ID de respuesta.",
                    det, _periodos(m))

    def i04(self) -> Hoja:
        por_id = self.df.groupby(ESTUDIANTE).size()
        umbral = max(100, 3 * por_id.quantile(0.999))
        relleno = sorted(por_id[por_id > umbral].index)
        d = self.df[self.df[ESTUDIANTE].isin(relleno)]
        det = d[BASE + [ESTUDIANTE, DOCENTE]]
        return Hoja("I04", "Id Estudiante de relleno",
                    f"{len(relleno)} valor(es) de Id Estudiante aparecen en {len(d):,} filas de cursos y respuestas "
                    f"distintos (un estudiante típico aparece en {int(por_id.median())} filas; se marca todo valor "
                    f"con más de {int(umbral)}). Además, el Id Estudiante cambia entre copias de una misma respuesta "
                    "(ver I01).",
                    "El Id Estudiante no es confiable para identificar estudiantes ni respuestas.",
                    "No se usa Id Estudiante en ninguna llave. Se solicita al cliente confirmar qué valor es el de "
                    "relleno.",
                    det, _periodos(d))

    def i05(self) -> Hoja:
        partes, recetas = [], {}
        for p, gp in self.df[~_vacio(self.df[CRUCE])].groupby(PERIODO):
            receta, fuera = _receta(gp, CRUCE, exacta=False)
            recetas[p] = _receta_texto(receta)
            x = gp[fuera].copy()
            x["Receta del periodo"] = recetas[p]
            varios_cruces, _ = _choques(gp, receta, CRUCE)
            x["Problema"] = varios_cruces[fuera].map(
                {True: "La misma combinación tiene más de un Código Cruce",
                 False: "Un Código Cruce agrupa combinaciones distintas"})
            partes.append(x)
        d = pd.concat(partes) if partes else self.df.iloc[0:0]
        resumen = "; ".join(f"{p}: {r}" for p, r in recetas.items())
        det = d.sort_values([PERIODO, CRUCE])[BASE + [DOCENTE, CRUCE, "Receta del periodo", "Problema"]]
        return Hoja("I05", "El Código Cruce Autoevaluación no sigue una regla única",
                    f"La combinación de columnas que mejor reproduce el Código Cruce cambia por periodo ({resumen}). "
                    f"{len(d):,} filas no cumplen ni la regla de su propio periodo.",
                    "Una regla propia para identificar la autoevaluación daría resultados distintos a los del cliente.",
                    "Se usa el Código Cruce del cliente tal como viene. Se solicita confirmar si el cambio de regla "
                    "fue intencional.",
                    det, _periodos(d))

    def i06(self) -> Hoja:
        filas = []
        for p, gp in self.df[~_vacio(self.df[CRUCE]) & ~_vacio(self.df[CONTEO])].groupby(PERIODO):
            por_cruce = gp.groupby(CRUCE)[CONTEO].nunique()
            por_conteo = gp.groupby(CONTEO)[CRUCE].nunique()
            if (por_cruce == 1).all() and (por_conteo == 1).all():
                regla = "Igual al Código Cruce"
            elif (por_cruce == 1).all():
                regla = f"Más agregado que el Código Cruce ({(por_conteo > 1).sum():,} conteos agrupan varios cruces)"
            else:
                receta, _ = _receta(gp, CONTEO, exacta=True)
                regla = (f"{_receta_texto(receta)} (exacta)" if receta
                         else "Ninguna combinación de columnas lo reproduce")
            filas.append({"Periodo": p, "Códigos Cruce distintos": gp[CRUCE].nunique(),
                          "Códigos Conteo distintos": gp[CONTEO].nunique(),
                          "Regla encontrada para el Código Conteo": regla})
        det = pd.DataFrame(filas)
        return Hoja("I06", "El Código Conteo no sigue una regla estable",
                    "La relación entre Código Conteo y Código Cruce cambia por periodo (ver detalle).",
                    "Posibles diferencias entre nuestros conteos de autoevaluaciones y los del tablero del cliente.",
                    "No se usa como llave. Se solicita al cliente explicar su uso.",
                    det, _periodos(det))

    def i07(self) -> Hoja:
        k = [PERIODO, CRUCE]
        d = self.df[~_vacio(self.df[CRUCE])].copy()
        cols = [self.p23, self.p24, self.p25]
        for c in cols:
            d[f"_l{self.etiquetas[c]}"] = self.t.largo(d[c])
        largos = [f"_l{self.etiquetas[c]}" for c in cols]
        g = d.groupby(k)
        mascara = pd.concat([g[l].transform("nunique") > 1 for l in largos], axis=1).any(axis=1)
        y = d[mascara].copy()
        y["Grupo de cruce"] = _grupo(y, k)
        y = y.drop_duplicates(k + largos)
        y = self._comentarios(y, cols)
        det = y.sort_values("Grupo de cruce")[
            ["Grupo de cruce"] + BASE + [CRUCE] +
            [x for c in cols for x in (f"{self.etiquetas[c]} (comentario oculto)",
                                       f"Largo {self.etiquetas[c]} (caracteres)")]]
        return Hoja("I07", "Autoevaluación con respuestas abiertas distintas dentro del mismo Código Cruce",
                    f"En {y['Grupo de cruce'].nunique():,} Códigos Cruce las filas no traen el mismo texto (o la "
                    "misma no-respuesta) en P23, P24 o P25. El detalle muestra una fila por cada variante.",
                    "Una misma autoevaluación tendría dos textos posibles.",
                    "Se elige la copia con comentario y, entre esas, la más larga. Se solicita revisión al cliente.",
                    det, _periodos(y))

    def i08(self) -> Hoja:
        ev = self.df[self.items].apply(pd.to_numeric, errors="coerce")
        malo = ev.isna().any(axis=1)
        z = self.df[malo].copy()
        z["Ítems con NR"] = ev[malo].isna().sum(axis=1)
        z["Promedio informado por el cliente"] = pd.to_numeric(z["Promedio Evaluación - General"], errors="coerce")
        z["Promedio de ítems 2-18 sin NR"] = ev[malo][self.items[1:]].mean(axis=1).round(4)
        z["Situación"] = z["Promedio informado por el cliente"].isna().map(
            {True: "Promedio vacío", False: "Promedio presente y no coincide con la regla"})
        presentes = (z["Situación"] != "Promedio vacío").sum()
        det = z.sort_values("Situación", ascending=False)[
            BASE + ["Ítems con NR", "Promedio informado por el cliente", "Promedio de ítems 2-18 sin NR", "Situación"]]
        return Hoja("I08", "Promedio general de evaluación cuando hay ítems «NR»",
                    "Cuando algún ítem es «NR», el cliente deja vacío el Promedio Evaluación - General. "
                    f"En {presentes:,} filas el promedio está presente pero no coincide con la regla (media de ítems 2 a 18).",
                    "Informativo: confirma la regla del cliente y señala las excepciones.",
                    "Se replica la regla del cliente: media de ítems 2 a 18; sin promedio si hay ítems «NR».",
                    det, _periodos(z))

    def i09(self) -> Hoja:
        c = self.t.es_comentario
        s = self.df.drop_duplicates([PERIODO, ID, DOCENTE])
        s = s[~_vacio(s["Observaciones eval"])]
        w = s[_si(s["Observaciones eval"]) & ~c(s[self.p19]) & ~c(s[self.p20])].copy()
        w["Encuesta"], w["Marca del cliente"] = "Evaluación (estudiante)", w["Observaciones eval"]
        a = self.df[~_vacio(self.df["Observaciones Auto"])].drop_duplicates([PERIODO, CRUCE])
        v = a[_si(a["Observaciones Auto"]) & ~c(a[self.p23]) & ~c(a[self.p24]) & ~c(a[self.p25])].copy()
        v["Encuesta"], v["Marca del cliente"] = "Autoevaluación (docente)", v["Observaciones Auto"]
        cols = [self.p19, self.p20, self.p23, self.p24, self.p25]
        # Cada encuesta muestra solo sus preguntas: en una fila de autoevaluación,
        # P19 y P20 son de un estudiante cualquiera y no vienen al caso.
        for parte, propias in ((w, [self.p19, self.p20]), (v, [self.p23, self.p24, self.p25])):
            for col in cols:
                parte[f"{self.etiquetas[col]} (comentario oculto)"] = (
                    self.t.ocultar(parte[col]) if col in propias else "")
        d = pd.concat([w, v])
        sin_marca = sorted(set(self.df[PERIODO]) - set(self.df.loc[
            ~_vacio(self.df["Observaciones eval"]) | ~_vacio(self.df["Observaciones Auto"]), PERIODO]))
        nota = (f" En {', '.join(sin_marca)} esas columnas están vacías en todas las filas, así que no permiten "
                "validar esos periodos." if sin_marca else "")
        det = d[["Encuesta"] + BASE + ["Marca del cliente"] +
                [f"{self.etiquetas[c]} (comentario oculto)" for c in cols]]
        return Hoja("I09", "Definición de «comentó» distinta a la del proyecto",
                    "Las columnas Observaciones eval y Observaciones Auto marcan «Si» respuestas que solo dicen "
                    f"«Ninguna», «Nada» o similares ({len(w):,} en evaluación, {len(v):,} en autoevaluación).{nota}",
                    "La tasa de participación del proyecto sale menor que la que resultaría de esas columnas.",
                    "Se mantiene la definición del proyecto (lista de no-respuesta de la configuración). Las "
                    "columnas Observaciones eval/Auto no se usan.",
                    det, _periodos(d))

    def i10(self) -> Hoja:
        campos = [("Codigo Materia", "Código Materia vacío"), ("Materia", "Materia vacía"),
                  (MARCA_TIEMPO, "Marca de tiempo vacía"),
                  ("Tiempo necesario para completar (segundos)", "Tiempo de diligenciamiento vacío"),
                  (self.p19, "P19 vacía (sin NR)"), (self.p20, "P20 vacía (sin NR)"),
                  (self.p23, "P23 vacía (sin NR)"), (self.p24, "P24 vacía (sin NR)"),
                  (self.p25, "P25 vacía (sin NR)"), ("N° Inscritos", "N° Inscritos vacío")]
        partes = []
        for col, etiqueta in campos:
            vacio = (self.df[col].astype(str) == "") if col == MARCA_TIEMPO else _vacio(self.df[col])
            q = self.df[vacio].copy()
            q["Campo con problema"] = etiqueta
            partes.append(q)
        d = pd.concat(partes)
        conteo = d["Campo con problema"].value_counts()
        return Hoja("I10", "Campos vacíos",
                    "Filas con campos vacíos: " + "; ".join(f"{k} ({v:,})" for k, v in conteo.items()) + ".",
                    "Menor; puede impedir asignar el curso o calcular tasas de respuesta.",
                    "Las preguntas abiertas vacías se tratan como no-respuesta; el resto se reporta sin cambios.",
                    d[["Campo con problema"] + BASE], _periodos(d))

    def i11(self) -> Hoja:
        filas = [{"Columna": c, "Valor": v, "Filas": n}
                 for c in ("Evaluación", "Autoevaluación") for v, n in self.df[c].value_counts().items()]
        det = pd.DataFrame(filas)
        variantes = det.assign(n=det["Valor"].map(normalizar)).groupby(["Columna", "n"])["Valor"].nunique()
        return Hoja("I11", "Valores con mayúsculas inconsistentes en Evaluación y Autoevaluación",
                    f"Las columnas Evaluación y Autoevaluación escriben el mismo valor de varias formas "
                    f"({(variantes > 1).sum()} valores con más de una escritura, p. ej. «Si»/«SI»).",
                    "Menor; un filtro sensible a mayúsculas separaría los valores.",
                    "Se normalizan sin distinguir mayúsculas.", det)

    def i12(self) -> Hoja:
        k = [PERIODO, CRUCE]
        cols = [self.p23, self.p24, self.p25]
        d = self.df[~_vacio(self.df[CRUCE])]
        un_caracter = pd.concat([d[c].astype(str).str.strip().str.len() == 1 for c in cols], axis=1).any(axis=1)
        y = d[un_caracter].drop_duplicates(k).copy()
        ceros = pd.concat([y[c].astype(str).str.strip() == "0" for c in cols], axis=1).any(axis=1)
        por_periodo = y[ceros][PERIODO].value_counts().sort_index()
        for c in cols:
            y[f"{self.etiquetas[c]}: largo del texto"] = y[c].astype(str).str.strip().str.len()
        det = y.rename(columns={ID: "ID de respuesta (una fila de ejemplo del cruce)"})[
            [PERIODO, "ID de respuesta (una fila de ejemplo del cruce)"] + BASE[2:] + [CRUCE] +
            [f"{self.etiquetas[c]}: largo del texto" for c in cols]]
        detalle_ceros = ", ".join(f"{p}: {n:,}" for p, n in por_periodo.items()) or "ninguno"
        return Hoja("I12", "No-respuesta del docente codificada como «0»",
                    f"{len(y):,} autoevaluaciones tienen una respuesta de un solo carácter en P23, P24 o P25. "
                    f"En {ceros.sum():,} ese carácter es «0» ({detalle_ceros}): la no-respuesta del docente se "
                    "registró como «0» en lugar de «NR». En la columna de largo, 1 identifica la respuesta afectada.",
                    "Si no se reconoce como no-respuesta, se contaría como un comentario y aumentaría "
                    "artificialmente la participación de los docentes.",
                    "El «0» está en la lista de no-respuesta del proyecto. Se solicita confirmar si es intencional.",
                    det, _periodos(y))

    def i13(self) -> Hoja:
        k = [PERIODO, CRUCE]
        d = self.df[~_vacio(self.df[CRUCE])].copy()
        auto = d["Autoevaluación"].map(normalizar)
        llave = [d[PERIODO], d[CRUCE]]
        mezcla = ((auto == "si").groupby(llave).transform("any")
                  & (auto == "no").groupby(llave).transform("any"))
        y = d[mezcla].copy()
        y["Grupo de cruce"] = _grupo(y, k)
        det = y.sort_values(["Grupo de cruce", "Autoevaluación"]).rename(
            columns={ID: "ID de respuesta (fila del estudiante)"})[
            ["Grupo de cruce", PERIODO, "ID de respuesta (fila del estudiante)"] + BASE[2:] + [CRUCE, "Autoevaluación"]]
        return Hoja("I13", "Autoevaluación marcada «Sí» y «No» dentro del mismo Código Cruce",
                    f"En {y['Grupo de cruce'].nunique():,} autoevaluaciones (mismo periodo y Código Cruce), unas filas "
                    "tienen Autoevaluación = Sí y otras Autoevaluación = No.",
                    "Según la fila que se tome, el docente aparecería como si hubiera respondido o no.",
                    "Se considera respondida si alguna fila dice Sí. Se solicita al cliente revisar la marca.",
                    det, _periodos(y))

    def i14(self) -> Hoja:
        ts = self.df[MARCA_TIEMPO].astype(str)
        es_fecha = ts.str.match(r"^\s*\d{1,4}[-/]\d{1,2}[-/]\d{1,4}")
        # Un espacio en blanco cuenta aquí (no es fecha) y no en I10 (no está vacío).
        d = self.df[(ts != "") & ~es_fecha].copy()
        d["Marca de tiempo (largo del valor)"] = d[MARCA_TIEMPO].astype(str).str.len()
        return Hoja("I14", "Marca de tiempo con un valor que no es fecha",
                    f"En {len(d):,} filas la columna Marca de tiempo trae un valor que no es una fecha y hora.",
                    "Menor; no afecta los comentarios.", "Se reporta sin cambios.",
                    d[BASE + ["Evaluación", "Marca de tiempo (largo del valor)"]], _periodos(d))


# --------------------------------------------------------------------------- acreditación

def _variantes(s: pd.Series) -> pd.Series:
    """True en las filas cuyo valor es una escritura minoritaria de otro (mayúsculas, tildes)."""
    n = s.map(normalizar)
    conteo = s.groupby([n, s]).transform("size")
    maximo = conteo.groupby(n).transform("max")
    escrituras = s.groupby(n).transform("nunique")
    return (escrituras > 1) & (conteo < maximo)


class RevisionAcreditacion:
    def __init__(self, df: pd.DataFrame, textos: Textos, columna_abierta: str):
        self.df, self.t, self.abierta = df, textos, columna_abierta
        self.anterior = df.columns[list(df.columns).index(columna_abierta) - 1]

    def hojas(self) -> list[Callable[[], Hoja]]:
        return [self.i15, self.i16, self.i17]

    def i15(self) -> Hoja:
        d = self.df[_variantes(self.df["Población"])]
        escrituras = ", ".join(f"«{v}» ({n:,})" for v, n in d["Población"].value_counts().items())
        return Hoja("I15", "Población escrita de varias formas en acreditación",
                    f"La columna Población trae escrituras distintas del mismo valor: {escrituras or 'ninguna'}. "
                    "El filtro de Excel no distingue mayúsculas y las muestra juntas; Power BI las trataría como "
                    "categorías distintas.",
                    "Un tablero filtrado por población separaría a un mismo grupo en dos.",
                    "Se unifican con la hoja roles de la configuración.",
                    d[[PERIODO, ID, "llave", "Facultad", "Programa", "Población"]], _periodos(d))

    def i16(self) -> Hoja:
        d = self.df
        envio = [PERIODO, ID]
        cerrado = d[self.abierta].map(normalizar).isin(self.t.cerrados)
        comentario = self.t.es_comentario(d[self.abierta]) & ~cerrado
        x = d[cerrado].copy()
        valor = f"Valor en la pregunta abierta ({self.abierta.strip()})"
        igual = f"¿Igual a la columna anterior ({self.anterior.strip()})?"
        x[valor] = x[self.abierta]
        x[igual] = _sino(x[self.abierta].map(normalizar) == x[self.anterior].map(normalizar))
        x["Copias del envío"] = d.groupby(envio)[ID].transform("size")[cerrado]
        x["Copias del envío con valor cerrado"] = cerrado.groupby([d[PERIODO], d[ID]]).transform("sum")[cerrado]
        x["¿Otra copia del envío trae un comentario?"] = _sino(
            comentario.groupby([d[PERIODO], d[ID]]).transform("any")[cerrado])
        envios = x.groupby(envio).ngroups
        iguales = (x[igual] == "Sí").sum()
        valores = ", ".join(f"«{v}» ({n:,})" for v, n in x[self.abierta].value_counts().items())
        cols = [PERIODO, ID, "Población", "llave", valor, igual, "Copias del envío",
                "Copias del envío con valor cerrado", "¿Otra copia del envío trae un comentario?"]
        return Hoja("I16", "Respuesta de escala cerrada en la columna de la pregunta abierta (acreditación)",
                    f"En {len(x):,} filas de {envios:,} envíos, la columna de la pregunta abierta trae un valor de "
                    f"escala cerrada en lugar de un comentario: {valores or 'ninguno'}. En {iguales:,} de esas filas "
                    "el valor es igual al de la columna anterior. Dentro de un mismo envío unas copias traen el valor "
                    "cerrado y otras no, lo que apunta a un error al consolidar la base.",
                    "Sin tratamiento, esos envíos aparecerían con un comentario que no existe, y el modelo de "
                    "sentimiento leería «De acuerdo» como una opinión positiva.",
                    "Si el texto completo es un valor de escala cerrada, se trata como no-respuesta y la respuesta se "
                    "marca (valor_cerrado_en_pregunta_abierta). Si otra copia del envío trae un comentario real, se "
                    "conserva ese comentario. Se solicita al cliente revisar el proceso de consolidación.",
                    x[cols], _periodos(x))

    def i17(self, propias: tuple[str, ...] = ("SNIES", "Facultad", "Departamento", "Programa", "Ciudad/Énfasis",
                                              "Nivel de Formación", "Nivel Académico", "Tamaño de muestra",
                                              "llave", "Año", "Población")) -> Hoja:
        d = self.df
        envio = [PERIODO, ID]
        tiempo = "Tiempo necesario para completar (segundos)"
        cerradas = [c for c in d.columns if c not in set(propias) | {PERIODO, ID, tiempo, self.abierta}]
        rep = d[d.duplicated(envio, keep=False)]
        norm = rep[cerradas].apply(lambda s: s.map(normalizar))
        g = norm.groupby([rep[PERIODO], rep[ID]])
        distintas = g.nunique() > 1
        t_distinto = rep.groupby(envio)[tiempo].nunique() > 1
        copias = rep.groupby(envio).size()
        n_dif = distintas.sum(axis=1)
        filas = []
        for llave in copias.index:
            if n_dif[llave] == 0 and not t_distinto[llave]:
                continue
            cols = [c.strip() for c in distintas.columns[distintas.loc[llave]]]
            filas.append({"Periodo": llave[0], "ID de respuesta": llave[1], "Copias del envío": copias[llave],
                          "Preguntas cerradas con respuesta distinta": int(n_dif[llave]),
                          "Preguntas con diferencia (primeras 6)": ", ".join(cols[:6]),
                          "¿Tiempo de diligenciamiento distinto entre copias?": "Sí" if t_distinto[llave] else "No"})
        det = pd.DataFrame(filas, columns=["Periodo", "ID de respuesta", "Copias del envío",
                                           "Preguntas cerradas con respuesta distinta",
                                           "Preguntas con diferencia (primeras 6)",
                                           "¿Tiempo de diligenciamiento distinto entre copias?"])
        return Hoja("I17", "Envío de acreditación repetido por programa con respuestas cerradas distintas entre copias",
                    f"Cada envío se repite una vez por programa asociado ({len(copias):,} envíos con más de una copia). "
                    f"En {(n_dif > 0).sum():,} envíos alguna pregunta cerrada cambia entre copias (comparación sin "
                    f"distinguir mayúsculas, tildes ni puntuación) y en {int(t_distinto.sum()):,} cambia el tiempo de "
                    "diligenciamiento. Se excluyen de la comparación las columnas propias del programa.",
                    "Si las copias son el mismo envío, sus respuestas deberían ser idénticas; las diferencias impiden "
                    "saber cuál es la respuesta original.",
                    "Una respuesta por envío (Periodo + ID de respuesta); los programas se registran aparte (tabla "
                    "respuestas_programa). Se solicita al cliente explicar por qué cambian las respuestas entre copias.",
                    det, _periodos(det))


# --------------------------------------------------------------------------- calidad de servicios

class RevisionCalidad:
    ANIO, AREA, POBLACION, PREGUNTA, RESPUESTA, ACCESO = (
        "Año", "Facultad/Departamento/Área", "Población", "Pregunta", "Respuesta", "ID de Acceso")

    def __init__(self, df: pd.DataFrame, textos: Textos, codigos: pd.Series):
        self.df = df.assign(_codigo=codigos.values)
        self.t = textos

    def hojas(self) -> list[Callable[[], Hoja]]:
        return [self.i18, self.i19, self.i20, self.i21]

    def i18(self) -> Hoja:
        d, area = self.df, self.AREA
        valores = d.groupby(area).agg(Filas=(area, "size"),
                                      Años=(self.ANIO, lambda s: ", ".join(sorted(s.unique()))))
        n = pd.Series({v: normalizar(v) for v in valores.index})
        sin_prefijo = n.str.replace(r"^(facultad|departamento) de ", "", regex=True)
        problemas = {}
        for v in valores.index:
            p = []
            if v != v.strip():
                p.append("Espacios al inicio o al final")
            if (n == n[v]).sum() > 1:
                p.append("Misma área escrita de otra forma (mayúsculas, tildes o puntuación)")
            otros = [o for o in valores.index if o != v and sin_prefijo[o] == sin_prefijo[v] and n[o] != n[v]]
            if otros:
                p.append(f"Mismo nombre que «{otros[0].strip()}» con o sin el prefijo «Facultad de»")
            if n[v] in VALORES_RELLENO | self.t.no_respuesta:
                p.append("Valor de relleno: no identifica el área")
            if p:
                problemas[v] = "; ".join(p)
        det = valores.loc[list(problemas)].reset_index().rename(columns={area: "Valor en la base"})
        det.insert(1, "Problema", det["Valor en la base"].map(problemas))
        det["Valor en la base"] = det["Valor en la base"].map(lambda v: f"«{v}»")
        return Hoja("I18", "Facultad/Departamento/Área con valores no estándar (calidad)",
                    f"{len(det)} valores de la columna Facultad/Departamento/Área tienen un problema de escritura o "
                    f"no identifican el área ({int(det['Filas'].sum()) if len(det) else 0} filas). El detalle muestra "
                    "un valor por fila, entre comillas para que se vean los espacios.",
                    "Un tablero filtrado por facultad o área separaría o perdería esas respuestas.",
                    "Se reportan sin cambios; se solicita al cliente corregirlos en la base o enviar la equivalencia.",
                    det, _periodos(det, "Años"))

    def i19(self) -> Hoja:
        d = self.df
        varias = d.groupby("_codigo")[self.PREGUNTA].transform("nunique") > 1
        det = (d[varias].groupby(["_codigo", self.ANIO, self.PREGUNTA])
               .agg(Poblaciones=(self.POBLACION, lambda s: ", ".join(sorted(s.unique()))),
                    Filas=(self.PREGUNTA, "size"))
               .reset_index()
               .rename(columns={"_codigo": "Código de pregunta (proyecto)", self.PREGUNTA: "Redacción en la base"}))
        n_codigos = det["Código de pregunta (proyecto)"].nunique()
        return Hoja("I19", "La misma pregunta aparece con varias redacciones (calidad)",
                    f"{n_codigos} preguntas aparecen con más de una redacción ({len(det)} redacciones), según el año "
                    "y la población. El proyecto las agrupa con un código (hoja preguntas de la configuración).",
                    "Informativo: si las redacciones no son equivalentes, comparar respuestas entre años o "
                    "poblaciones no sería válido.",
                    "Se agrupan con el código del proyecto. Se solicita al cliente confirmar que son la misma pregunta.",
                    det, _periodos(det, self.ANIO))

    def i20(self) -> Hoja:
        d = self.df
        anios = sorted(d[self.ANIO].unique())
        partes = []
        for tipo, col in (("Pregunta (código del proyecto)", "_codigo"), ("Población", self.POBLACION)):
            t = pd.crosstab(d[col], d[self.ANIO]).reindex(columns=anios, fill_value=0)
            t.insert(0, "Tipo", tipo)
            t["Años con respuestas"] = (t[anios] > 0).sum(axis=1)
            partes.append(t.reset_index().rename(columns={col: "Valor"}))
        det = pd.concat(partes, ignore_index=True)
        det.columns = [str(c) for c in det.columns]
        parciales = det[det["Años con respuestas"] < len(anios)]
        return Hoja("I20", "Preguntas y poblaciones que no se repiten todos los años (calidad)",
                    f"De {len(det)} preguntas y poblaciones, {len(parciales)} no tienen respuestas en todos los años "
                    f"({', '.join(anios)}). El detalle cruza cada una con el año.",
                    "Informativo: las tendencias entre años solo son comparables para las preguntas y poblaciones "
                    "presentes en ambos años.",
                    "Se reporta sin cambios; los tableros deben mostrar solo los años con datos.",
                    det, ", ".join(anios))

    def i21(self) -> Hoja:
        d = self.df
        nr = ~self.t.es_comentario(d[self.RESPUESTA])
        x = d[nr].copy()
        x["Valor (normalizado)"] = x[self.RESPUESTA].map(normalizar)
        valores = ", ".join(f"«{v}» ({n:,})" for v, n in x["Valor (normalizado)"].value_counts().items())
        det = x.rename(columns={"_codigo": "Código de pregunta (proyecto)"})[
            [self.ANIO, self.ACCESO, self.POBLACION, "Código de pregunta (proyecto)", "Valor (normalizado)"]]
        return Hoja("I21", "Respuestas abiertas que son no-respuesta (calidad)",
                    f"La base de respuestas abiertas trae {len(x):,} filas cuyo texto es una no-respuesta: "
                    f"{valores or 'ninguna'}.",
                    "Si se contaran como comentarios, inflarían la participación y el modelo leería «gracias» como "
                    "opinión positiva.",
                    "Se tratan como no-respuesta con la lista de la configuración; se conservan en respuestas.csv "
                    "marcadas (es_no_respuesta).",
                    det, _periodos(det, self.ANIO))


# --------------------------------------------------------------------------- Excel

def _escribir_excel(hojas: list[Hoja], ruta: Path, id_corrida: str, fecha: str, bases: list[str]) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    def fuente(**k):
        return Font(name="Arial", **k)

    relleno = PatternFill("solid", fgColor="1F3864")
    linea = Side(style="thin", color="BFBFBF")
    borde = Border(top=linea, bottom=linea, left=linea, right=linea)
    ajuste = Alignment(wrap_text=True, vertical="top")

    wb = Workbook()
    ws = wb.active
    ws.title = "Resumen"
    ws["A1"] = "Informe de inconsistencias de las bases del cliente"
    ws["A1"].font = fuente(bold=True, size=14)
    notas = [
        f"Corrida de preparación {id_corrida}, generada el {fecha} (UTC). Bases revisadas: {'; '.join(bases)}.",
        "Cómo ubicar cada caso en la base original: busque el «ID de respuesta» (o «ID de Acceso») y confirme con "
        "el periodo y el curso. No se indica número de fila porque cambia si el archivo se ordena o filtra.",
        f"Los comentarios aparecen como {OCULTO}: el texto completo está en la base original. Solo se muestran los "
        "valores de no-respuesta o de escala cerrada. Las columnas «(codificado)» se muestran como vienen en la base.",
        "«Filas en el detalle» se calcula con una fórmula sobre cada hoja. Las reglas por periodo (I05, I06) y el "
        "valor de relleno (I04) se calculan en cada corrida.",
    ]
    for i, texto in enumerate(notas):
        c = ws.cell(row=3 + i, column=1, value=texto)
        c.font, c.alignment = fuente(size=10), ajuste
        ws.merge_cells(start_row=3 + i, start_column=1, end_row=3 + i, end_column=8)
        ws.row_dimensions[3 + i].height = 30
    encabezado = ["N°", "Inconsistencia", "Qué se observó", "Periodos", "Filas en el detalle",
                  "Impacto en el análisis de preguntas abiertas", "Cómo se trata en el proyecto", "Hoja de detalle"]
    fila0 = 8
    for j, h in enumerate(encabezado, 1):
        c = ws.cell(row=fila0, column=j, value=h)
        c.font, c.fill, c.border = fuente(bold=True, color="FFFFFF"), relleno, borde
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for i, h in enumerate(hojas, 1):
        r = fila0 + i
        tiene = h.detalle is not None and not h.error
        if tiene:
            cuenta = f"=MAX(COUNTA('{h.codigo}'!A:A)-1,0)"
        else:
            cuenta = "—"
        observado = h.observado if not h.error else f"No se pudo calcular en esta corrida: {h.error}"
        valores = [h.codigo, h.titulo, observado, h.periodos, cuenta, h.impacto, h.tratamiento,
                   h.codigo if tiene else "—"]
        for j, v in enumerate(valores, 1):
            c = ws.cell(row=r, column=j, value=v)
            c.font, c.alignment, c.border = fuente(size=10), ajuste, borde
            if j == 5:
                c.number_format = "#,##0"
        if tiene:
            c = ws.cell(row=r, column=8)
            c.hyperlink = f"#'{h.codigo}'!A1"
            c.font = fuente(size=10, color="0563C1", underline="single")
    for col, ancho in zip("ABCDEFGH", [6, 30, 55, 14, 12, 38, 45, 10]):
        ws.column_dimensions[col].width = ancho
    ws.freeze_panes = f"A{fila0 + 1}"

    for h in hojas:
        if h.detalle is None or h.error:
            continue
        d = h.detalle.rename(columns={**NOMBRES, **NOMBRES_CORTOS})
        s = wb.create_sheet(h.codigo)
        for j, nombre in enumerate(d.columns, 1):
            c = s.cell(row=1, column=j, value=str(nombre))
            c.font, c.fill = fuente(bold=True, color="FFFFFF"), relleno
            c.alignment = Alignment(wrap_text=True, vertical="center")
        for i, fila in enumerate(d.itertuples(index=False), 2):
            for j, v in enumerate(fila, 1):
                if v is None or (isinstance(v, float) and v != v):
                    v = None
                elif hasattr(v, "item"):
                    v = v.item()
                s.cell(row=i, column=j, value=v).font = fuente(size=10)
        for j, nombre in enumerate(d.columns, 1):
            largo = d.iloc[:, j - 1].astype(str).str.len().quantile(0.9) if len(d) else 8
            s.column_dimensions[get_column_letter(j)].width = min(max(10, max(len(str(nombre)) * 0.9, largo) + 2), 45)
        s.row_dimensions[1].height = 42
        s.freeze_panes = "A2"
        if len(d.columns):
            s.auto_filter.ref = s.dimensions
    ruta.parent.mkdir(parents=True, exist_ok=True)
    wb.save(ruta)


# --------------------------------------------------------------------------- entrada

def _seguro(funcion: Callable[[], Hoja], codigo: str) -> Hoja:
    try:
        return funcion()
    except Exception as exc:  # una revisión que falla no detiene las demás
        return Hoja(codigo, "Revisión no calculada", "", "", "", error=f"{type(exc).__name__}: {exc}")


def revisar_bases(crudas: dict[str, pd.DataFrame], config: str | Path,
                  no_respuesta: frozenset[str], valores_cerrados: frozenset[str]) -> list[Hoja]:
    """Corre todas las revisiones que aplican a las encuestas presentes.

    Args:
        crudas: {encuesta: hoja cruda completa, leída con dtype=str y sin NaN}.
    """
    textos = Textos(no_respuesta, valores_cerrados)
    hojas: list[Hoja] = []
    bloques: list[tuple[list[str], Callable[[], object]]] = []

    evadoc = next((crudas[e] for e in EVADOC if e in crudas), None)
    if evadoc is not None:
        bloques.append(([f"I{n:02d}" for n in range(1, 15)], lambda: RevisionEvaDoc(evadoc, textos)))
    if ACREDITACION in crudas:
        def acred():
            df = crudas[ACREDITACION]
            alias = preguntas_activas(config, ACREDITACION)[0]["alias"]
            return RevisionAcreditacion(df, textos, resolver_columna(list(df.columns), alias))
        bloques.append((["I15", "I16", "I17"], acred))
    if CALIDAD in crudas:
        def calidad():
            df = crudas[CALIDAD]
            codigos = asignar_preguntas(df[RevisionCalidad.PREGUNTA], preguntas_activas(config, CALIDAD), CALIDAD)
            return RevisionCalidad(df, textos, codigos)
        bloques.append((["I18", "I19", "I20", "I21"], calidad))

    for codigos, crear in bloques:
        try:
            revision = crear()
        except Exception as exc:
            hojas += [Hoja(c, "Revisión no calculada", "", "", "", error=f"{type(exc).__name__}: {exc}")
                      for c in codigos]
            continue
        for codigo, funcion in zip(codigos, revision.hojas()):
            hojas.append(_seguro(funcion, codigo))
    return hojas


def generar_informe(crudas: dict[str, pd.DataFrame], config: str | Path, no_respuesta: frozenset[str],
                    valores_cerrados: frozenset[str], id_corrida: str, fecha: str,
                    bases: list[str], carpeta: str | Path = CARPETA_INFORME) -> tuple[Path, list[Hoja]]:
    """Revisa las bases y escribe el informe. Devuelve la ruta y las hojas (para avisos)."""
    hojas = revisar_bases(crudas, config, no_respuesta, valores_cerrados)
    ruta = Path(carpeta) / f"Informe_inconsistencias_corrida_{id_corrida}.xlsx"
    _escribir_excel(hojas, ruta, id_corrida, fecha, bases)
    return ruta, hojas
