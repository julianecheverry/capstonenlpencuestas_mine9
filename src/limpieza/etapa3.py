"""
Etapa 3: limpieza del texto de las cláusulas y n-gramas (contrato v0.2, sección 6).

Lee `clausulas.csv` y `metadatos_nlp.csv` de la carpeta del NLP y escribe en
`data/processed/limpieza/`:

  clausulas_texto.csv      una fila por cláusula con opinión
  ngramas.csv              una fila por n-grama distinto (1, 2 y 3 palabras) de cada cláusula
  metadatos_limpieza.csv   una fila por corrida

Reglas principales:
- Las negaciones («no», «nunca», «sin»…) se conservan: «no explica» no es «explica».
- Las stopwords de varias palabras («sin embargo») se quitan antes que las de una.
- Las stopwords se comparan normalizadas, así «más» quita «mas».
- `texto_nubes` además quita los adverbios que no valoran; «bien», «mal»,
  «mejor», «peor», «siempre» y «nunca» se quedan (columna `valora`).
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from limpieza import VERSION_LIMPIEZA
from limpieza.listas import cargar_listas
from limpieza.textos import limpiar, ngramas, tokenizar
from preparacion.normalizacion import normalizar

RAIZ = Path(__file__).resolve().parents[2]
UTILIDADES = RAIZ / "data" / "utilities"
CARPETA_NLP = RAIZ / "data" / "processed" / "nlp"
CARPETA_SALIDA = RAIZ / "data" / "processed" / "limpieza"
CODIFICACION = "utf-8-sig"
SEPARADOR_ID = "|"

# Decisiones abiertas (especificación, sección 5): parámetros con el valor de hoy.
LEMATIZAR = False
ENFOQUES_SIN_NUBE: tuple[str, ...] = ()

COLUMNAS_TEXTO = ["clausula_id", "texto_normalizado", "texto_sin_stopwords", "texto_nubes"]
COLUMNAS_NGRAMAS = ["clausula_id", "n", "ngrama", "ngrama_mostrar", "frecuencia", "es_nube"]


def _leer(ruta: Path) -> pd.DataFrame:
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}.")
    return pd.read_csv(ruta, dtype=str, keep_default_na=False, encoding=CODIFICACION)


def _enfoques(catalogo: Path | None) -> dict[str, str]:
    if not ENFOQUES_SIN_NUBE:
        return {}
    if catalogo is None:
        raise ValueError("ENFOQUES_SIN_NUBE necesita el catalogo_preguntas.csv de la Etapa 1.")
    c = _leer(catalogo)
    return dict(zip(c["pregunta_codigo"], c["enfoque"]))


def validar_salida(clausulas: pd.DataFrame, textos: pd.DataFrame, ngr: pd.DataFrame,
                   negaciones: frozenset[str]) -> list[str]:
    """Chequeos del contrato (especificación, sección 6). Devuelve los errores encontrados."""
    errores = []
    con_opinion = clausulas.loc[clausulas["es_opinion"].str.lower() == "true", "clausula_id"]
    if set(textos["clausula_id"]) != set(con_opinion) or len(textos) != len(con_opinion):
        errores.append("clausulas_texto no tiene exactamente una fila por cláusula con opinión.")
    if not set(ngr["clausula_id"]) <= set(clausulas["clausula_id"]):
        errores.append("ngramas tiene clausula_id que no existen en clausulas.")
    if ngr.duplicated(["clausula_id", "n", "ngrama"]).any():
        errores.append("ngramas repite clausula_id + n + ngrama.")
    if not ngr["n"].isin([1, 2, 3]).all():
        errores.append("ngramas tiene n distinto de 1, 2 o 3.")
    if (ngr["frecuencia"] < 1).any():
        errores.append("ngramas tiene frecuencia menor que 1.")
    solas = ngr["es_nube"] & ngr["ngrama"].map(lambda g: all(p in negaciones for p in g.split()))
    if solas.any():
        errores.append(f"{int(solas.sum())} n-gramas de nube son solo negaciones.")
    return errores


def ejecutar_limpieza(carpeta_nlp: str | Path = CARPETA_NLP,
                      carpeta_salida: str | Path | None = CARPETA_SALIDA,
                      utilidades: str | Path = UTILIDADES,
                      catalogo_preguntas: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Corre la Etapa 3 y devuelve sus tres tablas.

    Args:
        carpeta_nlp: carpeta con clausulas.csv y metadatos_nlp.csv.
        carpeta_salida: dónde escribir los CSV; None para no escribir (pruebas).
        utilidades: carpeta con stopwords.xlsx y negaciones.xlsx.
        catalogo_preguntas: solo si ENFOQUES_SIN_NUBE no está vacío.

    Raises:
        ValueError: si una lista falta o está vacía, o la salida no cumple el contrato.
    """
    if LEMATIZAR:
        raise NotImplementedError("La lematización está por decidir (especificación, sección 5).")
    carpeta_nlp, utilidades = Path(carpeta_nlp), Path(utilidades)
    listas = cargar_listas(utilidades / "stopwords.xlsx", utilidades / "negaciones.xlsx")
    for aviso in listas.avisos:
        print(f"Aviso: {aviso}")

    clausulas = _leer(carpeta_nlp / "clausulas.csv")
    meta_nlp = _leer(carpeta_nlp / "metadatos_nlp.csv")
    faltan = sorted({"clausula_id", "respuesta_id", "texto_clausula", "es_opinion"} - set(clausulas.columns))
    if faltan:
        raise ValueError(f"A clausulas.csv le faltan las columnas {faltan}.")
    ids_nlp = sorted(set(meta_nlp["id_corrida_nlp"]))
    if len(ids_nlp) != 1:
        raise ValueError(f"metadatos_nlp.csv debe traer un solo id_corrida_nlp; trae {ids_nlp}.")

    enfoques = _enfoques(Path(catalogo_preguntas) if catalogo_preguntas else None)
    opinion = clausulas[clausulas["es_opinion"].str.lower() == "true"]
    partes = opinion["respuesta_id"].str.split(SEPARADOR_ID)
    encuestas = partes.str[0]
    preguntas = partes.str[-1]
    sin_lista = sorted(set(encuestas) - {e for e, p in listas.palabras_particulares.items() if p})
    if sin_lista:
        print(f"Aviso: encuestas sin stopwords particulares: {sin_lista}.")

    filas_texto, por_clausula, formas = [], [], defaultdict(Counter)
    for cid, texto, encuesta, pregunta in zip(opinion["clausula_id"], opinion["texto_clausula"],
                                              encuestas, preguntas):
        palabras, frases = listas.stopwords(encuesta)
        tokens = tokenizar(texto)
        limpios = limpiar(tokens, palabras, frases)
        nube = [t for t in limpios if t[0] not in listas.adverbios_no_valoran]
        filas_texto.append({
            "clausula_id": cid,
            "texto_normalizado": normalizar(texto),
            "texto_sin_stopwords": " ".join(t[0] for t in limpios),
            "texto_nubes": " ".join(t[0] for t in nube),
        })
        en_nube = set() if enfoques.get(pregunta) in ENFOQUES_SIN_NUBE else {
            (n, g) for (n, g, _m) in ngramas(nube)}
        cuenta = Counter()
        for (n, g, mostrar), k in ngramas(limpios).items():
            cuenta[(n, g)] += k
            formas[(n, g)][mostrar] += k
        por_clausula.append((cid, cuenta, en_nube))

    # La forma con tildes más frecuente de cada n-grama en todo el corpus
    # (en empate, la primera en orden alfabético, para que la salida no cambie).
    mostrar = {clave: min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0] for clave, c in formas.items()}
    filas_ng = []
    for cid, cuenta, en_nube in por_clausula:
        for (n, g), k in cuenta.items():
            sola_negacion = all(p in listas.negaciones for p in g.split())
            filas_ng.append({"clausula_id": cid, "n": n, "ngrama": g, "ngrama_mostrar": mostrar[(n, g)],
                             "frecuencia": k, "es_nube": (n, g) in en_nube and not sola_negacion})

    textos = pd.DataFrame(filas_texto, columns=COLUMNAS_TEXTO)
    ngr = pd.DataFrame(filas_ng, columns=COLUMNAS_NGRAMAS)
    errores = validar_salida(clausulas, textos, ngr, listas.negaciones)
    if errores:
        raise ValueError("La salida de la Etapa 3 no cumple el contrato:\n- " + "\n- ".join(errores))

    fecha = datetime.now(timezone.utc)
    huella_insumo = hashlib.sha256((carpeta_nlp / "clausulas.csv").read_bytes()).hexdigest()
    huella_corrida = hashlib.sha256(
        (huella_insumo + listas.huella_stopwords + listas.huella_negaciones
         + VERSION_LIMPIEZA).encode()).hexdigest()[:10]
    id_corrida = f"{fecha:%Y%m%d-%H%M%S}_{huella_corrida}"
    metadatos = pd.DataFrame([{
        "id_corrida_limpieza": id_corrida,
        "id_corrida_nlp": ids_nlp[0],
        "fecha_corrida": fecha.isoformat(timespec="seconds"),
        "version_limpieza": VERSION_LIMPIEZA,
        "huella_stopwords": listas.huella_stopwords,
        "huella_negaciones": listas.huella_negaciones,
        "n_clausulas": len(textos),
        "n_ngramas": len(ngr),
        "n_ngramas_nube": int(ngr["es_nube"].sum()),
    }])

    if carpeta_salida is not None:
        salida = Path(carpeta_salida)
        salida.mkdir(parents=True, exist_ok=True)
        textos.to_csv(salida / "clausulas_texto.csv", index=False, encoding=CODIFICACION)
        ngr.to_csv(salida / "ngramas.csv", index=False, encoding=CODIFICACION)
        metadatos.to_csv(salida / "metadatos_limpieza.csv", index=False, encoding=CODIFICACION)
        print(f"\nEtapa 3 lista: corrida {id_corrida}, {len(textos):,} cláusulas, "
              f"{len(ngr):,} n-gramas ({int(ngr['es_nube'].sum()):,} de nube), archivos en {salida}")
    return {"clausulas_texto": textos, "ngramas": ngr, "metadatos_limpieza": metadatos}
