"""
TEMPORAL: convierte la salida actual del NLP al formato de clausulas.csv del contrato.

El NLP todavía no entrega clausulas.csv (especificación del NLP, tareas 1 y 2).
Mientras tanto, este adaptador lee data/raw/clausulas_documentos_unificado.xlsx
(obs19, hoja `unificado`, encabezado en la fila 2) y escribe un clausulas.csv
mínimo para probar la Etapa 3 con datos reales.

Sus identificadores son PROVISIONALES y no cruzan con la Etapa 1: el archivo del
NLP no trae respuesta_id ni Id Docente y viene del corpus anterior. Lo que salga
con este adaptador sirve para revisar la limpieza, no para cargar en Power BI.

Se borra cuando el NLP entregue su clausulas.csv (checklist del contrato).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
ARCHIVO_NLP = RAIZ / "data" / "raw" / "clausulas_documentos_unificado.xlsx"
CARPETA_ADAPTADA = RAIZ / "data" / "processed" / "nlp_adaptado"
CODIFICACION = "utf-8-sig"


def adaptar(archivo: str | Path = ARCHIVO_NLP, carpeta_salida: str | Path = CARPETA_ADAPTADA,
            encuesta: str = "evadoc_evaluacion",
            pregunta_codigo: str = "obs19_aspectos_resaltar") -> pd.DataFrame:
    """Escribe clausulas.csv y metadatos_nlp.csv en `carpeta_salida` y devuelve las cláusulas.

    - respuesta_id provisional = encuesta|Periodo|doc<doc_id>|pregunta_codigo
    - clausula_id = respuesta_id|orden, con orden = clausula_id del NLP + 1 (el NLP numera desde 0)
    - es_opinion = estado == CON_OPINION. Las respuestas con alerta van completas y sin
      sentimiento (acuerdo del 15-nov), y las sin opinión tampoco tienen sentimiento.
    """
    archivo = Path(archivo)
    df = pd.read_excel(archivo, sheet_name="unificado", header=1,
                       usecols=["doc_id", "estado", "Periodo", "clausula_id", "clausula"], dtype=str)
    respuesta_id = encuesta + "|" + df["Periodo"] + "|doc" + df["doc_id"] + "|" + pregunta_codigo
    orden = (df["clausula_id"].astype(int) + 1).astype(str)
    clausulas = pd.DataFrame({
        "clausula_id": respuesta_id + "|" + orden,
        "respuesta_id": respuesta_id,
        "orden": orden,
        "texto_clausula": df["clausula"].fillna(""),
        "es_opinion": (df["estado"] == "CON_OPINION").map({True: "True", False: "False"}),
    })
    if clausulas["clausula_id"].duplicated().any():
        raise ValueError("El adaptador produjo clausula_id repetidos.")
    huella = hashlib.sha256(archivo.read_bytes()).hexdigest()[:10]
    meta = pd.DataFrame([{"id_corrida_nlp": f"adaptador_{huella}",
                          "origen": "adaptador_nlp_actual (temporal)",
                          "archivo": archivo.name, "n_clausulas": len(clausulas)}])
    salida = Path(carpeta_salida)
    salida.mkdir(parents=True, exist_ok=True)
    clausulas.to_csv(salida / "clausulas.csv", index=False, encoding=CODIFICACION)
    meta.to_csv(salida / "metadatos_nlp.csv", index=False, encoding=CODIFICACION)
    return clausulas
