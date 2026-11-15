"""
Etapa 1 — Preparación de datos.

Convierte las bases crudas del cliente en la tabla `respuestas`: una fila por
cada respuesta de una persona a una pregunta abierta, con una llave estable
(`respuesta_id`), sin duplicados y con la no-respuesta marcada (no eliminada).
Es el insumo de la Etapa 2 (NLP) y de los tableros de Power BI.

La especificación completa está en el contrato de salida del pipeline
(documento "Contrato de salida del pipeline NLP → Power BI").

Este paquete no depende de la limpieza de texto (nltk, stopwords): la
preparación de datos y el procesamiento de lenguaje son etapas separadas.
"""

VERSION_ETAPA1 = "1.7.0"
