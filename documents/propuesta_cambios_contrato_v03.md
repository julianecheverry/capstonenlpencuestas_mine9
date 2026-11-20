# Propuesta de cambios al contrato de datos (v0.2 → v0.3) — salidas del NLP (Etapa 2)

Fecha: 2026-11-19 · Origen: plan de cambios al NLP v2 (`documents/plan_cambios_nlp_etapa2.md`, Puntos 1–7), rama
`feature/contrato-etapa2-nlp`. **Es una propuesta para llevar a Cris: no se edita el contrato ni la spec.** Hasta que se
actualice, las validaciones del pipeline (`validar_salidas_nlp`) están **adelantadas** al contrato v0.2 y Power BI no debe
conectarse a estas tablas.

## Cambios

| # | Cambio | Dónde afecta (v0.2) | Punto |
|---|---|---|---|
| 1 | La entrada del NLP son solo las respuestas con `es_no_respuesta = False` **y** `tiene_alerta = False`. Una respuesta con alerta no pasa al NLP aunque la alerta esté negada. | §5.1 y principios | 1 |
| 2 | Las alertas salen del NLP: `alertas.csv`, `n_alertas` y `huella_alertas` ya no son del NLP (los publica la Etapa 1 v1.10, sin `clausula_id` ni `categoria_alerta`). El NLP publica **4 tablas**: `clausulas`, `auditoria_clasificacion`, `catalogo_aspectos`, `metadatos_nlp`. | §5.2 (regla de alertas), §5.3, §5.4 | 1 |
| 3 | `SIN_ASPECTO` pasa a ser el residual que HDBSCAN dejó sin grupo: `aspecto_origen = hdbscan`, `grupo_visual = «Sin grupo visual»` (valor nuevo del dominio). El origen `sistema` ya no se emite (se conserva en el dominio por compatibilidad). Las cláusulas sin opinión (`es_opinion = False`) también salen `SIN_ASPECTO` / `hdbscan`; las distingue `es_opinion`. | §3.3, §5.3 (`aspecto_origen`, `grupo_visual`) | 4 |
| 4 | `aspecto_id` de los temas de HDBSCAN = **primer sustantivo** de `palabras_clave` (spaCy `es_core_news_lg` en contexto; `NOUN`, sin `PROPN`), en MAYÚSCULAS, sin tildes (Ñ→N), con `_` por signos; **ya no** `EMERGENTE_<periodo>_<n>`. Choque con un aspecto fijo de la taxonomía de la encuesta: prefijo `EMERGENTE_`; choque entre temas de la misma pregunta: sufijo `_2`, `_3`; el mismo sustantivo en otra pregunta de la encuesta es el mismo aspecto (una fila de catálogo). Sin sustantivo: primer término + aviso (`TEMA_<n>` si no hay términos). `aspecto_nombre` = el sustantivo; `palabras_clave` separadas por coma; `grupo_visual` sigue «Temas emergentes». `taxonomia.xlsx` no cambia. | §3.3, §5.3 (`catalogo_aspectos`) | 4 |
| 5 | `metadatos_nlp`: `pct_aspecto_hdbscan` cuenta solo los temas emergentes y `pct_sin_aspecto` cuenta `SIN_ASPECTO`; las cuatro `pct_*` suman 100. | §5.3 (`metadatos_nlp`) | 4 |
| 6 | **No cambia** que `clausulas.csv` trae todas las cláusulas, con y sin opinión (se descartó el filtro `es_opinion = True`). Único ajuste: «cada respuesta con comentario tiene ≥ 1 cláusula» admite como excepción las respuestas procesables que quedan **vacías o de ruido** al limpiarse, que no generan cláusulas; `n_respuestas` de `metadatos_nlp` cuenta todas las procesadas. | §5.2, §5.4 | 5 |
| 7 | El NLP **no** publica `respuestas.csv` (ya existe el de la Etapa 1). Publica, en la carpeta de su corrida, el Excel `clausulas_documentos_unificado.xlsx` (hoja `unificado`: cláusulas internas + datos de su respuesta + `es_no_respuesta` y `tiene_alerta`); no es una tabla del contrato y no entra en el hash de archivos. | §5.3 | 6 |
| 8 | `VALORACIÓN GLOBAL`, `COMENTARIO GENERAL` y la clase `OPINIÓN GENERAL` de U4 son **una sola categoría**: `aspecto_id = OPINION_GENERAL`, `aspecto_nombre = «Opinión general»`; `aspecto_origen = regla` (sin contenido o comentario corto) o `clasificador` (la predijo U4). «Sin aspecto» y «Ninguno de la lista», si U4 las predijera, también salen `OPINION_GENERAL` (ninguna taxonomía actual las usa). La regla «cláusula sin contenido → Opinión general» vale también en respuestas cortas (`sin_contenido_en_cortas = True`, sin pasar por U4). **Columna nueva en `clausulas.csv`: `motivo_opinion_general`** (texto, después de `aspecto_origen`; `sin_contenido` | `comentario_corto` | `clasificador`; obligatoria con `aspecto_id = OPINION_GENERAL` y vacía en el resto). Coherencia: origen `regla` ⇒ `sin_contenido` o `comentario_corto`; origen `clasificador` ⇒ `clasificador`. | §5.2 (reglas de aspecto), §5.3 (`clausulas`: columna nueva) | 7 |
| 9 | Ruta de salida: `data/processed/nlp/<id_corrida_nlp>/` para las tablas del contrato y `data/processed/nlp/modelado/<encuesta>/<run_id>/` para los artefactos internos de modelado (antes `outputs/`). | §5.3 (ubicación) | 2 |

## Cosas que quedan abiertas o sin probar con datos reales
- La **elección del sustantivo con spaCy real** (`es_core_news_lg` no está instalado en el entorno de desarrollo): validar en Colab con grupos reales.
- Impacto de `sin_contenido_en_cortas = True` (cláusulas sin contenido de respuestas cortas que antes terminaban en un aspecto real): medir con el modelo U4 entrenado y GPU
  (cuántas de esas cláusulas asigna hoy U4 a un aspecto real frente a «Opinión general») y, si el efecto no es aceptable, dejar el parámetro en `False`.
- Corrida real de obs19 (24.347 respuestas procesables = 24.433 con comentario − 86 con alerta) y comparación de cifras con la spec.
- Cuántas respuestas vacías o de ruido tiene cada pregunta (no generan cláusulas).
- `aspecto_origen = hdbscan` en cláusulas sin opinión no viene de HDBSCAN; `pct_sin_aspecto` mezcla sin opinión y residual sin grupo. Separarlos exigiría una columna nueva.
- `documents/ejemplos_salidas_nlp/` conserva el esquema v0.2 (5 tablas); se regenera con datos sintéticos si se aprueba.
