# Prompt v2 para Claude Code — ajustes al NLP (Etapa 2) tras mover las alertas a la Etapa 1

> Pégalo en una sesión nueva de Claude Code abierta en la raíz del repo (`capstonenlpencuestas_mine9`), rama `feature/contrato-etapa2-nlp`. Antes, haz tú el merge de `feature/etapa1-alertas` (Punto 0 del plan).

---

Vas a ayudarme a ajustar el NLP del proyecto Prisma (`notebooks/Prisma_unificado.ipynb` y `src/nlp_utilidades/modulos_nlp.py`, más `src/nlp_utilidades/salidas_contrato.py` y `taxonomia.py` donde el plan lo exija) después de una decisión del equipo: **las alertas pasaron de la Etapa 2 a la Etapa 1**. Trabajamos **punto por punto, con mi aprobación manual al final de cada uno**.

## Lee primero (completo)
1. `documents/plan_cambios_nlp_etapa2.md` — plan v2 con los 7 puntos. Es tu guía; síguelo.
2. `documents/cambios etapa1 alertas.txt` — qué cambió en la Etapa 1 v1.10. **Manda sobre la spec y el contrato en todo lo relativo a alertas.**
3. `documents/especificaciones/spec_nlp_etapa2.md` y `documents/especificaciones/contrato_datos_v0.2.md` — vigentes salvo en lo que el plan cambie (plan, secciones 8 y 11).

## Estado que debes asumir
- Las fases 0–4 del plan anterior ya están hechas y commiteadas (ingesta desde la Etapa 1, rutas, conciliación de utilities, salidas del contrato v0.2 en `salidas_contrato.py`, Paso 12b del cuaderno). No las rehagas.
- Los 7 puntos nuevos: (1) ingestar solo `es_no_respuesta = "False"` y `tiene_alerta = "False"`; (2) salida de modelado de `RAIZ/outputs` a `RAIZ/data/processed/nlp`; (3) archivos exportados sin datos de respuestas que no cumplen esas condiciones; (4) ajustes a `catalogo_aspectos.csv` (`SIN_ASPECTO` y temas de HDBSCAN con ID = primer sustantivo de `palabras_clave`), también en `taxonomia.xlsx` y `clausulas.csv`; (5) `clausulas.csv` solo con `es_opinion = True`; (6) `respuestas.csv` con todos los registros y las columnas `es_no_respuesta` y `tiene_alerta`; (7) unificar `VALORACIÓN GLOBAL`, `OPINIÓN GENERAL` y `COMENTARIO GENERAL`.

## Reglas (no negociables)
1. **Puntos y aprobaciones.** Un punto a la vez (Punto 0 → 1 → … → 7). Al terminar cada uno, detente, entrega el informe de cierre (plan §12) y espera mi mensaje exacto `Aprobado Punto N`. No avances sin él.
2. **Decisiones de una en una.** Cada aprobación o decisión que necesites de mí la preguntas **por separado**, con `AskUserQuestion`: **una sola pregunta por mensaje**, 2–4 opciones, la recomendada primero y marcada «(Recomendada)», y esperas mi respuesta antes de seguir. No agrupes preguntas, no las dejes para el final y no asumas la respuesta. Las «Pregunta N.x» del plan son las previstas; si surge otra, la haces igual, también de una en una. Antes de preguntar, verifica en el código/datos todo lo que puedas verificar solo.
3. **Git:** NO ejecutes `git add`, `commit`, `push`, `merge`, `checkout`, `switch`, `reset`, `stash` ni `rebase`. Solo lectura (`status`, `diff`, `log`, `branch --show-current`). Los commits los hago yo; propónme el mensaje al cerrar cada punto. No toques los cambios ajenos del árbol (`README.md` y `data/utilities/esquema de los corpus.csv` borrados, `.claude/`, `historico_entregables/README_24ago2026.md`).
4. **Pruebas de humo obligatorias** para todo el código nuevo o modificado (están descritas en cada punto del plan), ejecutadas **antes** de pedirme aprobación y reportadas con conteos (pasa/falla). Las duraderas van a `tests/` (`test_nlp_*.py`). `poetry run pytest` completo debe pasar sin regresiones respecto de la línea base que midas en el Punto 0. Si algo no se puede ejecutar aquí (GPU, spaCy grande, torch), dilo y prueba con datos sintéticos (por ejemplo, inyectando el etiquetador POS).
5. **Datos del cliente:** no imprimas textos de comentarios ni de cláusulas ni salidas de celdas del cuaderno. Revisa con conteos, nombres de columnas, IDs de aspecto y valores de categorías. `data/raw/`, `data/processed/etapa1/` y `data/etiquetas_manuales/` son de solo lectura; solo se escribe en las carpetas de salida que el plan indica. Nada de eso va a git.
6. **Cuaderno:** edítalo con `NotebookEdit` o con un script `nbformat` que deje intactas las demás celdas; no lo ejecutes completo (entrenar U4 exige GPU). Valida con `nbformat.validate` y compilando cada celda de código (ignora líneas `%` y `!`). Cita las celdas por número.
7. **No cambies el modelo ni sus parámetros** (hiperparámetros, umbrales, `max_palabras_respuesta_corta`) salvo donde un punto lo pida. `stopwords.xlsx` y `negaciones.xlsx` los lee también la Etapa 3: no los modifiques. Las salidas internas de modelado y las tablas del contrato no deben cambiar de columnas salvo lo que el plan indique.
8. **Si algo no cuadra con el plan** (un archivo no existe, una cifra no coincide, la estructura del código es distinta, el corpus no trae `tiene_alerta`), no improvises: informa y pregunta (de una en una).

## Supuestos del plan que debes confirmar conmigo en su punto
S1 (Punto 3): «no mostrar datos de respuestas que cumplan las condiciones» se lee como «no mostrar datos de las que **no** las cumplen». · S2 (Punto 5): el filtro es `es_opinion == "True"` en el archivo (Excel lo muestra como VERDADERO). · S3 (Punto 4.1): `SIN_ASPECTO` pasa a ser el residual que HDBSCAN dejó sin grupo. · S4 (Punto 4): qué significa «aplica también a `taxonomia.xlsx`». · S5 (Punto 1): las alertas salen de las salidas del NLP.

## Cómo empezar
Empieza **solo con el Punto 0** (línea base; sin cambios de código). Primero dime en una frase qué vas a hacer; luego ejecútalo y entrégame el informe de cierre. Si `data/processed/etapa1/` no tiene una corrida con `tiene_alerta`, ejecuta `python run_etapa1.py` solo si el `git status` confirma que la Etapa 1 v1.10 ya está en la rama; si no, dímelo y espera. No pases al Punto 1 hasta que yo escriba `Aprobado Punto 0`.
