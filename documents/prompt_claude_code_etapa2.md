# Prompt para Claude Code — cambios al NLP (Etapa 2)

> Pégalo en una sesión nueva de Claude Code abierta en la raíz del repo (`capstonenlpencuestas_mine9`), en la rama `feature/contrato-etapa2-nlp`.

---
# cambios al NLP (Etapa 2)

Vas a ayudarme a adaptar el NLP del proyecto Prisma al contrato de datos v0.2. Trabajamos **por fases, con mi aprobación manual al final de cada una**.

## Documentos que debes leer primero (completos)
1. `documents/plan_cambios_nlp_etapa2.md` — el plan por fases. Es tu guía; síguelo.
2. `documents/especificaciones/spec_nlp_etapa2.md` — especificación de cambios (§2 Insumos, §3 Salidas, §4 tareas, §5 decisiones abiertas, §6 criterios de aceptación).
3. `documents/especificaciones/contrato_datos_v0.2.md` — contrato (sobre todo §3 y §5). **Si la spec y el contrato se contradicen, manda el contrato.**

Los archivos a modificar son **solo** estos dos (más pruebas nuevas en `tests/` y, donde el plan lo diga, los Excel de `data/utilities/` con mi aprobación):
- `notebooks/Prisma_unificado.ipynb`
- `src/nlp_utilidades/modulos_nlp.py`

## Reglas (no negociables)
1. **Fases y aprobaciones.** Haz una sola fase a la vez (Fase 0 → 1 → 2 → 3 → 4). Al terminar, detente, entrega el informe de cierre (plan §7) y espera mi mensaje exacto `Aprobado Fase N`. Antes de editar en cada fase, muéstrame brevemente qué vas a cambiar y espera mi visto bueno solo si hay una decisión abierta (las marcadas en el plan).
2. **Git:** NO ejecutes `git add`, `commit`, `push`, `merge`, `checkout`, `switch`, `reset`, `stash` ni `rebase`. Solo lectura (`status`, `diff`, `log`, `branch --show-current`). Los commits los hago yo; propónme un mensaje de commit al cerrar cada fase.
3. **Pruebas de humo obligatorias** para todo el código nuevo o modificado, ejecutadas antes de pedirme aprobación (plan: sección "Pruebas de humo" de cada fase). Guarda las duraderas en `tests/` (p. ej. `tests/test_utilities_conciliacion.py`, `tests/test_nlp_salidas.py`) y corre `poetry run pytest` completo: la línea base es **98 aprobadas** y no puede haber regresiones. Si algo no se puede ejecutar aquí (GPU, dependencias pesadas), dilo explícitamente y prueba lo que sí se pueda con datos sintéticos.
4. **Datos del cliente:** no imprimas textos de comentarios, de cláusulas ni salidas de celdas del cuaderno. Revisa con conteos, nombres de columnas y valores de categorías. No modifiques `data/raw/`, `data/processed/` ni `data/etiquetas_manuales/` (solo lectura). No subas nada de eso a git.
5. **Cuaderno:** edítalo con `NotebookEdit` o con un script `nbformat` que deje intactas las demás celdas. No lo ejecutes completo. Valida con `nbformat.validate` y compilando cada celda de código (ignora líneas que empiezan con `%` o `!`). Al citar celdas usa su número y la sección de la que provienen.
6. **Excel de `data/utilities/`:** no borres ni sobrescribas sin mi aprobación; antes de modificar uno, copia el original a `data/utilities/_respaldo/`. `stopwords.xlsx` y `negaciones.xlsx` los lee también la Etapa 3 (`src/limpieza/listas.py`): no cambies sus hojas, columnas ni valores de TIPO sin que lo aprobemos.
7. **No cambies el modelo ni sus parámetros** salvo donde la fase lo pida. Las decisiones abiertas (spec §5) quedan como parámetros con el valor de hoy.
8. **Si algo no cuadra con el plan** (un archivo no existe, una cifra no coincide, la estructura del código es distinta), no improvises: informa y pregunta.

## Alcance de las fases (resumen; el detalle está en el plan)
- **Fase 0 — Línea base:** rama, `pytest` (98), importabilidad de `modulos_nlp`, dependencias que faltan, inventario de uso de `raiz`/`RAIZ` y `doc_id`. Sin cambios de código.
- **Fase 1 — Conciliación de Excel** en `data/utilities`: `negaciones.xlsx` vs `negaciones_1.xlsx` (son idénticos), `stopwords.xlsx` vs `stopwords_1.xlsx` (informe de diferencias y opciones A/B/C; recomiendo A), y `taxonomia.xlsx` vs etiquetas únicas de `data/etiquetas_manuales` (tabla de conciliación; no edites la taxonomía hasta que yo decida cada caso: `Vocación` en obs24, `Ninguno de la lista` y cuál archivo de acreditación es el vigente, `Formación docente` en obs20, `Avance progresivo` en obs25).
- **Fase 2 — Rutas** del cuaderno (y de `modulos_nlp.py` donde haga falta) hacia la estructura real del repo: módulo `nlp_utilidades.modulos_nlp`, `data/utilities`, `data/etiquetas_manuales` con archivo explícito por perfil, `models/modelos/u4_aspectos`, calibrador en `src/nlp_utilidades/`, corrección de `col_respuesta` en obs23–25, y un bloque único `RUTAS` con `assert` de existencia.
- **Fase 3 — Ingesta desde la Etapa 1** (spec §2 y tareas 1–2): leer la corrida más reciente con `preparacion.corridas.ultima_corrida`, `respuestas.csv` con `dtype=str, keep_default_na=False, encoding="utf-8-sig"`, filtrar `es_no_respuesta == "False"` y la pregunta procesada, segmentar `texto_original`, llevar `respuesta_id`, registrar `id_corrida_preparacion`. Incluye los pasos previos A–D de la spec si el corpus viejo de obs19 está disponible (si no, repórtalo y déjalos pendientes).
- **Fase 4 — Salidas** (spec §3 y tareas 2–10): `clausulas.csv`, `auditoria_clasificacion.csv`, `alertas.csv`, `catalogo_aspectos.csv`, `metadatos_nlp.csv` en `data/processed/nlp/`, IDs `respuesta_id|orden`, parámetros para las decisiones abiertas, y `validar_salidas_nlp` con los criterios de la spec §6 / contrato §5.4. Conserva las salidas internas actuales en `outputs/`.

## Cómo empezar
Empieza **solo con la Fase 0**. Primero dime en una frase qué vas a hacer, luego ejecútala y entrégame el informe de cierre. No pases a la Fase 1 hasta que yo escriba `Aprobado Fase 0`.
