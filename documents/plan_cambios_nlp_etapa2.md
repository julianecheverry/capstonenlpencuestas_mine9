# Plan de cambios al NLP (Etapa 2) — `Prisma_unificado.ipynb` y `modulos_nlp.py`

Fecha: 2026-11-17 · Rama de trabajo: `feature/contrato-etapa2-nlp` (ya existe y está activa)
Referencias: `documents/especificaciones/spec_nlp_etapa2.md` (specs del 15-nov) y `documents/especificaciones/contrato_datos_v0.2.md` (si algo de la spec contradice el contrato, manda el contrato).

## 0. Cómo se trabaja (condiciones)

1. **Un punto = una fase = una aprobación manual.** Al terminar cada fase Code se detiene, presenta el informe de cierre (sección "Informe de cierre" abajo) y espera un mensaje explícito: `Aprobado Fase N`. No avanza sin él.
2. **Sin git de escritura.** Code no ejecuta `git add`, `commit`, `push`, `merge`, `checkout`, `reset` ni `stash`. Solo `git status`, `git diff`, `git log`. Los commits los haces tú al cerrar cada fase.
3. **Pruebas de humo obligatorias** sobre todo código nuevo, antes de pedir aprobación. Se ejecutan y se informan con conteos (pasa / falla). Si no se pueden ejecutar (p. ej. falta GPU), se dice y se prueba lo que sí se puede.
4. **Datos del cliente:** nunca imprimir textos de comentarios, cláusulas ni salidas de celdas del cuaderno. Solo conteos, nombres de columnas y valores de categorías. No tocar `data/raw/`. `data/processed/` y `data/etiquetas_manuales/` están en `.gitignore`: solo se leen.
5. **El cuaderno no se reescribe a mano ni se ejecuta completo** (entrenar U4 exige GPU/Colab). Se edita con `NotebookEdit` o con un script `nbformat` que conserve el resto de celdas intactas, y se valida con `nbformat.validate` + compilación de cada celda de código.
6. **No cambiar el modelo ni sus parámetros** salvo donde una fase lo pida. Las decisiones abiertas (spec §5) se dejan como parámetros con el valor de hoy.
7. **Los archivos de `data/utilities/` no se borran ni se sobrescriben sin aprobación.** Antes de modificar uno, Code deja una copia en `data/utilities/_respaldo/` (carpeta nueva; confirmar con `.gitignore` antes de que la veas en `git status`).

## 1. Diagnóstico de partida (lo que ya se verificó en la carpeta)

| Tema | Hallazgo |
|---|---|
| Estado git | `modulos_nlp.py` (8.010 líneas) y `Prisma_unificado.ipynb` están **sin seguimiento** (`??`). Sin un commit base no habrá `git diff` útil. |
| `data/processed/etapa1/` | **No existe** todavía: hay que correr `python run_etapa1.py` antes de la Fase 3. |
| `negaciones.xlsx` vs `negaciones_1.xlsx` | **Idénticos byte a byte** (mismo hash, hojas `negaciones` 44×5 y `_LEEME`). |
| `stopwords.xlsx` vs `stopwords_1.xlsx` | **Distintos.** `stopwords.xlsx` (hojas `base` 313 NLTK, `globales` 24, `particulares` 96 con claves antiguas `stopwords_evaluaciondocente…`, `adverbios` 70) lo lee también la **Etapa 3** (`src/limpieza/listas.py`, con esas claves y esas hojas, y las pruebas). `stopwords_1.xlsx` (`globales` 87, `particulares` 58 con claves del catálogo, `_adverbios_excluidos`, `_LEEME`) es la versión del NLP: mueve a `globales` 63 adverbios de relleno (60 de ellos están en la hoja `adverbios` de `stopwords.xlsx`) y renombra las claves a las del catálogo. Las palabras particulares son las mismas salvo 1 (`figri`). |
| Efecto actual | `RecursosLexicos._leer_stopwords` busca `stopword` y toma `stopwords.xlsx` (orden alfabético). Con las claves antiguas, **las particulares nunca coinciden** con `clave_stopwords` (p. ej. `evadocevaluacion`); y las hojas `base` y `adverbios` entran como "claves" falsas. |
| Taxonomía vs etiquetas (columna `aspecto_final_entrenamiento`, comparación sin tildes ni mayúsculas) | obs19: coincide (19 escrituras → 17 aspectos; difieren solo en mayúsculas). obs20: etiquetas = 17, taxonomía = 18 (**`Formación docente` sin etiquetas**; 64 filas vacías). obs23: 31 = 31, coincide. obs24: **`Vocación` (49 filas) está en las etiquetas y no en la taxonomía** (32 vs 31). obs25: taxonomía = 31, etiquetas = 30 (**`Avance progresivo` sin etiquetas**). Acreditación: hay **dos archivos**; el del 11-nov trae **`Ninguno de la lista` (12 filas)** y 79 vacías (20 vs 19); el del 14-nov coincide 19 = 19 y no tiene vacías. |
| Rutas del cuaderno | El cuaderno busca todo en `RAIZ` plano (léxicos, catálogo, 6 CSV, calibrador) y en carpetas que ya no existen: `finetuning_pruebas/etiquetas_manuales`, `src/datacleaning`, módulo `Mod_Bertopic_Ft`. El módulo hoy es `nlp_utilidades/modulos_nlp.py`. El modelo se guarda en `<raiz>/modelos/u4_aspectos/…` y la carpeta real es `models/modelos/`. |
| Perfiles obs23–obs25 | Heredan `col_respuesta="respuesta_completa_original"`, pero esas hojas traen `respuesta_completa` (solo obs19 usa `_original`). Además sus archivos **no contienen "auditado"** en el nombre, así que el patrón de búsqueda no los encuentra. |
| Alertas | `alertas.xlsx` existe en `data/utilities/` (112 términos) pero **solo tiene la columna `TERMINO`**; el contrato exige `categoria_alerta`. |

## 2. Fase 0 — Línea base (antes de tocar nada)

**Tú (manual):** haz un commit de línea base de `modulos_nlp.py` y `Prisma_unificado.ipynb` (y de `alertas.xlsx` si quieres versionarlo) en la rama actual. Si prefieres no hacerlo, Code deja copias `*.bak_fase0` junto a cada archivo.

**Code:**
1. Confirmar rama (`git branch --show-current`) y estado limpio de lo que se va a tocar.
2. `poetry run pytest` → debe dar **98 aprobadas** (spec, instrucciones de GitHub). Anotar el resultado como línea base.
3. Comprobar que `nlp_utilidades.modulos_nlp` se puede importar en este entorno; anotar qué dependencias pesadas faltan (torch, sentence-transformers, hdbscan, umap, bertopic, spacy). Si faltan, las pruebas de humo de las fases siguientes deben usar funciones puras (pandas) y `pytest.importorskip` para lo demás.
4. Verificar que `data/raw/` tiene los archivos que espera `run_etapa1.py` (ojo: el catálogo declara `Encuestas_Autoevaluación.xlsx` y en la carpeta está `Encuestas Autoevaluación--calidad.xlsx`; informar, no corregir).
5. Inventario del uso de `raiz`/`RAIZ` y de `doc_id` en `modulos_nlp.py` y en el cuaderno (solo conteos y líneas), para dimensionar las fases 2–4.

**Pruebas de humo:** `pytest` (98), `python -c "import nlp_utilidades.modulos_nlp"`, `python run_validar_taxonomia.py`.
**Aprobación:** informe de línea base. No hay cambios de código en esta fase.

## 3. Fase 1 — Conciliación de los Excel de `data/utilities`

Objetivo: un solo archivo por insumo y una taxonomía que cubra todas las etiquetas manuales.

### 1.1 `negaciones.xlsx` / `negaciones_1.xlsx`
- Code demuestra la igualdad (hash + `DataFrame.equals` de ambas hojas) y propone **conservar `negaciones.xlsx`** (es el compartido con la Etapa 3; el contrato prohíbe cambiar columnas o valores de TIPO sin versión nueva).
- `negaciones_1.xlsx` queda **marcado para retiro**: Code lo mueve a `data/utilities/_respaldo/` solo con tu aprobación (no se borra).

### 1.2 `stopwords.xlsx` / `stopwords_1.xlsx` — decisión tuya antes de editar
Code entrega un informe de diferencias (palabras, no textos de clientes): palabras solo en `_1`, solo en `stopwords.xlsx`, claves por hoja, y qué claves resuelve hoy cada lector (NLP y Etapa 3). Opciones:

| Opción | Qué hace | Riesgo |
|---|---|---|
| **A (recomendada)** | `stopwords.xlsx` queda como único archivo y **no cambia su estructura** (la Etapa 3 sigue igual). El lector del NLP (`_leer_stopwords`) pasa a leer sus hojas explícitamente (`base`, `globales`, `particulares`, `adverbios` con `valora = no`), traduce las claves antiguas a las del catálogo con un mapa documentado y reproduce lo que hacía `stopwords_1`. Las pocas palabras que existan solo en `_1` se agregan a `stopwords.xlsx` solo si las apruebas. `stopwords_1.xlsx` se retira. | Bajo; hay que confirmar que agregar palabras a `globales` no cambia los n-gramas de la Etapa 3 (se prueba con `pytest`). |
| B | Adoptar la estructura de `stopwords_1.xlsx` como oficial y adaptar `src/limpieza/listas.py` (mapa `PARTICULARES_POR_ENCUESTA`, hojas, pruebas). | Toca la Etapa 3 y su contrato; requiere acuerdo del equipo. |
| C | Dos archivos con nombres claros (p. ej. `stopwords.xlsx` para Etapa 3 y `stopwords_nlp.xlsx`). | Mantiene la duplicación que se quiere eliminar. |

### 1.3 `taxonomia.xlsx` vs etiquetas únicas de `data/etiquetas_manuales`
Code genera una tabla de conciliación por pregunta (etiqueta → ¿en taxonomía? → nº de filas; solo conteos) y propone, **sin editar** hasta que decidas cada caso:
- `Vocación` (obs24, 49 filas): ¿se agrega a la taxonomía (nuevo `aspecto_id`, ¿compartido con obs23/obs25?) o se reasigna en las etiquetas?
- `Ninguno de la lista` (acreditación 11-nov, 12 filas): ¿aspecto real o equivale a `SIN_ASPECTO`? Y **¿cuál de los dos archivos de acreditación es el vigente?** (el cuaderno apunta hoy a la hoja `acreditacion_formulada` del 11-nov; el del 14-nov es más nuevo).
- `Formación docente` (obs20) y `Avance progresivo` (obs25): están en la taxonomía sin etiquetas; avisar al entrenar (el clasificador no los aprenderá).
- Filas sin etiqueta (obs20: 64; acreditación 11-nov: 79): confirmar que se excluyen del entrenamiento.
- Normalización de escritura (`Relación estudiantes`/`Relación Estudiantes`, `opinión general`/`Opinión general`): confirmar que el entrenamiento ya las unifica (sin tildes ni mayúsculas) y documentarlo.

Tras tu decisión, Code edita `taxonomia.xlsx` (con respaldo) y corre `python run_validar_taxonomia.py`.

### Pruebas de humo (Fase 1)
- Nuevo `tests/test_utilities_conciliacion.py`: `leer_taxonomia` sin errores; todos los `aspecto_id` válidos; negaciones con las 5 columnas y los 5 TIPO; claves de stopwords resuelven para las 4 encuestas del catálogo; `cargar_listas` de la Etapa 3 sigue pasando; cada etiqueta única (normalizada) de cada archivo manual está en la taxonomía de su pregunta (`pytest.skip` si `data/etiquetas_manuales/` no existe).
- `RecursosLexicos.cargar(data/utilities)` carga sin avisos de claves sin coincidencia y devuelve conteos esperados.
- `poetry run pytest` completo: **98 + nuevas**, sin regresiones.

**Aprobación Fase 1:** tabla de conciliación, decisión A/B/C aplicada y cambios hechos en los Excel.

## 4. Fase 2 — Rutas del cuaderno y del módulo

Objetivo: que todo apunte a la estructura real, tanto local como en Colab (Drive con la misma estructura del repo).

| Qué | Hoy | Debe quedar |
|---|---|---|
| Módulo | `import Mod_Bertopic_Ft as ft`, buscado en `RAIZ` y `RAIZ/src` | `sys.path` con `RAIZ/src` e `from nlp_utilidades import modulos_nlp as ft`; revisar `ft.VERSION`/`HUELLA_ARCHIVO` y los `assert` (hoy exigen `ft-r4`) |
| `sys.path` | `RAIZ`, `RAIZ/src`, `RAIZ/src/datacleaning` | `RAIZ/src` |
| `RAIZ_COLAB` | `…/--topicos-unificado` | Carpeta del proyecto en Drive (`RAIZ_COLAB = "/content/drive/MyDrive/Maestria_MINE/SEMESTRE 4/3.OpcionGrado--Wilmer/capstonenlpencuestas_mine9"`); local sigue detectándose por `pyproject.toml` |
| Léxicos (`DIR_LEXICOS`) | `RAIZ` | `RAIZ/data/utilities` |
| Catálogo | `buscar_archivo(RAIZ, "preguntas_abiertas")` | `RAIZ/data/utilities/preguntas_abiertas.xlsx` |
| Calibrador | `RAIZ/calibrador_sentimiento.json` | `RAIZ/src/nlp_utilidades/calibrador_sentimiento.json` |
| Etiquetas manuales | `carpeta_etiquetas="finetuning_pruebas/etiquetas_manuales"` + patrón `*auditado*` | `data/etiquetas_manuales` con **`archivo_etiquetas` explícito por perfil** (los de obs23–25 no llevan "auditado") |
| Perfiles obs23–obs25 | `col_respuesta="respuesta_completa_original"` | `respuesta_completa` (confirmado en las hojas) |
| Acreditación | hoja `acreditacion_formulada` (11-nov) | archivo/hoja que decidas en la Fase 1 |
| Modelos U4 | `<raiz>/modelos/u4_aspectos/…` | `models/modelos/u4_aspectos/…`: separar `raiz` (datos) de `raiz_modelos` en las funciones de `modulos_nlp.py` que hoy usan un único `raiz`, o pasar `RAIZ/"models"` solo donde corresponda (Code decide tras el inventario de la Fase 0 y lo muestra antes de editar) |
| Caché de embeddings | `RAIZ/cache/embeddings_ft` | Dentro de `models/` (ya ignorado por git), p. ej. `models/cache/embeddings_ft` |
| Salidas | `RAIZ/outputs` | `RAIZ/outputs` (existe); las 5 tablas del contrato irán a `data/processed/nlp/` en la Fase 4 |
| Léxicos descritos | markdown del Paso 0a/1 menciona `accents.xlsx` y "un único juego en RAIZ" | Actualizar el texto: `accents.xlsx` ya no se usa (decisión 16-nov; hoy cae a `unicodedata`) y los léxicos viven en `data/utilities` |

Además: una sola celda con **todas** las rutas (bloque `RUTAS`), derivadas de `RAIZ`, con un `assert` por cada ruta de entrada que debe existir; quitar el doble montaje de Drive (celdas 3 y 7) sin romper Colab.

### Pruebas de humo (Fase 2)
- `nbformat.validate` + `ast.parse` de cada celda de código (ignorando líneas `%`/`!`).
- Script que ejecuta solo las celdas de configuración y rutas con `EN_COLAB = False` y verifica que **cada ruta existe** y que ya no queda ninguna referencia a `finetuning_pruebas`, `datacleaning`, `Mod_Bertopic_Ft`, `topicos-unificado` ni a archivos sueltos en `RAIZ`.
- `ft.resolver_archivo_etiquetas(...)` devuelve, para cada una de las 6 preguntas, un archivo existente (solo se imprime el nombre del archivo).
- `ft.RecursosLexicos.cargar(DIR_LEXICOS)` y `ft.cargar_taxonomias`/`leer_taxonomia` funcionan con las rutas nuevas.
- `pytest` completo sin regresiones.

**Aprobación Fase 2:** tabla "antes → después" de rutas y resultado de las pruebas.

## 5. Fase 3 — Ingesta desde la Etapa 1 (spec §2, tareas 1–2)

Objetivo: el cuaderno deja de leer los 6 CSV `{encuesta}__{clave}.csv` (separador `;`) y lee `respuestas.csv` de la corrida más reciente.

### 3.0 Prerrequisitos (spec "Antes de la tarea 1", pasos A–C) — decide si entran ahora
Cambiar de insumo no debe romper las etiquetas del cliente. Code verifica si el **corpus viejo de obs19** (24.380 respuestas, 25.714 cláusulas) sigue en el disco; hoy no aparece con ese nombre en `data/processed/`.
- **Paso A:** congelar etiquetas con su texto y llave en un archivo que no se regenera (p. ej. `data/etiquetas_manuales/congeladas/`, ignorado por git).
- **Paso B:** fijar versiones de spaCy y su modelo en `pyproject.toml` y las reglas de partición en un solo lugar; registrar `version_segmentacion`.
- **Paso C:** prueba de regresión (mismas 25.714 cláusulas, mismo orden). Si el corpus viejo no está, se informa y el paso queda **pendiente documentado**; no se inventa.
- **Paso D** (tabla de correspondencia cláusula vieja → `clausula_id` nuevo) se hace al final de la fase, con el insumo nuevo ya integrado.

### 3.1 Cambios
1. Ejecutar `python run_etapa1.py` (si no hay corrida). Usar `from preparacion.corridas import ultima_corrida` → `id_corrida, carpeta`.
2. Lectura: `pd.read_csv(ruta, dtype=str, keep_default_na=False, encoding="utf-8-sig")` de `respuestas.csv` y `catalogo_preguntas.csv` (verificar su ubicación real: la spec dice `data/processed/etapa1/`).
3. Filtrar `es_no_respuesta == "False"` y `pregunta_codigo == CLAVE_FOCAL`; comprobar que todas las filas traen el mismo `id_corrida_preparacion` y guardarlo para `metadatos_nlp`.
4. **Segmentar `texto_original`**, nunca un texto limpiado. Llevar `respuesta_id` en todo el flujo en lugar de `doc_id`.
5. Reemplazar `cargar_encuestas_institucion` (o añadir un adaptador `cargar_encuesta_etapa1(...)` que devuelva el mismo objeto `Encuesta` que consume el resto del cuaderno), de modo que el cambio aguas abajo sea mínimo. Reemplazar también las celdas "Paso 2.1 auditoría de columna de texto" (ya no hay `alias_redaccion` que resolver).
6. Parámetro `ruta_etapa1` en el bloque `RUTAS`; sin ruta quemada a la corrida.
7. Validar la cifra de referencia con `metadatos_preparacion.csv`: **158.731 respuestas, 60.621 con comentario; obs19 = 24.433** (spec). Informar diferencias por pregunta.

### Pruebas de humo (Fase 3)
- Prueba unitaria con un `respuestas.csv` sintético (3 preguntas, no-respuestas, un periodo `2025-1`, un valor `NA`) en un directorio temporal: el periodo sigue siendo texto, «NA» no se pierde, se filtran no-respuestas, falla con mensaje claro si no hay corridas o si hay dos `id_corrida_preparacion`.
- Con la corrida real: conteos por pregunta = `metadatos_preparacion.csv`; sin imprimir textos.
- El adaptador devuelve un objeto `Encuesta` con `respuesta_id` único y sin nulos; segmentar 20 respuestas sintéticas produce cláusulas que son fragmentos literales de `texto_original`.
- Pasos B/C si aplican; `pytest` completo.

**Aprobación Fase 3:** cifras por pregunta, resultado de la prueba de regresión (o su estado pendiente) y tabla de correspondencia de etiquetas (recuperadas exactas / en revisión).

## 6. Fase 4 — Salidas según el contrato v0.2 (spec §3, tareas 2–10)

Cinco CSV en `data/processed/nlp/` (`utf-8-sig`, coma, punto decimal, `True/False`, vacío para ausentes, nunca «NA»). Se **conservan** las salidas internas actuales en `outputs/` (zip, scorecard, auditoría); lo nuevo se suma.

| Archivo | Una fila = | Puntos clave |
|---|---|---|
| `clausulas.csv` | una cláusula | `clausula_id = respuesta_id\|orden`, `orden` desde 1 sin saltos (incluye sin opinión); `texto_clausula`; `es_opinion` (False si `estado = NO_OPINION`); `aspecto_id` (código de taxonomía; `SIN_ASPECTO`; `EMERGENTE_<periodo>_<n>`); `aspecto_origen` ∈ {clasificador, hdbscan, regla, sistema}; `confianza_aspecto`; `sentimiento` vacío si no es opinión; `confianza_sentimiento`; `version_modelo` (clasificador + sentimiento) |
| `auditoria_clasificacion.csv` | cláusula y corrida | `confianza_pos/neu/neg` antes del ajuste; top‑3 de las columnas `p_u4_*` traducidas a código; vacíos los 3 aspectos si no pasó por el clasificador; `visto_en_entrenamiento` |
| `alertas.csv` | un término encontrado | `alerta_id = respuesta_id\|n`; `termino`, `texto_coincidente`, `categoria_alerta`, `negada`. Una respuesta con alerta no se fragmenta: queda una cláusula (`orden = 1`) `SIN_ASPECTO` sin sentimiento |
| `catalogo_aspectos.csv` | un aspecto posible | taxonomía de las preguntas procesadas (`Aspectos definidos`) + emergentes (`Temas emergentes`, con `palabras_clave`) + `SIN_ASPECTO`; `orden` según la taxonomía |
| `metadatos_nlp.csv` | una pregunta procesada | `id_corrida_nlp = AAAAMMDD-HHMMSS_<huella>` (UTC; huella de insumos, taxonomía, alertas y versión del modelo); `id_corrida_preparacion`; `version_taxonomia`; `umbral_respaldo_hdbscan`; `accuracy_aspecto`; `metricas_sentimiento`; `regla_ajuste_sentimiento`; `huella_alertas`; conteos y porcentajes por origen |

Equivalencias de columnas (doc_id → respuesta_id, clausula → texto_clausula, asignacion → aspecto_origen, etc.) están en la spec §3; Code las implementa en **una sola función de construcción** y no renombra columnas internas del modelo.

### Decisiones abiertas → parámetros (valor de hoy), spec §5
`umbral_respaldo_hdbscan`, `regla_opinion_general` (palabras | contenido), `alerta_reemplaza_aspecto`, `alerta_negacion`, `regla_ajuste_sentimiento`. «Comentario General», «VALORACIÓN GLOBAL» y «ALERTA DE CASO ESPECIAL» quedan configurables (mapeo a `aspecto_id`/`aspecto_origen` en un diccionario, no en la lógica). Se implementa la alternativa solo donde sea barata.

### Puntos que requieren tu decisión durante la fase
- **`categoria_alerta`:** `alertas.xlsx` no tiene categorías. Propuesta: leer una columna `CATEGORIA` si existe y, si no, dejar el campo vacío y avisar; tú agregas las categorías al Excel.
- **`negada`:** hoy no se detecta por término; se deja el parámetro `alerta_negacion` y, si se activa, una ventana de N palabras con los NEGADOR de `negaciones.xlsx`.
- **Ubicación del código:** se escribe en `modulos_nlp.py` en una sección delimitada (`# === SALIDAS CONTRATO v0.2 ===`). Si prefieres un módulo aparte (`salidas_contrato.py`) por el tamaño del archivo, se dice antes de empezar.
- **Estabilidad (tarea 9):** semillas fijas y guardar UMAP/HDBSCAN con `prediction_data=True`; si el cambio es grande se propone dejarlo como subfase 4b.
- **Dependencias (tarea 11):** limpiar `pyproject.toml` y `poetry lock` se hace al final y con tu revisión (toca archivos fuera de los dos que indicaste).

### Validaciones del contrato dentro del pipeline (spec §6)
Función `validar_salidas_nlp(...)` que se detiene con mensaje claro: cobertura de respuestas (≥ 1 cláusula, ninguna no-respuesta), unicidad de IDs, `orden` desde 1 sin saltos, integridad referencial (`respuesta_id`, `aspecto_id`, `clausula_id`), dominios (`sentimiento`, `aspecto_origen`), confianzas en [0, 1] y suma de las tres = 1 ± 0,001, `sentimiento` vacío ⇔ `es_opinion = False`, `auditoria` con una fila por cláusula y aspectos ordenados, conteos de `metadatos_nlp` = archivos, `id_corrida_preparacion` registrado, determinismo (dos corridas, mismo hash de archivos) y relectura con `dtype=str`, `utf-8-sig`.

### Pruebas de humo (Fase 4)
- `tests/test_nlp_salidas.py` con un mini corpus sintético (≈ 10 respuestas: opinión, sin opinión, alerta, alerta negada, respuesta corta, una con 3 cláusulas) y probabilidades simuladas: se construyen los 5 CSV y pasa `validar_salidas_nlp`.
- Pruebas negativas: IDs duplicados, `orden` con salto, `aspecto_id` fuera del catálogo, confianzas que no suman 1 → la validación **falla** con el mensaje esperado.
- Determinismo: dos construcciones seguidas dan el mismo hash.
- Relectura de los CSV (`dtype=str`, `utf-8-sig`): periodo `2025-1` intacto, sin «NA», decimales con punto.
- Si hay GPU/Colab: una corrida real de **obs19** y comparación de conteos contra la spec (24.433 respuestas con comentario). Si no, se deja el procedimiento escrito en el informe.
- `pytest` completo.

**Aprobación Fase 4:** los 5 CSV de ejemplo (sintéticos), resultado de las validaciones y, si hubo corrida real, el resumen pedido por la spec: cláusulas por `aspecto_origen` (cantidad y %), cláusulas sin opinión, alertas y criterios no cumplidos.

## 7. Informe de cierre de cada fase (formato que Code debe entregar)

1. Qué cambió (archivos y celdas del cuaderno por número, nº de líneas).
2. Resultado de las pruebas de humo (comando → pasa/falla, conteos).
3. Decisiones tomadas y las que quedan abiertas.
4. Riesgos o supuestos.
5. Mensaje sugerido de commit (para que lo hagas tú).
6. Pregunta final: «¿Apruebas la Fase N para pasar a la N+1?».

## 8. Riesgos transversales

- **Etapa 3 comparte `stopwords.xlsx` y `negaciones.xlsx`**: cualquier cambio de estructura rompe sus pruebas (por eso la opción A de la Fase 1).
- **Corpus viejo de obs19 no localizado** → los pasos B–D de la spec pueden quedar pendientes y las 25.714 cláusulas no se podrán comparar.
- **Entrenar U4 exige GPU**: las pruebas locales son de humo, no de resultados.
- **Reproceso del histórico de obs19** lo aprueba el cliente (contrato 5.2); esta implementación no lo ejecuta.
- **Archivos con datos del cliente** (`data/etiquetas_manuales`, `data/raw`, `data/processed`): no se imprimen ni se suben.

## 9. Informes de cierre por fase

### Fase 0 — Línea base (2026-11-17)

**1. Qué cambió.** Ningún archivo de código ni de datos. Solo se creó el entorno: `poetry install --with dev` (autorizado), que instaló pytest, scikit-learn, scipy, gensim y wordcloud, y el proyecto `pln-encuestas 0.3.0`.

**2. Pruebas de humo**

| Comando | Resultado |
|---|---|
| `git branch --show-current` | `feature/contrato-etapa2-nlp` |
| `git status` | Solo `documents/prompt_claude_code_etapa2.md` modificado (ajeno a esta fase) |
| `poetry run pytest` | **98 aprobadas, 0 fallas** (24,5 s). Antes de instalar el entorno no corría (no había pytest). |
| `import nlp_utilidades.modulos_nlp` | Pasa (`VERSION = ft-r4.15`) |
| `python run_validar_taxonomia.py` | Pasa: versión `3e392f6ff0`, 147 filas, 66 aspectos |

Dependencias pesadas que faltan (no están en `pyproject`): torch, sentence-transformers, hdbscan, umap, bertopic, spacy, transformers. No hay GPU. `modulos_nlp` importa porque las carga de forma diferida; las pruebas de las fases siguientes usan funciones puras de pandas y `pytest.importorskip`.

**3. Diferencias con el plan (informadas, sin corregir)**
- `modulos_nlp.py` y el cuaderno ya están versionados (commit `35a693c`); no hacen falta copias `.bak_fase0`.
- No existen `data/processed/etapa1/` ni `data/processed/corpus_independientes/`; no hay corpus viejo de obs19 (pasos B–D de la Fase 3 quedarían pendientes). `models/modelos/` está vacía.
- `data/raw/` vs. nombres declarados en `preguntas_abiertas.xlsx` (hoja `archivos`): evaluación/autoevaluación docente existe con espacios en vez de guiones bajos; acreditación existe como `Encuestas Autoevaluación--calidad.xlsx`; `Consolidado_Preguntas_Abiertas.xlsx` (calidad) no aparece. `run_etapa1.py` podría no correr completo (no se probó).
- `data/utilities/_respaldo/` no está en `.gitignore`; hay que ignorarla antes de la Fase 1.

**4. Inventario (celdas del cuaderno en base 0; 137 celdas, 61 de código)**
- `modulos_nlp.py` (8.010 líneas): `raiz` 27, `RAIZ` 3, `doc_id` 37 (13 funciones; principales `construir_unidades` 18, `_verificar_union_clausulas_documentos` 10), `respuesta_id` 0. Funciones con parámetro `raiz`: `carpeta_modelos_u4`, `resolver_archivo_etiquetas`, `cargar_datos_u4`, `entrenar_u4`, `modelo_u4_de_clave`, `ejecutar_pipeline`.
- Cuaderno: `RAIZ` 43 ocurrencias en 18 celdas (1, 2, 4, 5, 7, 8, 10, 11, 13, 37, 43, 45, 47, 122, 127, 128, 129, 136); `raiz` 1 (127); `doc_id` 12 en celdas 43, 54, 56, 57, 122, 123. Rutas obsoletas: `finetuning_pruebas` (1, 4, 43), `datacleaning` (7), `Mod_Bertopic_Ft` (0, 4, 8, 37, 133), `topicos-unificado` (4), `respuesta_completa_original` (4), `accents` (2, 11, 13), `modelos` (43, 122, 127, 128, 131), `cache` (7, 26, 70).

**5. Decisiones abiertas**
- Fase 2: confirmar la separación `raiz` (datos) / `raiz_modelos` (6 funciones).
- Aclarar los nombres de `data/raw/` antes de la Fase 3.

**6. Riesgos.** Sin corpus viejo ni modelos entrenados no hay regresión de las 25.714 cláusulas ni corrida real local. Etapa 1 puede no correr con los nombres actuales de `data/raw/`.

**7. Commit sugerido.** Sin cambios de código; solo este informe: `docs: informe de cierre Fase 0 NLP etapa 2 (línea base 98 pruebas)`.

**8.** ¿Apruebas la Fase 0 para pasar a la Fase 1?

### Fase 1 — Conciliación de los Excel de `data/utilities` (cierre, 2026-11-17)

**1. Qué cambió**
- `data/utilities/taxonomia.xlsx` (hoja `aspectos`, 147 → 146 filas, 66 aspectos): + `VOCACION` en obs24 (mismo nombre y descripción que en obs19/obs20; obs24 = 32); − `FORMACION_DOCENTE` en obs20 (= 17); − `AVANCE_PROGRESIVO` en obs25 (= 30). Nueva versión de taxonomía: `9517d60bf5` (antes `3e392f6ff0`).
- `data/utilities/stopwords.xlsx`: hojas y columnas intactas. `globales` 24 → 28 (`dos`, `tres`, `queda`, `quedan`); `particulares` 96 → 101 (`figri` en las 5 claves `stopwords_calidad*`).
- `negaciones_1.xlsx` y `stopwords_1.xlsx` movidos (no borrados) a `data/utilities/_respaldo/`, junto con copias de los originales (`taxonomia.original.xlsx`, `stopwords.original.xlsx`). `negaciones.xlsx` sin cambios.
- `.gitignore`: + `data/utilities/_respaldo/`.
- `src/nlp_utilidades/modulos_nlp.py` (+73 / −19 líneas): constantes `CLAVES_PARTICULARES_NLP`, `ADVERBIOS_NO_STOPWORD_NLP`, `GLOBALES_SOLO_NLP` y nuevo `RecursosLexicos._leer_stopwords`: lee las hojas por nombre, toma `adverbios` con `valora = no`, traduce las claves antiguas a código de encuesta del catálogo (conserva también la clave antigua para el cuaderno actual) y reproduce exactamente las 87 globales de `stopwords_1`. El cuaderno no se tocó.
- Prueba nueva: `tests/test_utilities_conciliacion.py` (12 pruebas).

**2. Pruebas de humo**

| Comando | Resultado |
|---|---|
| `poetry run pytest` | **110 aprobadas** (98 de línea base + 12 nuevas), 0 fallas |
| `pytest tests/test_utilities_conciliacion.py` | 12 pasan: taxonomía válida y conteos por pregunta; negaciones 44×5 con los 5 TIPO; `_1` retirados; `cargar_listas` de la Etapa 3 con las 4 encuestas; `stopwords.xlsx` sin cambios de estructura y sin las 6 valorativas en `globales`; lector NLP (87 globales, claves del catálogo, sin `base`/`adverbios` como claves); cada etiqueta manual está en la taxonomía de su pregunta y cada aspecto tiene etiquetas |
| `python run_validar_taxonomia.py` | Pasa: 146 filas, 66 aspectos |
| `RecursosLexicos.cargar(data/utilities)` | 306 NLTK + 87 globales + 10 claves de encuesta; desaparece el aviso "2 coincidencias". Quedan 2 avisos esperados: `accents.xlsx` no encontrado (Fase 2) y sin particulares para acreditación (igual que la Etapa 3) |

**3. Decisiones tomadas**
1. Stopwords: **opción A**. El cuaderno no se edita por esto.
2. Extras de `_1`: agregar `dos`, `tres`, `queda`, `quedan` y `figri` al Excel. Las 6 valorativas (`bien`, `mal`, `mejor`, `nunca`, `peor`, `siempre`) y las 10 adverbios excluidos quedan como constantes del lector NLP.
3. `Vocación` en obs24: agregar a la taxonomía reutilizando `VOCACION`, solo en obs24.
4. Acreditación: archivo vigente = **14-nov**, hoja `acreditacion_etiquetas` (el del 11-nov queda sin uso; así `Ninguno de la lista` deja de ser problema).
5. `negaciones_1` y `stopwords_1` a `_respaldo/` y esa carpeta al `.gitignore`.
6. `Formación docente` (obs20) y `Avance progresivo` (obs25) se quitan de la taxonomía.
7. Se confirma que las filas sin etiqueta (obs20: 64) se excluyen y que las grafías distintas se unifican; ya lo hace `cargar_datos_u4` y queda cubierto por la prueba de etiquetas.

**4. Riesgos y supuestos**
- Agregar `dos`, `tres`, `queda`, `quedan` a `globales` y `figri` a las listas de calidad cambia también los n-gramas de la Etapa 3 (sus pruebas pasan, pero no se midió el efecto sobre datos reales). Reversible: `_respaldo/stopwords.original.xlsx`.
- En `stopwords_1`, las palabras de calidad estaban bajo la clave `autoevaluacion_acreditacion`; las traté como un error de nombre y las dejé en `calidadservicios`. Acreditación no tiene particulares, como en la Etapa 3.
- Calidad no tiene etiquetas ni taxonomía; su lista de stopwords es la unión de las 5 poblaciones.
- Quitar `FORMACION_DOCENTE` de obs20 y `AVANCE_PROGRESIVO` de obs25 deja las filas con etiquetas de obs24/obs23 en `AVANCE_PROGRESIVO` sin cambio. Si más adelante se pide ese aspecto en obs20/obs25, hay que reponerlo y reentrenar.
- Un modelo U4 ya entrenado con la taxonomía anterior (no hay ninguno en `models/modelos/`) quedaría desalineado con `comparar_con_clases`.
- Las rutas del cuaderno al archivo de acreditación (11-nov) se corrigen en la Fase 2.

**5. Commit sugerido (lo haces tú):**
`feat(nlp): conciliación de utilities Fase 1 — taxonomía (+VOCACION obs24, −2 sin etiquetas), stopwords opción A y lector NLP` (incluye `.gitignore`, `data/utilities/*.xlsx`, `modulos_nlp.py`, `tests/test_utilities_conciliacion.py` y este plan).

**6.** ¿Apruebas la Fase 1 para pasar a la Fase 2?
