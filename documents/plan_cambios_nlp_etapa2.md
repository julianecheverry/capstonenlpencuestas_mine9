# Plan de cambios al NLP (Etapa 2) — historial

Este archivo conserva **todas las versiones del plan**, en orden cronológico. No se sobrescribe: cada versión nueva se agrega al final.

| Parte | Fecha | Alcance | Estado |
|---|---|---|---|
| 1. Plan v1 | 2026-11-17 | Fases 0–4: línea base, conciliación de utilities, rutas, ingesta desde la Etapa 1, salidas del contrato v0.2 | Cerrado (commits `f240136`, `bb4ce40`, `7a65396`, `0cd65bc`) |
| 2. Plan v2 | 2026-11-19 | Puntos 0–7: alertas pasan a la Etapa 1, ingesta filtrada, ruta de salida, ajustes a catálogo/cláusulas/respuestas, unificación de categorías generales | Vigente |

Prompts para Claude Code: `prompt_claude_code_etapa2.md` (v1) y `prompt_claude_code_etapa2_v2.md` (v2).

---

# PARTE 1 — Plan v1 (2026-11-17) · cerrado

> Texto original del 17-nov tal como quedó en el commit `0cd65bc`, sin cambios de contenido (solo se quitó su título de nivel 1, ahora es el encabezado de esta parte).

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

### Fase 2 — Rutas del cuaderno y del módulo (cierre, 2026-11-17)

**1. Qué cambió**
- `notebooks/Prisma_unificado.ipynb`: 20 de 137 celdas modificadas (0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 36, 37, 38, 43, 45, 122, 127, 128, 133; numeración base 0). El resto quedó idéntico a HEAD y el formato del archivo se conservó (diff de una sola línea).
- `src/nlp_utilidades/modulos_nlp.py` (+52 / −30 líneas): `cargar_taxonomias` lee la hoja `aspectos` (y conserva el formato antiguo); `ejecutar_pipeline` recibe `dir_lexicos` y `raiz_modelos` (por defecto `raiz`, compatible hacia atrás) y busca la taxonomía por clave de pregunta; `ParametrosU4.carpeta_etiquetas` = `data/etiquetas_manuales`; `Config.dir_cache` = `models/cache/embeddings_ft`; el aviso de `accents.xlsx` pasa a `info`; docstring de uso apunta a `nlp_utilidades.modulos_nlp`.
- Prueba nueva: `tests/test_nlp_rutas.py` (12 pruebas).

**Tabla antes → después**

| Qué | Antes | Después |
|---|---|---|
| Módulo (celdas 8, 133) | `import Mod_Bertopic_Ft as ft`, buscado en `RAIZ` y `RAIZ/src` | `from nlp_utilidades import modulos_nlp as ft`; `assert` de existencia de `RUTAS["modulo"]`; los `assert` de `VERSION` (`ft-r4`) y de huella se mantienen |
| `sys.path` (celda 7) | `RAIZ`, `RAIZ/src`, `RAIZ/src/datacleaning` | solo `RAIZ/src` |
| `RAIZ_COLAB` (celda 4) | `…/--topicos-unificado` | `…/3.OpcionGrado--Wilmer/capstonenlpencuestas_mine9` |
| Drive (celdas 3 y 7) | montaje doble (el de la celda 3 no estaba protegido) | un solo montaje, en la celda 3 y solo si estamos en Colab |
| Bloque de rutas | repartido en celdas 4, 7, 11 | **un solo bloque `RUTAS` en la celda 7**, derivado de `RAIZ`, con `assert` por cada entrada obligatoria; la celda 11 ya no define rutas |
| Léxicos y catálogo | `RAIZ` y `buscar_archivo(RAIZ, …)` | `RAIZ/data/utilities` (`DIR_LEXICOS`, `RUTA_CATALOGO`); `assert` de los 5 léxicos obligatorios |
| Calibrador | `RAIZ/calibrador_sentimiento.json` | `RAIZ/src/nlp_utilidades/calibrador_sentimiento.json` (opcional, lo crea el Paso 16) |
| Etiquetas | `finetuning_pruebas/etiquetas_manuales` + patrón `*auditado*` | `data/etiquetas_manuales` con `archivo_etiquetas` explícito por perfil (obs23–25 ya no dependen del patrón) |
| `col_respuesta` obs23–25 | `respuesta_completa_original` (heredado) | `respuesta_completa` (default del perfil base; solo obs19 conserva `_original`) |
| Acreditación | 11-nov, hoja `acreditacion_formulada` | 14-nov, hoja `acreditacion_etiquetas` (decisión Fase 1) |
| Modelos U4 | `<RAIZ>/modelos/u4_aspectos/…` | `RAIZ/models/modelos/u4_aspectos/…`: el cuaderno pasa `RAIZ_MODELOS` a `entrenar_u4` y `modelo_u4_de_clave`; `params_u4()` entrega las rutas de etiquetas ya absolutas; `ejecutar_pipeline(…, raiz_modelos=RAIZ_MODELOS)` |
| Caché de embeddings | `RAIZ/cache/embeddings_ft` | `models/cache/embeddings_ft` (ignorada por git) |
| Salidas | `RAIZ/outputs` | `RAIZ/outputs` (sin cambio) |
| Taxonomía, Paso 3.1 (celdas 36–38) | hoja `taxonomia` (ENCUESTA/CODIGO/DESCRIPCION) por `encuesta.hoja` | hoja `aspectos` por clave de pregunta; **en modo `u4` los pasos 3.1 se omiten**, en `bertopic` se exige taxonomía de cada pregunta activa |
| Texto descriptivo | mencionaba `accents.xlsx`, `finetuning_pruebas`, "un único juego en `RAIZ`" | actualizado a `data/utilities` y a `accents.xlsx` fuera de uso |

**2. Pruebas de humo**

| Comando | Resultado |
|---|---|
| `poetry run pytest` | **122 aprobadas** (110 previas + 12 nuevas), 0 fallas |
| `nbformat.validate` | Pasa (137 celdas) |
| `ast.parse` de las 61 celdas de código (sin líneas `%`/`!`) | Pasa. La celda 6 (`!pip` dentro de `if EN_COLAB`) solo compila si esas líneas se sustituyen por `pass`; es anterior a esta fase |
| Celdas 4, 7, 8 y 11 ejecutadas con `EN_COLAB = False` para las 6 preguntas | Todas las `RUTAS` existen; el `assert` de etiquetas pasa en las 6 |
| `ft.resolver_archivo_etiquetas(RAIZ_MODELOS, …)` para las 6 preguntas | Devuelve un archivo existente en `data/etiquetas_manuales` en las 6 (hoja y columna de respuesta correctas) |
| `ft.carpeta_modelos_u4(RAIZ_MODELOS, clave)` | `models/modelos/u4_aspectos/<clave>` |
| `ft.RecursosLexicos.cargar(DIR_LEXICOS)` | 87 globales, 10 claves de encuesta |
| `ft.cargar_taxonomias(DIR_LEXICOS)` | 17 / 17 / 31 / 32 / 30 / 19 aspectos (obs19 / obs20 / obs23 / obs24 / obs25 / acreditación) |
| Celdas 37–38 | Modo `u4`: se omiten sin error. Modo `bertopic`: cargan `cfg.taxonomia` (19 códigos para acreditación) |
| Referencias viejas en el cuaderno (`finetuning_pruebas`, `datacleaning`, `Mod_Bertopic_Ft`, `topicos-unificado`) | 0 (`tests/test_nlp_rutas.py`) |

Lo que **no** se pudo probar aquí: ejecución real en Colab (montaje de Drive, `pip`), entrenamiento/etiquetado U4 y `ejecutar_pipeline` completo (faltan torch, sentence-transformers, spaCy, hdbscan, umap, bertopic y GPU). Para el montaje de Colab solo se revisó el código.

**3. Decisiones**
- Taxonomía (Paso 3.1): se implementaron las dos opciones: lectura de la hoja `aspectos` y omisión del paso en modo `u4`.
- Rutas de modelos: se adaptó el cuaderno a `models/modelos/u4_aspectos`. Para no cambiar firmas existentes, `entrenar_u4` y `modelo_u4_de_clave` reciben `RAIZ_MODELOS` como `raiz` y las etiquetas se pasan con ruta absoluta (`params_u4`). Solo `ejecutar_pipeline` ganó dos parámetros opcionales (`dir_lexicos`, `raiz_modelos`), porque necesita leer léxicos y modelos de carpetas distintas.
- Quedan abiertas (Fase 3): la lectura de datos (`cargar_encuestas_institucion` con los 6 CSV; hoy `CARPETA_DATOS_FT` apunta a `data/processed`) y el parámetro `ruta_etapa1` en `RUTAS`.

**4. Riesgos y supuestos**
- La celda 7 exige que existan los 5 léxicos y, en modo `u4`, el archivo de etiquetas de `CLAVE_FOCAL`; en Colab hay que tener `data/etiquetas_manuales` en Drive (está fuera de git).
- `RAIZ_COLAB` apunta a una carpeta de Drive con la misma estructura que el repo; no se pudo verificar que exista.
- `GUARDAR_EN_DRIVE = False` en Colab ahora lee todo de `RAIZ_COLAB` y solo envía `outputs` y la caché a `/content`.
- `ParametrosU4.desde` y las demás funciones del módulo siguen resolviendo `archivo_etiquetas` relativo a `raiz` si se les pasa una ruta relativa (compatibilidad); el cuaderno ya las pasa absolutas.
- La tabla de aspectos de las celdas 36–38 solo aplica a `bertopic`; en `u4` la comparación con las clases del modelo se hará en la Fase 4 (`comparar_con_clases`).

**5. Commit sugerido (lo haces tú)**
`feat(nlp): rutas de Prisma_unificado y modulos_nlp a la estructura del repo (Fase 2) — bloque RUTAS, etiquetas por perfil, models/modelos, taxonomía hoja aspectos`
(incluye `notebooks/Prisma_unificado.ipynb`, `src/nlp_utilidades/modulos_nlp.py`, `tests/test_nlp_rutas.py` y este plan).

**6.** ¿Apruebas la Fase 2 para pasar a la Fase 3?

### Fase 3 — Ingesta desde la Etapa 1 (cierre, 2026-11-17)

**1. Qué cambió**
- `src/nlp_utilidades/modulos_nlp.py` (+149 / −2 líneas): nueva sección «5b. Ingesta desde la Etapa 1» con `leer_respuestas_etapa1`, `cargar_encuestas_etapa1` y `comparar_con_metadatos_etapa1`; `Encuesta` gana el campo opcional `id_corrida_preparacion`; `ejecutar_pipeline` recibe `carpeta_etapa1` (tiene prioridad sobre `ruta_catalogo`/`carpeta_datos`). `cargar_encuestas_institucion` (los 6 CSV) se conserva sin cambios, ya sin uso en el cuaderno.
- `notebooks/Prisma_unificado.ipynb`: 12 de 137 celdas modificadas (7, 10, 11, 12, 15, 16, 17, 21, 22, 23, 24, 127; base 0). `RUTAS['etapa1']` (parámetro `ruta_etapa1`, alias `RUTA_ETAPA1`) reemplaza a `datos_csv`; la celda 7 toma la corrida con `ultima_corrida` (`ID_CORRIDA_PREPARACION`) y falla con mensaje claro si no hay ninguna; la celda 16 lee con `cargar_encuestas_etapa1`; las celdas 21–24 (antes «auditoría de columna de texto», con `alias_redaccion`) pasan a «auditoría de la ingesta» y guardan `auditoria_ingesta_etapa1.csv`; la celda 127 llama a `ejecutar_pipeline(carpeta_etapa1=RUTA_ETAPA1)`.
- `data/utilities/preguntas_abiertas.xlsx`: una fila nueva en la hoja `archivos` (`autoevaluacion_acreditacion` → `Encuestas Autoevaluación--calidad.xlsx`) para que la Etapa 1 encuentre el archivo real. Copia del original en `data/utilities/_respaldo/preguntas_abiertas.original.xlsx`. **Este Excel también lo lee la Etapa 1: el cambio va en tu commit.**
- Corrida real generada con `python run_etapa1.py autoevaluacion_acreditacion`: `data/processed/etapa1/20261117-211304_3d69efc5d6/` (fuera de git) y su informe en `reportes/` (fuera de git).
- Pruebas: `tests/test_nlp_ingesta.py` (13 nuevas) y ajuste de `tests/test_nlp_rutas.py` (se omite si no hay corrida de la Etapa 1).

**2. Pruebas de humo**

| Comando | Resultado |
|---|---|
| `poetry run pytest` | **135 aprobadas** (122 previas + 13 nuevas), 0 fallas |
| Corrida sintética (3 preguntas, no-respuestas, periodo `2025-1`, valor `NA`, ceros a la izquierda) | El periodo sigue siendo texto, `NA` se conserva, las no-respuestas se filtran, `respuesta_id` único y sin vacíos |
| Errores esperados | Sin corridas (`No hay corridas de la Etapa 1… run_etapa1.py`), dos `id_corrida_preparacion`, `es_no_respuesta` fuera de True/False, `respuesta_id` repetido o vacío, pregunta fuera del catálogo y columna faltante: todos fallan con mensaje claro |
| Corrida más reciente | Con dos carpetas toma la de id mayor |
| 20 respuestas sintéticas segmentadas (`Segmentador` por reglas, sin spaCy) | 32 cláusulas; todas son fragmentos literales de `texto_original`; `respuesta_id` único en `documentos` y recuperable en `clausulas` por `doc_id` |
| Corrida real (acreditación) | 9.289 respuestas, 1.277 con comentario = `metadatos_preparacion.csv` (diferencia 0); `respuesta_id` único; periodos `2024-1`…`2026-1` como texto |
| Celdas 4, 7, 8, 11, 14, 16, 18, 19 y 22 ejecutadas con la corrida real | Sin errores; auditoría de la ingesta: `diferencia = 0` y `respuesta_id_unicos = True` |
| `nbformat.validate` y `ast.parse` de las celdas de código | Pasan |

No se pudo probar aquí: segmentación con spaCy (`es_core_news_lg` no está instalado) y el resto del pipeline sobre estos datos.

**3. Decisiones**
- Se mantiene `doc_id` (posición interna) como llave de unión del modelo y `respuesta_id` viaja como llave en `Encuesta.marco` y en `documentos`; `clausulas` la recibe uniendo por `doc_id`. No se agregó la columna a `clausulas` para no romper las uniones internas (`_verificar_union_clausulas_documentos`, exportaciones). La función única de la Fase 4 hace ese cruce y arma `clausula_id = respuesta_id|orden`.
- `catalogo_preguntas.csv` está dentro de la carpeta de cada corrida (no en `data/processed/etapa1/` como dice la spec); el adaptador busca primero allí y luego en la carpeta padre.
- Archivo de acreditación: se usó `Encuestas Autoevaluación--calidad.xlsx` (hoja `Autoevaluación`, columna `FNACNAANAi181i.`, pregunta 34 del diccionario) por instrucción tuya; es la pregunta abierta de acreditación.

**4. Pasos previos A–D de la spec: PENDIENTES**
- El corpus viejo de obs19 (24.380 respuestas / 25.714 cláusulas) no está en el disco, así que no hay regresión (C) ni tabla de correspondencia de etiquetas (D, «recuperadas exactas / en revisión»).
- Paso A (congelar etiquetas con llave): los archivos de `data/etiquetas_manuales` traen solo `id`, la respuesta, el fragmento y el aspecto; **no traen la llave** (Periodo, ID de respuesta, Código Cruce). Sin el corpus viejo no se puede reconstruir.
- Paso B (fijar spaCy y su modelo en `pyproject.toml`, `version_segmentacion`): `pyproject.toml` está fuera de los archivos autorizados y spaCy no está instalado aquí; queda para la tarea 11.

**5. Cifras de referencia de la spec: solo parcialmente verificables**
- Verificado: acreditación = 1.277 con comentario (coincide con `metadatos_preparacion.csv`).
- No verificable: 158.731 respuestas / 60.621 con comentario en total y obs19 = 24.433. La Etapa 1 no corrió para evaluación/autoevaluación docente porque el archivo real se llama `Consolidado Histórico Evaluación Docente Anonimizado.xlsx` (con espacios) y el catálogo registra `Consolidado_Histórico_…` (guiones bajos); calidad no tiene archivo en `data/raw/`. Con una fila de alias más en la hoja `archivos` (mismo arreglo) se correría y se compararía obs19.

**6. Riesgos**
- La celda 7 ahora exige una corrida en `data/processed/etapa1` (fuera de git): cada persona debe correr `python run_etapa1.py` o copiar una corrida; en Colab, subir `data/processed/etapa1` a Drive.
- Para CLAVE_FOCAL de evaluación docente (obs19, obs20, obs23, obs24, obs25) el cuaderno falla en el Paso 2 hasta que exista una corrida que incluya esas preguntas.
- Las cláusulas del segmentador por reglas se comprobaron como literales; con spaCy (producción) la spec señala 129 cláusulas no literales por normalización: sin verificar aquí.
- Etapa 1 solo se corrió para acreditación: la corrida más reciente contiene únicamente esa encuesta.

**7. Commit sugerido (lo haces tú)**
`feat(nlp): ingesta desde la Etapa 1 (Fase 3) — cargar_encuestas_etapa1, respuesta_id como llave, auditoría de la ingesta, alias de archivo de acreditación`
(incluye `src/nlp_utilidades/modulos_nlp.py`, `notebooks/Prisma_unificado.ipynb`, `data/utilities/preguntas_abiertas.xlsx`, `tests/test_nlp_ingesta.py`, `tests/test_nlp_rutas.py` y este plan).

**8.** ¿Apruebas la Fase 3 para pasar a la Fase 4?

### Fase 4 — Salidas según el contrato v0.2 (cierre, 2026-11-17)

**1. Qué cambió**
- **Nuevo `src/nlp_utilidades/salidas_contrato.py`** (módulo aparte, por tu decisión; pandas puro, sin torch/spaCy/UMAP): `construir_salidas_contrato` (una sola función de construcción), `ParametrosSalidas` (decisiones abiertas como parámetros), `leer_alertas`, `leer_negadores`, `detectar_alertas`, `entrada_desde_resultado`, `escribir_salidas_contrato`, `leer_salidas_contrato`, `huella_archivos` y `validar_salidas_nlp` (`ErrorSalidasNLP`).
- **`src/nlp_utilidades/modulos_nlp.py`** (tarea 9, estabilidad): `ModeloResidual`, `ajustar_residual_persistible`, `guardar_modelo_residual` y `cargar_modelo_residual`; `modelar_residual_u4` ajusta UMAP/HDBSCAN con `prediction_data=True` (misma semilla y parámetros) y añade la columna `p_residual` (pertenencia al grupo); `EtapasU4.residual_persistible`. Un fallo del reajuste solo se registra, no detiene la corrida. Las columnas internas del modelo no se renombraron.
- **Cuaderno:** celda 1 (tabla de pasos), 4 (`SALIDAS_CONTRATO` con las decisiones abiertas), 7 (`RUTAS['nlp']`) y **tres celdas nuevas, 124–126 («Paso 12b»)**. Por esa inserción, las celdas que antes eran 124–137 pasan a 127–140 (la celda 127 de las Fases 2 y 3 es ahora la 130 y la 133 es la 136).
- Pruebas nuevas: `tests/test_nlp_salidas.py` (28), `tests/test_nlp_residual_persistible.py` (5).
- Ejemplo sintético de las 5 tablas: `documents/ejemplos_salidas_nlp/` (13 cláusulas; sin datos del cliente).

**2. Pruebas de humo**

| Comando / prueba | Resultado |
|---|---|
| `poetry run pytest` | **168 aprobadas** (135 previas + 33 nuevas), 0 fallas |
| Mini corpus sintético (opinión, sin opinión, alerta, alerta negada, respuesta corta, 3 cláusulas, residual con y sin grupo, «NA» como comentario) | Se construyen los 5 CSV, se releen con `dtype=str`/`utf-8-sig` y pasan `validar_salidas_nlp` |
| Pruebas negativas (IDs de cláusula y alerta duplicados, `orden` con salto, `aspecto_id` fuera del catálogo, confianzas que no suman 1 o fuera de [0, 1], cobertura y no-respuestas con cláusulas, sentimiento distinto de es_opinion, `aspecto_origen` inválido, `id_corrida_preparacion` distinto, taxonomía distinta de las clases, auditoría desordenada, conteos de metadatos) | La validación **falla** con el mensaje esperado en todas |
| Determinismo | Dos construcciones con la misma fecha dan el mismo hash de los 5 archivos; cambiar la fecha cambia el id y el hash |
| Formato | BOM utf-8-sig, punto decimal, periodo `2025-1` intacto, sin «NA»/«nan»/«None» (salvo «NA» como texto de un comentario real) |
| Alternativas no implementadas | `regla_opinion_general = "contenido"` y `alerta_reemplaza_aspecto = False` lanzan `NotImplementedError` con mensaje claro |
| Integración con la corrida real de la Etapa 1 (acreditación, 1.277 respuestas con comentario) + cláusulas y probabilidades **sintéticas**; se ejecutó la celda 125 del cuaderno | 1.277 cláusulas, 1.277 filas de auditoría, 14 alertas (13 respuestas, usando `alertas.xlsx` real), catálogo de 20 aspectos, 1 fila de metadatos; pasa la validación contra `respuestas.csv` real; los archivos de prueba se borraron |
| Tarea 9 con dobles de UMAP/HDBSCAN | Mapa de grupos, probabilidades, guardado/carga y `asignar` verificados; ver «No probado» |

**No probado aquí (declarado):** corrida real de obs19 (faltan torch, sentence-transformers, spaCy, UMAP, HDBSCAN, BERTopic, GPU y el modelo U4 entrenado), por lo que no se compararon las cifras de la spec (24.433 respuestas). La tarea 9 con las librerías reales (UMAP/HDBSCAN con `prediction_data=True` y `approximate_predict`) y la columna `p_residual` dentro de `modelar_residual_u4` quedan sin ejecutar.

**Procedimiento para Colab:** (a) `python run_etapa1.py` incluyendo evadoc (ver Fase 3, punto 5); (b) `CLAVE_FOCAL = "obs19_aspectos_resaltar"` y «Ejecutar todo» hasta el Paso 12b; (c) comparar `n_respuestas` de `metadatos_nlp.csv` con 24.433, revisar el resumen del Paso 12b (cláusulas por `aspecto_origen`, sin opinión, alertas), confirmar que imprime «Residual guardado» y cargar el `.joblib` para llamar a `ModeloResidual.asignar` con unos embeddings nuevos.

**3. Decisiones tomadas**
- Ubicación: módulo aparte `salidas_contrato.py` (fuera de los dos archivos autorizados; autorizado por ti).
- «Comentario General» y «VALORACIÓN GLOBAL» → `OPINION_GENERAL` con `aspecto_origen = regla` (existe en las 6 taxonomías; diccionario `MAPEO_ASIGNACION_DEFECTO`, configurable). **Cambia las cifras de hoy:** ambas categorías se funden en una.
- Alertas: `categoria_alerta` sale de la columna `CATEGORIA` de `alertas.xlsx` (hoy no existe: queda vacía y se avisa); `alerta_negacion = False` (valor de hoy, `negada = False` en todas) con la ventana de palabras sobre los NEGADOR ya implementada y activable con el parámetro; una fila por término encontrado (un término contenido dentro de otro más largo, «acoso» en «acoso laboral», no cuenta aparte).
- Estabilidad (tarea 9): incluida ahora, sin poder probarla con las librerías reales.
- Salidas en `data/processed/nlp/<id_corrida_nlp>/` (una carpeta por corrida; no sobrescribe).
- Las respuestas con alerta quedan con **una sola cláusula `SIN_ASPECTO`, `es_opinion = False` y sin sentimiento** (contrato 5.2); hoy el pipeline les fuerza NEG: esa etiqueta interna no se publica.
- Valores de las decisiones abiertas (spec §5): `umbral_respaldo_hdbscan` = `Config.umbral_residual_u4` (0,10 en los perfiles), `regla_opinion_general` = palabras, `alerta_reemplaza_aspecto` = True, `alerta_negacion` = False, `regla_ajuste_sentimiento` = descripción automática del ajuste actual (margen NEU o calibrador).

**4. Supuestos y riesgos a revisar**
- `confianza_aspecto` queda **vacía** en las cláusulas de origen `regla` y `sistema` (el contrato la marca obligatoria pero no define su valor para ellas); se exige para `clasificador`. Para `hdbscan` es la pertenencia al grupo (`p_residual`); si el modelo persistible no se pudo ajustar, queda vacía y se avisa.
- Una cláusula residual que HDBSCAN dejó sin grupo se publica como `SIN_ASPECTO` con origen `sistema`.
- `EMERGENTE_<periodo>_<n>`: como un grupo reúne respuestas de varios periodos, `<periodo>` es el más reciente de la pregunta (parámetro `periodo_emergentes`) y `<n>` es único dentro de (encuesta, periodo) entre todas las preguntas, para que no choquen.
- `catalogo_aspectos` no trae la pregunta: un aspecto compartido entre preguntas de la misma encuesta (p. ej. `CLARIDAD` en obs19 y obs20) aparece **una vez por (aspecto, encuesta)**, con el `orden` más bajo. Si el cliente necesita el orden por pregunta, habría que sumar `pregunta_codigo` al contrato.
- `SIN_ASPECTO` se publica con `grupo_visual = Aspectos definidos` (el contrato solo define dos valores) y nombre «Sin aspecto».
- `metricas_sentimiento` queda vacía si el cuaderno no la entrega; `accuracy_aspecto` sale de `acierto_holdout`.
- Tarea 11 (limpiar `pyproject.toml` y `poetry lock`) **no se hizo**: toca archivos fuera de los autorizados y se revisa contigo al final.
- Siguen pendientes de la Fase 3 los pasos A–D de la spec (corpus viejo de obs19 ausente) y la comparación de cifras de obs19.

**5. Resumen pedido por la spec (§6)**
No hubo corrida real de obs19, así que no hay resumen sobre datos reales. Con la corrida real de acreditación y cláusulas **sintéticas** salió: clasificador 1.230 (96,32 %), regla 34 (2,66 %), sistema 13 (1,02 %), hdbscan 0; cláusulas sin opinión 13; alertas 14. Estas cifras validan el flujo, no el modelo.

**6. Criterios de aceptación de la spec §6**

| Criterio | Estado |
|---|---|
| Regresión de 25.714 cláusulas del corpus viejo | **Pendiente** (corpus no disponible) |
| Etiquetas recuperadas exactas / tabla de correspondencia | **Pendiente** (pasos A–D) |
| Cobertura, IDs, orden, integridad referencial, dominios, confianzas, auditoría, metadatos y relectura con `dtype=str` | **Implementado y probado** (`validar_salidas_nlp`) |
| Taxonomía = clases del modelo | Implementado (`comparar_con_clases` dentro de la validación); no ejecutado con un modelo real |
| Dos corridas = mismos archivos | Probado con datos sintéticos (fecha fija); con el modelo real depende de la semilla de UMAP, ya fijada |

**7. Commit sugerido (lo haces tú)**
`feat(nlp): salidas del contrato v0.2 (Fase 4) — salidas_contrato.py, validar_salidas_nlp, Paso 12b del cuaderno y modelo persistible del residual`
(incluye `src/nlp_utilidades/salidas_contrato.py`, `src/nlp_utilidades/modulos_nlp.py`, `notebooks/Prisma_unificado.ipynb`, `tests/test_nlp_salidas.py`, `tests/test_nlp_residual_persistible.py`, `documents/ejemplos_salidas_nlp/` y este plan).

**8.** ¿Apruebas la Fase 4?


---

# PARTE 2 — Plan v2 (2026-11-19) · vigente

> Reemplaza al plan v1 para el trabajo pendiente. Las Fases 0–4 de la Parte 1 ya están hechas. Texto del 19-nov sin cambios de contenido (solo se quitó su título de nivel 1, ahora es el encabezado de esta parte).

Fecha: 2026-11-19 · Rama de trabajo: `feature/contrato-etapa2-nlp`
Reemplaza al plan del 17-nov. Las Fases 0–4 de ese plan ya están hechas y commiteadas (conciliación de utilities, rutas, ingesta desde la Etapa 1, salidas del contrato v0.2 con `salidas_contrato.py`). Este plan cubre los **7 puntos nuevos**, numerados igual que tu lista: **Punto 1 … Punto 7**, más un **Punto 0** previo.

Referencias (en `documents/`):
- `cambios etapa1 alertas.txt` — Etapa 1 v1.10 (rama `feature/etapa1-alertas`): las alertas pasan de la Etapa 2 a la Etapa 1. **Manda sobre la spec y el contrato en todo lo que toque alertas** (ambos están desactualizados en eso).
- `especificaciones/spec_nlp_etapa2.md` y `especificaciones/contrato_datos_v0.2.md` — siguen valiendo salvo en lo que este plan cambie (ver sección 11 de esta parte, "Cambios que este plan pide al contrato").

## 0. Cómo se trabaja

1. **Un punto = una aprobación.** Al terminar cada punto, Code se detiene con el informe de cierre (sección 12 de esta parte) y espera `Aprobado Punto N`.
2. **Decisiones de una en una.** Cuando Code necesite una aprobación tuya, la pregunta **una sola por mensaje**, con `AskUserQuestion` (2–4 opciones, la recomendada primero), y espera la respuesta antes de hacer la siguiente. No agrupa preguntas ni asume la respuesta. Las preguntas previstas están listadas en cada punto como **«Pregunta N.x»**; si aparece otra, la hace igual, de una en una.
3. **Sin git de escritura.** Code no ejecuta `git add`, `commit`, `push`, `merge`, `checkout`, `switch`, `reset`, `stash` ni `rebase`. Solo `status`, `diff`, `log`, `branch --show-current`. Los commits los haces tú; Code propone el mensaje al cerrar cada punto.
4. **Pruebas de humo obligatorias** sobre todo código nuevo o modificado, ejecutadas antes de pedir aprobación y reportadas con conteos (pasa/falla). Las duraderas van a `tests/` (`test_nlp_*.py`). `poetry run pytest` completo debe pasar sin regresiones respecto de la línea base del Punto 0. Lo que no se pueda ejecutar (GPU, spaCy grande, torch) se dice explícitamente y se prueba con datos sintéticos.
5. **Datos del cliente:** nunca imprimir textos de comentarios, de cláusulas ni salidas de celdas del cuaderno. Solo conteos, nombres de columnas, valores de categorías e IDs de aspecto. `data/raw/`, `data/processed/` y `data/etiquetas_manuales/` son de solo lectura salvo las carpetas de salida de este plan; nada de eso va a git.
6. **Cuaderno:** se edita con `NotebookEdit` o con un script `nbformat` que deje intactas las demás celdas; no se ejecuta completo (entrenar U4 exige GPU). Se valida con `nbformat.validate` + compilación de cada celda de código (ignorando líneas `%`/`!`). Citar celdas por número.
7. **No cambiar el modelo ni sus parámetros** (hiperparámetros, umbrales) salvo donde un punto lo pida expresamente. `stopwords.xlsx` y `negaciones.xlsx` los lee también la Etapa 3: no se tocan.
8. **Si algo no cuadra con el plan** (un archivo no existe, una cifra no coincide), Code no improvisa: informa y pregunta.

## 1. Estado de partida (verificado el 19-nov en la carpeta)

| Tema | Hallazgo |
|---|---|
| Rama | `feature/contrato-etapa2-nlp`, último commit `0cd65bc` (Fase 4: `salidas_contrato.py`, `validar_salidas_nlp`, Paso 12b del cuaderno). |
| Etapa 1 con alertas | **Aún no está en esta rama** (vive en `feature/etapa1-alertas`). La corrida disponible, `20261117-211304_3d69efc5d6`, **no trae `tiene_alerta`** ni `alertas.csv` (columnas actuales: `respuesta_id, encuesta, pregunta_codigo, rol_respondiente, periodo, id_origen, texto_original, es_no_respuesta, marca_inconsistencia, id_corrida_preparacion`). Hay que traer esa rama y volver a correr `run_etapa1.py` antes del Punto 1. |
| Alertas en el NLP hoy | Siguen en `modulos_nlp.py` (motivo `ALERTA`, código −4, `RecursosLexicos.detectar_alerta`) y en `salidas_contrato.py` (`alertas.csv`, `n_alertas`, `huella_alertas`, `leer_alertas`, `detectar_alertas`, `ParametrosSalidas.alerta_*`). Con la decisión del 19-nov dejan de ser del NLP. |
| Cuaderno: ingesta | Celda 16: `ft.cargar_encuestas_etapa1(RUTA_ETAPA1)`; el adaptador filtra solo `es_no_respuesta == "False"` y `COLUMNAS_RESPUESTAS_ETAPA1` no incluye `tiene_alerta`. |
| Cuaderno: salidas | Celda 7: `RUTAS["salida"] = RAIZ/"outputs"` (corridas internas `exportar_resultado` → `<salida>/<encuesta>/<run_id>/`, Paso 12, 15, 16 con `SALIDA.glob("*/*/manifest.json")`) y `RUTAS["nlp"] = RAIZ/"data"/"processed"/"nlp"` (las 5 tablas del contrato, una carpeta `<id_corrida_nlp>/` por corrida, celda 125). Hoy el cuaderno **no escribe** `respuestas.csv`: solo lo lee de la Etapa 1 para validar. |
| Catálogo hoy | `SIN_ASPECTO` sale con `aspecto_origen = "sistema"` y `grupo_visual = "Aspectos definidos"`; los grupos del residual salen como `EMERGENTE_<periodo>_<n>` con `grupo_visual = "Temas emergentes"` y `palabras_clave` (5–10 términos c-TF-IDF). `taxonomia.xlsx` tiene solo `pregunta_codigo, aspecto_id, aspecto_nombre, descripcion` (146 filas, 6 preguntas) y **no** contiene `SIN_ASPECTO` (el validador `taxonomia.py` lo reserva). |
| Umbrales vigentes (cuaderno, celda 4) | `max_palabras_respuesta_corta = 5`, `umbral_comentario_general = 0.80` (provisional; el default de `Config` es 0.50), residual: `umbral = 0.10`, `min_palabras = 6`. |
| Árbol de trabajo | Hay cambios ajenos a este plan que **no se tocan**: `README.md` y `data/utilities/esquema de los corpus.csv` aparecen borrados, `.claude/` y `historico_entregables/README_24ago2026.md` sin seguimiento. |
| Ejemplos | `documents/ejemplos_salidas_nlp/catalogo_aspectos.csv` está modificado y **corrupto**: la fila del tema emergente quedó entre comillas como un solo campo (probable guardado desde Excel). No usarlo como referencia; Code lo regenera al final con datos sintéticos si tú lo apruebas. |

## 2. Supuestos que adopto (Code los confirma contigo en el punto indicado)

| # | Supuesto | Dónde se confirma |
|---|---|---|
| S1 | **Punto 3** dice «que no muestren datos … que cumplan es_no_respuesta = False & tiene_alerta = False». Lo leo como lo inverso (contradice el Punto 1 si no): **los archivos exportados no deben mostrar datos de respuestas que NO cumplen** (no-respuestas y respuestas con alerta). La excepción es `respuestas.csv` (Punto 6), que lleva todo. | Pregunta 3.1 |
| S2 | **Punto 5** escribe `es_opinion == "VERDADERO"` (así lo muestra Excel en español). El contrato exige `True`/`False` en el archivo. Se filtra por `True` y el archivo sigue escribiendo `True`. | Pregunta 5.1 |
| S3 | **Punto 4.1**: con los Puntos 1 y 5, las únicas cláusulas con opinión que pueden quedar en `SIN_ASPECTO` son los **residuales que HDBSCAN dejó sin grupo**. Por eso `SIN_ASPECTO` pasa a `aspecto_origen = "hdbscan"` y `grupo_visual = "Sin grupo visual"`. Deja de significar «cláusula sin opinión». | Pregunta 4.1 |
| S4 | **Punto 4, «el cambio también aplica a taxonomia.xlsx»**: `taxonomia.xlsx` hoy no tiene `aspecto_origen` ni `grupo_visual`, y no es dónde nacen los temas de HDBSCAN. No está claro qué se quiere ahí; se pregunta, no se asume. | Preguntas 4.2 y 4.5 |
| S5 | Las alertas salen de las salidas del NLP (`alertas.csv`, `n_alertas`, `huella_alertas`), como dice el txt de la Etapa 1 en «Pendiente». | Pregunta 1.1 |

## 3. Punto 0 — Previo y línea base

**Tú (manual), antes de empezar con Code:**
1. Trae la rama de la Etapa 1 a la tuya: `git merge feature/etapa1-alertas` (o la integración que uses con Cris). Si hay conflictos, `git merge --abort` y consulta.
2. Haz el commit de lo pendiente que sea tuyo (no los borrados ajenos del árbol).

**Code (sin cambios de código):**
1. Rama actual; `git status` (informar, sin tocar lo ajeno).
2. `poetry run pytest` → anotar la línea base (con la Etapa 1 v1.10 debería sumar las 23 pruebas de `tests/test_alertas.py`).
3. Ejecutar `python run_etapa1.py` para generar una corrida nueva y confirmar con conteos: columna `tiene_alerta` en `respuestas.csv`, existencia de `alertas.csv`, y en `metadatos_preparacion.csv` las columnas `respuestas_con_alerta`, `alertas`, `alertas_negadas`, `huella_alertas`, `huella_negaciones`. Cifras de control del txt: obs19 = **86** respuestas con alerta; total **665**.
4. Inventario (solo conteos y líneas) de todo uso de `alerta`/`ALERTA`/`alertas` en `modulos_nlp.py`, `salidas_contrato.py`, `tests/test_nlp_*.py` y el cuaderno, para dimensionar el Punto 1.
5. Inventario de **todos los archivos que escribe el cuaderno** (nombre, celda, función que lo escribe): sirve a los Puntos 2, 3, 5 y 6.

**Pruebas de humo:** `pytest` completo; `python -c "import nlp_utilidades.modulos_nlp, nlp_utilidades.salidas_contrato"`.
**Aprobación:** informe de línea base (tabla de inventarios).

## 4. Punto 1 — Ingesta: solo `es_no_respuesta = "False"` y `tiene_alerta = "False"`

Objetivo: solo estas respuestas se dividen en cláusulas, se les modelan aspectos y se les calcula sentimiento.

**Pregunta 1.1 (de una):** ¿Retiramos del NLP la detección de alertas (motivo `ALERTA`/−4, `alertas.csv`, `n_alertas`, `huella_alertas`, `alertas.xlsx` como insumo)? *Recomendada:* sí en las **salidas** (el contrato las hace de la Etapa 1) y dejar el código legacy inactivo con un aviso, sin borrarlo todavía; la limpieza del código muerto se pide aparte.

Cambios:
1. `COLUMNAS_RESPUESTAS_ETAPA1` incluye `tiene_alerta`; validación de que solo toma `True`/`False` (mismo patrón que `es_no_respuesta`) y mensaje claro si falta la columna (corrida anterior a v1.10: «vuelva a correr run_etapa1.py»). Sin retrocompatibilidad silenciosa.
2. `cargar_encuestas_etapa1`: filtro `es_no_respuesta == "False"` **y** `tiene_alerta == "False"`; por pregunta devuelve los conteos de excluidas por cada motivo (para la auditoría). Documentar en el docstring que las respuestas con alerta **no** pasan al NLP aunque la alerta esté negada.
3. `comparar_con_metadatos_etapa1` y la auditoría de la ingesta (Paso 2.1, celda 24): `leidas = respuestas con comentario − respuestas_con_alerta` por pregunta, contra `metadatos_preparacion.csv`.
4. `salidas_contrato.py` / validador / `ParametrosSalidas` según la respuesta 1.1 (sin `alertas.csv` ni `n_alertas` en `metadatos_nlp` si se retiran; actualizar tests).
5. Texto de las celdas markdown del Paso 2 y 12b que hablan de alertas o de «las 5 tablas».

**Pruebas de humo:**
- Unitaria, con un `respuestas.csv` sintético en directorio temporal (≥ 3 preguntas; filas con las 4 combinaciones de `es_no_respuesta`/`tiene_alerta`; un periodo `2025-1`; un valor `NA`): solo pasan las `False/False`; periodo sigue siendo texto; falla con mensaje claro si falta `tiene_alerta` o trae un valor distinto de True/False.
- Corrida real: por pregunta, `procesadas = con_comentario − con_alerta`; obs19 esperado **24.433 − 86** si la corrida nueva no cambió esos totales (se informa la diferencia si cambió). Sin imprimir textos.
- `pytest` completo.

**Aprobación Punto 1:** tabla por pregunta (total, no-respuestas, con alerta, procesadas) y resultado de pruebas.

## 5. Punto 2 — Ruta de salida de modelado: `RAIZ/outputs` → `RAIZ/data/processed/nlp`

Hoy hay dos destinos: `RUTAS["salida"]` (`outputs/`, corridas internas) y `RUTAS["nlp"]` (`data/processed/nlp`, tablas del contrato).

**Pregunta 2.1 (de una): estructura dentro de `data/processed/nlp/`.** *Recomendada:* `data/processed/nlp/<id_corrida_nlp>/` para las tablas del contrato (como hoy) y `data/processed/nlp/modelado/<encuesta>/<run_id>/` para los artefactos internos (documentos, cláusulas internas, tópicos, ABSA, scorecard, manifest, auditoría humana). Así `SALIDA.glob("*/*/manifest.json")` de los Pasos 15–16 sigue funcionando apuntando a `.../modelado/`.

Cambios:
1. Celda 7: `RUTAS["salida"]` → `RAIZ/data/processed/nlp/modelado` (o lo que decidas); `SALIDA` y los `mkdir` coherentes; `BASE_SALIDA` en Colab sin romper (`/content` si `GUARDAR_EN_DRIVE = False`).
2. `modulos_nlp.py`: ningún default que escriba en `outputs/` (revisar `output_dir`, `exportar_resultado(raiz_salida)`, rutas del Excel de entrenamiento y de las auditorías); `Config`/funciones que reciben la carpeta de salida usan la nueva.
3. Pasos 15, 16 y siguientes (celdas 132–139): zip `outputs_full`, subida de auditorías y recalificación leen/escriben en la nueva ruta; textos markdown que mencionan `outputs/`.
4. La carpeta `outputs/` existente **no se borra ni se mueve**: Code solo informa qué contiene (conteos) y tú decides si la migras.
5. Confirmar que `data/processed/` ya está en `.gitignore` (lo está) para que nada de esto entre a git.

**Pruebas de humo:** nbformat + compilación de celdas; script que ejecuta solo la celda de rutas con `EN_COLAB = False` y verifica que `SALIDA`, `RUTAS["nlp"]` y el caché existen y cuelgan de `data/processed/nlp`; búsqueda de que no quede ninguna referencia a `"outputs"` como destino de escritura (`grep` con conteos); `exportar_resultado` sobre un `Resultado` mínimo sintético escribe bajo la nueva ruta (si el entorno lo permite); `pytest`.

**Aprobación Punto 2:** tabla «antes → después» de rutas.

## 6. Punto 3 — Archivos exportados: sin datos de respuestas que no cumplen las condiciones

Con el Punto 1 el corpus ya entra filtrado, así que los archivos internos nacen sin esas respuestas; este punto **verifica y cierra los huecos**.

**Pregunta 3.1 (de una):** confirmar el supuesto S1 (los archivos exportados no muestran datos de no-respuestas ni de respuestas con alerta; `respuestas.csv` es la excepción por el Punto 6).
**Pregunta 3.2 (de una):** ¿la auditoría de la ingesta (`auditoria_ingesta_etapa1.csv`, que solo trae **conteos** de excluidas por pregunta) se mantiene? *Recomendada:* sí, porque no muestra datos de ninguna respuesta.

Cambios:
1. Con el inventario del Punto 0, para **cada archivo que escribe el cuaderno** (documentos, cláusulas, tópicos, ABSA, grid, residuales, scorecard, `auditoria.xlsx` con `muestra_manual`/`muestra_pureza`, `clausulas_documentos_unificado.xlsx`, `auditoria_ingesta_etapa1.csv`, zip, las tablas del contrato) verificar que solo se alimenta del corpus filtrado y nunca de `respuestas.csv` completo (excepción: la validación y el Punto 6).
2. En `clausulas_documentos_unificado.xlsx` (celda 123): hoy une documentos y cláusulas con `how="left"` y deja los «documentos sin cláusulas (vacíos, ruido)»; revisar que no reintroduzca respuestas excluidas.
3. Ajustar lo que haga falta y las notas «Cómo leer esta salida» que hablen de alertas o de no-respuestas.

**Pruebas de humo:** prueba que arma un corpus sintético con las 4 combinaciones, corre la parte pura de exportación y verifica que ningún `respuesta_id` excluido aparece en ningún archivo (conjunto de IDs vs conjunto de IDs); en la corrida real, mismo chequeo con conteos sobre los archivos ya escritos (si existen); `pytest`.

**Aprobación Punto 3:** tabla archivo → fuente de datos → resultado del chequeo.

## 7. Punto 4 — `catalogo_aspectos.csv`, `taxonomia.xlsx` y `clausulas.csv`

### 4.1 `SIN_ASPECTO`
Registros con `aspecto_id == "SIN_ASPECTO"`: `aspecto_origen = "hdbscan"` y `grupo_visual = "Sin grupo visual"`.
**Pregunta 4.1 (de una):** confirmar el supuesto S3 (SIN_ASPECTO = residual que HDBSCAN dejó sin grupo) y que `aspecto_origen = "sistema"` deja de usarse (se conserva en el dominio del validador por compatibilidad).
**Pregunta 4.2 (de una): qué significa «aplica también a taxonomia.xlsx».** Opciones que Code presenta: (a) agregar a `taxonomia.xlsx` las columnas `aspecto_origen` y `grupo_visual` (todas las filas actuales: `clasificador` / `Aspectos definidos`) y una fila `SIN_ASPECTO` por pregunta (`hdbscan` / `Sin grupo visual`), de modo que `catalogo_aspectos` salga de una sola fuente; (b) dejar `taxonomia.xlsx` como está y que el cambio viva solo en `catalogo_aspectos.csv` y `clausulas.csv`; (c) otra que indiques. Si es (a): hay que relajar `RESERVADOS` y `leer_taxonomia` en `taxonomia.py` (y sus tests) y revisar que la Etapa 3 y `run_validar_taxonomia.py` no se rompan.
En `clausulas.csv` aplica a `aspecto_id` y `aspecto_origen` (la tabla no tiene `grupo_visual`). En `metadatos_nlp`, `pct_sin_aspecto` y `pct_aspecto_hdbscan` se redefinen sin solaparse: **Pregunta 4.3 (de una)** — ¿`pct_aspecto_hdbscan` cuenta solo los grupos emergentes y `pct_sin_aspecto` los sin grupo? *Recomendada:* sí.

### 4.2 Temas de HDBSCAN: `aspecto_id` = primer sustantivo de `palabras_clave`
Registros con `aspecto_origen == "hdbscan"` y `aspecto_id != "SIN_ASPECTO"`: el `aspecto_id` pasa de `EMERGENTE_<periodo>_<n>` al **primer sustantivo** de `palabras_clave` (los términos ya vienen ordenados por relevancia c-TF-IDF).
Cuestiones de diseño que Code resuelve contigo, **una por una**:
- **Pregunta 4.4 — Cómo detectar el sustantivo.** *Recomendada:* spaCy `es_core_news_lg` (ya es dependencia del segmentador), etiquetando cada término **dentro de su contexto de grupo** (unirlos en una frase corta) en vez de palabra suelta, porque el POS aislado es poco fiable («dinámicas» puede salir adjetivo); aceptar `NOUN` y excluir `PROPN`. Fallback si ningún término es sustantivo: primer término + aviso (o conservar `EMERGENTE_…`; se pregunta).
- **Pregunta 4.5 — Formato y colisiones del ID.** El validador exige `^[A-Z0-9]+(_[A-Z0-9]+)*$`: MAYÚSCULAS, sin tildes, no alfanumérico → `_` (la ñ ya se conserva en nombres de columna pero no en IDs; definir `Ñ`→`N`). Si dos grupos de la misma pregunta dan el mismo sustantivo, o choca con un `aspecto_id` de la taxonomía (p. ej. `TECNOLOGIA`), ¿sufijo numérico (`CLASES_2`), prefijo `EMERGENTE_` o el periodo? Nota: al quitar el periodo del ID, un mismo sustantivo en periodos distintos se unifica en el tablero (¿deseado?), pero el contrato dice que los grupos cambian por corrida y no se mezclan con los fijos: por eso se recomienda **mantener `grupo_visual = "Temas emergentes"`** como separador y un prefijo/sufijo que evite choques con la taxonomía.
- **Pregunta 4.6 — `aspecto_nombre` del emergente** (hoy la etiqueta c-TF-IDF): ¿se conserva o pasa a ser el sustantivo?
- **Aplicación a `taxonomia.xlsx`** según la respuesta 4.2: los emergentes **no** deberían entrar a la taxonomía fija (el contrato dice que solo se «proponen» cuando se repiten en varios periodos); si la respuesta 4.2 pide registrarlos, Code lo implementa como una hoja/lista aparte de **propuestas** y no como filas de `aspectos`, salvo que digas otra cosa.

**Pruebas de humo (Punto 4):**
- `catalogo_aspectos`/`clausulas` sintéticos: `SIN_ASPECTO` siempre con `hdbscan` + `Sin grupo visual`; ningún `EMERGENTE_` residual si hay sustantivo; IDs que cumplen `PATRON_ID`; sin duplicados `(encuesta, aspecto_id)`; colisión simulada resuelta según la regla elegida; todo `aspecto_id` de `clausulas` existe en el catálogo.
- Extracción del sustantivo con un etiquetador **inyectable** (`pos_fn`) para probar sin cargar spaCy; si `es_core_news_lg` está disponible, 10 grupos sintéticos con el modelo real (solo conteos y los IDs resultantes).
- `validar_salidas_nlp` actualizado: dominio de `grupo_visual` ampliado y reglas del Punto 4; pruebas negativas (SIN_ASPECTO con origen `sistema` → falla).
- `pytest` completo (incluye `test_taxonomia.py` y `test_limpieza.py`).

**Aprobación Punto 4:** catálogo y cláusulas de ejemplo **sintéticos** (solo IDs/orígenes), resultado de pruebas.

## 8. Punto 5 — `clausulas.csv` solo con `es_opinion == True`

**Pregunta 5.1 (de una):** confirmar S2 (`True` en el archivo; `VERDADERO` es solo cómo lo ve Excel).
**Pregunta 5.2 (de una): `orden` con saltos.** Si se retiran las cláusulas sin opinión, `orden` (que sale de la segmentación completa) queda con huecos y el validador actual exige «sin saltos». Opciones: (a) *recomendada* — **conservar el `orden` original** (los IDs siguen siendo estables y cruzan con las etiquetas y con la tabla de correspondencia) y cambiar la regla a «empieza en 1 solo si la primera cláusula es opinión; estrictamente creciente y único»; (b) renumerar 1…n (rompe la trazabilidad con la segmentación y con las etiquetas manuales).
**Pregunta 5.3 (de una):** cobertura: hoy el contrato dice «cada respuesta con comentario tiene ≥ 1 cláusula». Con este punto, una respuesta cuyas cláusulas son todas no-opinión (en obs19 ≈ 118) **no aparecerá** en `clausulas.csv`. ¿Se acepta, y `metadatos_nlp.n_respuestas` sigue contando las procesadas (entrada) y no las que tienen cláusulas? *Recomendada:* sí, con una verificación contra la tabla interna (que sí conserva todas las cláusulas) en vez de contra el CSV.

Cambios: `construir_salidas_contrato` filtra por `es_opinion == "True"` **al final** (después de calcular IDs, orden, auditoría); `auditoria_clasificacion` queda 1:1 con `clausulas`; `n_clausulas` y los `pct_*` se calculan sobre lo publicado; `sentimiento` ya nunca es vacío (se conserva la columna y la regla «vacío ⇔ es_opinion = False» se mantiene como verificación trivial); actualizar validador, tests y la celda 125 (resumen).

**Pruebas de humo:** corpus sintético con cláusulas opinión y no-opinión en la misma respuesta: `clausulas.csv` solo con `True`; IDs idénticos a los de la versión sin filtro; `auditoria` 1:1; respuesta con solo no-opinión ausente de `clausulas` pero presente en la tabla interna; determinismo (dos construcciones, mismo hash); `pytest`.

**Aprobación Punto 5.**

## 9. Punto 6 — `respuestas.csv`: todos los registros, con `es_no_respuesta` y `tiene_alerta`

Hoy el cuaderno no escribe `respuestas.csv`. Se escribe en la carpeta de la corrida del NLP (`data/processed/nlp/<id_corrida_nlp>/respuestas.csv`).
**Pregunta 6.1 (de una): alcance.** (a) *Recomendada* — todas las filas de la corrida de la Etapa 1 **de las preguntas procesadas en esta corrida del NLP**, y las columnas de la Etapa 1 (incluye `texto_original`, dato del cliente; la carpeta está fuera de git); (b) todas las filas de todas las preguntas de la corrida de la Etapa 1; (c) un subconjunto de columnas (`respuesta_id`, `encuesta`, `pregunta_codigo`, `periodo`, `rol_respondiente`, `es_no_respuesta`, `tiene_alerta`, `id_corrida_preparacion`; sin el texto).
Cambios: leer el `respuestas.csv` de la corrida (`dtype=str`, `utf-8-sig`), verificar que trae `es_no_respuesta` y `tiene_alerta` y que el conjunto de `respuesta_id` coincide con el de la Etapa 1 (todos, sin perder ninguno), escribirlo con el mismo formato (utf-8-sig, True/False, vacíos en blanco); `validar_salidas_nlp` pasa a validar contra **este** archivo (integridad referencial: `respuesta_id` de `clausulas` ⊆ respuestas con `es_no_respuesta=False` y `tiene_alerta=False`); `huella_archivos`/determinismo incluyen el archivo.
**Pruebas de humo:** sintético: el archivo escrito tiene todas las filas y las dos columnas, ninguna celda «NA», periodo como texto; `clausulas.respuesta_id` ⊆ procesadas; falla si una cláusula apunta a una respuesta con alerta o no-respuesta; corrida real: conteo de filas = Etapa 1 (según alcance elegido) y `procesadas = con_comentario − con_alerta`; `pytest`.
**Aprobación Punto 6.**

## 10. Punto 7 — Unificar `VALORACIÓN GLOBAL`, `OPINIÓN GENERAL` y `COMENTARIO GENERAL`

### 7.1 Qué genera cada categoría hoy (verificado en `modulos_nlp.py` y `salidas_contrato.py`)

| Categoría | Código interno / `asignacion` | Condición que la genera | Sale como |
|---|---|---|---|
| **VALORACIÓN GLOBAL** | `-3` · `valoracion_global` (motivo `VALORACION_GLOBAL`) | Cláusula **con opinión** (no ruido, no «no opinión») cuyo **texto de aspecto queda vacío** al neutralizarla con `neutralizacion.xlsx` (se quitan valorativos, persona, negadores y nombres propios): no dice *de qué* habla, solo valora («excelente profesor»). Si la respuesta es **larga** (> 5 palabras) **no pasa por el clasificador** y se queda aquí (regla g). Si la respuesta es **corta** (≤ 5 palabras) sí pasa por el clasificador y puede terminar en un aspecto, en Comentario General o aquí. | `OPINION_GENERAL`, origen `regla` (mapeo por defecto) |
| **COMENTARIO GENERAL** | `-5` · `regla_comentario_general` | Cláusula **con opinión** (`ASPECTO` o `VALORACION_GLOBAL`) de una **respuesta corta** (texto original sin limpiar con ≤ `max_palabras_respuesta_corta` = 5 palabras, `split()`), cuya **probabilidad máxima del clasificador** es **< `umbral_comentario_general`** (cuaderno: **0,80**, provisional; default del módulo 0,50). Es decir: texto breve que el clasificador no logra asignar con seguridad. | `OPINION_GENERAL`, origen `regla` |
| **OPINIÓN GENERAL** | Clase supervisada «Opinión general» (existe en las 6 taxonomías como `OPINION_GENERAL`); `CLASES_SIN_ASPECTO_U4` también incluye «Sin aspecto» y «Ninguno de la lista» | La **predice el clasificador U4** (aprendió de las etiquetas manuales): la cláusula es una valoración sin aspecto concreto. Además es el **destino** al que `MAPEO_ASIGNACION_DEFECTO` envía a las dos anteriores. | `OPINION_GENERAL`, origen `clasificador` (si la predijo U4) o `regla` (si vino de las otras dos) |

Consecuencia: hoy `OPINION_GENERAL` en `clausulas.csv` mezcla **tres orígenes** y no hay forma de distinguirlos salvo por `aspecto_origen` (clasificador vs regla), que no separa valoración global de comentario corto. En el notebook y el scorecard siguen existiendo por separado (`-3`, `-5`, métricas `pct_valoracion_global`, `pct_comentario_general`, gates).

### 7.2 Sugerencias de unificación y flujo

**Opción A (recomendada) — una sola categoría «Opinión general» con un flujo único, en dos etapas:**
- *Etapa A (bajo riesgo):* una sola categoría de cara al usuario: `aspecto_id = OPINION_GENERAL`, `aspecto_nombre = "Opinión general"` en **todas** las tablas, cuaderno y gráficos; se conservan los códigos internos `-3`/`-5` para no tocar scorecard ni ABSA, y se agrega una columna **interna** `motivo_opinion_general ∈ {sin_contenido, comentario_corto, clasificador}` para trazabilidad (no va a las tablas del contrato; no se cambian columnas del contrato).
- *Etapa B (opcional, riesgo medio):* colapsar `-3` y `-5` en un solo código dentro de `modulos_nlp.py` y unificar métricas/gates.

Flujo propuesto por cláusula **con opinión**, en este orden:
1. **Sin contenido** (antes VALORACIÓN GLOBAL): texto de aspecto vacío → `OPINION_GENERAL`, origen `regla`, `motivo = sin_contenido`. Propuesta: aplicarlo también a respuestas cortas (hoy pasan por U4), porque una cláusula sin contenido no tiene aspecto que predecir; **se mide primero** (Code compara, con la muestra etiquetada, cuántas de estas cláusulas U4 asigna hoy a un aspecto real y cuántas a «Opinión general»; solo conteos).
2. **El clasificador** predice «Opinión general» (o «Sin aspecto»/«Ninguno de la lista») → `OPINION_GENERAL`, origen `clasificador`, `motivo = clasificador`.
3. **Respuesta corta con baja confianza** (antes COMENTARIO GENERAL): ≤ 5 palabras y p máx < umbral → `OPINION_GENERAL`, origen `regla`, `motivo = comentario_corto`.
4. Si no, el aspecto del clasificador o el residual de HDBSCAN (Punto 4).

Decisiones que se dejan **como parámetros** (valor de hoy), alineadas con la spec §5: `regla_opinion_general = "palabras" | "contenido"` (la alternativa «contenido» = solo `n_contenido = 0` va a OG; las cortas con contenido van al clasificador), `umbral_comentario_general` (0,80 vs 0,50 del módulo: unificar con la muestra etiquetada, acierto frente a cobertura) y `max_palabras_respuesta_corta`.
**Opción B:** mantener tres categorías visibles (distintos `aspecto_id`) — descartada por contradecir lo que pides y por no aportar al tablero.
**Opción C:** unificar solo en la capa de salida (lo que ya hace el mapeo) — es la «Etapa A» sin la columna de trazabilidad; deja la mezcla de orígenes sin explicar.

**Pregunta 7.1 (de una):** ¿Opción A con las Etapas A y B, solo la Etapa A, o la Opción C? *Recomendada:* solo la Etapa A ahora.
**Pregunta 7.2 (de una):** ¿el paso 1 (sin contenido → Opinión general también en respuestas cortas) se activa o se deja como hoy hasta medirlo? *Recomendada:* medir primero y dejarlo apagado por parámetro.
**Pregunta 7.3 (de una):** ¿«Sin aspecto» y «Ninguno de la lista» entran en la unificación? *Recomendada:* sí, como clases que el clasificador puede predecir y que se publican como `OPINION_GENERAL` (ninguna taxonomía actual las usa; Code lo confirma con conteos).

**Pruebas de humo:** sintético con una cláusula de cada tipo (sin contenido, comentario corto de baja confianza, clase predicha, caso normal): las 3 primeras salen como `OPINION_GENERAL`, la cuarta con su aspecto; `motivo_opinion_general` correcto; cambio de parámetros (`regla_opinion_general`, umbral) cambia el resultado como se espera; ninguna tabla del contrato cambia de columnas; scorecard/ABSA internos sin regresión (`pytest`); conteos antes/después de la unificación sobre la corrida real (si hay modelo y GPU) o sobre el corpus sintético.
**Aprobación Punto 7.**

## 11. Cambios que este plan pide al contrato (para llevar a Cris: v0.3)

No se editan solos; Code los lista al final del Punto 7 en `documents/propuesta_cambios_contrato_v03.md`:
1. Entrada del NLP = `es_no_respuesta = False` **y** `tiene_alerta = False` (§5.1 y principios).
2. `alertas.csv`, `n_alertas` y `huella_alertas` salen del NLP (§5.3, §5.4, reglas de alertas §5.2 punto 5); la Etapa 1 publica `alertas.csv` sin `clausula_id` ni `categoria_alerta`.
3. `grupo_visual` admite **«Sin grupo visual»**; `SIN_ASPECTO` pasa a `aspecto_origen = hdbscan` (el dominio de `aspecto_origen` ya no necesita `sistema` salvo compatibilidad).
4. `aspecto_id` de los temas emergentes ya no es `EMERGENTE_<periodo>_<n>` sino derivado del primer sustantivo de `palabras_clave` (§3.3/§5.3).
5. ~~`clausulas.csv` solo con `es_opinion = True`~~ **Descartado en el Punto 5 (decisión del usuario, 2026-11-19): `clausulas.csv` conserva las cláusulas con y sin opinión.** Único ajuste: la regla «cada respuesta con comentario tiene ≥ 1 cláusula» admite como excepción las respuestas procesables que quedan vacías o de ruido al limpiarse, que no generan cláusulas (§5.2, §5.4); `n_respuestas` de `metadatos_nlp` cuenta todas las procesadas.
6. ~~El NLP publica `respuestas.csv`~~ **Descartado en el Punto 6 (decisión del usuario, 2026-11-19): `respuestas.csv` ya existe (Etapa 1).** En su lugar, el NLP publica en la carpeta de su corrida el Excel `clausulas_documentos_unificado.xlsx` (hoja `unificado`; no es una tabla del contrato) con `es_no_respuesta` y `tiene_alerta`.
7. Categorías `VALORACIÓN GLOBAL`/`COMENTARIO GENERAL`/`OPINIÓN GENERAL` unificadas en `OPINION_GENERAL` (documentar la regla).
8. Ruta de salida de modelado: `data/processed/nlp/` (carpeta propuesta en la spec).

> **Versión final de estos cambios (tras los Puntos 1–7 y las decisiones del usuario): `documents/propuesta_cambios_contrato_v03.md`.** Allí el cambio 5 (filtro `es_opinion`) y el 6 (`respuestas.csv`) figuran como descartados/reemplazados.

## 12. Informe de cierre de cada punto (formato que Code entrega)

1. Qué cambió (archivos, celdas del cuaderno por número, nº de líneas).
2. Pruebas de humo (comando → pasa/falla, conteos).
3. Decisiones tomadas (con tu respuesta a cada pregunta) y las que siguen abiertas.
4. Riesgos o supuestos.
5. Mensaje sugerido de commit (para que lo hagas tú).
6. «¿Apruebas el Punto N para pasar al N+1?».

## 13. Riesgos transversales

- **Etapa 1 v1.10 sin integrar**: sin `tiene_alerta` el Punto 1 no puede probarse con datos reales; por eso el Punto 0.
- **Contrato v0.2 vs. cambios**: varios puntos contradicen el contrato vigente (sección 11); hasta que Cris lo actualice, las validaciones del pipeline quedan **adelantadas** al contrato y Power BI no debe conectarse a estas tablas.
- **Sustantivo por POS**: etiquetar términos aislados es ruidoso; por eso se recomienda etiquetar en contexto y registrar cuántos grupos caen en el fallback.
- **IDs emergentes sin periodo** pueden unificar grupos distintos entre corridas; decidir la regla (Pregunta 4.5) antes de implementar.
- **GPU**: entrenar U4 y medir la unificación sobre datos reales exige Colab; localmente solo hay pruebas de humo con datos sintéticos.
- **Datos del cliente**: `respuestas.csv` del Punto 6 contiene `texto_original`; vive en `data/processed/` (fuera de git).

## 14. Informes de cierre por punto (plan v2)

### Punto 0 — Previo y línea base (2026-11-19)

**1. Qué cambió.** Ningún archivo del repo ni de datos. Solo scripts de apoyo en el scratchpad (fuera del proyecto).

**2. Pruebas y verificaciones**

| Comando / verificación | Resultado |
|---|---|
| `git branch --show-current` | `feature/contrato-etapa2-nlp` |
| `git status` | Limpio. El merge `0493a17` ya trajo la Etapa 1 v1.10 (`c28174c`). |
| `poetry run pytest` | **191 pasan, 0 fallan** (168 de la Fase 4 + 23 de `test_alertas.py`). **Línea base.** |
| `import nlp_utilidades.modulos_nlp, nlp_utilidades.salidas_contrato` | Pasa |
| Corrida `20261119-224804_57895bd79b` | Un solo `id_corrida_preparacion`. `respuestas.csv` trae `tiene_alerta` (tras `es_no_respuesta`); existe `alertas.csv` (`alerta_id, respuesta_id, termino, texto_coincidente, negada`); `metadatos_preparacion.csv` trae `respuestas_con_alerta`, `alertas`, `alertas_negadas`, `huella_alertas`, `huella_negaciones`. |
| Consistencia | `tiene_alerta` solo `True`/`False`; 0 casos con `es_no_respuesta = True` y `tiene_alerta = True`. |

Conteos por pregunta (`procesadas = con comentario − con alerta`):

| Pregunta | Total | No-respuestas | Con comentario | Con alerta | Procesadas |
|---|---:|---:|---:|---:|---:|
| obs19_aspectos_resaltar | 63.417 | 38.984 | 24.433 | 86 | 24.347 |
| obs20_oportunidades_mejora | 63.417 | 46.714 | 16.703 | 511 | 16.192 |
| obs23_percepcion_progreso | 6.876 | 1.334 | 5.542 | 21 | 5.521 |
| obs24_aspectos_resaltar | 6.876 | 1.451 | 5.425 | 3 | 5.422 |
| obs25_oportunidades_mejora | 6.876 | 1.556 | 5.320 | 8 | 5.312 |
| **Total** | 147.462 | 90.039 | 57.423 | 629 | **56.794** |

`alertas.csv`: 767 filas, 69 negadas, 629 respuestas distintas. Controles del txt: obs19 = 86 con alerta (coincide); obs19 con comentario = 24.433 (coincide con la spec).

Inventario de alertas (solo conteos):
- `modulos_nlp.py`: 72 apariciones (`ALERTA` −4, `MOTIVO_ALERTA`, `DOC_ALERTA`, mapeo «ALERTA DE CASO ESPECIAL», `RecursosLexicos._leer_alertas`/`detectar_alerta`, clasificación de respuesta ~l.2997, exclusión de ABSA l.4942, scorecard y gates l.5336–5412, regla U4 l.7121–7176, `p_*` forzado a NEG l.7673–7682, conteos `documentos_alerta`/`clausulas_alerta`).
- `salidas_contrato.py`: 55 (`alertas.csv`, `leer_alertas`, `detectar_alertas`, `ParametrosSalidas.alerta_*`, `n_alertas`, `huella_alertas`, mapeo `"alerta"`, validaciones l.642–653 y 708).
- `tests/test_nlp_salidas.py`: 30 (`test_alertas.py`, 40, es de la Etapa 1).
- Cuaderno (140 celdas, 62 de código): celdas 2, 4 (3), 7, 25, 43, 51, 58, 69, 89, 111, 113, 124 (4), 125 (12), 126 (4).

Inventario de lo que escribe el cuaderno:

| Celda | Escribe |
|---|---|
| 24 | `auditoria_ingesta_etapa1.csv` (`to_csv`) |
| 121 | `exportar_resultado` (`SALIDA/<encuesta>/<run_id>/`) |
| 123 | `clausulas_documentos_unificado.xlsx` (`to_excel`) |
| 125 | `escribir_salidas_contrato` (5 tablas) y `guardar_modelo_residual` |
| 132, 139 | `make_archive` (zips) |
| 136 | `.zip` de auditorías |

Celdas con `outputs`/`SALIDA`: 7, 24, 26, 132, 136, 137, 138, 139. `outputs/` existe y está vacío; `data/processed/nlp/` está vacía.

**3. Decisiones tomadas / abiertas.** Ninguna. Pendiente: confirmar si se vuelve a correr `run_etapa1.py` para que la corrida más reciente incluya acreditación.

**4. Hallazgos y riesgos**
- La corrida nueva solo trae las 5 preguntas de evadoc (calidad no tiene archivo en `data/raw/`; acreditación estaba en la corrida anterior). Como el cuaderno toma `ultima_corrida`, acreditación no se puede procesar con ella.
- Totales de alertas distintos a los del txt (629 / 767 / 69 frente a 665 / 809 / 70): la diferencia (36 / 42 / 1) es compatible con la ausencia de acreditación; inferencia no verificada.
- `COLUMNAS_RESPUESTAS_ETAPA1` aún no incluye `tiene_alerta` (Punto 1).

**5. Commit sugerido.** Sin cambios de código; solo este informe: `docs: informe de cierre Punto 0 NLP etapa 2 v2 (línea base 191 pruebas)`.

**6.** ¿Apruebas el Punto 0 para pasar al Punto 1?

### Punto 1 — Ingesta: solo `es_no_respuesta = "False"` y `tiene_alerta = "False"` (2026-11-19)

**1. Qué cambió**
- `src/nlp_utilidades/modulos_nlp.py` (+89 / −195):
  - Ingesta: `COLUMNAS_RESPUESTAS_ETAPA1` incluye `tiene_alerta`; `leer_respuestas_etapa1` valida que solo tome `True`/`False` y, si falta la columna, dice «vuelva a correr `python run_etapa1.py`». `cargar_encuestas_etapa1` conserva solo `es_no_respuesta = False` y `tiene_alerta = False` (las respuestas con alerta no pasan aunque estén negadas) y guarda en `Encuesta.conteos_ingesta` (campo nuevo) `total`, `no_respuestas`, `con_alerta`, `procesadas` por pregunta. `comparar_con_metadatos_etapa1` compara contra `respuestas_con_comentario − respuestas_con_alerta` y devuelve además `total`, `no_respuestas` y `con_alerta`.
  - Retiro del código de alertas (decisión 1.1): `ALERTA` (−4), `MOTIVO_ALERTA`, `DOC_ALERTA`, su entrada en `ETIQUETAS_ESPECIALES`, `RecursosLexicos.alertas`/`_leer_alertas`/`detectar_alerta` (y su aporte a `huella()`), la rama de alerta de `construir_unidades` (también la columna interna `palabra_alerta`), la exclusión en `muestra_manual`, las filas y chequeos de alerta en `conciliar`, las ramas de `etiquetar_corpus_u4` y `procesar_encuesta` (incluido el forzado NEG) y las métricas `documentos_alerta`/`clausulas_alerta`. Docstrings actualizados.
- `src/nlp_utilidades/salidas_contrato.py` (+34 / −156): pasa de 5 a **4 tablas** (sin `alertas.csv`); `metadatos_nlp` sin `huella_alertas` ni `n_alertas`; se eliminan `leer_alertas`, `detectar_alertas`, `leer_negadores`, `ParametrosSalidas.alerta_*`/`ventana_negacion` y el mapeo `alerta`; la huella de la corrida ya no incluye alertas. `validar_salidas_nlp` exige `tiene_alerta` en `respuestas.csv`, cobertura = respuestas con `es_no_respuesta = False` y `tiene_alerta = False`, y falla si una no-respuesta o una respuesta con alerta tiene cláusulas.
- Cuaderno (18 de 140 celdas; formato intacto, diff de una línea): 1, 2, 4 (quita los 3 parámetros de alertas de `SALIDAS_CONTRATO`), 7 (`alertas.xlsx` ya no es léxico obligatorio), 15, 21, 23 (Paso 2/2.1: filtro y auditoría con `total`, `no_respuestas`, `con_alerta`), 25, 43, 51, 58, 69, 89, 111, 113 (textos sin ALERTA/−4), 124, 125 (4 tablas, sin `leer_alertas`), 126.
- Pruebas: `tests/test_nlp_ingesta.py` (+6 nuevas y ajustes), `tests/test_nlp_salidas.py` (se retiran 5 de alertas, se agregan 4), `tests/test_nlp_alertas_retiradas.py` (nuevo, 5).

**2. Pruebas de humo**

| Comando / verificación | Resultado |
|---|---|
| `poetry run pytest` | **202 pasan, 0 fallan** (línea base 191) |
| Ingesta sintética (obs19/obs20; 4 combinaciones; periodo `2025-1`; «NA») | Solo pasan `False/False`; el periodo sigue siendo texto; «NA» se conserva; los textos con alerta no llegan al corpus; `conteos_ingesta` correctos; (True, True) también se excluye; todo con alerta → «no tiene respuestas procesables» |
| Errores esperados | Falta `tiene_alerta` en `respuestas.csv` → pide volver a correr la Etapa 1; valor distinto de True/False → falla; falta `respuestas_con_alerta` en `metadatos_preparacion.csv` → pide volver a correr |
| Salidas | 4 tablas pasan `validar_salidas_nlp`; fallan con mensaje esperado: respuesta con alerta con cláusulas, falta `tiene_alerta`, y todas las negativas previas |
| Sin restos | Prueba que busca los nombres retirados en los dos módulos, la API pública y el cuaderno: 0 restos |
| Corrida real `20261119-224804_57895bd79b` (celdas 6, 4, 7, 8, 11, 14, 16, 18, 19, 22 ejecutadas) | `diferencia = 0` y `respuesta_id` únicos en las 5 preguntas |
| `nbformat.validate` y `ast.parse` de las 62 celdas de código | Pasan (con las líneas `%`/`!` sustituidas por `pass`) |
| `python run_validar_taxonomia.py` | Pasa: 146 filas, 66 aspectos |

Tabla por pregunta (corrida real, solo conteos):

| Pregunta | Total | No-respuestas | Con alerta | Procesadas |
|---|---:|---:|---:|---:|
| obs19_aspectos_resaltar | 63.417 | 38.984 | 86 | **24.347** (24.433 − 86) |
| obs20_oportunidades_mejora | 63.417 | 46.714 | 511 | 16.192 |
| obs23_percepcion_progreso | 6.876 | 1.334 | 21 | 5.521 |
| obs24_aspectos_resaltar | 6.876 | 1.451 | 3 | 5.422 |
| obs25_oportunidades_mejora | 6.876 | 1.556 | 8 | 5.312 |

No se pudo probar aquí: `construir_unidades` y el resto del pipeline sobre la corrida real (faltan spaCy `es_core_news_lg`, torch, UMAP/HDBSCAN y GPU).

**3. Decisiones**
- 1.1: **retirar de las salidas y borrar el código muerto ya** (elegido por el usuario, en lugar de la opción recomendada de dejarlo inactivo).
- Solo evadoc: la corrida vigente es `20261119-224804_57895bd79b` (sin acreditación ni calidad).
- Se eliminó también la columna interna `palabra_alerta` de `documentos`/`clausulas` (parte del código muerto; no es del contrato).

**4. Riesgos y supuestos**
- La regla 7 pide no cambiar columnas internas salvo lo que indique el plan; se retiró `palabra_alerta` y las métricas `documentos_alerta`/`clausulas_alerta` por ser parte de lo que 1.1 mandó borrar. Si algún consumidor externo las leía (Etapa 3 y adaptador no las usan: se verificó con `grep`), habría que restituirlas.
- `huella()` de `RecursosLexicos` cambia (ya no incluye alertas): cambia el id de las corridas y de los modelos que dependan de ella.
- Un modelo U4 ya entrenado queda válido (no se tocó el modelo).
- `documents/ejemplos_salidas_nlp/` aún tiene `alertas.csv` y `metadatos_nlp.csv` del contrato anterior; se regenera al final (Punto 7), con tu aprobación.
- `src/limpieza/adaptador_nlp_actual.py` (Etapa 3) menciona alertas solo en un comentario; no se tocó.
- Las celdas 124–126 siguen describiendo `outputs/` y `SIN_ASPECTO` con origen `sistema`: se ajustan en los Puntos 2, 4 y 5.

**5. Commit sugerido (lo haces tú)**
`feat(nlp): ingesta solo con es_no_respuesta=False y tiene_alerta=False (Punto 1) — se retiran las alertas del NLP y de las salidas (4 tablas)`
(incluye `src/nlp_utilidades/modulos_nlp.py`, `src/nlp_utilidades/salidas_contrato.py`, `notebooks/Prisma_unificado.ipynb`, `tests/test_nlp_ingesta.py`, `tests/test_nlp_salidas.py`, `tests/test_nlp_alertas_retiradas.py` y este plan).

**6.** ¿Apruebas el Punto 1 para pasar al Punto 2?

### Punto 2 — Ruta de salida de modelado: `RAIZ/outputs` → `RAIZ/data/processed/nlp` (2026-11-19)

**1. Qué cambió**
- `src/nlp_utilidades/modulos_nlp.py`: `Config.output_dir` pasa de `outputs/ft` a `data/processed/nlp/modelado` (único default que escribía en `outputs`); docstrings de `exportar_resultado`/recalificación/auditorías actualizados; nueva `comprimir_carpeta(carpeta, destino_zip)` (zip que no se incluye a sí mismo ni a otros `.zip`, necesaria porque el zip queda dentro de la carpeta que comprime).
- Cuaderno (7 de 140 celdas; formato intacto): 7 (`RUTAS["salida"]` = `BASE_SALIDA/data/processed/nlp/modelado`; `RUTAS["nlp"]` sin cambio), 24 (comentario), 124, 131, 133 (textos), 132 (Paso 15: zip con `ft.comprimir_carpeta`), 139 (Paso 16: zip recalificado).
- Pruebas: `tests/test_nlp_salida_modelado.py` (6 nuevas).
- No se movió ni se borró nada: `outputs/` (vacía, 0 archivos) queda tal cual.

**Antes → después**

| Qué | Antes | Después |
|---|---|---|
| Salidas internas (`SALIDA`, corridas `<encuesta>/<run_id>/`, `auditoria_ingesta_etapa1.csv`) | `RAIZ/outputs` | `RAIZ/data/processed/nlp/modelado` |
| Tablas del contrato | `RAIZ/data/processed/nlp/<id_corrida_nlp>/` | igual |
| `Config.output_dir` (default) | `outputs/ft` | `data/processed/nlp/modelado` |
| Zip del Paso 15 (local) | `RAIZ/outputs_full.zip` | `…/nlp/modelado/outputs_full.zip` |
| Zip del Paso 16 (local) | `RAIZ/outputs_full_recalificado.zip` | `…/nlp/modelado/outputs_full_recalificado.zip` |
| Colab | `/content/outputs_full*.zip`; salidas en Drive o `/content` según `GUARDAR_EN_DRIVE` | igual (zips en `/content`; carpeta `modelado` en Drive o `/content`) |
| Caché de embeddings y modelos U4 | `models/…` | igual (ya no estaban en `outputs`) |

**2. Pruebas de humo**

| Comando / verificación | Resultado |
|---|---|
| `poetry run pytest` | **208 pasan, 0 fallan** (202 + 6 nuevas) |
| Celda de rutas ejecutada con `EN_COLAB = False` | `SALIDA` = `RAIZ/data/processed/nlp/modelado`, existe y cuelga de `RUTAS["nlp"]`; todas las `RUTAS` existen |
| Búsqueda de `outputs` como destino | 0 referencias en `modulos_nlp.py` y en las 62 celdas de código; 0 en las celdas de texto (`` `outputs/ ``) |
| `exportar_resultado` con un `Resultado` sintético | Escribe `clausulas.csv`, `manifest.json`, `scorecard.json` y `auditoria.xlsx` bajo `…/modelado/<encuesta>/<run_id>/`; sin raíz explícita usa el default de `Config`; no crea `outputs`; el glob `*/*/manifest.json` de los Pasos 15–16 lo encuentra |
| `comprimir_carpeta` | No se mete a sí mismo ni zips previos; `testzip` limpio; crea la carpeta de destino |
| `nbformat.validate`, `ast.parse` de las 62 celdas, celdas 6–22 con la corrida real | Pasan; auditoría de la ingesta con diferencia 0 |

No probado aquí: el Paso 15/16 en Colab (descarga, `files.upload`) y `exportar_resultado` con un `Resultado` real (faltan torch/spaCy/GPU); `clausulas.parquet` se omite porque `pyarrow` no está instalado.

**3. Decisiones**
- 2.1: estructura **`modelado/` aparte** (tablas del contrato en `nlp/<id_corrida_nlp>/`, internas en `nlp/modelado/<encuesta>/<run_id>/`).
- Zips: dentro de `nlp/modelado` (decisión adicional, elegida por el usuario).

**4. Riesgos y supuestos**
- El Excel de entrenamiento y los diagnósticos de U4 ya se escribían en `models/…` (carpeta del modelo), no en `outputs`; no cambian.
- Las corridas antiguas de `outputs/` (vacía hoy) no se migran; si apareciera alguna, el Paso 16 no la vería hasta copiarla a `modelado/`.
- Quien tenga un `outputs_full.zip` previo y lo suba al Paso 16 en Colab lo extrae en `modelado/` con la misma estructura `<encuesta>/<run_id>`: compatible.
- La carpeta `data/processed/nlp/modelado` quedó creada vacía al ejecutar la celda de rutas (carpeta de salida permitida por el plan).

**5. Commit sugerido (lo haces tú)**
`feat(nlp): salidas de modelado a data/processed/nlp/modelado (Punto 2) — RUTAS, Config.output_dir, comprimir_carpeta para los zips de los Pasos 15-16`
(incluye `src/nlp_utilidades/modulos_nlp.py`, `notebooks/Prisma_unificado.ipynb`, `tests/test_nlp_salida_modelado.py` y este plan).

**6.** ¿Apruebas el Punto 2 para pasar al Punto 3?

### Punto 3 — Archivos exportados sin datos de respuestas que no cumplen las condiciones (2026-11-19)

**1. Qué cambió**
- `src/nlp_utilidades/salidas_contrato.py`: dos funciones puras (pandas) nuevas: `ids_respuestas_excluidas(respuestas, preguntas=None)` (no-respuestas y respuestas con alerta) y `verificar_archivos_sin_excluidas(carpeta, respuestas)`, que revisa todos los CSV y todas las hojas de los XLSX de una carpeta (columnas `respuesta_id`, y `clausula_id`/`alerta_id` con forma `respuesta_id|n`) y falla con `ErrorSalidasNLP` (solo conteos, nunca textos) si encuentra una respuesta excluida. Devuelve una tabla por archivo/hoja con `columnas_id`, `filas_revisadas` y `excluidas`.
- Cuaderno (3 de 140 celdas): 122 (texto: los archivos solo traen respuestas procesadas; `respuestas.csv` del NLP será la excepción), 123 (tras escribir el Excel unificado, llama a la verificación sobre la carpeta de la corrida), 125 (la misma verificación sobre las tablas del contrato).
- Pruebas: `tests/test_nlp_archivos_sin_excluidas.py` (7 nuevas).
- `modulos_nlp.py` no cambió.

**Tabla archivo → fuente de datos → resultado del chequeo**

| Archivo (celda) | Fuente de datos | Chequeo |
|---|---|---|
| `documentos.csv`, `clausulas.csv`, `topicos.csv`, `absa.csv`, `grid.csv`, `topicos_residuales.csv`, `manifest.json`, `scorecard.json`, `auditoria.xlsx` (121, `exportar_resultado`) | `resultado`, construido solo con `encuestas` (ya filtradas por el Punto 1); ninguno lee `respuestas.csv` | `documentos.csv` trae `respuesta_id` y se verifica por id; los demás cuelgan de él por `doc_id` (declarado como `columnas_id = 0`) |
| `clausulas_documentos_unificado.xlsx` (123) | `documentos.csv` ⟕ `clausulas.csv` con `how="left"`: conserva documentos sin cláusulas (vacíos, ruido), que son respuestas procesables | Verificado por id: 0 excluidas; no reintroduce respuestas excluidas |
| `auditoria_ingesta_etapa1.csv` (24) | `comparar_con_metadatos_etapa1`: solo conteos | Sin ids ni textos; se mantiene (decisión 3.2) |
| `clausulas.csv`, `auditoria_clasificacion.csv`, `catalogo_aspectos.csv`, `metadatos_nlp.csv` (125) | `construir_salidas_contrato` sobre `resultado` | Verificado por `respuesta_id`/`clausula_id`; `validar_salidas_nlp` ya falla si una no-respuesta o una respuesta con alerta tiene cláusulas |
| zips (132, 139) | Carpeta `modelado` | Comprimen archivos ya verificados |
| `respuestas.csv` (Punto 6) | Etapa 1 completa | Excepción prevista: lleva todas las filas con `es_no_respuesta` y `tiene_alerta` |

**2. Pruebas de humo**

| Comando / verificación | Resultado |
|---|---|
| `poetry run pytest` | **215 pasan, 0 fallan** (208 + 7 nuevas) |
| Corpus sintético con las combinaciones de `es_no_respuesta`/`tiene_alerta` (ingesta → `construir_unidades` → `documentos.csv`, `clausulas.csv`, Excel unificado de la celda 123) | `documentos.csv` = las 3 respuestas procesables; 0 excluidas en todos los archivos; el unificado `left` no reintroduce ninguna |
| Pruebas negativas | Se detecta una excluida en un CSV, en una hoja de un XLSX de varias hojas (`auditoria.xlsx/muestra`) y en `clausulas.csv` de las tablas del contrato (por `respuesta_id` y por `clausula_id`); el mensaje solo trae conteos (no aparece el texto) |
| Tablas del contrato sintéticas | Pasan la verificación: `auditoria_clasificacion.csv` se revisa por `clausula_id` |
| Corrida real (`20261119-224804_57895bd79b`), ids del corpus que entra al modelado | 147.462 respuestas = 56.794 procesadas + 90.668 excluidas (90.039 no-respuestas + 629 con alerta); 0 excluidas en los 5 `documentos` (24.347 / 16.192 / 5.521 / 5.422 / 5.312 filas); el control negativo con una excluida real se detecta |
| `nbformat.validate`, `ast.parse` de las 62 celdas, celdas 6–22 con la corrida real | Pasan |

No probado: los archivos reales de modelado (aún no hay corrida con el modelo; faltan spaCy, torch, UMAP/HDBSCAN y GPU), por lo que el chequeo real se hizo sobre los ids que entran al modelado, no sobre archivos ya escritos. El cuaderno ejecutará la verificación en las celdas 123 y 125.

**3. Decisiones**
- 3.1 (S1): confirmada la lectura «los archivos exportados no muestran datos de las respuestas que NO cumplen las condiciones», con `respuestas.csv` (Punto 6) como única excepción.
- 3.2: `auditoria_ingesta_etapa1.csv` se mantiene (solo conteos).

**4. Riesgos y supuestos**
- La verificación por id no ve `clausulas.csv`, `topicos.csv`, etc. de modelado (solo `doc_id`); se confía en que nacen del mismo corpus que `documentos.csv`. Si más adelante se agrega `respuesta_id` a esos archivos, la función los cubre sola.
- Un XLSX de muchas hojas grandes se lee completo: la verificación puede tardar unos segundos con 24 mil filas.
- Si una hoja de `auditoria.xlsx` copiara texto de respuestas sin id, no se detectaría (hoy `muestra_manual` y `muestra_pureza` salen de `clausulas` del corpus filtrado).

**5. Commit sugerido (lo haces tú)**
`feat(nlp): verificar que los archivos exportados no traen respuestas excluidas (Punto 3) — verificar_archivos_sin_excluidas y comprobación en las celdas 123 y 125`
(incluye `src/nlp_utilidades/salidas_contrato.py`, `notebooks/Prisma_unificado.ipynb`, `tests/test_nlp_archivos_sin_excluidas.py` y este plan).

**6.** ¿Apruebas el Punto 3 para pasar al Punto 4?

### Punto 4 — `catalogo_aspectos.csv`, `clausulas.csv` y `taxonomia.xlsx` (2026-11-19)

**1. Qué cambió**
- `src/nlp_utilidades/salidas_contrato.py`:
  - **`SIN_ASPECTO`** sale siempre con `aspecto_origen = hdbscan` y, en el catálogo, `grupo_visual = «Sin grupo visual»` (nueva constante `GRUPO_SIN_GRUPO`). El origen `sistema` ya no se emite (se conserva en `ORIGENES` por compatibilidad). `no_opinion` y `ruido` pasan a `(SIN_ASPECTO, hdbscan)` y desaparecen de `clausulas.csv` con el Punto 5.
  - **Temas de HDBSCAN**: `aspecto_id` = primer sustantivo de `palabras_clave`. Funciones nuevas: `crear_pos_spacy` (spaCy `es_core_news_lg` en contexto: un `Doc` con los términos del grupo como frase corta; sin retokenizar), `parsear_palabras_clave`, `normalizar_id` (MAYÚSCULAS, sin tildes, Ñ→N, signos→`_`), `primer_sustantivo` (acepta `NOUN`, excluye `PROPN`; los términos de varias palabras se parten) e `id_de_tema` (resuelve choques). `construir_salidas_contrato` recibe `pos_fn` (inyectable; por defecto spaCy, solo si hay temas). Se eliminan `ParametrosSalidas.periodo_emergentes` y el contador `EMERGENTE_<periodo>_<n>`.
  - `aspecto_nombre` del tema = el sustantivo; `palabras_clave` se escribe separada por coma (antes podía salir con « | »).
  - `metadatos_nlp`: `pct_aspecto_hdbscan` cuenta solo temas y `pct_sin_aspecto` cuenta `SIN_ASPECTO`; las cuatro `pct_*` suman 100.
  - `validar_salidas_nlp`: `SIN_ASPECTO` ⇒ origen `hdbscan`; formato de todo `aspecto_id` (`PATRON_ID`); sin ids `EMERGENTE_<periodo>_<n>`; dominio de `grupo_visual` ampliado; `SIN_ASPECTO` ⇒ `hdbscan` / «Sin grupo visual»; temas emergentes ⇒ `hdbscan`; aspectos definidos ⇒ no `hdbscan`; las `pct_*` suman 100.
- Cuaderno (4 de 140 celdas): 4 (quita `periodo_emergentes`), 124 (explica la regla de IDs y `SIN_ASPECTO`), 125 (resumen con conteo de `SIN_ASPECTO`), 126 (texto).
- `taxonomia.xlsx` y `taxonomia.py` **no se tocan** (decisión 4.2).
- Pruebas: `tests/test_nlp_temas_hdbscan.py` (nuevo, 28 + 1 omitida) y ajustes a `tests/test_nlp_salidas.py`.

**2. Pruebas de humo**

| Comando / verificación | Resultado |
|---|---|
| `poetry run pytest` (incluye `test_taxonomia.py` y `test_limpieza.py`) | **243 pasan, 1 omitida, 0 fallan** (215 + 28 nuevas; la omitida es la de spaCy real, que no está instalado) |
| `python run_validar_taxonomia.py` | Pasa: 146 filas, 66 aspectos (sin cambios) |
| `SIN_ASPECTO` | Siempre `hdbscan` en `clausulas.csv`; catálogo con `hdbscan` / «Sin grupo visual»; ningún `sistema` |
| Temas con etiquetador inyectable | IDs = primer sustantivo; ningún `EMERGENTE_<periodo>_<n>`; todos cumplen `PATRON_ID`; sin duplicados `(encuesta, aspecto_id)`; todo `aspecto_id` de `clausulas` está en el catálogo; palabras clave separadas por coma |
| Colisiones | Con la taxonomía (`claridad` → `EMERGENTE_CLARIDAD`; también con la taxonomía de otra pregunta de la encuesta); entre temas de la misma pregunta → `CLASES`, `CLASES_2`, `CLASES_3`; mismo sustantivo en otra pregunta de la encuesta → un solo aspecto |
| Fallback y casos borde | Sin sustantivo → primer término + aviso con conteo; sin términos → `TEMA_<n>`; `sin aspecto` como sustantivo → `EMERGENTE_SIN_ASPECTO`; el etiquetador se llama una sola vez con todas las palabras (contexto); sin temas no pide spaCy; sin spaCy y con temas, falla con mensaje claro |
| 7 grupos sintéticos | Todos válidos y con ids únicos |
| Validador, pruebas negativas | Falla: `SIN_ASPECTO` con origen `sistema`; `SIN_ASPECTO` mal marcado en el catálogo; id antiguo `EMERGENTE_2025-1_0` o con formato inválido; tema emergente sin origen `hdbscan`; `pct_*` que no suman 100 |
| Determinismo | Dos construcciones con temas dan el mismo hash |
| `nbformat.validate`, `ast.parse` de 62 celdas, celdas 6–22 con la corrida real | Pasan |

**Ejemplo sintético** (solo IDs y orígenes; generado en el scratchpad, `documents/ejemplos_salidas_nlp/` no se tocó):

| aspecto_id | origen | grupo_visual |
|---|---|---|
| CLARIDAD, PUNTUALIDAD, OPINION_GENERAL | clasificador | Aspectos definidos |
| EMERGENTE_CLARIDAD (choque con la taxonomía) | hdbscan | Temas emergentes |
| CLASES, CLASES_2 (mismo sustantivo en dos temas) | hdbscan | Temas emergentes |
| PROPONE (sin sustantivo: primer término + aviso) | hdbscan | Temas emergentes |
| SIN_ASPECTO | hdbscan | Sin grupo visual |

En ese ejemplo, `pct_aspecto_clasificador` 45,45 · `pct_aspecto_hdbscan` 36,36 · `pct_aspecto_regla` 0 · `pct_sin_aspecto` 18,18 (suman 100). En `clausulas.csv`: cada cláusula con opinión sale con su aspecto del clasificador o su tema; la residual sin grupo y la cláusula sin opinión salen `SIN_ASPECTO` / `hdbscan`.

No probado aquí: spaCy real (`es_core_news_lg` no está instalado: falta validar en Colab que el etiquetador en contexto elija bien en grupos reales; hay una prueba lista que se activa sola cuando el modelo está disponible) y una corrida real con HDBSCAN.

**3. Decisiones** (con tus respuestas)
- 4.1 (S3): confirmada: `SIN_ASPECTO` = residual sin grupo, `hdbscan` / «Sin grupo visual».
- 4.2 (S4): **(b)** `taxonomia.xlsx` no se toca.
- 4.3: `pct_aspecto_hdbscan` = solo temas; `pct_sin_aspecto` = `SIN_ASPECTO`.
- 4.4: spaCy en contexto; 4.4b: fallback = primer término + aviso.
- 4.5: sufijo numérico entre temas de la misma pregunta; prefijo `EMERGENTE_` solo en choques con la taxonomía; 4.5b: el mismo sustantivo en otra pregunta de la encuesta se unifica en un solo ID.
- 4.6: `aspecto_nombre` = el sustantivo.

**4. Riesgos y supuestos**
- Al quitar el periodo del ID, un mismo sustantivo en distintas corridas se parece en el tablero aunque sea un grupo distinto; el contrato dice que los grupos cambian por corrida: se mantiene `grupo_visual = «Temas emergentes»` como separador.
- Plural y lemas: el ID usa el sustantivo tal como sale en `palabras_clave` (p. ej. `CLASES`, no `CLASE`); `clases` y `clase` serían temas distintos.
- Hasta el Punto 5, las cláusulas sin opinión salen como `SIN_ASPECTO` con origen `hdbscan` (semánticamente incorrecto, transitorio); `pct_sin_aspecto` las incluye.
- `palabras_clave` ahora siempre va separada por coma; el contrato v0.2 ya lo pedía.
- `documents/ejemplos_salidas_nlp/` sigue con el esquema anterior; se regenera al final con tu aprobación.

**5. Commit sugerido (lo haces tú)**
`feat(nlp): SIN_ASPECTO = residual de HDBSCAN y temas con aspecto_id del primer sustantivo (Punto 4) — normalizar_id, primer_sustantivo, pos_fn inyectable y validador`
(incluye `src/nlp_utilidades/salidas_contrato.py`, `notebooks/Prisma_unificado.ipynb`, `tests/test_nlp_temas_hdbscan.py`, `tests/test_nlp_salidas.py` y este plan).

**6.** ¿Apruebas el Punto 4 para pasar al Punto 5?

### Punto 5 — `clausulas.csv`: se conservan las cláusulas con y sin opinión (2026-11-19)

**Cambio de alcance (decisión del usuario).** El plan pedía filtrar `clausulas.csv` a `es_opinion == True`. Se descartó: **`clausulas.csv` conserva todos los registros, `es_opinion = True` y `False`**. Con eso, las preguntas 5.1 (supuesto S2) y 5.2 (`orden` con saltos) dejan de aplicar. Respuesta a «¿a qué subconjunto corresponden las cláusulas con opinión?»: las cláusulas de `clausulas.csv` ya son solo las de respuestas procesables (`es_no_respuesta = False` y `tiene_alerta = False`, Punto 1); las que tienen opinión (`es_opinion = True`, asignaciones `u4`, `residual_hdbscan`, `regla_comentario_general`, `valoracion_global`) son un subconjunto de ellas; la Etapa 1 no entrega cláusulas, solo respuestas.

**1. Qué cambió**
- `src/nlp_utilidades/salidas_contrato.py` (+25 / −5):
  - Hallazgo: las respuestas procesables que quedan **vacías o de ruido** tras limpiarse (`DOC_VACIA`/`DOC_RUIDO`) no generan ninguna cláusula, y el validador exigía ≥ 1 cláusula por procesable (habría fallado con datos reales; lectura del código, no probado con datos reales).
  - `SalidasContrato.sin_clausulas` (conjunto de `respuesta_id`) se llena desde `documentos.estado ∈ ESTADOS_SIN_CLAUSULAS = (VACIA, RUIDO)`; no agrega columnas a las tablas.
  - `validar_salidas_nlp(..., sin_clausulas=())`: toda procesable tiene ≥ 1 cláusula, salvo las declaradas; falla si una declarada sí tiene cláusulas, si no es procesable de las preguntas procesadas, o si falta una respuesta que no está declarada. `n_respuestas` de `metadatos_nlp` = procesadas = con cláusulas + declaradas.
- Cuaderno (3 de 140 celdas): 124 (texto: se conservan las cláusulas con y sin opinión; vacías/ruido), 125 (pasa `sin_clausulas=salidas.sin_clausulas` y muestra su conteo), 126 (texto).
- Pruebas: `tests/test_nlp_clausulas_con_y_sin_opinion.py` (10 nuevas).
- Sección 11 de este plan: el cambio 5 al contrato queda tachado y reescrito (no hay filtro; solo la excepción de cobertura).

**2. Pruebas de humo**

| Comando / verificación | Resultado |
|---|---|
| `poetry run pytest` | **253 pasan, 1 omitida, 0 fallan** (243 + 10 nuevas; la omitida es la de spaCy real) |
| `clausulas.csv` sintético | Conserva `True` y `False`; las sin opinión salen `SIN_ASPECTO` / `hdbscan` (regla del Punto 4), sin sentimiento; no se descarta ninguna cláusula |
| `orden` y auditoría | `orden` de 1 a n sin saltos por respuesta; `auditoria_clasificacion` 1:1 con `clausulas` |
| Vacías/ruido | Se declaran (`sin_clausulas` = 2 en el caso sintético), ninguna tiene cláusulas y `n_respuestas` = 10 (8 con cláusulas + 2 declaradas); la validación pasa |
| Pruebas negativas | Falla si no se declaran las vacías/ruido; si se declara una respuesta que sí tiene cláusulas; si se declara una no-respuesta; si falta una respuesta no declarada aunque haya otras declaradas; si `n_respuestas` no cuenta las declaradas |
| Sin estado en `documentos` | No declara nada (comportamiento anterior) |
| Determinismo | Dos construcciones, mismo hash |
| `nbformat.validate`, `ast.parse` de 62 celdas, celdas 6–22 con la corrida real | Pasan |

No probado: pipeline real (faltan spaCy, torch, UMAP/HDBSCAN y GPU), por lo que no se sabe cuántas respuestas vacías/ruido hay en obs19 (el plan estimaba ≈118 con todas sus cláusulas sin opinión, que ahora **sí** aparecen en `clausulas.csv`).

**3. Decisiones**
- 5.1 y 5.2: no aplican (no hay filtro). 5.1 se había respondido «filtrar por True», antes del cambio de alcance.
- Sin opinión: se mantiene `SIN_ASPECTO` con `aspecto_origen = hdbscan` y «Sin grupo visual» (regla del Punto 4); `es_opinion = False` las distingue del residual sin grupo.
- Cobertura: el validador declara las vacías/ruido (opción recomendada).

**4. Riesgos**
- `aspecto_origen = hdbscan` en cláusulas sin opinión no viene de HDBSCAN: se distingue solo por `es_opinion = False`.
- `pct_sin_aspecto` mezcla sin opinión y residual sin grupo; separarlos exigiría otra columna (cambio de contrato).
- Si `documentos` no trae la columna `estado` (otro origen), las vacías/ruido no se declaran y la validación las reporta como faltantes.

**5. Commit sugerido (lo haces tú)**
`feat(nlp): clausulas.csv conserva cláusulas con y sin opinión (Punto 5) — el validador declara las respuestas vacías/ruido sin cláusulas`
(incluye `src/nlp_utilidades/salidas_contrato.py`, `notebooks/Prisma_unificado.ipynb`, `tests/test_nlp_clausulas_con_y_sin_opinion.py` y este plan).

**6.** ¿Apruebas el Punto 5 para pasar al Punto 6?

### Punto 6 — Excel unificado junto a `clausulas.csv` y `metadatos_nlp.csv` (2026-11-19)

**Cambio de alcance (decisión del usuario).** El plan pedía escribir `respuestas.csv` desde el NLP. Se **ignora** (ya existe el de la Etapa 1; la validación contra él ya está en `validar_salidas_nlp` y `verificar_archivos_sin_excluidas`). En su lugar, el Excel que consolida cláusulas y respuestas (`clausulas_documentos_unificado.xlsx`) se **traslada a la carpeta de la corrida del NLP** (`data/processed/nlp/<id_corrida_nlp>/`), junto a `clausulas.csv` y `metadatos_nlp.csv`, con las banderas `es_no_respuesta` y `tiene_alerta` al final (decisión de contenido: «igual que hoy + las dos banderas»).

**1. Qué cambió**
- `src/nlp_utilidades/salidas_contrato.py` (+58): `ARCHIVO_UNIFICADO`, `HOJA_UNIFICADO`, `construir_unificado(documentos, clausulas, respuestas)` (la misma unión de la celda 123: `left` por `doc_id`, conserva documentos sin cláusulas; + `es_no_respuesta`/`tiene_alerta` por `respuesta_id`; elimina caracteres de control; falla con `ErrorSalidasNLP` si la unión cambia el número de filas, si un `respuesta_id` no está en `respuestas.csv` o si llega una no-respuesta o una respuesta con alerta) y `escribir_unificado(unificado, carpeta)`. El Excel **no** entra en `ARCHIVOS` ni en `huella_archivos` (el XLSX no es determinista byte a byte).
- Cuaderno (5 de 140 celdas; formato intacto): 122 (texto), 123 (en modo `u4` ya no escribe el Excel; en `bertopic` lo escribe en la carpeta de modelado; mantiene la verificación del Punto 3), 124 y 126 (textos), 125 (escribe el Excel en `CARPETA_NLP` tras validar y lo cubre `verificar_archivos_sin_excluidas(CARPETA_NLP, …)`).
- Pruebas: `tests/test_nlp_excel_unificado.py` (7 nuevas); `tests/test_nlp_archivos_sin_excluidas.py` ahora usa `construir_unificado`/`escribir_unificado`.
- Sección 11 de este plan: el cambio 6 queda tachado y reescrito.

**2. Pruebas de humo**

| Comando / verificación | Resultado |
|---|---|
| `poetry run pytest` | **260 pasan, 1 omitida, 0 fallan** (253 + 7 nuevas; la omitida es la de spaCy real) |
| Unión sintética | Una fila por cláusula + el documento sin cláusulas; las dos banderas al final, siempre `False`; mismas columnas de hoy (`p_u4_*`, `asignacion`…) |
| Casos negativos | Falla con bandera `True` (no-respuesta o alerta), con `respuesta_id` ausente de `respuestas.csv`, con columna faltante en `respuestas`/`documentos` |
| Caracteres de control | Se eliminan; el Excel se escribe sin `IllegalCharacterError` |
| Ubicación | El Excel queda en la misma carpeta que `clausulas.csv` y `metadatos_nlp.csv`; no cambia el hash de los 4 archivos del contrato |
| Verificación del Punto 3 | Cubre el Excel por `respuesta_id` (detecta una excluida colada) |
| Compatibilidad Etapa 3 | `adaptador_nlp_actual.py` lee por nombre (`usecols`): columnas nuevas no lo rompen (prueba de lectura con `usecols`) |
| Corrida real `20261119-224804_57895bd79b` (obs19, ids reales + cláusulas sintéticas) | 24.347 documentos → 48.694 filas; banderas solo `False`; Excel de 1,3 MB; 0 excluidas |
| `nbformat.validate`, `ast.parse` de 62 celdas, celdas 6–22 con la corrida real | Pasan |

No probado: el pipeline real (faltan spaCy, torch, UMAP/HDBSCAN y GPU); la celda 125 completa no se ejecutó.

**3. Decisiones**
- `respuestas.csv` del NLP: se ignora.
- Contenido: A + `es_no_respuesta` y `tiene_alerta`.
- Modo `bertopic` (sin carpeta `<id_corrida_nlp>`): el Excel queda en la carpeta de modelado.

**4. Riesgos**
- En modo `u4`, si la celda 125 falla antes de escribirlo, no queda Excel (ya no se escribe en la celda 123).
- El Excel lleva `documentos.csv` y `clausulas.csv` internos, que traen `respuesta_completa` y los textos de las cláusulas (datos del cliente): vive en `data/processed/` (fuera de git).
- Excel admite ~1,05 M de filas; obs19 produce ~49 mil.

**5. Commit sugerido (lo haces tú)**
`feat(nlp): Excel unificado junto a clausulas.csv y metadatos_nlp.csv con es_no_respuesta y tiene_alerta (Punto 6) — construir_unificado y escribir_unificado`
(incluye `src/nlp_utilidades/salidas_contrato.py`, `notebooks/Prisma_unificado.ipynb`, `tests/test_nlp_excel_unificado.py`, `tests/test_nlp_archivos_sin_excluidas.py` y este plan).

**6.** ¿Apruebas el Punto 6 para pasar al Punto 7?

### Punto 7 — Unificar `VALORACIÓN GLOBAL`, `OPINIÓN GENERAL` y `COMENTARIO GENERAL` (2026-11-19)

**1. Qué cambió**
- `src/nlp_utilidades/modulos_nlp.py` (+39 / −8):
  - Nombre único: `NOMBRE_OPINION_GENERAL = "Opinión general"`; `ETIQUETAS_ESPECIALES[-3]` y `[-5]` ya no se llaman «VALORACIÓN GLOBAL (sin aspecto)» ni «Comentario General» (`NOMBRE_COMENTARIO_GENERAL` queda como alias). Los códigos internos -3/-5 se conservan (scorecard y ABSA sin cambios; las tablas se indexan por `topico`).
  - **7.2 (activado):** `Config.sin_contenido_en_cortas` (por defecto `True`, entra en `_CAMPOS_U4`, en la huella solo en modo u4 y en el manifiesto): la cláusula sin contenido (`VALORACION_GLOBAL`) de una respuesta **corta** ya no pasa por U4 y va a «Opinión general»; con `False` se comporta como antes. `conciliar` solo cuenta las globales cortas que sí pasaron por U4 (`topico != -3`).
  - Columna `motivo_opinion_general` en `clausulas` (`sin_contenido` = -3, `comentario_corto` = -5, `clasificador` = U4 predijo «Opinión general»/«Sin aspecto»/«Ninguno de la lista»; vacía en el resto). **Actualización posterior (a pedido del usuario): también va en `clausulas.csv` del contrato** (ver «Ajuste posterior» al final del informe).
- `src/nlp_utilidades/salidas_contrato.py` (+25 / −5): `CLASES_SIN_ASPECTO` (espejo de `CLASES_SIN_ASPECTO_U4`); **7.3**: «Sin aspecto» y «Ninguno de la lista» que U4 predijera salen `OPINION_GENERAL` / `clasificador` (sin ellas en la taxonomía ya no falla «clase sin aspecto_id»; el validador no las exige en la taxonomía); `_top3` suma las probabilidades de clases que comparten `aspecto_id` para no repetir aspectos. Las tablas del contrato **no cambiaban de columnas** en la primera entrega (ver el ajuste posterior).
- Cuaderno (6 de 140 celdas): 4 (`sin_contenido_en_cortas=True` en el perfil y en `campos_config`), 25, 43, 69, 124, 126 (textos: una sola categoría, precedencia, conciliación).
- `documents/propuesta_cambios_contrato_v03.md` (nuevo, §11 de este plan): los 9 cambios propuestos al contrato y lo que queda sin probar.
- `documents/ejemplos_salidas_nlp/` regenerado con datos sintéticos (4 tablas, esquema nuevo; se elimina `alertas.csv`; incluye un tema emergente `DINAMICAS`, `SIN_ASPECTO` con y sin opinión, y `OPINION_GENERAL` con origen regla y clasificador); pasa `validar_salidas_nlp`.
- Pruebas: `tests/test_nlp_opinion_general.py` (12 nuevas).

**2. Pruebas de humo**

| Comando / verificación | Resultado |
|---|---|
| `poetry run pytest` | **272 pasan, 1 omitida, 0 fallan** (260 + 12 nuevas; la omitida es la de spaCy real) |
| `etiquetar_corpus_u4` con un U4 simulado (5 cláusulas: sin contenido corta, comentario corto, U4 predice «Opinión general», aspecto normal, ruido) | Sin contenido → -3, `valoracion_global`, `sin_contenido`, sin pasar por U4 (`p_aspecto_u4` vacío); comentario corto → -5, `comentario_corto`; U4 «Opinión general» → `clasificador`; aspecto normal → motivo vacío; las tres con nombre «Opinión general» |
| Parámetro apagado | La cláusula sin contenido de respuesta corta pasa por U4 (p = 0,80 → «Opinión general», `clasificador`): comportamiento anterior |
| Conciliación (`conciliar`) | Todos los `chequeo` valen 1 con la regla encendida y apagada |
| Huella | `sin_contenido_en_cortas` cambia la huella en modo u4 y no en bertopic |
| Salidas del contrato | Las tres vías salen `OPINION_GENERAL` (origen `regla`, `regla`, `clasificador`); un solo `OPINION_GENERAL` en el catálogo; (primera entrega) las tablas conservaban sus columnas; ver el ajuste posterior |
| 7.3 sintético | Con U4 prediciendo «Sin aspecto» → `OPINION_GENERAL`; el top-3 de la auditoría no repite aspectos (confianza 0,1 + 0,5 + 0,2 = 0,8); el validador acepta esas clases fuera de la taxonomía; una clase desconocida sigue fallando |
| Taxonomías | «Opinión general» está en las 6; «Sin aspecto» y «Ninguno de la lista» en ninguna (y en ninguna etiqueta manual: la prueba de la Fase 1 exige que toda etiqueta esté en la taxonomía) |
| `campos_config` de los 6 perfiles → `Config` | Válido; `sin_contenido_en_cortas = True` en todos |
| `nbformat.validate`, `ast.parse` de 62 celdas, celdas 6–22 con la corrida real | Pasan |

**No probado:** el efecto real de `sin_contenido_en_cortas = True` (cuántas cláusulas sin contenido de respuestas cortas U4 asignaba a un aspecto real) exige el modelo U4 entrenado y GPU; **procedimiento para Colab:** correr obs19 dos veces (`sin_contenido_en_cortas` en `True` y en `False`, mismo modelo), y comparar de `clausulas` internas las cláusulas con `motivo = VALORACION_GLOBAL` y `respuesta_corta = True`: cuántas el modo `False` asigna a un aspecto real (`aspecto_u4` ≠ «Opinión general», `asignacion = u4`) y cuántas a «Opinión general»; solo conteos. Tampoco se probó el pipeline completo (faltan spaCy, torch, UMAP/HDBSCAN y GPU) ni el sentimiento de las cláusulas -3 de respuestas cortas (no cambia: sigue calculándose para toda cláusula con opinión).

**3. Decisiones** (con tus respuestas)
- 7.1: **solo la Etapa A** (una categoría de cara al usuario + columna interna `motivo_opinion_general`; códigos -3/-5 intactos; Etapa B —colapsar códigos y métricas— no se hace).
- 7.2: **activarlo ya** (en lugar de la opción recomendada, que era dejarlo apagado hasta medir); queda como parámetro `sin_contenido_en_cortas` para apagarlo.
- 7.3: «Sin aspecto» y «Ninguno de la lista» entran como Opinión general.
- Ejemplos sintéticos: regenerados (aprobado).

**4. Riesgos**
- **7.2 activo sin medir:** cambia resultados del modelo para las cláusulas sin contenido de respuestas cortas (antes podían terminar en un aspecto real). Reversible con el parámetro; hay que medirlo en Colab.
- **Nombres duplicados en tablas internas:** `topicos.csv`/ABSA pueden mostrar varias filas llamadas «Opinión general» (códigos -3, -5 y la clase de U4); se distinguen por `topico`. Ningún código indexa por el nombre (verificado con búsqueda).
- La huella de las corridas en modo u4 cambia (campo nuevo).
- Los textos del cuaderno que siguen hablando de «Comentario General» y «Valoración global» (celdas 43, 51, 55, 57, 67, 111, 113, 118) describen el **mecanismo** interno (códigos -3/-5) y se conservan; lo que se publica es «Opinión general».
- `metricas` y gates del scorecard (`pct_comentario_general`, `pct_valoracion_global`) no cambian de nombre.

**5. Commit sugerido (lo haces tú)**
`feat(nlp): VALORACIÓN GLOBAL, COMENTARIO GENERAL y OPINIÓN GENERAL como una sola categoría (Punto 7) — motivo_opinion_general, sin_contenido_en_cortas, clases sin aspecto, propuesta de contrato v0.3 y ejemplos sintéticos`
(incluye `src/nlp_utilidades/modulos_nlp.py`, `src/nlp_utilidades/salidas_contrato.py`, `notebooks/Prisma_unificado.ipynb`, `tests/test_nlp_opinion_general.py`, `documents/propuesta_cambios_contrato_v03.md`, `documents/ejemplos_salidas_nlp/` y este plan).

**6.** ¿Apruebas el Punto 7? Con él terminan los 7 puntos del plan v2.

**Ajuste posterior al Punto 7 (a pedido del usuario, 2026-11-19): `motivo_opinion_general` va en `clausulas.csv`.**
- `salidas_contrato.py`: `COLUMNAS["clausulas"]` incluye `motivo_opinion_general` **después de `aspecto_origen`**; valores `sin_contenido` | `comentario_corto` | `clasificador` (`MOTIVOS_OPINION_GENERAL`, espejo de `modulos_nlp`); se calcula en `construir_salidas_contrato` desde la `asignacion` interna (`valoracion_global` → `sin_contenido`; `regla_comentario_general` → `comentario_corto`; cualquier otra asignación que termine en `OPINION_GENERAL`, es decir la clase que predice U4 → `clasificador`) y queda **vacía** si el aspecto no es `OPINION_GENERAL` (también si el mapeo configurable manda una regla a otro aspecto). Solo `clausulas` la lleva (es la tabla con una fila por cláusula y aspecto); `auditoria`, `catalogo` y `metadatos` no cambian.
- `validar_salidas_nlp`: dominio de la columna; obligatoria con `OPINION_GENERAL` y vacía en el resto; coherencia con el origen (`regla` ⇒ `sin_contenido`/`comentario_corto`; `clasificador` ⇒ `clasificador`).
- Cuaderno (celdas 43, 124, 126: textos); `documents/propuesta_cambios_contrato_v03.md` (cambio 8: columna nueva); `documents/ejemplos_salidas_nlp/` **regenerado** (4 tablas; `clausulas.csv` con la columna: `OPINION_GENERAL` con `clasificador` ×2, `comentario_corto` ×1 y `sin_contenido` ×1; el resto vacío; pasa el validador).
- Pruebas: `tests/test_nlp_opinion_general.py` pasa de 12 a 18 (columna en el contrato, vacía fuera de `OPINION_GENERAL`, coincide con la columna interna del pipeline, relectura como texto sin «NA», 5 pruebas negativas del validador y mapeo a otro aspecto).
- `poetry run pytest`: ver el recuento más abajo en la respuesta de cierre; riesgo nuevo: Power BI/Etapa 3 que lean `clausulas.csv` por posición verían la columna corrida (deben leer por nombre; contrato v0.3 pendiente).

### Punto 8 — Excel de entrenamiento de U4: `_test` (hold-out) y `_train` (entrenamiento) (2026-11-19)

**Punto añadido por el usuario tras aprobar el Punto 7.** 8.1: renombrar `entrenamiento_u4.xlsx` (resultados y métricas en el hold-out) a `entrenamiento_u4_test.xlsx`. 8.2: generar `entrenamiento_u4_train.xlsx` con la estructura de `entrenamiento_u4.xlsx` para los resultados y métricas en el conjunto de entrenamiento.

**1. Qué cambió** (`src/nlp_utilidades/modulos_nlp.py`, sin tocar el modelo, sus hiperparámetros, `metricas.json`, `holdout_predicciones.csv` ni la huella)
- Constantes `ARCHIVO_ENTRENAMIENTO_TEST = "entrenamiento_u4_test.xlsx"` y `ARCHIVO_ENTRENAMIENTO_TRAIN = "entrenamiento_u4_train.xlsx"`; se sustituyen los 3 usos del nombre antiguo (la escritura y las dos de `etiquetar_corpus_u4`, que añade las hojas `diag_cg_<fecha>`/`diag_res_<fecha>` al `_test`).
- `_tabla_predicciones_u4(d, idx, y, P, clases)`: helper que arma la tabla de predicciones (extraída del código en línea; mismo resultado para el hold-out).
- `_escribir_excel_entrenamiento(..., parte="test"|"train")`: mismas 8 hojas y columnas en ambos (`resumen`, `recall_por_clase`, `confusion_conteos`, `confusion_normalizada`, `predicciones`, `datos_usados`, `errores_frecuentes`, `config`); el `_test` conserva además el marcador `diagnostico_comentario_general`. En `_train` el `resumen` usa `acierto_train`, `f1_macro_train`, `f1_macro_train_todas_las_clases` y `n_train_evaluable`, y su nota aclara que no mide generalización.
- `entrenar_u4`: tras la evaluación del hold-out evalúa las filas `rol = entrenamiento` evaluables (`usa_evaluacion`) con `_softmax_lotes` y `_metricas_holdout` (genérica), registra una línea de log («ENTRENAMIENTO (no mide generalización)…») y escribe el `_train` junto al `_test`.
- Cuaderno (2 de 140 celdas): 55 (lee las matrices de `ft.ARCHIVO_ENTRENAMIENTO_TEST`) y 122 (texto con los dos archivos).
- Pruebas: `tests/test_nlp_entrenamiento_excel.py` (11 nuevas).

**2. Pruebas de humo**

| Comando / verificación | Resultado |
|---|---|
| `poetry run pytest` | **289 pasan, 1 omitida, 0 fallan** (278 + 11 nuevas; la omitida es la de spaCy real) |
| Escritura sintética de `_test` y `_train` | Se crean los dos archivos y no el nombre antiguo; mismas hojas y mismas columnas hoja por hoja (el `_test` además trae el marcador de diagnóstico) |
| `resumen` | El `_test` conserva sus campos `*_holdout`; el `_train` usa `*_train` (n = 8 filas de entrenamiento frente a 2 de hold-out), nota propia, y el resto del resumen (clave, run_id, modelo, md5, filas, época) es igual |
| Predicciones de train | Solo filas con `rol = entrenamiento`; coinciden con la hoja `predicciones`; la tabla del helper tiene las columnas esperadas y funciona vacía |
| Diagnósticos del corpus | `etiquetar_corpus_u4` (U4 simulado) añade `diag_cg_*`/`diag_res_*` al `_test`, no toca el `_train` ni crea el nombre antiguo; sin el `_test` no escribe nada |
| Casos límite | `parte` inválida falla con mensaje claro; conjunto de entrenamiento vacío escribe el Excel sin filas |
| Nombre antiguo | 0 usos como archivo en el módulo y en el cuaderno |
| `nbformat.validate`, `ast.parse` de 62 celdas, celdas 6–22 con la corrida real | Pasan |

**No probado:** `entrenar_u4` completo (faltan torch, sentence-transformers y GPU). **Procedimiento para Colab:** entrenar una pregunta con `reutilizar_modelo = False`, comprobar que la carpeta del modelo trae `entrenamiento_u4_test.xlsx` y `entrenamiento_u4_train.xlsx` (y no `entrenamiento_u4.xlsx`), que `n_train_evaluable` = filas `entrenamiento` evaluables de `datos_usados`, y comparar `acierto_train` con `acierto_holdout` (un train mucho mayor indica sobreajuste).

**3. Decisiones** (con tus respuestas)
- 8.2 evalúa **solo las filas `rol = entrenamiento`** (evaluables, igual criterio que el hold-out); la validación interna no se evalúa en ningún archivo.
- 8.2 **omite la hoja `diagnostico_comentario_general`** en `_train`.

**4. Riesgos**
- Un modelo que se **reutiliza** (misma huella) no se reentrena: no se genera `_train`, y los modelos entrenados antes en otras máquinas conservan `entrenamiento_u4.xlsx`, por lo que `etiquetar_corpus_u4` no les añadirá los diagnósticos (no hay modelos en este repo: `models/modelos` está vacía).
- Un pase de inferencia más por las filas de entrenamiento (segundos en GPU).
- Los fragmentos de entrenamiento tienen más peso en las métricas del `_train` que ningún otro grupo; no sirve para comparar modelos, solo para detectar sobreajuste.

**5. Commit sugerido (lo haces tú)**
`feat(nlp): Excel de métricas de U4 en hold-out (entrenamiento_u4_test.xlsx) y en entrenamiento (entrenamiento_u4_train.xlsx) (Punto 8)`
(incluye `src/nlp_utilidades/modulos_nlp.py`, `notebooks/Prisma_unificado.ipynb`, `tests/test_nlp_entrenamiento_excel.py` y este plan).

**6.** ¿Apruebas el Punto 8?
