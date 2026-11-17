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
