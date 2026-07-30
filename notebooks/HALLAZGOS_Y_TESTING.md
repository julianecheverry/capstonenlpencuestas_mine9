# Informe de hallazgos y testing -- pipeline de limpieza de encuestas

**Módulos evaluados:** `settings.py`, `datacleaning/limpieza.py` (clase `Cleaner`), `datacleaning/synonyms.py` (clase `SynonymReplacer`)
**Entorno de prueba:** Python 3.12, pandas 3.0.3, nltk 3.9.4, openpyxl 3.1.5, gensim 4.4.0, pytest 9.1.1, gestionado con Poetry
**Resultado de la suite:** `94 passed` -- 0 fallidos
**Cobertura de código:** 99% combinado (`limpieza.py`: 97%, `synonyms.py`: 100%, `settings.py`: 100%)
**Alcance:** esta es la **tercera revisión** de este proyecto. Desde la revisión anterior, el proyecto creció significativamente: se reorganizó la estructura de carpetas (`src/datacleaning/` para los módulos de limpieza, `src/settings.py` como configuración central), se incorporó `synonyms.py` (normalización de sinónimos vía Word2Vec) y se amplió `settings.py` para centralizar todo parámetro potencialmente ajustable, leyendo configuración desde archivos Excel reales del repositorio (`data/utilities/accents.xlsx` y `data/utilities/stopwords.xlsx`). `EDA.py` queda fuera del alcance de esta suite, por decisión explícita del equipo.

**Cómo correr esta suite:**
```bash
poetry install
poetry run python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
poetry run pytest --cov=src --cov-report=term-missing
```

**Nota sobre alcance de cobertura en el repo completo:** al correr `--cov=src` en el repositorio real (que incluye `EDA.py`, `Engineering.py`, `Mod_Bertopic.py`), esos archivos aparecerán con 0% de cobertura. Esto es esperado y no debe interpretarse como un descuido: están fuera del alcance de esta suite de tests, por acuerdo con el equipo.

---

## 1. Resumen ejecutivo

De los hallazgos documentados en la revisión anterior (sobre la versión de `limpieza.py` sin `synonyms.py` ni el `settings.py` actual), **todos quedaron resueltos**. Esta revisión confirma en ejecución **5 hallazgos nuevos**, incluyendo uno de severidad alta que afecta directamente la funcionalidad de stopwords globales.

### Hallazgos de la revisión anterior -- todos resueltos

| # | Hallazgo (revisión anterior) | Estado en esta revisión |
|---|---|---|
| 1 | `eliminate_stopwords` dependía de `self.cleaned_data`, sin auto-generación | ✅ **Resuelto** -- el bloque de auto-generación (`if not hasattr(self, 'cleaned_data')...`) ya está activo, sin comentar |
| 2 | `save_cleaned_data` no llamaba a `clean_key_column` internamente, fallaba con `AttributeError` críptico si se llamaba primero | ✅ **Resuelto** -- ahora lanza `ValueError` con un mensaje instructivo indicando qué método correr antes |
| 3 | Faltaba el recurso `punkt_tab` de NLTK, solo se descargaba `punkt` | ✅ **Resuelto** -- `punkt_tab` se descarga explícitamente a nivel de módulo |
| 4 | El docstring de `clean_key_column` decía `Returns: None` pese a devolver un DataFrame | ✅ **Resuelto** -- el docstring ahora documenta correctamente el retorno |
| 5 | El diccionario `accents` estaba duplicado, hardcodeado tanto en `settings.py` como dentro de `limpieza.py` | ✅ **Resuelto** -- `limpieza.py` ahora importa `accents` desde `settings`, que a su vez lo carga desde `data/utilities/accents.xlsx` |
| 6 | Código muerto: ramas `else: raise ValueError(...)` en `clean_key_column` y `save_cleaned_data`, inalcanzables porque `load_data()` nunca devuelve `None` | ✅ **Resuelto** -- ambas ramas `else` fueron eliminadas del código |

### Hallazgos nuevos, confirmados en esta revisión

| # | Hallazgo nuevo | Severidad |
|---|---|---|
| 1 | `stop_words.update(stopwords_global)` agrega las **claves** del diccionario, no las palabras -- las stopwords globales del Excel nunca se aplican al texto | **Alta** |
| 2 | Los coloquialismos multi-palabra en `settings.colombian_colloquialisms` (`"una chimba"`, `"una nota"`, `"un paseo"`) son inalcanzables por cómo tokeniza `SynonymReplacer.replace_synonyms` | Media |
| 3 | `settings.py` viola la regla de idioma del equipo: la función `cargar_stopwords_desde_excel`, su docstring y sus comentarios están en español | Baja (regla del equipo) |
| 4 | `settings.py` hace I/O (lee 2 archivos Excel) al momento de importarse, y `cargar_stopwords_desde_excel` atrapa cualquier excepción, la imprime por consola, y devuelve `{}` en silencio | Media |
| 5 | La firma de `cargar_stopwords_desde_excel` declara `sheet_name: int`, pero en la práctica recibe strings (`'particulares'`, `'globales'`) | Muy baja |

---

## 2. Metodología

Se mantienen los cuatro principios ya establecidos en revisiones anteriores (comportamiento sobre implementación, confirmación en ejecución antes de documentar, tests unitarios complementados con integración, y reconfirmación activa de hallazgos previos -- nunca asumir que siguen vigentes o que están resueltos sin volver a probarlos).

Para esta revisión se añade un quinto principio, específico del crecimiento del proyecto:

5. **Los hallazgos sobre datos de configuración (Excel) se verifican contra el contenido real del repositorio, no contra datos sintéticos.** Donde fue posible, los tests que dependen del contenido de `accents.xlsx` o `stopwords.xlsx` se escribieron para funcionar correctamente sin importar qué contengan esos archivos (parametrizando sobre su contenido real, o usando nombres de encuesta garantizados como inexistentes). El hallazgo de las stopwords globales fue además confirmado manualmente contra el contenido real de `stopwords.xlsx` del equipo, no solo contra un ejemplo construido para la prueba.

Un cambio de infraestructura de testing, sin tocar código fuente: se agregó una sección `[tool.coverage.report]` en `pyproject.toml` con `exclude_also`, excluyendo del cálculo de cobertura las funciones `_download_model` y `load_model` de `synonyms.py`. Ambas requieren descargar (~1GB) y cargar un modelo Word2Vec real desde internet, algo inviable de ejercitar en una suite automatizada. El resto de `synonyms.py` sí se prueba en su totalidad, usando un modelo simulado (mock). Esta exclusión es una decisión de configuración de tests, documentada con comentarios en el propio `pyproject.toml` -- no modifica `synonyms.py`.

---

## 3. Hallazgos nuevos, en detalle

### Hallazgo 1 -- BUG (severidad alta): las stopwords globales nunca se aplican

**Código relevante (`limpieza.py`, línea 20, a nivel de módulo):**
```python
stop_words = set(nltk.corpus.stopwords.words("spanish"))
stop_words.update(stopwords_global)
```

**Problema:** `stopwords_global` (definido en `settings.py`) no es una lista de palabras -- es un diccionario, resultado de agrupar la hoja `'globales'` de `stopwords.xlsx` por la columna `encuesta`. Cuando se llama `set.update()` pasándole un diccionario completo, Python itera sobre sus **claves**, no sobre sus valores. El resultado: la clave del diccionario entra al conjunto de stopwords, y las palabras reales -- que están en los valores -- nunca llegan.

**Confirmación en ejecución, contra el contenido real de `stopwords.xlsx` del equipo:**

La hoja `'globales'` del Excel real tiene la siguiente estructura: todas las filas usan el mismo valor, `'stopwords_global'`, en la columna `encuesta`, junto a 8 palabras reales en la columna `palabra`: `nr`, `academico`, `academia`, `universidad`, `facultad`, `ademas`, `tambien`, `ma`.

```python
>>> from settings import stopwords_global
>>> stopwords_global
{'stopwords_global': ['nr', 'academico', 'academia', 'universidad', 'facultad', 'ademas', 'tambien', 'ma']}

>>> demo = set()
>>> demo.update(stopwords_global)
>>> demo
{'stopwords_global'}
>>> 'nr' in demo
False
>>> 'academico' in demo
False
>>> 'universidad' in demo
False
```

La única "palabra" que termina en el conjunto de stopwords es el string literal `'stopwords_global'` -- que no es una palabra real y nunca va a aparecer en el texto de una encuesta. Las 8 palabras reales, que el equipo definió explícitamente para filtrarse de **todas** las encuestas, nunca se aplican a ningún texto procesado por el pipeline.

**Por qué esto importa más allá del bug en sí:** varias de las palabras afectadas (`universidad`, `academico`, `facultad`) son justamente el tipo de término institucional genérico que uno esperaría filtrar de cualquier encuesta del dominio académico -- es decir, el caso de uso más evidente para el que existe la hoja `'globales'`. El bug no es un caso límite raro: neutraliza por completo la funcionalidad que la hoja fue diseñada para proveer.

**Test que documenta el hallazgo (`tests/test_stopwords.py`):**
```python
def test_set_update_over_a_dict_adds_keys_not_word_lists():
    demo = set()
    demo.update({"stopwords_global": ["nr", "academico", "universidad"]})

    assert demo == {"stopwords_global"}   # la clave entro al set
    assert "nr" not in demo               # las palabras reales, no
    assert "academico" not in demo
    assert "universidad" not in demo


def test_global_stopword_words_never_reach_the_module_stop_words():
    # Parametrizado sobre el contenido REAL de settings.stopwords_global
    if not stopwords_global:
        pytest.skip("No global stopwords defined in stopwords.xlsx")

    nltk_spanish = set(limpieza.nltk.corpus.stopwords.words("spanish"))

    for key, words in stopwords_global.items():
        assert key in limpieza.stop_words   # la clave, si
        for word in words:
            if word not in nltk_spanish and word != key:
                assert word not in limpieza.stop_words   # las palabras, no
```

**Solución propuesta (pendiente):**
```python
# Opcion simple, si solo se espera un grupo bajo la clave 'stopwords_global':
stop_words.update(stopwords_global.get('stopwords_global', []))

# Opcion general, si en el futuro pudiera haber mas de un grupo global:
stop_words.update(
    word for words in stopwords_global.values() for word in words
)
```

---

### Hallazgo 2 -- Coloquialismos multi-palabra inalcanzables en `SynonymReplacer`

**Código relevante (`settings.py`):**
```python
colombian_colloquialisms = {
    "bacano": "bueno",
    "chevere": "bueno",
    "chimba": "bueno",
    "una nota": "bueno",
    "berraco": "bueno",
    "una chimba": "bueno",
    ...
    "un paseo": "bueno",
}
```

**Código relevante (`synonyms.py`, `replace_synonyms`):**
```python
words = nltk.word_tokenize(text)
replaced_words = [
    self.synonyms_dict.get(word.lower(), word)
    for word in words
]
```

**Problema:** `replace_synonyms` tokeniza el texto **palabra por palabra** con `nltk.word_tokenize`, y busca cada token individual en el diccionario. Las claves de `colombian_colloquialisms` que contienen un espacio (`"una nota"`, `"una chimba"`, `"un paseo"`) nunca pueden coincidir con un token individual -- un token nunca va a ser igual a una frase de dos o tres palabras.

**Confirmación en ejecución** (con un modelo Word2Vec simulado, ya que el mecanismo no depende de las predicciones reales del modelo):
```python
>>> replacer.replace_synonyms("el curso fue una chimba")
'el curso fue una bueno'
```

Notá que solo la palabra suelta `"chimba"` se reemplazó por `"bueno"` -- la frase completa `"una chimba"` nunca se evaluó como unidad, y la palabra `"una"` sobrevivió intacta en el resultado. La entrada de diccionario `"una chimba": "bueno"` es, en la práctica, configuración muerta: nunca puede dispararse tal como está escrito el método.

**Test que documenta el hallazgo (`tests/test_synonyms.py`):**
```python
def test_multi_word_colloquialisms_are_unreachable():
    multi_word_keys = [k for k in colombian_colloquialisms if " " in k]
    if not multi_word_keys:
        pytest.skip("No multi-word colloquialisms defined in settings")

    result = replacer.replace_synonyms("el curso fue una chimba")

    assert "una" in result.split()      # la frase NO se reemplazo como unidad
    assert "chimba" not in result.split()  # solo la palabra suelta matcheo
```

**Solución propuesta (pendiente):** antes de tokenizar, reemplazar las frases multi-palabra directamente sobre el string completo (por ejemplo, con `str.replace` o una expresión regular, iterando las claves multi-palabra de mayor a menor longitud para evitar coincidencias parciales), y solo después tokenizar el resto del texto para el reemplazo palabra por palabra.

---

### Hallazgo 3 -- `settings.py` viola la regla de idioma del equipo

**Código relevante (`settings.py`):**
```python
def cargar_stopwords_desde_excel(path_excel, sheet_name: int):
    """Carga y transforma el xlsx en el diccionario de stopwords esperado."""
    try:
        df = pd.read_excel(path_excel, sheet_name= sheet_name)
        # Agrupamos por la columna 'encuesta' y convertimos a lista cada grupo
        # dropna() asegura que no tengamos celdas vacías
        return df.groupby('encuesta')['palabra'].apply(list).to_dict()
    except Exception as e:
        print(f"Error cargando stopwords: {e}")
        return {}
```

**Problema:** el nombre de la función, su docstring, y sus dos comentarios internos están en español, mientras que el resto del código base (nombres de clases, métodos, parámetros, docstrings en `limpieza.py` y `synonyms.py`) está consistentemente en inglés -- siguiendo la regla explícita del equipo: código en inglés, datos de prueba/texto de encuestas en español.

**No es un bug funcional** -- el código corre igual sin importar el idioma de sus identificadores. Se documenta como hallazgo de consistencia porque rompe el patrón que el resto del proyecto sigue, y dificulta la búsqueda de código y la incorporación de nuevos colaboradores que trabajen solo en inglés.

**No se escribió un test para este hallazgo** -- es una observación de revisión de código, no un comportamiento verificable con un assert.

**Solución propuesta (pendiente):**
```python
def load_stopwords_from_excel(excel_path, sheet_name: str):
    """Loads and transforms the xlsx into the expected stopwords dict."""
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        # Group by the 'encuesta' column and convert each group to a list
        return df.groupby('encuesta')['palabra'].apply(list).to_dict()
    except Exception as e:
        print(f"Error loading stopwords: {e}")
        return {}
```
(Nota: los nombres de columnas `'encuesta'` y `'palabra'` del propio archivo Excel no están sujetos a esta regla -- son datos, no código, igual que los textos de encuesta.)

---

### Hallazgo 4 -- `settings.py` hace I/O al importarse y traga errores en silencio

**Código relevante (`settings.py`):**
```python
df_accents = pd.read_excel(ACCENTS_PATH, sheet_name='accents')
accents = dict(zip(df_accents['accented'], df_accents['not_accented']))

def cargar_stopwords_desde_excel(path_excel, sheet_name: int):
    try:
        df = pd.read_excel(path_excel, sheet_name=sheet_name)
        return df.groupby('encuesta')['palabra'].apply(list).to_dict()
    except Exception as e:
        print(f"Error cargando stopwords: {e}")
        return {}

stopwords_dict = cargar_stopwords_desde_excel(STOPWORDS_PATH, sheet_name='particulares')
stopwords_global = cargar_stopwords_desde_excel(STOPWORDS_PATH, sheet_name='globales')
```

**Problema, en dos partes:**

1. **I/O a nivel de módulo:** la lectura de `accents.xlsx` y las dos lecturas de `stopwords.xlsx` ocurren automáticamente en el momento en que **cualquier cosa** hace `import settings` (o `from settings import ...`), no cuando se necesita el dato. Esto significa que cualquier código que solo necesite, por ejemplo, `SYNONYMS_SIMILARITY_THRESHOLD` (un número fijo, sin relación con Excel) igual dispara 3 lecturas de archivo en el momento del import.

2. **Manejo de errores silencioso:** `cargar_stopwords_desde_excel` atrapa **cualquier excepción** (`except Exception`), la imprime por consola con `print`, y devuelve un diccionario vacío `{}`. Si `stopwords.xlsx` no existiera, estuviera corrupto, o tuviera un nombre de columna distinto al esperado, el pipeline completo seguiría corriendo con **cero stopwords específicas de encuesta y cero stopwords globales**, sin lanzar ninguna excepción -- solo un mensaje impreso en la consola, fácil de pasar por alto en un entorno de producción o en un notebook con mucho output.

**Confirmación en ejecución:**
```python
>>> from settings import cargar_stopwords_desde_excel
>>> cargar_stopwords_desde_excel("ruta/que/no/existe.xlsx", sheet_name="particulares")
Error cargando stopwords: [Errno 2] No such file or directory: 'ruta/que/no/existe.xlsx'
{}
```

**Por qué importa:** combinado con el Hallazgo 1 (las stopwords globales ya no se aplican por el bug de `.update()`), un fallo silencioso adicional en la carga haría que el problema sea todavía más difícil de detectar -- el pipeline "funciona" (no lanza errores) incluso si la configuración de stopwords está completamente vacía por un archivo faltante o mal formado.

**Test que documenta el hallazgo (`tests/test_stopwords.py`):**
```python
def test_stopwords_loader_swallows_errors_and_returns_empty_dict(capsys):
    result = cargar_stopwords_desde_excel(
        "ruta/que/no/existe.xlsx", sheet_name="particulares"
    )
    assert result == {}
    assert "Error cargando stopwords" in capsys.readouterr().out
```

**Solución propuesta (pendiente):** decidir con el equipo si un archivo de stopwords faltante o corrupto debería:
(a) detener la ejecución con una excepción clara (`raise ValueError(...)` en vez de `except...: return {}`), o
(b) seguir devolviendo `{}` pero emitir una advertencia más visible (`warnings.warn(...)` en vez de `print`), para que sea imposible de ignorar accidentalmente.
La I/O a nivel de módulo podría además diferirse (cargar solo la primera vez que se accede al valor, no en el import) si el tiempo de arranque del pipeline se vuelve relevante.

---

### Hallazgo 5 -- Firma `sheet_name: int` que recibe strings

**Código relevante (`settings.py`):**
```python
def cargar_stopwords_desde_excel(path_excel, sheet_name: int):
    ...

stopwords_dict = cargar_stopwords_desde_excel(
                    STOPWORDS_PATH, sheet_name='particulares')
```

**Problema:** el type hint del parámetro declara `sheet_name: int`, pero en las dos únicas llamadas reales a la función se le pasan strings (`'particulares'`, `'globales'`). `pandas.read_excel` sí acepta ambos tipos para `sheet_name` (un índice numérico o un nombre de hoja), así que el código funciona correctamente en la práctica -- el problema es exclusivamente que el type hint miente sobre lo que la función realmente acepta y usa.

**Severidad muy baja:** no causa ningún comportamiento incorrecto. Se documenta únicamente para que quien lea la firma de la función no se confunda pensando que solo acepta índices numéricos.

**No se escribió un test para este hallazgo** -- un type hint incorrecto no es, por sí mismo, algo verificable con un assert en tiempo de ejecución (Python no impone los type hints).

**Solución propuesta (pendiente):**
```python
def cargar_stopwords_desde_excel(path_excel, sheet_name: str | int):
```

---

## 4. Estructura de archivos de la suite

```
tests/
├── conftest.py                 (configura sys.path para la estructura de dos niveles src/ + src/datacleaning/)
├── test_clean_text.py          (23 tests -- accents parametrizado desde settings.accents real)
├── test_load_data.py           (14 tests -- constructor con survey_name)
├── test_clean_key_column.py    (6 tests)
├── test_save_cleaned_data.py   (4 tests -- regresion de orden de llamada confirmada resuelta)
├── test_stopwords.py           (9 tests -- incluye los Hallazgos 1, 3 y 4 de esta revision)
├── test_synonyms.py            (16 tests -- NUEVO, modelo Word2Vec mockeado, incluye el Hallazgo 2)
└── test_integration.py         (4 tests -- cadena real Cleaner + SynonymReplacer, sin mocks salvo el modelo)
```

**Resultado de ejecución:**
```
94 passed in 8.53s

Name                           Stmts   Miss  Cover   Missing
------------------------------------------------------------
src/datacleaning/limpieza.py      75      2    97%   92-93
src/datacleaning/synonyms.py      58      0   100%
src/settings.py                   22      0   100%
------------------------------------------------------------
TOTAL                            155      2    99%
```

Las 2 líneas restantes sin cubrir en `limpieza.py` (92-93) corresponden a la rama `except pd.errors.DatabaseError`, la misma de bajo valor documentada en revisiones anteriores -- difícil de provocar con un CSV/Excel real sin recurrir a artificios.

`synonyms.py` alcanza 100% gracias a la exclusión configurada de `_download_model` y `load_model` (ver Sección 2) -- no porque esas funciones estén probadas, sino porque se excluyeron deliberadamente del cálculo, con el motivo documentado en `pyproject.toml`.

---

## 5. Próximos pasos sugeridos

1. **Prioridad alta:** corregir el Hallazgo 1 (`stop_words.update(stopwords_global)`). Es un cambio de una línea, y actualmente neutraliza por completo la funcionalidad de stopwords globales -- las 8 palabras reales configuradas en el Excel del equipo nunca se aplican a ningún texto.
2. Decidir el tratamiento del Hallazgo 2 (coloquialismos multi-palabra): o se ajusta `replace_synonyms` para reconocer frases antes de tokenizar, o se documenta explícitamente que `colombian_colloquialisms` solo admite entradas de una palabra, y se depuran las entradas multi-palabra existentes para no dejar configuración muerta.
3. Revisar con el equipo el criterio de manejo de errores del Hallazgo 4 (¿fallo silencioso o excepción explícita cuando falta el Excel de stopwords?), antes de que un archivo faltante en producción pase desapercibido.
4. Renombrar `cargar_stopwords_desde_excel` y su docstring/comentarios al inglés (Hallazgo 3), y corregir el type hint de `sheet_name` (Hallazgo 5) -- ambos de baja prioridad, agrupables en un solo commit de limpieza.
5. Si se decide diferir la carga de Excel en `settings.py` (en vez de hacerla en el import), coordinar el cambio con quien más dependa del tiempo de arranque del pipeline, ya que altera cuándo ocurren las excepciones de archivo faltante.
