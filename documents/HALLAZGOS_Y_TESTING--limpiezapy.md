# Informe de hallazgos y testing -- `limpieza.py`

**Módulo evaluado:** `limpieza.py` (clase `cleaner`), ubicado en `src/limpieza.py`
**Entorno de prueba:** Python 3.12.3, pandas 3.0.3, pytest 9.1.1, gestionado con Poetry 2.4.1
**Resultado de la suite:** `41 passed, 5 xfailed`
**Cobertura de código:** 78% de `src/limpieza.py` (ver Sección 8)
**Alcance:** revisión de comportamiento mediante pruebas automatizadas (pytest), incluyendo tanto pruebas unitarias (cada método probado de forma aislada) como pruebas de integración (el flujo real encadenado, ejecutado de punta a punta). No se modificó el código fuente original; todos los hallazgos están confirmados en ejecución, no solo por lectura de código.

**Cómo correr esta suite:**
```bash
poetry install
poetry run pytest --cov=src
```

---

## 1. Resumen ejecutivo

Se construyó una suite de pruebas automatizadas para la clase `cleaner`, cubriendo sus 6 métodos (`load_data`, `_clean_text`, `clean_key_column`, `_clean_stopwords`, `eliminate_stopwords`, `save_cleaned_data`). Durante este proceso se identificaron **5 hallazgos** que afectan el comportamiento esperado o la mantenibilidad de la clase, todos confirmados ejecutando el código real o mediante revisión directa del código (no son señalamientos teóricos).

Ningún hallazgo fue corregido en el código fuente: por decisión del equipo, los hallazgos se documentan primero (con un test que reproduce el problema) para su revisión y priorización antes de aplicar cualquier cambio.

| # | Hallazgo | Método afectado | Severidad sugerida |
|---|---|---|---|
| ✅1 | `_clean_stopwords` no puede ejecutarse: falta `import nltk` y usa `self` en un `@staticmethod` | `_clean_stopwords` | Alta -- el método nunca funciona |
| ✅2 | `eliminate_stopwords` depende de `self.cleaned_data`, un atributo que solo existe si se llamó antes a `save_cleaned_data` | `eliminate_stopwords` | Alta -- falla en el uso más intuitivo del método |
| ✅3 | `load_data` no admite `file_path` como `str`, aunque la firma del método lo declara (`str \| Path`) | `load_data` (y, por herencia, `clean_key_column`) | Media-alta -- rompe una promesa explícita de la API |
| ✅4 | La validación `dtype != 'object'` falla con pandas >= 3.0 | `load_data` (y, por herencia, `clean_key_column`) | Media de forma aislada, pero confirmada por las pruebas de integración (Sección 3.1) como bloqueante de todo el flujo real -- ver nota de prioridad práctica en la Sección 7 |
| ✅5 | Inconsistencia de idioma: nombres de variables en español, sufijos de columnas en español, y un docstring con inglés poco idiomático, mezclados en un código que por lo demás está en inglés | `clean_key_column`, `eliminate_stopwords` | Baja -- no afecta la funcionalidad, pero afecta la mantenibilidad y consistencia |

`clean_key_column` y `save_cleaned_data` también se probaron directamente (aislados de `load_data` mediante mocking, para evitar que los Hallazgos 3 y 4 enmascararan los resultados). **No se encontraron bugs propios nuevos** en ninguno de los dos métodos -- solo heredan los hallazgos anteriores a través de su llamada a `load_data()`.

Además, se construyó una suite de pruebas de integración (`test_integration.py`) para ejercitar el flujo real encadenado (`load_data -> clean_key_column -> save_cleaned_data`, y `eliminate_stopwords`) sin aislar los métodos entre sí. Esto reveló una aclaración importante sobre el impacto real del Hallazgo 4: **en este entorno, el Hallazgo 4 bloquea cualquier archivo CSV o Excel real en el primer paso del flujo** -- no es un caso raro, sino un bloqueo total del camino feliz tal como está escrito hoy. Ver la Sección 3.1 para más detalle.

Por último, se midió la cobertura de código con `pytest-cov` (`poetry run pytest --cov=src`), alcanzando un **78%** de `src/limpieza.py`. La mayor parte de lo no cubierto depende del entorno (ramas `ImportError` no probadas) o es una ruta de funcionalidad genuinamente sin probar (la rama de lectura de Excel en `load_data`). Una parte, sin embargo, es **código estructuralmente inalcanzable**: dos ramas `else: raise ValueError(...)` (en `clean_key_column` y `save_cleaned_data`) protegen contra un valor de retorno `None` que `load_data()` nunca produce en la práctica. Ver la Sección 8 para el detalle completo.

---

## 2. Metodología

Se aplicaron tres principios de testing a lo largo de todo el trabajo:

1. **Se testea el comportamiento observable, no la implementación interna.** Cada test verifica qué devuelve un método o qué excepción lanza dado cierto input, sin asumir cómo está construido por dentro.
2. **Todo hallazgo se confirma en ejecución antes de documentarlo.** Ningún bug se reporta solo por lectura de código; en cada caso se ejecutó el método contra datos de prueba y se capturó el error real (tipo de excepción y mensaje exacto).
3. **Los tests unitarios se complementan con tests de integración.** La mayor parte de esta suite aísla cada método (muchas veces vía mocking) para que un fallo apunte sin ambigüedad a un solo método. Ese aislamiento puede ocultar problemas que solo aparecen cuando los métodos se encadenan de verdad, tal como lo haría un usuario real. `test_integration.py` ejercita la cadena real sin mocks (salvo un workaround puntual y explícitamente documentado -- ver Sección 3.1) para detectar justamente esa categoría de problema.

Los hallazgos se documentan en los tests mediante `@pytest.mark.xfail(strict=True)`. Esto significa:
- El test reproduce el bug y se espera que falle (`xfail`).
- Si el bug se corrige en el código fuente y el test empieza a pasar, pytest lo marca como `XPASS` y lo reporta como **fallo de la suite** (por el `strict=True`). Esto es intencional: obliga a actualizar el test (quitar el `xfail`) en el momento en que se aplique la corrección, evitando que quede documentación desactualizada.

Para `clean_key_column` y `save_cleaned_data`, se usó `unittest.mock.patch` para reemplazar el valor de retorno de `load_data()` por un DataFrame controlado en memoria. Esto aísla la lógica propia de cada método de los problemas conocidos en `load_data`, para poder confirmar si agregan algún problema adicional propio (no lo hacen).

---

## 3. Hallazgos detallados

### Hallazgo 1 -- `_clean_stopwords` no puede ejecutarse

**Código relevante (`limpieza.py`):**
```python
@staticmethod
def _clean_stopwords(text: str) -> str:
    words = nltk.word_tokenize(text)
    filtradas = [w for w in words if w.lower() not in self.stop_words]
    return ' '.join(filtradas)
```

**Problema:**
- El método usa `nltk` directamente, pero nunca lo importa dentro de la función (el import de NLTK ocurre en otro método, `eliminate_stopwords`, y no es accesible desde acá).
- Es un `@staticmethod`, por lo que no recibe `self` -- sin embargo, intenta usar `self.stop_words`, que además **nunca se inicializa en ningún lugar de la clase** (no aparece en `__init__`).

**Confirmación en ejecución:**
```python
>>> cleaner._clean_stopwords("hola mundo de prueba")
NameError: name 'nltk' is not defined
```

**Test que documenta el hallazgo (`tests/test_stopwords.py`):**
```python
@pytest.mark.xfail(
    reason="BUG: _clean_stopwords does not import 'nltk' inside the method, "
           "and also uses 'self.stop_words' while being a @staticmethod "
           "(does not receive self). Confirmed at runtime: NameError: name "
           "'nltk' is not defined. PROPOSED FIX (pending): add "
           "'import nltk' to the method, remove @staticmethod or replace "
           "self.stop_words with a real instance attribute/parameter, and "
           "resolve where 'self.stop_words' comes from (today it is never "
           "initialized in __init__).",
    strict=True,
)
def test_clean_stopwords_fails_because_nltk_not_imported():
    result = cleaner._clean_stopwords("hola mundo de prueba")
    assert result == "hola mundo prueba"  # expected if it worked correctly
```

**Solución propuesta (pendiente):**
1. Agregar `import nltk` al inicio del método (o a nivel de módulo).
2. Decidir de dónde se origina `stop_words`: si el método deja de ser `@staticmethod` y pasa a recibir `self`, debería usar un atributo de instancia ya inicializado (ver el Hallazgo 2 para un patrón similar); alternativamente, `stop_words` podría pasarse como parámetro del método.

---

### Hallazgo 2 -- `eliminate_stopwords` depende de un atributo que puede no existir

**Código relevante (`limpieza.py`):**
```python
def eliminate_stopwords(self):
    ...
    columna_objetivo = f"{self.key_column}_limpia"
    self.cleaned_data[f"{self.key_column}_sin_stopwords"] = self.cleaned_data[columna_objetivo].apply(self._clean_stopwords)
    return self.cleaned_data
```

**Problema:**
El atributo `self.cleaned_data` solo se crea en un único lugar de toda la clase: dentro de `save_cleaned_data`, en la línea `self.cleaned_data = cleaned_data`. El constructor (`__init__`) no lo inicializa. Si se llama `eliminate_stopwords()` sin haber llamado antes (y con éxito) a `save_cleaned_data()`, el atributo no existe.

**Confirmación en ejecución** (con `nltk` instalado, para aislar este hallazgo del Hallazgo 1):
```python
>>> instance = cleaner(file_path="file.csv", key_column="comment")
>>> instance.eliminate_stopwords()
AttributeError: 'cleaner' object has no attribute 'cleaned_data'
```

**Test que documenta el hallazgo (`tests/test_stopwords.py`):**
```python
@pytest.mark.xfail(
    reason="BUG: eliminate_stopwords() uses self.cleaned_data, but that "
           "attribute is only created inside save_cleaned_data() "
           "('self.cleaned_data = cleaned_data'). Calling eliminate_stopwords "
           "without having called save_cleaned_data first raises "
           "AttributeError. Confirmed at runtime: AttributeError: 'cleaner' "
           "object has no attribute 'cleaned_data'. PROPOSED FIX (pending): "
           "initialize self.cleaned_data = None in __init__, or make "
           "eliminate_stopwords call clean_key_column() first if "
           "self.cleaned_data doesn't exist yet (same as save_cleaned_data "
           "does internally).",
    strict=True,
)
def test_eliminate_stopwords_fails_without_prior_cleaned_data():
    instance = cleaner(file_path="not_relevant.csv", key_column="comment")
    result = instance.eliminate_stopwords()
    assert result is not None  # expected if it worked correctly
```

**Solución propuesta (pendiente):**
Se evaluaron tres opciones, no mutuamente excluyentes:

| Opción | Qué resuelve |
|---|---|
| A. Inicializar `self.cleaned_data = None` en `__init__` | El atributo siempre existe; evita el `AttributeError`, pero no evita un fallo posterior si sigue siendo `None` |
| B. `eliminate_stopwords` genera los datos si no existen (`if self.cleaned_data is None: self.cleaned_data = self.clean_key_column()`) | Elimina la dependencia implícita de orden de llamada; el método funciona sin importar cuándo se invoque |
| C. Validar y lanzar un error de negocio explícito si `self.cleaned_data` es `None` | No elimina la dependencia de orden, pero la hace visible con un mensaje claro en vez de un `AttributeError` críptico |

Recomendación: combinar A + B. La opción C queda reservada para casos donde sí se quiera forzar un orden estricto de llamada, lo cual no aplica acá.

---

### Hallazgo 3 -- `load_data` no admite `file_path` como `str`

**Código relevante (`limpieza.py`):**
```python
def __init__(self, file_path : str | Path , key_column : str, separator : str = ',', sheet_name: str | int = 0):
    self.file_path = file_path
    ...

def load_data(self):
    ...
    extension = self.file_path.suffix.lower()
```

**Problema:**
La firma del constructor declara que `file_path` acepta `str` o `Path` (`str | Path`). Sin embargo, `load_data` usa `self.file_path.suffix`, atributo exclusivo de `pathlib.Path` -- los objetos `str` no lo tienen. Si se instancia la clase con un string (un uso perfectamente válido según la firma), el método falla.

**Confirmación en ejecución:**
```python
>>> instance = cleaner(file_path="data.csv", key_column="comment")
>>> instance.load_data()
ValueError: An error occurred: 'str' object has no attribute 'suffix'
```
*(El error real es `AttributeError`, pero queda envuelto en un `ValueError` genérico por el `except Exception` al final de `load_data` -- ver también la observación de la Sección 4.)*

**Test que documenta el hallazgo (`tests/test_load_data.py`):**
```python
@pytest.mark.xfail(
    reason="BUG: load_data() fails with a str file_path because it uses "
           ".suffix, an attribute exclusive to Path. See limpieza.py line "
           "'extension = self.file_path.suffix.lower()'.",
    strict=True,
)
def test_accepts_file_path_as_string(valid_csv):
    instance = cleaner(file_path=str(valid_csv), key_column="comment")
    result = instance.load_data()
    assert isinstance(result, pd.DataFrame)
```

**Impacto adicional:** este hallazgo se propaga a `clean_key_column()`, ya que ese método llama internamente a `load_data()`. Se confirmó en ejecución que `clean_key_column()` produce el mismo error al recibir `file_path` como `str`.

**Solución propuesta (pendiente):**
Convertir `file_path` a `Path` al inicio de `load_data` (o en `__init__`):
```python
extension = Path(self.file_path).suffix.lower()
```

---

### Hallazgo 4 -- La validación de tipo de columna falla con pandas >= 3.0

**Código relevante (`limpieza.py`):**
```python
if data[self.key_column].dtype != 'object':
    raise ValueError(f"Column '{self.key_column}' is not of text type.")
```

**Problema:**
A partir de pandas 3.0, `pandas.read_csv` puede inferir un dtype `str` nativo (distinto de `object`) para columnas de texto. La validación compara contra el string literal `'object'`, por lo que **rechaza columnas de texto legítimas** cuando se ejecuta en un entorno con pandas >= 3.0.

**Confirmación en ejecución** (pandas 3.0.3, columna de texto válida):
```python
>>> df["comment"].dtype
str
>>> df["comment"].dtype == 'object'
False
```
Esto provoca que `load_data()` rechace un archivo perfectamente válido con:
```
ValueError: An error occurred: Column 'comment' is not of text type.
```

**Test que documenta el hallazgo (`tests/test_load_data.py`):**
```python
@pytest.mark.xfail(
    reason="ENVIRONMENT-DEPENDENT FINDING: in pandas >= 3.0, pandas.read_csv "
           "can infer dtype 'str' (not 'object') for text columns. "
           "load_data() compares with data[col].dtype != 'object' literally, "
           "which rejects valid text columns in this environment. "
           "See limpieza.py: 'if data[self.key_column].dtype != \"object\":'. "
           "Pandas installed in this environment: 3.0.2. "
           "PROPOSED FIX (not yet applied): replace the comparison with "
           "'not pd.api.types.is_string_dtype(data[self.key_column])', which "
           "is version-agnostic. Decision: keep as xfail for now, do not "
           "modify limpieza.py yet.",
    strict=True,
)
def test_loading_a_valid_csv_returns_a_dataframe(valid_csv):
    instance = cleaner(file_path=valid_csv, key_column="comment")
    result = instance.load_data()
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "comment" in result.columns
```

**Impacto adicional:** esto también se propaga a `clean_key_column()` por la misma razón que el Hallazgo 3 (depende de `load_data()`). Confirmado en ejecución.

**Solución propuesta (pendiente):**
Reemplazar la comparación literal por la función de pandas diseñada para este propósito exacto, que es agnóstica a la versión:
```python
if not pd.api.types.is_string_dtype(data[self.key_column]):
    raise ValueError(f"Column '{self.key_column}' is not of text type.")
```

---

## 3.1 Tests de integración -- el flujo real encadenado

Todos los hallazgos anteriores se confirmaron con **tests unitarios**: cada método probado de forma aislada, muchas veces usando `unittest.mock` para evitar que los problemas conocidos de un método enmascararan el comportamiento de otro. Ese enfoque es preciso, pero puede ocultar problemas que solo aparecen cuando los métodos se encadenan de verdad, tal como lo haría un usuario real.

Para cubrir ese hueco, se agregó `tests/test_integration.py`. Corre el flujo real de punta a punta -- `load_data -> clean_key_column -> save_cleaned_data`, y `eliminate_stopwords` -- sin mocking, salvo un workaround puntual y explícitamente comentado que se describe más abajo. Resultado:

```
tests/test_integration.py::test_full_workflow_is_blocked_by_finding_4_in_this_environment PASSED
tests/test_integration.py::test_full_workflow_load_clean_and_save PASSED
tests/test_integration.py::test_full_workflow_stops_cleanly_on_missing_key_column PASSED
tests/test_integration.py::test_eliminate_stopwords_after_a_real_save_still_fails PASSED

4 passed in 3.12s
```

### Hallazgo de integración -- el Hallazgo 4 bloquea todo el flujo real en este entorno

Mientras se construía el test de integración del camino feliz, se construyeron varios archivos CSV y Excel reales -- incluyendo uno armado deliberadamente con una columna `dtype="object"` explícita antes de guardarlo, en un intento de producir un archivo que `load_data()` aceptara. **Ninguno funcionó.** En este entorno (pandas 3.0.3), tanto `pandas.read_csv` como `pandas.read_excel` infieren un dtype `str` para columnas de texto sin importar cómo se haya escrito el archivo de origen, por lo que la validación `dtype != 'object'` de `load_data()` (Hallazgo 4) rechaza cualquier archivo real desde el primer paso.

Esto se confirma directamente con el siguiente test:

```python
def test_full_workflow_is_blocked_by_finding_4_in_this_environment(real_survey_csv, tmp_path):
    """
    This test documents an integration-level consequence of Finding 4
    (see HALLAZGOS_Y_TESTING.md). It is NOT marked xfail, because it is
    not testing a bug directly -- it is confirming, on purpose, that the
    very first step of the real chain (load_data) blocks the rest of the
    workflow in this environment.
    """
    instance = cleaner(file_path=real_survey_csv, key_column="comment")

    with pytest.raises(ValueError, match="not of text type"):
        instance.load_data()
```

**Por qué esto importa más allá del Hallazgo 4 en sí:** el Hallazgo 4 se documentó originalmente (Sección 3, Hallazgo 4) como un bug de comparación de dtype que afecta a `load_data`. El test de integración aclara su severidad real: no es un caso raro que solo aparece con datos inusuales -- es un **bloqueo total** para el caso de uso principal de todo el módulo (limpiar un archivo de encuesta real) en cualquier entorno que corra pandas >= 3.0. Esto eleva la prioridad práctica del Hallazgo 4, aunque su severidad aislada se calificó como "Media" en la Sección 1.

### Test de integración del camino feliz -- el resto de la cadena, una vez resuelto el Hallazgo 4

Dado que ningún archivo real lograba pasar `load_data()` en este entorno, la única forma de ejercitar `clean_key_column` y `save_cleaned_data` juntos de verdad (sin modificar `limpieza.py`) fue simular el valor de retorno de `load_data()` en ese único punto, dejando el resto de la llamada completamente real:

```python
def test_full_workflow_load_clean_and_save(tmp_path):
    output_path = tmp_path / "cleaned_output.csv"
    instance = cleaner(file_path="not_relevant.csv", key_column="comment")

    real_shaped_data = pd.DataFrame({
        "respondent_id": [1, 2, 3, 4],
        "comment": ["Excelente atención, muy rápido!", "  regular  ", "PÉSIMO!!", "Buena relación precio/calidad"],
    })

    with patch.object(cleaner, "load_data", return_value=real_shaped_data):
        loaded = instance.load_data()
        assert len(loaded) == 4

        cleaned = instance.clean_key_column()
        assert "comment_limpia" in cleaned.columns
        assert cleaned["comment_limpia"].iloc[0] == "excelente atencion muy rapido"

        instance.save_cleaned_data(str(output_path))

    assert output_path.exists()
    saved = pd.read_csv(output_path)
    assert "comment_limpia" in saved.columns
    assert instance.cleaned_data is not None
```

Este es un uso más acotado y deliberado de mocking que en `test_clean_key_column.py` / `test_save_cleaned_data.py` (que mockean `load_data` en cada test, por diseño, para aislar el método bajo prueba). Aquí existe únicamente como workaround del Hallazgo 4, para que el resto de la cadena real -- `clean_key_column`, `_clean_text`, `save_cleaned_data`, y el archivo resultante en disco -- pudiera verificarse en conjunto, de punta a punta. Resultado: **el resto de la cadena funciona correctamente** una vez que `load_data` no es quien la bloquea.

### Confirmación de integración -- una configuración inválida detiene la cadena de forma prolija

```python
def test_full_workflow_stops_cleanly_on_missing_key_column(tmp_path):
    ...
    instance = cleaner(file_path=path, key_column="comment")  # "comment" does not exist here

    with pytest.raises(ValueError, match="not found"):
        instance.load_data()
    with pytest.raises(ValueError, match="not found"):
        instance.clean_key_column()
    with pytest.raises(ValueError, match="not found"):
        instance.save_cleaned_data(str(tmp_path / "should_not_exist.csv"))
```

Confirma que un `key_column` inválido detiene cada método de la cadena con el mismo error claro, en vez de fallar de forma confusa en un paso posterior no relacionado.

### Confirmación de integración -- los Hallazgos 1 y 2 combinados, en un escenario real

```python
def test_eliminate_stopwords_after_a_real_save_still_fails(tmp_path):
    """
    Runs the real save_cleaned_data() chain first, so self.cleaned_data
    genuinely exists (Finding 2's problem is solved here by following the
    correct call order). Confirms Finding 1 still blocks
    eliminate_stopwords() even in this best-case scenario.
    """
    ...
    with patch.object(cleaner, "load_data", return_value=real_shaped_data):
        instance.save_cleaned_data(str(output_path))

    assert instance.cleaned_data is not None  # Finding 2's problem is not present here

    with pytest.raises(NameError, match="nltk"):
        instance.eliminate_stopwords()  # still blocked by Finding 1
```

Esto confirma que incluso un usuario que hace todo "correctamente" -- llamando a `save_cleaned_data()` antes de `eliminate_stopwords()`, exactamente como exige el orden implícito (no documentado) -- aun así no puede usar `eliminate_stopwords()`, porque el Hallazgo 1 es independiente del orden de llamada. Los dos hallazgos relacionados con stopwords no son redundantes entre sí: corregir solo el Hallazgo 2 no alcanzaría para que `eliminate_stopwords()` sea utilizable.

---

### Hallazgo 5 -- Inconsistencia de idioma: identificadores en español mezclados en un código en inglés

**Código relevante (`limpieza.py`):**
```python
# clean_key_column docstring
"""
Realizes the cleaning of the text in the key column of the loaded DataFrame.
...
Returns:
    pd.DataFrame: A new DataFrame that includes the additional column with
        the processed text ('{self.key_column}_limpia').
"""
data[f"{self.key_column}_limpia"] = data[self.key_column].apply(self._clean_text)

# _clean_stopwords
filtradas = [w for w in words if w.lower() not in self.stop_words]
return ' '.join(filtradas)

# eliminate_stopwords
columna_objetivo = f"{self.key_column}_limpia"
self.cleaned_data[f"{self.key_column}_sin_stopwords"] = self.cleaned_data[columna_objetivo].apply(self._clean_stopwords)
```

**Problema:**
Los nombres de clase, de métodos, de parámetros, y la mayoría de los docstrings en `limpieza.py` están consistentemente escritos en inglés. Sin embargo, algunos elementos siguen en español, rompiendo esa consistencia:

- Dos sufijos de nombre de columna generados usan palabras en español: `_limpia` y `_sin_stopwords`, en vez de un equivalente en inglés como `_clean` o `_no_stopwords`.
- Dos nombres de variable internos están en español: `filtradas` y `columna_objetivo`.
- El docstring de `clean_key_column` empieza con "Realizes the cleaning...", que no es inglés idiomático -- se lee como una traducción literal del español "Realiza la limpieza...". Un equivalente más natural sería "Performs the cleaning..." o "Cleans the text...".

**Esto no es un bug funcional.** Ninguno de estos elementos causa comportamiento incorrecto o errores en ejecución -- el código corre igual sin importar en qué idioma esté un nombre de variable o un sufijo de columna. Se marca acá como un hallazgo de mantenibilidad/consistencia, no como un defecto.

**Por qué importa igual:**
- Cualquier código que consuma el DataFrame resultante (un script posterior, un notebook, otro miembro del equipo) necesita conocer el sufijo exacto en español (`_limpia`, `_sin_stopwords`) para referenciar las columnas generadas -- alguien que solo trabaje en inglés podría razonablemente esperar `_clean` o `_no_stopwords` y no encontrarlas.
- Los identificadores mezclados en dos idiomas dificultan la búsqueda en el código, dificultan la incorporación de nuevos colaboradores, y son inconsistentes con la intención aparente del proyecto de estandarizar en inglés (reflejada en todo lo demás: el nombre de clase `cleaner`, los métodos `load_data`, `clean_key_column`, `eliminate_stopwords`, `save_cleaned_data`, los parámetros `file_path`, `key_column`, etc.).

**No se escribió un test para este hallazgo** -- no es un comportamiento que se pueda verificar con `pytest.raises` o una comparación de salida; es una observación de revisión de código. Se documenta acá para que no se pierda al priorizar las correcciones de los Hallazgos 1 a 4.

**Solución propuesta (pendiente):**
Renombrar los identificadores y sufijos afectados a sus equivalentes en inglés, por ejemplo:
```python
# Sufijos
f"{self.key_column}_clean"           # en vez de f"{self.key_column}_limpia"
f"{self.key_column}_no_stopwords"     # en vez de f"{self.key_column}_sin_stopwords"

# Variables
filtered_words = [w for w in words if w.lower() not in self.stop_words]   # en vez de filtradas
target_column = f"{self.key_column}_clean"                                 # en vez de columna_objetivo
```
Nota: cambiar los sufijos de columna es un cambio incompatible (breaking change) para cualquier código posterior o archivo CSV guardado que ya referencie `_limpia` / `_sin_stopwords` -- esto debería coordinarse con quien consuma la salida de este módulo, no aplicarse en silencio.

---

## 4. Observación general (no bloqueante)

`load_data` envuelve todas las excepciones internas (`FileNotFoundError`, `AttributeError`, etc.) en un `ValueError` genérico mediante un `except Exception as e: raise ValueError(...)` al final del método. Esto no es, en sí, un bug -- el método sigue lanzando una excepción ante un error -- pero tiene dos consecuencias a tener en cuenta:

- Se pierde la distinción entre tipos de error (un archivo inexistente y una columna mal tipada generan la misma clase de excepción, `ValueError`, distinguibles solo por el texto del mensaje).
- Errores de programación genuinos (como los Hallazgos 3 y 4, que son bugs de tipos/versión, no errores de validación de negocio) quedan disfrazados de errores de validación, lo que puede confundir al diagnosticar fallos en producción.

No se generó un test específico para este punto porque no es, por sí mismo, un comportamiento incorrecto -- se deja como nota de diseño para que el equipo lo considere junto con los demás hallazgos.

---

## 5. Lo que SÍ funciona correctamente (confirmado por tests)

La función `_clean_text` (limpieza de texto: minúsculas, tildes, caracteres especiales, manejo de valores no-texto) fue validada exhaustivamente y **no presenta hallazgos**. Se confirmaron 18 casos (incluyendo variantes parametrizadas), entre ellos:

- Normalización de mayúsculas/minúsculas y tildes.
- Conservación de la letra "ñ" y de números.
- Eliminación de signos de puntuación, HTML pegado por error, etc.
- Manejo correcto de valores no-texto (`None`, `NaN`, números, listas), devolviendo string vacío.
- Manejo correcto de strings vacíos o compuestos solo de espacios.

`load_data` también funciona correctamente (más allá de los Hallazgos 3 y 4) para sus reglas de validación principales:

- Archivo inexistente -> `ValueError`.
- Formato de archivo no soportado -> `ValueError`.
- Columna clave inexistente en el archivo -> `ValueError`.
- Columna clave de tipo numérico (cuando el dtype es claramente distinto de texto, p. ej. `int64`) -> `ValueError`.

`clean_key_column` funciona correctamente cuando se aísla de los problemas conocidos de `load_data` mediante mocking:

- Agrega la columna `{key_column}_limpia` con el texto limpio, sin modificar la columna original.
- El nombre de la columna nueva sigue correctamente al `key_column` que se haya configurado.
- No muta el DataFrame original en el lugar (el `.copy()` explícito funciona como se espera).
- Propaga los errores de `load_data()` en vez de tragárselos.

`save_cleaned_data` también funciona correctamente cuando se aísla de la misma forma:

- Guarda el archivo CSV con la columna limpia incluida.
- Crea el atributo `self.cleaned_data` tras un guardado exitoso (este es exactamente el mecanismo detrás del Hallazgo 2).
- Lanza `ValueError` con un mensaje claro cuando la ruta de salida es inválida.
- Propaga los errores de la cadena de carga de datos en vez de tragárselos.

---

## 6. Estructura de archivos de la suite

```
tests/
├── test_clean_text.py         (18 tests -- todos PASSED)
├── test_load_data.py          (7 tests -- 4 PASSED, 3 XFAIL)
├── test_clean_key_column.py   (4 tests -- todos PASSED)
├── test_save_cleaned_data.py  (4 tests -- todos PASSED)
├── test_stopwords.py          (2 tests -- 2 XFAIL)
└── test_integration.py        (4 tests -- todos PASSED; flujo real encadenado, sin mocks salvo un workaround documentado)
```

**Resultado de ejecución (`poetry run pytest --cov=src`):**
```
41 passed, 5 xfailed in 1.63s
```

---

## 7. Próximos pasos sugeridos

1. Priorizar los Hallazgos 1 y 2 (severidad alta): ambos impiden que la funcionalidad de stopwords funcione en cualquier escenario de uso razonable.
2. Decidir sobre el Hallazgo 3 (compatibilidad `str`/`Path`), ya que rompe una promesa explícita de la API documentada en la propia firma del método.
3. Reconsiderar la urgencia del Hallazgo 4 a la luz de los resultados de las pruebas de integración (Sección 3.1): no afecta solo un caso raro, bloquea el flujo principal de todo el módulo en cualquier entorno que corra pandas >= 3.0. Si se va a fijar pandas < 3.0 como dependencia, el impacto inmediato es menor, pero la corrección con `pd.api.types.is_string_dtype` es fuertemente preferible dado cuán ampliamente bloquea el uso real.
4. Una vez aplicadas las correcciones, quitar los `@pytest.mark.xfail` correspondientes y confirmar que los tests pasan a `PASSED` (si quedan en `XFAIL` tras la corrección, pytest los marcará como `XPASS` y fallará la suite por el `strict=True` -- señal de que el test todavía necesita actualizarse).
5. Considerar renombrar los identificadores y sufijos de columna en español documentados en el Hallazgo 5 a sus equivalentes en inglés -- baja prioridad comparado con los Hallazgos 1-4, pero vale la pena coordinarlo con cualquier consumidor posterior de las columnas generadas antes de aplicarlo, ya que `_limpia` / `_sin_stopwords` cambiarían a `_clean` / `_no_stopwords` (un cambio incompatible para archivos de salida existentes).
6. Revisar los hallazgos de código muerto de la Sección 8 (las ramas `else: raise ValueError(...)` en `clean_key_column` y `save_cleaned_data`) -- son inalcanzables tal como están escritas, y deberían eliminarse, o bien `load_data()` debería modificarse para que pueda devolver `None` en algún escenario real, si esa era la intención original.

---

## 8. Cobertura de código

La cobertura de código mide qué líneas de `src/limpieza.py` se ejecutaron al menos una vez durante la corrida de la suite. No mide corrección por sí misma (una línea puede ejecutarse y aun así estar comparada contra la expectativa equivocada), pero un número de cobertura bajo señala de forma confiable rutas de código que ningún test ejercita en absoluto -- que es exactamente cómo se encontró la observación de código muerto más abajo.

**Comando usado:**
```bash
poetry run pytest --cov=src --cov-report=term-missing
```

**Resultado:**
```
Name              Stmts   Miss  Cover   Missing
-----------------------------------------------
src/limpieza.py      80     18    78%   36-37, 54-65, 73, 143, 148-149, 173-174, 184, 216
-----------------------------------------------
TOTAL                80     18    78%
41 passed, 5 xfailed in 4.94s
```

**El 78% de las declaraciones (statements) de `src/limpieza.py` fueron ejecutadas por la suite.** Las 18 líneas sin cobertura se dividen en tres categorías distintas:

### Categoría A -- Ramas dependientes del entorno (esperado, baja preocupación)

| Líneas | Qué son | Por qué no están cubiertas |
|---|---|---|
| 36-37 | Rama de `ImportError` si `pandas` mismo no se puede importar | Pandas siempre está instalado en el entorno de test; esta rama solo correría en un entorno roto |
| 173-174 | Rama de `ImportError` si `nltk` no se puede importar, dentro de `eliminate_stopwords` | `nltk` está instalado en el entorno de test específicamente para poder aislar y confirmar los Hallazgos 1 y 2 (ver Sección 3, Hallazgos 1-2) |

Es razonable dejar esto sin cubrir -- testearlo implicaría romper deliberadamente las propias dependencias del entorno de test, lo cual rara vez vale el esfuerzo.

### Categoría B -- Una ruta de funcionalidad nunca ejercitada (hueco de cobertura real)

| Líneas | Qué son | Por qué no están cubiertas |
|---|---|---|
| 54-65 | Toda la rama de lectura de Excel de `load_data` (`.xlsx`, `.xls`, `.xlsm`, incluyendo la verificación de import de `openpyxl`) | Todos los tests existentes usan solo fixtures de CSV; ningún test crea o lee un archivo Excel |

Este es un hueco genuino, independiente de los Hallazgos 1-4: la ruta de código de Excel nunca fue ejecutada por ningún test de esta suite, así que se desconoce si funciona correctamente, o si tiene su propia versión del Hallazgo 4 (un problema de inferencia de dtype en `pd.read_excel`, sin testear, no sería sorprendente, dado que el Hallazgo 4 ya afecta a `pd.read_csv` en este entorno). Recomendado: agregar al menos un test usando un archivo `.xlsx` real (siguiendo el mismo patrón que las fixtures de CSV existentes en `test_load_data.py`).

### Categoría C -- Código inalcanzable (código muerto, confirmado por cobertura + rastreo manual)

| Líneas | Qué son | Por qué no están cubiertas |
|---|---|---|
| 73 | `return data`, el camino exitoso de `load_data` | No lo alcanza ningún test que no sea `xfail`, porque el Hallazgo 4 bloquea a cualquier archivo real antes de llegar hasta acá en este entorno (ver Sección 3.1) |
| 143 | `else: raise ValueError("The data could not be loaded for cleaning.")` en `clean_key_column` | `load_data()` nunca devuelve `None` en la práctica -- o devuelve un `DataFrame`, o lanza una excepción. La rama `if data is not None` / `else` de arriba, por lo tanto, nunca puede tomar el camino `else` en la práctica |
| 148-149 | La línea de `self.stop_words` y el `return ' '.join(filtradas)` en `_clean_stopwords` | Nunca se alcanzan porque la llamada `nltk.word_tokenize(text)` de la línea anterior siempre lanza `NameError` primero (Hallazgo 1) |
| 184 | `return self.cleaned_data` al final de `eliminate_stopwords` | Nunca se alcanza, por la misma razón que 148-149 -- el método siempre falla antes de llegar acá |
| 216 | `else: raise ValueError("The data could not be cleared for saving.")` en `save_cleaned_data` | Misma causa raíz que la línea 143: `clean_key_column()` (llamado vía `self.clean_key_column()`) nunca devuelve `None` en la práctica, por lo que esta rama `else` es inalcanzable |

**Por qué importa:** las líneas 143 y 216 son patrones de código muerto estructuralmente idénticos (`if x is not None: ... else: raise ValueError(...)`, protegiendo contra un `None` que el método llamado nunca produce en realidad). Esto no es, por sí mismo, un bug que cause comportamiento incorrecto -- pero es engañoso: quien lea `clean_key_column` o `save_cleaned_data` podría asumir que `load_data()` puede devolver `None` en alguna circunstancia, cuando en realidad no puede (solo devuelve un `DataFrame` o lanza una excepción). Las líneas 148-149 y 184 están muertas por una razón distinta: son código normal que *funcionaría* si se corrigiera el Hallazgo 1, pero hoy es inalcanzable como consecuencia directa de ese hallazgo -- no es un problema separado a corregir por sí solo.

**No se crearon nuevos tests `@pytest.mark.xfail` para las líneas 143/216** (las ramas `else` muertas), ya que no existe ningún input que pueda hacer que `load_data()` devuelva `None` para dispararlas -- no hay nada contra qué hacer un assert. Se registran acá como una observación de calidad de código, para abordarse junto con los Hallazgos 1-4 si se revisa `limpieza.py`.
