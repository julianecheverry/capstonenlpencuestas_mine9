# Informe de hallazgos y testing -- `limpieza.py`

**Módulo evaluado:** `limpieza.py` (clase `Cleaner`) y `settings.py`, ubicados en `src/`
**Entorno de prueba:** Python 3.12.3, pandas 3.0.2, nltk 3.9.4, openpyxl 3.1.5, pytest 9.1.1, gestionado con Poetry
**Resultado de la suite:** `78 passed` -- **0 xfailed**
**Cobertura de código:** 95% combinado (`src/limpieza.py`: 94%, `src/settings.py`: 100%)
**Alcance:** esta es la **segunda revisión** de este módulo. La primera revisión (ver historial de commits y la versión anterior de este mismo archivo) documentó 5 hallazgos sobre una versión previa de `limpieza.py`. Un compañero del equipo actualizó el código fuente -- renombrando la clase de `cleaner` a `Cleaner`, agregando un archivo `settings.py`, y corrigiendo varios de los problemas identificados -- antes de que ese cambio fuera aprobado en su Pull Request. Esta revisión confirma, en ejecución real, qué se resolvió, qué persiste, y qué problemas nuevos aparecieron.

**Cómo correr esta suite:**
```bash
poetry install
poetry run python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
poetry run pytest --cov=src
```

---

## 1. Resumen ejecutivo

De los 5 hallazgos documentados en la revisión anterior, **4 quedaron resueltos** y **1 persiste** (en una forma distinta). Además, se confirmaron en ejecución **6 hallazgos nuevos**, incluyendo una regresión de severidad alta.

### Hallazgos de la revisión anterior

| # | Hallazgo (versión anterior) | Estado en esta revisión |
|---|---|---|
| 1 | `_clean_stopwords` no podía ejecutarse (`self`/`nltk` mal usados en un `@staticmethod`) | ✅ **Resuelto** -- ahora es un método de instancia normal, con `nltk` importado a nivel de módulo |
| 2 | `eliminate_stopwords` dependía de `self.cleaned_data`, nunca inicializado | ⚠️ **Persiste**, en una forma nueva (ver Hallazgo 2 actualizado, Sección 3) |
| 3 | `load_data` no admitía `file_path` como `str` | ✅ **Resuelto** -- `__init__` ahora convierte con `Path(file_path)` |
| 4 | La validación de dtype fallaba con pandas >= 3.0 | ✅ **Resuelto** -- ahora usa `pandas.api.types.is_string_dtype()` |
| 5 | Inconsistencia de idioma (identificadores en español) | ✅ **Resuelto** -- todo el código revisado está en inglés |

### Hallazgos nuevos, confirmados en esta revisión

| # | Hallazgo nuevo | Severidad sugerida |
|---|---|---|
| A | `save_cleaned_data` ya no llama a `clean_key_column()` internamente -- ahora también depende del orden de llamada (regresión) | **Alta** -- rompe un uso que antes funcionaba de forma autónoma |
| B | El recurso `punkt_tab` de NLTK es necesario además de `punkt`, pero `__init__` solo descarga `punkt` | Media -- bloquea `_clean_stopwords` en entornos nuevos hasta que se descargue manualmente |
| C | El docstring de `clean_key_column` dice `Returns: None`, pero el método retorna un `DataFrame` | Baja -- no afecta funcionalidad, sí afecta confiabilidad de la documentación |
| D | `settings.py` duplica exactamente el diccionario `accents` que ya existe (hardcodeado) dentro de `_clean_text` en `limpieza.py`; `limpieza.py` no importa `settings.py` | Media -- riesgo de que ambos diccionarios se desincronicen en el futuro |
| E | Los imports de `openpyxl` y `nltk` pasaron a ser a nivel de módulo (antes eran diferidos, dentro de los métodos, con manejo de `ImportError`) | Media -- ahora el módulo entero falla al importarse si falta cualquiera de las dos dependencias, incluso para quien solo necesita `_clean_text` |
| F | El `except Exception` genérico que envolvía todo error en `ValueError` fue eliminado; ahora `FileNotFoundError` se propaga directamente | Informativo -- es una mejora de diseño, pero rompe la compatibilidad con tests que esperaban `ValueError` |

El hallazgo de código muerto de la revisión anterior (las ramas `else: raise ValueError(...)` en `clean_key_column` y `save_cleaned_data`) **persiste sin cambios** -- ver Sección 5.

El hallazgo de integración más importante de la revisión anterior -- que el bug de dtype bloqueaba el 100% de los flujos reales -- **se revirtió por completo**: el flujo real, con un archivo real, ahora funciona de punta a punta sin ningún workaround. Ver Sección 4.

---

## 2. Metodología

Se mantienen los mismos tres principios de la revisión anterior (comportamiento sobre implementación, confirmación en ejecución antes de documentar, tests unitarios complementados con tests de integración). A esto se suma un cuarto principio, específico de esta segunda revisión:

4. **Cada hallazgo de la revisión anterior se reconfirma activamente, no se asume.** No se da por sentado que un hallazgo viejo sigue vigente ni que está resuelto solo porque el código "se ve" distinto -- cada uno de los 5 hallazgos anteriores se volvió a ejecutar contra el código nuevo, igual que la primera vez, antes de marcarlo como resuelto o persistente.

Como resultado de esto, **ningún test de esta revisión usa `@pytest.mark.xfail`**: los hallazgos que persisten (como el nuevo Hallazgo 2) se confirman con asserts directos sobre el comportamiento real, no como "fallos esperados", porque no se está a la espera de un fix específico ya identificado de antemano -- son hallazgos nuevos para que el equipo decida qué hacer con ellos.

---

## 3. Hallazgos detallados

### Hallazgo 1 (anterior) -- RESUELTO: `_clean_stopwords` ya funciona correctamente

**Código anterior:**
```python
@staticmethod
def _clean_stopwords(text: str) -> str:
    words = nltk.word_tokenize(text)
    filtradas = [w for w in words if w.lower() not in self.stop_words]
    return ' '.join(filtradas)
```

**Código actual:**
```python
def _clean_stopwords(self, text: str) -> str:
    words = nltk.word_tokenize(text)
    filtered_words = [w for w in words if w.lower() not in self.stop_words]
    return ' '.join(filtered_words)
```

El método ya no es `@staticmethod` -- recibe `self` correctamente, y `self.stop_words` se inicializa en `__init__`. `nltk` se importa a nivel de módulo. Confirmado en ejecución:

```python
>>> instance = Cleaner(file_path="survey.csv", key_column="comment")
>>> instance._clean_stopwords("el gato come pescado")
'gato come pescado'
```

**Nota de entorno relacionada:** ver Hallazgo B más abajo -- esta confirmación requiere que el recurso `punkt_tab` de NLTK esté descargado, no solo `punkt`.

---

### Hallazgo 2 (anterior) -- PERSISTE, en una forma nueva: `eliminate_stopwords` sigue dependiendo de un atributo no garantizado

**Código actual relevante:**
```python
def clean_key_column(self) -> pd.DataFrame:
    ...
    if data is not None:
        data[f"{self.key_column}_clean"] = (
            data[self.key_column].apply(self._clean_text)
        )
        self.cleaned_data = data        # <- self.cleaned_data nace AQUI ahora
        return self.cleaned_data
    ...

def eliminate_stopwords(self) -> pd.DataFrame:
    # if not hasattr(self, 'cleaned_data') or self.cleaned_data is None:
    #      self.cleaned_data = self.clean_key_column()

    goal_column = f"{self.key_column}_clean"
    self.cleaned_data[f"{self.key_column}_no_stopwords"] = (
        self.cleaned_data[goal_column].apply(
            lambda x: self._clean_stopwords(x))
    )
    return self.cleaned_data
```

**Qué cambió y qué no cambió:** en la versión anterior, `self.cleaned_data` nacía únicamente dentro de `save_cleaned_data()`. En esta versión, nace dentro de `clean_key_column()` en su lugar. **El problema de fondo es el mismo**: `__init__` sigue sin inicializar `self.cleaned_data`, así que `eliminate_stopwords()` sigue fallando si se llama sin haber ejecutado antes el método correcto (ahora `clean_key_column()`, antes `save_cleaned_data()`).

**Dato relevante para el equipo:** hay código comentado dentro de `eliminate_stopwords()` que implementa casi exactamente la Opción B que se recomendó en el informe anterior (autogenerar los datos si no existen). Parece un intento de corrección que quedó sin terminar de aplicar.

**Confirmación en ejecución:**
```python
>>> instance = Cleaner(file_path="survey.csv", key_column="comment")
>>> instance.eliminate_stopwords()
AttributeError: 'Cleaner' object has no attribute 'cleaned_data'
```

**Solución propuesta (pendiente):** descomentar y completar el bloque ya escrito:
```python
if not hasattr(self, 'cleaned_data') or self.cleaned_data is None:
    self.cleaned_data = self.clean_key_column()
```
Esto resolvería tanto este hallazgo como el Hallazgo A (la misma regresión en `save_cleaned_data`), si se aplica un patrón equivalente en ambos métodos. También se recomienda inicializar `self.cleaned_data = None` en `__init__`, para que el atributo exista siempre de forma predecible.

---

### Hallazgo 3 (anterior) -- RESUELTO: `load_data` ya admite `file_path` como `str`

**Código actual:**
```python
def __init__(self, file_path: str | Path, ...) -> None:
    self.file_path = Path(file_path)
```

La conversión explícita a `Path` en el constructor resuelve el problema de raíz. Confirmado en ejecución:
```python
>>> instance = Cleaner(file_path="survey.csv", key_column="comment")  # string, no Path
>>> instance.load_data()
# Funciona correctamente, sin error
```

---

### Hallazgo 4 (anterior) -- RESUELTO: la validación de dtype ya funciona con pandas >= 3.0

**Código actual:**
```python
from pandas.api.types import is_string_dtype
...
if not is_string_dtype(data[self.key_column]):
    raise ValueError(f"Column '{self.key_column}' is not of text type.")
```

Esta es exactamente la solución propuesta en el informe anterior. Confirmado en ejecución, con una columna que pandas 3.0.2 infiere como `dtype: str`:
```python
>>> df["comment"].dtype
str
>>> is_string_dtype(df["comment"])
True
```

**Impacto:** este era el hallazgo que bloqueaba el 100% de los flujos reales en la revisión anterior. Su resolución es la causa directa de que el flujo de integración completo (Sección 4) ahora funcione sin workarounds.

---

### Hallazgo 5 (anterior) -- RESUELTO: ya no hay identificadores en español

Se revisó el código completo (`limpieza.py`) y no se encontraron variables, sufijos de columna, ni docstrings en español. Los sufijos ahora son `_clean` y `_no_stopwords` (antes `_limpia` y `_sin_stopwords`), y las variables internas usan nombres en inglés (`filtered_words`, `goal_column`, antes `filtradas`, `columna_objetivo`).

---

### Hallazgo A (nuevo) -- REGRESIÓN: `save_cleaned_data` ahora también depende del orden de llamada

**Código anterior** (la línea relevante, ya eliminada):
```python
def save_cleaned_data(self, out_path: str) -> None:
    cleaned_data = self.clean_key_column()   # llamaba a clean_key_column internamente
    if cleaned_data is not None:
        ...
```

**Código actual:**
```python
def save_cleaned_data(self, out_path: str) -> None:
    # cleaned_data = self.clean_key_column()
    if self.cleaned_data is not None:
        ...
```

**Problema:** en la versión anterior, `save_cleaned_data()` funcionaba correctamente incluso como primer método llamado sobre una instancia nueva, porque generaba los datos limpios por sí misma. En esta versión, esa llamada interna está comentada, por lo que `save_cleaned_data()` ahora **también** requiere que `clean_key_column()` se haya ejecutado antes -- exactamente la misma categoría de problema que el Hallazgo 2, pero en un método que antes no la tenía.

**Confirmación en ejecución:**
```python
>>> instance = Cleaner(file_path="survey.csv", key_column="comment")
>>> instance.save_cleaned_data("output.csv")
AttributeError: 'Cleaner' object has no attribute 'cleaned_data'
```

**Por qué se marca como regresión y no solo como "hallazgo nuevo":** un test que pasaba en la versión anterior (llamar `save_cleaned_data` directamente sobre una instancia nueva) ahora falla en la versión actual. Esto es un cambio de comportamiento que rompe un uso previamente válido de la clase.

**Solución propuesta (pendiente):** la misma recomendada para el Hallazgo 2 -- un chequeo de auto-generación al inicio del método, o como mínimo, descomentar la línea `cleaned_data = self.clean_key_column()` si la intención no era deliberadamente cambiar este comportamiento.

---

### Hallazgo B (nuevo) -- El recurso `punkt_tab` de NLTK falta en la descarga automática

**Código actual:**
```python
def __init__(self, ...) -> None:
    ...
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    self.stop_words = set(nltk.corpus.stopwords.words("spanish"))
```

**Problema:** `__init__` descarga los recursos `punkt` y `stopwords`, pero `_clean_stopwords` usa `nltk.word_tokenize`, que en la versión de NLTK usada en este entorno (3.9.4) requiere específicamente el recurso `punkt_tab`, no solo `punkt`. Sin descargarlo, `_clean_stopwords` falla.

**Confirmación en ejecución** (en un entorno donde solo se descargaron `punkt` y `stopwords`, no `punkt_tab`):
```python
>>> instance._clean_stopwords("el gato come pescado")
LookupError:
**********************************************************************
  Resource 'punkt_tab' not found.
  Please use the NLTK Downloader to obtain the resource:
  >>> nltk.download('punkt_tab')
**********************************************************************
```

**Solución propuesta (pendiente):** agregar `nltk.download('punkt_tab', quiet=True)` junto a las otras descargas en `__init__`. Mientras tanto, este requisito se documenta en `README.md`.

---

### Hallazgo C (nuevo) -- Docstring de `clean_key_column` inconsistente con su comportamiento real

**Código actual:**
```python
def clean_key_column(self) -> pd.DataFrame:
    """
    ...
    Returns:
        None
    ...
    """
    ...
    return self.cleaned_data
```

**Problema:** el docstring declara `Returns: None`, pero el método retorna `self.cleaned_data`, un `pd.DataFrame` -- confirmado en ejecución (`isinstance(resultado, pd.DataFrame)` es `True`). El type hint de la firma del método (`-> pd.DataFrame`) sí es correcto; es específicamente el texto del docstring el que quedó desactualizado.

**Por qué importa:** alguien que solo lea la documentación (sin mirar el código o el type hint) podría asumir que no puede usar el valor de retorno, cuando en realidad sí puede y probablemente deba hacerlo.

**Solución propuesta (pendiente):** actualizar el texto del docstring a algo como `Returns: pd.DataFrame: The cleaned DataFrame, also stored in self.cleaned_data.`

---

### Hallazgo D (nuevo) -- `settings.py` duplica el diccionario de acentos sin que `limpieza.py` lo use

**`settings.py` (archivo nuevo):**
```python
accents = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'ä': 'a', 'ë': 'e', 'ï': 'i', 'ö': 'o', 'ü': 'u',
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
            'Ä': 'A', 'Ë': 'E', 'Ï': 'I', 'Ö': 'O', 'Ü': 'U'}
```

**`limpieza.py` (dentro de `_clean_text`, sin importar `settings.py`):**
```python
accents = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
           'ä': 'a', 'ë': 'e', 'ï': 'i', 'ö': 'o', 'ü': 'u',
          'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
          'Ä': 'A', 'Ë': 'E', 'Ï': 'I', 'Ö': 'O', 'Ü': 'U'}
```

**Confirmación:** se comparó ambos diccionarios programáticamente -- son exactamente idénticos hoy. El riesgo no es el estado actual, sino el mantenimiento futuro: si alguien actualiza uno de los dos (por ejemplo, agregando soporte para otro idioma) y olvida actualizar el otro, ambos quedan desincronizados sin ningún aviso, porque no hay ninguna relación de import entre los dos archivos.

**Por qué probablemente existe esta duplicación:** todo indica que `settings.py` se creó con la intención de centralizar este diccionario (posiblemente para reutilizarlo también desde el notebook de pruebas mencionado por el usuario), pero `limpieza.py` nunca se actualizó para importarlo desde ahí.

**Solución propuesta (pendiente):**
```python
# en limpieza.py
from settings import accents
```
y eliminar el diccionario hardcodeado dentro de `_clean_text`.

---

### Hallazgo E (nuevo) -- Los imports de `openpyxl` y `nltk` ahora son obligatorios a nivel de módulo

**Código anterior** (import diferido, con manejo de error):
```python
def load_data(self):
    ...
    elif extension in {'.xlsx', '.xls', '.xlsm'}:
        try:
            import openpyxl
        except ImportError as exc:
            raise ImportError(
                "Excel files require 'openpyxl' to be installed. "
                "Install it with: pip install openpyxl"
            ) from exc
```

**Código actual:**
```python
# al inicio de limpieza.py
import openpyxl
import nltk
```

**Problema:** antes, `openpyxl` solo se importaba (y solo era necesario) si efectivamente se intentaba leer un archivo Excel -- alguien que solo usara CSVs y `_clean_text` no necesitaba tenerlo instalado, y si no lo tenía, recibía un mensaje de error claro y específico. Ahora, **el módulo entero falla al importarse** (`ModuleNotFoundError`) si `openpyxl` o `nltk` no están instalados, sin importar qué funcionalidad se piense usar, y sin el mensaje personalizado que existía antes.

**Confirmación:** se agregaron `openpyxl` y `nltk` como dependencias obligatorias en `pyproject.toml` (antes `openpyxl` no estaba listada en absoluto, ya que solo se necesitaba opcionalmente). Sin este ajuste, `poetry install` no instalaría algo que el módulo ya necesita solo para poder importarse.

**Es una mejora o una regresión, según el caso de uso:** para un proyecto donde siempre se van a usar todas las funcionalidades (que parece ser el caso de este equipo), esto simplifica el código. Para una librería de uso más general, sería preferible volver a los imports diferidos. Se documenta como hallazgo informativo, no como un defecto a corregir necesariamente.

---

### Hallazgo F (nuevo, informativo) -- Cambio de comportamiento: ya no todo error se envuelve en `ValueError`

**Código anterior:**
```python
def load_data(self):
    try:
        ...
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")
```

**Código actual:** ese `try/except Exception` genérico que envolvía absolutamente todo ya no existe. Ahora, por ejemplo, un archivo inexistente propaga `FileNotFoundError` directamente, sin convertirse en `ValueError`.

**Confirmación en ejecución:**
```python
>>> Cleaner(file_path="no_existe.csv", key_column="comment").load_data()
FileNotFoundError: Dataset not found in the file path: no_existe.csv
```

**Por qué es una mejora:** esto es exactamente lo que se recomendaba en la "Observación general" del informe anterior -- distinguir tipos de error en vez de disfrazar todo como `ValueError`. Se marca como hallazgo informativo (no como bug) porque es un cambio deliberado y positivo, pero se documenta porque **rompe la compatibilidad** de cualquier código (incluidos los tests de la revisión anterior) que esperara `ValueError` para este caso.

Adicionalmente, ahora `pd.errors.EmptyDataError` (archivo vacío) se atrapa explícitamente y se convierte en `ValueError` -- un caso que antes no tenía manejo dedicado.

---

## 4. Tests de integración -- el flujo real ahora funciona de punta a punta

En la revisión anterior, la Sección 3.1 documentaba que el Hallazgo 4 bloqueaba el 100% de los flujos reales: ningún archivo CSV o Excel lograba pasar la validación de `load_data()` en un entorno con pandas >= 3.0, y el único modo de probar el resto de la cadena era mockeando `load_data()`.

**Esto se revirtió por completo.** El siguiente test corre la cadena completa -- `load_data -> clean_key_column -> save_cleaned_data -> eliminate_stopwords` -- con un archivo real en disco y **cero mocks**:

```python
def test_full_workflow_load_clean_save_and_remove_stopwords(real_survey_csv, tmp_path):
    output_path = tmp_path / "cleaned_output.csv"
    instance = Cleaner(file_path=real_survey_csv, key_column="comment")

    loaded = instance.load_data()
    assert len(loaded) == 4

    cleaned = instance.clean_key_column()
    assert "comment_clean" in cleaned.columns

    instance.save_cleaned_data(str(output_path))
    assert output_path.exists()

    final = instance.eliminate_stopwords()
    assert "comment_no_stopwords" in final.columns
```

Resultado: **PASSED**, sin ningún workaround.

Adicionalmente, se confirmó en un escenario real (sin mocks) que la dependencia de orden de llamada (Hallazgos 2 y A) también aplica en este contexto:

```python
def test_calling_methods_out_of_order_fails_even_with_real_data(real_survey_csv, tmp_path):
    instance = Cleaner(file_path=real_survey_csv, key_column="comment")

    with pytest.raises(AttributeError, match="cleaned_data"):
        instance.save_cleaned_data(str(tmp_path / "x.csv"))

    with pytest.raises(AttributeError, match="cleaned_data"):
        instance.eliminate_stopwords()
```

Y que, una vez corregido el orden, la misma instancia se recupera sin problema:

```python
def test_correct_call_order_recovers_from_the_above(real_survey_csv, tmp_path):
    instance = Cleaner(file_path=real_survey_csv, key_column="comment")

    with pytest.raises(AttributeError):
        instance.eliminate_stopwords()

    instance.clean_key_column()
    instance.save_cleaned_data(str(tmp_path / "output.csv"))
    result = instance.eliminate_stopwords()

    assert "comment_no_stopwords" in result.columns
```

Esto confirma que los Hallazgos 2 y A son específicamente sobre **orden de llamada**, no sobre que la instancia quede en un estado roto de forma permanente.

---

## 5. Código muerto -- persiste sin cambios

Las dos ramas `else: raise ValueError(...)` documentadas en la revisión anterior siguen presentes, en las mismas condiciones:

```python
# clean_key_column
if data is not None:
    ...
    return self.cleaned_data
else:
    raise ValueError("The data could not be loaded for cleaning.")  # inalcanzable: load_data() nunca devuelve None

# save_cleaned_data
if self.cleaned_data is not None:
    ...
else:
    raise ValueError("The data could not be cleared for saving.")  # inalcanzable, por el mismo motivo
```

`load_data()` sigue sin tener ningún camino que devuelva `None` -- siempre devuelve un `DataFrame` o lanza una excepción. Confirmado por la cobertura de código: las líneas 168 y 240 (los dos `else`) aparecen como no ejecutadas en el reporte de `pytest-cov`.

---

## 6. Cobertura de código

```
Name              Stmts   Miss  Cover   Missing
-----------------------------------------------
src/limpieza.py      72      4    94%   76-77, 168, 240
src/settings.py       1      0   100%
-----------------------------------------------
TOTAL                73      4    95%
78 passed in 3.83s
```

La cobertura subió de 78% (revisión anterior) a 95% combinado. Las líneas restantes sin cubrir:

| Líneas | Qué son | Por qué no están cubiertas |
|---|---|---|
| 76-77 | Rama `except pd.errors.DatabaseError` en `load_data` | Difícil de provocar con un CSV/Excel real sin recurrir a artificios; se considera de bajo valor forzarla |
| 168 | `else: raise ValueError(...)` en `clean_key_column` | Código muerto (ver Sección 5) -- `load_data()` nunca devuelve `None` |
| 240 | `else: raise ValueError(...)` en `save_cleaned_data` | Código muerto (ver Sección 5), misma causa raíz |

`settings.py` tiene cobertura del 100%, ya que el test `test_each_accent_in_settings_is_normalized` (en `test_clean_text.py`) parametriza directamente sobre su diccionario `accents`, ejercitando cada entrada.

---

## 7. Estructura de archivos de la suite

```
tests/
├── test_clean_text.py         (48 tests -- todos PASSED; incluye parametrización desde settings.accents)
├── test_load_data.py          (11 tests -- todos PASSED; sin xfail, Hallazgos 3 y 4 resueltos)
├── test_clean_key_column.py   (6 tests -- todos PASSED; incluye el hallazgo del docstring)
├── test_save_cleaned_data.py  (3 tests -- todos PASSED; incluye la regresión del Hallazgo A)
├── test_stopwords.py          (5 tests -- todos PASSED; sin xfail, Hallazgo 1 resuelto, Hallazgo 2 confirmado en su nueva forma)
└── test_integration.py        (5 tests -- todos PASSED; flujo real completo sin mocks, algo imposible en la revisión anterior)
```

**Resultado de ejecución (`poetry run pytest --cov=src`):**
```
78 passed in 3.83s
```

---

## 8. Próximos pasos sugeridos

1. **Prioridad alta:** decidir sobre el Hallazgo A (regresión en `save_cleaned_data`) y el Hallazgo 2 (persistente en `eliminate_stopwords`) juntos -- ambos comparten la misma causa raíz y la misma solución propuesta (el bloque ya comentado en el código). Esto es lo más urgente porque rompe un uso que antes era válido.
2. Completar la descarga de `punkt_tab` en `__init__` (Hallazgo B), para que el entorno funcione sin pasos manuales adicionales.
3. Decidir si `limpieza.py` debe importar `accents` desde `settings.py` (Hallazgo D), eliminando la duplicación, antes de que ambos diccionarios diverjan sin que nadie lo note.
4. Corregir el docstring de `clean_key_column` (Hallazgo C) para que coincida con su comportamiento real.
5. Evaluar si los imports a nivel de módulo de `openpyxl`/`nltk` (Hallazgo E) son la decisión de diseño deseada para este proyecto, o si conviene volver a imports diferidos.
6. Revisar el código muerto de la Sección 5 -- sigue sin resolverse desde la revisión anterior.
7. Confirmar con el equipo que el cambio de `ValueError` genérico a excepciones específicas (Hallazgo F) es intencional, y actualizar cualquier código externo que dependiera del comportamiento anterior.
