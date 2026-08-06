# capstonenlpencuestas_mine9

Proyecto capstone para la aplicación de **Procesamiento de Lenguaje Natural (PLN/NLP)** a las respuestas abiertas de las encuestas de la Oficina de Aseguramiento de la Calidad (UEXT).

El pipeline cubre: limpieza de texto → normalización de sinónimos → análisis exploratorio (EDA) → ingeniería de características → modelado de tópicos (BERTopic).

Repositorio: https://github.com/julianecheverry/capstonenlpencuestas_mine9

---

## 1. Instalación del ambiente con Poetry

El proyecto usa [Poetry](https://python-poetry.org/) para gestionar dependencias y el entorno virtual. Se requiere **Python ≥ 3.12, < 4.0**.

### 1.1 Instalar Poetry

Si aún no tienes Poetry instalado:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Verifica la instalación:

```bash
poetry --version
```

### 1.2 Clonar el repositorio

```bash
git clone https://github.com/julianecheverry/capstonenlpencuestas_mine9.git
cd capstonenlpencuestas_mine9
```

### 1.3 Instalar dependencias

Instalación base (limpieza, EDA, ingeniería de características):

```bash
poetry install
```

Esto crea automáticamente un entorno virtual (o usa `poetry.lock` para fijar versiones exactas) e instala, entre otras, `pandas`, `nltk`, `scikit-learn`, `gensim`, `matplotlib`, `seaborn` y `wordcloud`.

El grupo de dependencias **`topics`** (modelado de tópicos con BERTopic) es opcional y pesado (`bertopic`, `sentence-transformers`, `umap-learn`, `hdbscan`, `plotly`). Instálalo solo si vas a ejecutar el Paso 10 del notebook:

```bash
poetry install --with topics
```

Para el desarrollo (tests y linters: `pytest`, `pytest-cov`, `black`, `ruff`, `mypy`, `pylint`), el grupo `dev` se instala por defecto junto con `poetry install`.

### 1.4 Activar el entorno / ejecutar comandos

```bash
poetry shell        # abre una shell dentro del entorno virtual
# o, sin activar la shell:
poetry run python mi_script.py
poetry run pytest   # correr la suite de tests
poetry run jupyter lab   # abrir Jupyter para trabajar con los notebooks
```

---

## 2. Estructura del proyecto

```
capstonenlpencuestas_mine9/
├── .vscode/                     # Configuración del editor (settings, snippets)
├── data/
│   ├── Corpus Ejemplo PLN.xlsx  # Corpus crudo de encuestas (datos de entrada)
│   └── utilities/
│       ├── accents.xlsx         # Mapa de normalización de tildes/acentos
│       └── stopwords.xlsx       # Stopwords globales y particulares por encuesta
├── documents/                   # Carpeta destino para entregables/documentos generados
├── figures/                     # Figuras exportadas (nubes de palabras, n-gramas, etc.)
│   ├── calidad_estudiantes/
│   ├── ngrams/
│   └── wordclouds/
├── notebooks/
│   ├── pruebas_explorar.ipynb   # Notebook principal: pipeline PLN completo
│   ├── prueba_limpieza2.ipynb   # Notebook de pruebas de limpieza de texto
│   └── HALLAZGOS_Y_TESTING.md   # Notas de hallazgos y estrategia de testing
├── src/
│   ├── settings.py              # Configuración global (rutas, acentos, stopwords)
│   ├── EDA.py                   # Análisis exploratorio de datos
│   ├── Engineering.py           # Ingeniería de características
│   ├── Mod_Bertopic.py          # Modelado de tópicos con BERTopic
│   └── datacleaning/
│       ├── limpieza.py          # Limpieza y remoción de stopwords
│       └── synonyms.py          # Normalización de sinónimos vía Word2Vec
├── tests/                       # Suite de pruebas unitarias e integración (pytest)
│   ├── conftest.py
│   ├── test_clean_key_column.py
│   ├── test_clean_text.py
│   ├── test_integration.py
│   ├── test_load_data.py
│   ├── test_save_cleaned_data.py
│   ├── test_stopwords.py
│   └── test_synonyms.py
├── CODEOWNERS                   # Dueños/revisores por defecto del repositorio
├── pyproject.toml               # Configuración de Poetry, dependencias y pytest
├── poetry.lock                  # Versiones exactas de dependencias (lockfile)
└── README.md
```

---

## 3. Descripción de los archivos en `src`

El código en `src/` implementa el pipeline de PLN en fases secuenciales:

- **`settings.py`** — Módulo de configuración central. Carga el mapa de normalización de acentos (`accents.xlsx`) y construye los diccionarios de stopwords (`stopwords.xlsx`), separando las stopwords **globales** (aplican a todas las encuestas) de las **particulares** (específicas por encuesta). Es importado por el resto de módulos para mantener una única fuente de verdad.

- **`datacleaning/limpieza.py`** — Contiene la clase `Cleaner`, que carga una hoja de encuesta, normaliza la columna de texto clave (minúsculas, acentos, caracteres especiales) y elimina las stopwords combinando la lista base de NLTK en español con las listas de `settings.py`.

- **`datacleaning/synonyms.py`** — Contiene la clase `SynonymReplacer`, que reemplaza términos del texto por su sinónimo canónico usando un modelo Word2Vec preentrenado en español, combinado con un diccionario curado de coloquialismos colombianos. Es independiente de `Cleaner`: opera directamente sobre texto plano.

- **`EDA.py`** — Análisis Exploratorio de Datos. Consume el texto ya limpio y normalizado (columna `_no_stopwords_synonyms`) y ofrece análisis de distribución de longitud de texto, generación de nubes de palabras (global, por columna, por categoría, por encuesta) y análisis de frecuencia de n-gramas (uni/bi/tri), pensado para producir visualizaciones reutilizables en reportes/PPT.

- **`Engineering.py`** — Ingeniería de Características. Transforma el texto normalizado en matrices numéricas mediante Bag of N-Grams (`CountVectorizer`) y TF-IDF (`TfidfVectorizer`), e incluye utilidades para inspeccionar vocabulario y pesos, además de cuantificar el impacto (reducción de vocabulario) de la normalización de sinónimos.

- **`Mod_Bertopic.py`** — Modelado de Tópicos. Contiene la clase `TopicModeler`, que aplica BERTopic sobre el texto normalizado: genera embeddings con un Sentence-Transformer multilingüe, reduce dimensionalidad con UMAP, agrupa con HDBSCAN y extrae términos representativos por tópico vía c-TF-IDF. Las dependencias pesadas se importan de forma diferida (lazy) para no penalizar el resto del pipeline si no se usa modelado de tópicos.

---

## 4. Descripción de `notebooks/pruebas_explorar.ipynb`

Este notebook es el **notebook principal / de orquestación** del proyecto: encadena todas las fases del pipeline de PLN sobre el corpus completo de encuestas, en 10 pasos:

1. **Importación de módulos y configuración global** — instala/importa dependencias (incluye `gensim`) y define parámetros globales del corpus, incluida la carpeta de exportación de figuras para el PowerPoint final.
2. **Construcción del diccionario de sinónimos** — arma una sola vez el diccionario de sinónimos, reutilizado para todas las encuestas.
3. **Carga, limpieza y normalización** — define y ejecuta `process_survey()` para limpiar y normalizar (sinónimos) cada encuesta, verificando el "contrato de columnas" esperado.
4. **Impacto de la normalización de sinónimos** — mide cuantitativamente cuánto se reduce el vocabulario al normalizar, e inspecciona qué términos específicos fueron colapsados en la encuesta principal.
5. **Nubes de palabras por encuesta** — genera y analiza nubes de palabras por encuesta usando `SurveyWordClouds`, incluyendo un mapa de nubes a diapositivas.
6. **N-gramas con normalización de sinónimos** — calcula y grafica n-gramas comparativos para la encuesta principal y gráficos de unigramas compactos para todas las encuestas.
7. **Distribución de longitud de respuestas** — analiza histogramas de longitud de texto y explora la relación entre la extensión de la respuesta y la calificación otorgada, con una sección de interpretación.
8. **Ingeniería de características** — construye matrices Bag-of-Words y TF-IDF sobre el texto normalizado con sinónimos, compara vocabularios y muestra los términos con mayor TF-IDF promedio.
9. **EDA automatizado end-to-end** — corre el análisis exploratorio completo para una encuesta específica ("Calidad Estudiantes") y resume los entregables generados.
10. **Modelado de tópicos con BERTopic** — entrena `TopicModeler`, muestra el panorama de tópicos, sus términos y documentos representativos, genera visualizaciones interactivas (mapa intertópicos, términos por tópico, jerarquía) y persiste el modelo entrenado en disco.

En conjunto, el notebook funciona como el **caso de uso de referencia** que valida e integra todos los módulos de `src/`, y produce las figuras y hallazgos que alimentan los documentos/reportes finales del proyecto.

---

## 5. Ejecutar los tests

```bash
poetry run pytest
```

La configuración de `pytest` (en `pyproject.toml`) añade `src` y `src/datacleaning` al `pythonpath`, por lo que los módulos pueden importarse directamente (p. ej. `from limpieza import Cleaner`) sin necesidad de instalarlos como paquete.