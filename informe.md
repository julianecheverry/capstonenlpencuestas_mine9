# Pipeline de PLN para Análisis de Encuestas Universitarias

Informe técnico del proyecto Capstone — MINE009, Universidad Externado.
Documenta el trabajo realizado en **limpieza de datos, análisis exploratorio
(EDA), ingeniería de características y testing**, así como la arquitectura del
repositorio y las convenciones de calidad de código.

Repositorio: `https://github.com/julianecheverry/capstonenlpencuestas_mine9.git`

---

## 1. Objetivo del proyecto

Analizar respuestas abiertas de encuestas institucionales para transformar
texto libre no estructurado en conocimiento accionable. El corpus proviene de
`Corpus Ejemplo PLN.xlsx` y comprende cinco encuestas:

| Encuesta | Enfoque |
|---|---|
| Evaluación Docente | Valoración de docentes por estudiantes |
| Autoevaluación Docente | Reflexión del docente sobre su práctica |
| Calidad Docentes | Percepción de calidad del cuerpo docente |
| Calidad Administrativos | Percepción del servicio administrativo |
| Calidad Estudiantes | Experiencia institucional del estudiante |

El reto no es la disponibilidad de datos sino su comprensión: convertir miles
de comentarios en patrones interpretables.

---

## 2. Arquitectura del pipeline

El proyecto sigue una arquitectura modular por fases, donde cada módulo tiene
una responsabilidad única y `settings.py` actúa como fuente única de
configuración.

```
                         settings.py
        (acentos · stopwords · semillas de sinónimos)
                              │
     ┌────────────────────────┼────────────────────────┐
     ▼                        ▼                          ▼
 Fase 1                    Fase 2                     Fase 3
 limpieza.py               EDA.py                     Engineering.py
 synonyms.py               (nubes, n-gramas,          (BoW, TF-IDF,
                            longitudes)                impacto sinónimos)
     │                        │                          │
     └────────► columna _no_stopwords_synonyms ◄─────────┘
                              │
                          Fase 4
                       Mod_Bertopic.py
                    (modelado de tópicos)
```

### Contrato de columnas

El flujo de transformación produce columnas con sufijos consistentes:

| Columna | Generada por | Contenido |
|---|---|---|
| `_clean` | `Cleaner.clean_key_column()` | Minúsculas, sin tildes ni símbolos |
| `_no_stopwords` | `Cleaner.eliminate_stopwords()` | Sin palabras vacías |
| `_no_stopwords_synonyms` | `SynonymReplacer.synonyms_to_dataframe()` | Variantes colapsadas a término canónico |

Todo el EDA, la ingeniería de características y el modelado de tópicos consumen
la columna `_no_stopwords_synonyms`.

### Estructura del repositorio

```
capstonenlpencuestas_mine9/
├── data/utilities/          # corpus, accents.xlsx, stopwords.xlsx
├── src/
│   ├── datacleaning/
│   │   ├── limpieza.py       # Fase 1: limpieza + stopwords
│   │   └── synonyms.py       # Fase 1: normalización de sinónimos
│   ├── EDA.py                # Fase 2
│   ├── Engineering.py        # Fase 3
│   ├── Mod_Bertopic.py       # Fase 4
│   └── settings.py           # configuración central
├── notebooks/
│   ├── pruebas_explorar.ipynb        # orquesta Fases 1–3
│   └── Pruebas_Mod_Bertopic.ipynb    # Fase 4 (modelado)
├── figures/                 # nubes y n-gramas exportados
├── models/                  # modelos BERTopic entrenados
├── tests/                   # pruebas unitarias
└── pyproject.toml           # dependencias y configuración de linters
```

---

## 3. Fase 1 — Limpieza de datos

### 3.1 Módulo `limpieza.py`

La clase `Cleaner` carga una hoja de encuesta, normaliza la columna de texto
abierto y elimina *stopwords*. Su constructor recibe la ruta del corpus, el
nombre de la encuesta, la columna clave y la ruta del archivo de *stopwords*.

**Pasos de normalización de texto (`_clean_text`):**
1. Recorte de espacios y conversión a minúsculas.
2. Eliminación de tildes mediante el mapa `accents` de `settings.py`.
3. Eliminación de caracteres especiales, conservando alfanuméricos y `ñ`.

### 3.2 Gestión de *stopwords* como fuente única

Las *stopwords* se combinan de dos orígenes en el momento de construcción:

- **NLTK** — lista base en español (313 términos).
- **`stopwords.xlsx`** — libro con dos hojas:
  - `globales`: columnas `encuesta` | `palabra`; aplican a todas las encuestas.
  - `particulares`: filtradas por el nombre de la encuesta.

Esto centraliza el control de *stopwords* fuera del código: el equipo edita el
Excel sin tocar Python.

### 3.3 Correcciones críticas resueltas

Durante el desarrollo se detectaron y corrigieron tres defectos:

**a) `stopwords.xlsx` no se leía.** La versión inicial solo cargaba la lista de
NLTK más tres palabras fijas, ignorando el Excel del equipo. En consecuencia,
términos como `nr` (9 425 ocurrencias en Evaluación Docente) sobrevivían y
contaminaban las nubes. Corregido: `Cleaner` ahora lee ambas hojas del libro.

**b) Referencias a variables inexistentes.** `_clean_stopwords` usaba una
variable global `stop_words` (inexistente) en lugar de `self.stop_words`, y
`_clean_text` referenciaba `accents` sin importarlo. Ambos corregidos.

**c) Desalineación de tildes entre texto y *stopwords*.** Este fue el defecto
más sutil. El texto se normalizaba quitando tildes (`más` → `mas`), pero la
lista de *stopwords* de NLTK conserva 84 términos acentuados (`más`, `también`,
`está`, `qué`, y todas las conjugaciones verbales). Como la comparación se hacía
entre formatos distintos, esas 84 *stopwords* **nunca coincidían** y se colaban
en el análisis.

La solución fue introducir un helper `_strip_accents` a nivel de módulo y
aplicarlo **tanto al texto como a las listas de *stopwords***, garantizando que
ambos lados de la comparación vivan en el mismo espacio normalizado. Resultado:
la lista pasó de 84 términos acentuados a 0, y palabras como `mas` o `tambien`
ahora se eliminan correctamente.

---

## 4. Fase 1b — Normalización de sinónimos

### 4.1 Módulo `synonyms.py`

Aporte integrado desde la rama de Julián. La clase `SynonymReplacer` colapsa
variantes léxicas sobre un término canónico combinando:

1. **Semillas canónicas** definidas en `settings.py`.
2. **Vecinos más cercanos** de cada semilla según un modelo Word2Vec en español
   (`most_similar`), filtrados por umbral de similitud coseno.
3. **Colombianismos curados** manualmente, que tienen prioridad sobre lo
   automático.

Ejemplos de colapso: *excelente*, *buenísimo*, *chévere* → `bueno`;
*cátedra*, *sesión* → `clase`.

### 4.2 Valor analítico

La normalización de sinónimos reduce el vocabulario de forma medible (más del
20 % en Evaluación Docente, con cerca de 970 variantes colapsadas). El efecto no
es cosmético: al concentrar la frecuencia en términos canónicos, los patrones
dominantes emergen con mayor nitidez en nubes y n-gramas.

---

## 5. Fase 2 — Análisis exploratorio (EDA)

Módulo `EDA.py`. No carga *stopwords* ni sinónimos propios: consume la columna
ya normalizada. Cuatro componentes principales:

### 5.1 `TextLengthAnalyzer`

Estadísticos y visualizaciones de longitud de respuesta (caracteres y palabras):
resumen descriptivo con percentiles, histogramas, diagramas de caja y densidad
comparada.

### 5.2 `WordCloudGenerator` y `SurveyWordClouds`

Generación de nubes de palabras a partir del texto canónico.
`SurveyWordClouds` produce **una nube por encuesta**, resuelve
automáticamente la columna `_synonyms` y exporta PNG de alta resolución
(200 dpi) a `figures/wordclouds/`, listos para la presentación.

**Hallazgos de las nubes:**
- Evaluación Docente y Autoevaluación se centran en el aula: `clase`, `tema`,
  `profesor`, `explica`, `aprendizaje`.
- Las encuestas de Calidad se desplazan a la experiencia institucional:
  `servicio`, `atención`, `mejora`, `información`.

### 5.3 `NgramAnalyzer`

Tablas de frecuencia y gráficos de uni/bi/trigramas. Los bigramas dominantes
(*buena explicación*, *dominio del tema*, *clase dinámica*) revelan estructuras
de opinión que las palabras aisladas no capturan.

### 5.4 Relación longitud–calificación

Análisis añadido para responder si la extensión de la respuesta se asocia con
la calificación. Cruzando la longitud en palabras con `Promedio Evaluación -
General` (escala 1–5) se encontró:

- **Correlación lineal casi nula** (Pearson ≈ 0.04): no hay relación directa
  "a más palabras, mejor o peor nota".
- **Patrón en forma de U**: las calificaciones extremas (1 y 5) generan
  respuestas más largas que las intermedias (3).

**Conclusión:** la longitud no predice la calificación, pero es un indicador de
**intensidad de opinión** — quienes valoran en los extremos elaboran más su
respuesta, sea de descontento o de reconocimiento.

---

## 6. Fase 3 — Ingeniería de características

Módulo `Engineering.py`. Transforma el texto normalizado en matrices numéricas.

### 6.1 `BagOfNgrams` y `TfIdfTransformer`

Envolturas sobre `CountVectorizer` y `TfidfVectorizer` de scikit-learn, con
propiedades protegidas (`matrix`, `feature_names`) que exigen ajuste previo,
utilidades de inspección de vocabulario y ranking de términos por peso TF-IDF.

### 6.2 Análisis de impacto de sinónimos

Dos funciones cuantifican el efecto de la normalización:

- `vocabulary_reduction(antes, después)` — mide la reducción absoluta y
  porcentual del vocabulario.
- `top_collapsed_terms(antes, después)` — lista qué variantes desaparecieron al
  mapearse a su término canónico.

Esta es la evidencia analítica de que la integración de sinónimos mejora las
representaciones aguas abajo.

---

## 7. Fase 4 — Modelado de tópicos (preliminar)

Módulo `Mod_Bertopic.py`. La clase `TopicModeler` encapsula BERTopic:
embeddings multilingües → UMAP → HDBSCAN → c-TF-IDF. Las dependencias pesadas se
importan de forma diferida (*lazy*) para no encarecer el resto del pipeline.

Ofrece: entrenamiento reproducible (`random_state=42`), inspección de tópicos y
términos, documentos representativos, métrica de coherencia (`coherence_score`),
resumen de *outliers*, reducción de tópicos y de *outliers*, visualizaciones
exportables a HTML y persistencia (`save`/`load`).

En una corrida sobre Evaluación Docente se descubrieron 62 tópicos
interpretables (reconocimiento general, metodología, resolución de dudas,
opiniones negativas). El detalle completo corresponde a la entrega 2.

**Corrección de estabilidad numérica.** La visualización de jerarquía de
tópicos (`plot_hierarchy`) fallaba con `ValueError: Distance matrix cannot
contain negative values`. La causa es de precisión de punto flotante: BERTopic
calcula la distancia entre tópicos como `1 - similitud_coseno`, y para tópicos
casi idénticos el redondeo puede producir distancias negativas diminutas
(del orden de `-1e-16`) que disparan la validación interna. Se resolvió pasando
una función de distancia propia (`_safe_cosine_distance`) que recorta los
valores a cero con `np.clip`, eliminando el artefacto numérico sin alterar la
estructura real del dendrograma.

---

## 8. Testing y calidad de código

### 8.1 Estrategia de pruebas

Las pruebas viven en `tests/` y se ejecutan con `pytest` y cobertura vía
`pytest-cov`. El foco está en la lógica determinista de cada módulo: limpieza de
texto, construcción del conjunto de *stopwords*, conteo de n-gramas y
construcción de matrices.

Cada corrección crítica se validó empíricamente sobre el corpus real:
eliminación de `nr` (9 425 → 0), eliminación de *stopwords* acentuadas
(84 → 0), ejecución completa del pipeline tras la ampliación del corpus a
64 697 filas, y estabilización numérica del dendrograma de tópicos
(distancias negativas por punto flotante → 0).

### 8.2 Estándares de código (obligatorios)

Todos los módulos `.py` y notebooks cumplen:

- **Estilo:** PEP 8 y PEP 257; formato Black a 88 caracteres; compatible con Ruff.
- **Tipado:** anotaciones completas compatibles con MyPy estricto.
- **Documentación:** docstrings estilo Google en todos los elementos públicos.
- **Diseño:** funciones pequeñas de responsabilidad única; DRY, KISS, SOLID.
- **Manejo de errores:** excepciones específicas, sin `except` desnudos.
- **Decoradores:** `@staticmethod`, `@property`, `@functools.cached_property`
  donde aplica.
- **Dependencias:** gestionadas con Poetry vía `pyproject.toml`, con secciones
  de configuración para Black, Ruff, MyPy y Pylint.

Los módulos de las fases 1 a 4 pasan Ruff (selección E, F, W, I, N, UP) y Black
sin observaciones.

---

## 9. Reproducibilidad

### 9.1 Instalación

```bash
poetry install                 # dependencias base (fases 1–3)
poetry install --with topics   # añade BERTopic y dependencias de la fase 4
```

### 9.2 Ejecución

1. **Exploración (fases 1–3):** abrir `notebooks/pruebas_explorar.ipynb`,
   reiniciar el kernel y ejecutar todo. El cuaderno resuelve las rutas del
   proyecto automáticamente buscando `pyproject.toml` y localiza el corpus en
   `data/utilities/`.
2. **Modelado (fase 4):** abrir `notebooks/Pruebas_Mod_Bertopic.ipynb`. Entrena
   un modelo por encuesta y exporta métricas y visualizaciones.

### 9.3 Notas operativas

- Tras reemplazar cualquier módulo, **reiniciar el kernel** para que el nuevo
  código se cargue (los objetos como `Cleaner` construyen su estado en el
  `__init__`).
- El corpus y los modelos pesados deben mantenerse fuera del control de
  versiones vía `.gitignore` para no inflar el repositorio.

---

## 10. Entregables generados

| Artefacto | Ubicación |
|---|---|
| Nubes de palabras por encuesta | `figures/wordclouds/*.png` |
| Gráficos de n-gramas | `figures/ngrams/*.png` |
| Visualizaciones de tópicos | `figures/topics/<encuesta>/*.html` |
| Modelos BERTopic entrenados | `models/bertopic_<encuesta>` |
| Cuaderno de exploración | `notebooks/pruebas_explorar.ipynb` |
| Cuaderno de modelado | `notebooks/Pruebas_Mod_Bertopic.ipynb` |

---

## 11. Próximos pasos (entrega 2)

- Interpretación completa de los tópicos BERTopic por encuesta.
- Afinamiento de `min_topic_size` según la coherencia observada.
- Tópicos transversales entre encuestas.
- Análisis de sentimiento que separe la carga emocional de la extensión de la
  respuesta, motivado por el hallazgo de la sección 5.4.
