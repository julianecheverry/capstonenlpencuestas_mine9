# Especificación de cambios al NLP (Etapa 2) — para implementar con Claude

Nov 15, 2026 · @Cris

Qué debe cambiar en el pipeline de NLP para que su salida cumpla el contrato de datos v0.2. Está escrita para pegarla en una sesión de Claude junto al código del NLP.

## 1. Cómo usar esta especificación

Pégala completa en tu sesión de Claude con el repositorio abierto y pídele que la implemente por tareas, en el orden de la sección 4, mostrando el plan antes de cada tarea. La referencia completa es el contrato de datos v0.2 (sección 5); si algo de aquí lo contradice, manda el contrato.

**Reglas de trabajo para Claude:**

1.  Trabajar en <span class="mark">una rama nueva</span> (por ejemplo feature/nlp-contrato-v02), con <span class="mark">un commit por tarea.</span> No hacer push sin que el encargado lo pida.

2.  No imprimir textos de comentarios en la consola ni en los mensajes: revisar con conteos, nombres de columnas y valores de categorías. Los datos son del cliente.

3.  No subir a git archivos de data/raw/ ni de data/processed/.

4.  No cambiar el modelo ni sus parámetros salvo donde la tarea lo pida; las decisiones abiertas (sección 5) se dejan como parámetros con el valor de hoy.

5.  Al terminar cada tarea, correr las verificaciones de la sección 6 que apliquen e informar los conteos.

**Contexto en una línea:** la Etapa 1 (src/preparacion, ya implementada) entrega una fila por respuesta con un identificador estable, respuesta_id. El NLP debe partir de ahí, <span class="mark">conservar ese identificador en todo lo que produce</span> y entregar las cinco tablas de la sección 3.

**Requisito previo:** la Etapa 1, taxonomia.xlsx y el validador (src/nlp_utilidades/taxonomia.py, run_validar_taxonomia.py) están en la rama feature/etapa1-preparacion, que aún no se ha subido a GitHub. Antes de empezar, Cristina sube esa rama y el encargado de NLP crea su rama a partir de ella. Para generar los insumos: python run_etapa1.py.

## 2. <span class="mark">Insumos</span>

| Archivo                | Dónde                                                                       | Qué se usa                                                                                                                                                                                                                 |
|------------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| respuestas.csv         | data/processed/etapa1/\<id_corrida_preparacion\>/ (la corrida más reciente) | Solo filas con es_no_respuesta = False. Columnas: respuesta_id, encuesta, pregunta_codigo, periodo, rol_respondiente, texto_original (sin limpiar), id_corrida_preparacion                                                 |
| catalogo_preguntas.csv | data/processed/etapa1/                                                      | pregunta_codigo → encuesta, enfoque, par_espejo                                                                                                                                                                            |
| taxonomia.xlsx         | data/utilities/ (hoja aspectos)                                             | pregunta_codigo, aspecto_id, aspecto_nombre, descripcion. Se valida con python run_validar_taxonomia.py                                                                                                                    |
| alertas.xlsx           | por confirmar: no está en el repositorio                                    | Lista de términos de alerta, con categoría                                                                                                                                                                                 |
| negaciones.xlsx        | data/utilities/ (hoja negaciones)                                           | Léxico de polaridad del NLP: TERMINO, TERMINO_SIN_TILDE, TIPO, ALCANCE, NOTA. Es compartido: la Etapa 3 lee las filas NEGADOR y CUANTIFICADOR_BAJO. No cambiar columnas ni valores de TIPO sin pasar por el contrato (5.1) |

Lectura de los CSV: pd.read_csv(ruta, dtype=str, keep_default_na=False, encoding="utf-8-sig"). Así el periodo y los identificadores quedan como texto y valores como «NA» no se pierden.

Cifras de la última corrida (20261115-224013_dffe2d4a92) para comprobar la lectura: 158.731 respuestas, 60.621 con comentario. En obs19 (obs19_aspectos_resaltar) son 24.433 con comentario, frente a las 24.380 del corpus anterior. Antes de cerrar la tarea 1, confirmar estas cifras por pregunta con metadatos_preparacion.csv.

## 3. <span class="mark">Salidas</span>

Cinco CSV en utf-8-sig, separados por coma, con decimales con punto, True/False y celdas vacías para los datos ausentes (nunca «NA»). Carpeta propuesta: data/processed/nlp/. El detalle de cada columna está en la sección 5.3 del contrato.

| Archivo                     | Una fila =             | Columnas                                                                                                                                                                                                                                                                                                                                                                                  |
|-----------------------------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| clausulas.csv               | una cláusula           | clausula_id, respuesta_id, orden, texto_clausula, es_opinion, aspecto_id, aspecto_origen, confianza_aspecto, sentimiento, confianza_sentimiento, version_modelo                                                                                                                                                                                                                           |
| auditoria_clasificacion.csv | una cláusula y corrida | clausula_id, id_corrida_nlp, version_modelo, confianza_pos, confianza_neu, confianza_neg, aspecto_1_id, confianza_aspecto_1, aspecto_2_id, confianza_aspecto_2, aspecto_3_id, confianza_aspecto_3, visto_en_entrenamiento                                                                                                                                                                 |
| alertas.csv                 | un término encontrado  | alerta_id, respuesta_id, clausula_id, termino, texto_coincidente, categoria_alerta, negada                                                                                                                                                                                                                                                                                                |
| catalogo_aspectos.csv       | un aspecto posible     | aspecto_id, aspecto_nombre, aspecto_origen, grupo_visual, palabras_clave, encuesta, orden                                                                                                                                                                                                                                                                                                 |
| metadatos_nlp.csv           | una corrida            | id_corrida_nlp, id_corrida_preparacion, encuesta, pregunta_codigo (una fila por pregunta procesada), fecha_corrida, version_modelo, version_taxonomia, umbral_respaldo_hdbscan, accuracy_aspecto, metricas_sentimiento, regla_ajuste_sentimiento, huella_alertas, n_respuestas, n_clausulas, n_alertas, pct_aspecto_clasificador, pct_aspecto_hdbscan, pct_aspecto_regla, pct_sin_aspecto |

**Equivalencia con la salida actual** (clausulas_documentos_unificado.xlsx, hoja unificado):

| Hoy                       | En el contrato                                          | Cambio                                                                                                                                                     |
|---------------------------|---------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| doc_id                    | respuesta_id                                            | Reemplazar por el respuesta_id de respuestas.csv                                                                                                           |
| clausula_id (desde 0)     | orden (desde 1) y clausula_id = respuesta_id\|orden     | Sumar 1 y construir el identificador                                                                                                                       |
| clausula                  | texto_clausula                                          | Renombrar                                                                                                                                                  |
| estado                    | es_opinion                                              | False si estado = NO_OPINION                                                                                                                               |
| aspecto (nombre)          | aspecto_id (código)                                     | Traducir nombre → código con taxonomia.xlsx; ver tarea 3 para las categorías especiales                                                                    |
| asignacion                | aspecto_origen                                          | u4 → clasificador; residual_hdbscan → hdbscan; regla_comentario_general y valoracion_global → regla; no_opinion → sistema; alerta → según decisión abierta |
| p_aspecto_u4              | confianza_aspecto                                       | Para HDBSCAN, la probabilidad de pertenencia al grupo                                                                                                      |
| sentimiento               | sentimiento                                             | Vacío si es_opinion = False                                                                                                                                |
| confianza                 | confianza_sentimiento                                   | Renombrar                                                                                                                                                  |
| modelo_u4_id              | version_modelo                                          | Incluir también la versión del modelo de sentimiento                                                                                                       |
| p_pos, p_neu, p_neg       | confianza_pos, confianza_neu, confianza_neg (auditoría) | Renombrar; son las del modelo, antes del ajuste                                                                                                            |
| 17 columnas p_u4\_\*      | aspecto_1_id a confianza_aspecto_3 (auditoría)          | Ordenar de mayor a menor y guardar las tres primeras con su código                                                                                         |
| u4_visto_en_entrenamiento | visto_en_entrenamiento (auditoría)                      | Renombrar                                                                                                                                                  |
| palabra_alerta            | termino (alertas)                                       | Llevar a la tabla alertas                                                                                                                                  |

Las demás columnas de hoy (clausula_clean, texto_aspecto, n_contenido, polaridad, tiene_negador, etc.) son internas del NLP y no se publican.

## <span class="mark">4. Cambios a implementar, en orden</span>

Cada tarea deja el pipeline funcionando. Empezar por obs19, que ya tiene salida para comparar, y luego extender a obs20, obs23 a obs25 y acreditación.

**Antes de la tarea 1: conservar la partición y las etiquetas del cliente.** El cliente ya revisó y etiquetó cláusulas de obs19, y cambiar de insumo no puede romper eso. Comprobado el 15-nov: el NLP fragmentó sobre el texto original, no sobre el limpio. Las 24.380 respuestas de obs19 cruzan por llave con el corpus viejo, y en las 23.318 donde el texto original y el limpio difieren, el NLP usó siempre el original. Además, 25.585 de las 25.714 cláusulas son pedazos literales de su respuesta. La Etapa 1 entrega ese mismo texto en texto_original. Cuatro pasos previos:

- **Paso previo A. Congelar las etiquetas con su texto.** Guardar las etiquetas del cliente en un archivo aparte que no se regenera. Cada fila lleva la llave de la respuesta (Periodo, ID de respuesta, Código Cruce Autoevaluación), el texto exacto de la cláusula y la etiqueta. Nunca guardar la etiqueta solo con un número de fila o un clausula_id, porque esos cambian.

- **Paso previo B. Fijar la fragmentación.** Fijar las versiones de las librerías que fragmentan (por ejemplo spaCy y su modelo) en pyproject.toml, dejar las reglas de partición en un solo lugar y registrar version_segmentacion en metadatos_nlp. Documentar qué explica las 129 cláusulas que no son literales; si es una normalización, también queda fija.

- **Paso previo C. Prueba de regresión.** Antes de cambiar de insumo, fragmentar con el código actual el corpus viejo de obs19 y comprobar que salen exactamente las mismas 25.714 cláusulas, con el mismo texto y en el mismo orden dentro de cada respuesta. Si no salen iguales, se corrige el código antes de seguir.

- **Paso previo D. Pasar las etiquetas a las cláusulas nuevas.** Después de la tarea 2, cruzar las etiquetas congeladas con las cláusulas nuevas por llave de la respuesta más texto de la cláusula, y entregar una tabla de correspondencia (cláusula vieja → clausula_id nuevo). Reportar cuántas etiquetas se recuperaron exactas y cuántas no. Las que no crucen van a revisión y no se reasignan a ciegas; serán sobre todo las de respuestas que la Etapa 1 colapsó al unir copias (2.807 filas en la conciliación).

Pasar obs19 al corpus de la Etapa 1 es un reproceso del histórico: según la regla «Lo ya procesado no se reclasifica» del contrato (5.2), lo aprueba el cliente, y la tabla del paso D es la correspondencia que esa regla pide.

1.  **Leer de la Etapa 1 la corrida más reciente.** Cada corrida queda en data/processed/etapa1/\<id_corrida_preparacion\>/. No hay aprobación del cliente (decisión del equipo, 16-nov): usar from preparacion.corridas import ultima_corrida y id_corrida, carpeta = ultima_corrida("data/processed/etapa1"); si no hay ninguna corrida, la función da error. Leer carpeta / "respuestas.csv" en lugar de data/processed/corpus_independientes/, filtrando es_no_respuesta = False y la pregunta que se procesa. Comprobar que todas las filas traen ese id_corrida_preparacion y copiarlo a metadatos_nlp. Llevar respuesta_id en todo el flujo en lugar de doc_id. Segmentar texto_original, no un texto limpiado.

2.  **Identificadores de cláusula.** orden desde 1 y sin saltos dentro de cada respuesta, incluidas las cláusulas sin opinión; clausula_id = respuesta_id + "\|" + str(orden).

3.  **Códigos de aspecto.** Validar la taxonomía y compararla con las clases del modelo antes de clasificar:

> from nlp_utilidades.taxonomia import leer_taxonomia, comparar_con_clases  
> from preparacion.configuracion import codigos_preguntas  
> tax = leer_taxonomia("data/utilities/taxonomia.xlsx",  
> codigos_preguntas("data/utilities/preguntas_abiertas.xlsx"))  
> dif = comparar_con_clases(tax, "obs19_aspectos_resaltar", list(modelo.classes\_))  
> assert not dif\["solo_en_taxonomia"\] and not dif\["solo_en_modelo"\], dif

Traducir el nombre del aspecto a aspecto_id con la taxonomía de la pregunta. Las cláusulas sin opinión llevan SIN_ASPECTO y aspecto_origen = sistema. Los grupos de HDBSCAN llevan EMERGENTE\_\<periodo\>\_\<n\> y aspecto_origen = hdbscan. «Comentario General», «VALORACIÓN GLOBAL» y «ALERTA DE CASO ESPECIAL» dependen de decisiones abiertas (sección 5): dejarlos configurables.

4.  **Sentimiento.** Vacío, con sus probabilidades, cuando es_opinion = False. Si se mantiene el ajuste neu_ajustado, describir la regla en metadatos_nlp.regla_ajuste_sentimiento.

5.  **Auditoría.** Escribir auditoria_clasificacion.csv desde las probabilidades que ya existen: p_pos/p_neu/p_neg y el top 3 de las columnas p_u4\_\* (con su código). Las cláusulas que no pasan por el clasificador dejan vacíos los tres aspectos.

6.  **Alertas.** Escribir alertas.csv con una fila por término encontrado. alerta_id = respuesta_id + "\|" + n, donde n numera las coincidencias dentro de la respuesta desde 1. Incluir el fragmento que coincidió. La negación y la categoría dependen de lo que se acuerde (sección 5).

7.  **Catálogo.** Escribir catalogo_aspectos.csv: la taxonomía de las preguntas procesadas (aspecto_origen = clasificador, grupo_visual = Aspectos definidos), más los grupos emergentes de la corrida (hdbscan, Temas emergentes, con palabras_clave), más SIN_ASPECTO (sistema).

8.  **Metadatos.** Escribir metadatos_nlp.csv. id_corrida_nlp tiene el formato AAAAMMDD-HHMMSS\_\<huella\>, en UTC, donde la huella sale de los insumos, la taxonomía, las alertas y la versión del modelo. Copiar id_corrida_preparacion de respuestas.csv. Calcular los conteos y porcentajes.

9.  **Estabilidad.** Fijar las semillas (por ejemplo random_state de UMAP) y guardar UMAP y HDBSCAN ajustados (prediction_data=True), para asignar las respuestas nuevas con approximate_predict sin reajustar.

10. **Validaciones al final.** Implementar los chequeos de la sección 6 dentro del pipeline, para que se detenga con un mensaje claro si alguno falla.

11. **Dependencias (tarea del 16-nov).** Al subir el código del NLP al repo, actualizar dependencies en pyproject.toml con las librerías que de verdad usa, y correr poetry lock. Las Etapas 1 y 3 solo necesitan pandas y openpyxl (y pytest para las pruebas). Hoy están listadas por el código anterior nltk, wordcloud, gensim, numpy, scipy, matplotlib, seaborn, scikit-learn e ipykernel: quitar las que no use nadie y pasar lo pesado o lo de notebooks a un grupo opcional, como ya está topics (BERTopic, UMAP, HDBSCAN). La lista final se revisa en el PR del NLP.

Queda para después, cuando lo anterior funcione: clasificar solo las respuestas nuevas en corridas sucesivas (sección 5.2 del contrato).

## 5. Decisiones abiertas: dejarlas como parámetros

Estas decisiones están en discusión con el equipo. No se resuelven en el código: cada una queda como un parámetro de configuración con el valor de hoy, para cambiarla sin tocar la lógica. El detalle y las cifras están en el informe «Revisión de la salida del NLP — P19».

| Decisión                           | Parámetro sugerido                           | Hoy                                                                         | Propuesta en discusión                                                                             |
|------------------------------------|----------------------------------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| Confianza mínima para usar HDBSCAN | umbral_respaldo_hdbscan                      | sin documentar; 49 % de las cláusulas de obs19 va a HDBSCAN                 | elegirlo con la muestra etiquetada (acierto frente a cobertura)                                    |
| Regla de comentarios cortos        | regla_opinion_general = palabras o contenido | palabras: respuesta de ≤ 5 palabras → Comentario General                    | contenido: solo si n_contenido = 0 → OPINION_GENERAL; las cortas con contenido van al clasificador |
| Valoración global                  | parte de la misma regla                      | categoría propia (276 cláusulas)                                            | OPINION_GENERAL                                                                                    |
| Alerta y aspecto                   | alerta_reemplaza_aspecto = True/False        | True: la alerta reemplaza el aspecto y fuerza NEG                           | False: la cláusula conserva su clasificación y la alerta va a alertas.csv                          |
| Negación en alertas                | alerta_negacion                              | no se detecta por término (57 de 86 cláusulas con alerta tienen un negador) | marcar negada por término                                                                          |
| Ajuste de sentimiento              | regla_ajuste_sentimiento                     | neu_ajustado (1.318 cláusulas), sin describir                               | describirlo o quitarlo                                                                             |

Las «propuestas en discusión» no están aprobadas. Implementar las dos opciones solo donde sea barato; si no, dejar la de hoy y el parámetro listo.

## 6. Criterios de aceptación

La corrida se entrega solo si se cumplen todos. Cada uno se verifica con conteos, sin imprimir textos.

- [ ] Prueba de regresión: el corpus viejo de obs19, fragmentado con el código actual, produce las mismas 25.714 cláusulas.

- [ ] Se informa cuántas etiquetas del cliente se recuperaron exactas en las cláusulas nuevas, con la tabla de correspondencia; las que no cruzan quedan listadas para revisión.

- [ ] Cada respuesta de respuestas.csv con es_no_respuesta = False de la pregunta procesada tiene al menos una cláusula. Ninguna no-respuesta tiene cláusulas. Para obs19: 24.433 respuestas.

- [ ] clausula_id y alerta_id no se repiten; orden empieza en 1 y no tiene saltos en ninguna respuesta.

- [ ] Todo respuesta_id de clausulas y alertas existe en respuestas.csv.

- [ ] Todo aspecto_id de clausulas y de la auditoría existe en catalogo_aspectos.

- [ ] La taxonomía coincide con las clases del modelo para cada pregunta procesada (comparar_con_clases sin diferencias).

- [ ] sentimiento solo toma POS, NEU, NEG o vacío, y está vacío exactamente cuando es_opinion = False.

- [ ] aspecto_origen solo toma clasificador, hdbscan, regla o sistema.

- [ ] Todas las confianzas están entre 0 y 1, y confianza_pos + confianza_neu + confianza_neg = 1 ± 0,001 cuando hay sentimiento.

- [ ] auditoria_clasificacion tiene una fila por cláusula de clausulas, y en ella los aspectos van de mayor a menor confianza.

- [ ] metadatos_nlp registra el id_corrida_preparacion de la corrida procesada, y sus conteos coinciden con los archivos.

- [ ] Con el mismo insumo y las mismas versiones, dos corridas producen los mismos archivos.

- [ ] Los CSV se leen con dtype=str, encoding="utf-8-sig" sin errores, y el periodo sigue siendo texto («2025-1»).

Al terminar, informar en un mensaje: cláusulas por aspecto_origen (cantidad y %), cláusulas sin opinión, alertas, y cualquier criterio que no se haya podido cumplir y por qué.

**<span class="mark">Wapp- instrucciones GITHUB</span>**

Lo que deben hacer tus compañeros (después de que se fusione el \#20):

<span class="mark">Guardar su trabajo</span>: con git status ven si tienen algo pendiente. Si es así, lo guardan con un commit en una rama propia, nunca en master. Si cambiaron algún Excel de data/utilities/, hacen además una copia fuera del repo.

<span class="mark">Crear rama de trabajo</span>

git checkout -b feature/contrato-etapa2-nlp

<span class="mark">Traer master:</span>

> git fetch origin
>
> git switch master
>
> git pull

<span class="mark">Si tienen una rama de trabajo, la actualizan con lo nuevo:</span>

> git switch feature/contrato-etapa2-nlp
>
> git merge master

<span class="mark">Si salen conflictos</span>, git merge --abort deja todo como estaba y consultan a Cris

4\. <span class="mark">Rehacer el entorno</span>: poetry install. Si alguna vez movieron la carpeta del repo, primero rmdir /s /q .venv.

5\. <span class="mark">Comprobar</span>: poetry run pytest, con 98 pruebas aprobadas.
