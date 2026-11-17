# Contrato de datos — preguntas abiertas de encuestas (v0.2)

Nov 13, 2026 · @Cris

Versión 0.2 · borrador para revisión del equipo.

## 1. Propósito y alcance

Este documento define <span class="mark">qué tablas produce cada pieza del flujo, qué columnas tienen y cómo se construyen sus identificadores</span>, para que la pieza siguiente las use sin ajustes y el cliente las conecte a sus tableros de Power BI. Si una pieza cambia una tabla o una columna de las que se describen aquí, se rompen las piezas que vienen después. Por eso este contrato solo cambia con una versión nueva, acordada por el equipo.

### Para quién es

| Pieza                                   | Responsable                                 | Secciones que necesita                            |
|-----------------------------------------|---------------------------------------------|---------------------------------------------------|
| Etapa 1 · Preparación                   | Cristina                                    | 3 y 4                                             |
| <span class="mark">Etapa 2 · NLP</span> | <span class="mark">Encargado del NLP</span> | <span class="mark">3, salidas de la 4, y 5</span> |
| Etapa 3 · Limpieza y n-gramas           | Encargado del EDA                           | 3, salida clausulas de la 5, y 6                  |
| Tableros de Power BI                    | Cliente (integra el modelo)                 | 3 y 7                                             |

### Encuestas incluidas

| Encuesta (código)           | Qué es                                                       | Preguntas abiertas                                                                                                  | Periodos                | Estado               |
|-----------------------------|--------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|-------------------------|----------------------|
| evadoc_evaluacion           | Evaluación docente hecha por los estudiantes (solo Pregrado) | obs19_aspectos_resaltar, obs20_oportunidades_mejora                                                                 | 2024-2 a 2026-1         | Etapa 1 implementada |
| evadoc_autoevaluacion       | Autoevaluación del docente (solo Pregrado)                   | obs23_percepcion_progreso, obs24_aspectos_resaltar, obs25_oportunidades_mejora                                      | 2024-2 a 2026-1         | Etapa 1 implementada |
| autoevaluacion_acreditacion | Autoevaluación para acreditación de programas                | acred_sugerencias_complementarias                                                                                   | 2024-1 a 2026-1         | Etapa 1 implementada |
| calidad_servicios           | Encuesta de calidad de los servicios de la Universidad       | 10 códigos con prefijo cal\_ (sugerencias generales, sugerencias sobre el programa, seis servicios y dos de motivo) | 2021, 2022, 2023 y 2025 | Etapa 1 implementada |

### Principios

1.  **Una sola puerta de entrada.** Solo la Etapa 1 lee las bases del cliente. Las demás piezas leen las tablas de la etapa anterior, así todas cuentan las mismas respuestas.

2.  **Un tipo de fila por tabla.** Cada tabla dice qué representa una fila y nunca mezcla dos cosas. Por ejemplo, la autoevaluación de un docente aparece una vez, no una vez por cada estudiante del grupo.

3.  **Los identificadores siempre se arman igual.** La misma respuesta tiene el mismo identificador en cualquier corrida (ver sección 3).

4.  **Nada se borra en silencio.** Las respuestas vacías o «NR» y los problemas de la base se marcan; no se eliminan.

5.  **El cliente aprueba antes de seguir.** El NLP solo procesa una corrida de la Etapa 1 que el cliente haya aprobado (ver sección 2).

6.  **Lo común en un lugar, lo propio en otro.** Lo que tienen todas las encuestas va en respuestas; lo propio de cada encuesta (grupo, docente, programa…) va en su tabla de contexto.

7.  **Datos personales mínimos.** No se usa el Id Estudiante. El Código Cruce Autoevaluación se usa tal como viene en la base que se recibe. Si una copia de los datos debe salir del cliente, se anonimiza la base cruda antes de la Etapa 1.

## 2. Flujo de datos

<img src="media/image1.png" style="width:6in;height:6.05357in" alt="flujo de datos · 2 etapas, limpieza por cláusula y 2 consumidores" />

flujo de datos · 2 etapas, limpieza por cláusula y 2 consumidores

Solo la Etapa 1 lee las bases del cliente. El NLP, la limpieza, el EDA y Power BI trabajan sobre las tablas que produce, así que todos cuentan las mismas respuestas. La limpieza va después del NLP porque los n-gramas y las nubes de palabras se calculan por cláusula. Entrega una sola tabla ngramas, que usan tanto el EDA (si quedará implementado) como Power BI. En verde está lo que ya existe; lo demás está en curso de implementación.

### Responsables y punto de control

| Etapa                   | Responsable | Lee                                                                                           | Produce                                                                                                                                                   |
|-------------------------|-------------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 · Preparación         | Cristina    | Bases del cliente y preguntas_abiertas.xlsx                                                   | respuestas, contexto\_\<encuesta\> (una por encuesta), respuestas_oferta, respuestas_programa, catalogo_preguntas, inconsistencias, metadatos_preparacion |
| 2 · NLP                 | Julián      | respuestas (<span class="mark">solo las que tienen comentario) de una corrida aprobada</span> | clausulas, alertas, auditoria_clasificacion, catalogo_aspectos, metadatos_nlp                                                                             |
| 3 · Limpieza y n-gramas | Jose        | clausulas                                                                                     | clausulas_texto, ngramas, metadatos_limpieza                                                                                                              |

**Punto de control entre la Etapa 1 y la Etapa 2.** <span class="mark">La Etapa 1 se puede correr muchas veces mientras el cliente corrige su base con ayuda del informe de inconsistencias.</span> El NLP no toma la última corrida, sino la que el cliente apruebe:

1.  **Cada corrida se guarda aparte.** La Etapa 1 escribe cada corrida en su propia carpeta, data/processed/etapa1/\<id_corrida_preparacion\>/, y nunca sobrescribe una anterior.

2.  **<span class="mark">Envío al cliente</span>.** Cristina envía al cliente el informe de inconsistencias y el metadatos_preparacion.csv de la corrida, citando su id_corrida_preparacion en el asunto del correo.

3.  **Revisión.** El cliente responde aprobando esa corrida o pidiendo correcciones. Si corrige su base, se hace una corrida nueva y se vuelve al paso 2.

4.  **Registro de la aprobación.** Con la respuesta del cliente, Cristina corre python run_aprobar_corrida.py \<id_corrida_preparacion\> "\<quién aprobó\>" "\<referencia del correo\>". El comando comprueba que la carpeta de esa corrida exista y agrega una fila a data/processed/etapa1/aprobaciones.csv con id_corrida_preparacion, fecha_aprobacion_utc, aprobado_por y referencia. El correo se guarda como evidencia.

5.  **El NLP <span class="mark">toma la última aprobada</span>.** Lee la última fila de aprobaciones.csv, carga la carpeta de esa corrida, comprueba que todas las filas de respuestas.csv traigan ese id_corrida_preparacion y lo copia a metadatos_nlp. Si no hay ninguna corrida aprobada, se detiene.

6.  **Una aprobación nueva no borra la anterior.** aprobaciones.csv guarda el historial completo. Lo que el NLP ya clasificó se trata según la regla «Lo ya procesado no se reclasifica» (sección 5.2).

Los pasos 1 y 4 los implementa la Etapa 1 (versión 1.7); el paso 5, el NLP.

Así, cualquier cifra del tablero se puede rastrear hasta la versión de la base que el cliente aprobó.

## 3. Convenciones comunes

Estas reglas aplican a todas las tablas de todas las etapas.

### 3.1 Archivos

- Un archivo CSV por tabla, con el mismo nombre de la tabla: respuestas.csv, clausulas.csv.

- <span class="mark">Codificación UTF-8</span> con BOM (utf-8-sig), para que Excel y Power BI muestren bien las tildes.

- Separador: coma. Los textos que traen comas o saltos de línea van entre comillas; pandas lo hace solo.

- Los decimales se escriben con punto (0.85). Como el formato regional de Colombia usa coma, en Power Query esas columnas se importan con la configuración regional «Inglés (Estados Unidos)».

- Las columnas de texto que parecen fechas o números (como el periodo «2025-1») se importan como texto: en Excel, con Datos → Desde texto/CSV, no con doble clic; en Power Query, fijando la columna como Texto.

- Los CSV no se abren con doble clic en Excel: se importan. Excel admite como máximo 1.048.576 filas por hoja y ngramas.csv puede pasar de esa cifra (obs19 sola tiene 487.573 filas). Con doble clic, Excel muestra solo las primeras filas, y si alguien guarda el archivo así, queda cortado sin que se note. Para revisar en Excel: Datos → Desde texto/CSV → Cargar en el **modelo de datos**, o filtrar antes en Power Query, por ejemplo por encuesta o por n. Power BI y pandas leen el archivo completo.

- Las <span class="mark">columnas sí / no se escriben True o False</span>.

- Un <span class="mark">dato ausente es una celda vacía</span>, nunca «NA», «null» ni «-».

### 3.2 Nombres de columnas

- En español, en minúsculas, sin tildes y con las palabras unidas por guion bajo. Ejemplo: «Ciudad/Énfasis» se convierte en ciudad_enfasis. La ñ se conserva.

- Las columnas que vienen del cliente conservan su nombre convertido así (codigo_del_grupo), salvo que se acuerde otro (llave se publica como llave_oac).

- Las columnas propias siguen estos patrones, para que el nombre diga qué contiene:

| Si la columna es…         | Su nombre…                 | Ejemplos                              |
|---------------------------|----------------------------|---------------------------------------|
| Un identificador          | termina en \_id            | respuesta_id, clausula_id, aspecto_id |
| Sí / no                   | empieza con es\_ o tiene\_ | es_no_respuesta, es_opinion           |
| Un conteo                 | empieza con n\_            | n_clausulas                           |
| Una probabilidad de 0 a 1 | empieza con confianza\_    | confianza_aspecto                     |
| Una versión               | empieza con version\_      | version_etapa1                        |
| Un texto                  | empieza con texto\_        | texto_original, texto_clausula        |

### 3.3 Identificadores

Un identificador es el valor que distingue una fila de todas las demás y que usan las otras tablas para unirse a ella. Todos se construyen con la misma receta, de modo que la misma respuesta produzca el mismo identificador en cualquier corrida.

| Identificador                                               | Cómo se arma                                                                    | Ejemplo                                                             |
|-------------------------------------------------------------|---------------------------------------------------------------------------------|---------------------------------------------------------------------|
| id_origen                                                   | Las llaves de la encuesta, sin el periodo, unidas con «-» (ver tabla siguiente) | 555001-27                                                           |
| respuesta_id                                                | encuesta, periodo, id_origen y pregunta_codigo, unidos con «\|»                 | evadoc_evaluacion\|2024-2\|555001-27\|obs20_oportunidades_mejora    |
| clausula_id                                                 | respuesta_id + «\|» + orden de la cláusula                                      | evadoc_evaluacion\|2024-2\|555001-27\|obs20_oportunidades_mejora\|2 |
| alerta_id                                                   | respuesta_id + «\|» + número de la coincidencia dentro de la respuesta          | …\|obs20_oportunidades_mejora\|1                                    |
| id_corrida_preparacion, id_corrida_nlp, id_corrida_limpieza | Fecha y hora UTC de la corrida + «\_» + huella de los insumos                   | 20261114-180717_dd32fed281                                          |

| Encuesta                    | id_origen                          |
|-----------------------------|------------------------------------|
| evadoc_evaluacion           | ID de respuesta + «-» + Id Docente |
| evadoc_autoevaluacion       | Código Cruce Autoevaluación        |
| autoevaluacion_acreditacion | ID de respuesta                    |
| calidad_servicios           | ID de Acceso                       |

### 3.4 Versiones

Cada etapa tiene una versión de su código (por ejemplo, version_etapa1 = 1.4.0) y la registra en sus metadatos. Si una etapa cambia lo que entrega, sube su versión y este contrato se actualiza antes de que la etapa siguiente use el cambio.

### 3.5 Glosario

- **Respuesta:** lo que una persona escribió en una pregunta abierta, en un periodo. Es una fila de respuestas.

- **No-respuesta:** respuesta vacía o sin contenido («NR», «Ninguna», «No aplica», «De acuerdo» en una pregunta abierta…). Se conserva, marcada con es_no_respuesta = True.

- **Copia:** cuando la base del cliente repite la misma respuesta en varias filas (por ejemplo, una por cada facultad que ofrece el curso). Se deja una sola respuesta y las copias se registran aparte.

- **Llave:** columnas de la base del cliente que, juntas, identifican una respuesta (por ejemplo, Periodo + ID de respuesta + Id Docente).

- **Cláusula:** fragmento de una respuesta que expresa una sola idea. «Explica muy bien, pero llega tarde» tiene dos.

- **Aspecto:** tema del que habla una cláusula (por ejemplo, Puntualidad).

- **N-grama:** secuencia de 1, 2 o 3 palabras seguidas de una cláusula; con ellos se arman las nubes y los «top» de expresiones.

- **Corrida:** cada ejecución de una etapa. Tiene su propio identificador.

- **Catálogo:** tabla con los valores posibles de algo (preguntas, aspectos), con su nombre para mostrar.

## <span class="mark">4. Etapa 1 · Preparación</span>

La Etapa 1 lee las bases del cliente y deja **una fila por cada respuesta de una persona a una pregunta abierta**, con su texto original, sin limpiar. Decide qué es no-respuesta, elige una sola copia cuando la base repite la misma respuesta y marca los problemas que encuentra en la base. No segmenta, no clasifica ni quita palabras: eso lo hacen las etapas siguientes.

- Código: src/preparacion/ (versión 1.6.1). Se ejecuta con python run_etapa1.py.

- Salidas: carpeta data/processed/etapa1/.

- Responsable: Cristina.

### 4.1 Insumos

**Bases del cliente**

| Encuesta                                  | Archivo                                                   | Hoja                | Cómo viene                                                                                                                                                 |
|-------------------------------------------|-----------------------------------------------------------|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| evadoc_evaluacion y evadoc_autoevaluacion | Consolidado_Histórico_Evaluación_Docente_Anonimizado.xlsx | EvaDoc Pregrado     | Una fila por evaluación de un estudiante; cada pregunta abierta es una columna. La autoevaluación del docente viene pegada en cada fila de sus estudiantes |
| autoevaluacion_acreditacion               | Encuestas_Autoevaluación.xlsx                             | Autoevaluación      | Una fila por envío y programa: el mismo envío se repite una vez por cada programa asociado                                                                 |
| calidad_servicios                         | Consolidado_Preguntas_Abiertas.xlsx                       | Respuestas Abiertas | Una fila por respuesta; la pregunta viene como texto en la columna Pregunta (formato largo)                                                                |

**Configuración** (data/utilities/preguntas_abiertas.xlsx). Todo lo que cambia entre encuestas se define aquí, no en el código:

| Hoja             | Qué define                                                                                                                                                    |
|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| preguntas        | Cada pregunta abierta: su código (clave_logica), las redacciones con que aparece en la base (alias_redaccion, puede haber varias por código) y si está activa |
| archivos         | Los nombres de archivo conocidos de cada encuesta                                                                                                             |
| etapa1           | Por encuesta: llaves, columnas de contexto y de copias, filtros, rol y renombres (ver tabla siguiente)                                                        |
| no_respuesta     | Textos que cuentan como no-respuesta (NR, Ninguna, No aplica…)                                                                                                |
| valores_cerrados | Valores de escala cerrada que, si son todo el texto de una respuesta abierta, también cuentan como no-respuesta (De acuerdo, N/S…)                            |
| roles            | Equivalencias para unificar el rol del respondiente                                                                                                           |

| Tipo (hoja etapa1)               | Para qué                                                                                                                |
|----------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| llave                            | Columnas que identifican una respuesta, en orden; la primera es el periodo                                              |
| contexto                         | Columnas que van a la tabla de contexto de la encuesta                                                                  |
| oferta / programa                | Columnas que se guardan por cada copia en respuestas_oferta o en respuestas_programa                                    |
| desempate                        | Columna para desempatar copias (orden alfabético)                                                                       |
| filtro                           | Condición para incluir una fila, de la forma «Columna = valor»                                                          |
| rol / columna_rol                | Rol fijo de la encuesta, o columna de donde se toma                                                                     |
| renombrar                        | Nombre de una columna en las salidas, de la forma «Columna = nombre» (p. ej. «llave = llave_oac»)                       |
| formato                          | «ancho» (cada pregunta es una columna; es el valor por defecto) o «largo» (cada fila es una respuesta, como en calidad) |
| columna_pregunta / columna_texto | Solo formato largo: columna con el texto de la pregunta y columna con la respuesta                                      |

### 4.2 Reglas

**1. Solo cuenta quien respondió la encuesta.** En la evaluación docente entran las filas con Evaluación = Sí y en la autoevaluación las de Autoevaluación = Sí. Si las copias de una misma respuesta tienen la marca mezclada, la respuesta entra si alguna copia dice Sí.

**2. Una respuesta por llave.** La llave dice qué filas de la base son la misma respuesta:

| Encuesta                    | Llave                                  | Por qué                                                                                                                                                                      |
|-----------------------------|----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| evadoc_evaluacion           | Periodo + ID de respuesta + Id Docente | Un curso ofrecido por varias facultades repite la respuesta una vez por facultad de oferta. No se usa el Id Estudiante porque cambia entre las copias de una misma respuesta |
| evadoc_autoevaluacion       | Periodo + Código Cruce Autoevaluación  | La autoevaluación se repite en cada fila de estudiante del grupo. El Código Conteo no se usa porque su receta cambia entre periodos                                          |
| autoevaluacion_acreditacion | Periodo + ID de respuesta              | El envío se repite una vez por programa asociado; se asume que es el mismo envío de la misma persona                                                                         |
| calidad_servicios           | Año + ID de Acceso                     | La base no trae copias: cada fila ya es una respuesta                                                                                                                        |

**3. Elección de copia.** Si una llave tiene varias copias, se elige una **por pregunta** (para obs19 puede ganar una copia y para obs20 otra):

1.  Gana la copia que tiene comentario.

2.  Entre las que tienen comentario, gana la más larga.

3.  Si siguen empatadas, gana la de Facultad de oferta primera en orden alfabético y, después, la que aparece primero en la base.

Las copias no se pierden: se guardan en respuestas_oferta (evaluación docente) o respuestas_programa (acreditación).

**4. No-respuesta.** Se normaliza el texto original (minúsculas, sin tildes ni puntuación, espacios simples) y se compara con las hojas no_respuesta y valores_cerrados. La no-respuesta se marca con es_no_respuesta = True y **no se elimina**. Se decide siempre sobre el texto original, nunca después de quitar palabras. Los comentarios cortos con contenido («bien», «excelente») sí son comentarios. Un valor cerrado solo cuenta si es el texto completo: «De acuerdo, pero falta…» es comentario.

**5. Marcas de inconsistencia.** Cuando la base trae un problema, la respuesta se marca en marca_inconsistencia; nunca se corrige en silencio. Si hay varios, se separan con «; ».

| Código                            | Significa                                                                                                             |
|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| comentarios_distintos_en_copias   | Las copias de la respuesta traen comentarios distintos; se conservó el más largo                                      |
| valor_cerrado_en_pregunta_abierta | Alguna copia trae en la pregunta abierta un valor de escala cerrada (p. ej. «De acuerdo»); se trató como no-respuesta |

**6. Rol unificado.** En evaluación docente y acreditación, el rol usa las etiquetas del cliente: Estudiantes, Docentes, Administrativos, Directivos y coordinadores, Graduados (evaluación = Estudiantes, autoevaluación = Docentes). En calidad, el rol es la población que registra el cliente (Docentes e investigadores, Estudiantes de posgrados, Personal operativo…).

**7. La corrida se detiene, con un mensaje claro,** si falta una columna configurada, si un valor de rol no tiene equivalencia en la hoja roles, si ninguna redacción registrada coincide con la base o, en formato largo, si alguna fila trae una redacción que no está registrada o una redacción aparece en dos códigos. Así nada nuevo se procesa por error.

**8. Código Cruce.** Se usa siempre tal como viene en la base que se recibe, sin codificarlo.

**9. Formato largo.** Cuando la pregunta viene como texto (calidad), cada fila recibe el código de su pregunta comparando su redacción normalizada con las redacciones registradas en la hoja preguntas. Así, las variantes que solo cambian en la numeración, los espacios o las mayúsculas quedan en el mismo código.

**10. Identificadores.** respuesta_id e id_origen se construyen con los datos de la llave, nunca con el número de fila, porque ese número cambia si el archivo se ordena o se filtra. Se guardan como texto, aunque parezcan números, para no perder ceros a la izquierda; periodo también es texto, con el formato del cliente («2025-1», o «2021» en calidad). El carácter «\|» está reservado para unir las partes y no puede aparecer dentro de ninguna; si aparece en una llave, la corrida debe detenerse (pendiente: la Etapa 1 aún no lo comprueba).

### 4.3 Salidas

| Archivo                                      | Una fila =                                                                | Para qué                                                                                                                                                                                  |
|----------------------------------------------|---------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| respuestas.csv                               | una respuesta de una persona a una pregunta abierta (todas las encuestas) | Entrada del NLP; comentario completo y % de personas que comentan en Power BI                                                                                                             |
| contexto\_\<encuesta\>.csv                   | una respuesta (la misma de respuestas)                                    | Columnas propias de cada encuesta para filtrar y unir con los datos del cliente                                                                                                           |
| respuestas_oferta.csv                        | una copia de una respuesta de evaluación docente                          | Que un comentario cuente para todas las facultades que ofrecen el curso, sin duplicarse                                                                                                   |
| respuestas_programa.csv                      | un programa asociado a un envío de acreditación                           | Que un comentario cuente para todos sus programas, sin duplicarse; filtro por programa                                                                                                    |
| catalogo_preguntas.csv                       | una pregunta abierta                                                      | Texto y rol de cada pregunta                                                                                                                                                              |
| inconsistencias.csv                          | una respuesta marcada y un código de inconsistencia                       | Revisión de los problemas de la base                                                                                                                                                      |
| metadatos_preparacion.csv                    | una pregunta en una corrida                                               | Conteos de control y trazabilidad de la corrida                                                                                                                                           |
| Informe_inconsistencias_corrida\_\<id\>.xlsx | una hoja por inconsistencia (I01 a I21) y una hoja de resumen             | Informe para el cliente. Va a reportes/etapa 1 preparación/ (fuera de git) y no lo lee ninguna etapa. Los comentarios aparecen como \[TEXTO\]. Si falla, la corrida sigue y solo se avisa |

**respuestas**

| Columna                | Tipo    | Obligatoria | De dónde sale                                                                                                                             | Descripción                                                                              |
|------------------------|---------|-------------|-------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| respuesta_id           | texto   | sí          | Construida: encuesta\|periodo\|id_origen\|pregunta_codigo                                                                                 | Identificador de la respuesta (sección 3.3)                                              |
| encuesta               | texto   | sí          | Configuración: hoja preguntas                                                                                                             | Código de la encuesta (sección 1)                                                        |
| pregunta_codigo        | texto   | sí          | Configuración: clave_logica de la hoja preguntas, asignada por el nombre de la columna (formato ancho) o por la redacción (formato largo) | Código de la pregunta; existe en catalogo_preguntas                                      |
| rol_respondiente       | texto   | sí          | Base del cliente (la columna que indique columna_rol, p. ej. Población) o valor fijo (rol), traducido con la hoja roles                   | Rol unificado (regla 6)                                                                  |
| periodo                | texto   | sí          | Base del cliente: primera columna de la llave (Periodo; Año en calidad)                                                                   | Como lo escribe el cliente: «2025-1», o el año en calidad                                |
| id_origen              | texto   | sí          | Construida: las demás columnas de la llave (hoja etapa1), unidas con -                                                                    | Las llaves de la respuesta sin el periodo (sección 3.3)                                  |
| texto_original         | texto   | no          | Base del cliente: la columna de la pregunta (formato ancho) o columna_texto (formato largo)                                               | La respuesta tal como está en la base, sin limpiar. Vacía si la persona no escribió nada |
| es_no_respuesta        | sí / no | sí          | Calculada: texto_original comparado con las hojas no_respuesta y valores_cerrados                                                         | True si es no-respuesta (regla 4)                                                        |
| marca_inconsistencia   | texto   | no          | Calculada por la Etapa 1 (regla 5)                                                                                                        | Código o códigos de inconsistencia; vacía si no hay                                      |
| id_corrida_preparacion | texto   | sí          | Generada en cada corrida: fecha y huella de los archivos                                                                                  | Corrida que generó la fila                                                               |

**contexto\_\<encuesta\>** (una fila por respuesta, unida por respuesta_id)

Cada tabla empieza con respuesta_id (construida, igual que en respuestas). Las demás columnas vienen de la base del cliente: se eligen en la hoja etapa1 (tipo contexto) y en la salida llevan el nombre original en snake_case.

| Encuesta                    | Columna en la salida        | Columna en la base del cliente |
|-----------------------------|-----------------------------|--------------------------------|
| evadoc_evaluacion           | id_de_respuesta             | ID de respuesta                |
| evadoc_evaluacion           | id_docente                  | Id Docente                     |
| evadoc_evaluacion           | codigo_cruce_autoevaluacion | Código Cruce Autoevaluación    |
| evadoc_evaluacion           | codigo_del_grupo            | Codigo del Grupo               |
| evadoc_evaluacion           | codigo_materia              | Codigo Materia                 |
| evadoc_evaluacion           | facultad_estudiante         | Facultad Estudiante            |
| evadoc_autoevaluacion       | codigo_cruce_autoevaluacion | Código Cruce Autoevaluación    |
| evadoc_autoevaluacion       | id_docente                  | Id Docente                     |
| evadoc_autoevaluacion       | codigo_del_grupo            | Codigo del Grupo               |
| evadoc_autoevaluacion       | codigo_materia              | Codigo Materia                 |
| autoevaluacion_acreditacion | id_de_respuesta             | ID de respuesta                |
| autoevaluacion_acreditacion | poblacion                   | Población                      |
| calidad_servicios           | id_de_acceso                | ID de Acceso                   |
| calidad_servicios           | poblacion                   | Población                      |
| calidad_servicios           | facultad_departamento_area  | Facultad/Departamento/Área     |

**respuestas_oferta** (solo evaluación docente)

| Columna                     | De dónde sale                                                                            | Descripción                                                             |
|-----------------------------|------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| respuesta_id                | Construida, igual que en respuestas                                                      | Existe en respuestas                                                    |
| facultad_de_oferta          | Base del cliente: Facultad de oferta                                                     | Facultad de oferta de la copia                                          |
| codigo_cruce_autoevaluacion | Base del cliente: Código Cruce Autoevaluación (solo evaluación; vacía en autoevaluación) | Valor de la copia (en 2024-2 el Código Cruce puede variar entre copias) |
| codigo_materia              | Base del cliente: Codigo Materia                                                         | Valor de la copia                                                       |
| codigo_del_grupo            | Base del cliente: Codigo del Grupo                                                       | Valor de la copia                                                       |

Las columnas de la base se eligen en la hoja etapa1 (tipo oferta) y llevan el nombre original en snake_case.

**respuestas_programa** (solo acreditación)

| Columna            | De dónde sale                                                          | Descripción                                                                                                                             |
|--------------------|------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| respuesta_id       | Construida, igual que en respuestas                                    | Existe en respuestas                                                                                                                    |
| llave_oac          | Base del cliente: llave (renombrada en la hoja etapa1, tipo renombrar) | Columna llave del cliente, sin cambios; sirve para cruzar con sus tableros. No es un código de programa: un programa puede tener varias |
| facultad           | Base del cliente: Facultad                                             | Atributos del programa asociado al envío                                                                                                |
| departamento       | Base del cliente: Departamento                                         | ídem                                                                                                                                    |
| programa           | Base del cliente: Programa                                             | ídem                                                                                                                                    |
| ciudad_enfasis     | Base del cliente: Ciudad/Énfasis                                       | ídem                                                                                                                                    |
| nivel_de_formacion | Base del cliente: Nivel de Formación                                   | ídem                                                                                                                                    |
| nivel_academico    | Base del cliente: Nivel Académico                                      | ídem                                                                                                                                    |
| snies              | Base del cliente: SNIES                                                | ídem                                                                                                                                    |

Las columnas de la base se eligen en la hoja etapa1 (tipo programa) y llevan el nombre original en snake_case, salvo las renombradas.

**catalogo_preguntas**

| Columna          | De dónde sale                                                                                                                                              | Descripción                                                                         |
|------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| pregunta_codigo  | Configuración: hoja preguntas, columna clave_logica                                                                                                        | Código de la pregunta                                                               |
| encuesta         | Configuración: hoja preguntas, columna encuesta                                                                                                            | Código de la encuesta                                                               |
| texto_pregunta   | Base del cliente: el encabezado de la columna que coincidió con alias_redaccion (hoja preguntas). En las bases en formato largo, el primer alias_redaccion | Redacción de la pregunta en la base                                                 |
| rol_respondiente | Configuración: hoja etapa1, rol; si el rol viene por fila, el texto «según columna '\<columna_rol\>'»                                                      | Quién la responde                                                                   |
| hoja             | Configuración: hoja preguntas, columna hoja_excel                                                                                                          | Hoja de la base de donde se leyó                                                    |
| enfoque          | Configuración: hoja preguntas, columna enfoque                                                                                                             | resaltar, mejora, progreso, general, servicio o motivo                              |
| par_espejo       | Configuración: hoja preguntas, columna par_espejo                                                                                                          | Pregunta equivalente del otro rol (obs19 ↔ obs24, obs20 ↔ obs25); vacía si no tiene |

**inconsistencias**: una fila por respuesta marcada y código de inconsistencia.

| Columna              | De dónde sale                      | Descripción                                                                 |
|----------------------|------------------------------------|-----------------------------------------------------------------------------|
| respuesta_id         | Copiada de respuestas              | Respuesta marcada                                                           |
| encuesta             | Copiada de respuestas              | Código de la encuesta                                                       |
| pregunta_codigo      | Copiada de respuestas              | Código de la pregunta                                                       |
| periodo              | Copiada de respuestas              | Periodo de la respuesta                                                     |
| id_origen            | Copiada de respuestas              | Llaves de la respuesta sin el periodo, para buscarla en la base del cliente |
| marca_inconsistencia | Calculada por la Etapa 1 (regla 5) | Un solo código por fila: una respuesta con dos códigos tiene dos filas      |
| detalle              | Tabla de códigos de la regla 5     | Explicación del código                                                      |

**metadatos_preparacion**: una fila por pregunta en cada corrida: identificación de la corrida, qué se leyó y conteos de control. Es lo que el cliente revisa, junto con el informe de inconsistencias, antes de aprobar una corrida.

| Columna                             | Grupo       | De dónde sale                                                                  | Descripción                                            |
|-------------------------------------|-------------|--------------------------------------------------------------------------------|--------------------------------------------------------|
| id_corrida_preparacion              | Corrida     | Generada: fecha y huella de los archivos leídos, la configuración y la versión | Identifica la corrida; es la que aprueba el cliente    |
| fecha_utc                           | Corrida     | Generada: reloj al correr                                                      | Fecha y hora de la corrida (UTC)                       |
| version_etapa1                      | Corrida     | Código: src/preparacion                                                        | Versión de la Etapa 1                                  |
| encuesta                            | Qué se leyó | Configuración: hoja preguntas                                                  | Encuesta                                               |
| pregunta_codigo                     | Qué se leyó | Configuración: hoja preguntas                                                  | Pregunta                                               |
| archivo                             | Qué se leyó | Configuración: hoja archivos                                                   | Base del cliente leída                                 |
| sha256_archivo                      | Qué se leyó | Calculada sobre el archivo                                                     | Huella: cambia si cambia la base                       |
| hoja                                | Qué se leyó | Configuración: hoja preguntas                                                  | Hoja del Excel leída                                   |
| columna_texto                       | Qué se leyó | Configuración: redacción de la pregunta o columna_texto                        | Columna de la base con la respuesta                    |
| filas_crudas                        | Conteo      | Calculada                                                                      | Filas de la base para esa pregunta                     |
| filas_excluidas_por_filtro          | Conteo      | Calculada (regla 1)                                                            | Filas que no son respondientes efectivos               |
| filas_sin_llave                     | Conteo      | Calculada                                                                      | Filas sin alguna columna de la llave                   |
| respuestas                          | Conteo      | Calculada                                                                      | Filas en respuestas                                    |
| respuestas_con_comentario           | Conteo      | Calculada                                                                      | Respuestas con es_no_respuesta = False                 |
| no_respuestas                       | Conteo      | Calculada (regla 4)                                                            | Respuestas con es_no_respuesta = True                  |
| copias_descartadas                  | Conteo      | Calculada (regla 3)                                                            | Copias que no se eligieron                             |
| llaves_con_varias_copias            | Conteo      | Calculada                                                                      | Respuestas que tenían más de una copia                 |
| llaves_con_comentarios_distintos    | Conteo      | Calculada (regla 5)                                                            | Respuestas con comentarios distintos entre copias      |
| llaves_con_valor_cerrado            | Conteo      | Calculada (regla 5)                                                            | Respuestas con un valor cerrado en la pregunta abierta |
| respuestas_con_marca_inconsistencia | Conteo      | Calculada (regla 5)                                                            | Respuestas con alguna marca                            |

### 4.4 Validaciones

La Etapa 1 revisa su propia salida al final de cada corrida y **se detiene** si algo falla:

- respuesta_id no se repite.

- Las columnas obligatorias de respuestas no tienen vacíos.

- marca_inconsistencia solo trae códigos de la tabla de la regla 5.

- Cada tabla de contexto tiene exactamente una fila por respuesta de su encuesta.

- Cada fila de respuestas_oferta y respuestas_programa corresponde a una respuesta que existe y es de una encuesta que usa esa tabla, y cada respuesta de esas encuestas tiene al menos una fila en su tabla de copias.

Además, dos corridas con los mismos insumos producen los mismos archivos (salvo el identificador y la fecha de la corrida).

### 4.5 Pendiente en esta etapa

- Hecho (16-nov, v1.8): la Etapa 1 genera el informe de inconsistencias para el cliente en cada corrida, con las 17 revisiones anteriores y 4 nuevas de calidad de servicios (src/preparacion/informe.py). Pero se socializa con el equipo el obtenido en el análisis.

## <span class="mark">5. Etapa 2 · NLP</span>

El NLP toma las respuestas con comentario de una corrida de la Etapa 1 aprobada por el cliente, las divide en cláusulas y a cada cláusula le asigna un aspecto y un sentimiento. También detecta las alertas.

- Responsable: encargado del NLP.

### 5.1 Insumos

- respuestas.csv de la corrida aprobada, solo las filas con es_no_respuesta = False. Se usa texto_original, sin limpiar.

- catalogo_preguntas.csv, para saber a qué encuesta y pregunta pertenece cada respuesta.

- data/utilities/taxonomia.xlsx, la lista fija de aspectos por pregunta (hoja aspectos: pregunta_codigo, aspecto_id, aspecto_nombre, descripcion). Su dueño es el NLP. Antes de cada corrida se valida con python run_validar_taxonomia.py, y aspecto_nombre debe coincidir con las clases del modelo entrenado (nlp_utilidades.taxonomia.comparar_con_clases); si no, ese aspecto nunca se asigna.

- alertas.xlsx, la lista de términos de alerta.

- data/utilities/negaciones.xlsx, el léxico de polaridad (hoja negaciones: TERMINO, TERMINO_SIN_TILDE, TIPO, ALCANCE, NOTA; la hoja \_LEEME explica cada valor). TIPO toma cinco valores: NEGADOR, ADVERSATIVO, CUANTIFICADOR_BAJO, INTENSIFICADOR y MODAL_ATENUANTE. Su dueño es el NLP, que lo usa para ajustar el sentimiento y cortar cláusulas. **Es un insumo compartido:** la Etapa 3 también lo lee (6.1), así que cambiar sus columnas o sus valores de TIPO pasa por el contrato.

### 5.2 Reglas

1.  **Segmentación sobre el texto original**, con su puntuación y sus negaciones. Cada respuesta con comentario tiene al menos una cláusula. Las cláusulas se numeran en orden desde 1, sin saltos, incluidas las que no expresan opinión (es_opinion = False).

2.  **Identificadores.** clausula_id y alerta_id se arman a partir del respuesta_id que entrega la Etapa 1, tal cual, sin volver a calcularlo, y se guardan como texto (sección 3.3).

3.  **Aspecto.** Lo decide el clasificador (MPNet ajustado + regresión logística) con la lista fija de aspectos de su pregunta (taxonomia.xlsx), definida en esta misma etapa 2 de NLP. Estudiantes y docentes tienen listas distintas. ACORDAR: Si ningún aspecto alcanza la confianza mínima, HDBSCAN asigna un grupo de respaldo; esos grupos cambian en cada corrida y se presentan aparte, como «Temas emergentes». Una cláusula sin aspecto lleva SIN_ASPECTO. Calidad aún no tiene lista ni etiquetado manual para entrenar el clasificador.

4.  **Sentimiento** por cláusula: POS, NEU o NEG, con su confianza. Las cláusulas sin opinión (es_opinion = False, lo que hoy el NLP marca como NO_OPINION) no llevan sentimiento y van con SIN_ASPECTO. No es lo mismo que es_no_respuesta: esa la decide la Etapa 1 para la respuesta completa (vacía, «NR», un valor de escala) y esas respuestas nunca llegan al NLP; es_opinion la decide el NLP para cada cláusula de un comentario real. En la salida actual de obs19 solo aparece en 118 respuestas de una sola cláusula sin palabras de contenido (n_contenido = 0); 38 de ellas ya las detecta la Etapa 1 con la lista no_respuesta. Pendiente: que el NLP confirme la regla exacta. Si una regla del NLP ajusta la clase del modelo, se documenta en metadatos_nlp y la auditoría conserva las probabilidades originales.

5.  **Alertas** (acuerdo del equipo, 15-nov; ya es lo que hace el NLP en obs19): una alerta aplica al comentario completo. Si una respuesta tiene un término de alertas.xlsx, no se fragmenta ni se clasifica: va completa a alertas (una fila por término encontrado) y en clausulas queda como una sola cláusula (orden = 1) con SIN_ASPECTO y sin sentimiento. Si una negación cercana anula el término («nunca ha sido irrespetuoso»), la alerta se marca como negada: se muestra, pero no cuenta. Costo conocido: esas respuestas no aportan a los aspectos (86 de 24.380 en obs19). Por decidir con el NLP: si una respuesta con alerta negada se procesa normal (57 de las 86 tienen un negador).

6.  **Lo ya procesado no se reclasifica.** Cada corrida clasifica solo las respuestas cuyo respuesta_id no tiene resultado vigente; las anteriores conservan su aspecto, sentimiento, grupo emergente y version_modelo, aunque el modelo cambie. Para que eso sea posible:

    - Semillas fijas (por ejemplo random_state de UMAP) y versiones de librerías y modelos registradas en metadatos_nlp.

    - UMAP y HDBSCAN se ajustan una vez y se guardan con versión (prediction_data=True); las respuestas nuevas se asignan con approximate_predict, sin reajustar.

    - Reprocesar el histórico es una decisión explícita: corrida completa con versión nueva, aprobada por el cliente, y una tabla que relacione los grupos emergentes viejos con los nuevos.

    - Si la Etapa 1 cambia una respuesta ya clasificada (por ejemplo, pasa a no-respuesta), el resultado anterior se retira y queda registrado en metadatos_nlp.

    - Un tema emergente que se repite en varios periodos se propone para la taxonomía fija.

### 5.3 Salidas

**clausulas**: una fila por cada cláusula de cada respuesta con comentario.

<table>
<colgroup>
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 10%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 23%" />
</colgroup>
<thead>
<tr class="header">
<th><mark>Columna</mark></th>
<th><mark>Tipo</mark></th>
<th><mark>Oblig.</mark></th>
<th><mark>De dónde sale</mark></th>
<th><mark>Hoy en la salida de obs19</mark></th>
<th><mark>Descripción</mark></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>clausula_id</td>
<td>texto</td>
<td>sí</td>
<td>Construida en la Etapa 2: respuesta_id|orden</td>
<td>doc_id + clausula_id (desde 0)</td>
<td>Identificador de la cláusula</td>
</tr>
<tr class="even">
<td>respuesta_id</td>
<td>texto</td>
<td>sí</td>
<td>Copiada de respuestas (Etapa 1)</td>
<td>no existe</td>
<td>Debe existir en respuestas</td>
</tr>
<tr class="odd">
<td>orden</td>
<td>entero</td>
<td>sí</td>
<td>Etapa 2: segmentación</td>
<td>clausula_id + 1</td>
<td>Posición de la cláusula dentro de la respuesta, desde 1</td>
</tr>
<tr class="even">
<td>texto_clausula</td>
<td>texto</td>
<td>sí</td>
<td>Etapa 2: segmentación de texto_original</td>
<td>clausula</td>
<td>Texto original de la cláusula (no el neutralizado)</td>
</tr>
<tr class="odd">
<td>es_opinion</td>
<td>booleano</td>
<td>sí</td>
<td>Etapa 2</td>
<td>estado ≠ NO_OPINION</td>
<td>Falso para cláusulas sin contenido evaluable</td>
</tr>
<tr class="even">
<td>aspecto_id</td>
<td>texto</td>
<td>sí</td>
<td>Etapa 2: clasificador, HDBSCAN o regla, con los códigos de taxonomia.xlsx</td>
<td>aspecto (nombre, no código)</td>
<td>Código estable; debe existir en catalogo_aspectos. Sin aspecto (incluidas las cláusulas sin opinión): SIN_ASPECTO</td>
</tr>
<tr class="odd">
<td>aspecto_origen</td>
<td>texto</td>
<td>sí</td>
<td>Etapa 2</td>
<td>asignacion</td>
<td>clasificador, hdbscan (respaldo cuando ninguna etiqueta encaja), regla (una regla del NLP, p. ej. opinión general), sistema (SIN_ASPECTO) o manual (asignado a mano por el cliente a una cláusula que había quedado en SIN_ASPECTO)</td>
</tr>
<tr class="even">
<td>confianza_aspecto</td>
<td>decimal 0–1</td>
<td>sí</td>
<td>Etapa 2</td>
<td>p_aspecto_u4 (solo clasificador; no hay columna para HDBSCAN)</td>
<td>Clasificador: probabilidad de la clase ganadora. HDBSCAN: probabilidad de pertenencia al grupo</td>
</tr>
<tr class="odd">
<td>sentimiento</td>
<td>texto</td>
<td>sí, salvo sin opinión</td>
<td>Etapa 2: modelo de sentimiento, con ajuste documentado</td>
<td>sentimiento (el modelo está en sentimiento_modelo)</td>
<td><p>POS, NEU o NEG. Vacío si es_opinion = False</p>
<p>Power BI los muestra como Positivo, Neutral y Negativo con una tabla de etiquetas.</p></td>
</tr>
<tr class="even">
<td>confianza_sentimiento</td>
<td>decimal 0–1</td>
<td>sí, salvo sin opinión</td>
<td>Etapa 2</td>
<td>confianza</td>
<td>Probabilidad de la clase ganadora</td>
</tr>
<tr class="odd">
<td>version_modelo</td>
<td>texto</td>
<td>sí</td>
<td>Etapa 2</td>
<td>modelo_u4_id (solo el clasificador)</td>
<td>Versión del clasificador y del modelo de sentimiento</td>
</tr>
</tbody>
</table>

clausulas lleva solo la clase ganadora y su confianza. Las probabilidades completas, para auditar, van en auditoria_clasificacion.

**auditoria_clasificacion**: una fila por cláusula clasificada y corrida del NLP. Se relaciona 1 a 1 con clausulas por clausula_id, así hereda el RLS. Se carga en Power BI en una página de auditoría, para que el cliente vea en un solo lugar por qué se asignó cada sentimiento y aspecto.

| Columna                | Tipo        | Oblig.                | De dónde sale                                                                    | Hoy en la salida de obs19                                         | Descripción                                                                                                       |
|------------------------|-------------|-----------------------|----------------------------------------------------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| clausula_id            | texto       | sí                    | Igual que en clausulas (Etapa 2)                                                 | doc_id + clausula_id                                              | Debe existir en clausulas                                                                                         |
| id_corrida_nlp         | texto       | sí                    | Generada en cada corrida del NLP                                                 | no existe                                                         | Corrida que produjo la fila                                                                                       |
| version_modelo         | texto       | sí                    | Etapa 2                                                                          | modelo_u4_id                                                      | Igual a la de clausulas                                                                                           |
| confianza_pos          | decimal 0–1 | sí, salvo sin opinión | Etapa 2: modelo de sentimiento, antes de cualquier ajuste                        | p_pos                                                             | Probabilidad de POS                                                                                               |
| confianza_neu          | decimal 0–1 | sí, salvo sin opinión | Etapa 2: modelo de sentimiento, antes de cualquier ajuste                        | p_neu                                                             | Probabilidad de NEU                                                                                               |
| confianza_neg          | decimal 0–1 | sí, salvo sin opinión | Etapa 2: modelo de sentimiento, antes de cualquier ajuste                        | p_neg                                                             | Probabilidad de NEG                                                                                               |
| aspecto_1_id           | texto       | sí                    | Etapa 2: la clase más probable del clasificador, con su código de taxonomia.xlsx | se deriva de las 17 columnas p_u4\_\* (aspecto_u4 trae el nombre) | Aspecto más probable según el clasificador (puede diferir de clausulas.aspecto_id si decidió HDBSCAN o una regla) |
| confianza_aspecto_1    | decimal 0–1 | sí                    | Etapa 2                                                                          | p_aspecto_u4                                                      | Probabilidad de aspecto_1_id                                                                                      |
| aspecto_2_id           | texto       | sí                    | Etapa 2: segunda clase del clasificador                                          | se deriva de p_u4\_\*                                             | Segundo más probable                                                                                              |
| confianza_aspecto_2    | decimal 0–1 | sí                    | Etapa 2                                                                          | se deriva de p_u4\_\*                                             | Probabilidad de aspecto_2_id                                                                                      |
| aspecto_3_id           | texto       | sí                    | Etapa 2: tercera clase del clasificador                                          | se deriva de p_u4\_\*                                             | Tercero más probable                                                                                              |
| confianza_aspecto_3    | decimal 0–1 | sí                    | Etapa 2                                                                          | se deriva de p_u4\_\*                                             | Probabilidad de aspecto_3_id                                                                                      |
| visto_en_entrenamiento | booleano    | sí                    | Etapa 2: la cláusula estaba en la muestra de entrenamiento                       | u4_visto_en_entrenamiento                                         | Verdadero si la cláusula estuvo en el entrenamiento del clasificador. Se excluye al medir el acierto              |

Reglas: las tres confianzas de sentimiento suman 1 (±0,001); el sentimiento de clausulas es el de mayor confianza, salvo un ajuste documentado del NLP, y queda vacío en las cláusulas sin opinión; los aspectos van de mayor a menor confianza y existen en catalogo_aspectos para esa pregunta. Decisión del equipo (15-nov): se guardan los tres más probables y no todos porque cada pregunta tiene un número distinto de aspectos. Las cláusulas que no pasan por el clasificador (hoy: sin opinión, alerta y valoración global) dejan vacíos los tres aspectos y sus confianzas.

**alertas**

Una fila por cada término de alertas.xlsx detectado en una respuesta. Si una respuesta dispara tres términos, tiene tres filas. En el tablero, el director ve el comentario completo (de respuestas) junto al término que lo disparó, para juzgar rápido si es una alerta real o una falsa alarma («nos acosa con trabajos»).

| Columna           | De dónde sale                                     | Hoy en la salida de obs19                                                                                                   | Tipo     | Oblig. | Descripción                                                                                                                                                                                  |
|-------------------|---------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|----------|--------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| alerta_id         | Construida en la Etapa 2                          | no existe                                                                                                                   | texto    | sí     | respuesta_id\|número de coincidencia                                                                                                                                                         |
| respuesta_id      | Copiada de respuestas (Etapa 1)                   | no existe (doc_id)                                                                                                          | texto    | sí     | Debe existir en respuestas                                                                                                                                                                   |
| clausula_id       | Igual que en clausulas (Etapa 2)                  | La marca va en la única cláusula de la respuesta (estado = ALERTA, n_clausulas = 1): esa cláusula es el comentario completo | texto    | sí     | La cláusula única de la respuesta (orden = 1), porque una respuesta con alerta no se fragmenta (regla de alertas, 5.2)                                                                       |
| termino           | Insumo: alertas.xlsx                              | palabra_alerta y palabra_alerta_doc; hoy dicen lo mismo, porque la respuesta tiene una sola cláusula                        | texto    | sí     | Término de la lista, tal como está en alertas.xlsx. Es para revisar y auditar: el tablero muestra categoria_alerta. Lo que se escribió exactamente va en texto_coincidente                   |
| texto_coincidente | Etapa 2, sobre texto_original                     | no existe                                                                                                                   | texto    | sí     | Fragmento exacto del comentario que coincidió                                                                                                                                                |
| categoria_alerta  | Insumo: alertas.xlsx                              | no existe                                                                                                                   | texto    | sí     | Etiqueta con que la alerta aparece en el tablero (p. ej., acoso, discriminación, salud mental), asignada a cada término en alertas.xlsx. Si la lista no tiene categorías, hay que agregarlas |
| negada            | Etapa 2: detección de negación cercana al término | no existe; tiene_negador es de toda la cláusula, no del término                                                             | booleano | sí     | Verdadero si una negación cercana anula el término («nunca ha sido irrespetuoso»); se muestra, pero no cuenta como alerta                                                                    |

**catalogo_aspectos**

Una fila por cada aspecto que puede aparecer en clausulas: los de cada pregunta en taxonomia.xlsx, los grupos que cree HDBSCAN y SIN_ASPECTO. Los grupos de HDBSCAN cambian en cada corrida, por lo que en el tablero se agrupan como «Temas emergentes» y no se mezclan con los aspectos fijos (si se mezclaran, las tendencias entre periodos se romperían).

**Lista fija.** Sale de taxonomia.xlsx (insumo del NLP). Cada pregunta tiene su propia lista. Un mismo aspecto_id en varias preguntas es el mismo aspecto, con el mismo nombre y descripción, y no cambia aunque cambie el nombre: es lo que une los periodos. El origen de la lista está en el informe de revisión del NLP.

| Columna        | Tipo   | Oblig.       | Descripción                                                                                                                                                                                                                                                                                                                                                          |
|----------------|--------|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| aspecto_id     | texto  | sí           | Estable. Clasificador: código fijo (p. ej., CLARIDAD). HDBSCAN: EMERGENTE\_\<periodo\>\_\<n\>                                                                                                                                                                                                                                                                        |
| aspecto_nombre | texto  | sí           | Nombre para mostrar. HDBSCAN: etiqueta propuesta a partir de sus palabras clave                                                                                                                                                                                                                                                                                      |
| aspecto_origen | texto  | sí           | clasificador, hdbscan, regla, sistema (para SIN_ASPECTO) o manual (aspecto que el cliente asignó a mano a una cláusula que había quedado en SIN_ASPECTO; decisión del equipo, 15-nov)                                                                                                                                                                                |
| grupo_visual   | texto  | sí           | Aspectos definidos (los de taxonomia.xlsx, que asigna el clasificador) o Temas emergentes (los grupos de HDBSCAN, con aspecto_origen = hdbscan). Es la etiqueta que ve el usuario del tablero. Decisión del equipo (15-nov): un tema emergente no se muestra en el tablero hasta que el cliente le asigna un aspecto; esas cláusulas pasan a aspecto_origen = manual |
| palabras_clave | texto  | solo HDBSCAN | Las 5 a 10 palabras más representativas del grupo, separadas por coma. Pueden ser los términos que hoy genera el pipeline NLP.                                                                                                                                                                                                                                       |
| encuesta       | texto  | sí           | Encuesta para la que aplica el aspecto. Qué aspectos tiene cada pregunta está en taxonomia.xlsx                                                                                                                                                                                                                                                                      |
| orden          | entero | sí           | Orden fijo en que los aspectos aparecen en los visuales: filas del mapa de calor, leyendas y listas. En Power BI se usa con «Ordenar por columna», para que no salgan en orden alfabético. Sale del orden de las filas de taxonomia.xlsx para cada pregunta; los temas emergentes van al final y SIN_ASPECTO de último                                               |

**metadatos_nlp**: una fila por cada pregunta procesada en cada corrida (llave: id_corrida_nlp + pregunta_codigo), porque cada encuesta, y cada pregunta, tiene su propio modelo y su propia lista de aspectos. Las columnas de corrida (id_corrida_nlp, id_corrida_preparacion, fecha_corrida, huella_alertas) se repiten en todas sus filas; las del modelo y los conteos son de esa pregunta. Sirve para auditar: si una cifra del tablero cambia entre dos periodos, aquí se ve si cambió la opinión o cambió el modelo.

| Columna                                                                           | De dónde sale                                                                                                | Hoy en la salida de obs19                                          | Descripción                                                                                                                                                     |
|-----------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| id_corrida_nlp                                                                    | Generada en cada corrida del NLP                                                                             | no existe                                                          | Identificador de la corrida (sección 3.3)                                                                                                                       |
| id_corrida_preparacion                                                            | Copiada de la corrida aprobada de la Etapa 1 (respuestas.csv)                                                | no existe                                                          | Corrida aprobada de la Etapa 1 que se procesó                                                                                                                   |
| fecha_corrida                                                                     | Generada: reloj al correr                                                                                    | no existe                                                          | Fecha y hora de la ejecución                                                                                                                                    |
| encuesta, pregunta_codigo                                                         | Copiadas de respuestas.csv (Etapa 1)                                                                         | no existen; el archivo es solo de obs19                            | Pregunta a la que corresponde la fila. Con id_corrida_nlp, forman la llave                                                                                      |
| version_modelo                                                                    | Etapa 2: identificador de los modelos entrenados                                                             | modelo_u4_id (solo el clasificador)                                | Versión del clasificador de aspectos y del modelo de sentimiento que se usaron en esa pregunta. Es la misma que lleva cada cláusula en clausulas.version_modelo |
| version_taxonomia                                                                 | Huella de taxonomia.xlsx (run_validar_taxonomia.py)                                                          | no existe                                                          | Qué versión de la lista de aspectos se usó                                                                                                                      |
| umbral_respaldo_hdbscan                                                           | Parámetro del NLP                                                                                            | no existe                                                          | Confianza del clasificador por debajo de la cual se usa HDBSCAN                                                                                                 |
| accuracy_aspecto                                                                  | Evaluación del NLP contra la muestra etiquetada por el cliente, sin las cláusulas vistas en el entrenamiento | no está en el archivo; el equipo informa 0,72                      | Exactitud del clasificador de esa pregunta en validación                                                                                                        |
| metricas_sentimiento                                                              | Evaluación del NLP                                                                                           | no existe                                                          | Métricas del modelo de sentimiento en validación                                                                                                                |
| regla_ajuste_sentimiento                                                          | Parámetro del NLP                                                                                            | neu_ajustado marca las cláusulas ajustadas, sin describir la regla | Qué regla ajusta la clase del modelo de sentimiento, si hay alguna                                                                                              |
| huella_alertas                                                                    | Huella de alertas.xlsx                                                                                       | no existe                                                          | Para saber qué versión de la lista se usó                                                                                                                       |
| n_respuestas, n_clausulas, n_alertas                                              | Contados sobre la salida de la corrida                                                                       | se pueden contar: 24.380, 25.714 y 86                              | Conteos de control de esa pregunta                                                                                                                              |
| pct_aspecto_clasificador, pct_aspecto_hdbscan, pct_aspecto_regla, pct_sin_aspecto | Contados sobre clausulas                                                                                     | la hoja «Resumen aspectos» trae conteos por aspecto                | Distribución del origen del aspecto                                                                                                                             |

### 5.4 Validaciones

La corrida del NLP solo se entrega si cumple todas estas reglas; lo ideal es que el propio pipeline las verifique al final y se detenga con un mensaje claro:

- Cada respuesta con comentario de la corrida aprobada tiene al menos una cláusula, y <span class="mark">ninguna no-respuesta tiene cláusulas</span>.

- clausula_id y alerta_id no se repiten; orden empieza en 1 y no tiene saltos.

- Todo respuesta_id de clausulas y alertas existe en respuestas, y todo aspecto_id existe en catalogo_aspectos.

- sentimiento solo toma POS, NEU o NEG; aspecto_origen solo los valores definidos; las confianzas están entre 0 y 1.

- metadatos_nlp registra el id_corrida_preparacion de la corrida que se procesó.

- Dos corridas con el mismo insumo y la misma versión producen los mismos archivos.

### 5.5 Pendiente en esta etapa

- Leer respuestas.csv de la Etapa 1 en lugar de data/processed/corpus_independientes/.

- <span class="mark">Fijar el umbral de respaldo de HDBSCAN y cómo se nombran sus grupos</span>.

- <span class="mark">Documentar el modelo de sentimiento y sus métricas.</span>

- <span class="mark">Agregar categorías a alertas.xlsx</span> y definir cómo se detectan las negaciones cercanas.

- Construir la lista de aspectos de calidad con el proceso acordado (ver Registro de decisiones).

- <span class="mark">Decidir con el equipo la regla de comentarios de menos de 5 palabras (aspecto OPINION_GENERAL).</span>

- Decidir si las preguntas de enfoque «motivo» (calidad) llevan sentimiento.

## 6. Etapa 3 · Limpieza y n-gramas

La Etapa 3 toma las cláusulas que entrega el NLP, limpia su texto y produce los n-gramas con los que se arman las nubes de palabras y los «top» de expresiones. Es la única fuente de n-gramas: el análisis exploratorio y Power BI usan la misma tabla.

- Responsable: encargado del EDA.

- Código: rediseño de src/datacleaning/, para que lea clausulas en lugar de las bases crudas.

### 6.1 Insumos

- clausulas.csv, solo las filas con es_opinion = True. Se usa texto_clausula.

- Listas de data/utilities/: stopwords.xlsx (hojas base, globales, particulares y adverbios, esta última con la columna valora) y el léxico de polaridad negaciones.xlsx del NLP (5.1), del que usa las filas con TIPO = NEGADOR o CUANTIFICADOR_BAJO. Las tildes no usan lista: las quita normalizar() con la normalización Unicode (NFKD), igual que la Etapa 1.

### 6.2 Reglas

1.  **Normalización:** minúsculas, sin tildes ni puntuación, espacios simples; la misma que usa la Etapa 1.

2.  **Las listas se comparan ya normalizadas.** Una palabra con tilde en la lista («más», «también») debe coincidir con el texto sin tilde.

3.  **Se conservan las palabras que invierten la polaridad**, aunque estén en la lista de stopwords o de adverbios: los negadores («no», «nunca», «sin», «falta»…) y los cuantificadores bajos («poco», «menos», «casi»…) del léxico negaciones.xlsx. Si no, «no explica» termina igual que «explica», y «poco claro» igual que «claro». Los intensificadores, adversativos y modales no se protegen, porque no invierten la polaridad.

4.  **Las stopwords particulares se aplican por encuesta,** con los códigos de encuesta de la sección 1. Incluyen los términos que solo repiten la pregunta («docente», «profesor»).

5.  **Las stopwords de varias palabras** («sin embargo», «por favor») se quitan como frase completa, antes de separar las palabras.

6.  **Para las nubes** se quitan además los adverbios que no valoran («ahora», «ahí», «después»). Los que valoran («bien», «mal», «mejor», «peor», «siempre», «nunca») se conservan.

7.  **Si una lista no se puede leer, la corrida se detiene.** Nunca se sigue con una lista vacía.

8.  **Identificadores.** Usa el clausula_id que entrega el NLP tal cual, como texto, y no crea identificadores nuevos: la llave de ngramas es clausula_id + n + ngrama.

### 6.3 Salidas

**clausulas_texto**: una fila por cláusula con opinión.

| Columna             | De dónde sale                                                                     | Hoy en src/datacleaning                               | Descripción                                                           |
|---------------------|-----------------------------------------------------------------------------------|-------------------------------------------------------|-----------------------------------------------------------------------|
| clausula_id         | Copiada de clausulas (Etapa 2)                                                    | no existe (trabaja sobre respuestas completas)        | Existe en clausulas                                                   |
| texto_normalizado   | Construida en la Etapa 3: texto_clausula con la regla 1                           | \<columna\>\_clean                                    | Texto de la cláusula normalizado (regla 1)                            |
| texto_sin_stopwords | Construida en la Etapa 3: texto_normalizado con las reglas 2 a 5 y stopwords.xlsx | \<columna\>\_no_stopwords (hoy pierde las negaciones) | Sin stopwords, conservando las negaciones; es la base de los n-gramas |
| texto_nubes         | Construida en la Etapa 3: texto_sin_stopwords con la regla 6 y la hoja adverbios  | \<columna\>\_no_stopwords_no_adverbs                  | Además sin adverbios que no valoran                                   |

**ngramas**: una fila por cada n-grama distinto (de 1, 2 y 3 palabras) de cada cláusula con opinión. Su llave es clausula_id + n + ngrama, que no se repite; no lleva identificador propio porque ninguna tabla se une a ella. Se entregan todos, no uno por cláusula; Power BI cuenta frecuencias y filtra por aspecto, sentimiento, docente o periodo a través de la cláusula.

| Columna        | Tipo    | Obligatoria | De dónde sale                                                                                                                                       | Descripción                                                                        |
|----------------|---------|-------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| clausula_id    | texto   | sí          | Copiada de clausulas (Etapa 2)                                                                                                                      | Existe en clausulas                                                                |
| n              | entero  | sí          | Construida en la Etapa 3                                                                                                                            | 1, 2 o 3                                                                           |
| ngrama         | texto   | sí          | Construida en la Etapa 3 a partir de texto_sin_stopwords                                                                                            | Forma normalizada, para contar                                                     |
| ngrama_mostrar | texto   | sí          | Construida en la Etapa 3: la forma del n-grama en texto_clausula (con tildes) más frecuente en el corpus                                            | Forma legible para mostrar                                                         |
| frecuencia     | entero  | sí          | Contada en la Etapa 3                                                                                                                               | Veces que aparece en la cláusula                                                   |
| es_nube        | sí / no | sí          | Construida en la Etapa 3: el n-grama también está en texto_nubes y no está formado solo por palabras protegidas (negadores o cuantificadores bajos) | True si el término va en las nubes de palabras (por ejemplo, un «no» suelto no va) |

**metadatos_limpieza**: una fila por corrida.

| Columna             | De dónde sale                                                    | Descripción                               |
|---------------------|------------------------------------------------------------------|-------------------------------------------|
| id_corrida_limpieza | Generada en cada corrida de la Etapa 3                           | Identificador de la corrida (sección 3.3) |
| id_corrida_nlp      | Copiada de metadatos_nlp (Etapa 2)                               | Corrida del NLP que se procesó            |
| fecha_corrida       | Generada: reloj al correr                                        | Fecha y hora de la ejecución              |
| version_limpieza    | Código de la Etapa 3                                             | Versión del código de limpieza            |
| huella_stopwords    | Calculada sobre stopwords.xlsx                                   | Qué versión de la lista se usó            |
| huella_negaciones   | Calculada sobre negaciones.xlsx, el léxico compartido con el NLP | Ídem                                      |
| n_clausulas         | Contada en la Etapa 3                                            | Cláusulas procesadas (es_opinion = True)  |
| n_ngramas           | Contada en la Etapa 3                                            | Filas de ngramas                          |
| n_ngramas_nube      | Contada en la Etapa 3                                            | Filas de ngramas con es_nube = True       |

Los nombres son una propuesta; los confirma el encargado del EDA.

### 6.4 Validaciones

- Todo clausula_id de clausulas_texto y de ngramas existe en clausulas.

- La combinación clausula_id + n + ngrama no se repite.

- n solo toma 1, 2 o 3, y frecuencia es al menos 1.

- Dos corridas con el mismo insumo y la misma versión producen los mismos archivos.

### 6.5 Pendiente en esta etapa

- Rediseñar src/datacleaning/ para que trabaje por cláusula y corregir lo que hoy falla: las stopwords particulares no se aplican (las busca con otro nombre de encuesta), las negaciones se pierden, las stopwords de varias palabras y las que tienen tilde no coinciden, y los errores al leer las listas no detienen la corrida.

- ~~Crear la lista de negaciones.~~ Resuelto el 16-nov: se usa el léxico de polaridad del NLP.

- Revisar la hoja adverbios para separar los que valoran de los que no.

- Decidir si se lematiza (por ejemplo, «explica» y «explicaba» como una sola forma).

## 7. Consumo

### 7.1 Power BI

**Relaciones entre las tablas del flujo**

<img src="media/image2.png" style="width:6in;height:4.54464in" alt="modelo relacional · 11 tablas, 2 llaves centrales" />

modelo relacional · 11 tablas, 2 llaves centrales

Las tablas de la Etapa 1 se unen a respuestas por respuesta_id, y las del NLP y la limpieza a clausulas por clausula_id. El modelo del cliente (y su seguridad por filas) se conecta por las tablas de contexto y de copias. La tabla de abajo da el mismo detalle por escrito.

| Tabla                                    | Se une con         | Por             | Relación     |
|------------------------------------------|--------------------|-----------------|--------------|
| contexto\_\<encuesta\>                   | respuestas         | respuesta_id    | una a una    |
| respuestas_oferta, respuestas_programa   | respuestas         | respuesta_id    | varias a una |
| respuestas                               | catalogo_preguntas | pregunta_codigo | varias a una |
| clausulas, alertas                       | respuestas         | respuesta_id    | varias a una |
| clausulas                                | catalogo_aspectos  | aspecto_id      | varias a una |
| ngramas                                  | clausulas          | clausula_id     | varias a una |
| auditoria_clasificacion, clausulas_texto | clausulas          | clausula_id     | una a una    |

**Unión con el modelo del cliente.** Las tablas de contexto y de copias traen las llaves con que el cliente identifica sus datos: id_docente, codigo_del_grupo, codigo_materia y codigo_cruce_autoevaluacion en la evaluación docente; facultad_de_oferta en respuestas_oferta; llave_oac y programa en respuestas_programa, y poblacion y facultad_departamento_area en el contexto de calidad.

**Reglas para el tablero**

- **Seguridad por filas:** las tablas nuevas deben quedar relacionadas con la tabla donde el cliente tiene su regla de seguridad (por programa o facultad). Si no, decanos y directores verían comentarios que no les corresponden.

- **Contar sin duplicar:** como una respuesta puede tener varias filas en respuestas_oferta o respuestas_programa, los conteos de respuestas se hacen con valores distintos de respuesta_id.

- **% de personas que comentan** = respuestas con es_no_respuesta = False ÷ todas las respuestas, en evaluación docente y acreditación. En calidad_servicios se muestra «% de respuestas registradas con contenido», que no se compara con las otras encuestas, y los indicadores se calculan como proporciones dentro de los comentarios, nunca como conteos absolutos entre poblaciones.

- **Sentimiento:** una tabla de etiquetas traduce POS, NEU y NEG a Positivo, Neutral y Negativo.

- **Temas emergentes:** los grupos de HDBSCAN se muestran aparte de los aspectos definidos, porque cambian en cada corrida.

### 7.2 Análisis exploratorio

Usa clausulas, clausulas_texto y ngramas, con las mismas definiciones que Power BI. Hoy no entrega tablas al tablero.

<img src="media/image3.png" style="width:6in;height:3.05357in" alt="modelo relacional del análisis exploratorio · 6 tablas" />

modelo relacional del análisis exploratorio · 6 tablas

El análisis parte de clausulas. Por clausula_id llega a sus textos limpios y a sus n-gramas, y por respuesta_id a la pregunta, el periodo y el rol. Así, un n-grama se puede filtrar por aspecto, sentimiento, pregunta o periodo sin cruzar con otra fuente.

## 8. Validaciones entre etapas

Cada etapa valida su propia salida (4.4, 5.4 y 6.4). Lo que ninguna puede revisar sola es cómo encajan entre ellas. Para eso hay un paso aparte, que se corre después de las tres etapas y antes de cargar Power BI.

**Dónde.** run_validar_flujo.py, con el módulo src/validacion/. No pertenece a ninguna etapa.

**Quién.** Quien publica en Power BI. Si algo falla, no se publica.

**Qué lee.** La última corrida aprobada de la Etapa 1 (data/processed/etapa1/\<id_corrida_preparacion\>/), la salida del NLP (data/processed/nlp/) y la de la Etapa 3 (data/processed/limpieza/), con sus metadatos.

**Qué comprueba:**

1.  **Cadena de corridas.** metadatos_nlp.id_corrida_preparacion es la última corrida aprobada (aprobaciones.csv), y metadatos_limpieza.id_corrida_nlp es la corrida del NLP que se va a publicar.

2.  **Integridad.** Todo respuesta_id de clausulas y alertas existe en respuestas. Todo clausula_id de auditoria_clasificacion, alertas, clausulas_texto y ngramas existe en clausulas. Todo aspecto_id de clausulas y de la auditoría existe en catalogo_aspectos.

3.  **Conteos.** Los totales que declara cada metadatos\_\* coinciden con las filas de sus archivos. Las respuestas con comentario de la Etapa 1 son las que el NLP dice haber procesado.

**Qué entrega.** validacion_flujo.csv, con una fila por comprobación: chequeo, resultado (ok o falla), filas_con_falla y detalle. Termina con error si hay al menos una falla.

**Lo que no hace:**

- La comparación con la base cruda le toca a la Etapa 1, porque es la única que la lee. En cada periodo, las respuestas con es_no_respuesta = False deben coincidir con las celdas con texto de la base, deduplicadas con las mismas llaves y sin la no-respuesta. Las columnas Observaciones eval y Observaciones Auto del cliente no sirven para validar, porque cuentan «Ninguna» y «Nada» como comentario.

- La estabilidad (que el mismo insumo, con las mismas versiones, produzca los mismos archivos) se prueba solo cuando cambia la versión de una etapa: se corre dos veces y se comparan las huellas.

El detalle para implementarlo está en la especificación de las validaciones entre etapas.

## 9. Registro de decisiones

Ante una cifra que no cuadre con los datos del cliente, la primera referencia es el informe de inconsistencias que se le entrega con cada corrida.

| Tema                                           | Decisión                                                                                                                                                                                                                                                                                                                          | Por qué                                                                                                                                                                                                | Fecha  |
|------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|
| Alcance                                        | Solo Pregrado en la evaluación docente                                                                                                                                                                                                                                                                                            | Posgrado está en otra base                                                                                                                                                                             | 13-nov |
| Llave de la evaluación                         | Periodo + ID de respuesta + Id Docente                                                                                                                                                                                                                                                                                            | El Id Estudiante cambia entre copias de una misma respuesta y tiene un valor de relleno                                                                                                                | 13-nov |
| Llave de la autoevaluación                     | Periodo + Código Cruce Autoevaluación                                                                                                                                                                                                                                                                                             | Se repite por cada estudiante del grupo; el Código Conteo cambia de receta entre periodos                                                                                                              | 13-nov |
| Elección de copia                              | Con comentario, luego la más larga, luego la Facultad de oferta primera alfabéticamente                                                                                                                                                                                                                                           | Que dos corridas elijan siempre la misma copia                                                                                                                                                         | 13-nov |
| No-respuesta                                   | Lista configurable, comparada sobre el texto original normalizado                                                                                                                                                                                                                                                                 | Decidirla después de quitar stopwords hacía perder comentarios reales («bien»)                                                                                                                         | 13-nov |
| No-respuestas en la salida                     | Se conservan, marcadas                                                                                                                                                                                                                                                                                                            | Son el denominador del % de personas que comentan                                                                                                                                                      | 13-nov |
| Definición de «comentó»                        | «Ninguna» y «Nada» son no-respuesta, aunque el cliente los cuente como comentario                                                                                                                                                                                                                                                 | Coherencia con la lista; se informa al cliente                                                                                                                                                         | 13-nov |
| Núcleo + contexto                              | Lo común a todas las encuestas en respuestas; lo propio, en contexto\_\<encuesta\>                                                                                                                                                                                                                                                | Sin columnas vacías por encuesta                                                                                                                                                                       | 14-nov |
| Rol                                            | Etiquetas del cliente: Estudiantes, Docentes, Administrativos, Directivos y coordinadores, Graduados                                                                                                                                                                                                                              | Un solo vocabulario para todas las encuestas                                                                                                                                                           | 14-nov |
| Respondientes efectivos                        | Solo filas con Evaluación / Autoevaluación = Sí                                                                                                                                                                                                                                                                                   | Los registros sin llenar inflaban el denominador                                                                                                                                                       | 14-nov |
| Marcas de inconsistencia                       | Se marcan, no se corrigen; se llaman «inconsistencia»                                                                                                                                                                                                                                                                             | Trazabilidad; evitar confusión con la encuesta de calidad                                                                                                                                              | 14-nov |
| Acreditación                                   | Una respuesta por envío (Periodo + ID de respuesta); programas en respuestas_programa; llave se publica como llave_oac                                                                                                                                                                                                            | El envío se repite por programa; se asume que es el mismo envío                                                                                                                                        | 14-nov |
| Valores cerrados en pregunta abierta           | «De acuerdo», «N/S»… como texto completo son no-respuesta y se marcan                                                                                                                                                                                                                                                             | Error de consolidación del cliente en acreditación                                                                                                                                                     | 14-nov |
| Código Cruce                                   | Se usa tal como viene; no se codifica                                                                                                                                                                                                                                                                                             | La protección se hace anonimizando la base antes, si sale del cliente                                                                                                                                  | 14-nov |
| Formato                                        | CSV en UTF-8 con BOM                                                                                                                                                                                                                                                                                                              | Es lo que produce la Etapa 1 y el cliente lo abre sin herramientas extra                                                                                                                               | 14-nov |
| Sentimiento                                    | POS, NEU, NEG                                                                                                                                                                                                                                                                                                                     | Son los códigos que ya produce el NLP                                                                                                                                                                  | 14-nov |
| Punto de control                               | El NLP procesa solo una corrida de la Etapa 1 aprobada por el cliente                                                                                                                                                                                                                                                             | El cliente puede corregir su base y correr la Etapa 1 varias veces                                                                                                                                     | 14-nov |
| Limpieza                                       | Va después del NLP, por cláusula; ngramas es la única fuente para EDA y Power BI                                                                                                                                                                                                                                                  | Los n-gramas se calculan por cláusula; evitar dos cálculos distintos                                                                                                                                   | 14-nov |
| Nubes de palabras                              | Se arman desde ngramas (es_nube)                                                                                                                                                                                                                                                                                                  | Una sola fuente                                                                                                                                                                                        | 14-nov |
| Calidad                                        | Entra al flujo como calidad_servicios; sus preguntas se agrupan en 10 códigos                                                                                                                                                                                                                                                     | Es la encuesta de servicios de la Universidad                                                                                                                                                          | 14-nov |
| Indicadores de calidad                         | Proporciones dentro de los comentarios, sin tasas sobre la población                                                                                                                                                                                                                                                              | El cliente no entrega el total de encuestados                                                                                                                                                          | 14-nov |
| Análisis exploratorio                          | No alimenta el tablero                                                                                                                                                                                                                                                                                                            | Su uso es el informe del equipo                                                                                                                                                                        | 14-nov |
| Roles de calidad                               | Las 8 poblaciones del cliente quedan como roles propios                                                                                                                                                                                                                                                                           | Decisión de Cristina; la población se conserva tal como la registra el cliente                                                                                                                         | 14-nov |
| Códigos de calidad                             | Prefijo cal\_; las sugerencias que mencionan «el programa» (2021–2022) tienen su propio código, separado de las generales (2023 y 2025)                                                                                                                                                                                           | Una redacción distinta puede llevar a contestar sobre temas distintos                                                                                                                                  | 14-nov |
| No-respuesta ampliada                          | Se agregan «Sin comentarios», «Sin observaciones», «No gracias» y «Gracias»                                                                                                                                                                                                                                                       | No expresan una opinión; aparecen en todas las encuestas                                                                                                                                               | 14-nov |
| Estudiante vs docente                          | Comparación general, pregunta contra pregunta, cada lado con su propia lista de aspectos; sin tabla de equivalencias. A nivel docente, su lado se muestra como lista de cláusulas, sin porcentajes                                                                                                                                | Las listas son distintas (17/18 frente a 31); las equivalencias complicarían el proyecto                                                                                                               | 14-nov |
| Aspectos de calidad                            | Tendrá lista propia con el mismo proceso: primera pasada sin lista, muestra que cubra todos los códigos y años, corrección del cliente, regresión logística y acierto por aspecto. Mientras tanto, solo temas emergentes                                                                                                          | Es el camino que ya funcionó en las demás encuestas                                                                                                                                                    | 14-nov |
| Corridas del NLP                               | Solo se clasifica lo nuevo; reprocesar el histórico es una decisión explícita con versión nueva                                                                                                                                                                                                                                   | Lo ya publicado en el tablero no cambia sin aviso                                                                                                                                                      | 14-nov |
| Auditoría de la clasificación                  | Probabilidades de sentimiento y 3 aspectos más probables en auditoria_clasificacion, cargada en Power BI; clausulas solo con la clase ganadora                                                                                                                                                                                    | clausulas lo más limpia posible y el cliente con la auditoría en un solo lugar                                                                                                                         | 14-nov |
| Aspectos en la auditoría                       | Se guardan los 3 aspectos más probables con su confianza, no todos. El NLP los obtiene ordenando sus probabilidades por clase (hoy p_u4\_\*, que no guarda el top 3). Quedan vacíos en las cláusulas que no pasan por el clasificador                                                                                             | Cada pregunta tiene un número distinto de aspectos, así que una columna por aspecto no cabe en una tabla común; más allá del tercero la probabilidad ya es baja (medianas en obs19: 0,39, 0,13 y 0,08) | 15-nov |
| Normalización del texto, la «ñ» y accents.xlsx | Se mantiene normalizar() tal como está, y lo usan la Etapa 1 y la Etapa 3. Quita las tildes con la normalización Unicode (NFKD), sin una lista fija, y por eso la «ñ» también queda en «n» («año» → «ano»). texto_original, los identificadores y ngrama_mostrar conservan la «ñ». accents.xlsx ya no se usa en las etapas nuevas | La normalización solo se usa para comparar y contar, y las confusiones del tipo «año»/«ano» son raras en estos comentarios. Cambiarla obliga a correr de nuevo la Etapa 1 y la Etapa 3                 | 16-nov |
| Léxico de negaciones compartido                | Un solo negaciones.xlsx, el del NLP, que es su dueño. La Etapa 2 lo usa entero; la Etapa 3 protege de la limpieza los tipos NEGADOR y CUANTIFICADOR_BAJO (23 términos). Reemplaza la lista de 10 palabras que había creado la Etapa 3                                                                                             | Los dos archivos tenían el mismo nombre y columnas distintas, así que uno rompía al otro. El del NLP es más completo, y además mostró que «poco claro» también se volvía «claro» en la limpieza        | 16-nov |
| Taxonomía                                      | taxonomia.xlsx es insumo del NLP, que es su dueño y la valida contra su modelo; la Etapa 1 no la usa                                                                                                                                                                                                                              | Es el conjunto de etiquetas del modelo; si la Etapa 1 dependiera de ella, un cambio en la lista obligaría a repetirla sin que cambien los datos                                                        | 15-nov |

## 10. Temas por resolver

**Equipo**

- Si las preguntas de enfoque «motivo» (calidad) llevan sentimiento.

- Si la Etapa 3 lematiza.

- Convención de idioma para el código (funciones, clases y archivos).

- Propuesta para controlar SIN_ASPECTO sin prohibirlo (forzar un aspecto en todo crea asignaciones falsas):

  1.  SIN_ASPECTO lleva un motivo: sin_opinion o no_asignable.

  2.  metadatos_nlp informa el porcentaje por corrida y por pregunta.

  3.  Tope acordado con el encargado de NLP (por ejemplo, 10 %); si se supera, la corrida no se aprueba sin revisión.

  4.  En Power BI aparece como categoría visible, para que los totales cuadren.

- **Alertas (validar con el equipo):**

  - alertas.xlsx no está: no aparece en la carpeta del repositorio ni en su historial de git. Hay que averiguar qué pasó y recuperar la lista con el encargado de NLP. Sin ella no se puede validar termino ni categoria_alerta.

  - Decidido el 15-nov: la alerta aplica al comentario completo, que no se fragmenta (regla de alertas, 5.2). Queda abierto si una alerta negada desvía el comentario: 57 de las 86 de obs19 tienen un negador.

- Los temas de modelado del NLP (umbral para pasar a HDBSCAN, regla de comentarios cortos, medición del acierto, ajuste de sentimiento) se discuten en el informe de revisión del NLP; aquí solo entra lo que cambie las tablas.

**NLP**

- Umbral de confianza para usar HDBSCAN, cómo se nombran sus grupos y qué porcentaje de cláusulas termina allí.

- Modelo de sentimiento, sus métricas de validación y si se aplica el calibrador.

- Alertas: validar cómo quedó construido el pipeline y acordar la regla de la sección 5.2 (grano, negación y categorías de alertas.xlsx). En obs19 la alerta va por cláusula (86), reemplaza el aspecto y el sentimiento, y 57 de esas 86 cláusulas tienen un negador: posibles falsos positivos.

- Lista de aspectos de calidad: primera pasada y muestra para que el cliente la corrija.

- Qué necesita el cliente para correr el pipeline (Python, GPU, memoria) y cómo se le entrega el modelo.

**Checklist antes de dar por cerrada la taxonomía:**

- [ ] El encargado de NLP entrega su taxonomia.xlsx final. La que hoy está en data/utilities/ es un borrador de Cristina, no la versión oficial.

- [ ] La versión del NLP pasa python run_validar_taxonomia.py sin errores.

- [ ] Coincide con las clases de cada modelo (comparar_con_clases): obs19 ya coincide; faltan obs20, obs23 a obs25 y acreditación.

- [ ] Se reemplaza el borrador en data/utilities/ y su version_taxonomia queda en metadatos_nlp.

**Checklist del código temporal por retirar:**

- [x] Conciliación con el insumo NLP v0: retirada el 15-nov (commit 3af1e52); el código queda en la etiqueta conciliacion-insumo-nlp-v0.

- [ ] src/limpieza/adaptador_nlp_actual.py y la opción --adaptador de run_limpieza.py: se borran cuando el NLP entregue su clausulas.csv.

- [ ] src/datacleaning/, sus pruebas, src/settings.py y data/utilities/accents.xlsx (que solo usa ese código): se borran cuando el NLP lea de la Etapa 1 (tarea 1 de su especificación) y el equipo apruebe la Etapa 3. Antes de borrar, poner una etiqueta como con la conciliación.

**Cliente**

- Identificador del docente: en la base que recibimos el Id Docente es un consecutivo, pero en su tablero es el número de documento. ¿Con cuál une nuestras tablas?

- ¿La seguridad por filas de su tablero filtra por docente o por programa, y con qué llave?

- ¿El cambio en la receta del Código Cruce Autoevaluación entre 2025-1 y 2025-2 fue intencional?

- ¿Para qué usa el Código Conteo?

- ¿Qué valor es el Id Estudiante que se repite en 291 filas, y por qué cambia entre las copias de una misma respuesta?

- Calidad: ¿la base incluye a todos los encuestados o solo a quienes escribieron algo? Ya indicó que tiene el total por año y población, pero no lo consideró necesario.

## Anexo A. Ejemplo trazado

La pestaña **Ejemplo trazado** sigue unas pocas filas de la base cruda a través de todas las etapas, hasta Power BI.

Ir a la pestaña: Ejemplo trazado
