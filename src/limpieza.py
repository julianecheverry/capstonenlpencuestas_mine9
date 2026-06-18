# Módulo para Limpieza de columna clave de texto

# Librerías python estándar
from pathlib import Path
import re

class dataloader:
    def __init__(self, ruta_archivo : str | Path , columna_clave : str, separator : str = ','):
        self.ruta_archivo = ruta_archivo
        self.separator = separator
        self.columna_clave = columna_clave

    def cargar_datos(self):
        """
        Carga y valida el archivo de datos para el procesamiento de texto.

        Este método intenta leer un archivo CSV en la ruta especificada, 
        verificando previamente la existencia del archivo, la presencia de la 
        columna clave y la validez del tipo de dato de dicha columna.

        Returns:
            pd.DataFrame: Un DataFrame de pandas con los datos cargados si la 
                operación fue exitosa.

        Raises:
            ImportError: Si la librería 'pandas' no está instalada.
            FileNotFoundError: Si el archivo no existe en la ruta proporcionada.
            ValueError: Si la columna clave no existe, no es de tipo texto, 
                o si ocurre cualquier error inesperado durante la lectura.
        """        
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("No se pudo importar pandas. Verificar que esté instalado en el ambiente de trabajo.") from exc
            # return None

        try:
            if not Path(self.ruta_archivo).exists():
                raise FileNotFoundError(f"Dataset no encontrado en la ruta: {self.ruta_archivo}")
            # Cargar el archivo CSV utilizando pandas
            datos = pd.read_csv(self.ruta_archivo, sep=self.separator)
            # Validaciones básicas para asegurar que la columna clave existe y es de tipo texto
            if self.columna_clave not in datos.columns:
                raise ValueError(f"La columna '{self.columna_clave}' no se encuentra en el archivo.")
            if datos[self.columna_clave].dtype != 'object':
                raise ValueError(f"La columna '{self.columna_clave}' no es de tipo texto.")
            
            return datos
        
        # Manejo de excepciones para errores comunes
        except Exception as e:
            raise ValueError(f"Ocurrió un error: {e}")


class cleaner(dataloader):

    # def __init__(self, ruta_archivo : str, columna_clave : str, separator : str = ','):
    #     super().__init__(ruta_archivo, columna_clave, separator)

    def limpiar_texto(self, texto):
        """
        Limpia una cadena de texto eliminando ruido y normalizando caracteres.

        La función realiza las siguientes operaciones en orden:
        1. Elimina espacios en blanco al inicio y al final.
        2. Convierte el texto a minúsculas.
        3. Normaliza caracteres acentuados (tildes) a sus contrapartes simples.
        4. Elimina caracteres especiales, manteniendo solo alfanuméricos, 
           espacios y la letra 'ñ'.

        Args:
            texto (str): La cadena de texto cruda a procesar.

        Returns:
            str: El texto procesado y normalizado.
        """       
        # Eliminar espacios en blanco al inicio y al final
        texto_limpio = texto.strip()

        # Convertir a minúsculas
        texto_limpio = texto_limpio.lower()

        # Eliminar letras tildadas y caracteres acentuados
        acentos = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
                  'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U'}
        for acento, letra in acentos.items():
            texto_limpio = texto_limpio.replace(acento, letra)

        # Eliminar otros caracteres especiales utilizando una expresión regular
        texto_limpio = re.sub(r'[^a-zA-Z0-9\sñÑ]', '', texto_limpio)

        return texto_limpio
    
    def limpiar_columna_clave(self):
        """
        Realiza la limpieza del texto en la columna clave del DataFrame cargado.

        Este método delega la carga de datos a 'cargar_datos', aplica una transformación
        de limpieza (definida en 'limpiar_texto') sobre la columna especificada en 
        'self.columna_clave' y genera una nueva columna resultante en el DataFrame.

        Returns:
            pd.DataFrame: Un nuevo DataFrame que incluye la columna adicional con 
                el texto procesado ('{self.columna_clave}_limpia').

        Raises:
            ValueError: Si el método 'cargar_datos' retorna None, indicando que no 
                fue posible obtener la fuente de datos original.
        """        
        datos = self.cargar_datos()
        if datos is not None:
            # Aplicar la función de limpieza a cada valor de la columna clave sin sobreescribir el original
            datos[f"{self.columna_clave}_limpia"] = datos[self.columna_clave].apply(self.limpiar_texto)
            return datos
        else:
            raise ValueError("No se pudieron cargar los datos para limpiar.")
        
    def eliminar_stopwords(self):
        """
        Elimina las palabras vacías (stopwords) de una columna específica del DataFrame.

        La función utiliza el corpus de NLTK para español, extendido con términos
        específicos del dominio académico. Filtra las palabras de la columna 
        definida en 'self.columna_clave' y genera una nueva columna con el texto limpio.

        Args:
            texto (str, opcional): Parámetro reservado para compatibilidad, 
                actualmente no se utiliza dentro de la lógica del método.

        Raises:
            ImportError: Si la librería 'nltk' no está instalada en el entorno.

        Returns:
            el DataFrame 'self.datos_limpios' con la nueva columna '{self.columna_clave}_sin_stopwords'.
        """        

        try:
            from nltk.corpus import stopwords
        except ImportError as exc:
            raise ImportError("No se pudo importar -nltk-. Verificar que esté instalado en el ambiente de trabajo.") from exc


        stop_words = set(stopwords.words("spanish"))
        stop_words = set(list(stop_words)+ ['académico', 'academia', 'universidad'])

            # 1. Definir la lógica de limpieza en una función pequeña
        def limpiar_texto(texto):
            words = nltk.word_tokenize(texto)
            filtradas = [w for w in words if w.lower() not in self.stop_words]
            return ' '.join(filtradas)

        # 2. Aplicar la función a toda la columna de una vez (vectorizado)
        columna_objetivo = f"{self.columna_clave}_limpia"
        self.datos_limpios[f"{self.columna_clave}_sin_stopwords"] = self.datos_limpios[columna_objetivo].apply(limpiar_texto)

        return self.datos_limpios


    def guardar_datos_limpios(self, ruta_salida : str):
        """
        Ejecuta el proceso de limpieza y exporta el resultado a un archivo CSV.

        Este método invoca internamente el flujo de limpieza de datos, y si el 
        proceso es exitoso, guarda el DataFrame resultante en la ruta especificada. 
        También actualiza el estado interno del objeto con los datos procesados.

        Args:
            ruta_salida (str): La ruta del sistema de archivos (incluyendo nombre 
                del archivo y extensión .csv) donde se guardarán los datos.

        Raises:
            ValueError: Si el proceso de limpieza falla o si ocurre un error 
                durante la escritura del archivo en el disco (ej. permisos, ruta inválida).

        Returns:
            None: La función no retorna un valor, pero imprime un mensaje de 
                confirmación en consola al finalizar exitosamente.
        """
        datos_limpios = self.limpiar_columna_clave()
        if datos_limpios is not None:
            try:
                datos_limpios.to_csv(ruta_salida, index=False)
                self.datos_limpios = datos_limpios
                print(f"Datos limpios guardados en '{ruta_salida}'.")
            except Exception as e:
                raise ValueError(f"Ocurrió un error al guardar el archivo: {e}")
        else:
            raise ValueError("No se pudieron limpiar los datos para guardar.")