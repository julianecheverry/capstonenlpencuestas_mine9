# Módulo para Limpieza de columna clave de texto

# from importlib.resources import path
from pathlib import Path
import re

class dataloader:
    def __init__(self, ruta_archivo : str | Path , columna_clave : str, separator : str = ','):
        self.ruta_archivo = ruta_archivo
        self.separator = separator
        self.columna_clave = columna_clave

    def cargar_datos(self):
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
        # Eliminar espacios en blanco al inicio y al final
        texto_limpio = texto.strip()

        # Convertir a minúsculas
        texto_limpio = texto_limpio.lower()

        # Eliminar caracteres especiales (puedes ajustar esto según tus necesidades)
        # caracteres_especiales = ['.', ',', '!', '?', ';', ':', '-', '_', '(', ')', '[', ']', '{', '}', '"', "'"]
        # for char in caracteres_especiales:
        #     texto_limpio = texto_limpio.replace(char, '')

        # Eliminar letras tildadas y caracteres acentuados
        acentos = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
                  'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U'}
        for acento, letra in acentos.items():
            texto_limpio = texto_limpio.replace(acento, letra)

        # Eliminar otros caracteres especiales utilizando una expresión regular
        texto_limpio = re.sub(r'[^a-zA-Z0-9\sñÑ]', '', texto_limpio)

        return texto_limpio
    
    def limpiar_columna_clave(self):
        datos = self.cargar_datos()
        if datos is not None:
            # Aplicar la función de limpieza a cada valor de la columna clave sin sobreescribir el original
            datos[f"{self.columna_clave}_limpia"] = datos[self.columna_clave].apply(self.limpiar_texto)
            return datos
        else:
            raise ValueError("No se pudieron cargar los datos para limpiar.")
            # return None
        
    def guardar_datos_limpios(self, ruta_salida : str):
        datos_limpios = self.limpiar_columna_clave()
        if datos_limpios is not None:
            try:
                datos_limpios.to_csv(ruta_salida, index=False)
                print(f"Datos limpios guardados en '{ruta_salida}'.")
            except Exception as e:
                raise ValueError(f"Ocurrió un error al guardar el archivo: {e}")
        else:
            raise ValueError("No se pudieron limpiar los datos para guardar.")
        

    def eliminar_stopwords(self, texto):

        try:
            from nltk.corpus import stopwords
        except ImportError as exc:
            raise ImportError("No se pudo importar -nltk-. Verificar que esté instalado en el ambiente de trabajo.") from exc


        stop_words = set(stopwords.words("spanish"))
        stop_words = set(list(stop_words)+ ['académico', 'academia', 'universidad'])

        for op in range(n):
            without_stop_words = []
            stopword = []
            sentence = AllReviews2[op]
            words = nltk.word_tokenize(sentence)
            for word in words:
                if word in stop_words:
                    stopword.append(word)
                else:
                    without_stop_words.append(word)
            AllReviews3[op]= ' '.join(without_stop_words)

        # # Lista de stopwords en español
        # stopwords = set([
        #     'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se',
        #     'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al',
        #     'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este',
        #     # Agrega más stopwords según sea necesario
        # ])
        # palabras = texto.split()
        # palabras_filtradas = [palabra for palabra in palabras if palabra not in stopwords]
        # return ' '.join(palabras_filtradas)