# Module for cleaning key text column

# Standard Python libraries
from pathlib import Path
import re

class dataloader:
    SUPPORTED_FORMATS = {'.csv', '.xlsx', '.xls'}
    def __init__(self, file_path : str | Path , key_column : str, separator : str = ',', sheet_name: str | int = 0):
        self.file_path = file_path
        self.separator = separator
        self.key_column = key_column
        self.sheet_name = sheet_name

    def load_data(self):
        """
        Carga y valida el archivo de data para el procesamiento de texto.

        Este método intenta leer un archivo CSV (.csv) o Excel (.xlsx, .xls) en la 
        ruta especificada, verificando previamente la existencia del archivo, el 
        formato soportado, la presencia de la columna clave y la validez del tipo 
        de dato de dicha columna.

        Returns:
            pd.DataFrame: Un DataFrame de pandas con los data cargados si la 
                operación fue exitosa.

        Raises:
            ImportError: Si la librería 'pandas' no está instalada, o si se intenta 
                leer un archivo Excel sin tener 'openpyxl' instalado.
            FileNotFoundError: Si el archivo no existe en la ruta proporcionada.
            ValueError: Si el formato del archivo no está soportado, si la columna 
                clave no existe, no es de tipo texto, o si ocurre cualquier error 
                inesperado durante la lectura.
        """      
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Pandas could not be imported. Please verify that it is installed in your working environment.") from exc
            # return None

        try:
            if not Path(self.file_path).exists():
                raise FileNotFoundError(f"Dataset not found in the file path: {self.file_path}")
            # Validar extensión
            extension = self.file_path.suffix.lower()
            if extension not in self.SUPPORTED_FORMATS:
                raise ValueError(
                f"Format '{extension}' is not supported. Supported formats: {self.SUPPORTED_FORMATS}"
            )

            try:
                if extension == '.csv':
                    data = pd.read_csv(self.file_path, sep=self.separator)

                elif extension in {'.xlsx', '.xls'}:
                    try:
                        import openpyxl  # noqa: F401
                    except ImportError as exc:
                        raise ImportError(
                            "Excel files require 'openpyxl' to be installed. "
                            "Install it with: pip install openpyxl"
                        ) from exc
                    data = pd.read_excel(self.file_path, sheet_name=self.sheet_name)

            except (pd.errors.ParserError, Exception) as exc:
                raise ValueError(f"Error reading the file: {exc}") from exc

            # Validations to ensure the key column exists and is of text type
            if self.key_column not in data.columns:
                raise ValueError(f"Column '{self.key_column}' not found in the file.")
            if data[self.key_column].dtype != 'object':
                raise ValueError(f"Column '{self.key_column}' is not of text type.")
            
            return data
        
        # Manejo de excepciones para errores comunes
        except Exception as e:
            raise ValueError(f"An error occurred: {e}")


class cleaner(dataloader):

    # def __init__(self, ruta_archivo : str, columna_clave : str, separator : str = ','):
    #     super().__init__(ruta_archivo, columna_clave, separator)

    # @staticmethod -> Para clases desacopladas, pero aquí queremos usar self.stop_words que es un atributo de instancia, 
    # así que no es adecuado usar @staticmethod --< CONFIRMAR SI SE DEBE USAR O NO >--
    def clean_text(self, text: str) -> str:
        """
        Limpia una cadena de texto eliminando ruido y normalizando caracteres.

        La función realiza las siguientes operaciones en orden:
        1. Elimina espacios en blanco al inicio y al final.
        2. Convierte el texto a minúsculas.
        3. Normaliza caracteres acentuados (tildes) a sus contrapartes simples.
        4. Elimina caracteres especiales, manteniendo solo alfanuméricos, 
           espacios y la letra 'ñ'.

        Args:
            text (str): La cadena de texto cruda a procesar.

        Returns:
            str: El texto procesado y normalizado.
        """       
        if not isinstance(text, str):
            return ''  # Retorna una cadena vacía si el valor no es texto para evitar que pandas represente como 'float'
        
        # Eliminar espacios en blanco al inicio y al final
        cleaned_text = text.strip()

        # Convertir a minúsculas
        cleaned_text = cleaned_text.lower()

        # Eliminar letras tildadas y caracteres acentuados
        accents = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
                  'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U'}
        for accent, letter in accents.items():
            cleaned_text = cleaned_text.replace(accent, letter)

        # Eliminar otros caracteres especiales utilizando una expresión regular
        cleaned_text = re.sub(r'[^a-zA-Z0-9\sñÑ]', '', cleaned_text)

        return cleaned_text
    
    def clean_key_column(self):
        """
        Realizes the cleaning of the text in the key column of the loaded DataFrame.

        This method delegates the data loading to 'load_data', applies a cleaning
        transformation (defined in 'clean_text') to the specified column in        
        'self.key_column' and generates a new resulting column in the DataFrame.
        
        Returns:
            pd.DataFrame: A new DataFrame that includes the additional column with 
                the processed text ('{self.key_column}_limpia').

        Raises:
            ValueError: If the method 'load_data' returns None, indicating that it 
            was not possible to obtain the original data source.
        """        
        data = self.load_data()
        if data is not None:
            # Aplicar la función de limpieza a cada valor de la columna clave sin sobreescribir el original
            data[f"{self.key_column}_limpia"] = data[self.key_column].apply(self.clean_text)
            return data
        else:
            raise ValueError("No se pudieron cargar los data para limpiar.")
        
    def eliminate_stopwords(self):
        """
        Elimina las palabras vacías (stopwords) de una columna específica del DataFrame.

        La función utiliza el corpus de NLTK para español, extendido con términos
        específicos del dominio académico. Filtra las palabras de la columna 
        definida en 'self.key_column' y genera una nueva columna con el texto limpio.

        Args:
            texto (str, opcional): Parámetro reservado para compatibilidad, 
                actualmente no se utiliza dentro de la lógica del método.

        Raises:
            ImportError: Si la librería 'nltk' no está instalada en el entorno.

        Returns:
            pd.DataFrame: The DataFrame 'self.cleaned_data' with the new column '{self.key_column}_sin_stopwords'.
        """        

        try:
            from nltk.corpus import stopwords
        except ImportError as exc:
            raise ImportError("Could not import -nltk-. Verify that it is installed in the working environment.") from exc


        stop_words = set(stopwords.words("spanish"))
        stop_words = set(list(stop_words)+ ['académico', 'academia', 'universidad'])

            # 1. Definir la lógica de limpieza en una función pequeña
        def clean_stopwords(text):
            words = nltk.word_tokenize(text)
            filtradas = [w for w in words if w.lower() not in self.stop_words]
            return ' '.join(filtradas)

        # 2. Aplicar la función a toda la columna de una vez (vectorizado)
        columna_objetivo = f"{self.key_column}_limpia"
        self.cleaned_data[f"{self.key_column}_sin_stopwords"] = self.cleaned_data[columna_objetivo].apply(clean_stopwords)

        return self.cleaned_data


    def save_cleaned_data(self, out_path : str):
        """
        Ejecuta el proceso de limpieza y exporta el resultado a un archivo CSV.

        Este método invoca internamente el flujo de limpieza de data, y si el 
        proceso es exitoso, guarda el DataFrame resultante en la ruta especificada. 
        También actualiza el estado interno del objeto con los data procesados.

        Args:
            out_path (str): La ruta del sistema de archivos (incluyendo nombre 
                del archivo y extensión .csv) donde se guardarán los data.

        Raises:
            ValueError: Si el proceso de limpieza falla o si ocurre un error 
                durante la escritura del archivo en el disco (ej. permisos, ruta inválida).

        Returns:
            None: La función no retorna un valor, pero imprime un mensaje de 
                confirmación en consola al finalizar exitosamente.
        """
        cleaned_data = self.clean_key_column()
        if cleaned_data is not None:
            try:
                cleaned_data.to_csv(out_path, index=False)
                self.cleaned_data = cleaned_data
                print(f"clean data saved in '{out_path}'.")
            except Exception as e:
                raise ValueError(f"An error occurred while saving the file.: {e}")
        else:
            raise ValueError("The data could not be cleared for saving.")