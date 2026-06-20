# Module for cleaning key text column

# Standard Python libraries
from pathlib import Path
import re

class cleaner:
    SUPPORTED_FORMATS = {'.csv', '.xlsx', '.xls', '.xlsm'}
    def __init__(self, file_path : str | Path , key_column : str, separator : str = ',', sheet_name: str | int = 0):
        self.file_path = file_path
        self.separator = separator
        self.key_column = key_column
        self.sheet_name = sheet_name

    def load_data(self):
        """
        Loads and validate the data file for text processing.

        This method attempts to read a CSV (.csv) or Excel (.xlsx, .xls) file in the specified path, 
        first verifying the existence of the file, the supported format, 
        the presence of the key column, and the validity of the data type of that column.

        Returns:
            pd.DataFrame: A pandas DataFrame with the loaded data if the 
                operation was successful.

        Raises:
            ImportError: If the 'pandas' library is not installed, or if an attempt is made 
                to read an Excel file without having 'openpyxl' installed.
            FileNotFoundError: If the file does not exist in the specified path.
            ValueError: If the file format is not supported, if the key column does not exist, 
                if it is not of text type, or if any unexpected error occurs during reading.
        """      
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Pandas could not be imported. Please verify that it is installed in your working environment.") from exc
            

        try:
            if not Path(self.file_path).exists():
                raise FileNotFoundError(f"Dataset not found in the file path: {self.file_path}")
            # Validate extension
            extension = self.file_path.suffix.lower()
            if extension not in self.SUPPORTED_FORMATS:
                raise ValueError(
                f"Format '{extension}' is not supported. Supported formats: {self.SUPPORTED_FORMATS}"
            )

            try:
                if extension == '.csv':
                    data = pd.read_csv(self.file_path, sep=self.separator)

                elif extension in {'.xlsx', '.xls', '.xlsm'}:
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
        
        # Exception handling for common errors
        except Exception as e:
            raise ValueError(f"An error occurred: {e}")


    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Cleans a text string by removing noise and normalizing characters.

        The function performs the following operations in order:

        1. Removes leading and trailing whitespace.

        2. Converts the text to lowercase.

        3. Normalizes accented characters (tildes) to their plaintext counterparts.

        4. Removes special characters, keeping only alphanumeric characters, spaces, and the letter 'ñ'.

        Args:
            text (str): The raw text string to process.

        Returns:
            str: The processed and normalized text.
        """       
        if not isinstance(text, str):
            return ''  # Returns an empty string if the value is not text to avoid pandas representing it as 'float'
        
        # Remove leading and trailing whitespace
        cleaned_text = text.strip()

        # Convert to lowercase
        cleaned_text = cleaned_text.lower()

        # Remove accented letters
        accents = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
                  'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U'}
        for accent, letter in accents.items():
            cleaned_text = cleaned_text.replace(accent, letter)

        # Eliminate other special characters using a regular expression
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
        data = self.load_data().copy()  # Make a copy to avoid modifying the original DataFrame
        if data is not None:
            # Apply the cleaning function to each value in the key column without overwriting the original column.
            data[f"{self.key_column}_limpia"] = data[self.key_column].apply(self._clean_text)
            return data
        else:
            raise ValueError("The data could not be loaded for cleaning.")
        
    @staticmethod
    def _clean_stopwords(text: str) -> str:
        words = nltk.word_tokenize(text)
        filtradas = [w for w in words if w.lower() not in self.stop_words]
        return ' '.join(filtradas)

    def eliminate_stopwords(self):
        """
        Eliminates stopwords from a specific column in the DataFrame.

        The function uses the NLTK corpus for Spanish, extended with terms
        specific to the academic domain. It filters the words from the column

        defined in 'self.key_column' and generates a new column with the clean text.

        Args:
            text (str, optional): Reserved parameter for compatibility, 
                currently not used within the method's logic.

        Raises:
            ImportError: If the 'nltk' library is not installed in the environment.

        Returns:
            pd.DataFrame: The DataFrame 'self.cleaned_data' with the new column '{self.key_column}_sin_stopwords'.
        """        

        try:
            from nltk.corpus import stopwords
        except ImportError as exc:
            raise ImportError("Could not import -nltk-. Verify that it is installed in the working environment.") from exc


        stop_words = set(stopwords.words("spanish"))
        stop_words = set(list(stop_words)+ ['académico', 'academia', 'universidad'])

        # 2. Apply the function to the entire column at once (vectorized)
        columna_objetivo = f"{self.key_column}_limpia"
        self.cleaned_data[f"{self.key_column}_sin_stopwords"] = self.cleaned_data[columna_objetivo].apply(self._clean_stopwords)

        return self.cleaned_data


    def save_cleaned_data(self, out_path : str):
        """
        Run the cleaning process and export the result to a CSV file.

        This method internally invokes the data cleaning workflow, and if the 
        process is successful, it saves the resulting DataFrame to the specified path. 
        It also updates the internal state of the object with the processed data.

        Args:
            out_path (str): The file system path (including the file name and 
                .csv extension) where the cleaned data will be saved.

        Raises:
            ValueError: If the cleaning process fails or if an error occurs
            during writing the file to disk (e.g., permissions, invalid path).

        Returns:
            None: The function does not return a value, but prints a confirmation message
                in the console upon successful completion.
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