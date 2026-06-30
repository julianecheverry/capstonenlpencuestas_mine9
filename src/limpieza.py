# Module for cleaning key text column

# Standard Python libraries import
from pathlib import Path
import re
import pandas as pd # Allowing type hints for pandas DataFrame
from pandas.api.types import is_string_dtype
import openpyxl
import nltk
from settings import ( accents,
                      stopwords_global,
                    stopwords_evaluaciondocente,
                    stopwords_autoevaluaciondocente,
                    stopwords_calidaddocentes,
                    stopwords_calidadadministrativos,
                    stopwords_calidadestudiantes,
                    stopwords_calidaddirectivos,
                    stopwords_calidadegresados
                      )

# NLTK elements import

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(nltk.corpus.stopwords.words("spanish"))
stop_words.update(stopwords_global)


# Class definition
class Cleaner:
    SUPPORTED_FORMATS = {'.csv', '.xlsx', '.xls', '.xlsm'}
    def __init__(self, file_path: str | Path ,
                 key_column: str,
                 separator: str = ',',
                 sheet_name: str | int = 0) -> None:
        self.file_path = Path(file_path)
        self.separator = separator
        self.key_column = key_column
        self.sheet_name = sheet_name

    def load_data(self) -> pd.DataFrame:
        """
        Loads and validate the data file for text processing.

        This method attempts to read a CSV (.csv) or Excel (.xlsx, .xls) file
        in the specified path, first verifying the existence of the file,
        the supported format, the presence of the key column, and the validity
        of the data type of that column.

        Returns:
            pd.DataFrame: A pandas DataFrame with the loaded data if the
                operation was successful.

        Raises:
            ImportError: If the 'pandas' library is not installed, or if an
            attempt is made to read an Excel file without having 'openpyxl'
            installed.
            FileNotFoundError: If the file does not exist in the specified
            path.
            ValueError: If the file format is not supported, if the key column
            does not exist, if it is not of text type, or if any unexpected
            error occurs during reading.
        """

        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset not found in the file path: "
                                    f"{self.file_path}")
        # Validate extension with the suffix attribute
        extension = self.file_path.suffix.lower()
        if extension not in self.SUPPORTED_FORMATS:
            raise ValueError(
            f"Format '{extension}' is not supported. Supported formats: "
            f"{self.SUPPORTED_FORMATS}"
        )

        try:
            if extension == '.csv':
                data = pd.read_csv(self.file_path, sep=self.separator)

            elif extension in {'.xlsx', '.xls', '.xlsm'}:
                data = pd.read_excel(self.file_path,
                                        sheet_name=self.sheet_name)

        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ValueError(
                f"Error reading the file format: {exc}"
            ) from exc

        except pd.errors.DatabaseError as exc:
            raise ValueError(
                f"Error reading data from file: {exc}"
            ) from exc

        # Validations to ensure the key column exists and is of text type
        if self.key_column not in data.columns:
            raise ValueError(f"Column '{self.key_column}' "
                                f"not found in the file.")

        # Avoids pandas representing NaN values as 'float' in the key column
        data[self.key_column] = data[self.key_column].fillna('')

        if not is_string_dtype(data[self.key_column]):
            raise ValueError(f"Column '{self.key_column}' "
                                f"is not of text type.")

        return data

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Cleans a text string by removing noise and normalizing characters.
        The function performs the following operations in order:
        1. Removes leading and trailing whitespace.
        2. Converts the text to lowercase.
        3. Normalizes accented characters (tildes) to their plaintext
        counterparts.
        4. Removes special characters, keeping only alphanumeric characters,
        spaces, and the letter 'ñ'.

        Args:
            text (str): The raw text string to process.

        Returns:
            str: The processed and normalized text.
        """
        # Returns an empty string if the value is not text
        # to avoid pandas representing it as 'float'
        if not isinstance(text, str):
            return ''

        # Remove leading and trailing whitespace
        cleaned_text = text.strip()

        # Convert to lowercase
        cleaned_text = cleaned_text.lower()

        # Remove accented letters
        for accent, letter in accents.items():
            cleaned_text = cleaned_text.replace(accent, letter)

        # Eliminate other special characters using a regular expression
        cleaned_text = re.sub(r'[^a-zA-Z0-9\sñÑ]', '', cleaned_text)

        return cleaned_text

    def clean_key_column(self) -> pd.DataFrame:
        """
        Performs the cleaning of the text in the key column of the loaded
        DataFrame.

        This method delegates the data loading to 'load_data', applies a
        cleaning transformation (defined in 'clean_text') to the specified
        column in 'self.key_column' and generates a new resulting column
        in the DataFrame.

        Returns:
            None

        Raises:
            ValueError: If the method 'load_data' returns None, indicating
            that it was not possible to obtain the original data source.
        """
        # Make a copy to avoid modifying the original DataFrame
        data = self.load_data().copy()
        if data is not None:
            # Apply the cleaning function to each value in the key column
            # without overwriting the original column.
            data[f"{self.key_column}_clean"] = (
                data[self.key_column].apply(self._clean_text)
                )

            self.cleaned_data = data

            return self.cleaned_data

        else:
            raise ValueError("The data could not be loaded for cleaning.")


    def _clean_stopwords(self, text: str) -> str:
        words = nltk.word_tokenize(text)
        filtered_words = [w for w in words if w.lower() not in stop_words]
        return ' '.join(filtered_words)

    def eliminate_stopwords(self) -> pd.DataFrame:
        """
        Eliminates stopwords from a specific column in the DataFrame.

        The function uses the NLTK corpus for Spanish, extended with terms
        specific to the academic domain. It filters the words from the column
        defined in 'self.key_column' and generates a new column with
        the clean text.

        Args:
            text (str, optional): Reserved parameter for compatibility,
                currently not used within the method's logic.

        Raises:
            ImportError: If the 'nltk' library is not installed in the
            environment.

        Returns:
            pd.DataFrame: The DataFrame 'self.cleaned_data' with
            the new column '{self.key_column}_no_stopwords'.
        """
        # if not hasattr(self, 'cleaned_data') or self.cleaned_data is None:
        #      self.cleaned_data = self.clean_key_column()

        # Apply the function to the entire column at once (vectorized)
        goal_column = f"{self.key_column}_clean"
        self.cleaned_data[f"{self.key_column}_no_stopwords"] = (
            self.cleaned_data[goal_column].apply(
                lambda x:self._clean_stopwords(x))
            )

        return self.cleaned_data


    def save_cleaned_data(self, out_path: str) -> None:
        """
        Runs the cleaning process and export the result to a CSV file.

        This method internally invokes the data cleaning workflow, and if the
        process is successful, it saves the resulting DataFrame to the
        specified path. It also updates the internal state of the object with
        the processed data.

        Args:
            out_path (str): The file system path (including the file name and
                .csv extension) where the cleaned data will be saved.

        Raises:
            ValueError: If the cleaning process fails or if an error occurs
            during writing the file to disk (e.g., permissions, invalid path).

        Returns:
            None: The function does not return a value, but prints a
            confirmation message in the console upon successful completion.
        """
        # cleaned_data = self.clean_key_column()
        if self.cleaned_data is not None:
            try:
                self.cleaned_data.to_csv(out_path, index=False)
                print(f"clean data saved in '{out_path}'.")
            except Exception as e:
                raise ValueError(f"An error occurred while saving the file.:"
                                 f" {e}")
        else:
            raise ValueError("The data could not be cleared for saving.")