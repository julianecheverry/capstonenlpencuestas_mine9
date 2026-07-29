# Settings

import pandas as pd
from pathlib import Path

UTILITIES_DIR = Path(__file__).resolve().parent.parent.parent
ACCENTS_PATH = UTILITIES_DIR / 'data' / 'utilities' / 'accents.xlsx'

# Accents to remove

df_accents = pd.read_excel(ACCENTS_PATH, sheet_name='accents')

accents = dict(zip(df_accents['accented'], df_accents['not_accented']))

#################
# Stopwords to remove Per specific survey:
# Loading stop words from Excel
# The xlsx file has two columns: 'encuesta' and 'palabra',
# where 'encuesta' is the name of the survey and
# 'palabra' is the corresponding stop word.

def cargar_stopwords_desde_excel(path_excel, sheet_name: int):
    """Carga y transforma el xlsx en el diccionario de stopwords esperado."""
    try:
        df = pd.read_excel(path_excel, sheet_name= sheet_name)
        # Agrupamos por la columna 'encuesta' y convertimos a lista cada grupo
        # dropna() asegura que no tengamos celdas vacías
        return df.groupby('encuesta')['palabra'].apply(list).to_dict()
    except Exception as e:
        print(f"Error cargando stopwords: {e}")
        return {}

STOPWORDS_PATH= UTILITIES_DIR / 'data' / 'utilities' / 'stopwords.xlsx'
# The stopwords_dict dictionary is now loaded dynamically
stopwords_dict = cargar_stopwords_desde_excel(
                    STOPWORDS_PATH, sheet_name='particulares')

# The stopwords_global dictionary is now loaded dynamically
stopwords_global = cargar_stopwords_desde_excel(
                    STOPWORDS_PATH, sheet_name='globales')