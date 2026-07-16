# Settings

# Accents to remove -> 🅿⚠ traer también desde un archivo excel
accents = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'ä': 'a', 'ë': 'e', 'ï': 'i', 'ö': 'o', 'ü': 'u',
            'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
            'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',

            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
            'Ä': 'A', 'Ë': 'E', 'Ï': 'I', 'Ö': 'O', 'Ü': 'U',
            'À': 'A', 'È': 'E', 'Ì': 'I', 'Ò': 'O', 'Ù': 'U',
            'Â': 'A', 'Ê': 'E', 'Î': 'I', 'Ô': 'O', 'Û': 'U'}

#################
# Per specific survey:
# carga de stopwords desde excel
# el xlsx tiene dos columnas: 'encuesta' y 'palabra',
# donde 'encuesta' es el nombre de la encuesta y
# 'palabra' es la stopword correspondiente.

import pandas as pd

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

# El diccionario ahora se carga dinámicamente
stopwords_dict = cargar_stopwords_desde_excel(
                    'data/raw/stopwords.xlsx', sheet_name='particulares')

# Stopwords to remove
# Global level

stopwords_global = cargar_stopwords_desde_excel(
                    'data/raw/stopwords.xlsx', sheet_name='globales')