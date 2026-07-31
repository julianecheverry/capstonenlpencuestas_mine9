# Settings

import pandas as pd
from pathlib import Path

UTILITIES_DIR = Path(__file__).resolve().parent.parent
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

# --- Synonyms to replace --- #

# --- Word2Vec model configuration ---

# Local path where the pretrained Word2Vec model should be stored/loaded from.
SYNONYMS_MODEL_PATH = UTILITIES_DIR / 'models' / 'SBW-vectors-300-min5.bin'

# Official download source for the SBWCE Word2Vec pretrained model (Spanish,
# 300 dimensions, binary Word2Vec format). Reference:
# Cristian Cardellino - Spanish Billion Word Corpus and Embeddings.
SYNONYMS_MODEL_URL = (
    "http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5.bin.gz"
)

# --- Canonical synonym seeds ---
# Each key is the "canonical" term that will replace its associated
# synonyms/variants in the survey text. The lists are SEED words used to
# query `most_similar` in Word2Vec and auto-expand the dictionary; they are
# not the final exhaustive list.
canonical_synonyms_seeds = {
    "bueno": ["bueno", "excelente", "maravilloso"],
    "malo": ["malo", "terrible", "pesimo"],
    "rapido": ["rapido", "veloz"],
    "lento": ["lento", "demorado"],
    "docente": ["docente", "profesor", "profesora", "maestro", "maestra"],
    "estudiante": ["estudiante", "alumno", "alumna"],
    "facil": ["facil", "simple", "sencillo"],
    "clase": ["clase", "curso", "materia"],
    # Add more canonical categories relevant to the survey domain here.
}

# --- Manual colloquialisms (Colombian Spanish) ---
# These are added/overridden manually because Word2Vec pretrained on formal
# corpora (Wikipedia, news) rarely captures regional slang correctly.
# Format: {variant_word: canonical_word}
colombian_colloquialisms = {
    "bacano": "bueno",
    "chevere": "bueno",
    "chimba": "bueno",
    "una nota": "bueno",
    "berraco": "bueno",
    "una chimba": "bueno",
    "10/10": "bueno",
    "1010": "bueno",
    "una lata": "malo",
    "un paseo": "bueno",
}

# --- Similarity parameters ---
SYNONYMS_SIMILARITY_THRESHOLD = 0.65   # Minimum cosine similarity accepted
SYNONYMS_TOPN = 15                     # Candidates retrieved per seed word