"""
Ejecuta la Etapa 3 (limpieza y n-gramas).

Uso (desde la raíz del repositorio):

    python run_limpieza.py                 # lee data/processed/nlp/ (salida del NLP)
    python run_limpieza.py --adaptador     # TEMPORAL: convierte primero el archivo
                                           # actual del NLP (obs19) y lee de ahí

Con --adaptador los identificadores son provisionales: sirve para revisar la
limpieza, no para cargar en Power BI.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from limpieza.etapa3 import CARPETA_NLP, ejecutar_limpieza  # noqa: E402

if __name__ == "__main__":
    if "--adaptador" in sys.argv[1:]:
        from limpieza.adaptador_nlp_actual import CARPETA_ADAPTADA, adaptar
        adaptar()
        ejecutar_limpieza(carpeta_nlp=CARPETA_ADAPTADA)
    else:
        ejecutar_limpieza(carpeta_nlp=CARPETA_NLP)
