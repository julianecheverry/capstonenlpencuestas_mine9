import sys
from pathlib import Path

# Necesario porque orquestador.py vive en src/, no en la raíz
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orquestador import process_survey

if __name__ == "__main__":
    process_survey("evadoc_evaluacion")
    process_survey("evadoc_autoevaluacion")
    process_survey("autoevaluacion_acreditacion")