"""
Ejecuta la Etapa 1 (preparación de datos) para todas las encuestas
configuradas en la hoja `etapa1` de data/utilities/preguntas_abiertas.xlsx.

Uso (desde la raíz del repositorio):

    poetry run python run_etapa1.py

Opcional: nombres de encuesta como argumentos para procesar solo esas.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from preparacion.etapa1 import ejecutar_etapa1  # noqa: E402

if __name__ == "__main__":
    ejecutar_etapa1(encuestas=sys.argv[1:] or None)
