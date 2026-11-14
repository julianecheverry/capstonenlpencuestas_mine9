"""
Ejecuta la Etapa 1 (preparación de datos) para todas las encuestas
configuradas en la hoja `etapa1` de data/utilities/preguntas_abiertas.xlsx.

Uso (desde la raíz del repositorio):

    poetry run python run_etapa1.py

La frase secreta para codificar el Código Cruce se toma de la variable de
entorno ETAPA1_SAL o, si no existe, del archivo `.etapa1_sal` en la raíz del
repositorio (una sola línea; está en .gitignore y nunca debe subirse).

Opcional: nombres de encuesta como argumentos para procesar solo esas.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from preparacion.etapa1 import ejecutar_etapa1  # noqa: E402

if __name__ == "__main__":
    ejecutar_etapa1(encuestas=sys.argv[1:] or None)
