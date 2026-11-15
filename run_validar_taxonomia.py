"""
Valida data/utilities/taxonomia.xlsx (lista fija de aspectos del NLP).

Uso (desde la raíz del repositorio):

    python run_validar_taxonomia.py

Revisa la forma del archivo contra las preguntas de preguntas_abiertas.xlsx.
La comparación con las clases del modelo se hace con
`nlp_utilidades.taxonomia.comparar_con_clases` desde el código del NLP.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nlp_utilidades.taxonomia import leer_taxonomia, version_taxonomia  # noqa: E402
from preparacion.configuracion import codigos_preguntas  # noqa: E402

UTILIDADES = Path(__file__).parent / "data" / "utilities"

if __name__ == "__main__":
    ruta = UTILIDADES / "taxonomia.xlsx"
    tax = leer_taxonomia(ruta, codigos_preguntas(UTILIDADES / "preguntas_abiertas.xlsx"))
    print(f"Taxonomía válida (versión {version_taxonomia(ruta)}): {len(tax)} filas, "
          f"{tax['aspecto_id'].nunique()} aspectos.")
    print(tax["pregunta_codigo"].value_counts().to_string())
