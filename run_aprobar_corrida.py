"""
Registra la aprobación del cliente de una corrida de la Etapa 1.

Uso (desde la raíz del repositorio), con la respuesta del cliente a la mano:

    python run_aprobar_corrida.py <id_corrida_preparacion> "<quién aprobó>" "<referencia del correo>"

Sin argumentos, lista las corridas guardadas y las aprobaciones registradas.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from preparacion.aprobacion import aprobar, corridas, leer_aprobaciones  # noqa: E402
from preparacion.etapa1 import CARPETA_SALIDA  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Corridas guardadas:", *corridas(CARPETA_SALIDA) or ["(ninguna)"], sep="\n  ")
        print("\nAprobaciones:")
        print(leer_aprobaciones(CARPETA_SALIDA).to_string(index=False))
    elif len(sys.argv) == 4:
        fila = aprobar(sys.argv[1], sys.argv[2], sys.argv[3], CARPETA_SALIDA)
        print(f"Corrida {fila['id_corrida_preparacion']} aprobada por {fila['aprobado_por']} "
              f"({fila['fecha_aprobacion_utc']}).")
    else:
        print(__doc__)
        sys.exit(1)
