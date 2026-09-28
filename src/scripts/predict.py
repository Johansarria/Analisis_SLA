import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import setup_logging, MODEL_ANS_XGB, MODEL_ORACULO_XGB
from src.core.inference import InferenceEngine

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Predice SLA o tiempo de resolución para tickets de Jira en formato CSV."
    )
    parser.add_argument(
        "--input_csv", required=True, type=str,
        help="Ruta al CSV con los tickets a predecir."
    )
    parser.add_argument(
        "--model", required=True, choices=["ans", "oracle"],
        help="Modelo a utilizar: 'ans' para clasificación de cumplimiento, 'oracle' para tiempo de resolución."
    )
    parser.add_argument(
        "--output_csv", required=True, type=str,
        help="Ruta donde se guardará el CSV con la columna 'Prediccion' agregada."
    )

    args = parser.parse_args()
    setup_logging()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    if not input_path.exists():
        logger.error(f"Archivo de entrada no encontrado: {input_path}")
        sys.exit(1)

    model_path = MODEL_ANS_XGB if args.model == "ans" else MODEL_ORACULO_XGB
    if not model_path.exists():
        logger.error(f"Modelo no encontrado: {model_path}. Entrena primero con run_pipeline.")
        sys.exit(1)

    logger.info(f"Cargando modelo desde {model_path}...")
    engine = InferenceEngine(model_path)

    df = pd.read_csv(input_path)
    logger.info(f"Registros a predecir: {len(df)}")

    try:
        predictions = engine.predict(df)
    except Exception as e:
        logger.error(f"Error durante la predicción: {e}")
        sys.exit(1)

    df['Prediccion'] = predictions
    df.to_csv(output_path, index=False)
    logger.info(f"Resultados guardados en {output_path}")


if __name__ == "__main__":
    main()
