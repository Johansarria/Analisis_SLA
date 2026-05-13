import sys
import logging
import argparse
from pathlib import Path

from src.data.etl import DataProcessor
from src.core.forecaster import DemandForecaster
from src.utils.visualization import Visualizer
from src.config import ensure_dirs, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_forecasting(input_file: str, days_to_predict: int = 30):
    logger.info("🚀 Iniciando Pipeline de Forecasting de Demanda...")
    
    # Asegurar estructura
    ensure_dirs()
    
    # 1. ETL
    processor = DataProcessor()
    df_ts = processor.process_demand_series(input_file)
    
    output_path = PROCESSED_DATA_DIR / "demanda_diaria.csv"
    processor.save_processed_data(df_ts, str(output_path))
    
    # 2. Entrenamiento (Forecaster)
    forecaster = DemandForecaster()
    forecaster.train(df_ts)
    
    # 3. Predicción
    df_pred = forecaster.predict_future(df_ts, days=days_to_predict)
    pred_path = PROCESSED_DATA_DIR / "prediccion_demanda_futura.csv"
    df_pred.to_csv(pred_path, index=False)
    logger.info(f"Predicciones guardadas en {pred_path}")
    
    # 4. Visualización
    viz = Visualizer()
    viz.plot_demand_forecast(df_ts, df_pred)
    
    logger.info("✅ Pipeline de Forecasting completado con éxito.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ejecutar el modelo de Forecasting de Demanda')
    parser.add_argument('--input', type=str, default=r'C:\MLpractica2\data\processed\Jira ATP resumen 2 años.csv',
                        help='Ruta al archivo histórico de 2 años')
    parser.add_argument('--days', type=int, default=30,
                        help='Cantidad de días a predecir hacia el futuro')
    
    args = parser.parse_args()
    
    try:
        run_forecasting(args.input, args.days)
    except Exception as e:
        logger.error(f"❌ Error en el pipeline: {e}")
        sys.exit(1)
