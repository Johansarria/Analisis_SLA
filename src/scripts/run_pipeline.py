import sys
import logging
from src.config import (
    ensure_dirs, FILE_JIRA_RAW, FILE_JIRA_CLEAN, 
    FILE_ANS_CLEAN, PROCESSED_DATA_DIR, RAW_DATA_DIR
)
from src.data.etl import DataProcessor
from src.core.models import ModelTrainer
from src.core.detector import AnomalyDetector
from src.utils.visualization import Visualizer
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_full_pipeline():
    """Ejecuta el pipeline completo de análisis SLA."""
    logger.info("🚀 Iniciando Pipeline Maestro de Análisis SLA...")
    
    # 1. Asegurar estructura
    ensure_dirs()
    
    processor = DataProcessor()
    trainer = ModelTrainer()
    detector = AnomalyDetector()
    viz = Visualizer()

    # 2. ETL
    try:
        # Jira ETL
        df_jira = processor.process_jira(str(FILE_JIRA_RAW))
        processor.save_processed_data(df_jira, str(FILE_JIRA_CLEAN))
        
        # ANS ETL (Si existe el archivo Excel)
        excel_ans = RAW_DATA_DIR / 'BASE PARA DATOS  O&M FEBRERO ZONAS-REVISADO CALI (2).xlsx'
        if excel_ans.exists():
            df_ans = processor.process_ans(str(excel_ans))
            processor.save_processed_data(df_ans, str(FILE_ANS_CLEAN))
            
            # 3. Entrenamiento (Solo si hay datos ANS)
            trainer.train_ans_classifier(df_ans)
        else:
            logger.warning(f"Archivo Excel {excel_ans} no encontrado. Saltando ETL de ANS.")

        # Oráculo Training
        trainer.train_oraculo_regressor(df_jira)

        # 4. Detección de Anomalías
        perfiles = detector.build_node_profiles(df_jira)
        perfiles_full, peligros = detector.detect_anomalies(perfiles)
        top_nodos = detector.get_top_critical_nodos(peligros)
        
        logger.info(f"Detección completada. Nodos críticos encontrados: {len(peligros)}")

        # 5. Visualización
        viz.plot_anomaly_radar(top_nodos)
        
        # Simulación de otros datos para gráficos (si existen los archivos de serie de tiempo)
        # Esto es un placeholder para demostrar la integración
        # if FILE_TIME_SERIES_ZONES.exists():
        #     df_zonas = pd.read_csv(str(FILE_TIME_SERIES_ZONES))
        #     viz.plot_operational_distribution(df_zonas)

        logger.info("✅ Pipeline completado con éxito.")

    except Exception as e:
        logger.error(f"❌ Error crítico en el pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_full_pipeline()
