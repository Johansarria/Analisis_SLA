import pandas as pd
from pathlib import Path
from src.core.models import ModelTrainer
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = Path(r"C:\MLpractica2\data\processed")
FILE_ANS = DATA_PATH / "datos_ans_limpios.csv"
FILE_JIRA = DATA_PATH / "datos_jira_regresion.csv"

def retrain():
    trainer = ModelTrainer()
    
    if FILE_ANS.exists():
        logger.info(f"Cargando datos ANS desde {FILE_ANS}")
        df_ans = pd.read_csv(FILE_ANS)
        trainer.train_ans_classifier(df_ans)
    else:
        logger.warning(f"No se encontró el archivo ANS en {FILE_ANS}")
        
    if FILE_JIRA.exists():
        logger.info(f"Cargando datos Jira desde {FILE_JIRA}")
        df_jira = pd.read_csv(FILE_JIRA)
        
        # Como los datos procesados externamente pueden no tener 
        # las características cíclicas que agregamos en el ETL de esta versión, 
        # las recreamos temporalmente si no existen.
        if 'Hora_sin' not in df_jira.columns and 'Hora_Del_Dia' in df_jira.columns:
            df_jira['Hora_sin'] = np.sin(2 * np.pi * df_jira['Hora_Del_Dia'] / 24.0)
            df_jira['Hora_cos'] = np.cos(2 * np.pi * df_jira['Hora_Del_Dia'] / 24.0)
            
        if 'Dia_sin' not in df_jira.columns and 'Dia_De_La_Semana' in df_jira.columns:
            df_jira['Dia_sin'] = np.sin(2 * np.pi * df_jira['Dia_De_La_Semana'] / 7.0)
            df_jira['Dia_cos'] = np.cos(2 * np.pi * df_jira['Dia_De_La_Semana'] / 7.0)
            
        trainer.train_oraculo_regressor(df_jira)
    else:
        logger.warning(f"No se encontró el archivo Jira en {FILE_JIRA}")

if __name__ == "__main__":
    retrain()
