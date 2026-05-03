import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging
from src.config import (
    FILE_JIRA_RAW, FILE_JIRA_CLEAN, FILE_ANS_CLEAN, 
    JIRA_COLUMNS_MAP, MONTHS_ES_TO_EN, 
    MIN_MINUTES_RESOLUTION, MAX_MINUTES_RESOLUTION,
    RAW_DATA_DIR
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """Clase para el procesamiento de datos ETL del proyecto SLA."""

    def __init__(self):
        self.months_map = MONTHS_ES_TO_EN

    def _translate_date(self, date_str: str) -> str:
        """Traduce meses en español a inglés para pd.to_datetime."""
        if pd.isna(date_str):
            return np.nan
        date_str = str(date_str).lower()
        for es, en in self.months_map.items():
            if es in date_str:
                date_str = date_str.replace(es, en)
                break
        return date_str

    def process_jira(self, input_file: str = str(FILE_JIRA_RAW)) -> pd.DataFrame:
        """Limpia y procesa los datos de Jira."""
        logger.info(f"Iniciando ETL de Jira desde {input_file}")
        
        try:
            try:
                df = pd.read_csv(input_file)
                if len(df.columns) < 3:
                    df = pd.read_csv(input_file, sep=';')
            except Exception:
                df = pd.read_csv(input_file, sep=';', encoding='latin-1')
        except Exception as e:
            logger.error(f"Error al leer archivo Jira: {e}")
            raise

        # Selección y renombramiento
        cols_needed = list(JIRA_COLUMNS_MAP.keys())
        df_limpio = df[cols_needed].copy()
        df_limpio.columns = [JIRA_COLUMNS_MAP[c] for c in cols_needed]

        # Limpieza de nulos
        df_limpio = df_limpio.dropna(subset=['Fecha_Solucion', 'Fecha_Creacion'])

        # Conversión de fechas
        df_limpio['Fecha_Creacion'] = pd.to_datetime(
            df_limpio['Fecha_Creacion'].apply(self._translate_date), errors='coerce'
        )
        df_limpio['Fecha_Solucion'] = pd.to_datetime(
            df_limpio['Fecha_Solucion'].apply(self._translate_date), errors='coerce'
        )
        df_limpio = df_limpio.dropna(subset=['Fecha_Creacion', 'Fecha_Solucion'])

        # Cálculo de minutos de resolución
        diff = df_limpio['Fecha_Solucion'] - df_limpio['Fecha_Creacion']
        df_limpio['Minutos_Resolucion'] = diff.dt.total_seconds() / 60.0

        # Ingeniería de Características
        df_limpio['Tipo_Falla'] = df_limpio['Resumen'].apply(
            lambda x: str(x).split('/')[-1].strip().upper()[:30]
        )
        df_limpio['Hora_Del_Dia'] = df_limpio['Fecha_Creacion'].dt.hour
        df_limpio['Dia_De_La_Semana'] = df_limpio['Fecha_Creacion'].dt.dayofweek
        df_limpio['Usuarios_Afectados'] = pd.to_numeric(
            df_limpio['Usuarios_Afectados'], errors='coerce'
        ).fillna(1)

        # Filtro de anomalías
        df_final = df_limpio[
            (df_limpio['Minutos_Resolucion'] >= MIN_MINUTES_RESOLUTION) & 
            (df_limpio['Minutos_Resolucion'] <= MAX_MINUTES_RESOLUTION)
        ]

        logger.info(f"Tickets procesados: {len(df_final)}")
        return df_final

    def process_ans(self, excel_file: str) -> pd.DataFrame:
        """Limpia y procesa los datos de ANS (Excel)."""
        logger.info(f"Iniciando ETL de ANS desde {excel_file}")
        
        try:
            df = pd.read_excel(excel_file, sheet_name='NACIONAL')
        except Exception as e:
            logger.error(f"Error al leer Excel ANS: {e}")
            raise

        cols_needed = ['Prioridad', 'CENTRAL', 'SUEPERVISOR', 'CUMPLE/NO CUMPLE']
        for col in cols_needed:
            if col not in df.columns:
                raise ValueError(f"Columna faltante en ANS: {col}")

        df_limpio = df[cols_needed].copy()
        df_limpio = df_limpio.dropna(subset=['CUMPLE/NO CUMPLE'])
        
        # Target binario: 1 si NO CUMPLE, 0 si CUMPLE
        df_limpio['Target'] = df_limpio['CUMPLE/NO CUMPLE'].apply(
            lambda x: 1 if str(x).strip().upper() == 'NO CUMPLE' else 0
        )

        logger.info(f"Tickets ANS procesados: {len(df_limpio)}")
        return df_limpio

    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """Guarda el DataFrame procesado en un CSV."""
        df.to_csv(output_path, index=False)
        logger.info(f"Datos guardados en {output_path}")

if __name__ == "__main__":
    # Prueba rápida
    processor = DataProcessor()
    # Aquí iría la lógica de prueba si se ejecuta solo este archivo
