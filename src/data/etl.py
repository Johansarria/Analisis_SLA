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
        ).fillna('DESCONOCIDO')
        
        # Consolidar nulos en Nodo
        if 'Nodo' in df_limpio.columns:
            df_limpio['Nodo'] = df_limpio['Nodo'].fillna('DESCONOCIDO').str.upper()
            
        # Codificación Cíclica del tiempo
        df_limpio['Hora_Del_Dia'] = df_limpio['Fecha_Creacion'].dt.hour
        df_limpio['Hora_sin'] = np.sin(2 * np.pi * df_limpio['Hora_Del_Dia'] / 24.0)
        df_limpio['Hora_cos'] = np.cos(2 * np.pi * df_limpio['Hora_Del_Dia'] / 24.0)
        
        df_limpio['Dia_De_La_Semana'] = df_limpio['Fecha_Creacion'].dt.dayofweek
        df_limpio['Dia_sin'] = np.sin(2 * np.pi * df_limpio['Dia_De_La_Semana'] / 7.0)
        df_limpio['Dia_cos'] = np.cos(2 * np.pi * df_limpio['Dia_De_La_Semana'] / 7.0)
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

    def process_demand_series(self, input_file: str) -> pd.DataFrame:
        """Extrae la serie de tiempo de demanda diaria (volumen de tickets) por Zona y Nodo."""
        logger.info(f"Iniciando extracción de serie de tiempo desde {input_file}")
        try:
            try:
                df = pd.read_csv(input_file)
                if len(df.columns) < 3:
                    df = pd.read_csv(input_file, sep=';')
            except Exception:
                df = pd.read_csv(input_file, sep=';', encoding='latin-1')
        except Exception as e:
            logger.error(f"Error al leer archivo Jira para TS: {e}")
            raise
            
        # Encontrar las columnas relevantes
        fecha_col_raw = None
        zona_col_raw = None
        nodo_col_raw = None
        
        for raw_col, clean_col in JIRA_COLUMNS_MAP.items():
            if clean_col == 'Fecha_Creacion' and raw_col in df.columns:
                fecha_col_raw = raw_col
            elif clean_col == 'Zona' and raw_col in df.columns:
                zona_col_raw = raw_col
            elif clean_col == 'Nodo' and raw_col in df.columns:
                nodo_col_raw = raw_col
                
        if not fecha_col_raw:
            for col in df.columns:
                if 'creada' in str(col).lower() or 'created' in str(col).lower() or 'fecha' in str(col).lower():
                    fecha_col_raw = col
                    break
        
        if not fecha_col_raw:
            raise ValueError("No se encontró la columna de fecha de creación en el archivo.")
            
        if not zona_col_raw:
            logger.warning("No se encontró la columna de Zona explícita. Se intentará usar 'Proyecto' o se asignará 'GLOBAL'.")
            for col in df.columns:
                if 'proyecto' in str(col).lower() or 'project' in str(col).lower():
                    zona_col_raw = col
                    break
                    
        if not nodo_col_raw:
            logger.warning("No se encontró la columna de Nodo explícita. Se asignará 'DESCONOCIDO'.")
            for col in df.columns:
                if 'nodo' in str(col).lower() or 'node' in str(col).lower():
                    nodo_col_raw = col
                    break
            
        df['Fecha_Creacion'] = pd.to_datetime(
            df[fecha_col_raw].apply(self._translate_date), errors='coerce'
        )
        df = df.dropna(subset=['Fecha_Creacion'])
        df['Fecha'] = df['Fecha_Creacion'].dt.date
        
        if zona_col_raw:
            df['Zona'] = df[zona_col_raw].fillna('DESCONOCIDO').astype(str)
        else:
            df['Zona'] = 'GLOBAL'
            
        if nodo_col_raw:
            df['Nodo'] = df[nodo_col_raw].fillna('DESCONOCIDO').astype(str)
        else:
            df['Nodo'] = 'DESCONOCIDO'
            
        # Agrupar por Fecha, Zona y Nodo
        df_ts = df.groupby(['Fecha', 'Zona', 'Nodo']).size().reset_index(name='Tickets_Total')
        df_ts['Fecha'] = pd.to_datetime(df_ts['Fecha'])
        
        # Filtro de seguridad: Mapeo estricto de Nodos válidos por Zona para corregir errores de digitación en Jira
        VALID_ZONES_NODES = {
            'Zona Suroccidente ': ['Chiminangos', 'Troncal', 'Colon', 'Camino Real', 'Palmira', 'Tristan', 'Santa Mónica', 'DESCONOCIDO'],
            'Zona Centro': ['Chapinero', 'Asturias', 'Española', 'Progreso', 'Toberín', 'DESCONOCIDO'],
            'Zona Noroccidente': ['Transversal', 'Guayabal', 'Envigado', 'Itagüí', 'Galán', 'Sabaneta', 'DESCONOCIDO'],
            'Zona Area Metropolitana': ['Mosquera ', 'Madrid', 'Villa de Leyva', 'DESCONOCIDO'],
            'Mantenimiento Preventivo': ['Troncal', 'Colon', 'Chiminangos', 'Palmira', 'Galán', 'Española', 'Progreso', 'DESCONOCIDO']
        }
        
        # En lugar de un producto cruzado completo, obtendremos solo las parejas válidas de (Zona, Nodo)
        valid_pairs = df_ts[['Zona', 'Nodo']].drop_duplicates()
        
        # Aplicar el filtro estricto: Conservar la fila solo si el Nodo pertenece a la lista válida de esa Zona
        filtered_pairs = []
        for _, row in valid_pairs.iterrows():
            z = str(row['Zona']).strip()
            n = str(row['Nodo']).strip()
            
            # Buscar coincidencia (ignorando espacios extras)
            z_key = next((k for k in VALID_ZONES_NODES.keys() if k.strip() == z), None)
            
            if z_key:
                # Normalizar para comparar (ej. 'Colon' vs 'Colón', 'Itagüí' vs 'Itag')
                valid_nodes_normalized = [vn.lower().replace('ó','o').replace('ü','u').replace('í','i').replace('á','a').strip() for vn in VALID_ZONES_NODES[z_key]]
                n_normalized = n.lower().replace('ó','o').replace('ü','u').replace('í','i').replace('á','a').strip()
                
                # Excepciones ortográficas encontradas en el dataset
                if 'itag' in n_normalized: n_normalized = 'itagui'
                if 'monica' in n_normalized: n_normalized = 'santa monica'
                if 'espa' in n_normalized and 'ola' in n_normalized: n_normalized = 'espanola'
                
                # Check si el nodo normalizado está en la lista de permitidos normalizada
                if n_normalized in valid_nodes_normalized or n_normalized == 'desconocido':
                    filtered_pairs.append(row)
            else:
                # Si la zona no está en el mapa, la conservamos por seguridad
                filtered_pairs.append(row)
                
        valid_pairs = pd.DataFrame(filtered_pairs)
        
        all_dates = pd.date_range(df_ts['Fecha'].min(), df_ts['Fecha'].max())
        
        # Crear un dataframe base con todas las combinaciones reales validas
        base_df_list = []
        for _, row in valid_pairs.iterrows():
            temp_df = pd.DataFrame({'Fecha': all_dates})
            temp_df['Zona'] = row['Zona']
            temp_df['Nodo'] = row['Nodo']
            base_df_list.append(temp_df)
            
        base_df = pd.concat(base_df_list, ignore_index=True)
        
        # Hacer merge para rellenar vacíos con 0
        df_ts = pd.merge(base_df, df_ts, on=['Fecha', 'Zona', 'Nodo'], how='left')
        df_ts['Tickets_Total'] = df_ts['Tickets_Total'].fillna(0)
        df_ts = df_ts.sort_values(['Zona', 'Nodo', 'Fecha']).reset_index(drop=True)
        
        # Ingeniería de Variables Temporales (Lag Features)
        df_ts['Dia_Semana'] = df_ts['Fecha'].dt.dayofweek
        df_ts['Mes'] = df_ts['Fecha'].dt.month
        
        # Variables de rezago (lags) aplicadas por ZONA y NODO
        df_ts['Lag_1'] = df_ts.groupby(['Zona', 'Nodo'])['Tickets_Total'].shift(1)
        df_ts['Lag_7'] = df_ts.groupby(['Zona', 'Nodo'])['Tickets_Total'].shift(7)
        
        # Promedios móviles aplicados por ZONA y NODO
        df_ts['Media_Movil_7d'] = df_ts.groupby(['Zona', 'Nodo'])['Tickets_Total'].transform(
            lambda x: x.shift(1).rolling(window=7).mean()
        )
        df_ts['Media_Movil_14d'] = df_ts.groupby(['Zona', 'Nodo'])['Tickets_Total'].transform(
            lambda x: x.shift(1).rolling(window=14).mean()
        )
        
        # Descartar las primeras filas con NaNs (que ocurren por el shift)
        df_ts = df_ts.dropna().reset_index(drop=True)
        
        logger.info(f"Serie de tiempo por zonas y nodos procesada. Registros válidos: {len(df_ts)}")
        return df_ts

    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """Guarda el DataFrame procesado en un CSV."""
        df.to_csv(output_path, index=False)
        logger.info(f"Datos guardados en {output_path}")

if __name__ == "__main__":
    # Prueba rápida
    processor = DataProcessor()
    # Aquí iría la lógica de prueba si se ejecuta solo este archivo
