import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Tuple, List, Optional
import logging
from src.config import FILE_JIRA_CLEAN

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Clase para la detección de anomalías en nodos de red."""

    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.is_fitted = False

    def build_node_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construye perfiles de salud por nodo a partir de datos de fallas."""
        logger.info("Construyendo perfiles de salud por nodo...")
        
        perfiles = df.groupby('Nodo').agg(
            Total_Fallas=('Tipo_Falla', 'count'),
            Tiempo_Promedio_Minutos=('Minutos_Resolucion', 'mean'),
            Fallas_Criticas=('Prioridad', lambda x: (x.str.upper() == 'ALTA').sum() + (x.str.upper() == 'CRÍTICA').sum())
        ).reset_index()

        perfiles['Porcentaje_Critico'] = (perfiles['Fallas_Criticas'] / perfiles['Total_Fallas']) * 100
        
        # Filtro de nodos con pocos datos (ruido)
        perfiles = perfiles[perfiles['Total_Fallas'] > 5].reset_index(drop=True)
        return perfiles

    def detect_anomalies(self, perfiles: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Aplica K-Means para identificar nodos en riesgo."""
        logger.info("Iniciando agrupación espacial (Clustering)...")
        
        features = ['Total_Fallas', 'Tiempo_Promedio_Minutos', 'Porcentaje_Critico']
        data_to_scale = perfiles[features]
        
        scaled_data = self.scaler.fit_transform(data_to_scale)
        perfiles['Grupo_IA'] = self.kmeans.fit_predict(scaled_data)
        self.is_fitted = True

        # Identificar el grupo crítico (mayor promedio de fallas)
        promedios_grupos = perfiles.groupby('Grupo_IA')['Total_Fallas'].mean()
        grupo_critico_id = promedios_grupos.idxmax()
        
        nodos_en_peligro = perfiles[perfiles['Grupo_IA'] == grupo_critico_id]
        return perfiles, nodos_en_peligro

    def get_top_critical_nodos(self, nodos_en_peligro: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Devuelve los top N nodos más críticos."""
        return nodos_en_peligro.sort_values(by='Total_Fallas', ascending=False).head(top_n)

if __name__ == "__main__":
    # Ejemplo de uso
    try:
        df_jira = pd.read_csv(str(FILE_JIRA_CLEAN))
        detector = AnomalyDetector()
        perfiles = detector.build_node_profiles(df_jira)
        _, peligros = detector.detect_anomalies(perfiles)
        top_10 = detector.get_top_critical_nodos(peligros)
        print(top_10)
    except Exception as e:
        logger.error(f"Error en ejecución de prueba: {e}")
