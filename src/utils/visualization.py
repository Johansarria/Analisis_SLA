import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
import logging
from src.config import PLOTS_DIR

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuración de estilo global
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]

class Visualizer:
    """Clase para la generación de reportes visuales."""

    @staticmethod
    def plot_risk_comparison(df_heat: pd.DataFrame, output_name: str = "grafico_riesgo_nodos.png"):
        """Genera un gráfico de barras comparando el real vs proyectado."""
        logger.info("Generando gráfico de comparación de riesgo...")
        top_10 = df_heat.head(10)
        
        fig, ax = plt.subplots()
        nodos = top_10['Nodo'].str.slice(0, 15)
        x = range(len(nodos))
        width = 0.35

        ax.bar([i - width/2 for i in x], top_10['Abril'], width, label='Real', color='#3498db')
        ax.bar([i + width/2 for i in x], top_10['Proyección Abril'], width, label='Proyectado', color='#e74c3c')

        ax.set_ylabel('Cantidad de Tickets')
        ax.set_title('ANÁLISIS DE RIESGO: REAL VS PROYECTADO (TOP 10 NODOS)')
        ax.set_xticks(x)
        ax.set_xticklabels(nodos, rotation=45)
        ax.legend()

        plt.tight_layout()
        save_path = PLOTS_DIR / output_name
        plt.savefig(save_path)
        logger.info(f"Gráfico guardado en {save_path}")
        plt.close()

    @staticmethod
    def plot_operational_distribution(df_zonas: pd.DataFrame, output_name: str = "grafico_distribucion_zonas.png"):
        """Genera un gráfico de pastel de la carga operativa por zona."""
        logger.info("Generando gráfico de distribución operativa...")
        
        # Eliminar columna temporal si existe
        data_to_sum = df_zonas.drop('Dia_Calendario', axis=1) if 'Dia_Calendario' in df_zonas.columns else df_zonas
        zonas_totales = data_to_sum.sum().sort_values(ascending=False)

        plt.figure(figsize=(10, 7))
        colores = sns.color_palette('pastel')[0:len(zonas_totales)]
        
        plt.pie(zonas_totales, 
                labels=zonas_totales.index, 
                autopct='%1.1f%%', 
                startangle=140, 
                colors=colores,
                explode=[0.1 if i == 0 else 0 for i in range(len(zonas_totales))])

        plt.title('DISTRIBUCIÓN DE CARGA OPERATIVA POR ZONA')
        save_path = PLOTS_DIR / output_name
        plt.savefig(save_path)
        logger.info(f"Gráfico guardado en {save_path}")
        plt.close()

    @staticmethod
    def plot_anomaly_radar(top_peligro: pd.DataFrame, output_name: str = "radar_anomalias.png"):
        """Genera una visualización del radar de anomalías."""
        logger.info("Generando visualización de radar de anomalías...")
        # (Implementación personalizada si se requiere un radar real, 
        # por ahora un gráfico de barras de criticidad)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_peligro, x='Total_Fallas', y='Nodo', hue='Porcentaje_Critico', palette='Reds')
        plt.title('Top Nodos con Comportamiento Anómalo')
        save_path = PLOTS_DIR / output_name
        plt.savefig(save_path)
        plt.close()
