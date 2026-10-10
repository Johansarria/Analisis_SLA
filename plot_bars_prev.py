import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.core.forecaster import DemandForecaster
from src.config import PROCESSED_DATA_DIR, PLOTS_DIR
import logging

logging.basicConfig(level=logging.WARNING)

def plot_bars_prev():
    # 1. Cargar datos
    df_ts = pd.read_csv(PROCESSED_DATA_DIR / "demanda_diaria.csv")
    df_ts['Fecha'] = pd.to_datetime(df_ts['Fecha'])
    
    # 2. Definir rango
    cutoff_date = pd.to_datetime('2026-04-12')
    df_train = df_ts[df_ts['Fecha'] < cutoff_date].copy()
    df_test = df_ts[df_ts['Fecha'] >= cutoff_date].copy()
    
    # 3. Entrenar y Predecir
    forecaster = DemandForecaster()
    forecaster.train(df_train)
    days_to_predict = (df_ts['Fecha'].max() - cutoff_date).days + 1
    df_pred = forecaster.predict_future(df_train, days=days_to_predict)
    
    # 4. Consolidar Mantenimiento Preventivo
    target_zone = 'Mantenimiento Preventivo'
    real_monthly = df_test[df_test['Zona'] == target_zone].groupby('Fecha')['Tickets_Total'].sum().reset_index()
    pred_monthly = df_pred[df_pred['Zona'] == target_zone].groupby('Fecha')['Prediccion_Tickets'].sum().reset_index()
    
    comparativa = pd.merge(real_monthly, pred_monthly, on='Fecha', how='inner')
    comparativa.rename(columns={'Tickets_Total': 'Real', 'Prediccion_Tickets': 'Prediccion'}, inplace=True)
    comparativa['Fecha_Str'] = comparativa['Fecha'].dt.strftime('%d-%b')
    
    # 5. Generar Gráfico de Barras Agrupadas
    x = np.arange(len(comparativa))
    width = 0.35
    
    plt.figure(figsize=(15, 7))
    plt.bar(x - width/2, comparativa['Real'], width, label='Real', color='#f1c40f', alpha=0.8)
    plt.bar(x + width/2, comparativa['Prediccion'], width, label='Predicción', color='#27ae60', alpha=0.8)
    
    plt.title('COMPARATIVA DIARIA: REAL VS PREDICCIÓN (MANTENIMIENTO PREVENTIVO)', fontsize=16, fontweight='bold')
    plt.xlabel('Día', fontsize=12)
    plt.ylabel('Cantidad de Tickets', fontsize=12)
    plt.xticks(x, comparativa['Fecha_Str'], rotation=45, fontsize=10)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Añadir valores sobre las barras
    for i, v in enumerate(comparativa['Real']):
        plt.text(i - width/2, v + 0.2, str(int(v)), ha='center', fontsize=9)
    for i, v in enumerate(comparativa['Prediccion']):
        plt.text(i + width/2, v + 0.2, str(int(v)), ha='center', fontsize=9)
        
    plt.tight_layout()
    
    plot_path = PLOTS_DIR / "barras_validacion_prev.png"
    plt.savefig(plot_path)
    print(f"\nGráfico de barras guardado en: {plot_path}")

if __name__ == "__main__":
    plot_bars_prev()
