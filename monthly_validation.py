import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.core.forecaster import DemandForecaster
from src.config import PROCESSED_DATA_DIR, PLOTS_DIR
import logging

logging.basicConfig(level=logging.WARNING)

def monthly_validation():
    # 1. Cargar datos
    df_ts = pd.read_csv(PROCESSED_DATA_DIR / "demanda_diaria.csv")
    df_ts['Fecha'] = pd.to_datetime(df_ts['Fecha'])
    
    # 2. Partir datos: Entrenar hasta el 11 de abril
    cutoff_date = pd.to_datetime('2026-04-12')
    df_train = df_ts[df_ts['Fecha'] < cutoff_date].copy()
    df_test = df_ts[df_ts['Fecha'] >= cutoff_date].copy()
    
    # 3. Entrenar modelo una vez
    forecaster = DemandForecaster()
    print(f"Entrenando modelo con histórico hasta {cutoff_date - pd.Timedelta(days=1)}...")
    forecaster.train(df_train)
    
    # 4. Predecir 30 días adelante (multistep)
    days_to_predict = (df_ts['Fecha'].max() - cutoff_date).days + 1
    print(f"Generando proyección multi-paso para {days_to_predict} días...")
    df_pred = forecaster.predict_future(df_train, days=days_to_predict)
    
    # 5. Consolidar resultados para la Zona Sur (Zona Suroccidente )
    target_zone = 'Zona Suroccidente '
    
    real_monthly = df_test[df_test['Zona'] == target_zone].groupby('Fecha')['Tickets_Total'].sum().reset_index()
    pred_monthly = df_pred[df_pred['Zona'] == target_zone].groupby('Fecha')['Prediccion_Tickets'].sum().reset_index()
    
    comparativa = pd.merge(real_monthly, pred_monthly, on='Fecha', how='inner')
    comparativa.rename(columns={'Tickets_Total': 'Real', 'Prediccion_Tickets': 'Prediccion'}, inplace=True)
    
    # 6. Generar Tabla y Gráfico
    print("\n--- COMPARATIVA MENSUAL ZONA SUR (12 ABRIL - 11 MAYO) ---")
    print(comparativa.to_string(index=False))
    
    plt.figure(figsize=(12, 6))
    plt.plot(comparativa['Fecha'], comparativa['Real'], label='Real', marker='o', color='#1f77b4', linewidth=2)
    plt.plot(comparativa['Fecha'], comparativa['Prediccion'], label='Predicción', marker='x', linestyle='--', color='#d62728', linewidth=2)
    
    plt.title('VALIDACIÓN MENSUAL: DEMANDA ZONA SUR (ABRIL - MAYO 2026)', fontsize=14, fontweight='bold')
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Cantidad de Tickets', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_path = PLOTS_DIR / "validacion_mensual_sur.png"
    plt.savefig(plot_path)
    print(f"\nGráfico guardado en: {plot_path}")

if __name__ == "__main__":
    monthly_validation()
