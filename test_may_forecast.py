import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.core.forecaster import DemandForecaster
from src.config import PROCESSED_DATA_DIR, PLOTS_DIR
from sklearn.metrics import mean_absolute_error

def test_may_forecast():
    print("Iniciando validación temporal del mes de Mayo...")
    
    # Cargar datos procesados (hasta el 11 de Mayo)
    df_ts = pd.read_csv(PROCESSED_DATA_DIR / "demanda_diaria.csv")
    df_ts['Fecha'] = pd.to_datetime(df_ts['Fecha'])
    
    # Partir los datos: Entrenar hasta Abril, Validar en Mayo
    cutoff_date = pd.to_datetime('2026-05-01')
    
    df_train = df_ts[df_ts['Fecha'] < cutoff_date].copy()
    df_test = df_ts[df_ts['Fecha'] >= cutoff_date].copy()
    
    print(f"Días de entrenamiento (Hasta Abril): {len(df_train)}")
    print(f"Días de validación (Mayo a la fecha): {len(df_test)}")
    
    if len(df_test) == 0:
        print("No hay datos de Mayo para validar.")
        return
        
    # Entrenar modelo solo con datos hasta Abril
    forecaster = DemandForecaster()
    print("Entrenando modelo con histórico hasta Abril...")
    forecaster.train(df_train)
    
    # Predecir los días equivalentes a Mayo
    print(f"Prediciendo {len(df_test)} días hacia el futuro...")
    df_pred = forecaster.predict_future(df_train, days=len(df_test))
    
    # Unir real vs predicho
    resultados = pd.merge(df_test[['Fecha', 'Tickets_Total']], df_pred, on='Fecha', how='inner')
    resultados.rename(columns={'Tickets_Total': 'Real', 'Prediccion_Tickets': 'Proyeccion'}, inplace=True)
    
    # Calcular métricas
    mae = mean_absolute_error(resultados['Real'], resultados['Proyeccion'])
    
    print("\n--- RESULTADOS DE MAYO ---")
    print(resultados.to_string(index=False))
    print(f"\nMAE en Mayo: {mae:.2f} tickets/día")
    print(f"Demanda Total Acumulada Real (Mayo): {resultados['Real'].sum()}")
    print(f"Demanda Total Acumulada Predicha (Mayo): {resultados['Proyeccion'].sum()}")
    
    # Visualizar
    plt.figure(figsize=(10, 5))
    plt.plot(resultados['Fecha'], resultados['Real'], marker='o', label='Real')
    plt.plot(resultados['Fecha'], resultados['Proyeccion'], marker='x', linestyle='--', label='Modelo', color='red')
    plt.title('VALIDACIÓN DE DEMANDA: MAYO (REAL VS MODELO)')
    plt.xlabel('Fecha')
    plt.ylabel('Cantidad de Tickets')
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plot_path = PLOTS_DIR / "validacion_mayo.png"
    plt.savefig(plot_path)
    print(f"\nGráfico guardado en {plot_path}")

if __name__ == "__main__":
    import logging
    logging.getLogger("src.core.forecaster").setLevel(logging.WARNING)
    test_may_forecast()
