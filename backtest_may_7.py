import pandas as pd
from src.core.forecaster import DemandForecaster
from src.config import PROCESSED_DATA_DIR
import logging

logging.basicConfig(level=logging.WARNING)

def backtest_may_7():
    # 1. Cargar datos
    df_ts = pd.read_csv(PROCESSED_DATA_DIR / "demanda_diaria.csv")
    df_ts['Fecha'] = pd.to_datetime(df_ts['Fecha'])
    
    # 2. Preparar entrenamiento (hasta el 6 de mayo para predecir el 7)
    train_end_date = pd.to_datetime('2026-05-06')
    df_train = df_ts[df_ts['Fecha'] <= train_end_date].copy()
    
    # 3. Entrenar modelo
    forecaster = DemandForecaster()
    print("Entrenando modelo con datos hasta el 6 de Mayo...")
    forecaster.train(df_train)
    
    # 4. Predecir 1 día (el 7 de mayo)
    print("Generando predicción para el 7 de Mayo...")
    df_pred = forecaster.predict_future(df_train, days=1)
    
    # 5. Obtener valores reales para comparar
    target_date = pd.to_datetime('2026-05-07')
    target_zone = 'Zona Suroccidente '
    
    real_values = df_ts[(df_ts['Fecha'] == target_date) & (df_ts['Zona'] == target_zone)].copy()
    pred_values = df_pred[(df_pred['Fecha'] == target_date) & (df_pred['Zona'] == target_zone)].copy()
    
    # Unir resultados
    comparativa = pd.merge(
        real_values[['Nodo', 'Tickets_Total']], 
        pred_values[['Nodo', 'Prediccion_Tickets']], 
        on='Nodo'
    )
    comparativa.rename(columns={'Tickets_Total': 'Real', 'Prediccion_Tickets': 'Prediccion'}, inplace=True)
    comparativa['Diferencia'] = comparativa['Prediccion'] - comparativa['Real']
    
    print("\n--- COMPARATIVA ZONA SUR: 7 DE MAYO (PASADO) ---")
    print("="*60)
    print(comparativa.to_string(index=False))
    print("="*60)
    print(f"TOTAL REAL: {comparativa['Real'].sum()}")
    print(f"TOTAL PREDICCIÓN: {comparativa['Prediccion'].sum()}")
    print(f"Error Total: {abs(comparativa['Prediccion'].sum() - comparativa['Real'].sum())}")

if __name__ == "__main__":
    backtest_may_7()
