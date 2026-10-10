import pandas as pd
import numpy as np
from src.core.forecaster import DemandForecaster
from src.config import PROCESSED_DATA_DIR
import logging
import random

logging.basicConfig(level=logging.WARNING)

def backtest_random_days():
    # 1. Cargar datos
    df_ts = pd.read_csv(PROCESSED_DATA_DIR / "demanda_diaria.csv")
    df_ts['Fecha'] = pd.to_datetime(df_ts['Fecha'])
    
    # 2. Definir rango del último mes (aprox)
    max_date = df_ts['Fecha'].max()
    min_date = max_date - pd.Timedelta(days=30)
    
    # Asegurar que hay suficientes días para rezagos (Lags)
    # El modelo usa Lag_7 y Media_Movil_14d, así que necesitamos al menos 14 días de historia antes de la predicción
    available_dates = df_ts[(df_ts['Fecha'] >= min_date) & (df_ts['Fecha'] <= max_date)]['Fecha'].unique()
    
    if len(available_dates) < 3:
        print("No hay suficientes datos para elegir 3 días aleatorios.")
        return
        
    random_days = sorted(random.sample(list(available_dates), 3))
    
    target_zone = 'Zona Suroccidente '
    results_summary = []

    for day in random_days:
        day = pd.to_datetime(day)
        print(f"\n--- Probando para el día: {day.strftime('%Y-%m-%d')} ---")
        
        # Train con datos hasta day - 1
        df_train = df_ts[df_ts['Fecha'] < day].copy()
        
        forecaster = DemandForecaster()
        forecaster.train(df_train)
        
        # Predecir el día
        df_pred = forecaster.predict_future(df_train, days=1)
        
        # Comparar
        real_values = df_ts[(df_ts['Fecha'] == day) & (df_ts['Zona'] == target_zone)]
        pred_values = df_pred[(df_pred['Fecha'] == day) & (df_pred['Zona'] == target_zone)]
        
        comp = pd.merge(real_values[['Nodo', 'Tickets_Total']], pred_values[['Nodo', 'Prediccion_Tickets']], on='Nodo')
        
        total_real = comp['Tickets_Total'].sum()
        total_pred = comp['Prediccion_Tickets'].sum()
        
        results_summary.append({
            'Fecha': day.strftime('%Y-%m-%d'),
            'Dia_Semana': day.day_name(),
            'Real': total_real,
            'Prediccion': total_pred,
            'Diferencia': total_pred - total_real
        })
        
        print(f"Resultado: Real={total_real}, Prediccion={total_pred}, Dif={total_pred - total_real}")

    print("\n" + "="*60)
    print("RESUMEN DE 3 DÍAS ALEATORIOS (ZONA SUR)")
    print("="*60)
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    backtest_random_days()
