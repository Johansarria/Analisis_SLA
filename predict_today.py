import pandas as pd
from src.core.forecaster import DemandForecaster
from src.config import PROCESSED_DATA_DIR
import logging

logging.basicConfig(level=logging.WARNING)

def predict_today():
    # Cargar datos procesados (hasta el 11 de Mayo)
    df_ts = pd.read_csv(PROCESSED_DATA_DIR / "demanda_diaria.csv")
    df_ts['Fecha'] = pd.to_datetime(df_ts['Fecha'])
    
    # Entrenar/Cargar el modelo (usaremos la clase DemandForecaster)
    forecaster = DemandForecaster()
    
    # Entrenamos con todos los datos disponibles (Optimizado con 50 trials)
    forecaster.train(df_ts, n_trials=50)
    
    # Predecir 2 días (12 y 13 de Mayo)
    df_pred = forecaster.predict_future(df_ts, days=2)
    
    # Filtrar para hoy miércoles 13 de mayo
    today = '2026-05-13'
    target_zone = 'Zona Suroccidente '
    
    result = df_pred[(df_pred['Fecha'] == today) & (df_pred['Zona'] == target_zone)]
    
    print("\nProyecciones para la Zona Suroccidente (Sur) - Miércoles 13 de Mayo:")
    print("="*60)
    print(result[['Nodo', 'Prediccion_Tickets']].to_string(index=False))
    print("="*60)
    print(f"Total proyectado para la zona: {result['Prediccion_Tickets'].sum()}")

if __name__ == "__main__":
    predict_today()
