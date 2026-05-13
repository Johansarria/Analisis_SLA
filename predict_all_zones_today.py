import pandas as pd
import numpy as np
from src.core.forecaster_v4 import DemandForecasterV4
from src.config import PROCESSED_DATA_DIR
import logging

# Configurar logging para silenciar info
logging.basicConfig(level=logging.WARNING)

def run():
    # Cargar datos históricos
    df = pd.read_csv(PROCESSED_DATA_DIR / "demanda_diaria.csv")
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Cargar el modelo V4 (con clima)
    fc = DemandForecasterV4()
    try:
        fc.load_model()
        # Cargar clima desde el caché (debería estar actualizado hasta hoy)
        fc.weather_df = pd.read_csv(PROCESSED_DATA_DIR / "clima_bogota.csv")
        fc.weather_df['Fecha'] = pd.to_datetime(fc.weather_df['Fecha'])
    except Exception as e:
        print(f"Error cargando modelo o clima: {e}")
        return

    # Hoy es 13 de mayo de 2026. Los datos en demanda_diaria llegan hasta el 11 de mayo?
    # Vamos a predecir para el día de hoy (13 de mayo)
    # predict_future calcula desde el último día en df + 1 hasta days.
    # Si df termina el 11, days=2 nos da el 13.
    
    last_date = df['Fecha'].max()
    today = pd.to_datetime('2026-05-13')
    days_to_predict = (today - last_date).days
    
    if days_to_predict <= 0:
        print(f"La fecha máxima en el dataset ({last_date.date()}) ya es igual o posterior a hoy ({today.date()}).")
        # Aun así podemos mostrar el dato si ya existe
        result = df[df['Fecha'] == today].groupby('Zona')['Tickets_Total'].sum().reset_index()
        print(f"\nResultados reales para hoy {today.date()}:")
    else:
        print(f"Prediciendo para hoy {today.date()} (Horizonte: {days_to_predict} días desde {last_date.date()})...")
        df_pred = fc.predict_future(df, days=days_to_predict)
        result = df_pred[df_pred['Fecha'] == today].groupby('Zona')['Prediccion_Tickets'].sum().reset_index()
        result.rename(columns={'Prediccion_Tickets': 'Tickets_Predichos'}, inplace=True)
        print(f"\nPredicciones para hoy {today.date()}:")

    print("="*60)
    print(result.to_string(index=False))
    print("="*60)
    print(f"Total Nacional Proyectado: {result.iloc[:,1].sum()}")

if __name__ == "__main__":
    run()
