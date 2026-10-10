import pandas as pd
from src.core.forecaster import DemandForecaster
from src.config import PROCESSED_DATA_DIR
import logging

logging.basicConfig(level=logging.WARNING)

def validate_optimized_zcen():
    # 1. Cargar datos
    df_ts = pd.read_csv(PROCESSED_DATA_DIR / "demanda_diaria.csv")
    df_ts['Fecha'] = pd.to_datetime(df_ts['Fecha'])
    
    # 2. Rango de prueba (Mayo)
    cutoff = pd.to_datetime('2026-05-01')
    df_train = df_ts[df_ts['Fecha'] < cutoff].copy()
    
    # 3. Entrenar con 50 trials para un mejor balance
    forecaster = DemandForecaster()
    print("Entrenando modelo optimizado (50 trials)...")
    forecaster.train(df_train, n_trials=50)
    
    # 4. Predecir Mayo (hasta el 11)
    df_pred = forecaster.predict_future(df_train, days=11)
    
    # 5. Comparar resultados para Zona Centro
    target_zone = 'Zona Centro'
    real = df_ts[(df_ts['Fecha'] >= cutoff) & (df_ts['Zona'] == target_zone)].groupby('Fecha')['Tickets_Total'].sum().reset_index()
    pred = df_pred[df_pred['Zona'] == target_zone].groupby('Fecha')['Prediccion_Tickets'].sum().reset_index()
    
    comp = pd.merge(real, pred, on='Fecha')
    comp.rename(columns={'Tickets_Total': 'Real', 'Prediccion_Tickets': 'Pred_Optimizada'}, inplace=True)
    comp['Error'] = comp['Pred_Optimizada'] - comp['Real']
    
    print("\n--- COMPARATIVA ZONA CENTRO (OPTIMIZADA CON 100 TRIALS) ---")
    print("="*70)
    print(comp.to_string(index=False))
    print("="*70)
    print(f"MAE Optimizada: {abs(comp['Error']).mean():.2f}")

if __name__ == "__main__":
    validate_optimized_zcen()
