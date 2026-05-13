"""Genera 3 graficos independientes para los ultimos 3 meses de Zona Centro."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.core.forecaster_v4 import DemandForecasterV4
from src.config import PROCESSED_DATA_DIR, PLOTS_DIR
import logging

logging.basicConfig(level=logging.INFO)

def plot_period(df_real, df_pred, title, filename):
    c = pd.merge(df_real, df_pred, on='Fecha', how='inner')
    c.rename(columns={'Tickets_Total': 'Real', 'Prediccion_Tickets': 'Pred'}, inplace=True)
    c['Lbl'] = c['Fecha'].dt.strftime('%d-%b')
    
    if len(c) == 0:
        print(f"Sin datos para {title}")
        return

    mae = np.abs(c['Real'] - c['Pred']).mean()
    
    x = np.arange(len(c))
    w = 0.4
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(x - w/2, c['Real'], w, label='Real', color='#9b59b6', alpha=0.8)
    ax.bar(x + w/2, c['Pred'], w, label='Prediccion V4', color='#2ecc71', alpha=0.8)
    
    ax.set_title(f"{title} - ZONA CENTRO", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(c['Lbl'], rotation=45, fontsize=8)
    ax.legend()
    ax.grid(axis='y', ls='--', alpha=0.3)
    
    # Text labels
    for i, v in enumerate(c['Real']):
        ax.text(i - w/2, v + 0.2, str(int(v)), ha='center', fontsize=7)
    for i, v in enumerate(c['Pred']):
        ax.text(i + w/2, v + 0.2, str(int(v)), ha='center', fontsize=7)
        
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=120)
    plt.close()
    print(f"Creado: {filename} (MAE={mae:.1f})")

def run():
    df = pd.read_csv(PROCESSED_DATA_DIR / "demanda_diaria.csv")
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df_zcen = df[df['Zona'] == 'Zona Centro'].copy()
    
    # Cargar modelo V4
    fc = DemandForecasterV4()
    fc.load_model()
    # Cargar clima para todo el rango
    fc.weather_df = pd.read_csv(PROCESSED_DATA_DIR / "clima_bogota.csv")
    fc.weather_df['Fecha'] = pd.to_datetime(fc.weather_df['Fecha'])

    # Para mostrar los 3 meses, necesitamos "predicciones" para todo el rango.
    # Como el modelo es Directo (Horizonte 1-30), simularemos una prediccion 
    # de ventana rodante o simplemente usaremos el modelo para inferir sobre el pasado.
    # Para simplicidad y visualizacion, usaremos el metodo predict_future con cutoffs mensuales.
    
    # Mes 1: Feb-Mar
    c1 = pd.to_datetime('2026-02-25') # Necesitamos algo de historia previa
    p1 = fc.predict_future(df[df['Fecha'] < c1], days=20)
    
    # Mes 2: Mar-Abr
    c2 = pd.to_datetime('2026-03-15')
    p2 = fc.predict_future(df[df['Fecha'] < c2], days=30)
    
    # Mes 3: Abr-May (El que ya tenemos)
    c3 = pd.to_datetime('2026-04-12')
    p3 = fc.predict_future(df[df['Fecha'] < c3], days=30)
    
    # Filtrar reales y graficar
    real_zcen = df_zcen.groupby('Fecha')['Tickets_Total'].sum().reset_index()
    
    p1_zcen = p1[p1['Zona'] == 'Zona Centro']
    p2_zcen = p2[p2['Zona'] == 'Zona Centro']
    p3_zcen = p3[p3['Zona'] == 'Zona Centro']
    
    plot_period(real_zcen, p1_zcen.groupby('Fecha')['Prediccion_Tickets'].sum().reset_index(), 
                "MES 1 (Feb-Mar)", "zcen_mes1.png")
    plot_period(real_zcen, p2_zcen.groupby('Fecha')['Prediccion_Tickets'].sum().reset_index(), 
                "MES 2 (Mar-Abr)", "zcen_mes2.png")
    plot_period(real_zcen, p3_zcen.groupby('Fecha')['Prediccion_Tickets'].sum().reset_index(), 
                "MES 3 (Abr-May)", "zcen_mes3.png")

if __name__ == "__main__":
    run()
