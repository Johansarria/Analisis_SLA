"""Validacion V4 - Con features de clima (Open-Meteo)."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.core.forecaster_v4 import DemandForecasterV4
from src.config import PROCESSED_DATA_DIR, PLOTS_DIR
import logging, time

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

ZONES = {
    'Zona Centro':              {'cr': '#9b59b6', 'cp': '#2ecc71', 'l': 'ZONA CENTRO'},
    'Zona Suroccidente ':       {'cr': '#3498db', 'cp': '#e74c3c', 'l': 'ZONA SUR'},
    'Zona Area Metropolitana':  {'cr': '#e67e22', 'cp': '#34495e', 'l': 'ZONA AREA METROPOLITANA'},
    'Zona Noroccidente':        {'cr': '#2ecc71', 'cp': '#f39c12', 'l': 'ZONA NOROCCIDENTE'},
    'Mantenimiento Preventivo': {'cr': '#f1c40f', 'cp': '#27ae60', 'l': 'MANTENIMIENTO PREVENTIVO'},
}
SUFFIXES = {
    'Zona Centro': 'zcen', 'Zona Suroccidente ': 'zsur',
    'Zona Area Metropolitana': 'zamp', 'Zona Noroccidente': 'znor',
    'Mantenimiento Preventivo': 'prev',
}

def run():
    df = pd.read_csv(PROCESSED_DATA_DIR / "demanda_diaria.csv")
    df['Fecha'] = pd.to_datetime(df['Fecha'])

    cutoff = pd.to_datetime('2026-04-12')
    df_train = df[df['Fecha'] < cutoff].copy()
    df_test  = df[df['Fecha'] >= cutoff].copy()
    days = (df['Fecha'].max() - cutoff).days + 1

    print("\n" + "=" * 60)
    print("  ENTRENANDO MODELO V4 (mejoras 1-8 con clima)")
    print("=" * 60)
    t0 = time.time()
    fc = DemandForecasterV4(quantile_alpha=0.6, ensemble_xgb_w=0.6, dow_damping=0.4)
    fc.train(df_train, n_trials=50)
    print(f"\n  Entrenamiento: {time.time()-t0:.1f}s")

    print("  Prediciendo...")
    df_pred = fc.predict_future(df_train, days=days)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = []

    for zn, s in ZONES.items():
        sfx = SUFFIXES[zn]
        real = df_test[df_test['Zona'] == zn].groupby('Fecha')['Tickets_Total'].sum().reset_index()
        pred = df_pred[df_pred['Zona'] == zn].groupby('Fecha')['Prediccion_Tickets'].sum().reset_index()
        c = pd.merge(real, pred, on='Fecha', how='inner')
        c.rename(columns={'Tickets_Total': 'Real', 'Prediccion_Tickets': 'Pred'}, inplace=True)
        c['Lbl'] = c['Fecha'].dt.strftime('%d-%b')
        if len(c) == 0:
            continue

        mae  = np.abs(c['Real'] - c['Pred']).mean()
        mape = (np.abs(c['Real'] - c['Pred']) / c['Real'].replace(0, np.nan)).mean() * 100
        rmse = np.sqrt(((c['Real'] - c['Pred'])**2).mean())
        metrics.append({'Zona': s['l'], 'MAE': round(mae,2), 'MAPE': f"{mape:.1f}%", 'RMSE': round(rmse,2)})

        x = np.arange(len(c)); w = 0.35
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.bar(x - w/2, c['Real'], w, label='Real', color=s['cr'], alpha=0.85)
        ax.bar(x + w/2, c['Pred'], w, label='Prediccion V4', color=s['cp'], alpha=0.85)
        ax.set_title(f"REAL VS PREDICCION V4 ({s['l']})\n[Incluye datos climaticos Open-Meteo]",
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Dia'); ax.set_ylabel('Tickets')
        ax.set_xticks(x); ax.set_xticklabels(c['Lbl'], rotation=45, fontsize=9)
        ax.legend(fontsize=11); ax.grid(axis='y', ls='--', alpha=0.4)
        for i, v in enumerate(c['Real']):
            ax.text(i - w/2, v+0.3, str(int(v)), ha='center', fontsize=8)
        for i, v in enumerate(c['Pred']):
            ax.text(i + w/2, v+0.3, str(int(v)), ha='center', fontsize=8)
        ax.text(0.99, 0.97, f"MAE={mae:.1f}  MAPE={mape:.1f}%",
                transform=ax.transAxes, ha='right', va='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"barras_v4_{sfx}.png", dpi=150)
        plt.close()
        print(f"  [OK] {s['l']} -> barras_v4_{sfx}.png")

    print("\n" + "=" * 60)
    print("  METRICAS V4 (con clima)")
    print("=" * 60)
    print(pd.DataFrame(metrics).to_string(index=False))
    print()

if __name__ == "__main__":
    run()
