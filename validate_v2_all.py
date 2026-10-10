"""
Validación V2 — Genera gráficos comparativos Real vs Predicción (V2)
para las 5 zonas: Centro, Sur, Noroccidente, Á. Metropolitana, Preventivos.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.core.forecaster_v2 import DemandForecasterV2
from src.config import PROCESSED_DATA_DIR, PLOTS_DIR
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

ZONES = {
    'Zona Centro':              {'color_real': '#9b59b6', 'color_pred': '#2ecc71', 'label': 'ZONA CENTRO'},
    'Zona Suroccidente ':       {'color_real': '#3498db', 'color_pred': '#e74c3c', 'label': 'ZONA SUR'},
    'Zona Area Metropolitana':  {'color_real': '#e67e22', 'color_pred': '#34495e', 'label': 'ZONA AREA METROPOLITANA'},
    'Zona Noroccidente':        {'color_real': '#2ecc71', 'color_pred': '#f39c12', 'label': 'ZONA NOROCCIDENTE'},
    'Mantenimiento Preventivo': {'color_real': '#f1c40f', 'color_pred': '#27ae60', 'label': 'MANTENIMIENTO PREVENTIVO'},
}

FILE_SUFFIXES = {
    'Zona Centro': 'zcen',
    'Zona Suroccidente ': 'zsur',
    'Zona Area Metropolitana': 'zamp',
    'Zona Noroccidente': 'znor',
    'Mantenimiento Preventivo': 'prev',
}


def run_validation():
    # 1. Cargar datos
    df_ts = pd.read_csv(PROCESSED_DATA_DIR / "demanda_diaria.csv")
    df_ts['Fecha'] = pd.to_datetime(df_ts['Fecha'])

    # 2. Split temporal
    cutoff_date = pd.to_datetime('2026-04-12')
    df_train = df_ts[df_ts['Fecha'] < cutoff_date].copy()
    df_test = df_ts[df_ts['Fecha'] >= cutoff_date].copy()
    days_to_predict = (df_ts['Fecha'].max() - cutoff_date).days + 1

    # 3. Entrenar V2 con 50 trials por zona
    print("\n" + "=" * 60)
    print("  ENTRENANDO MODELO V2 (4 mejoras aplicadas)")
    print("=" * 60)
    t0 = time.time()

    forecaster = DemandForecasterV2(quantile_alpha=0.6)
    forecaster.train(df_train, n_trials=50)

    print(f"\n  Entrenamiento completado en {time.time() - t0:.1f}s")

    # 4. Predecir
    print("  Generando predicciones directas...")
    df_pred = forecaster.predict_future(df_train, days=days_to_predict)

    # 5. Generar gráficos por zona
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_summary = []

    for zone_name, style in ZONES.items():
        suffix = FILE_SUFFIXES[zone_name]

        real = df_test[df_test['Zona'] == zone_name].groupby('Fecha')['Tickets_Total'].sum().reset_index()
        pred = df_pred[df_pred['Zona'] == zone_name].groupby('Fecha')['Prediccion_Tickets'].sum().reset_index()

        comp = pd.merge(real, pred, on='Fecha', how='inner')
        comp.rename(columns={'Tickets_Total': 'Real', 'Prediccion_Tickets': 'Prediccion'}, inplace=True)
        comp['Fecha_Str'] = comp['Fecha'].dt.strftime('%d-%b')

        if len(comp) == 0:
            print(f"  WARN: Sin datos para {zone_name}")
            continue

        # Métricas
        mae = np.abs(comp['Real'] - comp['Prediccion']).mean()
        mape = (np.abs(comp['Real'] - comp['Prediccion']) / comp['Real'].replace(0, np.nan)).mean() * 100
        rmse = np.sqrt(((comp['Real'] - comp['Prediccion']) ** 2).mean())
        metrics_summary.append({
            'Zona': style['label'], 'MAE': mae, 'MAPE': f"{mape:.1f}%", 'RMSE': rmse
        })

        # Gráfico
        x = np.arange(len(comp))
        width = 0.35

        fig, ax = plt.subplots(figsize=(16, 7))
        bars_r = ax.bar(x - width / 2, comp['Real'], width,
                        label='Real', color=style['color_real'], alpha=0.85)
        bars_p = ax.bar(x + width / 2, comp['Prediccion'], width,
                        label='Predicción V2', color=style['color_pred'], alpha=0.85)

        ax.set_title(
            f"COMPARATIVA DIARIA: REAL VS PREDICCIÓN V2 ({style['label']})",
            fontsize=15, fontweight='bold'
        )
        ax.set_xlabel('Día', fontsize=12)
        ax.set_ylabel('Cantidad de Tickets', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(comp['Fecha_Str'], rotation=45, fontsize=9)
        ax.legend(fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        # Valores sobre barras
        for i, v in enumerate(comp['Real']):
            ax.text(i - width / 2, v + 0.3, str(int(v)), ha='center', fontsize=8)
        for i, v in enumerate(comp['Prediccion']):
            ax.text(i + width / 2, v + 0.3, str(int(v)), ha='center', fontsize=8)

        # Anotar MAE en el gráfico
        ax.text(0.99, 0.97, f"MAE={mae:.1f}  MAPE={mape:.1f}%",
                transform=ax.transAxes, ha='right', va='top',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plot_path = PLOTS_DIR / f"barras_v2_{suffix}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  [OK] {style['label']} -> {plot_path.name}")

    # 6. Resumen de métricas
    print("\n" + "=" * 60)
    print("  RESUMEN DE MÉTRICAS V2")
    print("=" * 60)
    df_metrics = pd.DataFrame(metrics_summary)
    print(df_metrics.to_string(index=False))
    print()


if __name__ == "__main__":
    run_validation()
