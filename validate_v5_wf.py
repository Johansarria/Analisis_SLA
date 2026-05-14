"""
Validacion V5 - Walk-Forward Cross Validation con multiples ventanas.
Compara V4 (directo) vs V5 (iterativo + target-encoding + triple-quantile).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging, time
import sys

# Agregar raiz al path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.core.forecaster_v4 import DemandForecasterV4
from src.core.forecaster_v5 import DemandForecasterV5
from src.config import PROCESSED_DATA_DIR, PLOTS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

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


def evaluate_preds(df_real, df_pred, zona_label):
    """Calcula metricas por zona."""
    real = df_real.groupby('Fecha')['Tickets_Total'].sum().reset_index()
    pred = df_pred.groupby('Fecha')['Prediccion_Tickets'].sum().reset_index()
    c = pd.merge(real, pred, on='Fecha', how='inner')
    if len(c) == 0:
        return None
    mae = np.abs(c['Tickets_Total'] - c['Prediccion_Tickets']).mean()
    mape = (np.abs(c['Tickets_Total'] - c['Prediccion_Tickets']) / c['Tickets_Total'].replace(0, np.nan)).mean() * 100
    rmse = np.sqrt(((c['Tickets_Total'] - c['Prediccion_Tickets'])**2).mean())
    return {'Zona': zona_label, 'MAE': round(mae, 2), 'MAPE': f"{mape:.1f}%", 'RMSE': round(rmse, 2)}


def walk_forward_validate(df, n_splits=3, days_ahead=14, n_trials=15):
    """
    Realiza walk-forward validation con n_splits ventanas temporales.
    Cada ventana: entrena en train, predice 'days_ahead' dias del inicio de test.
    """
    # Crear cortes temporales
    n = len(df['Fecha'].unique())
    unique_dates = sorted(df['Fecha'].unique())
    # Necesitamos al menos ~60 dias de train + days_ahead de test por ventana
    min_train_size = 60
    if n < min_train_size + n_splits * days_ahead:
        logger.warning(f"Datos insuficientes para {n_splits} splits. Reduciendo a 1 split.")
        n_splits = 1

    step = max(1, (n - min_train_size - days_ahead) // n_splits)

    all_results_v4 = []
    all_results_v5 = []

    for split_idx in range(n_splits):
        train_end_idx = min_train_size + split_idx * step
        test_start_idx = train_end_idx + 1
        test_end_idx = test_start_idx + days_ahead - 1

        if test_end_idx >= n:
            test_end_idx = n - 1

        train_cutoff = unique_dates[train_end_idx]
        test_start = unique_dates[test_start_idx]
        test_end = unique_dates[test_end_idx]

        print(f"\n{'='*70}")
        print(f"  SPLIT {split_idx+1}/{n_splits}")
        print(f"  Train: {unique_dates[0].strftime('%Y-%m-%d')} -> {train_cutoff.strftime('%Y-%m-%d')}")
        print(f"  Test:  {test_start.strftime('%Y-%m-%d')} -> {test_end.strftime('%Y-%m-%d')}")
        print(f"{'='*70}")

        df_train = df[df['Fecha'] <= train_cutoff].copy()
        df_test = df[(df['Fecha'] >= test_start) & (df['Fecha'] <= test_end)].copy()

        # ---- V4 (directo) ----
        print("  [V4] Entrenando...")
        t0 = time.time()
        fc4 = DemandForecasterV4(quantile_alpha=0.6, ensemble_xgb_w=0.6, dow_damping=0.4)
        try:
            fc4.train(df_train, n_trials=n_trials)
            pred4 = fc4.predict_future(df_train, days=days_ahead)
            t4 = time.time() - t0
            print(f"  [V4] Entrenamiento+prediccion: {t4:.1f}s")
        except Exception as e:
            logger.error(f"V4 fallo en split {split_idx+1}: {e}")
            pred4 = pd.DataFrame()

        # ---- V5 (iterativo) ----
        print("  [V5] Entrenando...")
        t0 = time.time()
        fc5 = DemandForecasterV5(ensemble_xgb_w=0.6, target_smooth=10.0)
        try:
            fc5.train(df_train, n_trials=n_trials)
            pred5 = fc5.predict_future(df_train, days=days_ahead)
            t5 = time.time() - t0
            print(f"  [V5] Entrenamiento+prediccion: {t5:.1f}s")
        except Exception as e:
            logger.error(f"V5 fallo en split {split_idx+1}: {e}")
            pred5 = pd.DataFrame()

        # ---- Metricas por zona ----
        for zn, s in ZONES.items():
            real_zn = df_test[df_test['Zona'] == zn]
            if len(real_zn) == 0:
                continue

            if not pred4.empty:
                m4 = evaluate_preds(real_zn, pred4[pred4['Zona'] == zn], s['l'])
                if m4:
                    m4['Split'] = split_idx + 1
                    m4['Modelo'] = 'V4'
                    all_results_v4.append(m4)

            if not pred5.empty:
                m5 = evaluate_preds(real_zn, pred5[pred5['Zona'] == zn], s['l'])
                if m5:
                    m5['Split'] = split_idx + 1
                    m5['Modelo'] = 'V5'
                    all_results_v5.append(m5)

        # ---- Grafico comparativo por split ----
        if not pred4.empty and not pred5.empty:
            _plot_split_comparison(df_test, pred4, pred5, split_idx)

    # ---- Resumen agregado ----
    print("\n" + "="*70)
    print("  RESUMEN WALK-FORWARD VALIDATION")
    print("="*70)

    df_v4 = pd.DataFrame(all_results_v4)
    df_v5 = pd.DataFrame(all_results_v5)

    if not df_v4.empty:
        print("\n--- V4 (Directo) ---")
        summary4 = df_v4.groupby('Zona')[['MAE', 'RMSE']].mean().round(2)
        print(summary4.to_string())

    if not df_v5.empty:
        print("\n--- V5 (Iterativo + TargetEnc + TripleQ) ---")
        summary5 = df_v5.groupby('Zona')[['MAE', 'RMSE']].mean().round(2)
        print(summary5.to_string())

    if not df_v4.empty and not df_v5.empty:
        print("\n--- Diferencia V5 - V4 (negativo = mejora) ---")
        merged = pd.merge(
            df_v4.groupby('Zona')[['MAE', 'RMSE']].mean().reset_index(),
            df_v5.groupby('Zona')[['MAE', 'RMSE']].mean().reset_index(),
            on='Zona', suffixes=('_V4', '_V5')
        )
        merged['MAE_diff'] = merged['MAE_V5'] - merged['MAE_V4']
        merged['RMSE_diff'] = merged['RMSE_V5'] - merged['RMSE_V4']
        print(merged[['Zona', 'MAE_diff', 'RMSE_diff']].to_string(index=False))

    # Guardar CSVs
    from src.config import REPORTS_DIR
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not df_v4.empty:
        df_v4.to_csv(REPORTS_DIR / "walkforward_v4.csv", index=False)
    if not df_v5.empty:
        df_v5.to_csv(REPORTS_DIR / "walkforward_v5.csv", index=False)
    print(f"\n  Reportes guardados en {REPORTS_DIR}")


def _plot_split_comparison(df_test, pred4, pred5, split_idx):
    """Genera grafico comparativo por split para Zona Centro."""
    zn = 'Zona Centro'
    if zn not in ZONES:
        return
    real = df_test[df_test['Zona'] == zn].groupby('Fecha')['Tickets_Total'].sum().reset_index()
    p4 = pred4[pred4['Zona'] == zn].groupby('Fecha')['Prediccion_Tickets'].sum().reset_index()
    p5 = pred5[pred5['Zona'] == zn].groupby('Fecha')['Prediccion_Tickets'].sum().reset_index()

    c = real.merge(p4, on='Fecha', how='outer').merge(p5, on='Fecha', how='outer')
    c = c.sort_values('Fecha').fillna(0)
    c.rename(columns={'Prediccion_Tickets_x': 'V4', 'Prediccion_Tickets_y': 'V5'}, inplace=True)

    if len(c) == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(c['Fecha'], c['Tickets_Total'], 'k-o', label='Real', linewidth=2, markersize=6)
    ax.plot(c['Fecha'], c['V4'], 'b--s', label='V4 Directo', alpha=0.7)
    ax.plot(c['Fecha'], c['V5'], 'g-^', label='V5 Iterativo', alpha=0.7)
    ax.set_title(f"Walk-Forward Split {split_idx+1} - {zn}", fontsize=13, fontweight='bold')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Tickets')
    ax.legend()
    ax.grid(ls='--', alpha=0.4)
    plt.xticks(rotation=45)
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / f"wf_split{split_idx+1}_zcen.png", dpi=150)
    plt.close()
    print(f"  [OK] Grafico guardado: wf_split{split_idx+1}_zcen.png")


def run_single_holdout(df, cutoff_str='2026-04-12', days=None, n_trials=50):
    """Ejecuta un unico holdout (como V4) pero comparando V4 vs V5."""
    cutoff = pd.to_datetime(cutoff_str)
    df_train = df[df['Fecha'] < cutoff].copy()
    df_test = df[df['Fecha'] >= cutoff].copy()
    if days is None:
        days = (df['Fecha'].max() - cutoff).days + 1

    print("\n" + "="*70)
    print("  HOLDOUT VALIDATION (unico corte)")
    print(f"  Train: hasta {cutoff_str}  |  Test: {days} dias")
    print("="*70)

    # V4
    print("\n[V4] Entrenando...")
    fc4 = DemandForecasterV4(quantile_alpha=0.6, ensemble_xgb_w=0.6, dow_damping=0.4)
    fc4.train(df_train, n_trials=n_trials)
    pred4 = fc4.predict_future(df_train, days=days)

    # V5
    print("\n[V5] Entrenando...")
    fc5 = DemandForecasterV5(ensemble_xgb_w=0.6, target_smooth=10.0)
    fc5.train(df_train, n_trials=n_trials)
    pred5 = fc5.predict_future(df_train, days=days)
    pred5_direct = fc5.predict_future_direct(df_train, days=days)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = []

    for zn, s in ZONES.items():
        sfx = SUFFIXES[zn]
        real = df_test[df_test['Zona'] == zn].groupby('Fecha')['Tickets_Total'].sum().reset_index()

        for label, pred_df in [('V4', pred4), ('V5_Iter', pred5), ('V5_Direct', pred5_direct)]:
            p = pred_df[pred_df['Zona'] == zn].groupby('Fecha')['Prediccion_Tickets'].sum().reset_index()
            c = pd.merge(real, p, on='Fecha', how='inner')
            if len(c) == 0:
                continue
            mae = np.abs(c['Tickets_Total'] - c['Prediccion_Tickets']).mean()
            mape = (np.abs(c['Tickets_Total'] - c['Prediccion_Tickets']) / c['Tickets_Total'].replace(0, np.nan)).mean() * 100
            rmse = np.sqrt(((c['Tickets_Total'] - c['Prediccion_Tickets'])**2).mean())
            metrics.append({
                'Zona': s['l'], 'Modelo': label,
                'MAE': round(mae, 2), 'MAPE': f"{mape:.1f}%", 'RMSE': round(rmse, 2)
            })

        # Grafico comparativo
        p4 = pred4[pred4['Zona'] == zn].groupby('Fecha')['Prediccion_Tickets'].sum().reset_index()
        p5 = pred5[pred5['Zona'] == zn].groupby('Fecha')['Prediccion_Tickets'].sum().reset_index()
        c = pd.merge(real, p4, on='Fecha', how='inner').merge(p5, on='Fecha', how='inner')
        c.rename(columns={'Prediccion_Tickets_x': 'V4', 'Prediccion_Tickets_y': 'V5'}, inplace=True)
        if len(c) == 0:
            continue

        x = np.arange(len(c))
        w = 0.25
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.bar(x - w, c['Tickets_Total'], w, label='Real', color=s['cr'], alpha=0.85)
        ax.bar(x, c['V4'], w, label='V4 Directo', color='blue', alpha=0.6)
        ax.bar(x + w, c['V5'], w, label='V5 Iterativo', color='green', alpha=0.6)
        ax.set_title(f"REAL vs V4 vs V5 ({s['l']})", fontsize=14, fontweight='bold')
        ax.set_xlabel('Dia')
        ax.set_ylabel('Tickets')
        ax.set_xticks(x)
        ax.set_xticklabels(c['Fecha'].dt.strftime('%d-%b'), rotation=45, fontsize=9)
        ax.legend(fontsize=11)
        ax.grid(axis='y', ls='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"barras_v5_{sfx}.png", dpi=150)
        plt.close()
        print(f"  [OK] {s['l']} -> barras_v5_{sfx}.png")

    print("\n" + "="*70)
    print("  METRICAS HOLDOUT")
    print("="*70)
    df_metrics = pd.DataFrame(metrics)
    print(df_metrics.to_string(index=False))

    # Guardar
    from src.config import REPORTS_DIR
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(REPORTS_DIR / "holdout_v4_vs_v5.csv", index=False)


def main():
    # Cargar datos
    demand_path = PROCESSED_DATA_DIR / "demanda_diaria.csv"
    if not demand_path.exists():
        # Fallback: generar desde el ETL
        logger.info("demanda_diaria.csv no encontrado. Generando desde Jira...")
        from src.data.etl import DataProcessor
        processor = DataProcessor()
        raw_path = PROCESSED_DATA_DIR / "Jira ATP resumen 2 años.csv"
        if not raw_path.exists():
            raw_path = PROCESSED_DATA_DIR.parent / "raw" / "JiraATP.csv"
        df_ts = processor.process_demand_series(str(raw_path))
        processor.save_processed_data(df_ts, str(demand_path))
        logger.info(f"demanda_diaria.csv generado con {len(df_ts)} registros")

    df = pd.read_csv(demand_path)
    df['Fecha'] = pd.to_datetime(df['Fecha'])

    # 1) Holdout unico (rapido)
    run_single_holdout(df, cutoff_str='2026-04-12', n_trials=5)

    # 2) Walk-forward (mas robusto, 2 splits para no tardar mucho)
    # Descomentar para ejecutar:
    # walk_forward_validate(df, n_splits=2, days_ahead=14, n_trials=15)


if __name__ == "__main__":
    main()
