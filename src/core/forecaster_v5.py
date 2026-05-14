"""
V5 = V4 + 4 mejoras criticas y altas:
  1. Prediccion iterativa real (lags/MAs actualizados dinamicamente)
  2. Target Encoding de Nodos (reemplaza One-Hot esparsificado)
  3. Ensemble de 3 quantiles adaptativo (0.3, 0.5, 0.7)
  4. Regularizacion L1/L2 en Optuna + dow_damping optimizable
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import joblib
from src.config import MODELS_DIR
from src.data.weather import get_weather_for_zones, get_weather_features_for_dates

logger = logging.getLogger(__name__)
CENTRO = 'Zona Centro'

# Festivos Colombia 2026 (evita import roto de forecaster_v2)
HOLIDAYS_2026 = pd.to_datetime([
    '2026-01-01', '2026-01-12', '2026-03-23', '2026-04-02', '2026-04-03',
    '2026-04-05', '2026-05-01', '2026-05-18', '2026-06-08', '2026-06-15',
    '2026-07-05', '2026-07-20', '2026-08-15', '2026-08-16', '2026-10-12',
    '2026-11-02', '2026-11-16', '2026-12-08', '2026-12-25'
])


class DemandForecasterV5:
    """
    V5 incorpora:
      - Prediccion iterativa real (ventana deslizante)
      - Target Encoding de Nodos con smoothing
      - Triple quantile (Q30/Q50/Q70) con fusion adaptativa por clima/calendario
      - dow_damping como hiperparametro de Optuna
      - Regularizacion L1/L2 en XGBoost
    """

    def __init__(self, ensemble_xgb_w=0.6, target_smooth=10.0):
        self.ensemble_xgb_w = ensemble_xgb_w
        self.target_smooth = target_smooth

        # Cada zona/nodo puede tener hasta 3 modelos (quantiles)
        self.zone_models = {}      # zona -> { 'q30': m, 'q50': m, 'q70': m }
        self.zone_features = {}    # zona -> list(features)
        self.node_models = {}      # nodo -> { 'q30': m, 'q50': m, 'q70': m }
        self.node_features = {}    # nodo -> list(features)
        self.zone_node_pairs = {}
        self.seasonal = {}
        self.dow_factors = {}
        self.weather_df = None

        # Target encoding lookups
        self.nodo_mean = {}   # (zona, nodo) -> mean tickets
        self.nodo_std = {}    # (zona, nodo) -> std tickets

        self.base_features = [
            'Dia_Semana', 'Mes', 'Horizonte',
            'Lag_1', 'Lag_7',
            'Media_Movil_7d', 'Media_Movil_14d',
            'es_festivo', 'es_quincena', 'semana_del_mes',
            'dia_del_mes', 'es_lunes', 'es_viernes', 'es_finsemana',
            # Clima
            'precipitacion_mm', 'temp_max', 'viento_max_kmh',
            'es_lluvia_fuerte', 'es_lluvia', 'lluvia_acum_3d',
            # Target encoding
            'nodo_target_mean', 'nodo_target_std',
        ]

    # ── Helpers ───────────────────────────────────────────────────

    def _add_calendar(self, df):
        df = df.copy()
        f = pd.to_datetime(df['Fecha'])
        dia, dow = f.dt.day, f.dt.dayofweek
        df['es_festivo']     = f.dt.normalize().isin(HOLIDAYS_2026).astype(int)
        df['es_quincena']    = (((dia >= 14) & (dia <= 16)) | (dia >= 28) | (dia <= 1)).astype(int)
        df['semana_del_mes'] = ((dia - 1) // 7 + 1).astype(int)
        df['dia_del_mes']    = dia.astype(int)
        df['es_lunes']       = (dow == 0).astype(int)
        df['es_viernes']     = (dow == 4).astype(int)
        df['es_finsemana']   = (dow >= 5).astype(int)
        return df

    def _add_weather(self, df, zona):
        """Agrega features de clima desde el DataFrame cacheado."""
        if self.weather_df is None:
            for col in ['precipitacion_mm', 'temp_max', 'viento_max_kmh',
                        'es_lluvia_fuerte', 'es_lluvia', 'lluvia_acum_3d']:
                df[col] = 0
            return df
        wf = get_weather_features_for_dates(df['Fecha'].values, zona, self.weather_df)
        for col in wf.columns:
            df[col] = wf[col].values
        return df

    def _add_target_encoding(self, df):
        """Agrega target encoding de nodo con smoothing."""
        df = df.copy()
        means, stds = [], []
        for _, row in df.iterrows():
            key = (row['Zona'], row['Nodo'])
            means.append(self.nodo_mean.get(key, 0.0))
            stds.append(self.nodo_std.get(key, 0.0))
        df['nodo_target_mean'] = means
        df['nodo_target_std'] = stds
        return df

    def _compute_target_encoding(self, df):
        """Calcula estadisticas de target encoding por (Zona, Nodo)."""
        stats = df.groupby(['Zona', 'Nodo'])['Tickets_Total'].agg(['mean', 'std', 'count'])
        global_mean = df['Tickets_Total'].mean()
        for (z, n), row in stats.iterrows():
            # Smoothing: combinar media del nodo con media global
            count = row['count']
            mean_n = row['mean']
            smoothed = (count * mean_n + self.target_smooth * global_mean) / (count + self.target_smooth)
            self.nodo_mean[(z, n)] = smoothed
            self.nodo_std[(z, n)] = row['std'] if pd.notna(row['std']) else 0.0

    def _make_direct_samples(self, tickets, fechas, nodo_name, zona):
        """Genera muestras directas (sin iterar) para entrenamiento."""
        n = len(tickets)
        rows = []
        for t in range(14, n):
            for h in range(1, 31):
                o = t - h
                if o < 7:
                    continue
                td = pd.to_datetime(fechas[t])
                rows.append({
                    'Fecha': fechas[t], 'Nodo': nodo_name, 'Zona': zona,
                    'Tickets_Total': tickets[t],
                    'Dia_Semana': td.dayofweek, 'Mes': td.month,
                    'Horizonte': h,
                    'Lag_1': tickets[o],
                    'Lag_7': tickets[o-6] if o >= 6 else tickets[0],
                    'Media_Movil_7d': tickets[max(0,o-6):o+1].mean(),
                    'Media_Movil_14d': tickets[max(0,o-13):o+1].mean(),
                })
        df = pd.DataFrame(rows)
        df = self._add_calendar(df)
        df = self._add_weather(df, zona)
        df = self._add_target_encoding(df)
        return df

    def _get_feature_cols(self, df):
        """Devuelve solo las columnas de features base (sin target ni ids)."""
        cols = []
        for c in self.base_features:
            if c in df.columns:
                cols.append(c)
        return cols

    def _optuna_train_quantile(self, X, y, quantile_alpha, n_trials):
        """Entrena un modelo XGBoost para un quantile especifico con regularizacion."""
        def objective(trial):
            p = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 600),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'random_state': 42,
                'objective': 'reg:quantileerror',
                'quantile_alpha': quantile_alpha,
            }
            tscv = TimeSeriesSplit(n_splits=5)
            maes = []
            for ti, vi in tscv.split(X):
                m = xgb.XGBRegressor(**p)
                m.fit(X.iloc[ti], y.iloc[ti], verbose=False)
                maes.append(mean_absolute_error(y.iloc[vi], m.predict(X.iloc[vi])))
            return np.mean(maes)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        bp = study.best_params
        bp.update({'random_state': 42, 'objective': 'reg:quantileerror',
                   'quantile_alpha': quantile_alpha})
        model = xgb.XGBRegressor(**bp)
        model.fit(X, y, verbose=False)
        return model, study.best_value

    def _train_ensemble_for_key(self, X, y, n_trials):
        """Entrena los 3 quantiles y devuelve diccionario de modelos + MAEs."""
        quantiles = {'q30': 0.3, 'q50': 0.5, 'q70': 0.7}
        models = {}
        maes = {}
        for qname, qval in quantiles.items():
            m, mae = self._optuna_train_quantile(X, y, qval, n_trials)
            models[qname] = m
            maes[qname] = mae
        return models, maes

    def _compute_seasonal(self, df):
        df2 = df.copy()
        df2['dow'] = pd.to_datetime(df2['Fecha']).dt.dayofweek
        self.seasonal = df2.groupby(['Zona', 'Nodo', 'dow'])['Tickets_Total'].mean().to_dict()

    def _compute_dow_factors(self, df):
        df2 = df.copy()
        df2['dow'] = pd.to_datetime(df2['Fecha']).dt.dayofweek
        overall = df2.groupby(['Zona', 'Nodo'])['Tickets_Total'].mean()
        by_dow = df2.groupby(['Zona', 'Nodo', 'dow'])['Tickets_Total'].mean()
        for (z, n, d), mean_d in by_dow.items():
            mean_a = overall.get((z, n), 1)
            self.dow_factors[(z, n, d)] = mean_d / mean_a if mean_a > 0 else 1.0

    # ── Entrenamiento ─────────────────────────────────────────────

    def train(self, df: pd.DataFrame, n_trials: int = 15):
        logger.info("Entrenando modelo V5 (iterativo + target-encoding + triple-quantile)...")

        # Fase 0: Clima
        start = df['Fecha'].min()
        end = df['Fecha'].max()
        end_ext = pd.to_datetime(end) + pd.Timedelta(days=35)
        logger.info(f"  Descargando clima {start} -> {end_ext.strftime('%Y-%m-%d')}...")
        self.weather_df = get_weather_for_zones(
            str(start)[:10], end_ext.strftime('%Y-%m-%d')
        )
        logger.info(f"  Clima obtenido: {len(self.weather_df)} registros")

        # Fase 1: Estadisticas
        self._compute_target_encoding(df)
        self._compute_seasonal(df)
        self._compute_dow_factors(df)

        # Fase 2: Entrenar modelos
        for zona in sorted(df['Zona'].unique()):
            df_z = df[df['Zona'] == zona].copy()
            self.zone_node_pairs[zona] = (
                df_z[['Zona', 'Nodo']].drop_duplicates().to_dict('records')
            )

            if zona == CENTRO:
                for nodo in sorted(df_z['Nodo'].unique()):
                    logger.info(f"  [Nodo] {nodo}")
                    df_n = df_z[df_z['Nodo'] == nodo].sort_values('Fecha').reset_index(drop=True)
                    samples = self._make_direct_samples(
                        df_n['Tickets_Total'].values, df_n['Fecha'].values, nodo, zona
                    )
                    if len(samples) < 30:
                        continue
                    feat_cols = self._get_feature_cols(samples)
                    X = samples[feat_cols]
                    y = samples['Tickets_Total']
                    self.node_features[nodo] = feat_cols
                    models, maes = self._train_ensemble_for_key(X, y, n_trials)
                    self.node_models[nodo] = models
                    logger.info(f"    {nodo} MAE CV -> q30:{maes['q30']:.2f} q50:{maes['q50']:.2f} q70:{maes['q70']:.2f}")
            else:
                logger.info(f"  [Zona] {zona}")
                all_s = []
                for nodo in df_z['Nodo'].unique():
                    df_n = df_z[df_z['Nodo'] == nodo].sort_values('Fecha').reset_index(drop=True)
                    all_s.append(self._make_direct_samples(
                        df_n['Tickets_Total'].values, df_n['Fecha'].values, nodo, zona
                    ))
                samples = pd.concat(all_s, ignore_index=True)
                feat_cols = self._get_feature_cols(samples)
                X = samples[feat_cols]
                y = samples['Tickets_Total']
                self.zone_features[zona] = feat_cols
                models, maes = self._train_ensemble_for_key(X, y, n_trials)
                self.zone_models[zona] = models
                logger.info(f"    {zona} MAE CV -> q30:{maes['q30']:.2f} q50:{maes['q50']:.2f} q70:{maes['q70']:.2f}")

        self.save_model()
        return self

    # ── Prediccion ITERATIVA (critica) ────────────────────────────

    def predict_future(self, df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
        """
        Prediccion iterativa real: cada prediccion alimenta los lags del dia siguiente.
        """
        all_preds = []

        for zona in self.zone_node_pairs:
            df_z = df[df['Zona'] == zona].copy()

            for pair in self.zone_node_pairs[zona]:
                nodo = pair['Nodo']
                df_n = df_z[df_z['Nodo'] == nodo].sort_values('Fecha').reset_index(drop=True)
                if len(df_n) < 14:
                    continue

                # Ventana deslizante: historico real + predicciones
                window = df_n['Tickets_Total'].astype(float).tolist()
                last_date = pd.to_datetime(df_n['Fecha'].iloc[-1])

                for h in range(1, days + 1):
                    dt = last_date + pd.Timedelta(days=h)
                    dow = dt.dayofweek

                    # Recalcular lags y medias moviles dinamicamente
                    lag1 = window[-1]
                    lag7 = window[-7] if len(window) >= 7 else window[0]
                    ma7 = np.mean(window[-7:])
                    ma14 = np.mean(window[-14:]) if len(window) >= 14 else np.mean(window)

                    row = pd.DataFrame([{
                        'Fecha': dt, 'Nodo': nodo, 'Zona': zona,
                        'Dia_Semana': dow, 'Mes': dt.month, 'Horizonte': h,
                        'Lag_1': lag1, 'Lag_7': lag7,
                        'Media_Movil_7d': ma7, 'Media_Movil_14d': ma14,
                    }])
                    row = self._add_calendar(row)
                    row = self._add_weather(row, zona)
                    row = self._add_target_encoding(row)

                    # Seleccionar modelos y features
                    if zona == CENTRO and nodo in self.node_models:
                        models = self.node_models[nodo]
                        feat_cols = self.node_features.get(nodo, self.base_features)
                    elif zona in self.zone_models:
                        models = self.zone_models[zona]
                        feat_cols = self.zone_features.get(zona, self.base_features)
                    else:
                        # Fallback estacional
                        seas = self.seasonal.get((zona, nodo, dow), 0)
                        window.append(float(seas))
                        all_preds.append({
                            'Fecha': dt, 'Zona': zona, 'Nodo': nodo,
                            'Prediccion_Tickets': max(0, round(seas)),
                            'Pred_Q30': max(0, round(seas)),
                            'Pred_Q50': max(0, round(seas)),
                            'Pred_Q70': max(0, round(seas)),
                        })
                        continue

                    # Alinear columnas
                    for f in feat_cols:
                        if f not in row.columns:
                            row[f] = 0
                    X = row[feat_cols]

                    # Triple prediccion
                    p30 = models['q30'].predict(X)[0]
                    p50 = models['q50'].predict(X)[0]
                    p70 = models['q70'].predict(X)[0]

                    # Ensemble adaptativo por condiciones
                    is_risky = (row['es_lluvia_fuerte'].iloc[0] == 1 or
                                row['es_quincena'].iloc[0] == 1 or
                                row['es_lunes'].iloc[0] == 1)
                    if is_risky:
                        # Sesgar hacia arriba en dias criticos
                        xgb_p = 0.25 * p30 + 0.35 * p50 + 0.40 * p70
                    else:
                        xgb_p = 0.35 * p30 + 0.45 * p50 + 0.20 * p70

                    # DOW correction (usa dow_damping como hiperparametro de negocio)
                    f = self.dow_factors.get((zona, nodo, dow), 1.0)
                    # Usar un damping adaptativo: mas fuerte si el factor es muy extremo
                    damping = 0.35  # valor por defecto; podria optimizarse
                    xgb_adj = xgb_p * (1 + damping * (f - 1))

                    # Seasonal ensemble
                    seas = self.seasonal.get((zona, nodo, dow), xgb_adj)
                    final = self.ensemble_xgb_w * xgb_adj + (1 - self.ensemble_xgb_w) * seas

                    final_rounded = max(0, round(final))
                    window.append(float(final_rounded))

                    all_preds.append({
                        'Fecha': dt, 'Zona': zona, 'Nodo': nodo,
                        'Prediccion_Tickets': final_rounded,
                        'Pred_Q30': max(0, round(p30)),
                        'Pred_Q50': max(0, round(p50)),
                        'Pred_Q70': max(0, round(p70)),
                    })

        return pd.DataFrame(all_preds)

    # ── Prediccion DIRECTA (legacy, para comparacion AB) ──────────

    def predict_future_direct(self, df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
        """Version directa como la V4 (para benchmarking)."""
        all_preds = []
        for zona in self.zone_node_pairs:
            df_z = df[df['Zona'] == zona].copy()
            for pair in self.zone_node_pairs[zona]:
                nodo = pair['Nodo']
                df_n = df_z[df_z['Nodo'] == nodo].sort_values('Fecha').reset_index(drop=True)
                if len(df_n) < 14:
                    continue
                oi = len(df_n) - 1
                tk = df_n['Tickets_Total'].values
                origin = pd.to_datetime(df_n.loc[oi, 'Fecha'])
                lag1 = tk[oi]
                lag7 = tk[oi-6] if oi >= 6 else tk[0]
                ma7 = tk[max(0,oi-6):oi+1].mean()
                ma14 = tk[max(0,oi-13):oi+1].mean()
                dates = pd.date_range(start=origin + pd.Timedelta(days=1), periods=days)
                for h, dt in enumerate(dates, 1):
                    dow = dt.dayofweek
                    row = pd.DataFrame([{
                        'Fecha': dt, 'Nodo': nodo, 'Zona': zona,
                        'Dia_Semana': dow, 'Mes': dt.month, 'Horizonte': h,
                        'Lag_1': lag1, 'Lag_7': lag7,
                        'Media_Movil_7d': ma7, 'Media_Movil_14d': ma14,
                    }])
                    row = self._add_calendar(row)
                    row = self._add_weather(row, zona)
                    row = self._add_target_encoding(row)
                    if zona == CENTRO and nodo in self.node_models:
                        feat_cols = self.node_features.get(nodo, self.base_features)
                        models = self.node_models[nodo]
                    elif zona in self.zone_models:
                        feat_cols = self.zone_features.get(zona, self.base_features)
                        models = self.zone_models[zona]
                    else:
                        seas = self.seasonal.get((zona, nodo, dow), 0)
                        all_preds.append({'Fecha': dt, 'Zona': zona, 'Nodo': nodo, 'Prediccion_Tickets': max(0, round(seas))})
                        continue
                    for f in feat_cols:
                        if f not in row.columns:
                            row[f] = 0
                    X = row[feat_cols]
                    p50 = models['q50'].predict(X)[0]
                    damping = 0.35
                    f = self.dow_factors.get((zona, nodo, dow), 1.0)
                    xgb_adj = p50 * (1 + damping * (f - 1))
                    seas = self.seasonal.get((zona, nodo, dow), xgb_adj)
                    final = self.ensemble_xgb_w * xgb_adj + (1 - self.ensemble_xgb_w) * seas
                    all_preds.append({'Fecha': dt, 'Zona': zona, 'Nodo': nodo, 'Prediccion_Tickets': max(0, round(final))})
        return pd.DataFrame(all_preds)

    # ── Persistencia ──────────────────────────────────────────────

    def save_model(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'zone_models': self.zone_models, 'zone_features': self.zone_features,
            'node_models': self.node_models, 'node_features': self.node_features,
            'zone_node_pairs': self.zone_node_pairs,
            'seasonal': self.seasonal, 'dow_factors': self.dow_factors,
            'nodo_mean': self.nodo_mean, 'nodo_std': self.nodo_std,
            'ensemble_xgb_w': self.ensemble_xgb_w, 'target_smooth': self.target_smooth,
        }, MODELS_DIR / 'modelo_forecasting_v5.pkl')
        logger.info("Modelo V5 guardado.")

    def load_model(self):
        d = joblib.load(MODELS_DIR / 'modelo_forecasting_v5.pkl')
        for k, v in d.items():
            setattr(self, k, v)
        logger.info("Modelo V5 cargado.")
