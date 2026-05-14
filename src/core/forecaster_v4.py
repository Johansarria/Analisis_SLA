"""
V4 = V3 + Mejora 6: Features exogenas de clima (Open-Meteo).
Nuevas features: precipitacion_mm, temp_max, viento_max_kmh,
                 es_lluvia_fuerte, es_lluvia, lluvia_acum_3d
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

HOLIDAYS_2026 = pd.to_datetime([
    '2026-01-01', '2026-01-12', '2026-03-23', '2026-04-02', '2026-04-03',
    '2026-04-05', '2026-05-01', '2026-05-18', '2026-06-08', '2026-06-15',
    '2026-07-05', '2026-07-20', '2026-08-15', '2026-08-16', '2026-10-12',
    '2026-11-02', '2026-11-16', '2026-12-08', '2026-12-25'
])


class DemandForecasterV4:
    """
    V4 integra todas las mejoras:
      1-4 (V2): pred directa, calendario, quantile, per-zona
      5   (V3): per-nodo para Centro
      7   (V3): ensemble estacional
      8   (V3): correccion DOW
      6   (V4): features climaticas de Open-Meteo
    """

    def __init__(self, quantile_alpha=0.6, ensemble_xgb_w=0.6, dow_damping=0.4):
        self.quantile_alpha = quantile_alpha
        self.ensemble_xgb_w = ensemble_xgb_w
        self.dow_damping = dow_damping

        self.zone_models = {}
        self.zone_features = {}
        self.node_models = {}
        self.node_features = {}
        self.zone_node_pairs = {}
        self.seasonal = {}
        self.dow_factors = {}
        self.weather_df = None  # Cache del clima

        self.base_features = [
            'Dia_Semana', 'Mes', 'Horizonte',
            'Lag_1_origin', 'Lag_7_origin',
            'Media_Movil_7d_origin', 'Media_Movil_14d_origin',
            'es_festivo', 'es_quincena', 'semana_del_mes',
            'dia_del_mes', 'es_lunes', 'es_viernes', 'es_finsemana',
            # ── Mejora 6: clima ──
            'precipitacion_mm', 'temp_max', 'viento_max_kmh',
            'es_lluvia_fuerte', 'es_lluvia', 'lluvia_acum_3d',
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

    def _make_direct_samples(self, tickets, fechas, nodo_name, zona):
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
                    'Lag_1_origin': tickets[o],
                    'Lag_7_origin': tickets[o-6] if o >= 6 else tickets[0],
                    'Media_Movil_7d_origin': tickets[max(0,o-6):o+1].mean(),
                    'Media_Movil_14d_origin': tickets[max(0,o-13):o+1].mean(),
                })
        df = pd.DataFrame(rows)
        df = self._add_calendar(df)
        df = self._add_weather(df, zona)
        return df

    def _encode(self, df, key, is_train, use_nodo=True):
        if use_nodo and 'Nodo' in df.columns:
            df_e = pd.get_dummies(df, columns=['Nodo'], drop_first=False, dtype=int)
            nc = sorted(c for c in df_e.columns if c.startswith('Nodo_'))
        else:
            df_e = df.copy()
            nc = []
        if is_train:
            feats = self.base_features + nc
            if use_nodo:
                self.zone_features[key] = feats
            else:
                self.node_features[key] = feats
        else:
            feats = self.zone_features.get(key) or self.node_features.get(key, self.base_features)
        for f in feats:
            if f not in df_e.columns:
                df_e[f] = 0
        return df_e[feats]

    def _optuna_train(self, X, y, n_trials):
        def objective(trial):
            p = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 600),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': 42,
                'objective': 'reg:quantileerror',
                'quantile_alpha': self.quantile_alpha,
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
                   'quantile_alpha': self.quantile_alpha})
        model = xgb.XGBRegressor(**bp)
        model.fit(X, y, verbose=False)
        return model, study.best_value

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
        logger.info("Entrenando modelo V4 (con clima)...")

        # Fase 0: Descargar clima
        start = df['Fecha'].min()
        end = df['Fecha'].max()
        # Extender al futuro para cubrir el periodo de prediccion
        end_ext = pd.to_datetime(end) + pd.Timedelta(days=35)
        logger.info(f"  Descargando clima {start} -> {end_ext.strftime('%Y-%m-%d')}...")
        self.weather_df = get_weather_for_zones(
            str(start)[:10], end_ext.strftime('%Y-%m-%d')
        )
        logger.info(f"  Clima obtenido: {len(self.weather_df)} registros")

        # Fase 1: Estadisticas
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
                    X = self._encode(samples, nodo, is_train=True, use_nodo=False)
                    y = samples['Tickets_Total']
                    model, mae = self._optuna_train(X, y, n_trials)
                    self.node_models[nodo] = model
                    logger.info(f"    {nodo} MAE CV: {mae:.2f}")
            else:
                logger.info(f"  [Zona] {zona}")
                all_s = []
                for nodo in df_z['Nodo'].unique():
                    df_n = df_z[df_z['Nodo'] == nodo].sort_values('Fecha').reset_index(drop=True)
                    all_s.append(self._make_direct_samples(
                        df_n['Tickets_Total'].values, df_n['Fecha'].values, nodo, zona
                    ))
                samples = pd.concat(all_s, ignore_index=True)
                X = self._encode(samples, zona, is_train=True, use_nodo=True)
                y = samples['Tickets_Total']
                model, mae = self._optuna_train(X, y, n_trials)
                self.zone_models[zona] = model
                logger.info(f"    {zona} MAE CV: {mae:.2f}")

        self.save_model()
        return self

    # ── Prediccion ────────────────────────────────────────────────

    def predict_future(self, df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
        all_preds = []

        for zona in self.zone_node_pairs:
            df_z = df[df['Zona'] == zona].copy()

            for pair in self.zone_node_pairs[zona]:
                nodo = pair['Nodo']
                df_n = df_z[df_z['Nodo'] == nodo].sort_values('Fecha').reset_index(drop=True)
                if len(df_n) < 7:
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
                        'Lag_1_origin': lag1, 'Lag_7_origin': lag7,
                        'Media_Movil_7d_origin': ma7, 'Media_Movil_14d_origin': ma14,
                    }])
                    row = self._add_calendar(row)
                    row = self._add_weather(row, zona)

                    if zona == CENTRO and nodo in self.node_models:
                        X = self._encode(row, nodo, is_train=False, use_nodo=False)
                        xgb_p = self.node_models[nodo].predict(X)[0]
                    elif zona in self.zone_models:
                        X = self._encode(row, zona, is_train=False, use_nodo=True)
                        xgb_p = self.zone_models[zona].predict(X)[0]
                    else:
                        xgb_p = self.seasonal.get((zona, nodo, dow), 0)

                    # DOW correction
                    f = self.dow_factors.get((zona, nodo, dow), 1.0)
                    xgb_adj = xgb_p * (1 + self.dow_damping * (f - 1))

                    # Seasonal ensemble
                    seas = self.seasonal.get((zona, nodo, dow), xgb_adj)
                    final = self.ensemble_xgb_w * xgb_adj + (1 - self.ensemble_xgb_w) * seas

                    all_preds.append({
                        'Fecha': dt, 'Zona': zona, 'Nodo': nodo,
                        'Prediccion_Tickets': max(0, round(final)),
                    })

        return pd.DataFrame(all_preds)

    # ── Persistencia ──────────────────────────────────────────────

    def save_model(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'zone_models': self.zone_models, 'zone_features': self.zone_features,
            'node_models': self.node_models, 'node_features': self.node_features,
            'zone_node_pairs': self.zone_node_pairs,
            'seasonal': self.seasonal, 'dow_factors': self.dow_factors,
            'quantile_alpha': self.quantile_alpha,
            'ensemble_xgb_w': self.ensemble_xgb_w, 'dow_damping': self.dow_damping,
        }, MODELS_DIR / 'modelo_forecasting_v4.pkl')
        logger.info("Modelo V4 guardado.")

    def load_model(self):
        d = joblib.load(MODELS_DIR / 'modelo_forecasting_v4.pkl')
        for k, v in d.items():
            setattr(self, k, v)
        logger.info("Modelo V4 cargado.")
