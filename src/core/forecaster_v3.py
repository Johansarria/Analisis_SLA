import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import joblib
from src.config import MODELS_DIR
from src.core.forecaster_v2 import HOLIDAYS_2026

logger = logging.getLogger(__name__)

CENTRO_ZONE = 'Zona Centro'


class DemandForecasterV3:
    """
    V3 = V2 + 3 mejoras adicionales:
      5. Modelos por nodo para Zona Centro (Asturias, Progreso, etc.)
      7. Ensemble con componente estacional semanal
      8. Correccion post-procesamiento por dia de semana
    """

    def __init__(self, quantile_alpha=0.6, ensemble_xgb_w=0.6, dow_damping=0.4):
        self.quantile_alpha = quantile_alpha
        self.ensemble_xgb_w = ensemble_xgb_w
        self.dow_damping = dow_damping

        self.zone_models = {}       # {zona: model} para zonas != Centro
        self.zone_features = {}     # {zona: [features]}
        self.node_models = {}       # {nodo: model} para nodos de Centro (Mejora 5)
        self.node_features = {}     # {nodo: [features]}
        self.zone_node_pairs = {}   # {zona: [{Zona, Nodo}]}

        self.seasonal = {}          # {(zona, nodo, dow): mean_tickets} (Mejora 7)
        self.dow_factors = {}       # {(zona, nodo, dow): factor}       (Mejora 8)

        self.base_features = [
            'Dia_Semana', 'Mes', 'Horizonte',
            'Lag_1_origin', 'Lag_7_origin',
            'Media_Movil_7d_origin', 'Media_Movil_14d_origin',
            'es_festivo', 'es_quincena', 'semana_del_mes',
            'dia_del_mes', 'es_lunes', 'es_viernes', 'es_finsemana',
        ]

    # ── helpers (reutilizados de V2) ──────────────────────────────

    def _add_calendar(self, df):
        df = df.copy()
        f = pd.to_datetime(df['Fecha'])
        dia, dow = f.dt.day, f.dt.dayofweek
        df['es_festivo']    = f.dt.normalize().isin(HOLIDAYS_2026).astype(int)
        df['es_quincena']   = (((dia >= 14) & (dia <= 16)) | (dia >= 28) | (dia <= 1)).astype(int)
        df['semana_del_mes'] = ((dia - 1) // 7 + 1).astype(int)
        df['dia_del_mes']   = dia.astype(int)
        df['es_lunes']      = (dow == 0).astype(int)
        df['es_viernes']    = (dow == 4).astype(int)
        df['es_finsemana']  = (dow >= 5).astype(int)
        return df

    def _make_direct_samples(self, tickets, fechas, nodo_name=None):
        """Crea muestras de entrenamiento directo para un nodo."""
        n = len(tickets)
        rows = []
        for t_idx in range(14, n):
            for h in range(1, 31):
                o = t_idx - h
                if o < 7:
                    continue
                td = pd.to_datetime(fechas[t_idx])
                rows.append({
                    'Fecha': fechas[t_idx],
                    'Nodo': nodo_name or '',
                    'Tickets_Total': tickets[t_idx],
                    'Dia_Semana': td.dayofweek,
                    'Mes': td.month,
                    'Horizonte': h,
                    'Lag_1_origin': tickets[o],
                    'Lag_7_origin': tickets[o - 6] if o >= 6 else tickets[0],
                    'Media_Movil_7d_origin': tickets[max(0, o-6):o+1].mean(),
                    'Media_Movil_14d_origin': tickets[max(0, o-13):o+1].mean(),
                })
        return pd.DataFrame(rows)

    def _encode(self, df, key, is_train, use_nodo=True):
        """One-hot encode Nodo (solo si use_nodo=True) y seleccionar features."""
        if use_nodo and 'Nodo' in df.columns:
            df_enc = pd.get_dummies(df, columns=['Nodo'], drop_first=False, dtype=int)
            nodo_cols = sorted(c for c in df_enc.columns if c.startswith('Nodo_'))
        else:
            df_enc = df.copy()
            nodo_cols = []

        store = self.zone_features if key in self.zone_features or key not in self.node_features else self.node_features
        if is_train:
            feats = self.base_features + nodo_cols
            if use_nodo:
                self.zone_features[key] = feats
            else:
                self.node_features[key] = feats
        else:
            feats = self.zone_features.get(key) or self.node_features.get(key, self.base_features)

        for f in feats:
            if f not in df_enc.columns:
                df_enc[f] = 0
        return df_enc[feats]

    def _optuna_train(self, X, y, n_trials):
        """Entrena XGBoost con Optuna + quantile loss."""
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

    # ── Mejora 7: componente estacional ───────────────────────────

    def _compute_seasonal(self, df):
        """Calcula la media historica de tickets por (zona, nodo, dia_semana)."""
        df = df.copy()
        df['dow'] = pd.to_datetime(df['Fecha']).dt.dayofweek
        grp = df.groupby(['Zona', 'Nodo', 'dow'])['Tickets_Total'].mean()
        self.seasonal = grp.to_dict()
        logger.info(f"  Indices estacionales calculados: {len(self.seasonal)} combinaciones")

    # ── Mejora 8: factores DOW ────────────────────────────────────

    def _compute_dow_factors(self, df):
        """Calcula factor multiplicativo DOW por (zona, nodo)."""
        df = df.copy()
        df['dow'] = pd.to_datetime(df['Fecha']).dt.dayofweek
        overall = df.groupby(['Zona', 'Nodo'])['Tickets_Total'].mean()
        by_dow  = df.groupby(['Zona', 'Nodo', 'dow'])['Tickets_Total'].mean()

        for (zona, nodo, dow), mean_dow in by_dow.items():
            mean_all = overall.get((zona, nodo), 1)
            if mean_all > 0:
                self.dow_factors[(zona, nodo, dow)] = mean_dow / mean_all
            else:
                self.dow_factors[(zona, nodo, dow)] = 1.0
        logger.info(f"  Factores DOW calculados: {len(self.dow_factors)} combinaciones")

    # ── Entrenamiento ─────────────────────────────────────────────

    def train(self, df: pd.DataFrame, n_trials: int = 15):
        logger.info("Entrenando modelo V3...")

        # Fase 1: estadisticas (Mejoras 7 y 8)
        self._compute_seasonal(df)
        self._compute_dow_factors(df)

        # Fase 2: entrenar modelos
        for zona in sorted(df['Zona'].unique()):
            df_z = df[df['Zona'] == zona].copy()
            self.zone_node_pairs[zona] = (
                df_z[['Zona', 'Nodo']].drop_duplicates().to_dict('records')
            )

            if zona == CENTRO_ZONE:
                # ── Mejora 5: modelo por nodo ──
                for nodo in sorted(df_z['Nodo'].unique()):
                    logger.info(f"  [Nodo] {nodo}")
                    df_n = df_z[df_z['Nodo'] == nodo].sort_values('Fecha').reset_index(drop=True)
                    samples = self._make_direct_samples(
                        df_n['Tickets_Total'].values, df_n['Fecha'].values, nodo
                    )
                    if len(samples) < 30:
                        logger.warning(f"    {nodo}: pocas muestras ({len(samples)})")
                        continue
                    samples = self._add_calendar(samples)
                    X = self._encode(samples, nodo, is_train=True, use_nodo=False)
                    y = samples['Tickets_Total']
                    model, mae = self._optuna_train(X, y, n_trials)
                    self.node_models[nodo] = model
                    logger.info(f"    {nodo} MAE CV: {mae:.2f}")
            else:
                # ── Per-zone (igual que V2) ──
                logger.info(f"  [Zona] {zona}")
                all_samples = []
                for nodo in df_z['Nodo'].unique():
                    df_n = df_z[df_z['Nodo'] == nodo].sort_values('Fecha').reset_index(drop=True)
                    s = self._make_direct_samples(
                        df_n['Tickets_Total'].values, df_n['Fecha'].values, nodo
                    )
                    all_samples.append(s)
                samples = pd.concat(all_samples, ignore_index=True)
                samples = self._add_calendar(samples)
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
                lag7 = tk[oi - 6] if oi >= 6 else tk[0]
                ma7  = tk[max(0, oi-6):oi+1].mean()
                ma14 = tk[max(0, oi-13):oi+1].mean()

                dates = pd.date_range(start=origin + pd.Timedelta(days=1), periods=days)

                for h, dt in enumerate(dates, 1):
                    dow = dt.dayofweek
                    row = pd.DataFrame([{
                        'Fecha': dt, 'Nodo': nodo,
                        'Dia_Semana': dow, 'Mes': dt.month, 'Horizonte': h,
                        'Lag_1_origin': lag1, 'Lag_7_origin': lag7,
                        'Media_Movil_7d_origin': ma7, 'Media_Movil_14d_origin': ma14,
                    }])
                    row = self._add_calendar(row)

                    # 1. XGBoost prediction
                    if zona == CENTRO_ZONE and nodo in self.node_models:
                        X = self._encode(row, nodo, is_train=False, use_nodo=False)
                        xgb_p = self.node_models[nodo].predict(X)[0]
                    elif zona in self.zone_models:
                        X = self._encode(row, zona, is_train=False, use_nodo=True)
                        xgb_p = self.zone_models[zona].predict(X)[0]
                    else:
                        xgb_p = self.seasonal.get((zona, nodo, dow), 0)

                    # 2. Mejora 8: DOW correction (damped multiplicative)
                    f = self.dow_factors.get((zona, nodo, dow), 1.0)
                    xgb_adj = xgb_p * (1 + self.dow_damping * (f - 1))

                    # 3. Mejora 7: seasonal ensemble
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
            'zone_models': self.zone_models,
            'zone_features': self.zone_features,
            'node_models': self.node_models,
            'node_features': self.node_features,
            'zone_node_pairs': self.zone_node_pairs,
            'seasonal': self.seasonal,
            'dow_factors': self.dow_factors,
            'quantile_alpha': self.quantile_alpha,
            'ensemble_xgb_w': self.ensemble_xgb_w,
            'dow_damping': self.dow_damping,
        }, MODELS_DIR / 'modelo_forecasting_v3.pkl')
        logger.info("Modelo V3 guardado.")

    def load_model(self):
        d = joblib.load(MODELS_DIR / 'modelo_forecasting_v3.pkl')
        self.zone_models    = d['zone_models']
        self.zone_features  = d['zone_features']
        self.node_models    = d['node_models']
        self.node_features  = d['node_features']
        self.zone_node_pairs = d['zone_node_pairs']
        self.seasonal       = d['seasonal']
        self.dow_factors    = d['dow_factors']
        self.quantile_alpha = d.get('quantile_alpha', 0.6)
        self.ensemble_xgb_w = d.get('ensemble_xgb_w', 0.6)
        self.dow_damping    = d.get('dow_damping', 0.4)
        logger.info("Modelo V3 cargado.")
