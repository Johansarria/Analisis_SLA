import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path
from src.config import MODELS_DIR

logger = logging.getLogger(__name__)

# Festivos colombianos 2026 (Ley de Emiliani aplicada)
HOLIDAYS_2026 = pd.to_datetime([
    '2026-01-01', '2026-01-12', '2026-03-23',
    '2026-04-02', '2026-04-03',  # Semana Santa
    '2026-05-01', '2026-05-18', '2026-06-08',
    '2026-06-15', '2026-06-29', '2026-07-20',
    '2026-08-07', '2026-08-17', '2026-10-12',
    '2026-11-02', '2026-11-16', '2026-12-08', '2026-12-25'
])


class DemandForecasterV2:
    """
    Forecaster mejorado con:
    1. Predicción directa multi-horizonte (sin propagación de errores)
    2. Features de calendario (festivos, quincena, semana del mes)
    3. Quantile regression para capturar picos (alpha=0.6)
    4. Modelo separado por zona
    """

    def __init__(self, quantile_alpha=0.6):
        self.models = {}
        self.features = {}
        self.zone_node_pairs = {}
        self.quantile_alpha = quantile_alpha
        self.base_features = [
            'Dia_Semana', 'Mes', 'Horizonte',
            'Lag_1_origin', 'Lag_7_origin',
            'Media_Movil_7d_origin', 'Media_Movil_14d_origin',
            'es_festivo', 'es_quincena', 'semana_del_mes',
            'dia_del_mes', 'es_lunes', 'es_viernes', 'es_finsemana'
        ]

    # ------------------------------------------------------------------
    # FEATURE ENGINEERING
    # ------------------------------------------------------------------
    def _add_calendar_features(self, df):
        """Agrega features de calendario colombiano."""
        df = df.copy()
        fecha = pd.to_datetime(df['Fecha'])
        dia = fecha.dt.day
        dow = fecha.dt.dayofweek

        df['es_festivo'] = fecha.dt.normalize().isin(HOLIDAYS_2026).astype(int)
        df['es_quincena'] = (((dia >= 14) & (dia <= 16)) | (dia >= 28) | (dia <= 1)).astype(int)
        df['semana_del_mes'] = ((dia - 1) // 7 + 1).astype(int)
        df['dia_del_mes'] = dia.astype(int)
        df['es_lunes'] = (dow == 0).astype(int)
        df['es_viernes'] = (dow == 4).astype(int)
        df['es_finsemana'] = (dow >= 5).astype(int)
        return df

    def _prepare_features(self, df, zone_name, is_train=False):
        """Aplica One-Hot Encoding a Nodo y selecciona features."""
        df_encoded = pd.get_dummies(df, columns=['Nodo'], drop_first=False, dtype=int)
        nodo_cols = sorted([c for c in df_encoded.columns if c.startswith('Nodo_')])

        if is_train:
            self.features[zone_name] = self.base_features + nodo_cols

        for f in self.features[zone_name]:
            if f not in df_encoded.columns:
                df_encoded[f] = 0

        return df_encoded[self.features[zone_name]]

    # ------------------------------------------------------------------
    # MEJORA #1: PREPARACIÓN DE DATOS PARA PREDICCIÓN DIRECTA
    # ------------------------------------------------------------------
    def _prepare_direct_training_data(self, df_zone, zone_name):
        """
        Crea muestras de entrenamiento para predicción directa.
        Para cada fecha T y horizonte h, los lags provienen del punto
        de origen (T-h), que siempre es dato real. 
        Esto elimina la propagación de errores.
        """
        horizons = list(range(1, 31))
        samples = []

        for nodo in df_zone['Nodo'].unique():
            df_n = df_zone[df_zone['Nodo'] == nodo].sort_values('Fecha').reset_index(drop=True)
            tickets = df_n['Tickets_Total'].values
            fechas = df_n['Fecha'].values
            n = len(df_n)

            for target_idx in range(14, n):
                target_date = fechas[target_idx]
                target_val = tickets[target_idx]

                for h in horizons:
                    origin_idx = target_idx - h
                    if origin_idx < 7:
                        continue

                    # Lags desde el punto de origen (datos reales)
                    lag_1 = tickets[origin_idx]
                    lag_7 = tickets[origin_idx - 6] if origin_idx >= 6 else tickets[0]
                    ma_7 = tickets[max(0, origin_idx - 6):origin_idx + 1].mean()
                    ma_14 = tickets[max(0, origin_idx - 13):origin_idx + 1].mean()

                    td = pd.to_datetime(target_date)
                    samples.append({
                        'Fecha': target_date,
                        'Nodo': nodo,
                        'Tickets_Total': target_val,
                        'Dia_Semana': td.dayofweek,
                        'Mes': td.month,
                        'Horizonte': h,
                        'Lag_1_origin': lag_1,
                        'Lag_7_origin': lag_7,
                        'Media_Movil_7d_origin': ma_7,
                        'Media_Movil_14d_origin': ma_14,
                    })

        df_samples = pd.DataFrame(samples)
        df_samples = self._add_calendar_features(df_samples)
        return df_samples

    # ------------------------------------------------------------------
    # ENTRENAMIENTO — MEJORA #3 (Quantile) + #4 (Por Zona)
    # ------------------------------------------------------------------
    def _objective(self, trial, X, y):
        """Función objetivo para Optuna con quantile loss."""
        params = {
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

        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr, verbose=False)
            preds = model.predict(X_val)
            maes.append(mean_absolute_error(y_val, preds))

        return np.mean(maes)

    def train(self, df: pd.DataFrame, n_trials: int = 15):
        """Entrena un modelo separado por zona (Mejora #4)."""
        logger.info(f"Entrenando modelos V2 ({n_trials} trials por zona)...")

        for zona in sorted(df['Zona'].unique()):
            logger.info(f"  Zona: {zona}")
            df_zona = df[df['Zona'] == zona].copy()

            self.zone_node_pairs[zona] = (
                df_zona[['Zona', 'Nodo']].drop_duplicates().to_dict('records')
            )

            df_samples = self._prepare_direct_training_data(df_zona, zona)
            if len(df_samples) < 50:
                logger.warning(f"  {zona}: muy pocas muestras ({len(df_samples)}), saltando.")
                continue

            X = self._prepare_features(df_samples, zona, is_train=True)
            y = df_samples['Tickets_Total']

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction='minimize')
            study.optimize(
                lambda trial: self._objective(trial, X, y), n_trials=n_trials
            )

            best_params = study.best_params
            best_params['random_state'] = 42
            best_params['objective'] = 'reg:quantileerror'
            best_params['quantile_alpha'] = self.quantile_alpha

            logger.info(f"  {zona} — MAE CV: {study.best_value:.2f}")

            model = xgb.XGBRegressor(**best_params)
            model.fit(X, y, verbose=False)
            self.models[zona] = model

        self.save_model()
        return self.models

    # ------------------------------------------------------------------
    # PREDICCIÓN DIRECTA (sin iteración)
    # ------------------------------------------------------------------
    def predict_future(self, df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
        """
        Predicción directa: lags siempre provienen de datos reales.
        No hay iteración → no hay propagación de errores.
        """
        all_preds = []

        for zona, model in self.models.items():
            df_zona = df[df['Zona'] == zona].copy()

            for pair in self.zone_node_pairs[zona]:
                nodo = pair['Nodo']
                df_n = df_zona[df_zona['Nodo'] == nodo].sort_values('Fecha').reset_index(drop=True)

                if len(df_n) < 7:
                    continue

                oidx = len(df_n) - 1
                tickets = df_n['Tickets_Total'].values
                origin_date = pd.to_datetime(df_n.loc[oidx, 'Fecha'])

                # Lags calculados UNA VEZ desde datos reales
                lag_1 = tickets[oidx]
                lag_7 = tickets[oidx - 6] if oidx >= 6 else tickets[0]
                ma_7 = tickets[max(0, oidx - 6):oidx + 1].mean()
                ma_14 = tickets[max(0, oidx - 13):oidx + 1].mean()

                future_dates = pd.date_range(
                    start=origin_date + pd.Timedelta(days=1), periods=days
                )

                for h, date in enumerate(future_dates, start=1):
                    row = pd.DataFrame([{
                        'Fecha': date,
                        'Nodo': nodo,
                        'Dia_Semana': date.dayofweek,
                        'Mes': date.month,
                        'Horizonte': h,
                        'Lag_1_origin': lag_1,
                        'Lag_7_origin': lag_7,
                        'Media_Movil_7d_origin': ma_7,
                        'Media_Movil_14d_origin': ma_14,
                    }])
                    row = self._add_calendar_features(row)
                    X_pred = self._prepare_features(row, zona, is_train=False)

                    pred_val = max(0, round(model.predict(X_pred)[0]))
                    all_preds.append({
                        'Fecha': date,
                        'Zona': zona,
                        'Nodo': nodo,
                        'Prediccion_Tickets': pred_val,
                    })

        return pd.DataFrame(all_preds)

    # ------------------------------------------------------------------
    # PERSISTENCIA
    # ------------------------------------------------------------------
    def save_model(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = MODELS_DIR / 'modelo_forecasting_v2.pkl'
        joblib.dump({
            'models': self.models,
            'features': self.features,
            'zone_node_pairs': self.zone_node_pairs,
            'quantile_alpha': self.quantile_alpha,
        }, path)
        logger.info(f"Modelo V2 guardado en {path}")

    def load_model(self):
        path = MODELS_DIR / 'modelo_forecasting_v2.pkl'
        data = joblib.load(path)
        self.models = data['models']
        self.features = data['features']
        self.zone_node_pairs = data['zone_node_pairs']
        self.quantile_alpha = data.get('quantile_alpha', 0.6)
        logger.info(f"Modelo V2 cargado desde {path}")
