import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path
from src.config import MODELS_DIR

logger = logging.getLogger(__name__)

class DemandForecaster:
    """Clase para el modelo predictivo de demanda de tickets (Time Series) desagregado por Zonas y Nodos."""
    
    def __init__(self):
        self.model = None
        self.base_features = ['Dia_Semana', 'Mes', 'Lag_1', 'Lag_7', 'Media_Movil_7d', 'Media_Movil_14d']
        self.features = [] # Se populará dinámicamente con One-Hot Encoding
        self.target = 'Tickets_Total'
        self.zone_node_pairs = []
        
    def _prepare_features(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """Aplica One-Hot Encoding a las variables Zona y Nodo asegurando consistencia de columnas."""
        df_encoded = pd.get_dummies(df, columns=['Zona', 'Nodo'], drop_first=False, dtype=int)
        
        if is_train:
            # Capturar todas las características finales
            self.features = self.base_features + [c for c in df_encoded.columns if c.startswith('Zona_') or c.startswith('Nodo_')]
            self.zone_node_pairs = df[['Zona', 'Nodo']].drop_duplicates().to_dict('records')
        
        # Asegurar que todas las columnas de feature esperadas existan
        for f in self.features:
            if f not in df_encoded.columns:
                df_encoded[f] = 0
                
        return df_encoded[self.features]
        
    def _objective(self, trial, X, y):
        """Función objetivo para Optuna."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': 42,
            'objective': 'reg:squarederror'
        }
        
        tscv = TimeSeriesSplit(n_splits=5)
        maes = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, verbose=False)
            
            preds = model.predict(X_val)
            mae = mean_absolute_error(y_val, preds)
            maes.append(mae)
            
        return np.mean(maes)
        
    def train(self, df: pd.DataFrame, n_trials: int = 15):
        """Entrena el modelo de forecasting usando Optuna y validación temporal."""
        logger.info(f"Iniciando entrenamiento del modelo de Forecasting ({n_trials} trials)...")
        
        # Ordenar por fecha para que el TimeSeriesSplit tenga sentido lógico
        df = df.sort_values('Fecha').reset_index(drop=True)
        
        X = self._prepare_features(df, is_train=True)
        y = df[self.target]
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self._objective(trial, X, y), n_trials=n_trials)
        
        best_params = study.best_params
        best_params['random_state'] = 42
        best_params['objective'] = 'reg:squarederror'
        
        logger.info(f"Mejores parámetros para Forecasting: {best_params}")
        logger.info(f"MAE de validación cruzada estimado: {study.best_value:.2f}")
        
        self.model = xgb.XGBRegressor(**best_params)
        self.model.fit(X, y)
        
        self.save_model()
        return self.model
        
    def predict_future(self, df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
        """
        Predice iterativamente la demanda para N días futuros por cada par (Zona, Nodo).
        """
        logger.info(f"Generando predicción para {days} días para {len(self.zone_node_pairs)} pares de Zonas-Nodos...")
        
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado o cargado.")
            
        future_dates = pd.date_range(start=df['Fecha'].max() + pd.Timedelta(days=1), periods=days)
        all_predictions = []
        
        # Predecir iterativamente por cada pareja Zona-Nodo
        for pair in self.zone_node_pairs:
            zona = pair['Zona']
            nodo = pair['Nodo']
            
            # Filtrar solo la historia de este nodo en esta zona
            df_zn = df[(df['Zona'] == zona) & (df['Nodo'] == nodo)].copy().sort_values('Fecha').reset_index(drop=True)
            
            for date in future_dates:
                new_row = pd.DataFrame([{
                    'Fecha': date,
                    'Zona': zona,
                    'Nodo': nodo,
                    'Dia_Semana': date.dayofweek,
                    'Mes': date.month,
                    'Tickets_Total': np.nan 
                }])
                
                df_zn = pd.concat([df_zn, new_row], ignore_index=True)
                idx = len(df_zn) - 1
                
                # Recalcular lags solo para este nodo
                df_zn.loc[idx, 'Lag_1'] = df_zn.loc[idx - 1, 'Tickets_Total']
                df_zn.loc[idx, 'Lag_7'] = df_zn.loc[idx - 7, 'Tickets_Total'] if idx >= 7 else df_zn.loc[idx - 1, 'Tickets_Total']
                df_zn.loc[idx, 'Media_Movil_7d'] = df_zn.loc[idx - 7:idx - 1, 'Tickets_Total'].mean()
                df_zn.loc[idx, 'Media_Movil_14d'] = df_zn.loc[idx - 14:idx - 1, 'Tickets_Total'].mean()
                
                # Predecir este paso
                row_to_predict = df_zn.loc[[idx]]
                X_pred = self._prepare_features(row_to_predict, is_train=False)
                
                pred_val = self.model.predict(X_pred)[0]
                pred_val = max(0, round(pred_val))
                
                df_zn.loc[idx, 'Tickets_Total'] = pred_val
                
                all_predictions.append({
                    'Fecha': date,
                    'Zona': zona,
                    'Nodo': nodo,
                    'Prediccion_Tickets': pred_val
                })
                
        return pd.DataFrame(all_predictions)
        
    def save_model(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = MODELS_DIR / 'modelo_forecasting_xgb.pkl'
        joblib.dump({
            'model': self.model, 
            'features': self.features, 
            'zone_node_pairs': self.zone_node_pairs
        }, path)
        logger.info(f"Modelo Forecasting guardado en {path}")
