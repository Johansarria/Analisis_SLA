import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import classification_report, mean_absolute_error, f1_score
import joblib
from typing import Dict, List, Any, Tuple
import logging
from src.config import MODELS_DIR
import optuna
from category_encoders import TargetEncoder
import re

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Clase para el entrenamiento y gestión de modelos de ML optimizados."""

    def _sanitize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia los nombres de las columnas para compatibilidad con XGBoost."""
        regex = re.compile(r"\[|\]|<", re.IGNORECASE)
        df.columns = [regex.sub("_", col) if any(x in str(col) for x in "[]<") else col for col in df.columns]
        return df

    def train_ans_classifier(self, df: pd.DataFrame, model_name: str = "modelo_ans_xgboost.pkl"):
        """Entrena un clasificador XGBoost para cumplimiento de ANS usando Target Encoding y Optuna."""
        logger.info("Entrenando clasificador de ANS con Optuna...")
        
        X_raw = df.drop(['Target', 'CUMPLE/NO CUMPLE'], axis=1, errors='ignore')
        y = df['Target']
        
        # Identificar columnas categóricas
        cat_cols = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Codificación con Target Encoding (evitando la explosión de get_dummies)
        encoder = TargetEncoder(cols=cat_cols)
        X = encoder.fit_transform(X_raw, y)
        X = self._sanitize_columns(X)
        features = X.columns.tolist()

        # Balanceo de clases
        pos_count = len(y[y == 0])
        neg_count = len(y[y == 1])
        scale_weight = pos_count / neg_count if neg_count > 0 else 1

        def objective(trial):
            params = {
                'use_label_encoder': False,
                'eval_metric': 'logloss',
                'scale_pos_weight': scale_weight,
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0)
            }
            
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in cv.split(X, y):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = xgb.XGBClassifier(**params)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                scores.append(f1_score(y_val, preds, average='weighted'))
                
            return np.mean(scores)

        # Optimización
        study = optuna.create_study(direction='maximize')
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=10)
        
        best_params = study.best_params
        best_params['scale_pos_weight'] = scale_weight
        best_params['use_label_encoder'] = False
        best_params['eval_metric'] = 'logloss'
        
        logger.info(f"Mejores parámetros encontrados: {best_params}")

        # Entrenar modelo final con 80/20 para reporte final
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
        modelo_final = xgb.XGBClassifier(**best_params)
        modelo_final.fit(X_train, y_train)
        
        preds = modelo_final.predict(X_test)
        logger.info("\n" + classification_report(y_test, preds))

        save_path = MODELS_DIR / model_name
        joblib.dump({'modelo': modelo_final, 'features': features, 'encoder': encoder}, save_path)
        logger.info(f"Modelo y Encoder guardados en {save_path}")

    def train_oraculo_regressor(self, df: pd.DataFrame, model_name: str = "modelo_oraculo_xgb.pkl"):
        """Entrena un regresor XGBoost para predicción de tiempos con Optuna."""
        logger.info("Entrenando regresor Oráculo con Optuna...")
        
        cols_for_model = [
            'Prioridad', 'Nodo', 'Usuarios_Afectados', 
            'Tipo_Falla', 'Hora_sin', 'Hora_cos', 'Dia_sin', 'Dia_cos'
        ]
        
        # Filtro de columnas que existan en el df (por si faltan las cíclicas)
        cols_for_model = [c for c in cols_for_model if c in df.columns]
        
        X_raw = df[cols_for_model].copy()
        y = df['Minutos_Resolucion']
        
        cat_cols = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Target Encoding
        encoder = TargetEncoder(cols=cat_cols)
        X = encoder.fit_transform(X_raw, y)
        X = self._sanitize_columns(X)
        features = X.columns.tolist()

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_state': 42
            }
            
            cv = KFold(n_splits=3, shuffle=True, random_state=42)
            maes = []
            
            for train_idx, val_idx in cv.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = xgb.XGBRegressor(**params)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                maes.append(mean_absolute_error(y_val, preds))
                
            return np.mean(maes)

        study = optuna.create_study(direction='minimize')
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=10)
        
        best_params = study.best_params
        best_params['random_state'] = 42
        
        logger.info(f"Mejores parámetros encontrados: {best_params}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        modelo_final = xgb.XGBRegressor(**best_params)
        modelo_final.fit(X_train, y_train)
        
        preds = modelo_final.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        logger.info(f"Error Medio Absoluto Final (MAE): {mae:.2f} minutos")

        save_path = MODELS_DIR / model_name
        joblib.dump({'modelo': modelo_final, 'features': features, 'encoder': encoder}, save_path)
        logger.info(f"Modelo y Encoder guardados en {save_path}")

    def load_model(self, model_path: str) -> Dict[str, Any]:
        """Carga un modelo y sus características."""
        return joblib.load(model_path)
