import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error
import joblib
from typing import Dict, List, Any, Tuple
import logging
from src.config import MODELS_DIR

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

import re

class ModelTrainer:
    """Clase para el entrenamiento y gestión de modelos de ML."""

    def _sanitize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia los nombres de las columnas para compatibilidad con XGBoost."""
        regex = re.compile(r"\[|\]|<", re.IGNORECASE)
        df.columns = [regex.sub("_", col) if any(x in str(col) for x in "[]<") else col for col in df.columns]
        return df

    def train_ans_classifier(self, df: pd.DataFrame, model_name: str = "modelo_ans_xgboost.pkl"):
        """Entrena un clasificador XGBoost para cumplimiento de ANS."""
        logger.info("Entrenando clasificador de ANS...")
        
        # Preparación de datos
        X_raw = df.drop(['Target', 'CUMPLE/NO CUMPLE'], axis=1, errors='ignore')
        y = df['Target']
        X = pd.get_dummies(X_raw)
        X = self._sanitize_columns(X)
        features = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # Balanceo de clases
        pos_count = len(y_train[y_train == 0])
        neg_count = len(y_train[y_train == 1])
        scale_weight = pos_count / neg_count

        modelo = xgb.XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss',
            scale_pos_weight=scale_weight,
            max_depth=5,
            learning_rate=0.05,
            n_estimators=200
        )

        modelo.fit(X_train, y_train)
        
        # Evaluación
        preds = modelo.predict(X_test)
        logger.info("\n" + classification_report(y_test, preds))

        # Guardado
        save_path = MODELS_DIR / model_name
        joblib.dump({'modelo': modelo, 'features': features}, save_path)
        logger.info(f"Modelo guardado en {save_path}")

    def train_oraculo_regressor(self, df: pd.DataFrame, model_name: str = "modelo_oraculo_xgb.pkl"):
        """Entrena un regresor XGBoost para predicción de tiempos (Oráculo)."""
        logger.info("Entrenando regresor Oráculo...")
        
        # Solo usamos las columnas de interés para el modelo
        cols_for_model = [
            'Prioridad', 'Nodo', 'Usuarios_Afectados', 
            'Tipo_Falla', 'Hora_Del_Dia', 'Dia_De_La_Semana'
        ]
        X_raw = df[cols_for_model].copy()
        y = df['Minutos_Resolucion']
        
        # Encoding de categorías
        X = pd.get_dummies(X_raw)
        X = self._sanitize_columns(X)
        features = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        modelo = xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=6,
            random_state=42
        )

        modelo.fit(X_train, y_train)
        
        # Evaluación
        preds = modelo.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        logger.info(f"Error Medio Absoluto (MAE): {mae:.2f} minutos")

        # Guardado
        save_path = MODELS_DIR / model_name
        joblib.dump({'modelo': modelo, 'features': features}, save_path)
        logger.info(f"Modelo guardado en {save_path}")

    def load_model(self, model_path: str) -> Dict[str, Any]:
        """Carga un modelo y sus características."""
        return joblib.load(model_path)
