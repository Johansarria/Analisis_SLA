import logging
from typing import Dict, Any, Union
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from src.core.models import ModelTrainer

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Motor de inferencia para predicción con modelos entrenados."""

    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Carga modelo, encoder y lista de features desde el archivo .pkl."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")

        data = joblib.load(self.model_path)
        self.model = data['modelo']
        self.encoder = data.get('encoder')
        self.features = data.get('features', [])

        if not self.features:
            raise ValueError("El modelo no contiene la lista de features esperadas.")

        logger.info(f"Modelo cargado desde {self.model_path} con {len(self.features)} features.")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica el encoder almacenado y alinea las columnas con las features del modelo.

        category_encoders valida que la entrada tenga exactamente las mismas columnas
        usadas en el fit (pasa las numericas sin tocarlas), por eso se reconstruye la
        entrada con las columnas del entrenamiento antes de transformar.
        """
        X = df.copy()

        if self.encoder is not None:
            input_cols = getattr(self.encoder, 'feature_names_in_', None)
            if input_cols is None:
                input_cols = getattr(self.encoder, 'cols', [])
            input_cols = list(input_cols)

            for col in input_cols:
                if col not in X.columns:
                    X[col] = 'MISSING'
            X = self.encoder.transform(X[input_cols] if input_cols else X)

            output_cols = getattr(self.encoder, 'feature_names_out_', None)
            output_cols = list(output_cols) if output_cols is not None else []
            if len(output_cols) == len(X.columns):
                X.columns = output_cols

        # Sanitizar nombres (mismo paso que en entrenamiento)
        X = ModelTrainer._sanitize_columns(X)

        # Alinear columnas: agregar faltantes con 0, eliminar sobrantes
        for col in self.features:
            if col not in X.columns:
                X[col] = 0
        X = X[self.features]
        return X

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Genera predicciones para el DataFrame de entrada."""
        X = self._prepare_features(df)
        return self.model.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Genera probabilidades de clase (solo para clasificadores)."""
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError("El modelo cargado no soporta predict_proba.")
        X = self._prepare_features(df)
        return self.model.predict_proba(X)
