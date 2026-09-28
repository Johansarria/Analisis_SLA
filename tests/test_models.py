import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.core.models import ModelTrainer
from src.config import MODELS_DIR


def test_shouldTrainAndSaveAnsClassifierWithEncoder(df_ans: pd.DataFrame, tmp_path: Path):
    """El clasificador ANS debe entrenarse y guardarse con encoder y features."""
    trainer = ModelTrainer()
    model_path = tmp_path / "test_ans_model.pkl"

    trainer.train_ans_classifier(df_ans, model_name=str(model_path.name))

    # El modelo se guarda en MODELS_DIR, no en tmp_path; lo cargamos desde allí
    full_path = MODELS_DIR / model_path.name
    assert full_path.exists(), "El archivo del modelo no fue creado"

    data = joblib.load(full_path)
    assert 'modelo' in data
    assert 'encoder' in data
    assert 'features' in data
    assert len(data['features']) > 0

    # Cleanup
    full_path.unlink(missing_ok=True)


def test_shouldTrainAndSaveOracleRegressorWithEncoder(df_jira: pd.DataFrame, tmp_path: Path):
    """El regresor Oráculo debe entrenarse y guardarse con encoder y features."""
    trainer = ModelTrainer()
    model_path = tmp_path / "test_oracle_model.pkl"

    trainer.train_oraculo_regressor(df_jira, model_name=str(model_path.name))

    full_path = MODELS_DIR / model_path.name
    assert full_path.exists(), "El archivo del modelo no fue creado"

    data = joblib.load(full_path)
    assert 'modelo' in data
    assert 'encoder' in data
    assert 'features' in data
    assert len(data['features']) > 0

    # Cleanup
    full_path.unlink(missing_ok=True)


def test_shouldGenerateNumericPredictionWhenLoadingOracleModel(df_jira: pd.DataFrame, tmp_path: Path):
    """Al cargar el modelo Oráculo, una fila dummy debe producir una predicción numérica."""
    trainer = ModelTrainer()
    model_name = "test_oracle_pred.pkl"
    trainer.train_oraculo_regressor(df_jira, model_name=model_name)

    full_path = MODELS_DIR / model_name
    data = joblib.load(full_path)
    modelo = data['modelo']
    encoder = data['encoder']
    features = data['features']

    # Construir dummy con las mismas columnas crudas usadas en el entrenamiento
    cols_for_model = ['Prioridad', 'Nodo', 'Usuarios_Afectados', 'Tipo_Falla',
                      'Hora_sin', 'Hora_cos', 'Dia_sin', 'Dia_cos']
    X_dummy = df_jira[cols_for_model].head(1)
    X_encoded = encoder.transform(X_dummy)
    # Alinear con features del modelo
    for col in features:
        if col not in X_encoded.columns:
            X_encoded[col] = 0
    X_encoded = X_encoded[features]

    prediccion = modelo.predict(X_encoded)
    assert len(prediccion) == 1
    assert isinstance(prediccion[0], (np.floating, float))

    full_path.unlink(missing_ok=True)
