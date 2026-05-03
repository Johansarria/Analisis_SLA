import pytest
import pandas as pd
import joblib
import numpy as np
from src.config import (
    FILE_ANS_CLEAN, FILE_JIRA_CLEAN, FILE_TIME_SERIES_ZONES,
    MODEL_ANS_XGB, MODEL_ORACULO_XGB
)

# --- CONFIGURACIÓN DE RUTAS ---
FILES_TO_CHECK = [
    FILE_ANS_CLEAN,
    FILE_JIRA_CLEAN,
    FILE_TIME_SERIES_ZONES,
    MODEL_ANS_XGB,
    MODEL_ORACULO_XGB
]

# 1. PRUEBA DE INFRAESTRUCTURA: ¿Existen los archivos críticos?
@pytest.mark.parametrize("file_path", FILES_TO_CHECK)
def test_file_existence(file_path):
    assert file_path.exists(), f"❌ El archivo {file_path} no fue generado."

# 2. PRUEBA DE DATOS: ¿El procesamiento de Jira mantuvo la integridad?
def test_data_integrity():
    if FILE_JIRA_CLEAN.exists():
        df = pd.read_csv(FILE_JIRA_CLEAN)
        # Verificamos que no haya valores nulos en columnas críticas
        assert df['Minutos_Resolucion'].isnull().sum() == 0, "Hay valores nulos en la variable objetivo."
        # Verificamos que tengamos las columnas necesarias
        assert 'Nodo' in df.columns, "La columna 'Nodo' desapareció."
        assert 'Prioridad' in df.columns, "La columna 'Prioridad' desapareció."

# 3. PRUEBA DE MODELO (ORÁCULO): ¿El modelo de regresión puede predecir?
def test_model_prediction():
    if MODEL_ORACULO_XGB.exists():
        data = joblib.load(MODEL_ORACULO_XGB)
        modelo = data['modelo']
        features = data['features']
        
        # Fila de prueba dummy
        X_dummy = pd.DataFrame(np.zeros((1, len(features))), columns=features)
        prediccion = modelo.predict(X_dummy)
        
        assert len(prediccion) == 1, "El modelo no generó predicción."
        assert isinstance(prediccion[0], (np.float32, float, np.float64)), "La predicción no es numérica."

# 4. PRUEBA DE LÓGICA DE NEGOCIO: ¿Los perfiles de anomalía son coherentes?
def test_anomaly_logic():
    # Esta prueba puede ser expandida para validar la lógica de AnomalyDetector
    pass
