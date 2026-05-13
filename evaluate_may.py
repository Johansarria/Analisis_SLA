import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from category_encoders import TargetEncoder

from src.data.etl import DataProcessor
import re
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = Path(r"C:\MLpractica2\data\processed")
FILE_TRAIN_RAW = DATA_PATH / "Jira ATP marzo-abril.csv"
FILE_TEST_RAW = DATA_PATH / "Jira ATP mayo.csv"

def sanitize_columns(df):
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    df.columns = [regex.sub("_", col) if any(x in str(col) for x in "[]<") else col for col in df.columns]
    return df

def run_evaluation():
    print("Iniciando procesamiento de datos...")
    processor = DataProcessor()
    
    # Cargar y limpiar datos de entrenamiento (Marzo-Abril)
    try:
        df_train_full = processor.process_jira(str(FILE_TRAIN_RAW))
        df_test_full = processor.process_jira(str(FILE_TEST_RAW))
    except Exception as e:
        print(f"Error al procesar archivos: {e}")
        return
    
    # Asegurarnos de tener las columnas cíclicas
    for df in [df_train_full, df_test_full]:
        if 'Hora_sin' not in df.columns:
            df['Hora_sin'] = np.sin(2 * np.pi * df['Hora_Del_Dia'] / 24.0)
            df['Hora_cos'] = np.cos(2 * np.pi * df['Hora_Del_Dia'] / 24.0)
            df['Dia_sin'] = np.sin(2 * np.pi * df['Dia_De_La_Semana'] / 7.0)
            df['Dia_cos'] = np.cos(2 * np.pi * df['Dia_De_La_Semana'] / 7.0)
            
    cols_for_model = [
        'Prioridad', 'Nodo', 'Usuarios_Afectados', 
        'Tipo_Falla', 'Hora_sin', 'Hora_cos', 'Dia_sin', 'Dia_cos'
    ]
    
    cols_for_model = [c for c in cols_for_model if c in df_train_full.columns]
    
    X_train_raw = df_train_full[cols_for_model].copy()
    y_train = df_train_full['Minutos_Resolucion']
    
    X_test_raw = df_test_full[cols_for_model].copy()
    y_test = df_test_full['Minutos_Resolucion']
    
    print(f"Datos de Entrenamiento (Mar-Abr): {len(X_train_raw)} filas")
    print(f"Datos de Validación (Mayo): {len(X_test_raw)} filas")
    
    # Target Encoding
    cat_cols = X_train_raw.select_dtypes(include=['object', 'category']).columns.tolist()
    encoder = TargetEncoder(cols=cat_cols)
    
    X_train = encoder.fit_transform(X_train_raw, y_train)
    X_train = sanitize_columns(X_train)
    
    X_test = encoder.transform(X_test_raw)
    X_test = sanitize_columns(X_test)
    
    # Parámetros óptimos encontrados previamente
    params = {
        'n_estimators': 118, 
        'learning_rate': 0.034236, 
        'max_depth': 9, 
        'subsample': 0.81929, 
        'random_state': 42
    }
    
    print("\nEntrenando el modelo Oráculo con los datos hasta Abril...")
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    print("Validando el modelo con los datos nuevos de Mayo...")
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    print(f"\n--- RESULTADOS DE VALIDACIÓN (MAYO) ---")
    print(f"Error Medio Absoluto (MAE) en datos no vistos: {mae:.2f} minutos")
    
    # Calcular promedios para dar contexto
    promedio_real = y_test.mean()
    print(f"Tiempo promedio de resolución real en Mayo: {promedio_real:.2f} minutos")
    print(f"Desviación del modelo sobre la media real: {(mae / promedio_real) * 100:.2f}%")

if __name__ == "__main__":
    run_evaluation()
