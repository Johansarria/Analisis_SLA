import pytest
import pandas as pd
import os
import joblib
import numpy as np

# --- CONFIGURACIÓN DE RUTAS ---
FILES_TO_CHECK = [
    'datos_ans_limpios.csv',
    'datos_jira_regresion.csv',
    'serie_tiempo_zonas.csv',
    'modelo_ans_xgboost.pkl',
    'modelo_oraculo_xgb.pkl'
]

# 1. PRUEBA DE INFRAESTRUCTURA: ¿Existen los archivos críticos?
@pytest.mark.parametrize("file_name", FILES_TO_CHECK)
def test_file_existence(file_name):
    assert os.path.exists(file_name), f"❌ El archivo {file_name} no fue generado."

# 2. PRUEBA DE DATOS: ¿El procesamiento de Jira mantuvo la integridad?
def test_data_integrity():
    if os.path.exists('datos_jira_regresion.csv'):
        df = pd.read_csv('datos_jira_regresion.csv')
        # Verificamos que no haya valores nulos en columnas críticas
        assert df['Minutos_Resolucion'].isnull().sum() == 0, "Hay valores nulos en la variable objetivo."
        # Verificamos que tengamos las columnas necesarias para los sensores
        assert 'Nodo' in df.columns, "La columna 'Nodo' desapareció en el proceso."
        assert 'Prioridad' in df.columns, "La columna 'Prioridad' desapareció en el proceso."

# 3. PRUEBA DE MODELO (ORÁCULO): ¿El modelo de regresión puede predecir?
def test_model_prediction():
    if os.path.exists('modelo_oraculo_xgb.pkl'):
        # Cargamos el cerebro y la lista de sensores
        data = joblib.load('modelo_oraculo_xgb.pkl')
        modelo = data['modelo']
        features = data['features']
        
        # Creamos una fila de prueba con ceros (dummy data)
        X_dummy = pd.DataFrame(np.zeros((1, len(features))), columns=features)
        
        # Intentamos una predicción
        prediccion = modelo.predict(X_dummy)
        
        assert len(prediccion) == 1, "El modelo no pudo generar una predicción."
        assert isinstance(prediccion[0], (np.float32, float)), "La predicción no es un valor numérico."

# 4. PRUEBA DE LÓGICA DE NEGOCIO: ¿La proyección de fin de mes es coherente?
def test_projection_logic():
    if os.path.exists('mapa_calor_proyectado.csv'):
        df_heat = pd.read_csv('mapa_calor_proyectado.csv')
        # La proyección nunca debería ser menor a lo que ya se registró hoy
        for _, row in df_heat.iterrows():
            assert row['Proyección Abril'] >= row['Abril'], f"Error en Nodo {row['Nodo']}: La proyección es menor al real actual."

if __name__ == "__main__":
    print("\n🚀 Iniciando batería de pruebas para el Proyecto SLA...")
    pytest.main([__file__])