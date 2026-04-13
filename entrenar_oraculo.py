import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import numpy as np
import joblib
import re
import warnings
warnings.filterwarnings('ignore')

print("1. DESPERTANDO AL ORÁCULO DE TIEMPOS...")
df = pd.read_csv('datos_jira_regresion.csv')

print("2. INGENIERÍA DE SENSORES (One-Hot Encoding)...")
y = df['Minutos_Resolucion']
X_texto = df.drop('Minutos_Resolucion', axis=1)

# Convertimos las palabras a sensores binarios
X = pd.get_dummies(X_texto)

# --- HOTFIX: SANITIZADOR DE COLUMNAS PARA XGBOOST ---
# Reemplazamos [, ], y < por guiones bajos para que XGBoost no colapse
X.columns = X.columns.str.replace(r'[\[\]<]', '_', regex=True)

columnas_finales = X.columns.tolist()
print(f"✅ Sensores generados y sanitizados: {len(columnas_finales)} variables activas.")

print("\n3. DIVISIÓN ESTRATÉGICA (Train/Test Split al 80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("4. ENTRENANDO MOTOR DE REGRESIÓN XGBOOST...")
# Usamos XGBRegressor para predecir números exactos
modelo_oraculo = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=300,             
    learning_rate=0.05,           
    max_depth=6                   
)

modelo_oraculo.fit(X_train, y_train)

print("\n5. EVALUACIÓN DE PRECISIÓN (Métricas de Error)...")
predicciones = modelo_oraculo.predict(X_test)

mae = mean_absolute_error(y_test, predicciones)
rmse = np.sqrt(mean_squared_error(y_test, predicciones))

print("\n" + "="*50)
print(" 📊 REPORTE DE PRECISIÓN DEL ORÁCULO")
print("="*50)
print(f"Margen de Error Promedio (MAE):  ± {mae / 60:.1f} Horas")
print(f"Desviación Máxima (RMSE):        ± {rmse / 60:.1f} Horas")
print("="*50)

joblib.dump({'modelo': modelo_oraculo, 'features': columnas_finales}, 'modelo_oraculo_xgb.pkl')
print("\n>>> ¡Cerebro matemático guardado! Archivo 'modelo_oraculo_xgb.pkl' generado.")