import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error
import os, warnings

# 1. Silenciar alertas técnicas para un reporte limpio
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

print("=== 🧪 TEST DE ESTRÉS: EVALUANDO PRECISIÓN OPERATIVA (WAPE) ===\n")

# Cargar la base de datos por zonas
if not os.path.exists('serie_tiempo_zonas.csv'):
    print("❌ Error: No se encuentra 'serie_tiempo_zonas.csv'. Ejecuta primero el ETL.")
    exit()

df = pd.read_csv('serie_tiempo_zonas.csv')
zonas = df.columns[1:] # Omitimos la columna de fecha

for zona in zonas:
    # Preparación de datos por zona
    datos = df[zona].values.reshape(-1, 1)
    
    # Si la zona tiene muy pocos datos históricos, no se puede validar con rigor
    if len(datos) < 15:
        print(f"📍 ZONA: {zona} | ⚠️ Datos insuficientes para validación científica.")
        print("-" * 60)
        continue

    scaler = MinMaxScaler()
    datos_esc = scaler.fit_transform(datos)

    # Crear secuencias (Ventana de 7 días)
    ventana = 7
    X, y = [], []
    for i in range(ventana, len(datos_esc)):
        X.append(datos_esc[i-ventana:i, 0])
        y.append(datos_esc[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # DIVISIÓN DE DATOS: 
    # Entrenamos con el 85% de la historia.
    # El 15% restante (lo más reciente) es el "examen final" para la IA.
    split = int(len(X) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Construcción del modelo rápido
    model = Sequential([
        LSTM(30, activation='relu', input_shape=(ventana, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Entrenamiento silencioso
    model.fit(X_train, y_train, epochs=50, batch_size=4, verbose=0)

    # Predicción sobre el set de prueba
    pred_esc = model.predict(X_test, verbose=0)
    
    # Volvemos a la escala real de tickets de Jira
    pred_real = scaler.inverse_transform(pred_esc)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

    # --- CÁLCULO DE MÉTRICAS OPERATIVAS ---
    
    # 1. MAE: Error absoluto medio (En cuántos tickets se equivoca por día)
    mae = mean_absolute_error(y_test_real, pred_real)
    
    # 2. WAPE: Weighted Absolute Percentage Error (Precisión ponderada)
    # Evita el error de división por cero de las zonas pequeñas
    suma_errores = np.sum(np.abs(y_test_real - pred_real))
    suma_reales = np.sum(y_test_real)
    
    if suma_reales > 0:
        wape = suma_errores / suma_reales
        precision_operativa = max(0, (1 - wape) * 100)
    else:
        precision_operativa = 0

    # Imprimir reporte formateado
    print(f"📍 ZONA: {zona}")
    print(f"   |-- Precisión Operativa: {precision_operativa:.1f}%")
    print(f"   |-- Margen de error:    ± {mae:.1f} tickets diarios")
    print(f"   |-- Volumen testeo:     {int(suma_reales)} tickets analizados")
    print("-" * 60)

print("\n>>> Interpretación Logística:")
print("🟢 > 85%: Excelente para planeación de cuadrillas y turnos.")
print("🟡 70% - 85%: Útil como guía, pero requiere supervisión humana.")
print("🔴 < 70%: Datos muy volátiles; usar solo como referencia de tendencia.")