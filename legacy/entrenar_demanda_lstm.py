import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os, warnings, datetime

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

print("=== 🧠 INICIANDO ENJAMBRE DE IA MULTI-ZONA ===\n")
df = pd.read_csv('serie_tiempo_zonas.csv')
df['Dia_Calendario'] = pd.to_datetime(df['Dia_Calendario'])
ultima_fecha = df['Dia_Calendario'].max()

# Detectamos todas las zonas automáticamente
zonas = df.columns[1:] 
print(f"📡 Detectadas {len(zonas)} Zonas Operativas: {', '.join(zonas)}\n")

dias_semana_esp = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}

# Iniciamos el bucle: Entrenamos una IA por cada zona
for zona in zonas:
    print(f"⏳ Entrenando Red Neuronal para la Zona: [{zona}] ...")
    datos_zona = df[zona].values.reshape(-1, 1)
    promedio_zona = df[zona].mean()

    # Escalado de datos
    scaler = MinMaxScaler()
    datos_escalados = scaler.fit_transform(datos_zona)

    ventana = 7
    X, y = [], []
    for i in range(ventana, len(datos_escalados)):
        X.append(datos_escalados[i-ventana:i, 0])
        y.append(datos_escalados[i, 0])

    if len(X) == 0: continue

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Construimos un cerebro más ligero (se entrena más rápido para múltiples zonas)
    modelo = Sequential()
    modelo.add(LSTM(30, activation='relu', input_shape=(ventana, 1)))
    modelo.add(Dense(1))
    modelo.compile(optimizer='adam', loss='mse')

    # Entrenamiento express
    modelo.fit(X, y, epochs=50, verbose=0)

    # Predicción del futuro (7 días)
    ultimos_dias = datos_escalados[-ventana:]
    predicciones_futuras = []

    for _ in range(7): 
        x_input = ultimos_dias.reshape((1, ventana, 1))
        pred = modelo.predict(x_input, verbose=0)
        predicciones_futuras.append(pred[0,0])
        ultimos_dias = np.append(ultimos_dias[1:], pred, axis=0)

    pred_reales = scaler.inverse_transform(np.array(predicciones_futuras).reshape(-1, 1))

    # --- PANEL DE IMPRESIÓN REGIONAL ---
    print(f"📊 PRONÓSTICO: {zona} (Promedio normal: {promedio_zona:.0f})")
    print("-" * 65)
    
    for i, pred_val in enumerate(pred_reales):
        volumen = int(pred_val[0])
        if volumen < 0: volumen = 0 # No hay tickets negativos
        
        fecha_futura = ultima_fecha + datetime.timedelta(days=(i + 1))
        dia_str = f"{dias_semana_esp[fecha_futura.weekday()]} {fecha_futura.strftime('%d/%m')}"
        
        # Alertas relativas: Pico es +30% del promedio de esta zona, Valle es -30%
        if volumen > promedio_zona * 1.3:
            alerta = "🔴 ALERTA: Pico Operativo"
        elif volumen < promedio_zona * 0.7:
            alerta = "🔵 VALLE: Capacidad Libre"
        else:
            alerta = "🟢 NOMINAL: Tráfico normal"
            
        print(f"  {dia_str:<18} | {volumen:>3} tickets | {alerta}")
    print("=" * 65 + "\n")