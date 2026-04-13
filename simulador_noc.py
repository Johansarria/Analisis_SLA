import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=== RADAR DE RIESGO ANS (NOC) ===")

# 1. Cargar el cerebro entrenado
try:
    datos_modelo = joblib.load('modelo_ans_xgboost.pkl')
    modelo = datos_modelo['modelo']
    features_entrenamiento = datos_modelo['features']
    print("✅ Cerebro XGBoost cargado y en línea.\n")
except FileNotFoundError:
    print("❌ Error: No se encontró 'modelo_ans_xgboost.pkl'.")
    exit()

# 2. Simulamos la entrada de 3 Tickets Nuevos en el sistema de Jira
nuevos_tickets = pd.DataFrame([
    {'Ticket': 'INC-001', 'Prioridad': 'Critico', 'CENTRAL': 'Asturias', 'SUEPERVISOR': 'ANDRES CORTES'},
    {'Ticket': 'INC-002', 'Prioridad': 'Medio', 'CENTRAL': 'Progreso', 'SUEPERVISOR': 'JHON DELGADO'},
    {'Ticket': 'INC-003', 'Prioridad': 'Bajo', 'CENTRAL': 'Toberín', 'SUEPERVISOR': 'NELSON CLAVIJO'}
])

print("TICKETS RECIBIDOS EN EL MINUTO CERO:")
print(nuevos_tickets[['Ticket', 'Prioridad', 'CENTRAL', 'SUEPERVISOR']])
print("-" * 50)

# 3. Preparación de datos (Igualar al entrenamiento)
# Extraemos solo las características (sin el nombre del ticket)
X_nuevo = nuevos_tickets.drop('Ticket', axis=1)

# Aplicamos One-Hot Encoding
X_nuevo_codificado = pd.get_dummies(X_nuevo)

# CLAVE MLOps: Alinear las columnas. Si un ticket nuevo no tiene una central específica,
# debemos crear la columna y ponerla en 0 para que el modelo no colapse.
for col in features_entrenamiento:
    if col not in X_nuevo_codificado.columns:
        X_nuevo_codificado[col] = 0

# Ordenamos las columnas exactamente igual que en el entrenamiento
X_nuevo_codificado = X_nuevo_codificado[features_entrenamiento]

# 4. Inferencia (Predicción)
# predict_proba nos da la probabilidad de [SI CUMPLE, NO CUMPLE]
probabilidades = modelo.predict_proba(X_nuevo_codificado)

# 5. Panel de Resultados
for i in range(len(nuevos_tickets)):
    ticket = nuevos_tickets.iloc[i]['Ticket']
    prob_riesgo = probabilidades[i][1] * 100 # Probabilidad de NO CUMPLE (Clase 1)
    
    print(f"Ticket {ticket} | Riesgo de Incumplimiento ANS: {prob_riesgo:.1f}%")
    
    if prob_riesgo > 70:
        print("  🚨 ALERTA ROJA: Despachar apoyo preventivo o escalar caso de inmediato.")
    elif prob_riesgo > 50:
        print("  ⚠️ ALERTA AMARILLA: Monitorear de cerca. Riesgo latente.")
    else:
        print("  ✅ OPERACIÓN NOMINAL: Dejar que la cuadrilla opere normalmente.")
    print("")