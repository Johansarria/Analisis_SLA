import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

print("1. CARGANDO MATERIA PRIMA...")
df = pd.read_csv('datos_ans_limpios.csv')

print("2. INGENIERÍA DE CARACTERÍSTICAS (Limpieza de Fuga de Datos)...")
# ELIMINAMOS 'Target' (respuesta numérica) y 'CUMPLE/NO CUMPLE' (respuesta en texto)
# Esto obliga al modelo a predecir usando solo las variables operativas.
X_texto = df.drop(['Target', 'CUMPLE/NO CUMPLE'], axis=1)
y = df['Target']

# Aplicamos One-Hot Encoding: Convierte categorías en columnas de 0s y 1s
X = pd.get_dummies(X_texto)
columnas_finales = X.columns.tolist()
print(f"✅ Sensores generados: {len(columnas_finales)} variables activas.")

print("\n3. DIVISIÓN ESTRATÉGICA (Train/Test Split)...")
# Separamos el 20% para la evaluación final
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print("4. CALIBRANDO PESOS (Balanceo de Clases)...")
casos_cumplen = len(y_train[y_train == 0])
casos_no_cumplen = len(y_train[y_train == 1])

# Factor de compensación para que la IA no ignore los casos que 'SI CUMPLEN'
peso_compensacion = casos_cumplen / casos_no_cumplen 
print(f"   Factor de compensación aplicado: {peso_compensacion:.4f}")

print("\n5. ENTRENANDO IA XGBOOST (Detección de Riesgos Real)...")
modelo_ans = xgb.XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss',
    scale_pos_weight=peso_compensacion, 
    max_depth=5,            # Un poco más de profundidad para captar patrones complejos
    learning_rate=0.05,     # Aprendizaje más pausado y preciso
    n_estimators=200        # Más árboles para compensar la dificultad de los datos
)

modelo_ans.fit(X_train, y_train)

print("\n6. PRUEBA DE FUEGO (Resultados Reales de Operación)...")
predicciones = modelo_ans.predict(X_test)

print("\n" + "="*50)
print(" 📊 REPORTE DE PRECISIÓN DE O&M (MÉTRICAS REALES)")
print("="*50)
print(classification_report(y_test, predicciones, target_names=['SI CUMPLE (0)', 'NO CUMPLE (1)']))

# Guardamos el cerebro corregido
joblib.dump({'modelo': modelo_ans, 'features': columnas_finales}, 'modelo_ans_xgboost.pkl')
print("\n>>> ¡Cerebro guardado! Archivo 'modelo_ans_xgboost.pkl' generado.")