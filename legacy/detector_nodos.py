import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

print("1. INICIANDO RADAR DE ANOMALÍAS (Mantenimiento Preventivo)...")

# Cargamos la base de datos limpia que hicimos en el primer ETL
try:
    df = pd.read_csv('datos_jira_regresion.csv')
except FileNotFoundError:
    print("❌ Error: No encuentro 'datos_jira_regresion.csv'.")
    exit()

print("2. CONSTRUYENDO PERFILES DE SALUD POR NODO...")
# Agrupamos los datos para crear un "Historial Clínico" de cada Nodo
perfiles_nodos = df.groupby('Nodo').agg(
    Total_Fallas=('Tipo_Falla', 'count'),
    Tiempo_Promedio_Minutos=('Minutos_Resolucion', 'mean'),
    # Calculamos cuántas fallas fueron Críticas o Altas
    Fallas_Criticas=('Prioridad', lambda x: (x.str.upper() == 'ALTA').sum() + (x.str.upper() == 'CRÍTICA').sum())
).reset_index()

# Calculamos el porcentaje de criticidad
perfiles_nodos['Porcentaje_Critico'] = (perfiles_nodos['Fallas_Criticas'] / perfiles_nodos['Total_Fallas']) * 100

# Filtramos nodos que solo tuvieron 1 o 2 fallas aisladas en 3 meses (no forman un patrón)
perfiles_nodos = perfiles_nodos[perfiles_nodos['Total_Fallas'] > 5].reset_index(drop=True)
sensores = perfiles_nodos[['Total_Fallas', 'Tiempo_Promedio_Minutos', 'Porcentaje_Critico']]

print(f"✅ Se evaluarán {len(perfiles_nodos)} Nodos activos.")

print("\n3. AGRUPACIÓN ESPACIAL (Clustering K-Means)...")
# Estandarizamos los datos (poner todo en la misma escala para la IA)
scaler = StandardScaler()
sensores_escalados = scaler.fit_transform(sensores)

# Le pedimos a la IA que divida la red en 3 grupos (Clusters) de salud
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
perfiles_nodos['Grupo_IA'] = kmeans.fit_predict(sensores_escalados)

# Identificamos automáticamente cuál es el grupo "Rojo" (El que tiene el promedio de fallas más alto)
promedios_grupos = perfiles_nodos.groupby('Grupo_IA')['Total_Fallas'].mean()
grupo_critico = promedios_grupos.idxmax()

# Extraemos solo los nodos que cayeron en la zona roja
nodos_en_peligro = perfiles_nodos[perfiles_nodos['Grupo_IA'] == grupo_critico]

print("\n" + "="*65)
print(" 🚨 RADAR DE MANTENIMIENTO PREVENTIVO (BOMBAS DE TIEMPO)")
print("="*65)
print(f"La IA ha detectado {len(nodos_en_peligro)} Nodos con comportamiento anómalo y degradación.\n")

# Ordenamos del más crítico al menos crítico y mostramos el Top 10
top_peligro = nodos_en_peligro.sort_values(by='Total_Fallas', ascending=False).head(10)

print(f"{'NODO AFECTADO':<30} | {'TOTAL FALLAS':<14} | {'% CRÍTICAS':<10}")
print("-" * 65)
for index, row in top_peligro.iterrows():
    print(f"{str(row['Nodo'])[:28]:<30} | {row['Total_Fallas']:<14} | {row['Porcentaje_Critico']:.1f}%")
print("="*65)
print(">>> Acción recomendada: Enviar cuadrillas a revisar empalmes y potencia en estos nodos.")
