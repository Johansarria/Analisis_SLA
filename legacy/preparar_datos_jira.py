import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("1. INICIANDO ETL DE PRECISIÓN (JIRA ATP)...")
archivo_csv = 'JiraATP.csv'

# 1. Cargar datos
try:
    try:
        df = pd.read_csv(archivo_csv)
        if len(df.columns) < 3: df = pd.read_csv(archivo_csv, sep=';')
    except:
        df = pd.read_csv(archivo_csv, sep=';', encoding='latin-1')
except Exception as e:
    print(f"❌ Error al leer: {e}"); exit()

# 2. SELECCIÓN DE COLUMNAS EXACTAS SEGÚN ESCÁNER
columnas_necesarias = [
    'Resumen',
    'Prioridad',
    'Creada',
    'Campo personalizado (Solución afectación )',
    'Campo personalizado (Nodo Afectado)',
    'Campo personalizado (Usuarios Iniciales Afectados)'
]

df_limpio = df[columnas_necesarias].copy()

# Renombramos para que sea más fácil trabajar
df_limpio.columns = ['Resumen', 'Prioridad', 'Fecha_Creacion', 'Fecha_Solucion', 'Nodo', 'Usuarios_Afectados']

# Eliminamos filas que no tengan fecha de solución (casos aún abiertos)
df_limpio = df_limpio.dropna(subset=['Fecha_Solucion', 'Fecha_Creacion'])

# 3. TRADUCCIÓN DE FECHAS (Español a Formato Máquina)
print("⏳ Convirtiendo fechas y calculando tiempos de resolución...")
meses_esp_ing = {
    'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr', 
    'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'ago': 'Aug', 
    'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
}

def traducir_fecha(fecha_str):
    if pd.isna(fecha_str): return np.nan
    fecha_str = str(fecha_str).lower()
    for esp, ing in meses_esp_ing.items():
        if esp in fecha_str:
            fecha_str = fecha_str.replace(esp, ing)
            break
    return fecha_str

df_limpio['Fecha_Creacion'] = pd.to_datetime(df_limpio['Fecha_Creacion'].apply(traducir_fecha), errors='coerce')
df_limpio['Fecha_Solucion'] = pd.to_datetime(df_limpio['Fecha_Solucion'].apply(traducir_fecha), errors='coerce')

# Filtramos errores de formato
df_limpio = df_limpio.dropna(subset=['Fecha_Creacion', 'Fecha_Solucion'])

# 4. EL MOTOR MATEMÁTICO: Resta de fechas
# Restamos la solución menos la creación para obtener los minutos exactos
diferencia = df_limpio['Fecha_Solucion'] - df_limpio['Fecha_Creacion']
df_limpio['Minutos_Resolucion'] = diferencia.dt.total_seconds() / 60.0

# 5. INGENIERÍA DE CARACTERÍSTICAS
# Extraer tipo de falla del resumen
df_limpio['Tipo_Falla'] = df_limpio['Resumen'].apply(lambda x: str(x).split('/')[-1].strip().upper()[:30])

# Extraer el reloj biológico de la red
df_limpio['Hora_Del_Dia'] = df_limpio['Fecha_Creacion'].dt.hour
df_limpio['Dia_De_La_Semana'] = df_limpio['Fecha_Creacion'].dt.dayofweek # 0=Lunes, 6=Domingo
df_limpio['Usuarios_Afectados'] = pd.to_numeric(df_limpio['Usuarios_Afectados'], errors='coerce').fillna(1)

# 6. FILTRO DE ANOMALÍAS (Outliers Operativos)
# Un caso no se resuelve en 0 minutos (error humano) ni en 20 días (ticket fantasma)
# Filtramos solo casos reales (entre 15 minutos y 5 días - 7200 min)
df_final = df_limpio[(df_limpio['Minutos_Resolucion'] >= 15) & (df_limpio['Minutos_Resolucion'] <= 7200)]

print("\n" + "="*45)
print(" 📊 AUDITORÍA DEL ORÁCULO DE TIEMPOS")
print("="*45)
print(f"Tickets listos para entrenar IA: {len(df_final)}")
print(f"Tiempo Promedio de Resolución:   {df_final['Minutos_Resolucion'].mean() / 60:.1f} Horas")
print(f"Falla más rápida reportada:      {df_final['Minutos_Resolucion'].min():.0f} minutos")
print(f"Falla más demorada (límite):     {df_final['Minutos_Resolucion'].max() / 60:.1f} Horas")
print("="*45)

# Limpiamos las columnas que ya no necesita el modelo predictivo
columnas_exportar = ['Prioridad', 'Nodo', 'Usuarios_Afectados', 'Tipo_Falla', 'Hora_Del_Dia', 'Dia_De_La_Semana', 'Minutos_Resolucion']
df_final[columnas_exportar].to_csv('datos_jira_regresion.csv', index=False)
print(">>> Archivo 'datos_jira_regresion.csv' exportado. El terreno está listo.")