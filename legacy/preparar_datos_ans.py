import pandas as pd

print("1. INICIANDO EXTRACCIÓN DE DATOS O&M (Lectura Directa de Excel)...")

# 1. Tu archivo original exacto (Asegúrate de que esté en la misma carpeta C:\MLpractica2)
archivo_excel = 'BASE PARA DATOS  O&M FEBRERO ZONAS-REVISADO CALI (2).xlsx'

try:
    # Magia de Pandas: Leemos el Excel y le indicamos exactamente qué pestaña (sheet) queremos
    print("⏳ Leyendo archivo Excel (esto puede tomar unos segundos)...")
    df = pd.read_excel(archivo_excel, sheet_name='NACIONAL')
    print(f"✅ Pestaña 'NACIONAL' extraída. Total de tickets encontrados: {len(df)}")
except FileNotFoundError:
    print(f"❌ Error: Python no encuentra el archivo '{archivo_excel}' en esta carpeta.")
    exit()
except ValueError:
    print("❌ Error: Se encontró el Excel, pero no existe una pestaña llamada 'NACIONAL'.")
    exit()

# 2. Seleccionar los sensores (Pistas) y el Objetivo (Target)
columnas_clave = [
    'Prioridad',          
    'CENTRAL',            
    'SUEPERVISOR',        
    'CUMPLE/NO CUMPLE'    
]

# Validar que las columnas existan
for col in columnas_clave:
    if col not in df.columns:
        print(f"⚠️ Alerta: La columna '{col}' no existe en la pestaña NACIONAL. Revisa los nombres.")
        exit()

df_limpio = df[columnas_clave].copy()

# 3. Limpiar datos vacíos
df_limpio = df_limpio.dropna(subset=['CUMPLE/NO CUMPLE'])

# 4. Traducir el Objetivo a Binario (0 y 1)
df_limpio['Target'] = df_limpio['CUMPLE/NO CUMPLE'].apply(lambda x: 1 if str(x).strip().upper() == 'NO CUMPLE' else 0)

# 5. Auditoría de Balanceo
casos_incumplidos = df_limpio[df_limpio['Target'] == 1].shape[0]
casos_cumplidos = df_limpio[df_limpio['Target'] == 0].shape[0]

print("\n--- AUDITORÍA DE TICKETS ---")
print(f"Tickets limpios y listos: {len(df_limpio)}")
print(f"Casos que CUMPLIERON (0): {casos_cumplidos}")
print(f"Casos que NO CUMPLIERON RIESGO (1): {casos_incumplidos}")
print("-" * 28)

# Guardamos el resultado crudo en CSV para que el motor XGBoost lo consuma rapidísimo
df_limpio.to_csv('datos_ans_limpios.csv', index=False)
print(">>> Archivo 'datos_ans_limpios.csv' generado con éxito. Listo para Ingeniería de Características.")