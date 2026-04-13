import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("1. INICIANDO COMPRESIÓN TEMPORAL (Volumen Diario)...")

try:
    print("⏳ Extrayendo calendario (Modo Bulldozer: ignorando tickets malformados)...")
    # on_bad_lines='skip' es la magia que evitará que el script colapse
    df_origen = pd.read_csv('JiraATP.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
    
    # Validamos si el separador real era coma
    if len(df_origen.columns) < 3:
        df_origen = pd.read_csv('JiraATP.csv', sep=',', encoding='latin-1', on_bad_lines='skip')
        
except Exception as e:
    print(f"❌ Error crítico al leer JiraATP.csv: {e}")
    exit()

if 'Creada' not in df_origen.columns:
    print("❌ Error: No encuentro la columna 'Creada'. Revisa el separador.")
    exit()

# Traductor de meses
meses_esp_ing = {
    'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr', 
    'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'ago': 'Aug', 
    'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
}

def traducir_fecha(fecha_str):
    if pd.isna(fecha_str): return None
    fecha_str = str(fecha_str).lower()
    for esp, ing in meses_esp_ing.items():
        if esp in fecha_str: 
            return fecha_str.replace(esp, ing)
    return fecha_str

print("⏳ Agrupando y contando tickets por día...")
# Limpieza de fechas
df_origen['Fecha_Limpia'] = pd.to_datetime(df_origen['Creada'].apply(traducir_fecha), errors='coerce')
df_origen = df_origen.dropna(subset=['Fecha_Limpia'])

# Extraemos solo el Día
df_origen['Dia_Calendario'] = df_origen['Fecha_Limpia'].dt.date

# Contamos el volumen diario
demanda_diaria = df_origen.groupby('Dia_Calendario').size().reset_index(name='Total_Tickets')
demanda_diaria = demanda_diaria.sort_values('Dia_Calendario')

print("\n" + "="*45)
print(" 📈 AUDITORÍA DE DEMANDA DIARIA")
print("="*45)
print(f"Días de operación registrados: {len(demanda_diaria)} días")
print(f"Día de mayor caos (Pico):      {demanda_diaria['Total_Tickets'].max()} tickets")
print(f"Día más tranquilo (Valle):     {demanda_diaria['Total_Tickets'].min()} tickets")
print(f"Promedio de tickets diarios:   {demanda_diaria['Total_Tickets'].mean():.0f} tickets")
print("="*45)

# Guardar base
demanda_diaria.to_csv('serie_tiempo_jira.csv', index=False)
print(">>> Archivo 'serie_tiempo_jira.csv' exportado. Listo para la Red Neuronal.")