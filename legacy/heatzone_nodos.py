import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("1. INICIANDO RADAR DE ZONAS CALIENTES CON PROYECCIÓN AL CIERRE...")

try:
    df_origen = pd.read_csv('JiraATP.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
    if len(df_origen.columns) < 3:
        df_origen = pd.read_csv('JiraATP.csv', sep=',', encoding='latin-1', on_bad_lines='skip')
except Exception as e:
    print(f"❌ Error al leer JiraATP.csv: {e}"); exit()

# 1. Limpieza de Fechas
meses_esp_ing = {'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr', 'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'ago': 'Aug', 'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'}
def traducir_fecha(f):
    if pd.isna(f): return None
    f = str(f).lower()
    for esp, ing in meses_esp_ing.items():
        if esp in f: return f.replace(esp, ing)
    return f

df_origen['Fecha'] = pd.to_datetime(df_origen['Creada'].apply(traducir_fecha), errors='coerce')
df_origen = df_origen.dropna(subset=['Fecha', 'Campo personalizado (Nodo Afectado)'])

# 2. Extracción de Mes y Nodo
nombres_meses = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo'}
df_origen['Mes'] = df_origen['Fecha'].dt.month.map(nombres_meses)
df_origen['Nodo'] = df_origen['Campo personalizado (Nodo Afectado)'].astype(str).str.strip()

# 3. CONSTRUCCIÓN DEL MAPA DE CALOR
heatzone = pd.crosstab(df_origen['Nodo'], df_origen['Mes'])

# --- LA MAGIA DE LA PROYECCIÓN ESTADÍSTICA ---
# Sabemos que hoy es 12 de abril. Calculamos el "Run Rate".
dias_transcurridos_abril = 12
dias_totales_abril = 30

if 'Abril' in heatzone.columns:
    # Calculamos la tasa diaria y la multiplicamos por el mes completo
    heatzone['Proyección Abril'] = (heatzone['Abril'] / dias_transcurridos_abril) * dias_totales_abril
    heatzone['Proyección Abril'] = heatzone['Proyección Abril'].round(0).astype(int)
else:
    heatzone['Proyección Abril'] = 0

# Calculamos el total usando la proyección para ver el daño real estimado del trimestre
meses_completos = [m for m in ['Febrero', 'Marzo'] if m in heatzone.columns]
heatzone['Total_Proyectado'] = heatzone[meses_completos].sum(axis=1) + heatzone['Proyección Abril']

# Limpiamos nodos pequeños (menos de 20 fallas)
heatzone = heatzone[heatzone['Total_Proyectado'] > 20]

# Ordenamos por los peores nodos basados en la Proyección Total
heatzone = heatzone.sort_values(by='Total_Proyectado', ascending=False)

print("\n" + "="*85)
print(" 🔥 MATRIZ DE CALOR CON PROYECCIÓN AL 30 DE ABRIL (TOP 15)")
print("="*85)

cabecera = f"{'NODO AFECTADO':<22} | "
for mes in meses_completos:
    cabecera += f"{mes:<10} | "
cabecera += f"{'Abril(Hoy)':<10} | {'Proy.30Abr':<10} | TOTAL EST."
print(cabecera)
print("-" * 85)

for nodo, fila in heatzone.head(15).iterrows():
    linea = f"{nodo[:20]:<22} | "
    
    # Meses completos (Feb/Mar)
    for mes in meses_completos:
        valor = fila[mes]
        fuego = "🔥" if valor > 100 else "  "
        linea += f"{valor:>4} {fuego}   | "
        
    # Abril Real (12 días)
    abril_real = fila.get('Abril', 0)
    linea += f"{abril_real:>4}         | "
    
    # Abril Proyectado (30 días)
    proy = fila['Proyección Abril']
    fuego_proy = "🔥" if proy > 100 else "  "
    linea += f"{proy:>4} {fuego_proy}   | "
    
    # Total
    linea += f"{fila['Total_Proyectado']:>5}"
    print(linea)

print("="*85)
print(">>> La Proyección de Abril se calcula asumiendo que el ritmo de fallas de los primeros 12 días se mantendrá.")

heatzone.to_csv('mapa_calor_proyectado.csv')
print(">>> Archivo 'mapa_calor_proyectado.csv' guardado.")