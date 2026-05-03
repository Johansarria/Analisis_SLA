import pandas as pd
import warnings
warnings.filterwarnings('ignore')

archivo_csv = 'JiraATP.csv'

print(f"=== INICIANDO SONDA DE EXPLORACIÓN EN: {archivo_csv} ===\n")

try:
    # Intentamos leer el archivo. A veces Jira usa punto y coma (;) en vez de coma (,).
    # Este bloque intenta descubrir el separador automáticamente.
    try:
        df = pd.read_csv(archivo_csv)
        if len(df.columns) < 3: # Si solo lee 1 o 2 columnas gigantes, el separador está mal
            df = pd.read_csv(archivo_csv, sep=';')
    except Exception:
        # Plan B de lectura por si hay caracteres especiales
        df = pd.read_csv(archivo_csv, sep=';', encoding='latin-1')

    print("✅ Archivo leído con éxito.")
    print(f"📊 Total de tickets encontrados: {len(df)}")
    print(f"🗂️ Total de columnas detectadas: {len(df.columns)}\n")
    
    print("-" * 50)
    print("📋 LISTA EXACTA DE COLUMNAS DISPONIBLES:")
    print("-" * 50)
    
    # Imprimimos todas las columnas en forma de lista enumerada
    for i, col in enumerate(df.columns.tolist(), start=1):
        print(f"{i}. '{col}'")
        
    print("\n" + "-" * 50)
    print("🔍 MUESTRA DEL PRIMER TICKET (Para ver el formato de los datos):")
    print("-" * 50)
    # Mostramos los valores de la primera fila
    primer_ticket = df.iloc[0]
    for col in df.columns[:15]: # Solo las primeras 15 columnas para no saturar la pantalla
        print(f"{col}: {primer_ticket[col]}")
        
except FileNotFoundError:
    print(f"❌ Error: El archivo '{archivo_csv}' no está en la carpeta actual.")
except Exception as e:
    print(f"❌ Error inesperado: {e}")