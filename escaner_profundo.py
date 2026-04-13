import pandas as pd
import warnings
warnings.filterwarnings('ignore')

archivo_csv = 'JiraATP.csv'

print(f"=== ESCÁNER PROFUNDO DE DATOS: {archivo_csv} ===\n")

try:
    # Intento de lectura robusta
    try:
        df = pd.read_csv(archivo_csv)
        if len(df.columns) < 3: 
            df = pd.read_csv(archivo_csv, sep=';')
    except Exception:
        df = pd.read_csv(archivo_csv, sep=';', encoding='latin-1')

    print("✅ Archivo cargado correctamente.")
    print(f"📊 Total de Tickets (Filas): {df.shape[0]}")
    print(f"🗂️ Total de Columnas: {df.shape[1]}")
    
    # 1. Lista completa de columnas y calidad de datos
    print("\n" + "="*55)
    print(" 📋 INVENTARIO DE COLUMNAS (Auditoría de Calidad)")
    print("="*55)
    
    for i, col in enumerate(df.columns, start=1):
        # Calculamos qué porcentaje de esta columna está vacío
        porcentaje_nulos = df[col].isnull().sum() / len(df) * 100
        # Imprimimos el nombre y qué tan "vacía" está la columna
        print(f"{i:02d}. {col[:35]:<35} | Faltan datos: {porcentaje_nulos:5.1f}%")

    # 2. Muestra de 2 tickets completos (filtrando campos vacíos)
    print("\n" + "="*55)
    print(" 🔍 RADIOGRAFÍA DE 2 TICKETS AL AZAR")
    print("="*55)
    
    # Tomamos 2 muestras aleatorias (random_state asegura que siempre salgan los mismos 2 si repites)
    muestras = df.sample(n=2, random_state=42)
    
    for idx, (indice, fila) in enumerate(muestras.iterrows(), start=1):
        print(f"\n--- TICKET DE MUESTRA {idx} ---")
        # Mostramos solo las columnas que SI tienen algún dato útil
        for col in df.columns:
            valor = fila[col]
            if pd.notna(valor) and str(valor).strip() != "" and str(valor).lower() != "nan":
                # Acortamos textos muy largos para no saturar tu pantalla
                texto_mostrar = str(valor)[:150] + "..." if len(str(valor)) > 150 else str(valor)
                print(f"  {col}: {texto_mostrar}")

except FileNotFoundError:
    print(f"❌ Error: No encuentro el archivo '{archivo_csv}' en tu carpeta C:\MLpractica2.")
except Exception as e:
    print(f"❌ Error técnico al procesar: {e}")