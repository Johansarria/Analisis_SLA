import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración de estilo
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]

print("📊 GENERANDO TABLERO VISUAL DE OPERACIONES...")

# --- GRÁFICO 1: BARRAS - HEAT ZONE PROYECTADO ---
if os.path.exists('mapa_calor_proyectado.csv'):
    df_heat = pd.read_csv('mapa_calor_proyectado.csv')
    # Tomamos los top 10 peores nodos
    top_10 = df_heat.head(10)

    # Creamos el gráfico de barras comparativo
    fig, ax = plt.subplots()
    
    # Configuración de las barras
    nodos = top_10['Nodo'].str.slice(0, 15) # Acortamos nombres largos
    x = range(len(nodos))
    width = 0.35

    # Barras de Real vs Proyectado
    rects1 = ax.bar([i - width/2 for i in x], top_10['Abril'], width, label='Real (al día 12)', color='#3498db')
    rects2 = ax.bar([i + width/2 for i in x], top_10['Proyección Abril'], width, label='Proyectado (al día 30)', color='#e74c3c')

    ax.set_ylabel('Cantidad de Tickets')
    ax.set_title('ANÁLISIS DE RIESGO ABRIL: REAL VS PROYECTADO (TOP 10 NODOS)')
    ax.set_xticks(x)
    ax.set_xticklabels(nodos, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.savefig('grafico_riesgo_nodos.png')
    print("✅ Gráfico de barras guardado como 'grafico_riesgo_nodos.png'")
    plt.show()

# --- GRÁFICO 2: PASTEL - CARGA OPERATIVA POR ZONA ---
if os.path.exists('serie_tiempo_zonas.csv'):
    df_zonas = pd.read_csv('serie_tiempo_zonas.csv')
    # Sumamos el total de tickets por zona en todo el periodo
    zonas_totales = df_zonas.drop('Dia_Calendario', axis=1).sum().sort_values(ascending=False)

    plt.figure(figsize=(10, 7))
    colores = sns.color_palette('pastel')[0:len(zonas_totales)]
    
    plt.pie(zonas_totales, 
            labels=zonas_totales.index, 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=colores,
            explode=[0.1 if i == 0 else 0 for i in range(len(zonas_totales))]) # Resalta la zona más grande

    plt.title('DISTRIBUCIÓN DE CARGA OPERATIVA POR ZONA (TRIMESTRE)')
    plt.savefig('grafico_distribucion_zonas.png')
    print("✅ Gráfico de pastel guardado como 'grafico_distribucion_zonas.png'")
    plt.show()

print("\n>>> ¡Visualizaciones completadas! Ya tienes material para tu presentación.")