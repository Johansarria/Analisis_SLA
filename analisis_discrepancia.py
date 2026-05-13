import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/demanda_diaria.csv')
df['Fecha'] = pd.to_datetime(df['Fecha'])
cutoff = pd.to_datetime('2026-04-12')

zcen = df[df['Zona'] == 'Zona Centro']

# Autocorrelacion por nodo
print("=== AUTOCORRELACION POR NODO (Zona Centro) ===")
for nodo in zcen['Nodo'].unique():
    nodo_data = zcen[zcen['Nodo'] == nodo].sort_values('Fecha')['Tickets_Total']
    a1 = nodo_data.autocorr(lag=1) if len(nodo_data) > 1 else 0
    a7 = nodo_data.autocorr(lag=7) if len(nodo_data) > 7 else 0
    print(f"  {nodo}: lag1={a1:.3f}, lag7={a7:.3f}")

# Desglose picos
print("\n=== DESGLOSE PICOS (>55 tickets) ===")
daily = zcen.groupby('Fecha')['Tickets_Total'].sum().reset_index()
test_daily = daily[daily['Fecha'] >= cutoff]
picos = test_daily[test_daily['Tickets_Total'] > 55]
for _, row in picos.iterrows():
    fecha = row['Fecha']
    nodos = zcen[zcen['Fecha'] == fecha][['Nodo', 'Tickets_Total']]
    print(f"\n  {fecha.strftime('%Y-%m-%d')} (Total={int(row['Tickets_Total'])}):")
    for _, nr in nodos.iterrows():
        print(f"    {nr['Nodo']}: {int(nr['Tickets_Total'])}")

# Distribucion dia de la semana en test
print("\n=== TICKETS POR DIA DE SEMANA (ZONA CENTRO, periodo test) ===")
test_zcen = zcen[zcen['Fecha'] >= cutoff]
daily_test = test_zcen.groupby('Fecha').agg({'Tickets_Total': 'sum', 'Dia_Semana': 'first'}).reset_index()
for dow in range(7):
    dias = ['Lun','Mar','Mie','Jue','Vie','Sab','Dom']
    subset = daily_test[daily_test['Dia_Semana'] == dow]
    print(f"  {dias[dow]}: media={subset['Tickets_Total'].mean():.1f}, max={subset['Tickets_Total'].max():.0f}, min={subset['Tickets_Total'].min():.0f}")

# Lag propagation error analysis
print("\n=== IMPACTO ERROR ACUMULATIVO EN PREDICCION ITERATIVA ===")
print("La prediccion iterativa usa el Lag_1 = prediccion del dia anterior")
print("Si el modelo subestima dia 1, el lag_1 del dia 2 sera bajo, propagando el error")
train_daily = daily[daily['Fecha'] < cutoff]
print(f"\nMedia tickets/dia en train: {train_daily['Tickets_Total'].mean():.1f}")
print(f"Media tickets/dia en test:  {test_daily['Tickets_Total'].mean():.1f}")

# Coeficiente de variacion por nodo
print("\n=== COEFICIENTE DE VARIACION POR NODO (Zona Centro) ===")
for nodo in zcen['Nodo'].unique():
    nodo_data = zcen[zcen['Nodo'] == nodo]['Tickets_Total']
    cv = nodo_data.std() / nodo_data.mean() if nodo_data.mean() > 0 else 0
    print(f"  {nodo}: CV={cv:.2f} (media={nodo_data.mean():.1f}, std={nodo_data.std():.1f})")
