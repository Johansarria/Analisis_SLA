# 🧠 Análisis SLA — Sistema de IA para Gestión de Incidencias de O&M

Sistema de Machine Learning diseñado para analizar, predecir y alertar sobre el cumplimiento de **Acuerdos de Nivel de Servicio (ANS/SLA)** en operaciones de O&M (Operación y Mantenimiento), utilizando datos de seguimiento de incidencias extraídos de **Jira**.

---

## 🎯 Objetivo

Reducir los incumplimientos de SLA en el NOC (Network Operations Center) mediante inteligencia artificial capaz de:

- **Clasificar** si un ticket cumplirá o no el ANS antes de que venza.
- **Predecir** con cuántos minutos/horas se resolverá un incidente.
- **Detectar** los nodos y zonas geográficas de mayor riesgo operativo.
- **Proyectar** la demanda futura de incidencias por zona.
- **Alertar** en tiempo real al NOC cuando un ticket tiene alta probabilidad de incumplimiento.

---

## 🗂️ Estructura del Proyecto

```
MLpractica2/
│
├── 📥 Preparación de Datos
│   ├── preparar_datos_ans.py      # Limpia y estructura datos para el modelo de clasificacion ANS
│   ├── preparar_datos_jira.py     # Extrae y transforma el histórico de Jira para regresion
│   └── preparar_demanda.py        # Genera series de tiempo de demanda de incidencias por zona
│
├── 🤖 Entrenamiento de Modelos
│   ├── entrenar_ans.py            # Modelo XGBoost: predice CUMPLE/NO CUMPLE (clasificacion)
│   ├── entrenar_oraculo.py        # Modelo XGBoost: predice minutos de resolucion (regresion)
│   └── entrenar_demanda_lstm.py   # Red LSTM: proyecta volumen futuro de incidencias
│
├── 🔍 Análisis Exploratorio
│   ├── explorar_jira.py           # Exploración inicial del CSV de Jira
│   ├── escaner_profundo.py        # Análisis estadístico avanzado del dataset
│   ├── detector_nodos.py          # Identifica nodos críticos con mayor frecuencia de fallos
│   └── heatzone_nodos.py          # Genera mapas de calor por zonas geográficas
│
├── 📊 Visualización y Validación
│   ├── graficar_operacion.py      # Gráficos de tendencias operacionales
│   └── validar_precision_demanda.py # Evalúa precisión del modelo LSTM de demanda
│
├── 🚨 Sistema en Vivo (NOC)
│   └── simulador_noc.py           # Simula tickets en tiempo real y emite alertas de riesgo ANS
│
├── 🧠 Modelos Entrenados
│   ├── modelo_ans_xgboost.pkl     # Clasificador ANS (CUMPLE/NO CUMPLE)
│   └── modelo_oraculo_xgb.pkl    # Regresor de tiempo de resolución
│
└── 📄 Datos procesados
    ├── datos_ans_limpios.csv
    ├── datos_jira_regresion.csv
    ├── mapa_calor_nodos.csv
    ├── mapa_calor_proyectado.csv
    ├── serie_tiempo_jira.csv
    └── serie_tiempo_zonas.csv
```

---

## ⚡ Flujo de Trabajo

```
JiraATP.csv (datos crudos)
        │
        ▼
[preparar_datos_ans.py]     → datos_ans_limpios.csv
[preparar_datos_jira.py]    → datos_jira_regresion.csv
[preparar_demanda.py]       → serie_tiempo_zonas.csv
        │
        ▼
[entrenar_ans.py]           → modelo_ans_xgboost.pkl     (Clasificador ANS)
[entrenar_oraculo.py]       → modelo_oraculo_xgb.pkl     (Regresor de tiempo)
[entrenar_demanda_lstm.py]  → predicciones de demanda
        │
        ▼
[simulador_noc.py]          → Alertas en tiempo real para el equipo de O&M
```

---

## 🛠️ Tecnologías Utilizadas

| Librería        | Uso                                              |
|-----------------|--------------------------------------------------|
| `pandas`        | Manipulación y limpieza de datos                 |
| `numpy`         | Operaciones numéricas y estadísticas             |
| `scikit-learn`  | División de datos, métricas de evaluación        |
| `xgboost`       | Modelos de clasificación y regresión (ANS)       |
| `tensorflow`    | Red neuronal LSTM para predicción de demanda     |
| `joblib`        | Serialización y carga de modelos entrenados      |
| `matplotlib`    | Visualización de gráficos de operación           |
| `seaborn`       | Heatmaps y gráficos estadísticos                 |

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/Johansarria/Analisis_SLA.git
cd Analisis_SLA
```

### 2. Crear entorno virtual (Linux / WSL)
```bash
sudo apt install python3-venv python3-pip -y
python3 -m venv venv_linux
source venv_linux/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar el flujo completo
```bash
# Preparar datos
python preparar_datos_ans.py

# Entrenar modelos
python entrenar_ans.py
python entrenar_oraculo.py

# Ejecutar simulador NOC
python simulador_noc.py
```

---

## 📈 Resultados Esperados

El **Simulador NOC** emite alertas por cada ticket entrante:

```
Ticket INC-001 | Riesgo de Incumplimiento ANS: 87.3%
  🚨 ALERTA ROJA: Despachar apoyo preventivo o escalar caso de inmediato.

Ticket INC-002 | Riesgo de Incumplimiento ANS: 54.1%
  ⚠️  ALERTA AMARILLA: Monitorear de cerca. Riesgo latente.

Ticket INC-003 | Riesgo de Incumplimiento ANS: 12.8%
  ✅ OPERACIÓN NOMINAL: Dejar que la cuadrilla opere normalmente.
```

---

## ⚠️ Datos Sensibles

El archivo `JiraATP.csv` (datos crudos de Jira, ~145MB) **no se incluye en el repositorio** por contener información operacional sensible y superar el límite de GitHub. Está excluido en el `.gitignore`.

---

## 👤 Autor

**Johan Sarria**  
GitHub: [@Johansarria](https://github.com/Johansarria)
