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
├── 🧪 Pruebas
│   └── test_operacion_sla.py      # Suite pytest: pruebas unitarias e integración del sistema
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

## 🧪 Pruebas Unitarias e Integración

El proyecto incluye una suite de pruebas automatizadas con **pytest** en `test_operacion_sla.py` que valida el sistema de extremo a extremo.

### Tipos de Pruebas

| # | Prueba | Tipo | Descripción |
|---|--------|------|-------------|
| 1 | `test_file_existence[datos_ans_limpios.csv]` | Integración | Verifica que el pipeline de preparación generó el archivo |
| 2 | `test_file_existence[datos_jira_regresion.csv]` | Integración | Verifica que la transformación de Jira generó el dataset |
| 3 | `test_file_existence[serie_tiempo_zonas.csv]` | Integración | Verifica que la serie temporal de demanda fue generada |
| 4 | `test_file_existence[modelo_ans_xgboost.pkl]` | Integración | Verifica que el modelo clasificador fue entrenado y guardado |
| 5 | `test_file_existence[modelo_oraculo_xgb.pkl]` | Integración | Verifica que el modelo regresor fue entrenado y guardado |
| 6 | `test_data_integrity` | Unitaria | Valida que no hay nulos en `Minutos_Resolucion` y que las columnas `Nodo` y `Prioridad` existen |
| 7 | `test_model_prediction` | Unitaria | Carga el Oráculo, genera una predicción con datos dummy y verifica que retorna un `float` |
| 8 | `test_projection_logic` | Unitaria | Valida la lógica de negocio: la proyección de fin de mes nunca puede ser menor al valor real actual |

### Resultados de la Última Ejecución

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-7.4.3
rootdir: C:\MLpractica2

collected 8 items

test_operacion_sla.py::test_file_existence[datos_ans_limpios.csv]   PASSED [ 12%]
test_operacion_sla.py::test_file_existence[datos_jira_regresion.csv] PASSED [ 25%]
test_operacion_sla.py::test_file_existence[serie_tiempo_zonas.csv]   PASSED [ 37%]
test_operacion_sla.py::test_file_existence[modelo_ans_xgboost.pkl]   PASSED [ 50%]
test_operacion_sla.py::test_file_existence[modelo_oraculo_xgb.pkl]   PASSED [ 62%]
test_operacion_sla.py::test_data_integrity                            PASSED [ 75%]
test_operacion_sla.py::test_model_prediction                          PASSED [ 87%]
test_operacion_sla.py::test_projection_logic                          PASSED [100%]

============================== 8 passed in 1.86s ==============================
```

### Cómo Ejecutar las Pruebas

```bash
# Con entorno activado
source venv_linux/bin/activate

# Ejecutar suite completa
pytest test_operacion_sla.py -v

# Ejecutar con reporte de cobertura
pytest test_operacion_sla.py -v --tb=short
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
