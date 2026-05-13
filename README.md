# Analisis SLA -- Sistema de IA para Gestion de Incidencias de O&M

Sistema de Machine Learning disenado para analizar, predecir y alertar sobre el cumplimiento de **Acuerdos de Nivel de Servicio (ANS/SLA)** en operaciones de O&M, utilizando datos de seguimiento de incidencias extraidos de **Jira**.

---

## Objetivo

Reducir los incumplimientos de SLA en el NOC (Network Operations Center) mediante inteligencia artificial capaz de:

- **Clasificar** si un ticket cumplira o no el ANS antes de que venza.
- **Predecir** con cuantos minutos/horas se resolvera un incidente (Oraculo).
- **Detectar** los nodos y zonas geograficas de mayor riesgo operativo.
- **Proyectar** la demanda futura de incidencias por zona.

---

## Estructura del Proyecto (v2.0 -- Refactorizado)

El proyecto fue reestructurado siguiendo los principios **SOLID**, **Clean Code** y las directrices definidas en [AGENTES.md](AGENTES.md). Los scripts originales monoliticos se movieron a `legacy/` como referencia historica.

```
Analisis_SLA/
├── data/
│   ├── raw/                          # Datos originales (JiraATP.csv, Excel O&M)
│   └── processed/                    # Datasets limpios generados por el ETL
├── models/                           # Modelos entrenados (.pkl)
├── outputs/
│   ├── plots/                        # Graficos y visualizaciones generadas
│   └── reports/                      # Reportes exportados
├── src/
│   ├── core/
│   │   ├── models.py                 # ModelTrainer: clasificador ANS + regresor Oraculo (XGBoost)
│   │   └── detector.py               # AnomalyDetector: deteccion de nodos criticos (K-Means)
│   ├── data/
│   │   └── etl.py                    # DataProcessor: limpieza de Jira y Excel ANS
│   ├── utils/
│   │   └── visualization.py          # Visualizer: generacion de graficos (barras, radar, pie)
│   ├── scripts/
│   │   └── run_pipeline.py           # Pipeline maestro que orquesta todo el proceso
│   └── config.py                     # Configuracion centralizada de rutas, columnas y constantes
├── tests/
│   └── test_operacion_sla.py         # Suite de pruebas con Pytest (8 tests)
├── legacy/                           # Scripts originales preservados (14 archivos)
├── AGENTES.md                        # Guia maestra de desarrollo para agentes IA
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Modulos Principales

### `src/config.py` -- Configuracion Centralizada

Define todas las rutas del sistema (`DATA_DIR`, `MODELS_DIR`, `PLOTS_DIR`, etc.), mapeos de columnas Jira, traduccion de meses espanol-a-ingles, y constantes de negocio. La funcion `ensure_dirs()` garantiza que la estructura de directorios exista antes de ejecutar cualquier proceso.

### `src/data/etl.py` -- DataProcessor

Clase responsable de la extraccion, transformacion y carga (ETL):

| Metodo | Funcion |
|--------|---------|
| `process_jira()` | Lee `JiraATP.csv`, renombra columnas, convierte fechas, calcula `Minutos_Resolucion`, extrae `Tipo_Falla`, `Hora_Del_Dia`, `Dia_De_La_Semana` |
| `process_ans()` | Lee el Excel de ANS (hoja NACIONAL), genera columna `Target` binaria (1 = no cumple, 0 = cumple) |
| `save_processed_data()` | Persiste los DataFrames limpios en `data/processed/` |

### `src/core/models.py` -- ModelTrainer

Clase que gestiona el ciclo de vida completo de los modelos XGBoost:

| Metodo | Funcion |
|--------|---------|
| `train_ans_classifier()` | Entrena clasificador XGBoost con balanceo de clases (`scale_pos_weight`). Evalua con `classification_report`. Guarda modelo + features en `modelo_ans_xgboost.pkl` |
| `train_oraculo_regressor()` | Entrena regresor XGBoost para predecir minutos de resolucion. Evalua con MAE. Guarda en `modelo_oraculo_xgb.pkl` |
| `load_model()` | Carga modelos entrenados desde disco |

### `src/core/forecaster.py` -- DemandForecaster

Gestiona el modelo de series de tiempo para proyectar la demanda de incidencias a nivel granular:

| Metodo | Funcion |
|--------|---------|
| `train()` | Entrena modelo XGBoost usando TimeSeriesSplit, aplicando One-Hot Encoding doble a `Zona` y `Nodo`. Optimiza hiperparámetros con Optuna. |
| `predict_future()` | Predice iterativamente la demanda a N días aislando el cálculo de promedios móviles para combinaciones geográficas estrictas. |

### `src/core/detector.py` -- AnomalyDetector

Usa **K-Means** (scikit-learn) para agrupar nodos por perfil de fallas:

- `build_node_profiles()`: agrega Total_Fallas, Tiempo_Promedio_Minutos y Porcentaje_Critico por nodo.
- `detect_anomalies()`: escala datos con `StandardScaler`, aplica K-Means (3 clusters) e identifica el grupo critico (mayor promedio de fallas).
- `get_top_critical_nodos()`: devuelve los N nodos mas riesgosos del grupo critico.

### `src/utils/visualization.py` -- Visualizer

Genera graficos usando matplotlib y seaborn:

| Metodo | Grafico generado |
|--------|------------------|
| `plot_risk_comparison()` | Barras comparativas Real vs Proyectado (top 10 nodos) |
| `plot_operational_distribution()` | Pie chart de carga operativa por zona |
| `plot_anomaly_radar()` | Barras de criticidad de los nodos anomalos |
| `plot_demand_forecast()` | Multi-gráficos separados por Zona, conteniendo los pronósticos por Nodo |

### Scripts Orquestadores (`src/scripts/`)

- `run_pipeline.py`: Orquesta el flujo de clasificación y detección de anomalías (ANS y K-Means).
- `run_forecasting.py`: Orquesta de forma independiente el flujo de predicción de demanda de tickets (Time Series XGBoost).

Ejecucion: 
```bash
python -m src.scripts.run_pipeline
python -m src.scripts.run_forecasting
```

---

## Instalacion y Uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/Johansarria/Analisis_SLA.git
cd Analisis_SLA
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Preparar los datos

Colocar los archivos fuente en `data/raw/`:
- `JiraATP.csv` -- Export de incidencias desde Jira
- `BASE PARA DATOS O&M FEBRERO ZONAS-REVISADO CALI (2).xlsx` -- Datos de ANS

### 4. Ejecutar el Sistema

Pipeline general (Clasificación/Oráculo/Anomalías):
```bash
python -m src.scripts.run_pipeline
```

Predicción de demanda de tickets (Forecasting por Zona/Nodo):
```bash
python -m src.scripts.run_forecasting
```

### 5. Ejecutar pruebas de integridad

```bash
pytest tests/ -v
```

Pruebas incluidas:
- **Infraestructura**: existencia de archivos criticos (parametrizado, 5 asserts)
- **Integridad de datos**: ausencia de nulos en `Minutos_Resolucion`, presencia de columnas clave
- **Prediccion del modelo**: carga y prediccion dummy con el regresor Oraculo
- **Logica de negocio**: validacion de coherencia en perfiles de anomalia

---

## Tecnologias Utilizadas

| Libreria        | Uso                                              |
|-----------------|--------------------------------------------------|
| `pandas`        | Manipulacion y limpieza de datos                 |
| `numpy`         | Operaciones vectoriales y soporte numerico       |
| `xgboost`       | Modelos de clasificacion (ANS), regresion (Oraculo) y Time Series (Forecasting) |
| `scikit-learn`  | Preprocesamiento (StandardScaler), clustering (K-Means) y metricas |
| `matplotlib`    | Visualizacion de graficos de operacion           |
| `seaborn`       | Estilos y paletas para visualizaciones           |
| `optuna`        | Optimización automática de hiperparámetros       |
| `joblib`        | Serializacion de modelos entrenados              |
| `pytest`        | Framework de pruebas automatizadas               |
| `openpyxl`      | Lectura de archivos Excel (.xlsx)                |

---

## Cambios en v2.0 (Y extensiones Nodos)

Comparado con la version original de scripts monoliticos:

| Aspecto | v1.0 | v2.0+ (Forecasting Nodal) |
|---------|------|------|
| Arquitectura | 14 scripts sueltos en raiz | Paquete `src/` con módulos altamente cohesivos (SOLID) |
| Configuracion | Rutas hardcodeadas en cada script | `src/config.py` centralizado con mapeo diccionario estricto |
| Responsabilidades | Cada script hacia ETL + modelo + graficos | Separacion: `etl.py`, `models.py`, `detector.py`, `forecaster.py` |
| Orquestacion | Ejecucion manual script por script | `run_pipeline.py` y `run_forecasting.py` automatizados |
| Nivel Analítico | Predicción global única | Predicción geolocalizada hasta nivel **Nodo** (30 sub-entidades) |

---

## Autor

**Johan Sarria**
GitHub: [@Johansarria](https://github.com/Johansarria)
