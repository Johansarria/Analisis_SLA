# 🧠 Análisis SLA — Sistema de IA para Gestión de Incidencias de O&M

Sistema de Machine Learning diseñado para analizar, predecir y alertar sobre el cumplimiento de **Acuerdos de Nivel de Servicio (ANS/SLA)** en operaciones de O&M (Operación y Mantenimiento), utilizando datos de seguimiento de incidencias extraídos de **Jira**.

---

## 🎯 Objetivo

Reducir los incumplimientos de SLA en el NOC (Network Operations Center) mediante inteligencia artificial capaz de:

- **Clasificar** si un ticket cumplirá o no el ANS antes de que venza.
- **Predecir** con cuántos minutos/horas se resolverá un incidente.
- **Detectar** los nodos y zonas geográficas de mayor riesgo operativo.
- **Proyectar** la demanda futura de incidencias por zona.

---

## 🗂️ Nueva Estructura del Proyecto (v2.0)

El proyecto ha sido reestructurado siguiendo las mejores prácticas de **Clean Code** y **SOLID**, como se define en [AGENTES.md](AGENTES.md).

```text
MLpractica2/
├── data/
│   ├── raw/                # Datos originales (JiraATP.csv, Excel O&M)
│   └── processed/          # Datasets limpios (datos_jira_regresion.csv, etc.)
├── models/                 # Modelos entrenados (.pkl)
├── outputs/
│   └── plots/              # Gráficos y visualizaciones generadas
├── src/
│   ├── core/               # Lógica principal: Modelos y Detección de Anomalías
│   ├── data/               # Motores ETL para limpieza y transformación
│   ├── utils/              # Generación de reportes y visualizaciones
│   ├── scripts/            # Puntos de entrada y orquestación del pipeline
│   └── config.py           # Configuración centralizada de rutas y constantes
├── tests/                  # Suite de pruebas automatizadas con Pytest
└── legacy/                 # Scripts originales preservados para compatibilidad
```

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/Johansarria/Analisis_SLA.git
cd Analisis_SLA
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar el Pipeline Maestro
Ya no es necesario ejecutar múltiples scripts por separado. El pipeline maestro coordina todo el proceso:
```bash
python -m src.scripts.run_pipeline
```
Este comando realiza:
1. **ETL**: Limpieza de datos de Jira y Excel.
2. **Entrenamiento**: Calibración de modelos XGBoost para ANS y Oráculo de tiempos.
3. **Detección**: Identificación de nodos críticos.
4. **Reporte**: Generación de gráficos en `outputs/plots/`.

### 4. Ejecutar pruebas de integridad
```bash
pytest tests/ -v
```

---

## 🛠️ Tecnologías Utilizadas

| Librería        | Uso                                              |
|-----------------|--------------------------------------------------|
| `pandas`        | Manipulación y limpieza de datos                 |
| `xgboost`       | Modelos de clasificación y regresión (SLA)       |
| `scikit-learn`  | Preprocesamiento y métricas                      |
| `matplotlib`    | Visualización de gráficos de operación           |
| `pathlib`       | Manejo robusto de rutas del sistema              |

---

## 👤 Autor

**Johan Sarria**  
GitHub: [@Johansarria](https://github.com/Johansarria)
