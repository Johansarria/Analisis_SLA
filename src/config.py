import os
from pathlib import Path

# --- RUTAS BASE ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# --- ARCHIVOS DE DATOS ---
FILE_JIRA_RAW = RAW_DATA_DIR / "JiraATP.csv"
FILE_JIRA_CLEAN = PROCESSED_DATA_DIR / "datos_jira_regresion.csv"
FILE_ANS_CLEAN = PROCESSED_DATA_DIR / "datos_ans_limpios.csv"
FILE_TIME_SERIES_ZONES = PROCESSED_DATA_DIR / "serie_tiempo_zonas.csv"

# --- MODELOS ---
MODEL_ANS_XGB = MODELS_DIR / "modelo_ans_xgboost.pkl"
MODEL_ORACULO_XGB = MODELS_DIR / "modelo_oraculo_xgb.pkl"

# --- CONFIGURACIÓN DE NEGOCIO ---
MIN_MINUTES_RESOLUTION = 15
MAX_MINUTES_RESOLUTION = 7200  # 5 días

# --- COLUMNAS JIRA ---
JIRA_COLUMNS_MAP = {
    'Resumen': 'Resumen',
    'Prioridad': 'Prioridad',
    'Creada': 'Fecha_Creacion',
    'Campo personalizado (Solución afectación )': 'Fecha_Solucion',
    'Campo personalizado (Nodo Afectado)': 'Nodo',
    'Campo personalizado (Usuarios Iniciales Afectados)': 'Usuarios_Afectados'
}

# --- MESES ESPAÑOL A INGLÉS ---
MONTHS_ES_TO_EN = {
    'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr', 
    'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'ago': 'Aug', 
    'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
}

def ensure_dirs():
    """Asegura que todas las carpetas necesarias existan."""
    for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
