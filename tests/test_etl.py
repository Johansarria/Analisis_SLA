import pandas as pd
import numpy as np
from src.data.etl import DataProcessor
from src.config import MIN_MINUTES_RESOLUTION, MAX_MINUTES_RESOLUTION


def test_shouldTranslateSpanishDatesToEnglishWhenProcessingJira(df_jira: pd.DataFrame):
    """Las fechas en español deben parsearse correctamente a datetime."""
    assert df_jira['Fecha_Creacion'].notna().all()
    assert df_jira['Fecha_Solucion'].notna().all()
    assert pd.api.types.is_datetime64_any_dtype(df_jira['Fecha_Creacion'])
    assert pd.api.types.is_datetime64_any_dtype(df_jira['Fecha_Solucion'])


def test_shouldFilterOutliersOutsideValidRangeWhenProcessingJira(processor: DataProcessor, tmp_path):
    """Los tickets con tiempos de resolución fuera de rango deben eliminarse."""
    # Crear CSV temporal con outliers
    data = {
        'Resumen': ['Falla/TEST'] * 3,
        'Prioridad': ['Alta'] * 3,
        'Creada': ['01 ene 2023 08:00'] * 3,
        'Campo personalizado (Solución afectación )': ['01 ene 2023 08:05', '01 ene 2023 10:00', '10 ene 2023 08:00'],
        'Campo personalizado (Nodo Afectado)': ['NODO_TEST'] * 3,
        'Campo personalizado (Usuarios Iniciales Afectados)': [10] * 3,
        'Nombre del proyecto': ['Zona Centro'] * 3
    }
    df_raw = pd.DataFrame(data)
    csv_path = tmp_path / "outliers.csv"
    df_raw.to_csv(csv_path, index=False)

    df_clean = processor.process_jira(str(csv_path))

    assert (df_clean['Minutos_Resolucion'] >= MIN_MINUTES_RESOLUTION).all()
    assert (df_clean['Minutos_Resolucion'] <= MAX_MINUTES_RESOLUTION).all()
    # El de 5 minutos y el de 10 dias deben filtrarse
    assert len(df_clean) == 1


def test_shouldCreateEngineeredFeaturesWhenProcessingJira(df_jira: pd.DataFrame):
    """Deben existir las columnas de ingeniería de características."""
    assert 'Tipo_Falla' in df_jira.columns
    assert 'Hora_Del_Dia' in df_jira.columns
    assert 'Dia_De_La_Semana' in df_jira.columns
    assert df_jira['Usuarios_Afectados'].notna().all()


def test_shouldCreateBinaryTargetWhenProcessingAns(df_ans: pd.DataFrame):
    """El target binario debe ser 1 para NO CUMPLE y 0 para CUMPLE."""
    assert 'Target' in df_ans.columns
    assert set(df_ans['Target'].unique()).issubset({0, 1})

    # Verificar lógica explícita
    for _, row in df_ans.iterrows():
        raw = str(row['CUMPLE/NO CUMPLE']).strip().upper()
        expected = 1 if raw == 'NO CUMPLE' else 0
        assert row['Target'] == expected
