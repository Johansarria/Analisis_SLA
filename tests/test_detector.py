import pytest
import pandas as pd
from src.core.detector import AnomalyDetector


def test_shouldIdentifyCriticalGroupAsHighestAverageFailures(df_jira: pd.DataFrame):
    """El grupo crítico debe ser aquel con mayor promedio de Total_Fallas."""
    detector = AnomalyDetector(n_clusters=3)
    perfiles = detector.build_node_profiles(df_jira)

    if len(perfiles) < 3:
        pytest.skip("Datos insuficientes para 3 clusters")

    perfiles_full, peligros = detector.detect_anomalies(perfiles)

    # El grupo crítico es el de mayor promedio de fallas
    promedios = perfiles_full.groupby('Grupo_IA')['Total_Fallas'].mean()
    grupo_critico = promedios.idxmax()

    assert peligros['Grupo_IA'].unique()[0] == grupo_critico
    assert len(peligros) > 0


def test_shouldFilterNodesWithFewerThanFiveIncidents():
    """Nodos con 5 o menos incidentes deben excluirse del análisis."""
    detector = AnomalyDetector()
    df = pd.DataFrame({
        'Nodo': ['NODO_A'] * 3 + ['NODO_B'] * 10 + ['NODO_C'] * 6,
        'Tipo_Falla': ['FALLA'] * 19,
        'Minutos_Resolucion': [60.0] * 19,
        'Prioridad': ['Alta'] * 19
    })

    perfiles = detector.build_node_profiles(df)
    nodos = perfiles['Nodo'].tolist()

    assert 'NODO_A' not in nodos
    assert 'NODO_B' in nodos
    assert 'NODO_C' in nodos
