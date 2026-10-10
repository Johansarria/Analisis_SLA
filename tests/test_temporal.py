"""Pruebas del análisis temporal (src/core/temporal.py)."""
import pandas as pd
import pytest

from src.core import temporal as tp


def _detalle(filas, tz="America/Bogota") -> pd.DataFrame:
    fechas = pd.to_datetime([f for f, _ in filas]).tz_localize(tz)
    return pd.DataFrame({
        "Fecha": fechas,
        "Nodo": [n for _, n in filas],
        "Zona": "ZSUR",
        "Clave": [f"ZSUR-{i}" for i in range(len(filas))],
    })


@pytest.mark.parametrize("hora,esperado", [
    (0, "Madrugada"), (5, "Madrugada"), (6, "Mañana"), (11, "Mañana"),
    (12, "Tarde"), (17, "Tarde"), (18, "Noche"), (23, "Noche"),
])
def test_franja_horaria(hora, esperado):
    assert tp.franja_horaria(hora) == esperado


def test_franja_horaria_fuera_de_rango():
    with pytest.raises(ValueError):
        tp.franja_horaria(24)


def test_columnas_temporales():
    detalle = tp.add_temporal_features(_detalle([
        ("2026-07-05 02:30", "Troncal"),   # domingo de madrugada
        ("2026-07-06 16:45", "Palmira"),   # lunes en la tarde
        ("2026-07-11 20:10", "Troncal"),   # sábado de noche
    ]))
    assert list(detalle["Hora"]) == [2, 16, 20]
    assert list(detalle["Franja"]) == ["Madrugada", "Tarde", "Noche"]
    assert list(detalle["DiaSemana"]) == ["Domingo", "Lunes", "Sábado"]
    assert list(detalle["FinDeSemana"]) == [True, False, True]
    assert detalle["FechaDia"].dt.tz is None


def test_resumen_por_hora_cubre_las_24_horas():
    detalle = tp.add_temporal_features(_detalle([("2026-07-06 16:45", "Palmira")]))
    horas = tp.resumen_por_hora(detalle)
    assert len(horas) == 24 and horas.loc[16] == 1 and horas.loc[3] == 0


def test_resumen_por_dia_semana_ordenado():
    detalle = tp.add_temporal_features(_detalle([
        ("2026-07-06 08:00", "A"), ("2026-07-13 08:00", "A"), ("2026-07-07 08:00", "B"),
    ]))
    dias = tp.resumen_por_dia_semana(detalle)
    assert list(dias.index) == list(tp.DIAS)
    assert dias.loc["Lunes"] == 2 and dias.loc["Martes"] == 1


def test_serie_diaria_rellena_dias_sin_fallas_y_marca_el_ultimo():
    detalle = tp.add_temporal_features(_detalle([
        ("2026-07-06 08:00", "A"), ("2026-07-08 08:00", "A"), ("2026-07-09 08:00", "B"),
    ]))
    serie = tp.serie_diaria(detalle)
    assert len(serie) == 4                      # 6, 7, 8 y 9 de julio
    assert serie["Incidencias"].loc["2026-07-07"] == 0
    assert bool(serie["Incompleto"].iloc[-1]) is True
    assert serie["Incompleto"].iloc[:-1].sum() == 0


def test_dia_tipico_excluye_el_ultimo_dia():
    detalle = tp.add_temporal_features(_detalle([
        ("2026-07-06 08:00", "A"), ("2026-07-07 08:00", "A"), ("2026-07-07 09:00", "A"),
        ("2026-07-08 08:00", "A"), ("2026-07-08 09:00", "A"), ("2026-07-08 10:00", "A"),
    ]))
    assert tp.dia_tipico(detalle) == 1.5        # días 6 y 7 completos; el 8 es el último y se descarta


def test_proyeccion_repite_el_patron_semanal():
    filas = [("2026-07-06 08:00", "A"), ("2026-07-07 08:00", "A"), ("2026-07-07 10:00", "A"),
             ("2026-07-13 08:00", "A"), ("2026-07-14 08:00", "A"), ("2026-07-14 10:00", "A"),
             ("2026-07-15 08:00", "A")]
    proy = tp.proyeccion_por_dia_semana(tp.add_temporal_features(_detalle(filas)), semanas=2)
    assert proy["promedio_por_dia"].loc["Lunes"] == 1.0
    assert proy["promedio_por_dia"].loc["Martes"] == 2.0
    assert len(proy["proximas"]) == 14
    assert proy["inicio"].isoformat() == "2026-07-16T00:00:00"
    assert proy["total"] == pytest.approx(3.0 * 2, abs=0.05)   # 3 fallas por semana x 2 semanas


def test_metricas_en_lenguaje_llano():
    filas = [("2026-07-06 16:00", "Troncal"), ("2026-07-06 17:00", "Troncal"),
             ("2026-07-13 16:00", "Troncal"), ("2026-07-07 16:00", "Palmira")]
    detalle = tp.add_temporal_features(_detalle(filas))
    met = tp.metricas(detalle)
    assert met["total"] == 4
    assert met["hora_pico"] == 16
    assert met["franja_pico"] == "Tarde"
    assert met["dia_semana_pico"] == "Lunes"
    assert met["nodo_franja_pico"].startswith("Troncal")
    assert met["top_nodos"].index[0] == "Troncal"


def test_matrices_de_dia_y_hora():
    detalle = tp.add_temporal_features(_detalle([
        ("2026-07-06 16:00", "Troncal"), ("2026-07-07 02:00", "Palmira"),
    ]))
    dh = tp.matriz_dia_hora(detalle)
    assert dh.shape == (7, 24)
    assert dh.loc["Lunes", 16] == 1 and dh.loc["Martes", 2] == 1
    nf = tp.matriz_nodo_franja(detalle)
    assert list(nf.columns) == ["Madrugada", "Mañana", "Tarde", "Noche"]
    assert nf.loc["Troncal", "Tarde"] == 1
