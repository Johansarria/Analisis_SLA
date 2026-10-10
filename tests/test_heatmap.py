"""Pruebas del mapa de calor Nodo x Mes (src/core/heatmap.py)."""
import pandas as pd
import pytest

from src.core.heatmap import (COL_FECHA, COL_NODO, COL_ZONA, _normalizar_fechas,
                              build_heatmap, prepare_window, render_ascii)


def _export_sintetico() -> pd.DataFrame:
    """Export mínimo con el mismo formato que descarga Jira (4 meses de datos)."""
    filas = [
        # (Creada, Nodo, Proyecto, Clave)
        ("2026-06-15 08:00:00", "Palmira", "Zona Suroccidente", "ZSUR-1"),      # fuera de la ventana
        ("2026-06-25 08:00:00", "Palmira", "Zona Suroccidente", "ZSUR-2"),      # dentro
        ("2026-07-05 08:00:00", "Girardot", "Zona Suroccidente", "ZSUR-3"),
        ("2026-07-06 08:00:00", "Girardot", "Zona Suroccidente", "ZSUR-4"),
        ("2026-08-10 08:00:00", "Girardot", "Zona Suroccidente", "ZSUR-5"),
        ("2026-08-11 08:00:00", "Palmira", "Zona Suroccidente", "ZSUR-6"),
        ("2026-09-01 08:00:00", "Girardot", "Zona Suroccidente", "ZSUR-7"),
        ("2026-09-02 08:00:00", "Girardot", "Zona Suroccidente", "ZSUR-8"),
        ("2026-09-03 08:00:00", "Girardot", "Zona Suroccidente", "ZSUR-9"),
        ("2026-09-04 08:00:00", "Girardot", "Zona Suroccidente", "ZSUR-10"),
        ("2026-09-05 08:00:00", "Chapinero", "Zona Centro", "ZCEN-1"),          # otra zona
        ("", "SinFecha", "Zona Suroccidente", "ZSUR-11"),                       # se descarta
        ("2026-09-06 08:00:00", "  ", "Zona Suroccidente", "ZSUR-12"),          # sin nodo, se descarta
    ]
    return pd.DataFrame(filas, columns=[COL_FECHA, COL_NODO, COL_ZONA, "Clave de incidencia"])


def test_normalizar_fechas_espanol():
    serie = pd.Series(["12/abr/24 3:41", "2026-09-20 10:00:00"])
    fechas = _normalizar_fechas(serie)
    assert fechas.notna().all()
    assert fechas.iloc[0].month == 4 and fechas.iloc[0].day == 12
    assert fechas.iloc[1].month == 9


def test_ventana_alineada_a_meses_calendario():
    """El corte por defecto es el último dato y la ventana arranca el día 1 del mes."""
    ventana = prepare_window(_export_sintetico(), months=3)
    assert ventana["Fecha"].min().date().isoformat() == "2026-07-05"
    assert ventana["Fecha"].max().date().isoformat() == "2026-09-05"
    assert "SinFecha" not in set(ventana["Nodo"])
    assert len(ventana) == 9  # 13 filas menos las 2 de junio, la sin fecha y la sin nodo


def test_inicio_explicito_amplia_la_ventana():
    ventana = prepare_window(_export_sintetico(), months=3, inicio="2026-06-15")
    assert ventana["Fecha"].min().date().isoformat() == "2026-06-15"
    assert len(ventana) == 11  # las 2 de junio entran + las 9 de julio a septiembre


def test_madrugada_no_se_corre_de_mes():
    """Las horas locales no se interpretan como UTC: 02:00 del 1 no cae en el mes anterior."""
    filas = [("2026-09-01 02:00:00", "Girardot", "Zona Suroccidente", "ZSUR-1"),
             ("2026-09-10 02:00:00", "Girardot", "Zona Suroccidente", "ZSUR-2")]
    df = pd.DataFrame(filas, columns=[COL_FECHA, COL_NODO, COL_ZONA, "Clave de incidencia"])
    matriz = build_heatmap(df, months=3)["matriz"]
    assert matriz.loc["Girardot", "Septiembre 2026"] == 2
    assert "Agosto 2026" not in matriz.columns


def test_filtro_de_zona():
    ventana = prepare_window(_export_sintetico(), months=3, zonas=["ZSUR"])
    assert set(ventana["Zona"]) == {"Zona Suroccidente"}
    assert "Chapinero" not in set(ventana["Nodo"])


def test_matriz_y_proyeccion_mes_en_curso():
    resultado = build_heatmap(_export_sintetico(), months=3)
    matriz, corte = resultado["matriz"], resultado["corte"]
    assert corte["mes_curso"] == "Septiembre 2026"
    assert corte["dias_transcurridos"] == 5 and corte["dias_totales"] == 30

    columnas = list(matriz.columns)
    assert columnas[:3] == ["Julio 2026", "Agosto 2026", "Septiembre 2026"]
    assert "Proyección Septiembre 2026" in columnas

    girardot = matriz.loc["Girardot"]
    assert girardot["Julio 2026"] == 2 and girardot["Agosto 2026"] == 1
    assert girardot["Septiembre 2026"] == 4
    assert girardot["Proyección Septiembre 2026"] == round(4 / 5 * 30)
    assert girardot["Total ventana"] == 7
    assert girardot["Total proyectado"] == 2 + 1 + round(4 / 5 * 30)


def test_orden_y_render_ascii():
    resultado = build_heatmap(_export_sintetico(), months=3)
    matriz = resultado["matriz"]
    assert list(matriz.index)[0] == "Girardot"  # el nodo más golpeado va primero
    tabla = render_ascii(matriz, top=5)
    assert "NODO AFECTADO" in tabla and "Girardot" in tabla


def test_min_total_filtra_nodos_pequenos():
    resultado = build_heatmap(_export_sintetico(), months=3, min_total=20)
    assert list(resultado["matriz"].index) == ["Girardot"]


def test_export_sin_columnas_requeridas_falla():
    with pytest.raises(KeyError):
        build_heatmap(pd.DataFrame({"Resumen": ["x"]}), months=3)
