"""Análisis temporal de incidencias: hora, franja del día, día de la semana y fecha.

Se monta sobre el detalle que produce ``src.core.heatmap.prepare_window`` (una fila
por incidencia, con ``Fecha`` en hora local de Bogotá) y responde preguntas de
negocio: a qué hora se reportan las fallas, qué días se concentran, cómo evoluciona
la operación día a día y qué se puede esperar si el patrón se mantiene.

Nada de esto reemplaza al modelo de pronóstico de demanda: es un análisis
descriptivo + proyección por promedio estacional del día de la semana.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Franjas del día y días de la semana, en el orden en que se muestran en el informe.
FRANJAS = (("Madrugada", 0, 5), ("Mañana", 6, 11), ("Tarde", 12, 17), ("Noche", 18, 23))
DIAS = ("Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo")
HORAS = tuple(range(24))


def franja_horaria(hora: int) -> str:
    """Devuelve la franja del día (Madrugada/Mañana/Tarde/Noche) de una hora 0-23."""
    if not 0 <= int(hora) <= 23:
        raise ValueError(f"Hora fuera de rango (0-23): {hora}")
    for nombre, inicio, fin in FRANJAS:
        if inicio <= int(hora) <= fin:
            return nombre
    raise ValueError(f"Hora sin franja asignada: {hora}")


def add_temporal_features(detalle: pd.DataFrame) -> pd.DataFrame:
    """Añade Hora, Franja, DiaSemana, FinDeSemana y FechaDia al detalle de incidencias."""
    faltantes = [c for c in ("Fecha", "Nodo") if c not in detalle.columns]
    if faltantes:
        raise KeyError("Faltan columnas en el detalle: " + ", ".join(faltantes))

    fecha = detalle["Fecha"]
    if fecha.dt.tz is None:
        raise ValueError("La columna Fecha debe traer zona horaria resuelta (prepare_window).")

    salida = detalle.copy()
    salida["Hora"] = fecha.dt.hour.astype(int)
    salida["Franja"] = salida["Hora"].map(franja_horaria)
    salida["DiaSemana"] = fecha.dt.weekday.map(lambda i: DIAS[i])
    salida["FinDeSemana"] = fecha.dt.weekday >= 5
    salida["FechaDia"] = fecha.dt.normalize().dt.tz_localize(None)
    return salida


def resumen_por_hora(df: pd.DataFrame) -> pd.Series:
    """Incidencias por hora del día (24 posiciones, incluidas las horas sin casos)."""
    conteo = df["Hora"].value_counts().reindex(HORAS, fill_value=0).sort_index()
    conteo.index.name = "Hora"
    return conteo.astype(int)


def resumen_por_franja(df: pd.DataFrame) -> pd.Series:
    """Incidencias por franja del día, ordenadas de madrugada a noche."""
    orden = [nombre for nombre, _, _ in FRANJAS]
    conteo = df["Franja"].value_counts().reindex(orden, fill_value=0)
    conteo.index.name = "Franja"
    return conteo.astype(int)


def resumen_por_dia_semana(df: pd.DataFrame) -> pd.Series:
    """Incidencias por día de la semana, de lunes a domingo."""
    conteo = df["DiaSemana"].value_counts().reindex(DIAS, fill_value=0)
    conteo.index.name = "DiaSemana"
    return conteo.astype(int)


def serie_diaria(df: pd.DataFrame) -> pd.DataFrame:
    """Serie diaria de incidencias con promedio móvil de 7 días.

    El último día se marca como ``Incompleto``: el export de Jira corta a una hora
    del día, así que ese conteo no es comparable con los demás y no debe entrar en
    promedios ni en el día típico.
    """
    conteo = df.groupby("FechaDia").size().rename("Incidencias")
    if conteo.empty:
        raise ValueError("No hay incidencias para construir la serie diaria.")

    indice = pd.date_range(conteo.index.min(), conteo.index.max(), freq="D")
    serie = conteo.reindex(indice, fill_value=0).to_frame()
    serie.index.name = "Fecha"
    serie["Promedio7"] = serie["Incidencias"].rolling(7, min_periods=3).mean().round(2)
    serie["DiaSemana"] = [DIAS[d.weekday()] for d in serie.index]
    serie["FinDeSemana"] = [d.weekday() >= 5 for d in serie.index]
    serie["Incompleto"] = False
    serie.loc[serie.index[-1], "Incompleto"] = True
    return serie


def matriz_nodo_franja(df: pd.DataFrame, nodos: Optional[list] = None) -> pd.DataFrame:
    """Matriz Nodo x Franja del día (cuántas fallas y a qué hora se reportan)."""
    base = df if nodos is None else df[df["Nodo"].isin(nodos)]
    orden = [nombre for nombre, _, _ in FRANJAS]
    matriz = pd.crosstab(base["Nodo"], base["Franja"]).reindex(columns=orden, fill_value=0)
    matriz = matriz.loc[matriz.sum(axis=1).sort_values(ascending=False).index]
    matriz.index.name = "Nodo"
    return matriz.astype(int)


def matriz_dia_hora(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz Día de la semana x Hora: los momentos más críticos de la semana."""
    matriz = pd.crosstab(df["DiaSemana"], df["Hora"]).reindex(index=DIAS, columns=HORAS, fill_value=0)
    matriz.index.name = "DiaSemana"
    matriz.columns.name = "Hora"
    return matriz.astype(int)


def dia_tipico(df: pd.DataFrame) -> float:
    """Incidencias promedio por día completo (excluye el último día, que viene cortado)."""
    serie = serie_diaria(df)
    completos = serie[~serie["Incompleto"]]
    return float(completos["Incidencias"].mean().round(2)) if len(completos) else 0.0


def proyeccion_por_dia_semana(df: pd.DataFrame, semanas: int = 4,
                              excluir_ultimo_dia: bool = True) -> dict:
    """Proyecta las próximas ``semanas`` semanas con el promedio de cada día de la semana.

    Es una proyección de patrón, no un modelo: asume que el ritmo observado por día
    de la semana se mantiene. Devuelve el promedio por día, el calendario proyectado
    y el total esperado.
    """
    base = df
    if excluir_ultimo_dia:
        corte = base["FechaDia"].max()
        base = base[base["FechaDia"] < corte]
    if base.empty:
        raise ValueError("No hay días completos para proyectar.")

    por_fecha = base.groupby("FechaDia").size()
    tabla = pd.DataFrame({"Fecha": por_fecha.index, "Incidencias": por_fecha.values})
    tabla["DiaSemana"] = [DIAS[d.weekday()] for d in tabla["Fecha"]]
    promedio = tabla.groupby("DiaSemana")["Incidencias"].mean().round(2).reindex(DIAS)

    inicio = df["FechaDia"].max() + pd.Timedelta(days=1)
    fechas = pd.date_range(inicio, periods=7 * semanas, freq="D")
    proyectado = pd.DataFrame({"Fecha": fechas})
    proyectado["DiaSemana"] = [DIAS[d.weekday()] for d in fechas]
    proyectado["Esperado"] = proyectado["DiaSemana"].map(promedio).round(1)
    proyectado["Semana"] = [f"Semana {(i // 7) + 1}" for i in range(len(proyectado))]

    return {
        "promedio_por_dia": promedio,
        "proximas": proyectado,
        "total": float(proyectado["Esperado"].sum()),
        "inicio": inicio,
        "fin": fechas[-1],
        "semanas": semanas,
    }


def metricas(df: pd.DataFrame, proyeccion: Optional[dict] = None) -> dict:
    """Métricas y hallazgos en lenguaje llano, listos para el informe."""
    if df.empty:
        raise ValueError("No hay incidencias para resumir.")

    horas = resumen_por_hora(df)
    franjas = resumen_por_franja(df)
    dias = resumen_por_dia_semana(df)
    serie = serie_diaria(df)
    completo = serie[~serie["Incompleto"]]

    hora_pico = int(horas.idxmax())
    dia_pico = str(dias.idxmax())
    fecha_pico = completo["Incidencias"].idxmax()
    fin_semana = float(df["FinDeSemana"].mean() * 100)
    nodo_franja = matriz_nodo_franja(df)
    par_pico = nodo_franja.stack().idxmax() if not nodo_franja.empty else ("", "")

    return {
        "total": int(len(df)),
        "desde": df["Fecha"].min(),
        "hasta": df["Fecha"].max(),
        "dias_cubiertos": int(len(serie)),
        "dia_tipico": dia_tipico(df),
        "dia_max": int(completo["Incidencias"].max()),
        "fecha_max": fecha_pico.date().isoformat(),
        "hora_pico": hora_pico,
        "hora_pico_casos": int(horas.loc[hora_pico]),
        "franja_pico": str(franjas.idxmax()),
        "franja_pico_casos": int(franjas.max()),
        "franja_pico_pct": round(float(franjas.max() / len(df) * 100), 1),
        "dia_semana_pico": dia_pico,
        "dia_semana_pico_casos": int(dias.loc[dia_pico]),
        "sabado_casos": int(dias.loc["Sábado"]),
        "domingo_casos": int(dias.loc["Domingo"]),
        "pct_7_19": round(float(df["Hora"].between(7, 19).mean() * 100), 1),
        "fin_de_semana_pct": round(fin_semana, 1),
        "nodos": int(df["Nodo"].nunique()),
        "nodo_franja_pico": f"{par_pico[0]} en la {par_pico[1].lower()}",
        "nodo_franja_casos": int(nodo_franja.stack().max()) if not nodo_franja.empty else 0,
        "top_nodos": df["Nodo"].value_counts().head(5),
        "proyeccion_total": None if proyeccion is None else proyeccion["total"],
    }
