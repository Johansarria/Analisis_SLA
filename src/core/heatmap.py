"""Mapa de calor Nodo x Mes construido desde un export de Jira.

Reemplaza a ``legacy/heatzone_nodos.py``: ya no trae fechas, meses ni cortes
quemados. La ventana es parametrizable (por defecto los últimos 3 meses) y el
mes en curso se proyecta por run-rate para compararlo con los meses cerrados.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from src.config import MONTHS_ES_TO_EN

logger = logging.getLogger(__name__)

COL_FECHA = "Creada"
COL_NODO = "Campo personalizado (Nodo Afectado)"
COL_ZONA = "Nombre del proyecto"
COL_CLAVE = "Clave de incidencia"

MESES_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre",
    12: "Diciembre",
}

TZ_DEFECTO = "America/Bogota"


def read_jira_export(path: Union[Path, str]) -> pd.DataFrame:
    """Lee un export de Jira probando los separadores/codificaciones habituales."""
    ultimo_error: Optional[Exception] = None
    for sep, enc in ((",", "utf-8-sig"), (";", "latin-1"), (",", "latin-1")):
        try:
            df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False,
                             on_bad_lines="skip")
        except Exception as exc:  # pragma: no cover - depende del archivo
            ultimo_error = exc
            continue
        if len(df.columns) > 5:
            logger.info("Export leído: %s filas x %s columnas (sep=%r, encoding=%s)",
                        len(df), len(df.columns), sep, enc)
            return df
    raise ValueError(f"No se pudo leer el export de Jira {path}: {ultimo_error}")


def _normalizar_fechas(serie: pd.Series) -> pd.Series:
    """Convierte 'Creada' a datetime **naive** (hora local del export).

    Soporta meses en español (ej. ``28/sep/26 9:11:32 AM``). No se marca como UTC:
    el export de Jira trae la hora local del sitio, y asumir UTC desplazaba las
    incidencias de madrugada al día (y al mes) anterior.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fechas = pd.to_datetime(serie, errors="coerce")
        pendientes = fechas.isna()
        if pendientes.any():
            traducidas = pd.to_datetime(serie[pendientes].apply(_traducir_mes),
                                        errors="coerce", dayfirst=True)
            fechas.loc[pendientes] = traducidas
    return fechas


def _traducir_mes(valor):
    """Reemplaza el mes en español por su abreviatura en inglés para poder parsear."""
    if pd.isna(valor):
        return None
    texto = str(valor).lower()
    for esp, ing in MONTHS_ES_TO_EN.items():
        if esp in texto:
            return texto.replace(esp, ing)
    return texto


def _etiqueta_mes(periodo: pd.Period) -> str:
    return f"{MESES_ES[periodo.month]} {periodo.year}"


def prepare_window(df: pd.DataFrame, months: int = 3,
                   end: Optional[Union[str, pd.Timestamp]] = None,
                   tz: str = TZ_DEFECTO,
                   zonas: Optional[Sequence[str]] = None,
                   inicio: Optional[Union[str, pd.Timestamp]] = None) -> pd.DataFrame:
    """Filtra el export a la ventana de los últimos ``months`` meses.

    La ventana se alinea a **meses calendario**: con ``months=3`` y corte en
    septiembre entran julio, agosto y septiembre completos (no tres bloques de 30
    días que dejarían junio a medias). ``inicio`` permite fijar una fecha de
    arranque explícita en lugar del cálculo por meses.

    ``zonas`` limita el análisis a uno o varios proyectos (ej. ['ZSUR'] o
    ['Zona Suroccidente']); se compara contra 'Nombre del proyecto' y contra
    el prefijo de la clave de incidencia.
    Devuelve el detalle (una fila por incidencia) con columnas
    Fecha, Nodo, Zona, Clave, Periodo y Mes.
    """
    faltantes = [c for c in (COL_FECHA, COL_NODO) if c not in df.columns]
    if faltantes:
        raise KeyError("El export no trae las columnas requeridas: " + ", ".join(faltantes))

    detalle = pd.DataFrame({
        "Fecha": _normalizar_fechas(df[COL_FECHA]),
        "Nodo": df[COL_NODO].astype(str).str.strip(),
        "Zona": (df[COL_ZONA].astype(str) if COL_ZONA in df.columns
                 else (df[COL_CLAVE].astype(str).str.split("-").str[0] if COL_CLAVE in df.columns else "")),
        "Clave": df[COL_CLAVE].astype(str) if COL_CLAVE in df.columns else "",
    })
    detalle = detalle.dropna(subset=["Fecha"])
    detalle = detalle[detalle["Nodo"].notna() & (detalle["Nodo"] != "") & (detalle["Nodo"] != "nan")]
    if zonas:
        buscadas = [z.strip().lower() for z in zonas if str(z).strip()]
        zona_txt = detalle["Zona"].str.lower()
        prefijo = detalle["Clave"].str.split("-").str[0].str.lower()
        mascara = zona_txt.apply(lambda v: any(z in v for z in buscadas)) | \
            prefijo.apply(lambda v: any(v == z for z in buscadas))
        detalle = detalle[mascara]
        logger.info("Filtro de zona %s: quedan %s incidencias", buscadas, len(detalle))

    if detalle.empty:
        raise ValueError("El filtro de zona/fecha no dejó incidencias; revisa el export.")

    if detalle["Fecha"].dt.tz is None:
        detalle["Fecha"] = detalle["Fecha"].dt.tz_localize(tz)
    else:
        detalle["Fecha"] = detalle["Fecha"].dt.tz_convert(tz)

    fin = pd.Timestamp(end) if end is not None else detalle["Fecha"].max()
    fin = fin.tz_localize(tz) if fin.tzinfo is None else fin.tz_convert(tz)
    if inicio is not None:
        ini = pd.Timestamp(inicio)
        ini = ini.tz_localize(tz) if ini.tzinfo is None else ini.tz_convert(tz)
    else:
        # Primer día del mes que está ``months - 1`` meses antes del corte.
        ini = (fin.tz_localize(None).to_period("M") - (months - 1)).start_time.tz_localize(tz)
    ventana = detalle[(detalle["Fecha"] >= ini) & (detalle["Fecha"] <= fin)].copy()

    ventana["Periodo"] = ventana["Fecha"].dt.tz_localize(None).dt.to_period("M")
    ventana["Mes"] = ventana["Periodo"].apply(_etiqueta_mes)
    logger.info("Ventana %s -> %s: %s incidencias (%s nodos)",
                ini.date(), fin.date(), len(ventana), ventana["Nodo"].nunique())
    return ventana


def build_heatmap(df: pd.DataFrame, months: int = 3,
                  end: Optional[Union[str, pd.Timestamp]] = None,
                  min_total: float = 0, tz: str = TZ_DEFECTO,
                  zonas: Optional[Sequence[str]] = None,
                  inicio: Optional[Union[str, pd.Timestamp]] = None) -> dict:
    """Construye la matriz Nodo x Mes con proyección del mes en curso.

    ``df`` puede ser el export crudo de Jira o el detalle ya preparado.
    Devuelve un dict con ``matriz``, ``detalle``, ``orden_meses`` y ``corte``.
    """
    if "Periodo" in df.columns and "Fecha" in df.columns:
        ventana = df
    else:
        ventana = prepare_window(df, months=months, end=end, tz=tz, zonas=zonas, inicio=inicio)

    if ventana.empty:
        raise ValueError("La ventana no tiene incidencias con Nodo Afectado; revisa fechas/columnas.")

    periodos = pd.period_range(ventana["Periodo"].min(), ventana["Periodo"].max(), freq="M")
    etiquetas = [_etiqueta_mes(p) for p in periodos]
    meses_cerrados = etiquetas[:-1]
    mes_curso = etiquetas[-1]
    periodo_curso = periodos[-1].month

    matriz = pd.crosstab(ventana["Nodo"], ventana["Mes"])
    matriz = matriz.reindex(columns=etiquetas, fill_value=0)
    matriz = matriz.astype(int)

    fin = ventana["Fecha"].max()
    dias_totales = fin.days_in_month
    dias_transcurridos = max(fin.day, 1)
    col_proy = f"Proyección {mes_curso}"
    if dias_transcurridos < dias_totales:
        matriz[col_proy] = (matriz[mes_curso] / dias_transcurridos * dias_totales).round().astype(int)
    else:
        matriz[col_proy] = matriz[mes_curso]

    matriz["Total ventana"] = matriz[etiquetas].sum(axis=1)
    matriz["Total proyectado"] = matriz[meses_cerrados].sum(axis=1) + matriz[col_proy]
    matriz = matriz[matriz["Total proyectado"] >= min_total]
    matriz = matriz.sort_values("Total proyectado", ascending=False)

    logger.info("Mes en curso: %s (%s/%s días). Meses cerrados: %s",
                mes_curso, dias_transcurridos, dias_totales, ", ".join(meses_cerrados) or "ninguno")

    return {
        "matriz": matriz,
        "detalle": ventana.sort_values(["Fecha", "Nodo"], ascending=[False, True]),
        "orden_meses": list(periodos),
        "corte": {
            "mes_curso": mes_curso,
            "periodo_curso": periodo_curso,
            "dias_transcurridos": dias_transcurridos,
            "dias_totales": dias_totales,
            "meses_cerrados": meses_cerrados,
            "desde": ventana["Fecha"].min(),
            "hasta": fin,
        },
    }


def render_ascii(matriz: pd.DataFrame, top: int = 15, umbral: int = 100) -> str:
    """Tabla ASCII del top de nodos (lo que imprimía el script legacy)."""
    if matriz.empty:
        return "Sin datos para el top solicitado."
    columnas = list(matriz.columns)
    ancho_nodo = min(max(len(str(i)) for i in matriz.index) + 2, 26)
    lineas = [" | ".join([f"{'NODO AFECTADO':<{ancho_nodo}}"] + [f"{c:>14}" for c in columnas])]
    lineas.append("-" * len(lineas[0]))
    for nodo, fila in matriz.head(top).iterrows():
        celdas = []
        for col in columnas:
            valor = int(fila[col])
            marca = " 🔥" if ("Total" not in col and valor > umbral) else ""
            celdas.append(f"{valor:>12}{marca:<2}")
        lineas.append(" | ".join([f"{str(nodo)[:ancho_nodo - 2]:<{ancho_nodo}}"] + celdas))
    return "\n".join(lineas)
