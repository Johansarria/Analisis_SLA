"""Informe visual de patrones temporales (hora, día, fecha) para audiencia no tecnica.

Uso:
    python -m src.scripts.analisis_temporal --input data/incoming/Jira_ATP_nuevo.csv --zona ZSUR --months 3

Salidas:
    outputs/reports/analisis_temporal_<ZONA>_<AAAAMMDD>.pdf    informe de lectura (graficas + resumen)
    outputs/reports/analisis_temporal_<ZONA>_<AAAAMMDD>.xlsx   mismas cifras en hojas de Excel + graficas
    outputs/plots/temporal_<grafica>_<ZONA>_<AAAAMMDD>.png     cada grafica suelta, para pegar en una presentacion
    data/processed/temporal_<ZONA>_detalle.csv                 detalle con las columnas de hora/dia
"""
from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from src.config import (PLOTS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, ensure_dirs,
                        setup_logging)
from src.core import temporal as tp
from src.core.heatmap import MESES_ES, build_heatmap, read_jira_export

logger = logging.getLogger(__name__)

COLOR_FRANJA = {"Madrugada": "#3d5a80", "Mañana": "#2a9d8f",
                "Tarde": "#e9a13b", "Noche": "#8e6bd0"}
COLOR_PRINCIPAL = "#4a7fb5"
COLOR_ALERTA = "#c1121f"
GRIS = "#5a5a5a"
TAMANO_PAGINA = (11.69, 8.27)  # A4 horizontal


def _estilo() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 12,
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.labelsize": 12.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _subtitulo(fig, texto: str, ancho: int = 118) -> None:
    """Frase explicativa (envuelta en varias líneas) encima del grafico, en pagina A4."""
    for i, linea in enumerate(textwrap.wrap(texto, ancho)):
        fig.text(0.09, 0.955 - i * 0.034, linea, fontsize=11.5, color=GRIS, va="top")


def _marco(fig, bottom: float = 0.17, right: float = 0.95, left: float = 0.09) -> None:
    """Reserva el espacio del titulo/frase arriba, de la leyenda abajo y de las etiquetas."""
    fig.subplots_adjust(top=0.80, bottom=bottom, left=left, right=right)


# --------------------------------------------------------------------------- #
# Graficas
# --------------------------------------------------------------------------- #
def fig_hora(df: pd.DataFrame, met: dict):
    horas = tp.resumen_por_hora(df)
    fig, ax = plt.subplots(figsize=TAMANO_PAGINA)
    _marco(fig, bottom=0.20)
    colores = [COLOR_FRANJA[tp.franja_horaria(h)] for h in horas.index]
    barras = ax.bar(horas.index, horas.values, color=colores, edgecolor="white", width=0.82)
    tope = max(int(horas.max()), 1)
    for barra, valor in zip(barras, horas.values):
        if valor:
            ax.text(barra.get_x() + barra.get_width() / 2, valor + tope * 0.015,
                    str(int(valor)), ha="center", fontsize=10, color="#333333")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=10.5)
    ax.set_xlabel("Hora del día (0 = medianoche, 12 = mediodía, 23 = 11 de la noche)")
    ax.set_ylabel("Fallas reportadas")
    ax.set_ylim(0, tope * 1.18)
    ax.set_title("¿A qué hora del día se reportan las fallas?", pad=12)
    _subtitulo(fig, f"Cada barra es una hora del día. El pico está a las {met['hora_pico']:02d}:00 con "
                  f"{met['hora_pico_casos']} fallas; la {met['franja_pico'].lower()} concentra el "
                  f"{met['franja_pico_pct']}% de todos los reportes.")
    ax.legend(handles=[Patch(facecolor=c, label=f) for f, c in COLOR_FRANJA.items()],
              ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False,
              title="Franja del día")
    return fig


def fig_dia_semana(df: pd.DataFrame, met: dict):
    dias = tp.resumen_por_dia_semana(df)
    fig, ax = plt.subplots(figsize=TAMANO_PAGINA)
    _marco(fig, bottom=0.17)
    colores = [COLOR_ALERTA if d == met["dia_semana_pico"] else COLOR_PRINCIPAL for d in dias.index]
    barras = ax.bar(dias.index, dias.values, color=colores, edgecolor="white", width=0.7)
    tope = max(int(dias.max()), 1)
    for barra, valor in zip(barras, dias.values):
        ax.text(barra.get_x() + barra.get_width() / 2, valor + tope * 0.02, str(int(valor)),
                ha="center", fontsize=11.5, color="#333333")
    ax.set_ylabel("Fallas reportadas")
    ax.set_ylim(0, tope * 1.15)
    ax.set_title("¿Qué días de la semana se reportan más fallas?", pad=12)
    _subtitulo(fig, f"El día más cargado es el {met['dia_semana_pico'].lower()} con "
                  f"{met['dia_semana_pico_casos']} fallas, pero ojo: el sábado "
                  f"({met['sabado_casos']}) se comporta como un día entre semana. El que sí baja "
                  f"es el domingo, con {met['domingo_casos']} fallas.")
    return fig


def fig_serie(serie: pd.DataFrame, met: dict):
    fig, ax = plt.subplots(figsize=TAMANO_PAGINA)
    _marco(fig, bottom=0.22)
    for dia in serie.index:
        if dia.weekday() >= 5:
            ax.axvspan(dia, dia + pd.Timedelta(days=1), color="#999999", alpha=0.13, lw=0)
    ax.plot(serie.index, serie["Incidencias"], color=COLOR_PRINCIPAL, lw=1.5, label="Fallas por día")
    ax.plot(serie.index, serie["Promedio7"], color=COLOR_ALERTA, lw=2.6,
            label="Promedio de los últimos 7 días (quita el ruido diario)")
    ax.axhline(met["dia_tipico"], ls="--", color="#444444", lw=1.3,
               label=f"Día típico: {met['dia_tipico']:.0f} fallas")

    ultimo = serie.index[-1]
    ax.axvspan(ultimo, ultimo + pd.Timedelta(days=1), color=COLOR_FRANJA["Tarde"], alpha=0.35, lw=0)
    ax.annotate("último día\n(incompleto)", xy=(ultimo, serie["Incidencias"].iloc[-1]),
                xytext=(-90, 42), textcoords="offset points", fontsize=10.5, color="#8a5a00",
                arrowprops=dict(arrowstyle="-", color="#8a5a00", lw=1))
    pico = serie.loc[serie["Incidencias"].idxmax()]
    ax.annotate(f"día más crítico: {pico.name.strftime('%d/%m')} ({int(pico['Incidencias'])} fallas)",
                xy=(serie["Incidencias"].idxmax(), pico["Incidencias"]),
                xytext=(-30, 30), textcoords="offset points", fontsize=10.5, color=COLOR_ALERTA,
                arrowprops=dict(arrowstyle="-", color=COLOR_ALERTA, lw=1))

    ax.set_ylabel("Fallas reportadas por día")
    ax.set_title("¿Cómo se movió la operación día a día?", pad=12)
    _subtitulo(fig, "Las franjas grises son los fines de semana. En promedio se reportan "
                  f"{met['dia_tipico']:.0f} fallas por día; el peor día del periodo fue el "
                  f"{pd.Timestamp(met['fecha_max']).strftime('%d/%m')} con {met['dia_max']}.")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    fig.autofmt_xdate(rotation=30)
    return fig


def fig_nodo_franja(matriz: pd.DataFrame, met: dict, top: int = 8):
    datos = matriz.head(top)
    fig, ax = plt.subplots(figsize=TAMANO_PAGINA)
    _marco(fig, bottom=0.14, right=0.92, left=0.20)
    imagen = ax.imshow(datos.values, cmap="YlOrRd", aspect="auto")
    umbral = datos.values.max() * 0.6 if datos.values.size else 0
    for i in range(datos.shape[0]):
        for j in range(datos.shape[1]):
            valor = int(datos.values[i, j])
            ax.text(j, i, valor, ha="center", va="center", fontsize=11.5,
                    color="white" if valor > umbral else "#222222")
    ax.set_xticks(range(len(datos.columns)))
    rangos = {nombre: f"{ini:02d}:00-{fin:02d}:59" for nombre, ini, fin in tp.FRANJAS}
    ax.set_xticklabels([f"{c}\n{rangos.get(c, '')}" for c in datos.columns], fontsize=11.5)
    ax.set_yticks(range(len(datos.index)))
    ax.set_yticklabels([str(n)[:26] for n in datos.index], fontsize=11.5)
    ax.grid(False)
    ax.set_title("¿Qué nodos fallan más y a qué hora del día?", pad=12)
    _subtitulo(fig, f"Filas: los {len(datos)} nodos con más fallas. Columnas: momento del día. "
                  f"El caso más marcado es {met['nodo_franja_pico']}, con "
                  f"{met['nodo_franja_casos']} fallas en ese lapso.")
    fig.colorbar(imagen, ax=ax, shrink=0.75, label="Fallas reportadas")
    return fig


def fig_dia_hora(matriz: pd.DataFrame):
    fig, ax = plt.subplots(figsize=TAMANO_PAGINA)
    _marco(fig, bottom=0.16, right=0.92)
    imagen = ax.imshow(matriz.values, cmap="YlOrRd", aspect="auto")
    tope = matriz.values.max() if matriz.values.size else 0
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            valor = int(matriz.values[i, j])
            if valor >= max(tope * 0.55, 1):
                ax.text(j, i, valor, ha="center", va="center", fontsize=9,
                        color="white" if valor > tope * 0.75 else "#222222")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=9.5)
    ax.set_yticks(range(len(matriz.index)))
    ax.set_yticklabels(matriz.index, fontsize=11.5)
    ax.set_xlabel("Hora del día")
    ax.grid(False)
    pos = matriz.stack().idxmax()
    ax.set_title("El mapa completo de la semana: días y horas", pad=12)
    _subtitulo(fig, f"Cada cuadro es un día y una hora. El momento más crítico de la semana es el "
                  f"{pos[0].lower()} a las {int(pos[1]):02d}:00, con {int(matriz.stack().max())} fallas.")
    fig.colorbar(imagen, ax=ax, shrink=0.75, label="Fallas reportadas")
    return fig


def fig_proyeccion(proy: dict, met: dict):
    datos = proy["proximas"]
    fig, ax = plt.subplots(figsize=TAMANO_PAGINA)
    _marco(fig, bottom=0.20)
    colores = [COLOR_FRANJA["Noche"] if d.weekday() >= 5 else COLOR_PRINCIPAL for d in datos["Fecha"]]
    barras = ax.bar(datos["Fecha"], datos["Esperado"], color=colores, width=0.75, edgecolor="white")
    promedio = datos["Esperado"].mean()
    ax.axhline(promedio, ls="--", color=COLOR_ALERTA, lw=1.6,
               label=f"Promedio esperado: {promedio:.0f} fallas por día")
    for barra, valor, fecha in zip(barras, datos["Esperado"], datos["Fecha"]):
        if fecha.weekday() == 0:
            ax.text(barra.get_x() + barra.get_width() / 2, valor + 0.6, f"{valor:.0f}",
                    ha="center", fontsize=9.5, color="#333333")
    ax.set_ylabel("Fallas esperadas por día")
    ax.set_xticks(datos["Fecha"][::2])
    ax.set_xticklabels([f"{d.strftime('%d/%m')}" for d in datos["Fecha"][::2]], fontsize=10)
    ax.set_title(f"Próximas {proy['semanas']} semanas: qué esperar si el patrón se mantiene", pad=12)
    _subtitulo(fig, f"Del {proy['inicio'].strftime('%d/%m')} al {proy['fin'].strftime('%d/%m')} se "
                  f"esperan alrededor de {proy['total']:.0f} fallas ("
                  f"{proy['total'] / proy['semanas']:.0f} por semana), calculadas con el promedio de "
                  "cada día de la semana en los meses analizados.")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    return fig


# --------------------------------------------------------------------------- #
# Paginas de texto y armado del informe
# --------------------------------------------------------------------------- #
def _bloques_informe(met: dict, proy: dict, etiqueta: str, zona_txt: str) -> list:
    """Hallazgos en lenguaje llano (los que se imprimen en la primera pagina)."""
    top = met["top_nodos"]
    top_txt = ", ".join(f"{n} ({int(v)})" for n, v in top.items())
    porcentaje_top = round(float(top.sum() / met["total"] * 100), 1)
    return [
        ("Qué se analizó",
         f"Las {met['total']} fallas reportadas en {zona_txt} entre el {met['desde'].strftime('%d/%m/%Y')} "
         f"y el {met['hasta'].strftime('%d/%m/%Y')} ({met['dias_cubiertos']} días, {met['nodos']} nodos). "
         "Cada falla cuenta una vez y viene de la fecha de reporte en Jira."),
        ("Lo más importante",
         f"Las fallas no se reparten parejo en el día: el {met['pct_7_19']}% se reporta entre las 7 de "
         f"la mañana y las 7 de la noche, y la hora con más reportes es las {met['hora_pico']:02d}:00 "
         f"({met['hora_pico_casos']} casos). En la semana, el día más cargado es el "
         f"{met['dia_semana_pico'].lower()} ({met['dia_semana_pico_casos']} fallas), pero el sábado "
         f"({met['sabado_casos']}) se comporta igual que un día entre semana; el domingo es el único "
         f"que baja de verdad, con {met['domingo_casos']}."),
        ("Dónde duele",
         f"Los 5 nodos con más fallas son {top_txt}. Juntos concentran el {porcentaje_top}% del total, "
         f"así que una buena parte del trabajo se explica por unos pocos puntos. El caso más marcado es "
         f"{met['nodo_franja_pico']}, con {met['nodo_franja_casos']} fallas."),
        ("El día típico",
         f"En un día normal se reportan alrededor de {met['dia_tipico']:.0f} fallas. El peor día del "
         f"periodo fue el {pd.Timestamp(met['fecha_max']).strftime('%d/%m/%Y')} con {met['dia_max']}. "
         f"La hora de mayor reporte es las {met['hora_pico']:02d}:00, con {met['hora_pico_casos']} casos."),
        ("Qué esperar",
         f"Si el patrón de los últimos meses se mantiene, en las próximas {proy['semanas']} semanas "
         f"(del {proy['inicio'].strftime('%d/%m')} al {proy['fin'].strftime('%d/%m')}) se reportarían "
         f"cerca de {proy['total']:.0f} fallas, alrededor de {proy['total'] / proy['semanas']:.0f} por "
         f"semana. Las barras más altas de esa proyección son el {met['dia_semana_pico'].lower()} y el "
         "sábado, que son los dos días que más reportes generan."),
        ("Cómo leerlo (límites)",
         "La hora que aparece es la del registro en Jira, no necesariamente la del daño: si una falla se "
         "detecta en la madrugada pero se reporta a las 8 de la mañana, cuenta a las 8. La proyección es "
         "un promedio del patrón observado, no un modelo que incluya clima, obras, cortes de energía ni "
         "cambios de red."),
    ]


def _fecha_larga(ts) -> str:
    """Fecha en español (strftime('%B') devolvería el mes en inglés)."""
    return f"{ts.day} de {MESES_ES[ts.month].lower()} de {ts.year}"


def pagina_resumen(met: dict, proy: dict, etiqueta: str, zona_txt: str, generado: str):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.07, 0.945, "Informe de operación · patrones de falla", fontsize=21, weight="bold",
             color="#1f3b57")
    fig.text(0.07, 0.918, f"{zona_txt} · {_fecha_larga(met['desde'])} "
                          f"al {_fecha_larga(met['hasta'])}", fontsize=12.5, color=GRIS)
    fig.text(0.07, 0.895, f"Generado el {generado} · {etiqueta}", fontsize=9.5, color="#8a8a8a")

    y = 0.865
    for titulo, texto in _bloques_informe(met, proy, etiqueta, zona_txt):
        fig.text(0.07, y, titulo, fontsize=13.5, weight="bold", color="#1f3b57")
        y -= 0.027
        for linea in textwrap.wrap(texto, 94):
            fig.text(0.07, y, linea, fontsize=10.8, color="#222222")
            y -= 0.0205
        y -= 0.016

    fig.text(0.07, 0.055, "Las gráficas de las páginas siguientes muestran de dónde salen estas cifras.",
             fontsize=10.5, style="italic", color=GRIS)
    return fig


# --------------------------------------------------------------------------- #
# Salidas
# --------------------------------------------------------------------------- #
def exportar_excel(ruta: Path, met: dict, horas: pd.Series, franjas: pd.Series, dias: pd.Series,
                   serie: pd.DataFrame, matriz_nf: pd.DataFrame, matriz_dh: pd.DataFrame,
                   proy: dict, detalle: pd.DataFrame, graficas: list) -> Path:
    contexto = [
        ("Periodo analizado", f"{met['desde'].strftime('%d/%m/%Y')} a {met['hasta'].strftime('%d/%m/%Y')}"),
        ("Fallas analizadas", met["total"]),
        ("Días cubiertos", met["dias_cubiertos"]),
        ("Nodos con fallas", met["nodos"]),
        ("Fallas por día (día típico)", met["dia_tipico"]),
        ("Hora con más reportes", f"{met['hora_pico']:02d}:00 ({met['hora_pico_casos']} fallas)"),
        ("Franja con más reportes", f"{met['franja_pico']} ({met['franja_pico_pct']}%)"),
        ("Día de la semana más cargado", f"{met['dia_semana_pico']} ({met['dia_semana_pico_casos']} fallas)"),
        ("Peso del fin de semana", f"{met['fin_de_semana_pct']}%"),
        ("Peor día del periodo", f"{met['fecha_max']} ({met['dia_max']} fallas)"),
        ("Nodo crítico por franja", f"{met['nodo_franja_pico']} ({met['nodo_franja_casos']} fallas)"),
        ("Proyección próximas semanas", f"{proy['total']:.0f} fallas en {proy['semanas']} semanas"),
    ]
    resumen = pd.DataFrame(contexto, columns=["Pregunta", "Respuesta"])

    detalle_x = detalle.copy()
    if "Fecha" in detalle_x.columns and getattr(detalle_x["Fecha"].dt, "tz", None) is not None:
        detalle_x["Fecha"] = detalle_x["Fecha"].dt.tz_localize(None)
    for col in detalle_x.columns:
        if str(detalle_x[col].dtype).startswith("period"):
            detalle_x[col] = detalle_x[col].astype(str)

    ruta.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(ruta, engine="openpyxl") as writer:
        resumen.to_excel(writer, sheet_name="Resumen", index=False)
        horas.rename("Fallas").to_frame().to_excel(writer, sheet_name="Por hora")
        franjas.rename("Fallas").to_frame().to_excel(writer, sheet_name="Por franja")
        dias.rename("Fallas").to_frame().to_excel(writer, sheet_name="Por dia de semana")
        serie.to_excel(writer, sheet_name="Por fecha")
        matriz_nf.to_excel(writer, sheet_name="Nodo x franja")
        matriz_dh.to_excel(writer, sheet_name="Dia x hora")
        prom = (proy["promedio_por_dia"].rename("Fallas por día")
                .rename_axis("Día de la semana").reset_index())
        prom.to_excel(writer, sheet_name="Proyeccion", index=False)
        proy["proximas"].to_excel(writer, sheet_name="Proyeccion", startrow=len(proy["promedio_por_dia"]) + 4,
                                  index=False)
        detalle_x.to_excel(writer, sheet_name="Detalle", index=False)
        pd.DataFrame({"Gráficas": ["Las imágenes van debajo, una por fila"]}).to_excel(
            writer, sheet_name="Graficas", index=False)

        hoja = writer.sheets["Graficas"]
        try:
            from openpyxl.drawing.image import Image as ImagenXL
            fila = 3
            for png in graficas:
                imagen = ImagenXL(str(png))
                alto = int(imagen.height * (900 / imagen.width))
                imagen.width, imagen.height = 900, alto
                hoja.add_image(imagen, f"A{fila}")
                fila += int(alto / 19) + 2
        except Exception as exc:  # pragma: no cover - depende de Pillow
            logger.warning("No se pudieron incrustar las imágenes en Excel: %s", exc)
    logger.info("Informe Excel guardado en %s", ruta)
    return ruta


def exportar_pdf(ruta: Path, paginas: list) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(ruta) as pdf:
        for fig in paginas:
            pdf.savefig(fig)
    logger.info("Informe PDF guardado en %s", ruta)
    return ruta


def _entrada_por_defecto() -> Path:
    from src.config import DATA_DIR
    archivos = sorted((DATA_DIR / "incoming").glob("*.csv"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    if not archivos:
        raise FileNotFoundError("No hay CSV en data/incoming; pasa --input explícito.")
    return archivos[0]


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Informe visual de patrones temporales de fallas")
    parser.add_argument("--input", type=str, default=None,
                        help="Export CSV de Jira (por defecto: el más reciente en data/incoming)")
    parser.add_argument("--zona", type=str, default=None, help="Zona a analizar (ej. ZSUR)")
    parser.add_argument("--months", type=int, default=3, help="Ventana en meses calendario")
    parser.add_argument("--end", type=str, default=None, help="Fecha de corte (YYYY-MM-DD)")
    parser.add_argument("--inicio", type=str, default=None, help="Fecha inicial explícita (YYYY-MM-DD)")
    parser.add_argument("--semanas", type=int, default=4, help="Semanas a proyectar")
    parser.add_argument("--top-nodos", type=int, default=8, help="Nodos en la gráfica de nodo x franja")
    parser.add_argument("--tz", type=str, default="America/Bogota")
    args = parser.parse_args(argv)

    setup_logging()
    ensure_dirs()
    _estilo()

    zonas = [z for z in (args.zona or "").split(",") if z.strip()] or None
    ruta = Path(args.input) if args.input else _entrada_por_defecto()
    if not ruta.exists():
        logger.error("No existe el archivo de entrada: %s", ruta)
        return 1

    export = read_jira_export(ruta)
    resultado = build_heatmap(export, months=args.months, end=args.end, tz=args.tz,
                              zonas=zonas, inicio=args.inicio)
    detalle = tp.add_temporal_features(resultado["detalle"])
    if detalle["FechaDia"].nunique() < 14:
        logger.warning("El periodo tiene menos de 14 días: la proyección semanal es poco confiable.")

    proy = tp.proyeccion_por_dia_semana(detalle, semanas=args.semanas)
    met = tp.metricas(detalle, proy)
    serie = tp.serie_diaria(detalle)
    matriz_nf = tp.matriz_nodo_franja(detalle)
    matriz_dh = tp.matriz_dia_hora(detalle)

    etiqueta = (zonas[0].upper() if zonas and len(zonas) == 1 else "") or "todas"
    zona_txt = {"ZSUR": "la zona sur (ZSUR)", "ZCEN": "la zona centro (ZCEN)",
                "ZNOR": "la zona norte (ZNOR)", "ZAMP": "la zona área metropolitana (ZAMP)",
                "MANPR": "mantenimiento preventivo"}.get(etiqueta, f"la zona {etiqueta}" if etiqueta != "todas" else "todas las zonas")
    sello = datetime.now().strftime("%Y%m%d")
    generado = datetime.now().strftime("%d/%m/%Y %H:%M")

    graficas = []
    paginas = [pagina_resumen(met, proy, etiqueta, zona_txt, generado)]
    for nombre, fig in (("hora", fig_hora(detalle, met)),
                        ("dia_semana", fig_dia_semana(detalle, met)),
                        ("evolucion_diaria", fig_serie(serie, met)),
                        ("nodo_franja", fig_nodo_franja(matriz_nf, met, args.top_nodos)),
                        ("dia_hora", fig_dia_hora(matriz_dh)),
                        ("proyeccion", fig_proyeccion(proy, met))):
        png = PLOTS_DIR / f"temporal_{nombre}_{etiqueta}_{sello}.png"
        fig.savefig(png, dpi=150)
        graficas.append(png)
        paginas.append(fig)
        logger.info("Gráfica guardada en %s", png)

    pdf = exportar_pdf(REPORTS_DIR / f"analisis_temporal_{etiqueta}_{sello}.pdf", paginas)
    for fig in paginas:
        plt.close(fig)

    xlsx = exportar_excel(REPORTS_DIR / f"analisis_temporal_{etiqueta}_{sello}.xlsx", met,
                          tp.resumen_por_hora(detalle), tp.resumen_por_franja(detalle),
                          tp.resumen_por_dia_semana(detalle), serie, matriz_nf, matriz_dh,
                          proy, detalle, graficas)
    csv_interno = PROCESSED_DATA_DIR / f"temporal_{etiqueta}_detalle.csv"
    detalle.to_csv(csv_interno, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print(f" PATRONES DE FALLA · {zona_txt} · {met['desde'].date()} → {met['hasta'].date()}")
    print("=" * 100)
    for titulo, texto in _bloques_informe(met, proy, etiqueta, zona_txt):
        print(f"\n{titulo.upper()}")
        print(textwrap.fill(texto, 100))
    print("\n" + "=" * 100)
    print(f">>> Informe PDF:  {pdf}")
    print(f">>> Informe XLSX: {xlsx}")
    print(f">>> Gráficas:     {PLOTS_DIR} (temporal_*_{etiqueta}_{sello}.png)")
    print(f">>> Detalle:      {csv_interno}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
