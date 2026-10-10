"""Genera el mapa de calor Nodo x Mes (informe de nodos calientes).

Uso:
    python -m src.scripts.heatmap_nodos --input data/incoming/info_nuevo.csv --months 3

Salidas:
    outputs/reports/heatmap_nodos_<AAAAMMDD>.xlsx  (matriz + detalle)
    outputs/plots/heatmap_nodos_<AAAAMMDD>.png     (visual del top N)
    data/processed/heatmap_nodos_interno.csv       (CSV interno de trabajo)
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import (PLOTS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, ensure_dirs,
                        setup_logging)
from src.core.heatmap import (COL_NODO, build_heatmap, read_jira_export,
                              render_ascii)

logger = logging.getLogger(__name__)


def plot_heatmap(matriz: pd.DataFrame, corte: dict, top: int, salida: Path) -> Path:
    """Guarda una imagen PNG con el top de nodos de la matriz."""
    datos = matriz.head(top).copy()
    datos = datos.drop(columns=[c for c in ("Total ventana", "Total proyectado") if c in datos.columns])
    if datos.empty:
        raise ValueError("La matriz quedó vacía: no hay nada que graficar.")

    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(datos.columns) + 3), 0.45 * len(datos) + 3))
    imagen = ax.imshow(datos.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(datos.columns)))
    ax.set_xticklabels(datos.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(datos.index)))
    ax.set_yticklabels([str(n)[:28] for n in datos.index])
    maximo = max(datos.values.max(), 1)
    for i in range(datos.shape[0]):
        for j in range(datos.shape[1]):
            valor = int(datos.values[i, j])
            ax.text(j, i, valor, ha="center", va="center", fontsize=8,
                    color="white" if valor > maximo * 0.6 else "#222222")
    ax.set_title(
        f"Nodos calientes · {corte['desde'].date()} a {corte['hasta'].date()} · "
        f"{corte['mes_curso']} proyectado ({corte['dias_transcurridos']}/{corte['dias_totales']} días)",
        fontsize=11,
    )
    fig.colorbar(imagen, ax=ax, shrink=0.7, label="Incidencias")
    fig.tight_layout()
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=150)
    plt.close(fig)
    logger.info("Imagen guardada en %s", salida)
    return salida


def exportar_excel(matriz: pd.DataFrame, detalle: pd.DataFrame, corte: dict, salida: Path) -> Path:
    """Escribe el informe .xlsx (hoja Heatmap + hoja Detalle)."""
    matriz_x = matriz.copy()
    matriz_x.index.name = "Nodo afectado"

    detalle_x = detalle.copy()
    detalle_x["Fecha"] = detalle_x["Fecha"].dt.tz_localize(None)
    detalle_x = detalle_x.rename(columns={"Fecha": "Creada", "Nodo": COL_NODO,
                                          "Zona": "Proyecto", "Clave": "Incidencia"})
    detalle_x["Mes"] = detalle_x["Periodo"].astype(str)
    detalle_x = detalle_x.drop(columns=["Periodo"])

    contexto = pd.DataFrame({
        "Parámetro": ["Desde", "Hasta", "Mes en curso", "Días transcurridos",
                      "Días del mes", "Incidencias en ventana", "Nodos distintos"],
        "Valor": [corte["desde"].strftime("%Y-%m-%d %H:%M"), corte["hasta"].strftime("%Y-%m-%d %H:%M"),
                  corte["mes_curso"], corte["dias_transcurridos"], corte["dias_totales"],
                  len(detalle), len(matriz)],
    })

    salida.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(salida, engine="openpyxl") as writer:
        matriz_x.to_excel(writer, sheet_name="Heatmap")
        contexto.to_excel(writer, sheet_name="Heatmap", startrow=len(matriz_x) + 3, index=False)
        detalle_x.to_excel(writer, sheet_name="Detalle", index=False)
    logger.info("Informe Excel guardado en %s", salida)
    return salida


def _entrada_por_defecto() -> Path:
    """Toma el export más reciente de data/incoming (donde caen los adjuntos)."""
    from src.config import DATA_DIR
    incoming = DATA_DIR / "incoming"
    archivos = sorted(incoming.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not archivos:
        raise FileNotFoundError(f"No hay CSV en {incoming}; pasa --input explícito.")
    return archivos[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Mapa de calor Nodo x Mes desde un export de Jira")
    parser.add_argument("--input", type=str, default=None,
                        help="Ruta del export CSV de Jira (por defecto: el más reciente en data/incoming)")
    parser.add_argument("--months", type=int, default=3, help="Tamaño de la ventana en meses")
    parser.add_argument("--end", type=str, default=None, help="Fecha de corte (YYYY-MM-DD). Por defecto: último dato")
    parser.add_argument("--inicio", type=str, default=None,
                        help="Fecha inicial explícita (YYYY-MM-DD); por defecto se calcula por meses calendario")
    parser.add_argument("--zona", type=str, default=None,
                        help="Limita a una o varias zonas separadas por coma (ej. ZSUR,Zona Suroccidente)")
    parser.add_argument("--min-total", type=float, default=0, help="Filtra nodos con total proyectado menor a este valor")
    parser.add_argument("--top", type=int, default=20, help="Nodos a mostrar en la imagen")
    parser.add_argument("--umbral", type=int, default=100, help="Marcar en rojo valores por encima de este umbral")
    parser.add_argument("--tz", type=str, default="America/Bogota")
    args = parser.parse_args(argv)

    setup_logging()
    ensure_dirs()

    zonas = [z for z in (args.zona or "").split(",") if z.strip()] or None
    ruta = Path(args.input) if args.input else _entrada_por_defecto()
    if not ruta.exists():
        logger.error("No existe el archivo de entrada: %s", ruta)
        return 1

    logger.info("Analizando %s (ventana: %s meses, zonas: %s)", ruta, args.months, zonas or "todas")
    export = read_jira_export(ruta)
    resultado = build_heatmap(export, months=args.months, end=args.end,
                              min_total=args.min_total, tz=args.tz, zonas=zonas,
                              inicio=args.inicio)
    matriz, detalle, corte = resultado["matriz"], resultado["detalle"], resultado["corte"]

    etiqueta = (zonas[0].upper() if zonas and len(zonas) == 1 else "") or "todas"
    print("\n" + "=" * 110)
    print(f" 🔥 MAPA DE CALOR NODO x MES · {corte['desde'].date()} → {corte['hasta'].date()} "
          f"(mes en curso: {corte['mes_curso']}, {corte['dias_transcurridos']}/{corte['dias_totales']} días)")
    print("=" * 110)
    print(render_ascii(matriz, top=args.top, umbral=args.umbral))
    print("=" * 110)
    print(">>> La columna de proyección asume que el ritmo diario del mes en curso se mantiene hasta el día "
          f"{corte['dias_totales']}.")

    sello = datetime.now().strftime("%Y%m%d")
    xlsx = exportar_excel(matriz, detalle, corte, REPORTS_DIR / f"heatmap_nodos_{etiqueta}_{sello}.xlsx")
    png = plot_heatmap(matriz, corte, args.top, PLOTS_DIR / f"heatmap_nodos_{etiqueta}_{sello}.png")
    csv_interno = PROCESSED_DATA_DIR / f"heatmap_nodos_{etiqueta}_interno.csv"
    matriz.to_csv(csv_interno, encoding="utf-8-sig")

    print(f">>> Informe: {xlsx}")
    print(f">>> Imagen:  {png}")
    print(f">>> CSV interno: {csv_interno}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
