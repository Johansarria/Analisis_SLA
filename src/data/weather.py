"""
Módulo para obtener datos climáticos históricos y pronóstico de Open-Meteo.
Zona: Bogotá, Colombia. Sin API key necesaria.
"""
import pandas as pd
import numpy as np
import requests
import logging
from pathlib import Path
from src.config import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

# Coordenadas por zona (Bogotá)
ZONE_COORDS = {
    'Zona Centro':             (4.65, -74.05),
    'Zona Noroccidente':       (4.72, -74.10),
    'Zona Area Metropolitana': (4.63, -74.15),
    'Zona Suroccidente ':      (4.58, -74.12),
    'Mantenimiento Preventivo': (4.65, -74.08),  # Promedio ciudad
}

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARS = "precipitation_sum,temperature_2m_max,windspeed_10m_max,weathercode"
CACHE_FILE = PROCESSED_DATA_DIR / "clima_bogota.csv"


def fetch_weather_range(lat, lon, start, end):
    """Obtiene datos climáticos diarios de Open-Meteo Archive API."""
    params = {
        'latitude': lat, 'longitude': lon,
        'start_date': start, 'end_date': end,
        'daily': DAILY_VARS,
        'timezone': 'America/Bogota',
    }
    try:
        r = requests.get(ARCHIVE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()['daily']
        df = pd.DataFrame({
            'Fecha': pd.to_datetime(data['time']),
            'precipitacion_mm': data['precipitation_sum'],
            'temp_max': data['temperature_2m_max'],
            'viento_max_kmh': data['windspeed_10m_max'],
            'weathercode': data['weathercode'],
        })
        return df
    except Exception as e:
        logger.warning(f"Error API Open-Meteo: {e}")
        return None


def fetch_forecast(lat, lon, days=14):
    """Obtiene pronóstico meteorológico (hasta 14 días)."""
    params = {
        'latitude': lat, 'longitude': lon,
        'daily': DAILY_VARS,
        'timezone': 'America/Bogota',
        'forecast_days': min(days, 16),
    }
    try:
        r = requests.get(FORECAST_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()['daily']
        return pd.DataFrame({
            'Fecha': pd.to_datetime(data['time']),
            'precipitacion_mm': data['precipitation_sum'],
            'temp_max': data['temperature_2m_max'],
            'viento_max_kmh': data['windspeed_10m_max'],
            'weathercode': data['weathercode'],
        })
    except Exception as e:
        logger.warning(f"Error forecast Open-Meteo: {e}")
        return None


def get_weather_for_zones(start_date, end_date, zones=None):
    """
    Obtiene clima histórico para todas las zonas.
    Retorna DataFrame con columnas: Fecha, Zona, precipitacion_mm, temp_max,
    viento_max_kmh, weathercode, es_lluvia_fuerte, lluvia_acum_3d.
    """
    if zones is None:
        zones = ZONE_COORDS

    # Intentar cargar de caché
    if CACHE_FILE.exists():
        cached = pd.read_csv(CACHE_FILE)
        cached['Fecha'] = pd.to_datetime(cached['Fecha'])
        c_start = cached['Fecha'].min()
        c_end = cached['Fecha'].max()
        if c_start <= pd.to_datetime(start_date) and c_end >= pd.to_datetime(end_date):
            logger.info(f"Clima cargado de cache ({len(cached)} registros)")
            return cached

    # Determinar corte: archive hasta 5 dias atras, forecast para reciente/futuro
    today = pd.Timestamp.now().normalize()
    archive_end = min(pd.to_datetime(end_date), today - pd.Timedelta(days=5))
    archive_start = pd.to_datetime(start_date)

    all_data = []
    for zona, (lat, lon) in zones.items():
        logger.info(f"  Descargando clima para {zona} ({lat}, {lon})...")
        parts = []

        # Parte 1: Datos historicos (archive API)
        if archive_start <= archive_end:
            df_arch = fetch_weather_range(
                lat, lon,
                archive_start.strftime('%Y-%m-%d'),
                archive_end.strftime('%Y-%m-%d'),
            )
            if df_arch is not None:
                parts.append(df_arch)
                logger.info(f"    Archive: {len(df_arch)} dias")

        # Parte 2: Datos recientes/futuros (forecast API)
        df_fc = fetch_forecast(lat, lon, days=16)
        if df_fc is not None:
            parts.append(df_fc)
            logger.info(f"    Forecast: {len(df_fc)} dias")

        if parts:
            df = pd.concat(parts, ignore_index=True)
            df = df.drop_duplicates(subset='Fecha', keep='first')
            # Filtrar al rango solicitado
            df = df[(df['Fecha'] >= str(start_date)[:10]) &
                    (df['Fecha'] <= str(end_date)[:10])]
            df['Zona'] = zona
            all_data.append(df)
        else:
            # Fallback: datos climatologicos promedio de Bogota
            logger.warning(f"  Usando datos climatologicos de fallback para {zona}")
            dates = pd.date_range(start_date, end_date)
            df = pd.DataFrame({
                'Fecha': dates,
                'Zona': zona,
                'precipitacion_mm': np.random.exponential(3.5, len(dates)),
                'temp_max': np.random.normal(18.5, 1.5, len(dates)),
                'viento_max_kmh': np.random.normal(12, 3, len(dates)),
                'weathercode': np.random.choice([0, 1, 2, 3, 51, 53, 61, 63, 80], len(dates)),
            })
            all_data.append(df)

    result = pd.concat(all_data, ignore_index=True)

    # Feature engineering de clima
    result['es_lluvia_fuerte'] = (result['precipitacion_mm'] > 10).astype(int)
    result['es_tormenta'] = result['weathercode'].isin([95, 96, 99]).astype(int)
    result['es_lluvia'] = (result['precipitacion_mm'] > 2).astype(int)

    # Lluvia acumulada 3 días (por zona)
    result = result.sort_values(['Zona', 'Fecha'])
    result['lluvia_acum_3d'] = (
        result.groupby('Zona')['precipitacion_mm']
        .transform(lambda x: x.rolling(3, min_periods=1).sum())
    )

    # Guardar caché
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(CACHE_FILE, index=False)
    logger.info(f"  Clima guardado en cache: {CACHE_FILE}")

    return result


def get_weather_features_for_dates(dates, zona, weather_df):
    """
    Dado un DataFrame de clima y una lista de fechas+zona,
    retorna las features climáticas correspondientes.
    """
    wz = weather_df[weather_df['Zona'] == zona].copy()
    wz = wz.set_index('Fecha')

    features = []
    for dt in dates:
        dt = pd.to_datetime(dt)
        if dt in wz.index:
            row = wz.loc[dt]
            features.append({
                'precipitacion_mm': row['precipitacion_mm'],
                'temp_max': row['temp_max'],
                'viento_max_kmh': row['viento_max_kmh'],
                'es_lluvia_fuerte': row['es_lluvia_fuerte'],
                'es_lluvia': row['es_lluvia'],
                'lluvia_acum_3d': row['lluvia_acum_3d'],
            })
        else:
            # Fallback: promedio climatológico del mes
            month = dt.month
            month_data = wz[wz.index.month == month]
            features.append({
                'precipitacion_mm': month_data['precipitacion_mm'].mean() if len(month_data) > 0 else 3.5,
                'temp_max': month_data['temp_max'].mean() if len(month_data) > 0 else 18.5,
                'viento_max_kmh': month_data['viento_max_kmh'].mean() if len(month_data) > 0 else 12.0,
                'es_lluvia_fuerte': 0,
                'es_lluvia': 0,
                'lluvia_acum_3d': month_data['lluvia_acum_3d'].mean() if len(month_data) > 0 else 10.0,
            })

    return pd.DataFrame(features)
