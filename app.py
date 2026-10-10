from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import os
from werkzeug.utils import secure_filename

from src.config import PROCESSED_DATA_DIR, REPORTS_DIR, RAW_DATA_DIR, FILE_JIRA_RAW
from src.scripts.run_pipeline import run_full_pipeline
from src.scripts.run_forecasting import run_forecasting

app = Flask(__name__)

# Rutas de datos
FILE_DEMANDA = PROCESSED_DATA_DIR / "demanda_diaria.csv"
FILE_PRED = PROCESSED_DATA_DIR / "prediccion_demanda_futura.csv"
FILE_HOLDOUT = REPORTS_DIR / "holdout_v4_vs_v5.csv"

FILE_PERFILES = PROCESSED_DATA_DIR / "perfiles_nodos.csv"
FILE_CRITICOS = PROCESSED_DATA_DIR / "nodos_criticos.csv"

def get_dashboard_data(start_date=None, end_date=None):
    data = {
        'kpis': {
            'total_tickets_today': 0,
            'label_suffix': 'hoy',
            'last_update': '---',
            'sla_compliance': 98.4,
            'critical_nodes': 0,
            'active_zones': 0
        },
        'distribution': {},
        'predictions_summary': {},
        'node_breakdown': [],
        'forecast_line': {'labels': [], 'values': []},
        'precision_metrics': [],
        'anomalies': {'labels': [], 'values': []}
    }
    
    is_range = bool(start_date and end_date and start_date.strip() and end_date.strip())
    
    # 1. Datos Históricos
    if FILE_DEMANDA.exists():
        df_hist = pd.read_csv(FILE_DEMANDA)
        df_hist['Fecha'] = pd.to_datetime(df_hist['Fecha'])
        
        # Filtrar
        if start_date and start_date.strip():
            df_hist = df_hist[df_hist['Fecha'] >= pd.to_datetime(start_date)]
        if end_date and end_date.strip():
            df_hist = df_hist[df_hist['Fecha'] <= pd.to_datetime(end_date)]

        if not df_hist.empty:
            last_date = df_hist['Fecha'].max()
            
            if is_range:
                days_hist = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
                if days_hist <= 0: days_hist = 1
                total_tickets = df_hist['Tickets_Total'].sum()
                # Promedio diario por zona
                dist = (df_hist.groupby('Zona')['Tickets_Total'].sum() / days_hist).to_dict()
                label_suffix = f"promedio diario ({days_hist}d)"
            else:
                total_tickets = df_hist[df_hist['Fecha'] == last_date]['Tickets_Total'].sum()
                dist = df_hist[df_hist['Fecha'] == last_date].groupby('Zona')['Tickets_Total'].sum().to_dict()
                label_suffix = "hoy"

            data['kpis'].update({
                'total_tickets_today': int(total_tickets),
                'label_suffix': label_suffix,
                'last_update': last_date.strftime('%Y-%m-%d'),
                'active_zones': df_hist['Zona'].nunique()
            })
            data['distribution'] = dist

    # 2. Predicciones
    if FILE_PRED.exists():
        df_pred = pd.read_csv(FILE_PRED)
        df_pred['Fecha'] = pd.to_datetime(df_pred['Fecha'])
        
        mask = pd.Series([True] * len(df_pred))
        if start_date and start_date.strip():
            mask &= (df_pred['Fecha'] >= pd.to_datetime(start_date))
        if end_date and end_date.strip():
            mask &= (df_pred['Fecha'] <= pd.to_datetime(end_date))
        
        df_pred_filtered = df_pred[mask]

        if not df_pred_filtered.empty:
            days_pred = df_pred_filtered['Fecha'].nunique()
            if days_pred <= 0: days_pred = 1
            if 'Q30' not in df_pred_filtered.columns: df_pred_filtered['Q30'] = df_pred_filtered['Prediccion_Tickets'] * 0.8
            if 'Q70' not in df_pred_filtered.columns: df_pred_filtered['Q70'] = df_pred_filtered['Prediccion_Tickets'] * 1.2

            # Promedio diario proyectado
            data['predictions_summary'] = (df_pred_filtered.groupby('Zona')['Prediccion_Tickets'].sum() / days_pred).to_dict()
            
            node_agg = df_pred_filtered.groupby(['Nodo', 'Zona']).agg({
                'Prediccion_Tickets': 'mean',
                'Q30': 'mean',
                'Q70': 'mean'
            }).reset_index()
            data['node_breakdown'] = node_agg.to_dict(orient='records')
            
            line_data = df_pred_filtered.groupby('Fecha')['Prediccion_Tickets'].sum().reset_index()
            data['forecast_line'] = {
                'labels': line_data['Fecha'].dt.strftime('%d-%b').tolist(),
                'values': line_data['Prediccion_Tickets'].tolist()
            }

    # 3. Anomalías y Nodos Críticos
    if FILE_CRITICOS.exists():
        df_crit = pd.read_csv(FILE_CRITICOS)
        data['kpis']['critical_nodes'] = len(df_crit)
        data['anomalies'] = {
            'labels': df_crit['Nodo'].tolist(),
            'values': df_crit['Total_Fallas'].tolist(),
            'risk_level': df_crit['Porcentaje_Critico'].tolist()
        }

    # 4. Métricas de Precisión
    if FILE_HOLDOUT.exists():
        df_metrics = pd.read_csv(FILE_HOLDOUT)
        data['precision_metrics'] = df_metrics.to_dict(orient='records')

    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def api_data():
    start = request.args.get('start')
    end = request.args.get('end')
    try:
        # Validar fechas si vienen
        if start: pd.to_datetime(start)
        if end: pd.to_datetime(end)
        return jsonify(get_dashboard_data(start, end))
    except Exception as e:
        return jsonify({'error': f"Error en procesamiento de fechas o datos: {str(e)}"}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        # Si es Jira, lo guardamos como JiraATP.csv para que el pipeline lo tome
        if 'jira' in filename.lower():
            save_path = FILE_JIRA_RAW
        else:
            save_path = RAW_DATA_DIR / filename
            
        file.save(save_path)
        return jsonify({'message': f'Archivo {filename} subido con éxito a {save_path.name}'})

@app.route('/api/run-pipeline', methods=['POST'])
def run_pipeline():
    try:
        # 1. Ejecutar Pipeline Maestro (Clasificación, Anomalías, etc.)
        run_full_pipeline()
        
        # 2. Ejecutar Forecasting si el archivo de Jira existe
        if FILE_JIRA_RAW.exists():
            run_forecasting(str(FILE_JIRA_RAW), days_to_predict=30)
            
        return jsonify({'message': 'Procesamiento completo finalizado con éxito.'})
    except Exception as e:
        return jsonify({'error': f'Error durante el procesamiento: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
