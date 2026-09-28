import pytest
from pathlib import Path
import pandas as pd
from src.config import FILE_JIRA_RAW, FILE_ANS_RAW, FILE_JIRA_CLEAN, FILE_ANS_CLEAN, MODEL_ANS_XGB, MODEL_ORACULO_XGB
from src.scripts.run_pipeline import run_full_pipeline


def test_shouldGenerateAllOutputFilesWhenRunningFullPipeline(monkeypatch, tmp_path: Path, jira_csv_path: Path, ans_excel_path: Path):
    """El pipeline completo debe generar los archivos procesados y modelos."""
    # Redirigir rutas de entrada a los fixtures
    # run_pipeline importa directamente las constantes, por lo que debemos parchear ambos: src.config y src.scripts.run_pipeline
    monkeypatch.setattr('src.config.FILE_JIRA_RAW', jira_csv_path)
    monkeypatch.setattr('src.scripts.run_pipeline.FILE_JIRA_RAW', jira_csv_path)
    monkeypatch.setattr('src.config.FILE_ANS_RAW', ans_excel_path)
    monkeypatch.setattr('src.scripts.run_pipeline.FILE_ANS_RAW', ans_excel_path)

    # Redirigir salidas a directorio temporal para no contaminar el proyecto
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    plots_dir = tmp_path / "plots"
    reports_dir = tmp_path / "reports"

    monkeypatch.setattr('src.config.FILE_JIRA_CLEAN', processed_dir / "datos_jira_regresion.csv")
    monkeypatch.setattr('src.scripts.run_pipeline.FILE_JIRA_CLEAN', processed_dir / "datos_jira_regresion.csv")
    monkeypatch.setattr('src.config.FILE_ANS_CLEAN', processed_dir / "datos_ans_limpios.csv")
    monkeypatch.setattr('src.scripts.run_pipeline.FILE_ANS_CLEAN', processed_dir / "datos_ans_limpios.csv")
    monkeypatch.setattr('src.config.MODEL_ANS_XGB', models_dir / "modelo_ans_xgboost.pkl")
    monkeypatch.setattr('src.config.MODEL_ORACULO_XGB', models_dir / "modelo_oraculo_xgb.pkl")
    monkeypatch.setattr('src.config.PLOTS_DIR', plots_dir)
    monkeypatch.setattr('src.utils.visualization.PLOTS_DIR', plots_dir)
    monkeypatch.setattr('src.config.REPORTS_DIR', reports_dir)
    monkeypatch.setattr('src.config.PROCESSED_DATA_DIR', processed_dir)
    monkeypatch.setattr('src.config.MODELS_DIR', models_dir)
    monkeypatch.setattr('src.core.models.MODELS_DIR', models_dir)

    run_full_pipeline()

    assert (processed_dir / "datos_jira_regresion.csv").exists()
    assert (processed_dir / "datos_ans_limpios.csv").exists()
    assert (models_dir / "modelo_ans_xgboost.pkl").exists()
    assert (models_dir / "modelo_oraculo_xgb.pkl").exists()
    assert (plots_dir / "radar_anomalias.png").exists()
