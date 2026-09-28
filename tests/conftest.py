import pytest
from pathlib import Path
import pandas as pd
from src.data.etl import DataProcessor

BASE_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def processor() -> DataProcessor:
    return DataProcessor()


@pytest.fixture
def jira_csv_path() -> Path:
    return BASE_FIXTURES / "jira_sample.csv"


@pytest.fixture
def ans_excel_path() -> Path:
    return BASE_FIXTURES / "ans_sample.xlsx"


@pytest.fixture
def df_jira(processor: DataProcessor, jira_csv_path: Path) -> pd.DataFrame:
    return processor.process_jira(str(jira_csv_path))


@pytest.fixture
def df_ans(processor: DataProcessor, ans_excel_path: Path) -> pd.DataFrame:
    return processor.process_ans(str(ans_excel_path))
