import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.api import app
from app.predictor import ModelPredictor

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """Тест 1: Проверка эндпоинта /health"""
    response = client.get('/health')
    assert response.status_code == 200

    data = response.get_json()
    assert data['status'] == 'healthy'

def test_predict_success_v1(client):
    """Тест 2: Успешный запрос к /predict (проверяем структуру ответа)"""

    payload = {
        "client_id": "user_2",
        "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
        "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
        "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0,
        "BILL_AMT5": 0, "BILL_AMT6": 0, "PAY_AMT1": 0, "PAY_AMT2": 689,
        "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
    }

    response = client.post('/predict', json=payload)
    assert response.status_code == 200

    data = response.get_json()

    assert 'client_id' in data
    assert 'model_version' in data
    assert 'prediction' in data
    assert 'default_probability' in data

    assert data['client_id'] == 'user_2'
    assert data['model_version'] == 'v1'
    assert data['prediction'] == 1

def test_predict_success_v2(client):
    """Тест 3: Успешный запрос к /predict (проверяем попадание на v2)"""
    payload = {
        "client_id": "test_user_1",
        "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
        "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
        "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0,
        "BILL_AMT5": 0, "BILL_AMT6": 0, "PAY_AMT1": 0, "PAY_AMT2": 689,
        "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
    }

    response = client.post('/predict', json=payload)
    data = response.get_json()

    assert response.status_code == 200
    assert data['model_version'] == 'v2'
    assert data['prediction'] in [0, 1]
    assert 0.0 <= data['default_probability'] <= 1.0

def test_predict_missing_data(client):
    """Тест 4: Ошибка при отправке не JSON"""
    response = client.post('/predict', data="just a string, not json")
    assert response.status_code == 400
    assert 'error' in response.get_json()
