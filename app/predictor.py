import os
import json
import joblib
import numpy as np
import pandas as pd
import hashlib
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")

class ModelPredictor:
    """Класс для инкапсуляции логики загрузки моделей и предсказаний."""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self._load_artifacts()

    def _load_artifacts(self):
        """Загрузка моделей, скейлера и названий фичей при старте."""
        with open(os.path.join(self.model_dir, 'feature_names.json'), 'r') as f:
            self.feature_names = json.load(f)

        self.model_v1 = joblib.load(os.path.join(self.model_dir, 'model_v1.joblib'))
        self.model_v2 = joblib.load(os.path.join(self.model_dir, 'model_v2.joblib'))
        self.scaler_v2 = joblib.load(os.path.join(self.model_dir, 'scaler_v2.joblib'))

    def get_model_version(self, client_id: str) -> str:
        """Определяет версию модели для A/B теста на основе хэша client_id."""
        hash_hex = hashlib.md5(client_id.encode('utf-8')).hexdigest()
        last_char = hash_hex[-1]
        return "v1" if int(last_char, 16) % 2 == 0 else "v2"

    def predict(self, data: dict) -> dict:
        """Делает предсказание на основе входного словаря."""

        input_features = [data.get(feat, 0) for feat in self.feature_names]

        features_df = pd.DataFrame([input_features], columns=self.feature_names)

        version = self.get_model_version(str(data.get('client_id', 'unknown')))

        if version == "v1":
            prediction = int(self.model_v1.predict(features_df)[0])
            probability = float(self.model_v1.predict_proba(features_df)[0][1])
        else:
            features_scaled = self.scaler_v2.transform(features_df)
            prediction = int(self.model_v2.predict(features_scaled)[0])
            probability = float(self.model_v2.predict_proba(features_scaled)[0][1])

        return {
            "model_version": version,
            "prediction": prediction,
            "default_probability": round(probability, 4)
        }
