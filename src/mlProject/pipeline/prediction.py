import joblib 
import numpy as np
import pandas as pd
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        # Load trained model and scaler
        self.model = joblib.load(Path("artifacts/model_trainer/model.joblib"))
        self.scaler = joblib.load(Path("artifacts/model_trainer/scaler.pkl"))

    def predict(self, data: pd.DataFrame):
        """
        data: pandas DataFrame with same feature columns as training data
        """
        # Scale input data
        data_scaled = self.scaler.transform(data)

        # Predict using trained model
        prediction = self.model.predict(data_scaled)

        return prediction
