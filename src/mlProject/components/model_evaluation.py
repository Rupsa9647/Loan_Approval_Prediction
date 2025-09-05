import pandas as pd
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from mlProject.utils.common import save_json
from mlProject.entity.config_entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        acc = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average="binary")
        recall = recall_score(actual, pred, average="binary")
        f1 = f1_score(actual, pred, average="binary")
        try:
            roc_auc = roc_auc_score(actual, pred)
        except:
            roc_auc = None
        return acc, precision, recall, f1, roc_auc

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        # ✅ Load scaler
        scaler = joblib.load(self.config.scaler_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]

        # ✅ Scale test data
        test_x_scaled = scaler.transform(test_x)

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_labels = model.predict(test_x_scaled)

            (acc, precision, recall, f1, roc_auc) = self.eval_metrics(test_y, predicted_labels)

            scores = {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
            }
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            if roc_auc is not None:
                mlflow.log_metric("roc_auc", roc_auc)

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="LoanApprovalModel")
            else:
                mlflow.sklearn.log_model(model, "model")
