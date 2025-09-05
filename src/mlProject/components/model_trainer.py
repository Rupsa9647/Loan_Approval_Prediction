import pandas as pd
import os
from mlProject import logger
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from mlProject.entity.config_entity import ModelTrainerConfig
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(train_x)
        X_test_scaled = scaler.transform(test_x)

        model = LogisticRegression()
        model.fit(X_train_scaled, train_y)

        print("Validation Accuracy:", model.score(X_test_scaled, test_y))

        # Save model
        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))

        # ✅ Save scaler
        joblib.dump(scaler, os.path.join(self.config.root_dir, "scaler.pkl"))
