import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from mlProject.entity.config_entity import DataTransformationConfig

def clean_data(st):
    st = st.strip()
    return st
class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_spliting(self):
        # Load raw data
        df = pd.read_csv(self.config.data_path)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        # -------------------------
        # ✅ Data Preprocessing
        # -------------------------
        df.education = df.education.apply(clean_data)
        df.self_employed = df.self_employed.apply(clean_data)
        df.loan_status = df.loan_status.apply(clean_data)
        df['education'] = df['education'].astype('category')
        df['self_employed'] = df['self_employed'].astype('category')
        df['loan_status'] = df['loan_status'].astype('category')
         # Encode categorical variables
        df.self_employed = df.self_employed.replace(['No', 'Yes'],[0,1])
        df.loan_status = df.loan_status.replace(['Approved', 'Rejected'],[1,0])
        df['education'] = df['education'].replace(['Graduate', 'Not Graduate'],[1,0])
        # Feature engineering: create total_assets
        df['total_assets'] = (
            df['residential_assets_value'] +
            df['commercial_assets_value'] +
            df['luxury_assets_value'] +
            df['bank_asset_value']
        )

        # Drop unnecessary columns
        df = df.drop(columns=[
            'loan_id',
            'residential_assets_value',
            'commercial_assets_value',
            'luxury_assets_value',
            'bank_asset_value'
        ])

        # -------------------------
        # ✅ Train-test split
        # -------------------------
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(df, test_size=0.25, random_state=42)

        # Save to artifacts directory
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Encoded & engineered data, then split into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
        print(df.columns)
        print(df.head(4))
