import pandas as pd
import os
from carpriceproject import logger
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib
from carpriceproject.components.data_transformation import DataTransformation
from carpriceproject.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        transformer = DataTransformation(config=self.config)
        train_data = transformer.preprocess_data(train_data)
        test_data = pd.read_csv(self.config.test_data_path)
        test_data = transformer.preprocess_data(test_data)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]



        # Initialize and train RandomForestRegressor
        model = RandomForestRegressor(random_state=0)
        model.fit(train_x, train_y)

        # Save the trained model
        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))
        print(f"Model trained and saved to {os.path.join(self.config.root_dir, self.config.model_name)}")
