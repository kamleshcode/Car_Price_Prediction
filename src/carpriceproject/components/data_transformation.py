import os
from carpriceproject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from carpriceproject.entity.config_entity import DataTransformationConfig
from sklearn.feature_extraction.text import CountVectorizer

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def preprocess_data(self, data):
        # Drop missing values
        data.dropna(inplace=True)
        
        # Drop duplicate rows
        data.drop_duplicates(inplace=True)
        
        # Extract brand name
        def get_brand_name(car_name):
            car_name = car_name.split(' ')[0]
            return car_name.strip()
        
        data['name'] = data['name'].apply(get_brand_name)
        
        # Clean numerical values
        def clean_data(value):
            value = value.split(' ')[0]
            value = value.strip()
            if value == '':
                value = 0
            return float(value)
        
        data['mileage'] = data['mileage'].apply(clean_data)
        data['max_power'] = data['max_power'].apply(clean_data)
        data['engine'] = data['engine'].apply(clean_data)
        
        # Replace categorical values with numerical codes
        brand_mapping = {
            'Maruti': 1, 'Skoda': 2, 'Honda': 3, 'Hyundai': 4, 'Toyota': 5,
            'Ford': 6, 'Renault': 7, 'Mahindra': 8, 'Tata': 9, 'Chevrolet': 10,
            'Datsun': 11, 'Jeep': 12, 'Mercedes-Benz': 13, 'Mitsubishi': 14,
            'Audi': 15, 'Volkswagen': 16, 'BMW': 17, 'Nissan': 18, 'Lexus': 19,
            'Jaguar': 20, 'Land': 21, 'MG': 22, 'Volvo': 23, 'Daewoo': 24,
            'Kia': 25, 'Fiat': 26, 'Force': 27, 'Ambassador': 28, 'Ashok': 29,
            'Isuzu': 30, 'Opel': 31
        }
        data['name'].replace(brand_mapping, inplace=True)
        
        data['transmission'].replace({'Manual': 1, 'Automatic': 2}, inplace=True)
        data['seller_type'].replace({'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3}, inplace=True)
        data['fuel'].replace({'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4}, inplace=True)
        data['owner'].replace({'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3,
                               'Fourth & Above Owner': 4, 'Test Drive Car': 5}, inplace=True)
        
        # Reset index
        data.reset_index(inplace=True, drop=True)
        
        return data
        
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up


    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
        
