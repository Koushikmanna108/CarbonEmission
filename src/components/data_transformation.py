import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['Monthly_Grocery_Bill', 'Vehicle_Monthly_Distance_Km',
                                'Waste_Bag_Weekly_Count', 'How_Long_TV_PC_Daily_Hour',
                                'How_Many_New_Clothes_Monthly', 'How_Long_Internet_Daily_Hour']
            categorical_columns_ordinal = ["Body_Type","How_Often_Shower","Social_Activity","Frequency_of_Traveling_by_Air","Waste_Bag_Size"]
            categorical_columns = ["Gender","Diet","Heating_Energy_Source","Transport","Energy_Efficiency"]
            categories_order = [
                ['underweight', 'normal','overweight', 'obese' ],
                ['less frequently', 'daily', 'twice a day',  'more frequently'],
                ['never', 'sometimes', 'often'],
                ['never', 'rarely', 'frequently', 'very frequently'],
                ['small', 'medium', 'large', 'extra large']
            ]
            numerical_pipeline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy='median')),
                    ("Scaler",StandardScaler())
                ]
            )
            categorical_ordinal_pipeline = Pipeline(
                steps = [
                    ("Imputer",SimpleImputer(strategy='most_frequent')),
                    ("OrdinalEncoder",OrdinalEncoder(categories=categories_order)),
                    ("Scaler",StandardScaler(with_mean=False))
                ]
            )
            categorical_one_pipeline = Pipeline(
                steps = [
                    ("Imputer",SimpleImputer(strategy='most_frequent')),
                    ("OneHotEncoder",OneHotEncoder(drop='first')),
                    ("Scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical columns scaling completed")

            logging.info("Categorical ordinal columns encoding completed")
            logging.info("Categorical one columns encoding completed")
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",numerical_pipeline,numerical_columns),
                    ("cat_ord_pipeline",categorical_ordinal_pipeline,categorical_columns_ordinal),
                    ("cat_one_pipeline",categorical_one_pipeline,categorical_columns)
                ]
            )

            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessor object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "CarbonEmission"

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframe")

            input_feature_train_arr =  preprocessing_obj.fit_transform(input_feature_train_df)
            
            input_feature_test_arr =  preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            logging.info("Saved Preprocessing Object")
            save_object (
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)        