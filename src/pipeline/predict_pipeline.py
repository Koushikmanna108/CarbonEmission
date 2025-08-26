import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            model=load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
            Body_Type:str, Gender:str, Diet:str, How_Often_Shower:str,
       Heating_Energy_Source:str, Transport:str, Social_Activity:str,
       Monthly_Grocery_Bill:int, Frequency_of_Traveling_by_Air:str,
       Vehicle_Monthly_Distance_Km:int, Waste_Bag_Size:str,
       Waste_Bag_Weekly_Count:int, How_Long_TV_PC_Daily_Hour:int,
       How_Many_New_Clothes_Monthly:int, How_Long_Internet_Daily_Hour:int,
       Energy_Efficiency:str):
        self.Body_Type = Body_Type
        self.Gender = Gender
        self.Diet = Diet
        self.How_Often_Shower = How_Often_Shower
        self.Heating_Energy_Source = Heating_Energy_Source
        self.Transport = Transport
        self.Social_Activity = Social_Activity
        self.Monthly_Grocery_Bill = Monthly_Grocery_Bill
        self.Frequency_of_Traveling_by_Air = Frequency_of_Traveling_by_Air
        self.Vehicle_Monthly_Distance_Km = Vehicle_Monthly_Distance_Km
        self.Waste_Bag_Size = Waste_Bag_Size
        self.Waste_Bag_Weekly_Count = Waste_Bag_Weekly_Count
        self.How_Long_TV_PC_Daily_Hour = How_Long_TV_PC_Daily_Hour
        self.How_Many_New_Clothes_Monthly = How_Many_New_Clothes_Monthly
        self.How_Long_Internet_Daily_Hour = How_Long_Internet_Daily_Hour
        self.Energy_Efficiency = Energy_Efficiency



    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Body_Type": [self.Body_Type],
                "Gender": [self.Gender],
                "Diet": [self.Diet],
                "How_Often_Shower": [self.How_Often_Shower],
                "Heating_Energy_Source": [self.Heating_Energy_Source],
                "Transport": [self.Transport],
                "Social_Activity": [self.Social_Activity],
                "Monthly_Grocery_Bill": [self.Monthly_Grocery_Bill],
                "Frequency_of_Traveling_by_Air": [self.Frequency_of_Traveling_by_Air],
                "Vehicle_Monthly_Distance_Km": [self.Vehicle_Monthly_Distance_Km],
                "Waste_Bag_Size": [self.Waste_Bag_Size],
                "Waste_Bag_Weekly_Count": [self.Waste_Bag_Weekly_Count],
                "How_Long_TV_PC_Daily_Hour": [self.How_Long_TV_PC_Daily_Hour],
                "How_Many_New_Clothes_Monthly": [self.How_Many_New_Clothes_Monthly],
                "How_Long_Internet_Daily_Hour": [self.How_Long_Internet_Daily_Hour],
                "Energy_Efficiency": [self.Energy_Efficiency]                
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
