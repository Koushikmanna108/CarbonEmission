from flask import Flask, request, render_template
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route("/form")
def form():
    return render_template("form.html")


@app.route('/predictdata',methods=["GET","POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template('form.html')
    else:
        data = CustomData(
            Body_Type=request.form.get('Body_Type'),
            Gender=request.form.get('Gender'),
            Diet=request.form.get('Diet'),
            How_Often_Shower=request.form.get('How_Often_Shower'),
            Heating_Energy_Source=request.form.get('Heating_Energy_Source'),
            Transport=request.form.get('Transport'),
            Social_Activity=request.form.get('Social_Activity'),
            Monthly_Grocery_Bill=float(request.form.get('Monthly_Grocery_Bill')),
            Frequency_of_Traveling_by_Air=request.form.get('Frequency_of_Traveling_by_Air'),
            Vehicle_Monthly_Distance_Km=float(request.form.get('Vehicle_Monthly_Distance_Km')),
            Waste_Bag_Size=request.form.get('Waste_Bag_Size'),
            Waste_Bag_Weekly_Count=float(request.form.get('Waste_Bag_Weekly_Count')),
            How_Long_TV_PC_Daily_Hour=float(request.form.get('How_Long_TV_PC_Daily_Hour')),
            How_Many_New_Clothes_Monthly=float(request.form.get('How_Many_New_Clothes_Monthly')),
            How_Long_Internet_Daily_Hour=float(request.form.get('How_Long_Internet_Daily_Hour')),
            Energy_Efficiency=request.form.get('Energy_Efficiency')
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        if results[0] < 0:
            return render_template("result.html",results=0)
        else:
            return render_template("result.html",results=results[0])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0",port=port)

