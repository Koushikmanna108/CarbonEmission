## 🌍 Carbon Emission Estimator (Per Person)

This project estimates the carbon footprint of an individual based on lifestyle, transport, and energy usage.
It is built with Flask and provides a web interface for users to input their data and get predictions.

## 📌 Features

• Web app built using Flask

• Input lifestyle details via a form

• Predicts carbon emissions per person per month using ML

• Classifies results into Low, Medium, High categories

• Interactive HTML templates (form.html, home.html, result.html)

## 📂 Project Structure

```
carbon-emission-estimator/
│── notebook/                # Jupyter notebooks for experiments
│   └── Model Training.ipynb # Model training notebook
│── Data/                    # Dataset(s)
│   └── Carbon Emission.csv
│── src/                     # Source code
│   ├── components/          # Data ingestion, transformation, model training
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/            # Prediction pipeline & helpers
│   │   ├── predict_pipeline.py
│   │── exception.py         # For Exception Handling
│   │── logger.py            # Logging   
│   │── utils.py             # Basic Function
│── templates/               # HTML files (Flask views)
│   ├── form.html
│   ├── home.html
│   └── result.html
│── app.py                   # Flask app entry point
│── requirements.txt         # Dependencies
│── README.md                # Documentation
│── .gitignore
│── venv/                    # Virtual Environment
|── setup.py    

```

## ⚙️ Installation
### 1. Clone the repository
```
git clone https://github.com/Koushikmanna108/CarbonEmission.git

cd CarbonEmission
```
### 2. Create & activate environment (Windows + Conda)
```
conda create -p venv python=3.13 -y

conda activate venv/

```

### 3. Install dependencies
```
pip install -r requirements.txt
```

## 🚀 Usage
### Run the Flask app
```
python app.py

```
## 📊 Example Prediction

### Input (via form):

• Transportation: Car (20 km/day)

• Diet: Vegetarian

• Electricity: 150 kWh/month

• Waste: 2 kg/week

### Output:

• Estimated Carbon Footprint: 1.8 tons CO₂e/year

• Category: Low 🌱

## 📈 Model

• Algorithm: GradientBoosting Regressor (from training notebook)

• Dataset: Carbon Emission.csv

• Pipeline: Includes ingestion, transformation, and prediction


🤝 Contributing

Pull requests are welcome!

For major changes, please open an issue first to discuss.