Rainfall Prediction for India using Machine Learning

This project predicts whether it will rain tomorrow at Indian weather stations
using historical weather data and machine learning. The goal is to build a
realistic, explainable, and internship-ready rainfall prediction system.


DATASET

The dataset used in this project contains daily historical weather observations
collected from multiple Indian weather stations.

Dataset details:
- Covers multiple locations across India
- Daily records for each station
- Large-scale dataset suitable for real-world prediction

Key features in the dataset include:
- date_of_record: Date of observation
- month: Month of the year
- season: Season (Winter, Summer, Monsoon, etc.)
- station_name: Weather station name
- state: Indian state
- air_pressure: Atmospheric pressure (hPa)
- elevation: Elevation of station (meters)
- latitude: Geographic latitude
- longitude: Geographic longitude
- rainfall: Rainfall amount in millimeters (mm)

Target Variable:
RainTomorrow (Yes or No)

RainTomorrow is derived based on next-day rainfall values, where rainfall
greater than a threshold indicates rain occurrence.

Dataset size:
Approximately 970,000 records across multiple Indian locations.

Note:
The dataset file is not uploaded to GitHub due to size constraints.
It can be obtained from publicly available Indian weather datasets
(e.g., Kaggle or government meteorological sources).


TECHNOLOGIES USED
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Joblib


PROJECT STRUCTURE

rainfall_prediction_india/
|
|-- train_model.py          Train the model once and save it
|-- evaluate_model.py       Generate confusion matrix and classification report
|-- visualize_data.py       Generate exploratory data analysis plots
|-- predict_location.py     Predict rainfall for a specific location
|
|-- requirements.txt
|-- README.md


ABOUT SAVED MODEL FILES (.pkl)

This project uses serialized files to store trained components:

- rainfall_model.pkl  
  Contains the trained Random Forest classification model.
  This file is generated automatically when train_model.py is executed.

- encoders.pkl  
  Stores LabelEncoder objects used to encode categorical features
  such as station name, state, month, and season.
  This ensures consistent encoding during prediction.

These files are not included in the GitHub repository because they are
large in size and can be regenerated locally by running the training script.


INSTALLATION

1. Clone or download the project folder
2. Install required libraries by running:
   pip install -r requirements.txt


MODEL BUILDING STEPS

1. Missing values handled using forward fill
2. Categorical features encoded using LabelEncoder
3. Datetime and leakage columns removed
4. Class imbalance handled using class_weight balanced
5. Model trained using Random Forest Classifier
6. Trained model and encoders saved using joblib


EXPLORATORY DATA ANALYSIS

The following visualizations were created:
- Rainfall distribution histogram
- Rain vs No Rain class distribution
- Rainfall vs Air Pressure scatter plot
- Confusion matrix for model evaluation

These plots help understand data imbalance and weather patterns.


MODEL PERFORMANCE

Accuracy approximately 86 to 87 percent
Balanced precision and recall for rain and no-rain classes
Performance is realistic with no data leakage


HOW TO RUN THE PROJECT

Step 1: Train the model (run once)
python train_model.py

This generates:
- rainfall_model.pkl
- encoders.pkl


Step 2: Evaluate the model
python evaluate_model.py

Outputs:
- Classification report
- Confusion matrix


Step 3: Generate visualizations
python visualize_data.py

Outputs:
- EDA plots for rainfall and weather relationships


Step 4: Predict rainfall for a location
python predict_location.py

Predicts whether it will rain tomorrow for a selected Indian station.


CONCLUSION

This project demonstrates a complete machine learning workflow including
data analysis, preprocessing, modeling, evaluation, and prediction using
real Indian weather data.
