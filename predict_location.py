import joblib
import pandas as pd

model = joblib.load("rainfall_model.pkl")
encoders = joblib.load("encoders.pkl")

df = pd.read_excel("indian_weather.xlsx")
df = df.ffill()

sample = df.mean().to_frame().T
location = input("Enter station name: ")

if location in encoders["station_name"].classes_:
    sample["station_name"] = encoders["station_name"].transform([location])[0]
    pred = model.predict(sample)[0]
    print("Rain Tomorrow:", "Yes" if pred == 1 else "No")
else:
    print("Location not found")
