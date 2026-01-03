import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_excel("indian_weather.xlsx")
df = df.ffill()

df["RainTomorrow"] = df["rainfall"].shift(-1).apply(lambda x: 1 if x > 0 else 0)

label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

drop_cols = [c for c in ["date_of_record"] if c in df.columns]
X = df.drop(["RainTomorrow"] + drop_cols, axis=1)
y = df["RainTomorrow"]

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

joblib.dump(model, "rainfall_model.pkl")
joblib.dump(label_encoders, "encoders.pkl")

print("Model trained and saved")
