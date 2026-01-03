import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_excel("indian_weather.xlsx")
df = df.ffill()

df["RainTomorrow"] = df["rainfall"].shift(-1).apply(lambda x: 1 if x > 0 else 0)

encoders = joblib.load("encoders.pkl")
for col, le in encoders.items():
    df[col] = le.transform(df[col])

X = df.drop(["RainTomorrow", "date_of_record"], axis=1)
y = df["RainTomorrow"]

model = joblib.load("rainfall_model.pkl")
y_pred = model.predict(X)

print(classification_report(y, y_pred))

cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

