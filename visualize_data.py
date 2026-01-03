import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("indian_weather.xlsx")

# Rainfall distribution
plt.hist(df["rainfall"], bins=50)
plt.title("Rainfall Distribution")
plt.xlabel("Rainfall mm")
plt.ylabel("Frequency")
plt.show()

# Rain vs No Rain
df["RainTomorrow"] = df["rainfall"].shift(-1).apply(lambda x: 1 if x > 0 else 0)
df["RainTomorrow"].value_counts().plot(kind="bar")
plt.xticks([0,1], ["No Rain", "Rain"], rotation=0)
plt.title("Rain vs No Rain")
plt.show()

# Rainfall vs Pressure
plt.scatter(df["air_pressure"], df["rainfall"], alpha=0.3)
plt.xlabel("Air Pressure")
plt.ylabel("Rainfall")
plt.title("Rainfall vs Air Pressure")
plt.show()
