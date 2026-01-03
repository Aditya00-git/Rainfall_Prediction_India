# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# df = pd.read_excel("indian_weather.xlsx")

# print("Dataset shape:", df.shape)
# print(df.head())

# df = df.ffill()

# if "RainTomorrow" not in df.columns:
#     df["RainTomorrow"] = df["rainfall"].shift(-1)
# df["RainTomorrow"] = df["RainTomorrow"].apply(
#     lambda x: 1 if x > 0 else 0
# )

# label_encoders = {}

# for col in df.select_dtypes(include="object").columns:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#     label_encoders[col] = le

# drop_cols = []

# for col in ["Date", "date_of_record", "RISK_MM", "RainToday"]:
#     if col in df.columns:
#         drop_cols.append(col)

# X = df.drop(["RainTomorrow"] + drop_cols, axis=1)

# y = df["RainTomorrow"]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# model = RandomForestClassifier(
#     n_estimators=200,
#     random_state=42,
#     class_weight="balanced"
# )

# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# print("\nAccuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# cm = confusion_matrix(y_test, y_pred)

# plt.figure(figsize=(6, 4))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Rainfall Prediction - India")
# plt.show()

# if "Location" in label_encoders:
#     loc_encoder = label_encoders["Location"]

#     print("\nAvailable Locations Example:")
#     print(loc_encoder.classes_[:10])

#     user_location = input("\nEnter Indian Location: ")

#     if user_location in loc_encoder.classes_:
#         loc_val = loc_encoder.transform([user_location])[0]

#         sample = X.mean().to_frame().T
#         sample["Location"] = loc_val

#         prediction = model.predict(sample)[0]
#         print("\nRain Tomorrow Prediction:",
#               "Yes" if prediction == 1 else "No")
#     else:
#         print("Location not found in dataset")
