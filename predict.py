import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("full_clean.csv")
print("Columns in clean file:", df.columns.tolist())

# We will predict PM2.5
target = 'PM2.5'
if target not in df.columns:
    # if PM2.5 not there, use first number column
    target = df.select_dtypes(include='number').columns[0]

print(f"Predicting: {target}")

# X = all other pollutant columns
# drop non-number columns
X = df.select_dtypes(include='number').drop(columns=[target], errors='ignore')
y = df[target]

# Remove rows where target is missing
valid = y.notna()
X = X[valid]
y = y[valid]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, pred):.2f}")
print(f"R2 Score: {r2_score(y_test, pred):.3f}")
print("Model trained on WHOLE INDIA data!")

# Save
import joblib
joblib.dump(model, "aqi_model.pkl")

pd.DataFrame({'Actual': y_test, 'Predicted': pred}).to_csv("predictions.csv", index=False)
print("Saved: aqi_model.pkl and predictions.csv")
print("YOU CAN NOW PREDICT ANY CITY!")