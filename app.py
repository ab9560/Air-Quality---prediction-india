from flask import Flask, request, render_template
import pandas as pd
import pickle
import requests
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load and train model once when app starts
df = pd.read_csv("full_clean.csv")
target = 'PM2.5'
X = df.select_dtypes(include='number').drop(columns=[target], errors='ignore')
y = df[target]
valid = y.notna()
X = X[valid]
y = y[valid]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save feature order
FEATURES = X.columns.tolist()
print("Model trained on:", FEATURES)

API_KEY = "89893e54d2d3e297184035ba0a7d596f"

def get_live_pollution(city):
    try:
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
        geo = requests.get(geo_url, timeout=10).json()
        if not geo: return None
        lat, lon = geo[0]['lat'], geo[0]['lon']

        poll_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        data = requests.get(poll_url, timeout=10).json()
        return data['list'][0]['components']
    except:
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_by_city', methods=['POST'])
def predict_by_city():
    city = request.form.get('city_name')
    live = get_live_pollution(city)
    if not live:
        return f"<h2>City '{city}' not found or API not active yet (wait 2 hours)</h2><a href='/'>Go Back</a>"

    # Map live data to your model features
    # Your model features are like PM10, NO2, SO2 etc.
    input_data = {}
    for f in FEATURES:
        # try to match: PM2.5 -> pm2_5, PM10 -> pm10
        key = f.lower().replace('.','').replace('2.5','2_5')
        # common mapping
        if 'pm2' in f.lower(): input_data[f] = live.get('pm2_5', 0)
        elif 'pm10' in f.lower(): input_data[f] = live.get('pm10', 0)
        elif 'no2' in f.lower(): input_data[f] = live.get('no2', 0)
        elif 'so2' in f.lower(): input_data[f] = live.get('so2', 0)
        elif 'co' in f.lower(): input_data[f] = live.get('co', 0)
        elif 'o3' in f.lower() or 'ozone' in f.lower(): input_data[f] = live.get('o3', 0)
        else: input_data[f] = 0

    df_input = pd.DataFrame([input_data])[FEATURES]
    pred = model.predict(df_input)[0]

    return f"""
    <h2>Live AQI for {city}</h2>
    <p>PM2.5: {live['pm2_5']} | PM10: {live['pm10']} | NO2: {live['no2']}</p>
    <h1>Predicted PM2.5: {pred:.2f}</h1>
    <p>Model Features Used: {FEATURES}</p>
    <a href='/'>Check Another City</a>
    """

if __name__ == '__main__':
    app.run(debug=True)
