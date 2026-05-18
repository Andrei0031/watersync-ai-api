from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from prophet.serialize import model_from_json
import json

app = FastAPI()

# Enable CORS so your Namecheap website layout can fetch the JSON data securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained machine learning model payload
with open('watersync_prophet_model.json', 'r') as fin:
    model = model_from_json(json.load(fin))

@app.get("/api/forecast")
def generate_forecast(months: int = 6):
    # 1. Generate the timeline (6 months + 1 extra month to cover April seamlessly)
    future_calendar = model.make_future_dataframe(periods=months + 1, freq='M')
    forecast = model.predict(future_calendar)
    
    # 2. Grab the trailing rows needed to build the connected visualization timeline
    total_rows_needed = months + 1
    results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(total_rows_needed).copy()
    
    # 3. Format the dates into standard chart-ready YYYY-MM labels
    results['ds'] = results['ds'].dt.strftime('%Y-%m') 
    
    # 4. Enforce the zero-floor safety clamping for low seasonal valleys
    results['yhat'] = results['yhat'].apply(lambda x: max(0.0, x))
    results['yhat_lower'] = results['yhat_lower'].apply(lambda x: max(0.0, x))
    results['yhat_upper'] = results['yhat_upper'].apply(lambda x: max(0.0, x))
    
    return results.to_dict(orient="records")
