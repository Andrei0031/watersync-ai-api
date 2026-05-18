from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from prophet.serialize import model_from_json
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open('watersync_prophet_model.json', 'r') as fin:
    model = model_from_json(json.load(fin))

@app.get("/api/forecast")
def generate_forecast(months: int = 6):
    future_calendar = model.make_future_dataframe(periods=months, freq='M')
    
    # Force a minimum cap (floor) of 0 so the model never outputs negative revenue
    future_calendar['floor'] = 0
    
    forecast = model.predict(future_calendar)
    
    results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(months).copy()
    results['ds'] = results['ds'].dt.strftime('%Y-%m') 
    
    # Safety Check: If any calculation still slips below zero, clamp it manually to 0
    results['yhat'] = results['yhat'].clip(lower=0)
    results['yhat_lower'] = results['yhat_lower'].clip(lower=0)
    results['yhat_upper'] = results['yhat_upper'].clip(lower=0)
    
    return results.to_dict(orient="records")
