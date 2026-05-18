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
    forecast = model.predict(future_calendar)
    
    results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(months).copy()
    results['ds'] = results['ds'].dt.strftime('%Y-%m') 
    
    return results.to_dict(orient="records")
