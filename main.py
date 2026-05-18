@app.get("/api/forecast")
def generate_forecast(months: int = 6):
    # 1. We tell Prophet to generate data (6 months + 1 extra month to cover April)
    # This forces the model to calculate a prediction for the current month (April)
    future_calendar = model.make_future_dataframe(periods=months + 1, freq='M')
    forecast = model.predict(future_calendar)
    
    # 2. Grab the tail records (we want the current month + the 6 future months = 7 rows)
    total_rows_needed = months + 1
    results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(total_rows_needed).copy()
    
    # 3. Format dates to match your dashboard layout (YYYY-MM)
    results['ds'] = results['ds'].dt.strftime('%Y-%m') 
    
    # 4. Enforce our safety clamping so numbers never drop below zero
    results['yhat'] = results['yhat'].apply(lambda x: max(0.0, x))
    results['yhat_lower'] = results['yhat_lower'].apply(lambda x: max(0.0, x))
    results['yhat_upper'] = results['yhat_upper'].apply(lambda x: max(0.0, x))
    
    return results.to_dict(orient="records")
