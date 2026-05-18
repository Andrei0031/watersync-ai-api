@app.get("/api/forecast")
def generate_forecast(months: int = 6):
    # 1. Generate the full timeline (history + future periods)
    future_calendar = model.make_future_dataframe(periods=months, freq='M')
    forecast = model.predict(future_calendar)
    
    # 2. Grab the historical data rows and the future data rows separately
    # This ensures we get exactly the last real month to anchor our chart line
    history_count = len(model.history)
    
    # We take the last 1 month of history + the 6 months of future data (7 rows total)
    total_rows_needed = months + 1
    results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(total_rows_needed).copy()
    
    # 3. Clean up the dates to standard YYYY-MM formatting
    results['ds'] = results['ds'].dt.strftime('%Y-%m') 
    
    # 4. Apply our safety logic to ensure zero-clamping for valleys
    results['yhat'] = results['yhat'].apply(lambda x: max(0.0, x))
    results['yhat_lower'] = results['yhat_lower'].apply(lambda x: max(0.0, x))
    results['yhat_upper'] = results['yhat_upper'].apply(lambda x: max(0.0, x))
    
    # Return exactly the 7 records (1 anchor month + 6 forecast months)
    return results.to_dict(orient="records")
