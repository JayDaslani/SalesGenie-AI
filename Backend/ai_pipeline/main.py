from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import json
import os
import traceback
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, '/Users/jaydasalani/Desktop/SalesGenie AI/data')

from agent import chat
from insights import generate_sales_insights, detect_anomalies, get_customer_recommendations, generate_executive_summary, category_deep_dive

app = FastAPI(
    title='SalesGenie AI API',
    description="AI Powered Sales Intelligence",
    version='1.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

data_store = {
    "df": None,
    "rfm": None,
    "monthly": None
}

def load_all_datasets():
    train_path = os.path.join(DATA_DIR, 'train.csv')
    if os.path.exists(train_path):
        df = pd.read_csv(train_path, encoding='latin-1')
        df.columns = df.columns.str.strip()
        df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, format='mixed')
        data_store['df'] = df
        print("train.csv loaded successfully!")
    else:
        print(f"File not found : {train_path}")

    rfm_path = os.path.join(DATA_DIR, 'customer_segments.csv')
    if os.path.exists(rfm_path):
        rfm = pd.read_csv(rfm_path)
        rfm.columns = rfm.columns.str.strip()
        data_store['rfm'] = rfm
        print("customer_segments.csv loaded successfully!")
    else:
        print(f"File not found : {rfm_path}")

    monthly_path = os.path.join(DATA_DIR, 'monthly_features.csv')
    if os.path.exists(monthly_path):
        data_store['monthly'] = pd.read_csv(monthly_path)
        print("monthly_features.csv loaded successfully!")
    else:
        print(f"File not found : {monthly_path}")


load_all_datasets()


class ChatRequest(BaseModel):
    message: str
    session_id: str = 'default'

class ForecastRequest(BaseModel):
    months: int = 3

class CategoryRequest(BaseModel):
    category: str

@app.get("/")
def root():
    return {
        "message": "SalesGenie AI API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "df_loaded": data_store['df'] is not None,
        "rfm_laoded": data_store['rfm'] is not None,
        "monthly_loaded": data_store['monthly'] is not None
    }

@app.get("/data/overview")
def get_data_overview():
    """Overview of Dataset"""
    if data_store['df'] is None:
        raise HTTPException(
            status_code=404,
            detail="Data not loaded"
        )
    
    df = data_store['df']

    return {
        "total_rows": len(df),
        "total_revenue": round(df['Sales'].sum(), 2),
        "total_orders": df['Order ID'].nunique(),
        "total_customer": df['Customer ID'].nunique(),
        "total_products": df['Product ID'].nunique(),
        "data_range": {
            "start": str(df['Order Date'].min().date()),
            "end": str(df['Order Date'].max().date())
        },
        "categories": df['Category'].unique().tolist(),
        "regions": df['Region'].unique().tolist()
    }

@app.get('/data/sales/monthly')
def get_monthly_sales():
    """Monthly sales data"""
    if data_store['df'] is None:
        raise HTTPException(
            status_code=404,
            detail="Data not loaded"
        )
    
    df = data_store['df']

    monthly = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
    monthly['Order Date'] = monthly['Order Date'].astype(str)

    return {
        "monthly_sales": monthly.to_dict(orient='records')
    }

@app.get("/data/sales/category")
def get_category_sales():
    """Category wise sales"""
    if data_store['df'] is None:
        raise HTTPException(
            status_code=404,
            detail="Data not loaded"
        )
    
    df = data_store['df']

    cat_sales = df.groupby('Category')['Sales'].sum().reset_index()

    return {
        "category_sales": cat_sales.to_dict(orient='records')
    }

@app.get('/data/sales/region')
def get_region_sales():
    """Region wise sales"""
    if data_store['df'] is None:
        raise HTTPException(
            status_code=404,
            detail='Data not loaded'
        )
    
    df = data_store['df']

    reg_sales = df.groupby('Region')['Sales'].sum().reset_index()

    return {
        "region_sales": reg_sales.to_dict(orient='records')
    }

@app.post('/ml/forecast')
def get_forecast(request: ForecastRequest):
    """Sales forecast"""

    try:
        model_path = os.path.join(DATA_DIR, 'best_model-2.pkl')
        info_path = os.path.join(DATA_DIR, 'model_info.json')

        if not os.path.exists(model_path) or not os.path.exists(info_path):
            raise HTTPException(
                status_code=404,
                detail="Model files (pkl/json) not found in data directory"
            )
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(info_path, 'r') as f:
            model_info = json.load(f)

        
        
        monthly = data_store['monthly']
        if monthly is None:
            raise HTTPException(
                status_code=404,
                detail="Monthly data not found"
            )
        
        features = model_info['selected_features']

        available = [f for f in features if f in monthly.columns]

        data = monthly[available + ['Total_Sales']].dropna()

        last_row = data.iloc[-1]
        last_sales = data['Total_Sales'].values

        predictions = []
        for i in range(request.months):
            next_month = int(last_row['Month']+i)%12+1
            next_year = int(last_row['Year']) + (int(last_row['Month']+i)//12)

            next_feat = {}
            for f in available:
                if f == 'Year':
                    next_feat[f] = next_year
                elif f == 'Month':
                    next_feat[f] = next_month
                elif f == 'Quarter':
                    next_feat[f] = ((next_month-1)//3)+1 
                elif f == "Month_Sin":
                    next_feat[f] = np.sin(2*np.pi*next_month/12)
                elif f == "Month_Cos":
                    next_feat[f] = np.cos(2*np.pi*next_month/12)
                elif f == "Is_Holiday_Month":
                    next_feat[f] = int(next_month in [11, 12, 1])
                elif f == "Is_Quarter_END":
                    next_feat[f] = int(next_month in [3,6,9,12])
                elif f == "Lag_1":
                    next_feat[f] = float(last_sales[-1])
                elif f == "Lag_2":
                    next_feat[f] = float(last_sales[-2])
                elif f == 'Lag_3':
                    next_feat[f] = float(last_sales[-3])
                elif f == 'Rolling_3':
                    next_feat[f] = float(np.mean(last_sales[-3:]))

            pred = float(model.predict(pd.DataFrame([next_feat]))[0])

            predictions.append({
                'month': next_month,
                'year': next_year,
                'predicted_sales': round(pred, 2)
            })

            last_sales = np.append(last_sales, pred)

        return {
            "forecast": predictions,
            "model": model_info['best_model_name'],
            "accuracy": f"{100 - model_info['best_mape']:.1f}%"
        }
    
    except Exception as e:
        print("Forecast Error Traceback:")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e)

        )
    

@app.get('/ml/segments')
def get_segments():
    """Customer segments"""

    if data_store['rfm'] is None:
        raise HTTPException(
            status_code=404,
            detail='RFM data is not found'
        )
    
    rfm = data_store['rfm']
    seg_summary = rfm.groupby('Segment').agg(
        count=('Customer ID', 'count'),
        avg_recency=('Recency', 'mean'),
        avg_frequency=('Frequency', 'mean'),
        avg_monetary=('Monetary', 'mean'),
        total_revenue=('Monetary', 'sum')
    ).round(2).reset_index()

    return {
        "segments": seg_summary.to_dict(orient='records')
    }

@app.post('/ai/chat')
def ai_chat(request: ChatRequest):
    """AI Agent chat"""

    try:
        response = chat(
            request.message,
            session_id=request.session_id
        )
        return {
            "response": response,
            "session_id": request.session_id
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
@app.get("/ai/insights")
def get_insights():
    """AI generated insights"""
    if data_store['df'] is None:
        raise HTTPException(
            status_code=404,
            detail='Data not loaded'
        )
    
    try:
        insights = generate_sales_insights(data_store['df'])
        return {'insights': insights}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e)
           
        )

@app.get('/ai/anomalies')
def get_anomalies():
    """Anomaly detection"""
    if data_store['df'] is None:
        raise HTTPException(
            status_code=404,
            detail='Data not loaded'
        )
    
    try:
        anomalies = detect_anomalies(data_store['df'])
        return {'anomalies': anomalies}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
@app.get('/ai/recommendations')
def get_recommendations():
    """Customer Recommendations"""
    if data_store['rfm'] is None:
        raise HTTPException(
            status_code = 404,
            detail='RFM data not loaded'
        )
    try:
        recs = get_customer_recommendations(data_store['rfm'])
        return {"recommendations": recs}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    

@app.post("/ai/category")
def analyze_category(request: CategoryRequest):
    """Category deep dive"""
    if data_store['df'] is None:
        raise HTTPException(
            status_code=404,
            detail='Data not loaded'
        )
    try:
        analysis = category_deep_dive(data_store['df'], request.category)
        return {'analysis': analysis}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
@app.get("/ai/summary")
def get_executive_summary():
    """Executive summary"""
    if data_store['df'] is None:
        raise HTTPException(
            status_code=404,
            detail='Data not loaded'
        )
    try:
        summary = generate_executive_summary(data_store['df'], data_store['rfm'], [])
        return {"summary": summary}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    






        


