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

from pathlib import Path

# Dynamic relative path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))

from tools import session_store, detect_columns, get_active_session, set_active_session_id
from agent import chat
from insights import (
    generate_sales_insights,
    detect_anomalies,
    get_customer_recommendations,
    generate_executive_summary,
    category_deep_dive
)

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
    default_session = session_store.get_session("default")
    data_store['df'] = default_session.df
    data_store['rfm'] = default_session.rfm
    data_store['monthly'] = default_session.monthly
    print("All default datasets loaded into session_store and data_store successfully!")

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

@app.post("/data/upload")
async def upload_dataset(file: UploadFile = File(...), session_id: str = "default"):
    """Upload custom dataset (CSV or Excel) for dynamic analysis in a session."""
    try:
        session = session_store.load_dataset(file.file, session_id=session_id, source_name=file.filename)
        df = session.df
        if session_id == "default":
            data_store['df'] = df
        return {
            "status": "success",
            "message": f"Successfully loaded {file.filename}",
            "session_id": session_id,
            "total_rows": len(df),
            "detected_columns": session.columns,
            "columns": list(df.columns),
            "preview": df.head(5).to_dict(orient="records")
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to process file: {str(e)}")

@app.get("/data/overview")
def get_data_overview(session_id: str = "default"):
    """Overview of Dataset with auto-detected columns"""
    session = session_store.get_session(session_id)
    df = session.df
    if df is None or df.empty:
        raise HTTPException(
            status_code=404,
            detail="Data not loaded"
        )
    
    cols = session.columns
    sales_col = cols.get('sales')
    order_col = cols.get('order_id')
    cust_col = cols.get('customer_id')
    prod_col = cols.get('product_id')
    date_col = cols.get('order_date')
    cat_col = cols.get('category')
    reg_col = cols.get('region')

    total_rev = round(float(df[sales_col].sum()), 2) if sales_col and sales_col in df.columns else 0.0
    total_orders = int(df[order_col].nunique()) if order_col and order_col in df.columns else len(df)
    total_cust = int(df[cust_col].nunique()) if cust_col and cust_col in df.columns else 0
    total_prod = int(df[prod_col].nunique()) if prod_col and prod_col in df.columns else 0

    data_range = {"start": "N/A", "end": "N/A"}
    if date_col and date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        valid_dates = df[date_col].dropna()
        if not valid_dates.empty:
            data_range = {
                "start": str(valid_dates.min().date()),
                "end": str(valid_dates.max().date())
            }

    categories = df[cat_col].dropna().unique().tolist() if cat_col and cat_col in df.columns else []
    regions = df[reg_col].dropna().unique().tolist() if reg_col and reg_col in df.columns else []

    return {
        "total_rows": len(df),
        "total_revenue": total_rev,
        "total_orders": total_orders,
        "total_customer": total_cust,
        "total_products": total_prod,
        "data_range": data_range,
        "categories": categories,
        "regions": regions,
        "detected_columns": cols
    }

@app.get('/data/sales/monthly')
def get_monthly_sales(session_id: str = "default"):
    """Monthly sales data using dynamic columns"""
    session = session_store.get_session(session_id)
    df = session.df
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Data not loaded")

    cols = session.columns
    sales_col = cols.get('sales')
    date_col = cols.get('order_date')

    if not sales_col or not date_col or not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        return {"monthly_sales": []}

    valid = df.dropna(subset=[date_col, sales_col])
    monthly = valid.groupby(valid[date_col].dt.to_period('M'))[sales_col].sum().reset_index()
    monthly.columns = ['Order Date', 'Sales']
    monthly['Order Date'] = monthly['Order Date'].astype(str)

    return {
        "monthly_sales": monthly.to_dict(orient='records')
    }

@app.get("/data/sales/category")
def get_category_sales(session_id: str = "default"):
    """Category wise sales using dynamic columns"""
    session = session_store.get_session(session_id)
    df = session.df
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Data not loaded")

    cols = session.columns
    sales_col = cols.get('sales')
    cat_col = cols.get('category')

    if not sales_col or not cat_col:
        return {"category_sales": []}

    cat_sales = df.groupby(cat_col)[sales_col].sum().reset_index()
    cat_sales.columns = ['Category', 'Sales']

    return {
        "category_sales": cat_sales.to_dict(orient='records')
    }

@app.get('/data/sales/region')
def get_region_sales(session_id: str = "default"):
    """Region wise sales using dynamic columns"""
    session = session_store.get_session(session_id)
    df = session.df
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail='Data not loaded')

    cols = session.columns
    sales_col = cols.get('sales')
    reg_col = cols.get('region')

    if not sales_col or not reg_col:
        return {"region_sales": []}

    reg_sales = df.groupby(reg_col)[sales_col].sum().reset_index()
    reg_sales.columns = ['Region', 'Sales']

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

            total_months = int(last_row['Month'])+i+1

            next_month = ((total_months - 1)%12)+1
            next_year = int(last_row['Year']) + (total_months-1)//12

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
            
            pred_df = pd.DataFrame([next_feat])
            pred = float(model.predict(pred_df)[0])
            
            avg_monthly = float(np.mean(last_sales[-6:]))
            if pred > avg_monthly * 3:
                pred = avg_monthly * 1.1
            elif pred < avg_monthly * 0.3:
                pred = avg_monthly * 0.9

            
            predictions.append({
                'month': next_month,
                'year': next_year,
                'predicted_sales': round(pred, 2),
                'period': f"{next_year}-{next_month:02d}"
            })

            last_sales = np.append(last_sales, pred)
            avg_actual = float(np.mean(data['Total_Sales'].values[-6:]))

        return {
            "forecast": predictions,
            "model": model_info['best_model_name'],
            "accuracy": f"{100 - model_info['best_mape']:.1f}%",
            "avg_monthly_actual": round(avg_actual, 2)
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
    
