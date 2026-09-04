from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import json
import os
import io
import traceback
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

from agent import chat
from insights import generate_sales_insights, detect_anomalies, get_customer_recommendations, generate_executive_summary, category_deep_dive
from tools import (
    detect_columns,
    set_dataframe,
    set_rfm,
    set_monthly,
    set_model,
    set_model,
    get_data_store
)

app = FastAPI(
    title='SalesGenie AI API',
    description='AI Powered Sales Intelligence',
    version='2.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

def load_default_datasets():
    """Load default dataset on startup"""
    train_path = os.path.join(DATA_DIR, 'train.csv')

    if os.path.exists(train_path):
        df = pd.read_csv(train_path, encoding='latin-1')
        df.columns = df.columns.str.strip()

        detected = detect_columns(df)

        if detected['data']:
            df[detected['data']] = pd.to_datetime(df[detected['data']], dayfirst=True, format='mixed')
        
        set_dataframe(df, detected)
        print("Default dataset loaded!")
        print(f"Columns detected: {detected}")

    rfm_path = os.path.join(DATA_DIR, 'customer_segments.csv')
    if os.path.exists(rfm_path):
        rfm = pd.read_csv(rfm_path)
        rfm.columns = rfm.columns.str.strip()
        set_rfm(rfm)
        print("RFM data loaded!")

    monthly_path = os.path.join(DATA_DIR, 'monthly_features.csv')
    if os.path.exists(monthly_path):
        monthly = pd.read_csv(monthly_path)
        set_monthly(monthly)
        print("Monthly data loaded!")

    model_path = os.path.join(DATA_DIR, 'best_model-2.pkl')
    info_path = os.path.join(DATA_DIR, 'model_info.json')

    if os.path.exists(model_path) and os.path.exists(info_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(info_path, 'r') as f:
            model_info = json.load(f)

        set_model(model, model_info)
        print("ML Model loaded!")

load_default_datasets()

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
        "message": "SalesGenie AI API v2.0",
        "status": "running",
        "dynamic": True
    }

@app.get("/health")
def health():
    store = get_data_store()
    return {
        "status": "healthy",
        "df_loaded": store['df'] is not None,
        "rfm_loaded": store['rfm'] is not None,
        "monthly_loaded": store['monthly'] is not None,
        "model_loaded": store['model'] is not None,
        "detected_columns": store.get('cols', {})
    }

@app.post("/upload/csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload any CSV file dynamically.
    Columns are auto-detected.
    """
    try:
        content = await file.read()
        df = pd.read_csv(
            io.BytesIO(content),
            encoding='latin-1'
        )
        df.columns = df.columns.str.strip()

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail='CSV file is empty!'
            )
        
        detected = detect_columns(df)

        if detected['date']:
            try:
                df[detected['date']] = pd.to_datetime(
                    df[detected['date']],
                    dayfirst=True,
                    format='mixed'
                )
            except Exception:
                pass

        
        set_dataframe(df, detected)

        set_rfm(None)
        set_monthly(None)

        return {
            "message": "CSV uploaded successfully!",
            "filename": file.filename,
            "rows": len(df),
            "columns": df.columns.tolist(),
            "detected_columns": detected,
            "sample_data": df.head(3).to_dict()
        }
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get('/data/columns')
def get_columns():
    """Get detected columns of loaded dataset"""
    store = get_data_store()

    if store['df'] is None:
        raise HTTPException(status_code=400, detail='No data loaded')
    
    df = store['df']
    cols = store.get('cols', {})

    return {
        "all_columns": df.columns.tolist(),
        "detected_columns": cols,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "shape": {
            "rows": len(df),
            "columns": len(df.columns)
        }
    }

@app.post("/data/columns/map")
def update_column_mapping(mapping: dict):
    """
    Manually update column mapping.
    If auto-detection is wrong, user can correct it.
    """

    store = get_data_store()

    if store['df'] is None:
        raise HTTPException(status_code=404, detail='No data loaded')
    
    current_cols = store.get('cols', {})
    current_cols.update(mapping)
    set_dataframe(store['df'], current_cols)

    return {
        "message": "Column mapping updated!",
        "updated_mapping": current_cols
    }

@app.get("/data/overview")
def get_data_overview():
    """Dynamic overview of any dataset"""
    store = get_data_store()

    if store['df'] is None:
        raise HTTPException(status_code=404, detail='Data not loaded. Upload a CSV first.')
    
    df = store['df']
    cols = store.get('cols', {})

    overview = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "detected_columns": cols
    }

    if cols.get('sales'):
        sc = cols['sales']
        overview["total_revenue"] = round(df[sc].sum(), 2)
        overview["avg_transaction"] = round(df[sc].mean(), 2)

    if cols.get("order"):
        oc = cols["order"]
        overview['total_orders'] = int(df[oc].nunique())

    if cols.get("customer"):
        cc = cols["customer"]
        overview['total_customers'] = int(df[cc].nunique())

    if cols.get("product"):
        pc = cols["product"]
        overview["total_products"] = int(df[pc].nunique())

    if cols.get("date"):
        dc = cols["date"]
        overview["date_range"] = {"start": str(df[dc].min()), "end": str(df[dc].max())}

    if cols.get("category"):
        catc = cols["category"]
        overview["categories"] = (df[catc].unique().tolist())

    if cols.get("region"):
        rc = cols["region"]
        overview["regions"] = (df[rc].unique().tolist())

    return overview

@app.get("/data/sales/monthly")
def get_monthly_sales():
    """Dynamic monthly sales"""
    store = get_data_store()

    if store['df'] is None:
        raise HTTPException(status_code=404, detail='Data not loaded')
    
    df = store["df"]
    cols = store.get("cols", {})
    date_col = cols.get("date")
    sales_col = cols.get("sales")

    if not date_col or not sales_col:
        raise HTTPException(status_code=400, detail='Date or Sales column not detected')
    
    monthly = df.groupby(df[date_col].dt.to_period('M'))[sales_col].sum().reset_index()
    monthly[date_col] = monthly[date_col].astype(str)

    return {
        "monthly_sales": monthly.rename(
            columns={
                date_col: "period",
                sales_col: "sales"
            }
        ).to_dict(orient='records')
    }


@app.get("/data/sales/category")
def get_category_sales():
    """Dynamic monthly sales"""
    store = get_data_store()

    if store['df'] is None:
        raise HTTPException(status_code=404, detail='Data not loaded')
    
    df = store['df']
    cols = store.get('cols', {})
    cat_col = cols.get("category")
    sales_col = cols.get("sales")

    if not cat_col or not sales_col:
        raise HTTPException(status_code=400, detail='Category or Sales column not detected')
    
    cat_sales = df.groupby(cat_col)[sales_col].sum().reset_index()

    return {
        "category_sales": cat_sales.rename(
            columns={
                cat_col: "category",
                sales_col: "sales"
            }
        ).to_dict(orient='records')
    }


@app.get("/data/sales/region")
def get_region_sales():
    """Dynamic region sales"""
    store = get_data_store()

    if store['df'] is None:
        raise HTTPException(status_code=404, detail='Data not loaded')
    
    df = store['df']
    cols = cols.get('cols', {})
    reg_col = cols.get("region")
    sales_col = cols.get("sales")

    if not reg_col or not sales_col:
        raise HTTPException(status_code=400, detail='Region or Sales column not detected')
    
    reg_sales = df.groupby(reg_col)[sales_col].sum().reset_index()

    return {
        "region_sales": reg_sales.rename(
            columns={
                reg_col: "region",
                sales_col: "sales"
            }
        ).to_dict(orient='records')
    }

@app.post("/ml/forecast")
def get_forecast(request: ForecastRequest):
    """Sales forecast using stored model"""
    store = get_data_store()
    model = store.get('model')
    model_info = store.get('model_info')
    monthly = store.get('monthly')

    if model is None:
        raise HTTPException(status_code=404, detail="ML Model not loaded")
    
    if monthly is None:
        raise HTTPException(status_code=404, detail="Monthly data not available")
    
    try:
        features = model_info['selected_features']
        available = [f for f in features if f in monthly.columns]
        data = monthly[available + ['Total_Sales']].dropna()

        last_row = data.iloc[-1]
        last_sales = data['Total_Sales'].values

        predictions = []

        for i in range(request.months):
            next_month = int(last_row['Monht'] + i)% 12 + 1
            next_year = int(last_row['Year']) + (int(last_row['Month'] + i)// 12)

            next_feat = {}
            for f in available:
                if f == 'Year':
                    next_feat[f] = next_year
                elif f == "Month":
                    next_feat[f] = next_month
                elif f == "Quarter":
                    next_feat[f] = ((next_month - 1)// 3)+1
                elif f == "Month_Sin":
                    next_feat[f] = np.sin(2*np.pi*next_month/12)
                elif f == "Month_Cos":
                    next_feat[f] = np.cos(2*np.pi*next_month/12)
                elif f == "Is_Holiday_Month":
                    next_feat[f] = int(next_month in [11, 12, 1])
                elif f == "Is_Quarter_END":
                    next_feat[f] = int(next_month in [3, 6, 9, 12])
                elif f == "Lag_1":
                    next_feat[f] = float(last_sales[-1])
                elif f == "Lag_2":
                    next_feat[f] = float(last_sales[-2])
                elif f == "Lag_3":
                    next_feat[f] = float(last_sales[-3])
                elif f == "Rolling_3":
                    next_feat[f] = float(np.mean(last_sales[-3:]))

            
            pred = float(model.predict(pd.DataFrame([next_feat]))[0])

            predictions.append({
                "month": next_month,
                "year": next_year,
                "predicted_sales": round(pred, 2)
            })
            last_sales = np.append(last_sales, pred)

        return {
            "forecast": predictions,
            "model": model_info.get('best_model_name', 'ML Model'),
            "accuracy": f"{100 - model_info.get('best_mape', 0):.1f}%"
        }
    
    except Exception as e:
        traceback().print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/ml/segments')
def get_segments():
    store = get_data_store()

    if store['rfm'] is None:
        raise HTTPException(status_code=404, detail='RFM data not found')
    
    rfm = store['rfm']
    seg_summary = rfm.groupby('segment').agg(
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
        raise HTTPException(status_code=500, detai=str(e))
    

@app.get('/ai/insights')
def get_insights():
    store = get_data_store()
    if store['df'] is None:
        raise HTTPException(status_code=404, detail='Data not loaded')
    try:
        insights = generate_sales_insights(store['df'])
        return {'insights': insights}
    
    except Exception as e:
        raise HTTPException (status_code=500, detail=str(e))
    

@app.get('/ai/anomalies')
def get_anomalies():
    store = get_data_store()

    if store['df'] is None:
        raise HTTPException(status_code=404, detail='Data not loaded')
    
    try:
        anomalies = detect_anomalies(store['df'])
        return {'anomalies': anomalies}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get('/ai/recommendations')
def get_recommendations():
    store = get_data_store()

    if store['rfm'] is None:
        raise HTTPException(status_code=404, detail='RFM data not loaded')
    
    try:
        recs = get_customer_recommendations(store['rfm'])
        return {"recommendations": recs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/category")
def analyze_category(request: CategoryRequest):
    store = get_data_store()
    
    if store['df'] is None:
        raise HTTPException(status_code=404, detail='Data not loaded')
    
    try:
        analysis = category_deep_dive(store['df'], request.category)
        return {'analysis': analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/ai/summary")
def get_executive_summary():
    store = get_data_store()

    if store['df'] is None:
        raise HTTPException(status_code=404, detail='Data not loaded')
    
    try:
        summary = get_executive_summary(store['df'], store['rfm'], [])
        return {"summary": summary}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))