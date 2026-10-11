import pandas as pd
import numpy as np
import json
import os
import pickle
from langchain_core.tools import tool

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))               
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..")) 
DATA_DIR = os.getenv("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))


train_path = os.path.join(DATA_DIR, "train.csv")
if os.path.exists(train_path):
    df = pd.read_csv(train_path, encoding='latin-1')
    df.columns = df.columns.str.strip()
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, format='mixed')
else:
    df = pd.DataFrame()


rfm_path = os.path.join(DATA_DIR, "customer_segments.csv")
rfm = pd.read_csv(rfm_path) if os.path.exists(rfm_path) else pd.DataFrame()


info_path = os.path.join(DATA_DIR, "model_info.json")
model_info = {}
if os.path.exists(info_path):
    with open(info_path, "r") as f:
        model_info = json.load(f)


model_path = os.path.join(DATA_DIR, "best_model-2.pkl")
forecast_model = None
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        forecast_model = pickle.load(f)



def detect_columns(df: pd.DataFrame) -> dict:
    """
    Standardizes and detects analytical columns dynamically
    (date, sales, customer, order, category, region, sub_category).
    """
    detected = {
        'date': None,
        'sales': None,
        'customer': None,
        'order': None,
        'category': None,
        'region': None,
        'sub_category': None
    }
    
    if df is None or df.empty:
        return detected

    cols_lower = {str(c).lower().strip().replace(' ', '_'): c for c in df.columns}

    # Order Date
    for k in ['order_date', 'date', 'orderdate', 'transaction_date', 'sale_date', 'timestamp']:
        if k in cols_lower:
            detected['date'] = cols_lower[k]
            break

    # Sales / Revenue
    for k in ['sales', 'revenue', 'total_sales', 'amount', 'total_amount', 'price']:
        if k in cols_lower:
            detected['sales'] = cols_lower[k]
            break

    # Customer ID
    for k in ['customer_id', 'customer', 'customerid', 'client_id', 'user_id']:
        if k in cols_lower:
            detected['customer'] = cols_lower[k]
            break

    # Order ID
    for k in ['order_id', 'orderid', 'transaction_id', 'invoice_id', 'invoice']:
        if k in cols_lower:
            detected['order'] = cols_lower[k]
            break

    # Category
    for k in ['category', 'product_category', 'item_category']:
        if k in cols_lower:
            detected['category'] = cols_lower[k]
            break

    # Region
    for k in ['region', 'territory', 'zone', 'country', 'state']:
        if k in cols_lower:
            detected['region'] = cols_lower[k]
            break

    # Sub-Category
    for k in ['sub_category', 'sub-category', 'subcategory', 'product_sub_category']:
        if k in cols_lower:
            detected['sub_category'] = cols_lower[k]
            break

    return detected
        
@tool
def analyze_sales(query: str) -> str:
    """
    Analyzes Sales data. 
    Use this When you have question about Sales.
    Example: "best region", "top prodcut", "monthly trend"
    """

    query = query.lower()
    results = []

    total = df['Sales'].sum()
    results.append(f"Total : ${total:,.2f}")

    best_cat = df.groupby('Category')['Sales'].sum().idxmax()
    cat_sales = df.groupby('Category')['Sales'].sum().max()
    results.append(f"Best Category : {best_cat}"
                   f"(${cat_sales:,.2f})")
    
    best_reg = df.groupby('Region')['Sales'].sum().idxmax()
    reg_sales = df.groupby('Region')['Sales'].sum().max()
    results.append(f"Best Region : {best_reg}"
                   f"(${reg_sales:,.2f})")
    
    best_sub = df.groupby('Sub-Category')['Sales'].sum().idxmax()
    results.append(f"Best Sub Category : {best_sub}")

    yearly = df.groupby(df['Order Date'].dt.year)['Sales'].sum()
    for year, sales in yearly.items():
        results.append(f"Year  {year} : ${sales:,.2f}")

    return "\n".join(results)


@tool
def get_forcast(months: int = 3) -> str:
    """
    Predicts Future Sales.
    Use this when you need a future forecast.
    Input: How many months to predict.
    """

    try:
        monthly_path = os.path.join(DATA_DIR, "monthly_features.csv")
        monthly = pd.read_csv(monthly_path)
        features = model_info['selected_features']

        available = [f for f in features if f in monthly.columns]

        data = monthly[available + ['Total_Sales']].dropna()

        last_row = data.iloc[-1]
        last_sales = data['Total_Sales'].values

        predictions = []

        for i in range(months):
            next_month = int(last_row['Month'] + i) % 12 + 1
            next_year = int(last_row['Year']) + (int(last_row['Month']+i)//12) 
            next_feat = {}
            for f in available:
              if f == 'Year':
                  next_feat[f] = next_year
              elif f == 'Month':
                  next_feat[f] = next_year
              elif f == 'Quarter':
                  next_feat[f] = ((next_month-1)//3)+1
              elif f == 'Month_Sin':
                  next_feat[f] = np.sin(2*np.pi*next_month/12)
              elif f == 'Month_Cos':
                  next_feat[f] = np.cos(2*np.pi*next_month/12)
              elif f == 'Is_Holiday_Month':
                  next_feat[f] = int(next_month in [11, 12, 1])
              elif f == 'Is_Quarter_END':
                  next_feat[f] = int(next_month in [3, 6, 9, 12])
              elif f == 'Lag_1':
                  next_feat[f] = last_sales[-1]
              elif f == 'Lag_2':
                  next_feat[f] = last_sales[-2]
              elif f == 'Lag_3':
                  next_feat[f] = last_sales[-3]
              elif f == 'Rolling_3':
                  next_feat[f] = np.mean(last_sales[-3:])
        

            pred = forecast_model.predict(pd.DataFrame([next_feat]))[0]
            predictions.append({
               'month': next_month,
               'year': next_year,
               'prediction': pred

            })

        results = [f"Sales Forecast (Next {months} months):"]

        for p in predictions:
            results.append(
                    f"  {p['year']}-"
                    f"{p['month']:02d}: "
                    f"${p['prediction']:,.2f}"
            )
        return "\n".join(results)
    
    except Exception as e:
        return f"Forecast error : {str(e)}"
    


@tool
def get_customer_segments(query: str = 'all') -> str:
    """Identifies customer segments.
    Use this when you need customer information.
    Categories include : Champions, Loyal, At Risk and Lost.
    """
    global rfm
    
    if rfm is None or rfm.empty:
        return "Customer segmentation (RFM) data is currently not available."

    # Handle case-insensitive column names ('Segment' vs 'segment')
    seg_col = 'Segment' if 'Segment' in rfm.columns else ('segment' if 'segment' in rfm.columns else None)
    
    if not seg_col:
        return f"Segment column not found in RFM dataset. Available columns: {list(rfm.columns)}"

    result = ['Customer Segment Analysis : ']
    seg_count = rfm[seg_col].value_counts()

    monetary_col = 'Monetary' if 'Monetary' in rfm.columns else ('monetary' if 'monetary' in rfm.columns else None)

    for seg, count in seg_count.items():
        seg_data = rfm[rfm[seg_col] == seg]
        
        if monetary_col:
            avg_monetary = seg_data[monetary_col].mean()
            total_rev = seg_data[monetary_col].sum()
            result.append(
                f"\n{seg}:"
                f"\n  Count: {count}"
                f"\n  Avg Revenue: ${avg_monetary:,.2f}"
                f"\n  Total Revenue: ${total_rev:,.2f}"
            )
        else:
            result.append(
                f"\n{seg}:"
                f"\n  Count: {count}"
            )

    return "\n".join(result)


@tool
def get_quick_stats(metric: str) -> str:
    """
    Provides quick statistics.
    Use this when you need a specefic metric.
    Metrics : revenue, orders, customers, products, regions
    """

    metric = metric.lower()

    if 'revenue' in metric or 'sales' in metric:
        total = df['Sales'].sum()
        avg = df['Sales'].mean()
        max_sale = df['Sales'].max()

        return (
            f"Revenue Stats:\n"
            f"  Total: ${total:,.2f}\n"
            f"  Average Order: ${avg:,.2f}\n"
            f"  Max Order: ${max_sale:,.2f}"
        )
    
    elif 'order' in metric:
        total = df['Order ID'].nunique()
        return f"Total Orders : {total:,}"
    
    elif 'customer' in metric:
        total = df['Customer ID'].nunique()
        return f"Total Customers: {total:,}"
    
    elif 'product' in metric:
        total = df['Product ID'].nunique()
        return f"Total Products : {total:,}"
    
    elif 'region' in metric:
        regions = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
        result = ['Regional Sales : ']
        for reg, sales in regions.items():
            result.append(
                f"  {reg}: ${sales:,.2f}"
            )
        return "\n".join(result)
    
    else:
        return (
            "Availabel metrics : revenue, orders, customers, products, regions"
        )


