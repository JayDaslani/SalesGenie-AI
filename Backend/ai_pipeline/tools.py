import pandas as pd
import numpy as np
import os
import json
import pickle
from langchain_core.tools import tool

_data_store = {
    "df": None,
    "rfm": None,
    "monthly": None,
    "model": None,
    "model_info": None,
    "cols": {
        "date": None,
        "sales": None,
        "category": None,
        "region": None,
        "customer": None,
        "order": None,
        "product": None,
        "subcategory": None
    }
}

def get_data_store():
    return _data_store

def set_dataframe(df: pd.DataFrame, detected_cols: dict):
    _data_store['df'] = df
    _data_store['cols'] = detected_cols

def set_rfm(rfm: pd.DataFrame):
    _data_store['rfm'] = rfm

def set_monthly(monthly: pd.DataFrame):
    _data_store['monthly'] = monthly

def set_model(model, model_info: dict):
    _data_store['model'] = model
    _data_store['model_info'] = model_info

def detect_columns(df: pd.DataFrame) -> dict:
    """Auto detect column types from any CSV"""
    cols = {
        "date": None,
        "sales": None,
        "category": None,
        "region": None,
        "customer": None,
        "order": None,
        "product": None,
        "subcategory": None
    }

    for col in df.columns:
        col_lower = col.lower().strip()

        if cols['date'] is None and any(x in col_lower for x in ['date', 'time', 'day', 'ordered', 'created']):
            cols['date'] = col
        
        elif cols['sales'] is None and any(
            x in col_lower for x in
            ['sales', 'revenue', 'amount', 'price',
             'total', 'income', 'value', 'cost']
        ):
            cols['sales'] = col

        elif cols['category'] is None and any(
            x in col_lower for x in
            ['category', 'type', 'dept',
             'department', 'group']
        ):
            cols['category'] = col

        elif cols['subcategory'] is None and any(
            x in col_lower for x in
            ['subcategory', 'sub-category', 
             'sub_category', 'subcat']
        ):
            cols['subcategory'] = col

        elif cols['region'] is None and any(
            x in col_lower for x in
            ['region', 'city', 'state',
             'location', 'area', 'zone',
             'territory', 'country']
        ):
            cols['region'] = col

        elif cols['customer'] is None and any(
            x in col_lower for x in
            ['customer', 'client', 
             'user', 'buyer', 'member']
        ):
            cols['customer'] = col

        elif cols['order'] is None and any(
            x in col_lower for x in
            ['order', 'transaction',
             'invoice', 'receipt']
        ):
            cols['order'] = col

        elif cols['product'] is None and any(
            x in col_lower for x in
            ['prodcut', 'item', 'sku', 'goods', 'service']
        ):
            cols['product'] = col

    return cols


@tool
def analyze_sales(query: str) -> str:
    """
    Analyzes sales data dynamically.
    Use when asked about sales, revenue,
    categories, regions, or trends.
    Example: 'best region',
             'top category',
             'monthly trend'
    """

    df = _data_store['df']
    cols = _data_store['cols']

    if df is None:
        return "No data loaded. Please upload a CSV file first."
    
    sales_col = cols.get("sales")
    category_col = cols.get("category")
    region_col = cols.get("region")
    date_col = cols.get("date")
    order_col = cols.get("order")
    customer_col = cols.get("customer")
    product_col = cols.get("product")
    subcategory_col = cols.get("subcategory")

    results = []
    query = query.lower()

    if sales_col:
        total = df[sales_col].sum()
        results.append(f"Total Revenue: ${total:,.2f}")

        avg = df[sales_col].mean()
        results.append(f"Average per row: ${avg:,.2f}")

    if category_col and sales_col:
        best_cat = df.groupby(category_col)[sales_col].sum().idxmax()
        cat_sales = df.groupby(category_col)[sales_col].sum().max()
        results.append(
            f"Best Category: {best_cat} "
            f"(${cat_sales:,.2f})"
        )

    if subcategory_col and sales_col:
        best_sub = df.groupby(subcategory_col)[sales_col].sum().idxmax()
        results.append(
            f"Best Sub-Category: {best_sub}"
        )

    if region_col and sales_col:
        best_reg = df.groupby(region_col)[sales_col].sum().idxmax()
        reg_sales = df.groupby(region_col)[sales_col].sum().max()
        results.append(
            f"Best Region: {best_reg} "
            f"(${reg_sales:,.2f})"
        )

    if date_col and sales_col:
        yearly = df.groupby(df[date_col].dt.year)[sales_col].sum()
        for year, sales in yearly.items():
            results.append(
                f"Year {year}: ${sales:,.2f}"
            )

    if order_col:
        total_orders = df[order_col].nunique()
        results.append(
            f"Total Orders: {total_orders:,}"
        )

    if customer_col:
        total_customers = df[customer_col].nunique()
        results.append(
            f"Total Customers: {total_customers:,}"
        )

    if not results:
        return "Could not analyze - no matching columns found."
    
    return "\n".join(results)

@tool
def get_forecast(months: int=3) -> str:
    """
    Predict future sales.
    Use when asked about future predictions or forecasts.
    Input: number of months to forecast.
    """

    model = _data_store['model']
    model_info = _data_store['model_info']
    monthly = _data_store['monthly']

    if model is None or model_info is None:
        return (
            "Forecast model not available. "
            "Please upload a CSV with sales "
            "data to generate forecasts."
        )
    
    if monthly is None:
        return "Monthly data not available for forecasting."
    
    try:
        features = model_info['selected_features']
        available = [
            f for f in features
            if f in monthly.columns
        ]
        data = monthly[available + ['Total_Sales']].dropna()

        last_row = data.iloc[-1]
        last_sales = data['Total_Sales'].values

        predictions = []

        for i in range(months):
            next_month = int(last_row['Month'] + i) % 12 + 1
            next_year = int(last_row['Year']) + (int(last_row['Month'] + i)//12)

            next_feat = {}
            for f in available:
                if f == 'Year':
                    next_feat[f] = next_year
                elif f == 'Month':
                    next_feat[f] = next_month
                elif f == 'Quarter':
                    next_feat[f] = ((next_month - 1)//3)+1
                elif f == 'Month_Sin':
                    next_feat[f] = np.sin(2*np.pi*next_month/12)
                elif f == 'Month_Cos':
                    next_feat[f] = np.cos(2*np.pi*next_month/12)
                elif f == 'Is_Holiday_Month':
                    next_feat[f] = int(next_month in [11, 12, 1])
                elif f == 'Is_Quarter_END':
                    next_feat[f] = int(next_month in [3, 6, 9, 12])
                elif f == 'Lag_1':
                    next_feat[f] = float(last_sales[-1])
                elif f == 'Lag_2':
                    next_feat[f] = float(last_sales[-2])
                elif f == 'Lag_3':
                    next_feat[f] = float(last_sales[-3])
                elif f == 'Rolling_3':
                    next_feat[f] = float(np.mean(last_sales[-3:]))
            

        
            pred = float(model.predict(pd.DataFrame([next_feat]))[0])

            predictions.append({
                'month': next_month,
                'year': next_year,
                'predictions': pred
            })

            last_sales = np.append(last_sales, pred)

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
    """
    Identifies customer segments.
    Use when asked about customers, segments, or retention.
    Categories: Champions, Loyal, At Risk, Lost.
    """

    rfm = _data_store["rfm"]

    if rfm is None:
        return (
            "Customer segment data not available. "
            "RFM analysis requires historical "
            "customer purchase data."
        )
    
    result = ['Customer Segment Analysis:']
    seg_count = rfm['Segment'].value_counts()

    for seg, count in seg_count.items():
        seg_data = rfm[rfm['Segment'] == seg]
        avg_monetary = seg_data['Monetary'].mean()
        total_rev = seg_data['Monetary'].sum()
        result.append(
            f"\n{seg}:"
            f"\n  Count: {count}"
            f"\n  Avg Revenue: ${avg_monetary:,.2f}"
            f"\n  Total Revenue: ${total_rev:,.2f}"
        )

    return "\n".join(result)


@tool
def get_quick_stats(metric: str) -> str:
    """
    Provides quick statistics.
    Use for specific metrics.
    Metrics: revenue, orders, customers, products, regions, categories
    """

    df = _data_store['df']
    cols = _data_store['cols']

    if df is None:
        return "No data loaded. Please upload a CSV file."
    
    metric = metric.lower()
    sales_col = cols.get('sales')
    order_col = cols.get('order')
    customer_col = cols.get('customer')
    product_col = cols.get('product')
    region_col = cols.get('region')
    category_col = cols.get('category')

    if 'revenue' in metric or 'sales' in metric:
        if sales_col:
            total = df[sales_col].sum()
            avg = df[sales_col].mean()
            max_sale = df[sales_col].max()
            return (
                f"Revenue Stats:\n"
                f"  Total: ${total:,.2f}\n"
                f"  Average: ${avg:,.2f}\n"
                f"  Max: ${max_sale:,.2f}"
            )
        
    elif 'order' in metric:
        if order_col:
            total = df[order_col].nunique()
            return f"Total Orders: {total:,}"
        
    elif 'customer' in metric:
        if customer_col:
            total = df[customer_col].nunique()
            return f"Total Customers: {total:,}"
        
    elif 'product' in metric:
        if product_col:
            total = df[product_col].nunique()
            return f"Total Products: {total:,}"
        
    elif 'region' in metric:
        if region_col and sales_col:
            regions = df.groupby(region_col)[sales_col].sum().sort_values(ascending=False)
            result = ['Regional Sales:']
            for reg, sales in regions.items():
                result.append(f"  {reg}: ${sales:,.2f}")

            return "\n".join(result)
        
    elif 'category' in metric:
        if category_col and sales_col:
            cats = df.groupby(category_col)[sales_col].sum().sort_values(ascending=False)
            result = ['Category Sales:']
            for cat, sales in cat.items():
                result.append(f"  {cat}: ${sales:,.2f}")
            
            return "\n".join(result)
        
    return (
        "Available metrics: revenue, orders, "
        "customers, products, regions, categories"
    )




    