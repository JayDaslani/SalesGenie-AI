import pandas as pd
import numpy as np
import json
import os
import pickle
from langchain_core.tools import tool

df = pd.read_csv('/Users/jaydasalani/Desktop/SalesGenie AI/data/train.csv', encoding='latin-1')

df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')

rfm = pd.read_csv('/Users/jaydasalani/Desktop/SalesGenie AI/data/customer_segments.csv')

with open('/Users/jaydasalani/Desktop/SalesGenie AI/data/model_info.json') as f:
    model_info = json.load(f)

with open('/Users/jaydasalani/Desktop/SalesGenie AI/data/best_model-2.pkl', 'rb') as f:
    forecast_model = pickle.load(f)


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
        monthly = pd.read_csv('/Users/jaydasalani/Desktop/SalesGenie AI/data/monthly_features.csv')

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
    result = ['Customer Segment Analysis : ']

    seg_count = rfm['Segment'].value_counts()

    for seg, count in seg_count.items():
        seg_data = rfm[rfm['Segment'] == seg]
        avg_monetary = seg_data['Monetary'].mean()
        total_rev = seg_data['Monetary'].sum()

        result.append(
            f"\n{seg}:"
            f"\n  Count: {count}"
            f"\n  Avg Revenue: "
            f"${avg_monetary:,.2f}"
            f"\n  Total Revenue: "
            f"${total_rev:,.2f}"
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


