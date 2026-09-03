import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from pathlib import Path
from dotenv import load_dotenv
from tools import detect_columns

load_dotenv()

llm = ChatGroq(
    model='openai/gpt-oss-20b',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0.3
)

parser = StrOutputParser()

def generate_sales_insights(df: pd.DataFrame) -> str:
    """Generates automated insights from sales data."""
    cols = detect_columns(df)
    sales_col = cols.get('sales', 'Sales')
    date_col = cols.get('order_date', 'Order Date')
    cat_col = cols.get('category', 'Category')
    reg_col = cols.get('region', 'Region')

    total_sales = float(df[sales_col].sum()) if sales_col in df.columns else 0.0
    avg_order = float(df[sales_col].mean()) if sales_col in df.columns else 0.0

    yearly_data = "Yearly data not available"
    growth = 0.0
    if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        valid_dates = df.dropna(subset=[date_col, sales_col])
        if not valid_dates.empty:
            yearly = valid_dates.groupby(valid_dates[date_col].dt.year)[sales_col].sum()
            yearly_data = yearly.to_string()
            if len(yearly) > 1 and yearly.iloc[0] > 0:
                growth = ((yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0] * 100)

    best_category = "N/A"
    if cat_col in df.columns:
        cat_group = df.groupby(cat_col)[sales_col].sum()
        if not cat_group.empty:
            best_category = str(cat_group.idxmax())

    best_region = "N/A"
    worst_region = "N/A"
    if reg_col in df.columns:
        reg_group = df.groupby(reg_col)[sales_col].sum()
        if not reg_group.empty:
            best_region = str(reg_group.idxmax())
            worst_region = str(reg_group.idxmin())

    best_month_str = "N/A"
    worst_month_str = "N/A"
    month_names = {
        1:'Jan', 2:'Feb', 3:'Mar',
        4:'Apr', 5:'May', 6:'Jun',
        7:'Jul', 8:'Aug', 9:'Sep',
        10:'Oct', 11:'Nov', 12:'Dec'
    }
    if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        valid_dates = df.dropna(subset=[date_col, sales_col])
        if not valid_dates.empty:
            monthly = valid_dates.groupby(valid_dates[date_col].dt.month)[sales_col].mean()
            if not monthly.empty:
                best_month = monthly.idxmax()
                worst_month = monthly.idxmin()
                best_month_str = month_names.get(best_month, str(best_month))
                worst_month_str = month_names.get(worst_month, str(worst_month))

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior business analyst.
         Generate a clear, actionable insights from sales data.
         Keep it concise 5-6 bullet points.
         Focus on business impact."""),
         ("human", """Analyze this sales data and provide insights:
          
          Total Revenue : ${total_sales:,.2f}
          Average Order Value : ${avg_order:,.2f}
          Revenue Growth (2015 - 2018) : {growth:.1f}%
          Best Category : {best_category}
          Best Region : {best_region}
          Worst Region : {worst_region}
          Peak month : {best_month}
          Slowest month : {worst_month}
          
          Yearly Revenue : {yearly_data}
          
          Provide 5-6 key business insights.
          """.format(
              total_sales=total_sales,
              avg_order=avg_order,
              growth=growth,
              best_category=best_category,
              best_region=best_region,
              worst_region=worst_region,
              best_month=best_month_str,
              worst_month=worst_month_str,
              yearly_data=yearly_data
          ))
    ])

    chain = prompt | llm | parser
    return chain.invoke({})


def detect_anomalies(df: pd.DataFrame) -> str:
    """Detects Unusual patterns using dynamic columns"""
    cols = detect_columns(df)
    sales_col = cols.get('sales', 'Sales')
    date_col = cols.get('order_date', 'Order Date')

    if date_col not in df.columns or sales_col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        return "Insufficient time series or sales data to detect anomalies."

    valid = df.dropna(subset=[date_col, sales_col])
    monthly_sales = valid.groupby(valid[date_col].dt.to_period('M'))[sales_col].sum()

    mean = float(monthly_sales.mean())
    std = float(monthly_sales.std()) if len(monthly_sales) > 1 else 0.0

    anomalies = monthly_sales[
        (monthly_sales > mean + 2*std) |
        (monthly_sales < mean - 2*std)
    ]

    cat_monthly = df.groupby([df['Order Date'].dt.to_period('M'),'Category'])['Sales'].sum().reset_index()

    high_months = monthly_sales[monthly_sales > mean + std]
    low_months = monthly_sales[monthly_sales < mean - std]

    anomaly_data = f"""
    Monthly Sales Statistics : 
    Mean: ${mean:,.2f}
    Std: ${mean:,.2f}

    Anomalous Months (>2 std): {anomalies.to_string() if len(anomalies) > 0 else 'Not found'}

    High Performance Months (>1 std) : {high_months.head(5).to_string()}
    Low Performance Months (<1 std) : {low_months.head(5).to_string()}

    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a data analyst specializing in anomaly detection.
         Explain anomalies in business terms.
         Be specific and actionable."""),
         ("human", """
         Analyze these sales anomalies and explain : {data}
          
          Provide :
          1. What anomalies exist
          2. Possible business reasons
          3. What action to take
          """)
    ])

    chain = prompt | llm | parser
    return chain.invoke({'data': anomaly_data})

def get_customer_recommendations(rfm: pd.DataFrame) -> str:
    """Customer segment based recommendations"""

    seg_summary = rfm.groupby('Segment').agg(
        Count=('Customer ID', 'count'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean'),
        Total_Revenue=('Monetary', 'sum')
    ).round(2)

    prompt = ChatPromptTemplate.from_messages([
        ('system', """You are a CRM expert.
         Give specefic, actionable recommendations for each customer segment.
         Focus on retention and growth."""),
         ('human', """
         Based on customer segementation analysis : 
          {segment_data}
          
          Provide specific recommendations for:
          1. How to retain each segment
          2. Marketing strategies
          3. Expected revenue impact
          4. Priority actions
          
          Be specefic and business-focused.
          """)
    ])

    chain = prompt | llm | parser
    return chain.invoke({"segment_data": seg_summary.to_string()})

def generate_executive_summary(df: pd.DataFrame, rfm: pd.DataFrame, forecast: list) -> str:
    """Complete executive summary using dynamic columns"""
    cols = detect_columns(df)
    sales_col = cols.get('sales', 'Sales')
    cust_col = cols.get('customer_id', 'Customer ID')
    order_col = cols.get('order_id', 'Order ID')

    total_revenue = float(df[sales_col].sum()) if sales_col in df.columns else 0.0
    total_customer = int(df[cust_col].nunique()) if cust_col in df.columns else 0
    total_orders = int(df[order_col].nunique()) if order_col in df.columns else len(df)

    champions = len(rfm[rfm['Segment'] == 'Champions']) if (rfm is not None and not rfm.empty and 'Segment' in rfm.columns) else 0
    at_risk = len(rfm[rfm['Segment'] == 'At Risk']) if (rfm is not None and not rfm.empty and 'Segment' in rfm.columns) else 0

    forecast_text = "\n".join([
        f"  {p['year']}-{p['month']:02d}: "
        f"${p['prediction']:,.2f}"
        for p in forecast
    ]) if forecast else "Not available"

    prompt = ChatPromptTemplate.from_messages([
        ('system', """You are a C-suite business analyst.
         Write a concise executive summary. Max 200 words.
         Include key metrics, insights, and recommendations."""),
         ("human", """
         Write an executive summary for : 
          
          BUSINESS OVERVIEW : 
          Total Revenue : ${total_revenue:,.2f}
          Total Customer : {total_customer:,}
          Total Orders : {total_orders:,}
          Champions Customers : {champions}
          At Risk Customers : {at_risk}
          
          SALES FORECAST : {forecast_text}
          
          Include:
          - Performance summary
          - Key achievments 
          - Risk areas
          - Strategic recommendations
          """.format(
              total_revenue=total_revenue,
              total_customer=total_customer,
              total_orders=total_orders,
              champions=champions,
              at_risk=at_risk,
              forecast_text=forecast_text
          ))
      
    ])

    chain = prompt | llm | parser
    return chain.invoke({})

def category_deep_dive(df: pd.DataFrame, category: str) -> str:
    """Specific category analysis using dynamic columns"""
    cols = detect_columns(df)
    cat_col = cols.get('category', 'Category')
    sales_col = cols.get('sales', 'Sales')
    date_col = cols.get('order_date', 'Order Date')
    reg_col = cols.get('region', 'Region')
    sub_col = cols.get('sub_category')

    if cat_col not in df.columns or sales_col not in df.columns:
        return f"Category '{cat_col}' or sales '{sales_col}' column not found in dataset."

    cat_df = df[df[cat_col].astype(str).str.lower() == str(category).lower()]
    if cat_df.empty:
        return f"No records found for category '{category}'."

    total = float(cat_df[sales_col].sum())
    overall_total = float(df[sales_col].sum())
    pct = (total / overall_total * 100) if overall_total > 0 else 0.0

    if sub_col and sub_col in cat_df.columns:
        sub_sales = cat_df.groupby(sub_col)[sales_col].sum().sort_values(ascending=False)
        sub_sales_text = sub_sales.to_string()
    else:
        sub_sales_text = "Sub-category column not found in dataset."

    yearly_text = "N/A"
    if date_col in cat_df.columns and pd.api.types.is_datetime64_any_dtype(cat_df[date_col]):
        valid_dates = cat_df.dropna(subset=[date_col])
        if not valid_dates.empty:
            yearly = valid_dates.groupby(valid_dates[date_col].dt.year)[sales_col].sum()
            yearly_text = yearly.to_string()

    region_sales_text = "N/A"
    if reg_col in cat_df.columns:
        region_sales = cat_df.groupby(reg_col)[sales_col].sum().sort_values(ascending=False)
        region_sales_text = region_sales.to_string()

    data = f"""
    Category : {category}
    Total Sales : ${total:,.2f}
    Market share : {pct:.1f}%

    Sub-Category Performance: {sub_sales_text}

    Yearly Trend : {yearly_text}

    Regional Performance : {region_sales_text}
    """

    prompt = ChatPromptTemplate.from_messages([
        ('system', """You are a category management expert.
         Provide deep insights about this product category."""),
         ('human', """
          Analyzing this category data : {data}
          
          Provide:
          1. Category health assessment
          2. Growth opportunities
          3. Underperforming areas
          4. Strategic Recommendations
          """)
    ])

    chain = prompt | llm | parser
    return chain.invoke({'data': data})

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))

    train_path = DATA_DIR / 'train.csv'
    rfm_path = DATA_DIR / 'customer_segments.csv'

    df = pd.read_csv(train_path, encoding='latin-1')
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y', errors='coerce')

    rfm = pd.read_csv(rfm_path) if rfm_path.exists() else pd.DataFrame()

    print("=== SALES INSIGHTS ===")
    insights = generate_sales_insights(df)
    print(insights)

    print('=== ANOMALY DETECTION ===')
    anomalies = detect_anomalies(df)
    print(anomalies)

    print("=== CUSTOMER RECOMMENDATIONS ===")
    recs = get_customer_recommendations(rfm)
    print(recs)

    print('=== CATEGORY DEEP DIVE ===')
    tech = category_deep_dive(df, "Technology")
    print(tech)

    print('=== EXECUTIVE SUMMARY ===')
    summary = generate_executive_summary(df, rfm, [])
    print(summary)







