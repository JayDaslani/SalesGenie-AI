import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model='qwen/qwen3.6-27b',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0.3
)

parser = StrOutputParser()

def generate_sales_insights(df: pd.DataFrame) -> str:
    """Generates automated insights from sales data."""

    total_sales = df['Sales'].sum()
    avg_order = df['Sales'].mean()

    yearly = df.groupby(df['Order Date'].dt.year)['Sales'].sum()

    growth = ((yearly.iloc[-1] - yearly.iloc[0])/yearly.iloc[0]*100)

    best_category = df.groupby('Category')['Sales'].sum().idxmax()

    best_region = df.groupby('Region')['Sales'].sum().idxmax()

    worst_region = df.groupby('Region')['Sales'].sum().idxmin()

    monthly = df.groupby(df['Order Date'].dt.month)['Sales'].mean()

    best_month = monthly.idxmax()
    worst_month = monthly.idxmin()

    month_names = {
        1:'Jan', 2:'Feb', 3:'Mar',
        4:'Apr', 5:'May', 6:'Jun',
        7:'Jul', 8:'Aug', 9:'Sep',
        10:'Oct', 11:'Nov', 12:'Dec'
    }

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
              best_month=month_names[best_month],
              worst_month=month_names[worst_month],
              yearly_data=yearly.to_string()

          ))
    ])

    chain = prompt | llm | parser
    return chain.invoke({})


def detect_anomalies(df: pd.DataFrame) -> str:
    """Detects Unusual patterns"""

    monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum()

    mean = monthly_sales.mean()
    std = monthly_sales.std()

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
    """Complete executive summary"""

    total_revenue = df['Sales'].sum()
    total_customer = df['Customer ID'].nunique() 
    total_orders = df['Order ID'].nunique()

    champions = len(rfm[rfm['Segment'] == 'Champions'])
    at_risk = len(rfm[rfm['Segment'] == 'At Risk'])

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
    """Specific category analysis"""

    cat_df = df[df['Category'] == category]

    total = cat_df['Sales'].sum()
    pct = (total / df['Sales'].sum()) * 100 if df['Sales'].sum() > 0 else 0


    sub_col = None
    for col in cat_df.columns:
        clean_col = col.lower().replace('-', '').replace('_', '').replace(' ', '')
        if clean_col == 'subcategory':
            sub_col = col
            break
            
    if sub_col is not None:
        sub_sales = cat_df.groupby(sub_col)['Sales'].sum().sort_values(ascending=False)
        sub_sales_text = sub_sales.to_string()
    else:
        sub_sales_text = "Sub-category column not found in dataset."

    yearly = cat_df.groupby(cat_df['Order Date'].dt.year)['Sales'].sum()
    region_sales = cat_df.groupby('Region')['Sales'].sum().sort_values(ascending=False)

    data = f"""
    Category : {category}
    Total Sales : ${total:,.2f}
    Market share : {pct:.1f}%

    Sub-Category Performance: {sub_sales.to_string()}

    Yearly Tred : {yearly.to_string()}

    Regional Performance : {region_sales.to_string()}

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
    df = pd.read_csv('/Users/jaydasalani/Desktop/SalesGenie AI/data/train.csv', encoding='latin-1')

    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')

    rfm = pd.read_csv('/Users/jaydasalani/Desktop/SalesGenie AI/data/customer_segments.csv')

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







