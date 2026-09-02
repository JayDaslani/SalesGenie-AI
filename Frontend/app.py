import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import uuid
import time

st.set_page_config(
    page_title='SalesGenie AI',
    page_icon="🧞",
    layout='wide',
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    * { font-family: 'Inter', sans-serif; }

    /* Background */
    .stApp {
        background: linear-gradient(
            135deg,
            #0a0a0f 0%,
            #0d1117 40%,
            #0a0f1e 100%
        );
    }
            
    /* Hide default header */
    header { visibility: hidden; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0d1117 0%,
            #161b27 100%
        );
        border-right: 1px solid #21262d;
    }
    
    /* Main content padding */
    .main .block-container {
        padding: 1rem 2rem;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(
            135deg,
            #1a1a2e 0%,
            #16213e 50%,
            #0f3460 100%
        );
        border: 1px solid #00d4ff22;
        border-radius: 20px;
        padding: 30px 40px;
        margin-bottom: 25px;
        position: relative;
        overflow: hidden;
    }
            
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(
            circle,
            #00d4ff11 0%,
            transparent 70%
        );
    }
            
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(
            135deg,
            #00d4ff,
            #7b2ff7,
            #ff6b6b
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }
            
    .hero-subtitle {
        color: #8b949e;
        font-size: 1rem;
        margin-top: 8px;
        font-weight: 300;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(
            135deg,
            #161b27,
            #1c2333
        );
        border: 1px solid #21262d;
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
            
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(
            90deg, #00d4ff, #7b2ff7
        );
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00d4ff;
        margin: 8px 0 4px;
    }
            
    .metric-label {
        color: #8b949e;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-icon {
        font-size: 1.5rem;
        margin-bottom: 5px;
    }
            
    /* Section Headers */
    .section-header {
        color: #e6edf3;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 10px 0;
        border-bottom: 1px solid #21262d;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
            
    /* Chat Messages */
    .chat-user {
        background: linear-gradient(
            135deg, #1a3a5c, #1e4080
        );
        border: 1px solid #2d5a8e;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 18px;
        margin: 8px 0;
        color: #e6edf3;
        font-size: 0.95rem;
    }

    .chat-ai {
        background: linear-gradient(
            135deg, #1a1a2e, #16213e
        );
        border: 1px solid #00d4ff33;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 18px;
        margin: 8px 0;
        color: #e6edf3;
        font-size: 0.95rem;
    }
            
    /* Insight Cards */
    .insight-card {
        background: linear-gradient(
            135deg, #161b27, #1c2333
        );
        border: 1px solid #21262d;
        border-left: 4px solid #00d4ff;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 10px 0;
        color: #c9d1d9;
        font-size: 0.95rem;
        line-height: 1.6;
    }
            
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #161b27;
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid #21262d;
    }

    .stTabs [data-baseweb="tab"] {
        color: #8b949e;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(
            135deg, #00d4ff22, #7b2ff722
        ) !important;
        color: #00d4ff !important;
        border: 1px solid #00d4ff44 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(
            135deg, #00d4ff, #7b2ff7
        );
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
    }
            
    .stButton > button:hover {
        opacity: 0.85;
        transform: translateY(-1px);
    }

    /* Sidebar logo */
    .sidebar-logo {
        text-align: center;
        padding: 20px 0;
        border-bottom: 1px solid #21262d;
        margin-bottom: 20px;
    }
            
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(
            135deg, #00d4ff, #7b2ff7
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
            
    /* Status badge */
    .status-badge {
        display: inline-block;
        background: #1a3a1a;
        border: 1px solid #2ea043;
        color: #3fb950;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track {
        background: #0d1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #21262d;
        border-radius: 3px;
    }         
</style>
""", unsafe_allow_html=True)

def api_get(endpoint):
    try:
        r = requests.get(f"{API_URL}{endpoint}",timeout=30)
        return r.json() if r.ok else None
    except:
        return None
    
def api_post(endpoint, data):
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=data, timeout=60)
        return r.json() if r.ok else None
    except: 
        return None
    
CHART_THEME = {
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'font': {'color': '#c9d1d9', 'family': 'Inter'},
    'xaxis': {
        'gridcolor': '#21262d',
        'linecolor': '#21262d'
    },
    'yaxis': {
        'gridcolor': '#21262d',
        'linecolor': '#21262d'
    }
}

COLORS = [
    '#00d4ff', '#7b2ff7', '#ff6b6b',
    '#ffd700', '#00ff88', '#ff8c00'
]

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'overview' not in st.session_state:
    st.session_state.overview = api_get("/data/overview")

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size:2.5rem">🧞</div>
        <div class="sidebar-title">SalesGenie AI</div>
        <div style="color:#8b949e;font-size:0.75rem;
                    margin-top:4px">
            Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    health = api_get('/health')
    if health:
        st.markdown(
            '<span class="status-badge">'
            '● API Connected</span>',
            unsafe_allow_html=True
        )
    else:
        st.error("⚠️ API Offline")

    if st.session_state.overview:
        ov = st.session_state.overview
        st.markdown(
            "**📊 Dataset Info**",
        )
        date_info = ov.get('date_range') or ov.get('data_range') or {}
        st.caption(f"📅 {date_info.get('start', 'N/A')} → {date_info.get('end', 'N/A')}")
        
        
        total_cust = ov.get('total_customers') or ov.get('total_customer', 0)
        st.caption(
            f"📦 {ov['total_orders']:,} Orders"
        )
        st.caption(f"👥 {total_cust:,} Customers")
        st.caption(
            f"🏷️ {ov['total_products']:,} Products"
        )

    st.markdown("---")

    st.markdown("**🗺️ Navigation**",)

    nav_items = [
        "📊 Dashboard",
        "📈 Forecast",
        "👥 Segments",
        "🤖 AI Chat",
        "💡 Insights",
    ]
    for item in nav_items:
        st.caption(item)

    st.markdown("---")
    st.caption(
        "Built with Langchain + Langgraph\n"
        "Powered by Groq + Llama 3.3 70B"
    )

st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🧞 SalesGenie AI</div>
    <div class="hero-subtitle">
        AI-Powered Sales Intelligence Platform •
        ML Forecasting • Customer Segmentation •
        Natural Language Analytics
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "📈 Forecast",
    "👥 Segments",
    "🤖 AI Chat",
    "💡 Insights"
])

with tab1:
    ov = st.session_state.overview

    if ov:
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">💰</div>
                <div class="metric-value">
                    ${ov['total_revenue']:,.0f}
                </div>
                <div class="metric-label">
                    Total Revenue
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📦</div>
                <div class="metric-value">
                    {ov['total_orders']:,}
                </div>
                <div class="metric-label">
                    Total Orders
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            total_cust = ov.get('total_customers') or ov.get('total_customer', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">👥</div>
                <div class="metric-value">
                    {total_cust:,}
                </div>
                <div class="metric-label">
                    Customers
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with c4:
            avg = ov['total_revenue'] / ov['total_orders']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🎯</div>
                <div class="metric-value">
                    ${avg:,.0f}
                </div>
                <div class="metric-label">
                    Avg Order Value
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(
                '<div class="section-header">'
                '📈 Monthly Revenue Trend</div>',
                unsafe_allow_html=True
            )
            monthly_data = api_get("/data/sales/monthly")
            if monthly_data:
                df_m = pd.DataFrame(monthly_data['monthly_sales'])
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_m['Order Date'],
                    y=df_m['Sales'],
                    mode='lines+markers',
                    line=dict(
                        color='#00d4ff',
                        width=2.5
                    ),
                    marker=dict(size=5),
                    fill='tozeroy',
                    fillcolor='rgba(0,212,255,0.08)',
                    name='Revenue'
                ))
                fig.update_layout(
                    **CHART_THEME,
                    height=280,
                    margin=dict(
                        l=0, r=0, t=10, b=0
                    ),
                    showlegend=False
                )
                st.plotly_chart(fig,use_container_width=True)

        with col2:
            st.markdown(
                '<div class="section-header">'
                '🍕 Category Split</div>',
                unsafe_allow_html=True
            )
            cat_data = api_get("/data/sales/category")
            if cat_data:
                df_c = pd.DataFrame(cat_data['category_sales'])
                fig = go.Figure(go.Pie(
                    labels=df_c['Category'],
                    values=df_c['Sales'],
                    hole=0.6,
                    marker_colors=COLORS[:3]
                ))
                fig.update_layout(
                    **CHART_THEME,
                    height=280,
                    margin=dict(
                        l=0, r=0, t=10, b=0
                    ),
                    legend=dict(
                        font=dict(color='#8b949e')
                    )
                )
                st.plotly_chart(fig,use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown(
                '<div class="section-header">'
                '🗺️ Regional Performance</div>',
                unsafe_allow_html=True
            )
            reg_data = api_get("/data/sales/region")
            if reg_data:
                df_r = pd.DataFrame(reg_data['region_sales']).sort_values('Sales')
                fig = go.Figure(go.Bar(
                    x=df_r['Sales'],
                    y=df_r['Region'],
                    orientation='h',
                    marker=dict(
                        color=COLORS[:4],
                        line=dict(
                            color='rgba(0,0,0,0)'
                        )
                    )
                ))
                fig.update_layout(
                    **CHART_THEME,
                    height=250,
                    margin=dict(
                        l=0, r=0, t=10, b=0
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.markdown(
                '<div class="section-header">'
                '🏷️ Top Categories</div>',
                unsafe_allow_html=True
            )
            if cat_data:
                df_c2 = pd.DataFrame(cat_data['category_sales']).sort_values('Sales', ascending=True)
                fig = go.Figure(go.Bar(
                    x=df_c2['Sales'],
                    y=df_c2['Category'],
                    orientation='h',
                    marker=dict(
                        color=COLORS[1:4]
                    )
                ))
                fig.update_layout(
                    **CHART_THEME,
                    height=250,
                    margin=dict(
                        l=0, r=0, t=10, b=0
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown(
        '<div class="section-header">'
        '📈 ML Sales Forecasting</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        months = st.slider("Forecast Months", min_value=1, max_value=12, value=3)
        run_forecast = st.button("🚀 Generate Forecast")

    with col2:
        if run_forecast:
            with st.spinner("ML model predicting ..."):
                result = api_post("/ml/forecast", {"months": months})

            if result:
                st.success(
                    f"Model: {result['model']} | "
                    f"Accuracy: {result['accuracy']}"
                )

                df_f = pd.DataFrame(result['forecast'])
                df_f['Period'] = df_f.apply(
                    lambda x: f"{int(x['year'])}-{int(x['month']):02d}",
                    axis=1
                )
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_f['Period'],
                    y=df_f['predicted_sales'],
                    marker=dict(
                        color=COLORS[:len(df_f)],
                    ),
                    text=df_f[
                        'predicted_sales'
                    ].apply(
                        lambda x: f"${x:,.0f}"
                    ),
                    textposition='outside'
                ))
                max_sales = df_f['predicted_sales'].max() if len(df_f) > 0 else 100000
                fig.update_layout(
                    **CHART_THEME,
                    height=350,
                    title="Sales Forecast",
                    showlegend=False,
                    margin=dict(t=50, b=20, l=10, r=10),
                    
                )
                fig.update_yaxes(range=[0, max_sales*1.2])
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    df_f[[
                        'Period',
                        'predicted_sales'
                    ]].rename(columns={
                        'predicted_sales':
                        'Predicted Sales ($)'
                    }),
                    use_container_width=True
                )
        else:
            st.info(
                "👈 Select months and click "
                "Generate Forecast"
            )



with tab3:
    st.markdown(
        '<div class="section-header">'
        '👥 Customer Segmentation</div>',
        unsafe_allow_html=True
    )

    seg_data = api_get("/ml/segments")

    if seg_data:
        df_s = pd.DataFrame(
            seg_data['segments']
        )

        # Dynamic columns
        num_segs = len(df_s)
        cols = st.columns(num_segs)

        seg_colors = {
            'Champions': '#ffd700',
            'Loyal Customers': '#00d4ff',
            'At Risk': '#ff8c00',
            'Lost Customers': '#ff6b6b',
            'Potential Loyalists': '#00ff88'
        }

        seg_icons = {
            'Champions': '👑',
            'Loyal Customers': '💎',
            'At Risk': '⚠️',
            'Lost Customers': '❌',
            'Potential Loyalists': '🌱'
        }

        for idx, row in df_s.iterrows():
            color = seg_colors.get(
                row['Segment'], '#00d4ff'
            )
            icon = seg_icons.get(
                row['Segment'], '👤'
            )
            with cols[idx]:
                st.markdown(f"""
                <div class="metric-card"
                style="border-top:3px solid {color}">
                    <div style="font-size:1.8rem">
                        {icon}
                    </div>
                    <div style="color:{color};
                    font-weight:700;
                    font-size:0.8rem;
                    margin:6px 0;
                    text-transform:uppercase;
                    letter-spacing:1px">
                        {row['Segment']}
                    </div>
                    <div class="metric-value"
                    style="color:{color}">
                        {int(row['count']):,}
                    </div>
                    <div class="metric-label">
                        customers
                    </div>
                    <div style="color:#8b949e;
                    font-size:0.75rem;
                    margin-top:8px;
                    border-top:1px solid #21262d;
                    padding-top:8px">
                        💰 Avg: ${row['avg_monetary']:,.0f}
                        <br>
                        📦 Freq: {row['avg_frequency']:.1f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(
            "<br>", unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure(go.Pie(
                labels=df_s['Segment'],
                values=df_s['count'],
                hole=0.55,
                marker_colors=[
                    seg_colors.get(s, '#00d4ff')
                    for s in df_s['Segment']
                ],
                textinfo='label+percent'
            ))
            fig.update_layout(
                **CHART_THEME,
                height=320,
                title="Customer Distribution",
                showlegend=False
            )
            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:
            df_rev = df_s.sort_values(
                'total_revenue', ascending=False
            )
            fig = go.Figure(go.Bar(
                x=df_rev['Segment'],
                y=df_rev['total_revenue'],
                marker_color=[
                    seg_colors.get(s, '#00d4ff')
                    for s in df_rev['Segment']
                ],
                text=df_rev[
                    'total_revenue'
                ].apply(
                    lambda x: f"${x:,.0f}"
                ),
                textposition='outside'
            ))
            max_rev = df_rev['total_revenue'].max()
            fig.update_layout(
                **CHART_THEME,
                height=320,
                title="Revenue by Segment",
                margin=dict(t=50, b=10, l=0, r=0),
                
            )
            fig.update_yaxes(range=[0, max_rev*1.2])
            st.plotly_chart(
                fig,
                use_container_width=True
            )


with tab4:
    st.markdown(
        '<div class="section-header">'
        '🤖 Chat with Your Data</div>',
        unsafe_allow_html=True
    )

    st.markdown("**⚡ Quick Questions:**")

    quick_qs = [
        "What is total revenue?",
        "Best performing region?",
        "Forecast next 3 months",
        "Customer segments summary"
    ]

    q_cols = st.columns(4)

    for i, q in enumerate(quick_qs):
        with q_cols[i]:
            if st.button(q, key=f"quick_{i}",use_container_width=True):
                st.session_state.chat_history.append(
                    {"role": "user", "content": q}
                )
                with st.spinner("Thinking..."):
                    res = api_post(
                        "/ai/chat",
                        {
                            "message": q,
                            "session_id":
                            st.session_state.session_id
                        }
                    )
                if res:
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": res['response']
                        }
                    )

    st.markdown("---")

    chat_container = st.container()

    with chat_container:
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(
                    f'<div class="chat-user">'
                    f'👤 {msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-ai">'
                    f'🧞 {msg["content"]}</div>',
                    unsafe_allow_html=True
                )

    user_input = st.chat_input("Ask anything about your sales data ...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("🧞 Analyzing..."):
            res = api_post(
                "/ai/chat",
                {
                    "message": user_input,
                    "session_id":
                    st.session_state.session_id
                }
            )
        if res:
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": res['response']
                }
            )
        st.rerun()
    
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


with tab5:
    st.markdown(
        '<div class="section-header">'
        '💡 AI Generated Insights</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "📊 Sales Insights",
            use_container_width=True
        ):
            with st.spinner(
                "Generating insights..."
            ):
                res = api_get("/ai/insights")
            if res:
                st.markdown(
                    f'<div class="insight-card">'
                    f'{res["insights"]}</div>',
                    unsafe_allow_html=True
                )

    with col2:
        if st.button(
            "🔍 Anomaly Detection",
            use_container_width=True
        ):
            with st.spinner(
                "Detecting anomalies..."
            ):
                res = api_get("/ai/anomalies")
            if res:
                st.markdown(
                    f'<div class="insight-card">'
                    f'{res["anomalies"]}</div>',
                    unsafe_allow_html=True
                )

    
    with col3:
        if st.button(
            "👥 Recommendations",
            use_container_width=True
        ):
            with st.spinner(
                "Getting recommendations..."
            ):
                res = api_get(
                    "/ai/recommendations"
                )
            if res:
                st.markdown(
                    f'<div class="insight-card">'
                    f'{res["recommendations"]}</div>',
                    unsafe_allow_html=True
                )

    st.markdown("---")

    st.markdown(
        '<div class="section-header">'
        '📋 Executive Summary</div>',
        unsafe_allow_html=True
    )

    if st.button("📋 Generate Executive Summary",use_container_width=True):
        with st.spinner(
            "Generating executive summary..."
        ):
            res = api_get("/ai/summary")
        if res:
            st.markdown(
                f'<div class="insight-card">'
                f'{res["summary"]}</div>',
                unsafe_allow_html=True
            )

