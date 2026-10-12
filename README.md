# 🧞 SalesGenie AI — Intelligent Sales Analytics Platform

> **AI-powered end-to-end sales intelligence platform** combining 
> Machine Learning forecasting, RFM customer segmentation, 
> RAG-based natural language Q&A, and automated PDF reporting.

<img width="1470" height="956" alt="Screenshot 2026-10-10 at 1 02 36 PM" src="https://github.com/user-attachments/assets/f3f5e74c-25c3-4479-9c2d-7ebff4fda677" />
<img width="1470" height="956" alt="Screenshot 2026-10-10 at 1 02 47 PM" src="https://github.com/user-attachments/assets/71daab2f-006f-421a-8c22-eb38262e0052" />
<img width="1470" height="956" alt="Screenshot 2026-10-10 at 1 03 00 PM" src="https://github.com/user-attachments/assets/0444edc7-d7b0-49c8-b3bd-344c28ecbdeb" />
<img width="1470" height="956" alt="Screenshot 2026-10-10 at 1 03 09 PM" src="https://github.com/user-attachments/assets/f573f71a-8287-4e4f-b924-192c935f4e77" />
<img width="1470" height="956" alt="Screenshot 2026-10-10 at 1 03 55 PM" src="https://github.com/user-attachments/assets/6ba39697-b230-4357-8bd5-d2b37cd8d57d" />
<img width="1470" height="956" alt="Screenshot 2026-10-10 at 1 04 02 PM" src="https://github.com/user-attachments/assets/ba585f28-9e66-4445-933e-c046bf9ae522" />

---

## 📌 Overview

SalesGenie AI is a full-stack AI application that transforms 
raw sales data into actionable business intelligence. It 
combines traditional Machine Learning with modern Generative AI 
to provide sales forecasting, customer segmentation, natural 
language analytics, and executive PDF reports — all through 
an interactive web interface.

---

## ✨ Key Features

### 🤖 AI-Powered Chat (RAG + LangGraph)
- Natural language Q&A on sales data
- LangGraph stateful agent with tool calling
- Powered by Groq API (Llama 3.3 70B)
- Persistent conversation memory

### 📈 ML Sales Forecasting
- Time series forecasting with auto feature engineering
- Lag features, rolling averages, cyclical encoding
- Multiple models compared — best selected automatically
- Month-by-month predictions with accuracy score

### 👥 RFM Customer Segmentation
- Recency, Frequency, Monetary analysis
- Segments: Champions, Loyal, At Risk, Lost
- Revenue contribution per segment
- Actionable retention recommendations

### 💡 AI Insights Engine
- Automated sales insights generation
- Anomaly detection in monthly trends
- Category deep-dive analysis
- Executive summary generation

### 📄 PDF Report Generation
- One-click professional PDF reports
- KPI metrics, charts data, forecasts
- Customer segment analysis
- AI-generated insights included

### 📊 Interactive Dashboard
- Real-time KPI metrics
- Monthly revenue trend charts
- Regional performance analysis
- Category breakdown visualization

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit |
| **Backend** | FastAPI |
| **AI Agent** | LangGraph + LangChain |
| **LLM** | Groq API (Llama 3.3 70B) |
| **ML Models** | Scikit-learn (RF, GB, LR) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly |
| **PDF Generation** | FPDF2 |
| **Language** | Python 3.10+ |

---

## 🏗️ Project Architecture
SalesGenie-AI/

│

├── Backend/

│ └── ai_pipeline/

│ ├── main.py # FastAPI endpoints

│ ├── agent.py # LangGraph AI agent

│ ├── tools.py # LangChain tools

│ ├── insights.py # AI insights engine

│ └── pdf_report.py # PDF generation

│

├── Frontend/

│ └── app.py # Streamlit UI (969 lines)

│

├── data/

│ ├── train.csv # Sales dataset

│ ├── best_model-2.pkl # Trained ML model

│ ├── customer_segments.csv

│ ├── monthly_features.csv

│ ├── model_info.json

│ └── selected_features.json

│

├── notebooks/

│ ├── EDA.ipynb

│ ├── feature_engineering.ipynb

│ ├── Customer_Segmentation.ipynb

│ └── Hyperparameter_tuning.ipynb

│

└── requirements.txt

---

## 🚀 How It Works
Raw Sales CSV
↓
Feature Engineering + ML Training
↓
FastAPI Backend (10+ endpoints)
↓
LangGraph Agent + 4 Tools
↓
Streamlit Dashboard
↓
Insights + Forecasts + PDF Reports

---

## 💻 Local Setup

### Prerequisites
```bash
Python 3.10+
Groq API Key (free at groq.com)
```

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/JayDaslani/SalesGenie-AI.git
cd SalesGenie-AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup environment
cp .env.example .env
# Add your GROQ_API_KEY in .env

# 4. Start FastAPI backend
cd Backend/ai_pipeline
uvicorn main:app --reload --port 8000

# 5. Start Streamlit frontend (new terminal)
cd Frontend
streamlit run app.py
```

### Access
