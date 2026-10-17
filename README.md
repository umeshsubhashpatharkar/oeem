# Next-Day OEE Prediction using Machine Learning 
<!-- py -3.11 -m venv .venv -->
## 📌 Project Overview_1
This project predicts the next day's Overall Equipment Effectiveness (OEE) using Machine Learning.

The objective is to use the previous day's machine-level production and operational information to estimate the OEE for the following day.

The project combines:
- A **Machine Learning pipeline** (data preprocessing, model training, evaluation, saving).
- A **Flask API frontend** (`app.py`) that loads the saved model, accepts user input, validates data, makes predictions, and stores results in MySQL.

---

## 🎯 Problem Statement
In a manufacturing environment, OEE is an important performance indicator used to measure equipment effectiveness.

**Goal:**  
Given the previous day's machine information, predict the next day's OEE.  
This helps provide early indication of expected machine performance and supports production planning.

---

## 🔄 Machine Learning Workflow
1. Data loading  
2. Exploratory Data Analysis (EDA)  
3. Data quality checks (missing values, duplicates, invalid values)  
4. Distribution & outlier analysis  
5. Correlation analysis  
6. Time-series stationarity testing  
7. Lag-1 feature creation (previous day features)  
8. Time-based train/test split  
9. Model comparison (Linear Regression, Decision Tree, Random Forest)  
10. Model evaluation & best model selection  
11. Save trained model (`Model/OEE_Tomorrow_Model.pkl`)  
12. Load saved model in Flask and predict tomorrow’s OEE  

---

## 🌐 Flask API Overview
The **Flask app (`app.py`)** provides:
- `/` → Home route (renders `templates/index.html`)  
- `/predict` → Accepts JSON input, validates fields, predicts OEE, saves result in MySQL, returns prediction as JSON  
- `/predictions` → Returns all saved predictions from MySQL  

### Required JSON fields for `/predict`
```json
{
  "Machine_Age": 1,
  "OEE_Lag_1": 85.5,
  "Avail_Hrs_Lag_1": 10,
  "MC_Running_Hrs_Lag_1": 8.4,
  "Total_DT_Lag_1": 1.7,
  "Prod_Qty_Lag_1": 1000,
  "Rej_Qty_Lag_1": 28,
  "Maintenance_Flag_Lag_1": 1
}
