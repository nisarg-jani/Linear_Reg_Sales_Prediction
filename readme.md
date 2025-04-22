# 🧠 Advertising Sales Prediction Web App

An end-to-end machine learning project that predicts **product sales** based on advertising budget allocations across **TV**, **Radio**, and **Newspaper** channels. This project demonstrates the complete ML lifecycle from EDA to deployment, built using `scikit-learn` and deployed with `Streamlit Cloud`.

---

## 🚀 Live App

👉 [Try the Live Demo](https://linear-regression-sales-prediction.streamlit.app/)

---

## 📈 Problem Statement

Companies often invest in various advertising channels but struggle to quantify how this spend translates into product sales. This project leverages linear modeling techniques to:

- Understand the relationship between advertising spend and sales
- Predict future sales based on budget allocations
- Provide a user-friendly tool to simulate different advertising strategies

---

## ✅ Key Features

- 📊 Exploratory Data Analysis & Data Preprocessing
- ✂️ Outlier handling using IQR capping on the **Newspaper** column
- ⚖️ Feature Scaling using `StandardScaler`
- 🧪 Regularized Linear Regression using `LassoCV` (auto-selects best alpha)
- 📦 Model serialization using `pickle`
- 🌐 Deployed on Streamlit Cloud for interactive user input & real-time predictions

---

## 🧰 Tech Stack

| Component         | Library        |
|------------------|----------------|
| Language         | Python         |
| Data Handling    | Pandas, NumPy  |
| Modeling         | scikit-learn   |
| Visualization    | Matplotlib     |
| Web Interface    | Streamlit      |
| Deployment       | Streamlit Cloud|

---

## 🧪 How It Works

1. User inputs advertising spend for **TV**, **Radio**, and **Newspaper**
2. Input data is capped for outliers and scaled using pre-fit `StandardScaler`
3. The trained `LassoCV` model predicts expected sales
4. Prediction is instantly shown in the UI

---

## 📁 Project Structure
├── Advertising 1.3.ipynb      # Notebook with data cleaning, modeling, and pickle creation  
├── model_pipeline.pkl         # Serialized model + scaler + capping thresholds  
├── app.py                     # Streamlit web application code  
├── requirements.txt           # Dependencies for deployment  
└── README.md                  # Project documentation  


---

## 📊 Model Performance

| Metric    | Value    |
|-----------|----------|
| R² Score  | 0.895    |
| MAE       | 1.3775   |
| RMSE      | 1.6844   |

> *Model trained using Lasso Regression with cross-validation on Advertising dataset.*

---

## 💡 Learnings & Highlights

- Gained practical experience building and deploying an ML model
- Learned to handle outliers using IQR technique
- Applied `StandardScaler` to normalize feature distributions
- Explored regularization with Lasso to reduce overfitting
- Packaged a complete ML pipeline using `pickle`
- Successfully deployed and shared a real-time prediction app using Streamlit

---


## 📬 Connect With Me
  💼 **LinkedIn**: [linkedin.com/in/nisargjani](https://linkedin.com/in/nisargjani)

---

## 📌 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/sales-prediction-app.git
cd sales-prediction-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

