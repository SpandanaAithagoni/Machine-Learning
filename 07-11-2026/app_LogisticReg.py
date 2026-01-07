import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Page Config
st.set_page_config("Logistic Regression", layout="centered")
/* App background */
.stApp{
    background: linear-gradient(
        135deg,
        #ecfeff 0%,
        #d1fae5 45%,
        #f0fdfa 100%
    );
    min-height:100vh;
    font-family:'Segoe UI', sans-serif;
}

/* Global text */
body, p, span, label, div, li{
    color:#064e3b !important;
    opacity:1 !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6{
    color:#022c22 !important;
    font-weight:800;
}

/* Cards */
.card{
    background: linear-gradient(
        135deg,
        #ffffff,
        #ccfbf1
    );
    padding:26px;
    border-radius:18px;
    box-shadow:0 14px 28px rgba(6,78,59,0.18);
    margin-bottom:26px;
}

/* Dataset table */
[data-testid="stDataFrame"]{
    background: linear-gradient(
        135deg,
        #ecfeff,
        #ccfbf1
    );
    border-radius:14px;
    padding:10px;
    box-shadow:0 10px 22px rgba(6,78,59,0.18);
}

/* Table header */
[data-testid="stDataFrame"] th{
    background:#99f6e4 !important;
    color:#022c22 !important;
    font-weight:700;
    border-bottom:1px solid #5eead4;
}

/* Table body cells */
[data-testid="stDataFrame"] td{
    background:#ecfeff !important;
    color:#022c22 !important;
    border-bottom:1px solid #ccfbf1;
}

/* Metric cards */
div[data-testid="metric-container"]{
    background: linear-gradient(
        135deg,
        #99f6e4,
        #5eead4
    );
    border-radius:16px;
    padding:18px;
    box-shadow:0 8px 20px rgba(6,78,59,0.28);
}

div[data-testid="metric-container"] *{
    color:#022c22 !important;
}

/* Prediction box */
.prediction-box{
    background: linear-gradient(
        135deg,
        #2dd4bf,
        #14b8a6
    );
    padding:22px;
    border-radius:22px;
    text-align:center;
    font-size:22px;
    font-weight:800;
    color:#022c22 !important;
    margin-top:15px;
    box-shadow:
        0 10px 26px rgba(6,78,59,0.35),
        0 0 0 2px rgba(20,184,166,0.35);
}

# Title
st.markdown("""
<div class="card">
    <h1>Logistic Regression</h1> 
    <p>Predict <b>Customer Churn</b> using <b>Monthly Charges</b></p>
</div>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = load_data()

# Dataset Preview
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# Data Preparation
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

X = df[["MonthlyCharges"]]
y = df["Churn"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Visualization
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Monthly Charges vs Churn Probability")

x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
x_range_scaled = scaler.transform(x_range)
y_range_prob = model.predict_proba(x_range_scaled)[:, 1]

fig, ax = plt.subplots()
ax.scatter(X, y, color="red", alpha=0.4, label="Actual Data")
ax.plot(x_range, y_range_prob, color="blue", label="Logistic Regression Curve")
ax.set_xlabel("Monthly Charges")
ax.set_ylabel("Probability of Churn")
ax.legend()
ax.grid(True, alpha=0.2)

st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# Model Performance
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

cm = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix")
st.write(cm)

st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

st.markdown('</div>', unsafe_allow_html=True)

# Model Parameters
st.markdown(f"""
<div class="card">
    <h3>Model Parameters</h3>
    <p>
        <b>Coefficient:</b> {model.coef_[0][0]:.3f}<br>
        <b>Intercept:</b> {model.intercept_[0]:.3f}
    </p>
</div>
""", unsafe_allow_html=True)

# Prediction with Slider
st.markdown('<div class="card">', unsafe_allow_html=True)

bill = st.slider(
    "Select Monthly Charges ($)",
    float(X.min()),
    float(X.max()),
    70.0
)

bill_scaled = scaler.transform([[bill]])
prob = model.predict_proba(bill_scaled)[0][1]

st.markdown(
    f'<div class="prediction-box">Churn Probability : {prob:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)

