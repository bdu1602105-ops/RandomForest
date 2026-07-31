import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Page Configuration
st.set_page_config(
    page_title="Barley Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# App Title & Description
st.title("🌾 Barley Yield Prediction Dashboard")
st.write("""
This application uses a Machine Learning model (Random Forest) trained on categorical environment data 
and nutrient application rates to predict **Grain Yield (GY)** and **Biomass Yield (BY)**.
""")

st.markdown("---")

# Sidebar: File Upload
st.sidebar.header("1. Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your barley Excel file (.xlsx)", type=["xlsx", "xls"])

@st.cache_data
def load_and_clean_data(file):
    """Loads and cleans dataset according to notebook steps."""
    df = pd.read_excel(file)
    
    # Convert yield columns to numeric, coercion replaces invalid parsing with NaN
    if "GY" in df.columns:
        df["GY"] = pd.to_numeric(df["GY"], errors="coerce")
    if "BY" in df.columns:
        df["BY"] = pd.to_numeric(df["BY"], errors="coerce")
    
    # Drop missing values in target columns
    df = df.dropna(subset=["GY", "BY"]).reset_index(drop=True)
    return df

if uploaded_file is not None:
    df = load_and_clean_data(uploaded_file)
    st.sidebar.success("Dataset successfully loaded and preprocessed!")
else:
    st.info("👈 Please upload your `barley.xlsx` file in the sidebar to begin model training.")
    
    # Display expected structure
    st.subheader("Expected Data Format")
    sample_df = pd.DataFrame({
        "Nutrient": ["S", "S"],
        "Location": ["Gololcha", "Gololcha"],
        "Agro_Ecological_Zone": ["SH3", "SH3"],
        "SOIL_TYPE": ["Cambisols", "Cambisols"],
        "Rep": [1, 1],
        "Rate": [0, 10],
        "GY": [4443.0, 4067.0],
        "BY": [15333.0, 15667.0]
    })
    st.dataframe(sample_df)
    st.stop()

# Display Dataset Preview
with st.expander("🔍 View Preprocessed Dataset Preview"):
    st.dataframe(df.head(10))
    st.write(f"**Total Records:** {df.shape[0]} rows | **Columns:** {df.shape[1]}")

# Define Features and Targets
categorical_features = ["Nutrient", "Location", "Agro_Ecological_Zone", "SOIL_TYPE"]
numerical_features = ["Rate"]
target_features = ["GY", "BY"]

X = df[categorical_features + numerical_features]
y = df[target_features]

# Pipeline Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

# Build & Train Pipeline Model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
r2_gy = r2_score(y_test["GY"], y_pred[:, 0])
r2_by = r2_score(y_test["BY"], y_pred[:, 1])
mae_gy = mean_absolute_error(y_test["GY"], y_pred[:, 0])
mae_by = mean_absolute_error(y_test["BY"], y_pred[:, 1])

# Sidebar: Prediction Input Parameters
st.sidebar.header("2. Input Prediction Parameters")

def user_input_features():
    inputs = {}
    for col in categorical_features:
        unique_vals = sorted(df[col].dropna().unique().tolist())
        inputs[col] = st.sidebar.selectbox(f"Select {col}", unique_vals)
    
    min_rate = int(df["Rate"].min())
    max_rate = int(df["Rate"].max())
    default_rate = int(df["Rate"].median())
    
    inputs["Rate"] = st.sidebar.slider("Nutrient Rate", min_value=min_rate, max_value=max_rate, value=default_rate)
    
    return pd.DataFrame(inputs, index=[0])

input_df = user_input_features()

# Display Dashboard Tabs
tab1, tab2 = st.tabs(["🚀 Yield Prediction", "📊 Model Metrics & Performance"])

with tab1:
    st.subheader("Selected Inputs for Prediction")
    st.dataframe(input_df)
    
    if st.button("Predict Yield", type="primary"):
        prediction = model.predict(input_df)
        predicted_gy = prediction[0][0]
        predicted_by = prediction[0][1]
        
        st.markdown("### 🎯 Predicted Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Predicted Grain Yield (GY)",
                value=f"{predicted_gy:,.2f}"
            )
            
        with col2:
            st.metric(
                label="Predicted Biomass Yield (BY)",
                value=f"{predicted_by:,.2f}"
            )

with tab2:
    st.subheader("Random Forest Regressor Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Grain Yield (GY)")
        st.write(f"**R² Score:** {r2_gy:.4f}")
        st.write(f"**MAE:** {mae_gy:,.2f}")
        
    with col2:
        st.markdown("#### Biomass Yield (BY)")
        st.write(f"**R² Score:** {r2_by:.4f}")
        st.write(f"**MAE:** {mae_by:,.2f}")
