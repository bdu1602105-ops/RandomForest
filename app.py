import joblib
import pandas as pd
import streamlit as st
# Set up page config
st.set_page_config(
    page_title="Barley Yield Predictor", page_icon="🌾", layout="centered"
)

st.title("🌾 Barley Grain Yield (GY) Predictor")
st.write(
    "Enter the crop management and environmental details below to predict the Grain Yield."
)


# Load trained model pipeline
@st.cache_resource
def load_model():
    return joblib.load("barley_model_pipeline.pkl")


model = load_model()

# --- User Inputs ---
st.subheader("Input Features")

col1, col2 = st.columns(2)

with col1:
    nutrient = st.selectbox(
        "Nutrient Type", options=["N", "P", "K", "S", "Zn", "B"]
    )
    location = st.selectbox(
        "Location",
        options=[
            "Gololcha",
            "Bolososorie",
            "Gudoberet",
            "Kersa",
            "Goshebado",
            "Sinanana Dinsho",
        ],
    )
    agro_zone = st.selectbox(
        "Agro Ecological Zone", options=["SH3", "M2", "SM3", "H1"]
    )

with col2:
    soil_type = st.selectbox(
        "Soil Type", options=["Cambisols", "Nitisols", "Vertisols", "Luvisols"]
    )
    rep = st.number_input(
        "Replication (Rep)", min_value=1, max_value=5, value=1
    )
    rate = st.slider("Application Rate", min_value=0, max_value=300, value=50)

# --- Predict Button ---
if st.button("Predict Yield", type="primary"):
    # Construct input dataframe matching original features
    input_data = pd.DataFrame(
        [
            {
                "Nutrient": nutrient,
                "Location": location,
                "Agro_Ecological_Zone": agro_zone,
                "SOIL_TYPE": soil_type,
                "Rep": rep,
                "Rate": rate,
            }
        ]
    )

    # Predict
    prediction = model.predict(input_data)[0]

    # Display Result
    st.success(f"**Predicted Grain Yield (GY):** {prediction:.2f} kg/ha")
