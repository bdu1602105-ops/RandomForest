import os
import pickle
import streamlit as st

# Dynamically construct the path to model.pkl
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

# Load the model safely
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Model file not found at: {MODEL_PATH}")