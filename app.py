#%%
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Nassau AI Predictor", page_icon="🍬", layout="centered")

# --- 2. LOAD THE AI MODEL ---
# Using st.cache_resource ensures the model only loads once, keeping the website lightning fast!
@st.cache_resource
def load_ai():
    model = joblib.load('nassau_rf_model.pkl')
    features = joblib.load('model_features.pkl')
    return model, features

rf_model, expected_features = load_ai()

# --- 3. DASHBOARD HEADER ---
st.title("🍬 Nassau Candy Logistics AI")
st.markdown("### Intelligent Delivery Time Predictor")
st.write("Enter the details of a new simulated order below. The Random Forest AI will calculate the expected delivery time based on carrier speeds and current factory strain.")
st.divider()

# --- 4. USER INPUTS (The UI) ---
st.subheader("📦 Order Details")
col1, col2 = st.columns(2)

with col1:
    ship_mode_names = ['Same Day (Express)', 'First Class (Priority)', 'Second Class', 'Standard Class (Ground)']
    ship_mode = st.selectbox("Shipping Service Level Paid", [0, 1, 2, 3], format_func=lambda x: ship_mode_names[x])
    units = st.number_input("Number of Units Ordered", min_value=1, max_value=5000, value=50)
    sales = st.number_input("Total Sales Value ($)", min_value=1.0, value=250.0)

with col2:
    st.info("💡 **Factory Strain:** Adjust this to simulate how busy the factory is right now.")
    rolling_vol = st.slider("Current 30-Day Factory Volume", min_value=0, max_value=5000, value=1500)
    order_month = st.selectbox("Month Order Placed", range(1, 13), format_func=lambda x: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][x-1])
    is_holiday = 1 if order_month in [11, 12] else 0

st.divider()

# --- 5. THE PREDICTION ENGINE ---
if st.button("🔮 Predict Delivery Time", use_container_width=True):
    # Create a blank dictionary with 0s for ALL expected features
    # (This prevents the model from crashing due to missing columns)
    input_data = {col: 0 for col in expected_features}

    # Overwrite the 0s with the data the user just typed into the website
    input_data['Ship Mode_Encoded'] = ship_mode

    # THE FIX: Apply log transformation to the VALUE, but keep the key as 'Sales' and 'Units'
    input_data['Sales'] = np.log1p(sales)
    input_data['Units'] = np.log1p(units)

    input_data['Rolling_30D_Units'] = rolling_vol
    input_data['Order_Month'] = order_month
    input_data['Is_Holiday_Season'] = is_holiday

    # Convert to a DataFrame so the AI can read it
    input_df = pd.DataFrame([input_data])

    # Make the prediction
    prediction = rf_model.predict(input_df)[0]

    # Display the result beautifully!
    st.success("Prediction Generated Successfully!")
    st.metric(label="Estimated Delivery Time", value=f"{prediction:.1f} Days", delta="-1.55 Days (Error Margin)",
              delta_color="off")
    st.caption("Based on historical data, this prediction has an average absolute error of 1.55 days.")
