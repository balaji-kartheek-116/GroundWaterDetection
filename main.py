import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Dynamic_2017.csv')

# Drop unnecessary columns
df.drop(['S.no.', 'Name of State', 'Name of District'], axis=1, inplace=True)

# Handle missing values
df.fillna(0, inplace=True)  # Fill NaN values with 0, you may choose a different strategy based on your data

# Scale the features
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Split the dataset into features (X) and target variable (y)
X = df_scaled.drop("Net Ground Water Availability for future use", axis=1)
y = df_scaled["Net Ground Water Availability for future use"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Linear Regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

st.title("Predicting Ground Water Availability")

# Define the Streamlit app
def main():
    # Login
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    
    if username == "admin" and password == "admin123":
        st.success("Logged in as Admin")
        
        # Add user input fields
        st.subheader("Enter Feature Values:")
        recharge_rainfall_monsoon = st.slider("Recharge from rainfall During Monsoon Season", min_value=-10.0, max_value=10.0, step=0.1)
        recharge_other_sources_monsoon = st.slider("Recharge from other sources During Monsoon Season", min_value=-10.0, max_value=10.0, step=0.1)
        recharge_rainfall_non_monsoon = st.slider("Recharge from rainfall During Non Monsoon Season", min_value=-10.0, max_value=10.0, step=0.1)
        recharge_other_sources_non_monsoon = st.slider("Recharge from other sources During Non Monsoon Season", min_value=-10.0, max_value=10.0, step=0.1)
        total_annual_ground_water_recharge = st.slider("Total Annual Ground Water Recharge", min_value=-10.0, max_value=10.0, step=0.1)
        total_natural_discharges = st.slider("Total Natural Discharges", min_value=-10.0, max_value=10.0, step=0.1)
        annual_extractable_ground_water_resource = st.slider("Annual Extractable Ground Water Resource", min_value=-10.0, max_value=10.0, step=0.1)
        current_annual_ground_water_extraction_irrigation = st.slider("Current Annual Ground Water Extraction For Irrigation", min_value=-10.0, max_value=10.0, step=0.1)
        current_annual_ground_water_extraction_domestic_industrial = st.slider("Current Annual Ground Water Extraction For Domestic & Industrial Use", min_value=-10.0, max_value=10.0, step=0.1)
        total_current_annual_ground_water_extraction = st.slider("Total Current Annual Ground Water Extraction", min_value=-10.0, max_value=10.0, step=0.1)
        annual_gw_allocation_domestic_2025 = st.slider("Annual GW Allocation for Domestic Use as on 2025", min_value=-10.0, max_value=10.0, step=0.1)
        stage_of_ground_water_extraction = st.slider("Stage of Ground Water Extraction (%)", min_value=-10.0, max_value=10.0, step=0.1)
        
        # Predict button
        if st.button("Predict"):
            # Prepare input features
            input_features = np.array([
                recharge_rainfall_monsoon,
                recharge_other_sources_monsoon,
                recharge_rainfall_non_monsoon,
                recharge_other_sources_non_monsoon,
                total_annual_ground_water_recharge,
                total_natural_discharges,
                annual_extractable_ground_water_resource,
                current_annual_ground_water_extraction_irrigation,
                current_annual_ground_water_extraction_domestic_industrial,
                total_current_annual_ground_water_extraction,
                annual_gw_allocation_domestic_2025,
                stage_of_ground_water_extraction
            ]).reshape(1, -1)
            
            # Make prediction
            predicted_ground_water_availability = linear_reg.predict(input_features)
            
            # Display prediction
            st.subheader("Predicted Ground Water Availability:")
            st.write(predicted_ground_water_availability[0])
    
        # Logout button
        if st.sidebar.button("Logout"):
            st.warning("Logged out")
    elif username != "" or password != "":
        st.error("Invalid Username or Password")

if __name__ == "__main__":
    main()
