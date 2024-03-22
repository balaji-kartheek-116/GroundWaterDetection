import streamlit as st
import numpy as np
import joblib

# Load the trained Linear Regression model
lr_model = joblib.load('linear_regression_model.pkl')

def main():
    st.title("Ground Water Detection: Predicting Ground Water Availability")

    # Get user input for features
    st.subheader("Enter Feature Values:")
    recharge_rainfall_monsoon = st.slider("Recharge from rainfall During Monsoon Season", -10.0, 10.0, step=0.1)
    recharge_other_sources_monsoon = st.slider("Recharge from other sources During Monsoon Season", -10.0, 10.0, step=0.1)
    recharge_rainfall_non_monsoon = st.slider("Recharge from rainfall During Non Monsoon Season", -10.0, 10.0, step=0.1)
    recharge_other_sources_non_monsoon = st.slider("Recharge from other sources During Non Monsoon Season", -10.0, 10.0, step=0.1)
    total_annual_ground_water_recharge = st.slider("Total Annual Ground Water Recharge", -10.0, 10.0, step=0.1)
    total_natural_discharges = st.slider("Total Natural Discharges", -10.0, 10.0, step=0.1)
    annual_extractable_ground_water_resource = st.slider("Annual Extractable Ground Water Resource", -10.0, 10.0, step=0.1)
    current_annual_ground_water_extraction_irrigation = st.slider("Current Annual Ground Water Extraction For Irrigation", -10.0, 10.0, step=0.1)
    current_annual_ground_water_extraction_domestic_industrial = st.slider("Current Annual Ground Water Extraction For Domestic & Industrial Use", -10.0, 10.0, step=0.1)
    total_current_annual_ground_water_extraction = st.slider("Total Current Annual Ground Water Extraction", -10.0, 10.0, step=0.1)
    annual_gw_allocation_domestic_2025 = st.slider("Annual GW Allocation for Domestic Use as on 2025", -10.0, 10.0, step=0.1)
    stage_of_ground_water_extraction = st.slider("Stage of Ground Water Extraction (%)", -10.0, 10.0, step=0.1)

    # Predict button
    if st.button("Predict"):
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
        ])

        # Make predictions using the loaded Linear Regression model
        predicted_ground_water_availability = lr_model.predict(input_features.reshape(1, -1))

        # Display the predicted Ground Water Availability
        st.subheader("Predicted Ground Water Availability:")
        st.write(predicted_ground_water_availability[0])

if __name__ == "__main__":
    main()
