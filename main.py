import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# Load the dataset
dataset_path = 'Dynamic_2017.csv'
df = pd.read_csv(dataset_path)

# Drop unnecessary columns
df.drop(['S.no.', 'Name of State', 'Name of District'], axis=1, inplace=True)

# Handle missing values
df.fillna(0, inplace=True)

# Scale features
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Split the dataset into features (X) and target variable (y)
X = df_scaled.drop("Net Ground Water Availability for future use", axis=1)
y = df_scaled["Net Ground Water Availability for future use"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Extract min and max values from the dataset
min_values = df_scaled.min()
max_values = df_scaled.max()


def train_model(model_type):
    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Random Forest":
        model = RandomForestRegressor()
    elif model_type == "Decision Tree":
        model = DecisionTreeRegressor()
    elif model_type == "SVR":
        model = SVR()
    else:
        raise ValueError("Invalid model type selected.")
    model.fit(X_train, y_train)
    return model

    return model


def main():
    st.title("Ground Water Detection: Predicting Ground Water Availability")

    # Display the loaded dataset
    st.subheader("Loaded Dataset:")
    st.write(df_scaled)

    # Get user input for features
    st.subheader("Enter Feature Values:")
    recharge_rainfall_monsoon = st.slider("Recharge from rainfall During Monsoon Season",
                                          min_value=min_values['Recharge from rainfall During Monsoon Season'],
                                          max_value=max_values['Recharge from rainfall During Monsoon Season'],
                                          step=0.1)
    recharge_other_sources_monsoon = st.slider("Recharge from other sources During Monsoon Season",
                                               min_value=min_values[
                                                   'Recharge from other sources During Monsoon Season'],
                                               max_value=max_values[
                                                   'Recharge from other sources During Monsoon Season'],
                                               step=0.1)
    recharge_rainfall_non_monsoon = st.slider("Recharge from rainfall During Non Monsoon Season",
                                              min_value=min_values['Recharge from rainfall During Non Monsoon Season'],
                                              max_value=max_values['Recharge from rainfall During Non Monsoon Season'],
                                              step=0.1)
    recharge_other_sources_non_monsoon = st.slider("Recharge from other sources During Non Monsoon Season",
                                                   min_value=min_values[
                                                       'Recharge from other sources During Non Monsoon Season'],
                                                   max_value=max_values[
                                                       'Recharge from other sources During Non Monsoon Season'],
                                                   step=0.1)
    total_annual_ground_water_recharge = st.slider("Total Annual Ground Water Recharge",
                                                   min_value=min_values['Total Annual Ground Water Recharge'],
                                                   max_value=max_values['Total Annual Ground Water Recharge'], step=0.1)
    total_natural_discharges = st.slider("Total Natural Discharges", min_value=min_values['Total Natural Discharges'],
                                         max_value=max_values['Total Natural Discharges'], step=0.1)
    annual_extractable_ground_water_resource = st.slider("Annual Extractable Ground Water Resource",
                                                         min_value=min_values[
                                                             'Annual Extractable Ground Water Resource'],
                                                         max_value=max_values[
                                                             'Annual Extractable Ground Water Resource'],
                                                         step=0.1)
    current_annual_ground_water_extraction_irrigation = st.slider(
        "Current Annual Ground Water Extraction For Irrigation",
        min_value=min_values['Current Annual Ground Water Extraction For Irrigation'],
        max_value=max_values['Current Annual Ground Water Extraction For Irrigation'], step=0.1)
    current_annual_ground_water_extraction_domestic_industrial = st.slider(
        "Current Annual Ground Water Extraction For Domestic & Industrial Use",
        min_value=min_values['Current Annual Ground Water Extraction For Domestic & Industrial Use'],
        max_value=max_values['Current Annual Ground Water Extraction For Domestic & Industrial Use'], step=0.1)
    total_current_annual_ground_water_extraction = st.slider("Total Current Annual Ground Water Extraction",
                                                             min_value=min_values[
                                                                 'Total Current Annual Ground Water Extraction'],
                                                             max_value=max_values[
                                                                 'Total Current Annual Ground Water Extraction'],
                                                             step=0.1)

    annual_gw_allocation_domestic_2025 = st.slider("Annual GW Allocation for Domestic Use as on 2025",
                                                   min_value=min_values[
                                                       'Annual GW Allocation for Domestic Use as on 2025'],
                                                   max_value=max_values[
                                                       'Annual GW Allocation for Domestic Use as on 2025'], step=0.1)
    stage_of_ground_water_extraction = st.slider("Stage of Ground Water Extraction (%)",
                                                 min_value=min_values['Stage of Ground Water Extraction (%)'],
                                                 max_value=max_values['Stage of Ground Water Extraction (%)'], step=0.1)

    # Model selection
    model_type = st.selectbox("Select Model", ["Linear Regression", "Decision Tree", "Random Forest", "SVR"])

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

        model = train_model(model_type)
        predicted_ground_water_availability = model.predict(input_features.reshape(1, -1))

        # Display the predicted Ground Water Availability
        st.subheader("Predicted Ground Water Availability:")
        st.write(predicted_ground_water_availability[0])


if __name__ == "__main__":
    main()
