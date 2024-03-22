import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the Linear Regression Model with joblib
lr_model = joblib.load('models/linear_reg_model.pkl')

# Load the dataset
dataset_path = 'Dynamic_2017.csv'
df = pd.read_csv(dataset_path)

# Drop unnecessary columns
df.drop(['S.no.', 'Name of State', 'Name of District'], axis=1, inplace=True)

# Handle missing values
df.fillna(0, inplace=True)

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_scaled.drop("Net Ground Water Availability for future use", axis=1, inplace=True)

# Extract min and max values from the dataset
min_values = df_scaled.min()
max_values = df_scaled.max()

def main():
    st.title("Ground Water Detection: Predicting Ground Water Availability")

    # Display the loaded dataset
    st.subheader("Loaded Dataset:")
    st.write(df_scaled)

    # Get user input for features
    st.subheader("Enter Feature Values:")
    input_features = {}
    for column in df_scaled.columns:
        input_features[column] = st.slider(f"{column}",
                                           min_value=min_values[column],
                                           max_value=max_values[column],
                                           step=0.1)

    # Predict button
    if st.button("Predict"):
        input_features_arr = np.array([input_features[column] for column in df_scaled.columns])
        
        # Make predictions using the Linear Regression model
        predicted_ground_water_availability = lr_model.predict(input_features_arr.reshape(1, -1))

        # Display the predicted Ground Water Availability
        st.subheader("Predicted Ground Water Availability:")
        st.write(predicted_ground_water_availability[0])

if __name__ == "__main__":
    main()
