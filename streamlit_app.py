import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import lightgbm

# Streamlit App
st.title('Telco Churn Prediction Dockerized App')

# Read CSV into a Pandas DataFrame
df = pd.read_csv('final.csv')

# Display DataFrame
st.write("DataFrame:")
st.write(df)

# Mapping dictionary
mapping = {'Yes': 1, 'No': 0}

# Applying mapping to the 'churn' column
df['Churn'] = df['Churn'].map(mapping)

X = df.drop('Churn', axis=1)
y = df['Churn']

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the LightGBM model
lgbm_classifier = joblib.load('lgbm_model.pkl')


if st.button('Show Example for (Churn = 0)'):
    # Select the example from the test data where Churn is 0
    example_0 = X_test[y_test == 0].sample(n=1)
    st.subheader('Example for (Churn = 0):')
    st.write(example_0)

    # Make predictions for the example
    prediction = lgbm_classifier.predict_proba(example_0)

    # Extract the probability of churn (label 1) and no churn (label 0)
    probability_churn = prediction[0][1]
    probability_no_churn = prediction[0][0]

    # Determine the predicted label based on the probability
    predicted_label = 1 if probability_churn > 0.5 else 0

    # Display the prediction result
    st.subheader('Prediction for No Churn (0) Example:')
    if predicted_label == 1:
        st.error(f'Churn (1) - Probability (False Positive): {probability_churn:.5f}')
    else:
        st.success(f'No Churn (0) - Probability: {probability_no_churn:.5f}')

if st.button('Show Example for (Churn = 1)'):
    # Select the example from the test data where Churn is 1
    example_1 = X_test[y_test == 1].sample(n=1)
    st.subheader('Example for (Churn = 1):')
    st.write(example_1)

    # Make predictions for the example
    prediction = lgbm_classifier.predict_proba(example_1)

    # Extract the probability of churn (label 1) and no churn (label 0)
    probability_churn = prediction[0][1]
    probability_no_churn = prediction[0][0]

    # Determine the predicted label based on the probability
    predicted_label = 1 if probability_churn > 0.5 else 0

    # Display the prediction result
    st.subheader('Prediction for Churn (1) Example:')
    if predicted_label == 1:
        st.success(f'Churn (1) - Probability: {probability_churn:.5f}')
    else:
        st.error(f'No Churn (0) - Probability (False Negative): {probability_no_churn:.5f}')