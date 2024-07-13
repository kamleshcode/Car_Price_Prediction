# 



import streamlit as st
import pandas as pd

from carpriceproject.pipeline.prediction import PredictionPipeline
from carpriceproject.components.data_evaluation import ModelEvaluation
from carpriceproject.entity.config_entity import ModelEvaluationConfig

# Set page configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="auto"
)

# Define a custom CSS style for the header
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("static/background.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    .header {
        text-align: center;
        color: #FF5733;
        font-size: 3em;
        margin-bottom: 20px;
    }
    .subheader {
        text-align: center;
        color: #C70039;
        font-size: 1.5em;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Render the header and subheader
st.markdown('<div class="header">🚗 Car Price Prediction 🚗</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload your car data and predict prices</div>', unsafe_allow_html=True)

# Function to preprocess data for prediction
def preprocess_data(input_data):
    # Encode categorical variables
    input_data['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
                                 'Fourth & Above Owner', 'Test Drive Car'],
                                [1, 2, 3, 4, 5], inplace=True)
    input_data['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
    input_data['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    input_data['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    input_data['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
                                'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
                                'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
                                'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
                                'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                               inplace=True)
    
    return input_data

# Function to start prediction
def start_predicting():
    # User input for prediction
    name = st.selectbox('🚘 Select Car Brand', ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
                                               'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
                                               'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
                                               'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
                                               'Ambassador', 'Ashok', 'Isuzu', 'Opel'])
    year = st.slider('📅 Car Manufactured Year', 1994, 2024)
    km_driven = st.slider('🛣️ No of kms Driven', 11, 200000)
    fuel = st.selectbox('⛽ Fuel type', ['Diesel', 'Petrol', 'LPG', 'CNG'])
    seller_type = st.selectbox('🏢 Seller type', ['Individual', 'Dealer', 'Trustmark Dealer'])
    transmission = st.selectbox('🔄 Transmission type', ['Manual', 'Automatic'])
    owner = st.selectbox('👤 Owner type', ['First Owner', 'Second Owner', 'Third Owner',
                                           'Fourth & Above Owner', 'Test Drive Car'])
    mileage = st.slider('⚡ Car Mileage (kmpl)', 10, 40)
    engine = st.slider('🔧 Engine CC', 700, 5000)
    max_power = st.slider('💪 Max Power (bhp)', 0, 200)
    seats = st.slider('🪑 No of Seats', 2, 10)

    if st.button("💰 Predict Price"):
        input_data_model = pd.DataFrame(
            [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
            columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
        )
        
        # Preprocess input data
        input_data_model = preprocess_data(input_data_model)
        
        # Perform prediction using PredictionPipeline
        prediction_pipeline = PredictionPipeline()
        predictions = prediction_pipeline.predict(input_data_model)
        
        st.markdown(f'**Predicted Car Price:** ₹ {predictions[0]:.2f}')

# Function to evaluate the model
def evaluate_model():
    st.subheader("Model Evaluation")
    test_data_path = st.text_input("Enter the path to the test data CSV", value="path/to/test_data.csv")
    model_path = st.text_input("Enter the path to the model file", value="artifacts/model_trainer/model.joblib")
    target_column = st.text_input("Enter the target column name", value="price")
    metric_file_name = "metrics.json"

    if st.button("Evaluate", key="evaluate_button", help="Click to evaluate the model"):
        with st.spinner("🔍 Evaluating the model..."):
            evaluation_config = ModelEvaluationConfig(test_data_path, model_path, target_column, metric_file_name)
            evaluator = ModelEvaluation(config=evaluation_config)
            metrics = evaluator.save_results()
            st.success("Evaluation completed.")
            st.json(metrics)

# Run the application
if __name__ == "__main__":
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Predict", "Evaluate Model"])

    if selection == "Predict":
        start_predicting()
    elif selection == "Evaluate Model":
        evaluate_model()
