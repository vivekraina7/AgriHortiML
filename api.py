# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import pickle
# import os
# from datetime import datetime, timedelta
# from fastapi.middleware.cors import CORSMiddleware
# import gdown

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Your frontend's URL
#     allow_credentials=True,
#     allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
#     allow_headers=["*"],  # Allow all headers
# )

# # Paths for saving models and encoders
# MODEL_DIR = "models"
# os.makedirs(MODEL_DIR, exist_ok=True)

# # Input data model
# class PricePredictionInput(BaseModel):
#     State: str
#     District: str
#     Market: str
#     Commodity: str
#     Arrival_Date: str

# # Download dataset from Google Drive
# def download_dataset():
#     dataset_url = "https://drive.google.com/uc?id=YOUR_DATASET_FILE_ID"  # Replace with your dataset file ID
#     dataset_path = "data/Price_Agriculture_commodities_Week.csv"
#     os.makedirs("data", exist_ok=True)
#     if not os.path.exists(dataset_path):
#         gdown.download(dataset_url, dataset_path, quiet=False)

# # Load dataset
# try:
#     download_dataset()  # Download dataset from Google Drive
#     data = pd.read_csv("data/Price_Agriculture_commodities_Week.csv")
#     data["Arrival_Date"] = pd.to_datetime(data["Arrival_Date"])
# except Exception as e:
#     raise Exception(f"Error loading dataset: {str(e)}")

# # Preprocess the dataset
# def preprocess_data(data):
#     # Extract day, month, year from Arrival_Date
#     data["Year"] = data["Arrival_Date"].dt.year
#     data["Month"] = data["Arrival_Date"].dt.month
#     data["Day"] = data["Arrival_Date"].dt.day

#     # Encode categorical variables
#     encoders = {}
#     for col in ["State", "District", "Market", "Commodity"]:
#         encoder = LabelEncoder()
#         data[col] = encoder.fit_transform(data[col])
#         encoders[col] = encoder

#     return data, encoders

# # Download models and encoders from Google Drive
# def download_models():
#     model_files = {
#         "rf_min.pkl": "1TJ0IB81GV5WGu2U-NHVPlXLb40Lrn1I5",
#         "rf_max.pkl": "1TQd8D2TINBiXB2NrlqsHYt9_N6LXpP6W",
#         "rf_modal.pkl": "1jB3Rfy_dsie3s5lRjiDZnX-e3S718OzX",
#         "scaler.pkl": "1ergixNUGsA0knokEkqFACRLTTZo_7-Jv",
#         "State_encoder.pkl": "16vxMkxzgU6XEebJq5pTJ9aeJbpEMDtdj",
#         "District_encoder.pkl": "1Z9K5ZXW8xHtrnTIVrVjhxjZoLHLIokAl",
#         "Market_encoder.pkl": "15HQ_JSxLW-gnMR8yX0jXxu_ug4JVxXdP",
#         "Commodity_encoder.pkl": "1FeqOHQRg0Cyx4CxuYmfS3G6xUs3dX1gg",
#     }

#     for file_name, file_id in model_files.items():
#         file_path = os.path.join(MODEL_DIR, file_name)
#         if not os.path.exists(file_path):
#             print(f"Downloading {file_name} from Google Drive...")
#             try:
#                 gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)
#                 print(f"Downloaded {file_name} successfully!")
#             except Exception as e:
#                 print(f"Failed to download {file_name}: {str(e)}")
#                 raise

# # Train the models
# def train_models():
#     global scaler, rf_min, rf_max, rf_modal

#     # Preprocess data
#     preprocessed_data, encoders = preprocess_data(data)

#     # Features and targets
#     X = preprocessed_data[["State", "District", "Market", "Commodity", "Year", "Month", "Day"]]
#     y_min = preprocessed_data["Min Price"]
#     y_max = preprocessed_data["Max Price"]
#     y_modal = preprocessed_data["Modal Price"]

#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(X, pd.concat([y_min, y_max, y_modal], axis=1), test_size=0.2, random_state=42)
#     y_train_min, y_train_max, y_train_modal = y_train["Min Price"], y_train["Max Price"], y_train["Modal Price"]
#     y_test_min, y_test_max, y_test_modal = y_test["Min Price"], y_test["Max Price"], y_test["Modal Price"]

#     # Scale features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # Train models
#     rf_min = RandomForestRegressor(random_state=42)
#     rf_min.fit(X_train_scaled, y_train_min)

#     rf_max = RandomForestRegressor(random_state=42)
#     rf_max.fit(X_train_scaled, y_train_max)

#     rf_modal = RandomForestRegressor(random_state=42)
#     rf_modal.fit(X_train_scaled, y_train_modal)

#     # Save models and scaler
#     pickle.dump(rf_min, open(os.path.join(MODEL_DIR, "rf_min.pkl"), "wb"))
#     pickle.dump(rf_max, open(os.path.join(MODEL_DIR, "rf_max.pkl"), "wb"))
#     pickle.dump(rf_modal, open(os.path.join(MODEL_DIR, "rf_modal.pkl"), "wb"))
#     pickle.dump(scaler, open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb"))
#     for key, encoder in encoders.items():
#         pickle.dump(encoder, open(os.path.join(MODEL_DIR, f"{key}_encoder.pkl"), "wb"))

#     return X_test_scaled, y_test_min, y_test_max, y_test_modal

# # Initialize models
# download_models()  # Download models and encoders from Google Drive

# # Helper function to load models and encoders
# def load_models():
#     global rf_min, rf_max, rf_modal, scaler, encoders
#     rf_min = pickle.load(open(os.path.join(MODEL_DIR, "rf_min.pkl"), "rb"))
#     rf_max = pickle.load(open(os.path.join(MODEL_DIR, "rf_max.pkl"), "rb"))
#     rf_modal = pickle.load(open(os.path.join(MODEL_DIR, "rf_modal.pkl"), "rb"))
#     scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))

#     encoders = {}
#     for col in ["State", "District", "Market", "Commodity"]:
#         encoders[col] = pickle.load(open(os.path.join(MODEL_DIR, f"{col}_encoder.pkl"), "rb"))

# load_models()

# # API Endpoints
# @app.get("/")
# def home():
#     return {"message": "Welcome to the Price Prediction API!"}

# @app.post("/predict")
# def predict(input_data: PricePredictionInput):
#     try:
#         # Convert input to DataFrame
#         input_df = pd.DataFrame([input_data.dict()])

#         # Add date features
#         input_df["Arrival_Date"] = pd.to_datetime(input_df["Arrival_Date"])
#         input_df["Year"] = input_df["Arrival_Date"].dt.year
#         input_df["Month"] = input_df["Arrival_Date"].dt.month
#         input_df["Day"] = input_df["Arrival_Date"].dt.day

#         # Encode categorical features
#         for col in ["State", "District", "Market", "Commodity"]:
#             input_df[col] = encoders[col].transform(input_df[col])

#         # Prepare features
#         X_input = input_df[["State", "District", "Market", "Commodity", "Year", "Month", "Day"]]
#         X_input_scaled = scaler.transform(X_input)

#         # Predict prices
#         min_price = rf_min.predict(X_input_scaled)[0]
#         max_price = rf_max.predict(X_input_scaled)[0]
#         modal_price = rf_modal.predict(X_input_scaled)[0]

#         return {
#             "Min_Price": min_price,
#             "Max_Price": max_price,
#             "Modal_Price": modal_price,
#         }

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

# # Analysis Endpoint
# @app.post("/analysis")
# def analysis(input_data: PricePredictionInput):
#     """
#     Perform analysis using the input fields and provide historical and future price predictions.
#     """
#     try:
#         # Convert input to DataFrame
#         input_df = pd.DataFrame([input_data.dict()])

#         # Convert 'Arrival_Date' to datetime
#         input_df["Arrival_Date"] = pd.to_datetime(input_df["Arrival_Date"])

#         # Extract Year, Month, Day from 'Arrival_Date'
#         input_df["Year"] = input_df["Arrival_Date"].dt.year
#         input_df["Month"] = input_df["Arrival_Date"].dt.month
#         input_df["Day"] = input_df["Arrival_Date"].dt.day

#         # Encode categorical features
#         for col in ["State", "District", "Market", "Commodity"]:
#             input_df[col] = encoders[col].transform(input_df[col])

#         # Historical Analysis
#         print(f"Filtering historical data for Commodity: {input_df['Commodity'].iloc[0]} and Market: {input_df['Market'].iloc[0]}")
#         historical_data = data[
#             (data["Commodity"] == encoders["Commodity"].inverse_transform([input_df["Commodity"].iloc[0]])[0]) &
#             (data["Market"] == encoders["Market"].inverse_transform([input_df["Market"].iloc[0]])[0])
#         ]

#         historical_data = historical_data[["Arrival_Date", "Min Price", "Max Price", "Modal Price"]].tail(10)

#         # Future Predictions (next 5 days)
#         future_dates = [input_df["Arrival_Date"].iloc[0] + timedelta(days=i) for i in range(1, 6)]
#         future_data = pd.DataFrame({
#             "State": [input_df["State"].iloc[0]] * len(future_dates),
#             "District": [input_df["District"].iloc[0]] * len(future_dates),
#             "Market": [input_df["Market"].iloc[0]] * len(future_dates),
#             "Commodity": [input_df["Commodity"].iloc[0]] * len(future_dates),
#             "Year": [date.year for date in future_dates],
#             "Month": [date.month for date in future_dates],
#             "Day": [date.day for date in future_dates]
#         })

#         # Scale features for prediction
#         X_future_scaled = scaler.transform(future_data)

#         # Predict prices
#         min_price_pred = rf_min.predict(X_future_scaled)
#         max_price_pred = rf_max.predict(X_future_scaled)
#         modal_price_pred = rf_modal.predict(X_future_scaled)

#         future_predictions = []
#         for i, date in enumerate(future_dates):
#             future_predictions.append({
#                 "Arrival_Date": date.strftime("%d-%m-%Y"),
#                 "Min_Price": min_price_pred[i],
#                 "Max_Price": max_price_pred[i],
#                 "Modal_Price": modal_price_pred[i]
#             })

#         return {
#             "historical_data": historical_data.to_dict(orient="records"),
#             "future_predictions": future_predictions
#         }

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

# if __name__ == '__main__':
#     app.run(debug=True)



from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
import gdown
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths for saving models and encoders
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Input data model
class PricePredictionInput(BaseModel):
    State: str
    District: str
    Market: str
    Commodity: str
    Arrival_Date: str

# Download dataset from Google Drive
def download_dataset():
    dataset_url = "https://drive.google.com/uc?id=YOUR_DATASET_FILE_ID"  # Replace with your dataset file ID
    dataset_path = "data/Price_Agriculture_commodities_Week.csv"
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(dataset_path):
        logger.info("Downloading dataset from Google Drive...")
        try:
            gdown.download(dataset_url, dataset_path, quiet=False)
            logger.info("Dataset downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download dataset: {str(e)}")
            raise Exception(f"Error downloading dataset: {str(e)}")

# Global variables for models and encoders
rf_min = None
rf_max = None
rf_modal = None
scaler = None
encoders = {}
data = None

# Load dataset
def load_dataset():
    global data
    try:
        dataset_path = "data/Price_Agriculture_commodities_Week.csv"
        if os.path.exists(dataset_path):
            logger.info(f"Loading dataset from {dataset_path}")
            # Fix date parsing warning by explicitly setting the format
            data = pd.read_csv(dataset_path)
            data["Arrival_Date"] = pd.to_datetime(data["Arrival_Date"], dayfirst=True)
            logger.info(f"Dataset loaded successfully with {len(data)} rows")
            return data
        else:
            logger.warning("Dataset file not found, attempting to download")
            download_dataset()
            if os.path.exists(dataset_path):
                data = pd.read_csv(dataset_path)
                data["Arrival_Date"] = pd.to_datetime(data["Arrival_Date"], dayfirst=True)
                logger.info(f"Dataset loaded successfully with {len(data)} rows")
                return data
            else:
                raise FileNotFoundError("Dataset file not found after download attempt")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise Exception(f"Error loading dataset: {str(e)}")

# Preprocess the dataset
def preprocess_data(data):
    logger.info("Preprocessing data...")
    # Extract day, month, year from Arrival_Date
    data["Year"] = data["Arrival_Date"].dt.year
    data["Month"] = data["Arrival_Date"].dt.month
    data["Day"] = data["Arrival_Date"].dt.day

    # Encode categorical variables
    encoders = {}
    for col in ["State", "District", "Market", "Commodity"]:
        logger.info(f"Encoding column: {col}")
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        encoders[col] = encoder

    logger.info("Data preprocessing completed successfully")
    return data, encoders

# Download models and encoders from Google Drive
def download_models():
    model_files = {
        "rf_min.pkl": "1TJ0IB81GV5WGu2U-NHVPlXLb40Lrn1I5",
        "rf_max.pkl": "1TQd8D2TINBiXB2NrlqsHYt9_N6LXpP6W",
        "rf_modal.pkl": "1jB3Rfy_dsie3s5lRjiDZnX-e3S718OzX",
        "scaler.pkl": "1ergixNUGsA0knokEkqFACRLTTZo_7-Jv",
        "State_encoder.pkl": "16vxMkxzgU6XEebJq5pTJ9aeJbpEMDtdj",
        "District_encoder.pkl": "1Z9K5ZXW8xHtrnTIVrVjhxjZoLHLIokAl",
        "Market_encoder.pkl": "15HQ_JSxLW-gnMR8yX0jXxu_ug4JVxXdP",
        "Commodity_encoder.pkl": "1FeqOHQRg0Cyx4CxuYmfS3G6xUs3dX1gg",
    }

    for file_name, file_id in model_files.items():
        file_path = os.path.join(MODEL_DIR, file_name)
        if not os.path.exists(file_path):
            logger.info(f"Downloading {file_name} from Google Drive...")
            try:
                gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)
                logger.info(f"Downloaded {file_name} successfully!")
            except Exception as e:
                logger.error(f"Failed to download {file_name}: {str(e)}")
                # Continue with other files instead of raising an exception
                continue

# # Train the models
# def train_models():
#     global scaler, rf_min, rf_max, rf_modal, encoders, data
    
#     logger.info("Training models...")
    
#     if data is None:
#         data = load_dataset()
    
#     # Preprocess data
#     preprocessed_data, encoders = preprocess_data(data)

#     # Features and targets
#     X = preprocessed_data[["State", "District", "Market", "Commodity", "Year", "Month", "Day"]]
#     y_min = preprocessed_data["Min Price"]
#     y_max = preprocessed_data["Max Price"]
#     y_modal = preprocessed_data["Modal Price"]

#     # Split data
#     logger.info("Splitting data into train and test sets...")
#     X_train, X_test, y_train, y_test = train_test_split(X, pd.concat([y_min, y_max, y_modal], axis=1), test_size=0.2, random_state=42)
#     y_train_min, y_train_max, y_train_modal = y_train["Min Price"], y_train["Max Price"], y_train["Modal Price"]
#     y_test_min, y_test_max, y_test_modal = y_test["Min Price"], y_test["Max Price"], y_test["Modal Price"]

#     # Scale features
#     logger.info("Scaling features...")
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # Train models
#     logger.info("Training Random Forest for Min Price...")
#     rf_min = RandomForestRegressor(n_estimators=100, random_state=42)
#     rf_min.fit(X_train_scaled, y_train_min)

#     logger.info("Training Random Forest for Max Price...")
#     rf_max = RandomForestRegressor(n_estimators=100, random_state=42)
#     rf_max.fit(X_train_scaled, y_train_max)

#     logger.info("Training Random Forest for Modal Price...")
#     rf_modal = RandomForestRegressor(n_estimators=100, random_state=42)
#     rf_modal.fit(X_train_scaled, y_train_modal)

#     # Save models and scaler
#     logger.info("Saving models and encoders...")
#     pickle.dump(rf_min, open(os.path.join(MODEL_DIR, "rf_min.pkl"), "wb"))
#     pickle.dump(rf_max, open(os.path.join(MODEL_DIR, "rf_max.pkl"), "wb"))
#     pickle.dump(rf_modal, open(os.path.join(MODEL_DIR, "rf_modal.pkl"), "wb"))
#     pickle.dump(scaler, open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb"))
#     for key, encoder in encoders.items():
#         pickle.dump(encoder, open(os.path.join(MODEL_DIR, f"{key}_encoder.pkl"), "wb"))

#     logger.info("Model training completed successfully")
#     return X_test_scaled, y_test_min, y_test_max, y_test_modal

# Helper function to load models and encoders
def load_models():
    global rf_min, rf_max, rf_modal, scaler, encoders
    
    try:
        logger.info("Attempting to load pre-trained models...")
        rf_min = pickle.load(open(os.path.join(MODEL_DIR, "rf_min.pkl"), "rb"))
        rf_max = pickle.load(open(os.path.join(MODEL_DIR, "rf_max.pkl"), "rb"))
        rf_modal = pickle.load(open(os.path.join(MODEL_DIR, "rf_modal.pkl"), "rb"))
        scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))

        encoders = {}
        for col in ["State", "District", "Market", "Commodity"]:
            encoders[col] = pickle.load(open(os.path.join(MODEL_DIR, f"{col}_encoder.pkl"), "rb"))
        
        logger.info("Models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.info("Will train new models instead")
        return False

# Initialize models and data
@app.on_event("startup")
async def startup_event():
    global data, rf_min, rf_max, rf_modal, scaler, encoders
    
    logger.info("Starting application...")
    
    # Load dataset
    try:
        data = load_dataset()
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        # Will try to train with minimal data if available

    # Try to download models first
    try:
        download_models()
    except Exception as e:
        logger.error(f"Error downloading models: {str(e)}")
    
    # Try to load models
    models_loaded = load_models()
    
    # If models couldn't be loaded, train new ones
    if not models_loaded:
        try:
            logger.info("Training new models...")
            # train_models()
            # Try loading again
            load_models()
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise Exception(f"Could not initialize models: {str(e)}")
    
    logger.info("Application startup completed")

# Helper function to ensure models are loaded
def get_models():
    if rf_min is None or rf_max is None or rf_modal is None or scaler is None:
        raise HTTPException(status_code=500, detail="Models not initialized. Please try again later.")
    return rf_min, rf_max, rf_modal, scaler, encoders

# API Endpoints
@app.get("/")
def home():
    return {"message": "Welcome to the Price Prediction API!", "status": "operational"}

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    if rf_min is None or rf_max is None or rf_modal is None or scaler is None:
        return {"status": "models not loaded"}
    return {"status": "healthy"}

@app.post("/predict")
def predict(input_data: PricePredictionInput, models=Depends(get_models)):
    try:
        rf_min, rf_max, rf_modal, scaler, encoders = models
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Add date features
        input_df["Arrival_Date"] = pd.to_datetime(input_df["Arrival_Date"], dayfirst=True)
        input_df["Year"] = input_df["Arrival_Date"].dt.year
        input_df["Month"] = input_df["Arrival_Date"].dt.month
        input_df["Day"] = input_df["Arrival_Date"].dt.day

        # Encode categorical features
        for col in ["State", "District", "Market", "Commodity"]:
            try:
                input_df[col] = encoders[col].transform(input_df[col])
            except ValueError as e:
                return {"error": f"Value error for {col}: {str(e)}. Available values: {list(encoders[col].classes_)}"}

        # Prepare features
        X_input = input_df[["State", "District", "Market", "Commodity", "Year", "Month", "Day"]]
        X_input_scaled = scaler.transform(X_input)

        # Predict prices
        min_price = rf_min.predict(X_input_scaled)[0]
        max_price = rf_max.predict(X_input_scaled)[0]
        modal_price = rf_modal.predict(X_input_scaled)[0]

        return {
            "Min_Price": float(min_price),
            "Max_Price": float(max_price),
            "Modal_Price": float(modal_price),
        }

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# Analysis Endpoint
@app.post("/analysis")
def analysis(input_data: PricePredictionInput, models=Depends(get_models)):
    """
    Perform analysis using the input fields and provide historical and future price predictions.
    """
    try:
        rf_min, rf_max, rf_modal, scaler, encoders = models
        global data
        
        if data is None:
            data = load_dataset()
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Convert 'Arrival_Date' to datetime
        input_df["Arrival_Date"] = pd.to_datetime(input_df["Arrival_Date"], dayfirst=True)

        # Extract Year, Month, Day from 'Arrival_Date'
        input_df["Year"] = input_df["Arrival_Date"].dt.year
        input_df["Month"] = input_df["Arrival_Date"].dt.month
        input_df["Day"] = input_df["Arrival_Date"].dt.day

        # Encode categorical features
        encoded_values = {}
        for col in ["State", "District", "Market", "Commodity"]:
            try:
                input_df[col] = encoders[col].transform(input_df[col])
                encoded_values[col] = input_df[col].iloc[0]
            except ValueError as e:
                return {"error": f"Value error for {col}: {str(e)}. Available values: {list(encoders[col].classes_)}"}

        # Historical Analysis
        try:
            commodity_name = encoders["Commodity"].inverse_transform([encoded_values["Commodity"]])[0]
            market_name = encoders["Market"].inverse_transform([encoded_values["Market"]])[0]
            
            logger.info(f"Filtering historical data for Commodity: {commodity_name} and Market: {market_name}")
            
            historical_data = data[
                (data["Commodity"] == commodity_name) &
                (data["Market"] == market_name)
            ]

            if len(historical_data) == 0:
                logger.warning(f"No historical data found for Commodity: {commodity_name} and Market: {market_name}")
                historical_records = []
            else:
                historical_data = historical_data[["Arrival_Date", "Min Price", "Max Price", "Modal Price"]].tail(10)
                historical_records = []
                for _, row in historical_data.iterrows():
                    historical_records.append({
                        "Arrival_Date": row["Arrival_Date"].strftime("%d-%m-%Y"),
                        "Min_Price": float(row["Min Price"]),
                        "Max_Price": float(row["Max Price"]),
                        "Modal_Price": float(row["Modal Price"])
                    })
        except Exception as e:
            logger.error(f"Error in historical analysis: {str(e)}")
            historical_records = []

        # Future Predictions (next 5 days)
        future_dates = [input_df["Arrival_Date"].iloc[0] + timedelta(days=i) for i in range(1, 6)]
        future_data = pd.DataFrame({
            "State": [encoded_values["State"]] * len(future_dates),
            "District": [encoded_values["District"]] * len(future_dates),
            "Market": [encoded_values["Market"]] * len(future_dates),
            "Commodity": [encoded_values["Commodity"]] * len(future_dates),
            "Year": [date.year for date in future_dates],
            "Month": [date.month for date in future_dates],
            "Day": [date.day for date in future_dates]
        })

        # Scale features for prediction
        X_future_scaled = scaler.transform(future_data)

        # Predict prices
        min_price_pred = rf_min.predict(X_future_scaled)
        max_price_pred = rf_max.predict(X_future_scaled)
        modal_price_pred = rf_modal.predict(X_future_scaled)

        future_predictions = []
        for i, date in enumerate(future_dates):
            future_predictions.append({
                "Arrival_Date": date.strftime("%d-%m-%Y"),
                "Min_Price": float(min_price_pred[i]),
                "Max_Price": float(max_price_pred[i]),
                "Modal_Price": float(modal_price_pred[i])
            })

        return {
            "historical_data": historical_records,
            "future_predictions": future_predictions
        }

    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
