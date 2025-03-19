from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import joblib
import os
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
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

# Paths for models and encoders
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Input data model
class PricePredictionInput(BaseModel):
    State: str
    District: str
    Market: str
    Commodity: str
    Arrival_Date: str

# Global variables for models and encoders
rf_min = None
rf_max = None
rf_modal = None
scaler = None
encoders = {}
data = None

def display_paths():
    """Display the current working directory and model paths for debugging."""
    # Get current working directory
    current_dir = os.getcwd()
    
    # Get absolute path to the model directory
    model_dir = os.path.abspath(MODEL_DIR)
    
    # Check if model directory exists
    model_dir_exists = os.path.exists(model_dir)
    
    # List all files in the model directory if it exists
    model_files = os.listdir(model_dir) if model_dir_exists else []
    
    # Get all model paths
    model_paths = {
        'rf_min': os.path.join(model_dir, "rf_min.pkl"),
        'rf_max': os.path.join(model_dir, "rf_max.pkl"),
        'rf_modal': os.path.join(model_dir, "rf_modal.pkl"),
        'scaler': os.path.join(model_dir, "scaler.pkl"),
        'State_encoder': os.path.join(model_dir, "State_encoder.pkl"),
        'District_encoder': os.path.join(model_dir, "District_encoder.pkl"),
        'Market_encoder': os.path.join(model_dir, "Market_encoder.pkl"),
        'Commodity_encoder': os.path.join(model_dir, "Commodity_encoder.pkl"),
    }
    
    # Check if each model file exists
    model_exists = {name: os.path.exists(path) for name, path in model_paths.items()}
    
    # Return all paths and information
    return {
        'current_directory': current_dir,
        'model_directory': model_dir,
        'model_dir_exists': model_dir_exists,
        'model_files': model_files,
        'model_paths': model_paths,
        'model_files_exist': model_exists
    }

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
            logger.warning("Dataset file not found. Proceeding without dataset.")
            return None
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None

def load_models():
    global rf_min, rf_max, rf_modal, scaler, encoders
    
    model_dir = "/opt/render/project/src/models"
    logger.info(f"Loading models from specific path: {model_dir}")
    
    try:
        # Use compatibility options when loading with pickle
        with open(os.path.join(model_dir, "rf_min.pkl"), "rb") as f:
            rf_min = pickle.load(f, encoding='latin1', fix_imports=True)
        with open(os.path.join(model_dir, "rf_max.pkl"), "rb") as f:
            rf_max = pickle.load(f, encoding='latin1', fix_imports=True)
        with open(os.path.join(model_dir, "rf_modal.pkl"), "rb") as f:
            rf_modal = pickle.load(f, encoding='latin1', fix_imports=True)
        with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f, encoding='latin1', fix_imports=True)

        encoders = {}
        for col in ["State", "District", "Market", "Commodity"]:
            with open(os.path.join(model_dir, f"{col}_encoder.pkl"), "rb") as f:
                encoders[col] = pickle.load(f, encoding='latin1', fix_imports=True)
        
        logger.info("Models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        paths_info = display_paths()
        logger.error(f"Model paths: {paths_info['model_paths']}")
        logger.error(f"Model existence: {paths_info['model_files_exist']}")
        return False
        
# Initialize models and data
@app.on_event("startup")
async def startup_event():
    global data, rf_min, rf_max, rf_modal, scaler, encoders
    
    logger.info("Starting application...")
    
    # Display paths for debugging
    paths_info = display_paths()
    logger.info(f"Current directory: {paths_info['current_directory']}")
    logger.info(f"Model directory: {paths_info['model_directory']}")
    logger.info(f"Model directory exists: {paths_info['model_dir_exists']}")
    
    # Load dataset
    data = load_dataset()
    
    # Load the models
    models_loaded = load_models()
    if not models_loaded:
        logger.error("Failed to load models. Please ensure model files are in the correct location.")
    
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

@app.get("/paths")
def get_paths():
    """Endpoint to display the current path and model paths for debugging."""
    paths_info = display_paths()
    return paths_info

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
        historical_records = []
        if data is not None:
            try:
                commodity_name = encoders["Commodity"].inverse_transform([encoded_values["Commodity"]])[0]
                market_name = encoders["Market"].inverse_transform([encoded_values["Market"]])[0]
                
                logger.info(f"Filtering historical data for Commodity: {commodity_name} and Market: {market_name}")
                
                historical_data = data[
                    (data["Commodity"] == commodity_name) &
                    (data["Market"] == market_name)
                ]

                if len(historical_data) > 0:
                    historical_data = historical_data[["Arrival_Date", "Min Price", "Max Price", "Modal Price"]].tail(10)
                    for _, row in historical_data.iterrows():
                        historical_records.append({
                            "Arrival_Date": row["Arrival_Date"].strftime("%d-%m-%Y"),
                            "Min_Price": float(row["Min Price"]),
                            "Max_Price": float(row["Max Price"]),
                            "Modal_Price": float(row["Modal Price"])
                        })
                else:
                    logger.warning(f"No historical data found for Commodity: {commodity_name} and Market: {market_name}")
            except Exception as e:
                logger.error(f"Error in historical analysis: {str(e)}")

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
