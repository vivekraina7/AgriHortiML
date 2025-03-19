from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS for cross-origin resource sharing
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained models and the scaler
try:
    rf_min = pickle.load(open('rf_min_model.pkl', 'rb'))
    rf_max = pickle.load(open('rf_max_model.pkl', 'rb'))
    rf_modal = pickle.load(open('rf_modal_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
except Exception as e:
    print(f"Error loading models: {e}")
    rf_min = rf_max = rf_modal = scaler = label_encoder = None

# Load the dataset for historical data
try:
    df = pd.read_csv('data/Price_Agriculture_commodities_Week.csv')
    df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], format='%d-%m-%Y')
except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame()  # Empty DataFrame if file not loaded

@app.route('/')
def home():
    return jsonify({'message': 'API is running successfully!'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON input from the request
        data = request.get_json()

        # Prepare the input data
        input_data = pd.DataFrame(data)
        input_data['Year'] = pd.to_datetime(input_data['Arrival_Date']).dt.year
        input_data['Month'] = pd.to_datetime(input_data['Arrival_Date']).dt.month
        input_data['Day'] = pd.to_datetime(input_data['Arrival_Date']).dt.day

        # Encoding categorical features
        input_data['State'] = label_encoder.transform(input_data['State'])
        input_data['District'] = label_encoder.transform(input_data['District'])
        input_data['Market'] = label_encoder.transform(input_data['Market'])
        input_data['Commodity'] = label_encoder.transform(input_data['Commodity'])

        # Prepare features for prediction
        X_input = input_data[['State', 'District', 'Market', 'Commodity', 'Year', 'Month', 'Day']]
        X_input_scaled = scaler.transform(X_input)

        # Predict Min, Max, and Modal prices
        min_price = rf_min.predict(X_input_scaled)
        max_price = rf_max.predict(X_input_scaled)
        modal_price = rf_modal.predict(X_input_scaled)

        # Prepare the output
        output = {
            'Min Price': min_price[0],
            'Max Price': max_price[0],
            'Modal Price': modal_price[0]
        }

        return jsonify(output)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/analysis', methods=['POST'])
def analysis():
    try:
        data = request.get_json()

        # Extract input fields
        state = data.get('State')
        district = data.get('District')
        market = data.get('Market')
        commodity = data.get('Commodity')
        variety = data.get('Variety')
        arrival_date = data.get('Arrival_Date')
        min_price = data.get('Min_Price')
        max_price = data.get('Max_Price')

        # Validate input fields
        if not all([state, district, market, commodity, variety, arrival_date, min_price, max_price]):
            return jsonify({'error': 'Missing required fields (State, District, Market, Commodity, Variety, Arrival_Date)'}), 400

        # Convert Arrival_Date to datetime
        arrival_date = datetime.strptime(arrival_date, '%d-%m-%Y')

        # Filter historical data from the dataset
        historical_data = df[(df['Commodity'] == commodity) & (df['Market'] == market)]
        if historical_data.empty:
            return jsonify({'error': 'No historical data found for the given inputs.'}), 404

        # Generate predictions for the next 5 days, dynamically changing the Arrival_Date
        future_dates = [arrival_date + timedelta(days=i) for i in range(1, 6)]
        prediction_data = pd.DataFrame({
            'State': [state] * len(future_dates),
            'District': [district] * len(future_dates),
            'Market': [market] * len(future_dates),
            'Commodity': [commodity] * len(future_dates),
            'Variety': [variety] * len(future_dates),
            'Arrival_Date': [date.strftime('%Y-%m-%d') for date in future_dates],
            'Min_Price': [min_price] * len(future_dates),
            'Max_Price': [max_price] * len(future_dates)
        })

        # Preprocess the prediction data
        prediction_data['Year'] = pd.to_datetime(prediction_data['Arrival_Date']).dt.year
        prediction_data['Month'] = pd.to_datetime(prediction_data['Arrival_Date']).dt.month
        prediction_data['Day'] = pd.to_datetime(prediction_data['Arrival_Date']).dt.day

        # Encoding categorical features
        prediction_data['State'] = label_encoder.transform(prediction_data['State'])
        prediction_data['District'] = label_encoder.transform(prediction_data['District'])
        prediction_data['Market'] = label_encoder.transform(prediction_data['Market'])
        prediction_data['Commodity'] = label_encoder.transform(prediction_data['Commodity'])

        # Prepare features for prediction
        X_input = prediction_data[['State', 'District', 'Market', 'Commodity', 'Year', 'Month', 'Day']]
        X_input_scaled = scaler.transform(X_input)

        # Predict Min, Max, and Modal prices
        min_price_pred = rf_min.predict(X_input_scaled)
        max_price_pred = rf_max.predict(X_input_scaled)
        modal_price_pred = rf_modal.predict(X_input_scaled)

        # Prepare the output for future predictions
        future_output = []
        for i in range(len(future_dates)):
            future_output.append({
                'Arrival_Date': future_dates[i].strftime('%d-%m-%Y'),
                'Min Price': min_price_pred[i],
                'Max Price': max_price_pred[i],
                'Modal Price': modal_price_pred[i]
            })

        # Prepare the historical data to return
        historical_data = historical_data[['Arrival_Date']]

        return jsonify({
            'historical_data': historical_data.to_dict(orient='records'),
            'future_predictions': future_output
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)