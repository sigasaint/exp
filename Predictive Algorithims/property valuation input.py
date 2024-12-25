import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the pre-trained model (assuming you've trained one)
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Define a function to make predictions
def predict_property_price(bedrooms, yard_size, features, modern, location):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'bedrooms': [bedrooms],
        'yard_size': [yard_size],
        'features': [features],
        'modern': [modern],
        'location': [location]
    })

    # Convert categorical variables
    input_data['features'] = pd.Categorical(input_data['features'])
    input_data['location'] = pd.Categorical(input_data['location'])

    # One-hot encoding
    input_data = pd.get_dummies(input_data, columns=['features', 'location'])

    # Make predictions
    prediction = model.predict(input_data)

    return prediction[0]

# Example usage:
bedrooms = int(input("Enter number of bedrooms: "))
yard_size = float(input("Enter yard size (sqft): "))
features = input("Enter additional features (pool, gym, views, or none): ")
modern = int(input("Is the property modern? (0 = old, 1 = modern): "))
location = input("Enter location (city, suburb, or rural): ")

predicted_price = predict_property_price(bedrooms, yard_size, features, modern, location)
print(f"Predicted property price: ${predicted_price:.2f}")