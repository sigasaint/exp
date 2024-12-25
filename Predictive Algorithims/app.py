
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# Load the dataset
df = pd.read_csv('property_data.csv')

# Define a LabelEncoder
le = LabelEncoder()

# Apply label encoding to categorical variables
df['features'] = le.fit_transform(df['features'])
df['location'] = le.fit_transform(df['location'])

# Split data into training and testing sets
X = df.drop(['price'], axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Define a function to make predictions with label encoding
def predict_property_price(bedrooms, yard_size, features, modern, location):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'bedrooms': [bedrooms],
        'yard_size': [yard_size],
        'features': [features],
        'modern': [modern],
        'location': [location]
    })

    # Apply label encoding to categorical variables
    input_data['features'] = le.transform(input_data['features'])
    input_data['location'] = le.transform(input_data['location'])

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
