
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px
import numpy as np

# Load data (replace with your dataset)
df = pd.read_csv('property_data.csv')

# Preprocess data (handle missing values, convert categorical variables, etc.)
df = df.dropna()  # Drop rows with missing values
df['features'] =pd.Categorical(df['features'])
ohe = OneHotEncoder(sparse=False,dtype=np.float64)
df['features'] = ohe.fit_transform(df[['features']].values.reshape(-1,1))  # One-hot encoding

X = df.drop(['price'], axis=1)  # Features
y = df['price']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse:.2f}')

# Create a scatter plot of actual vs. predicted prices
fig = px.scatter(x=y_test, y=y_pred, title='Actual vs. Predicted Prices')
fig.write_html('prediction.html')