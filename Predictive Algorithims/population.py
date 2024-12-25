
import pandas as pd
import numpy as np
from scipy import optimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.DataFrame({
    "Year": [1980, 1990, 2000, 2010, 2020, 2022],
    "Population": [7570672, 10038000, 11631657, 12973808, 14438802, 15000000],
    "Birth Rate (per 1,000)": [43.8, 38.5, 34.2, 32.2, 30.4, 29.5],
    "Death Rate (per 1,000)": [12.5, 10.3, 21.1, 12.3, 9.5, 9.2],
    "GDP (USD billion)": [2.1, 6.3, 5.5, 10.4, 14.1, 15.5],
    "Life Expectancy (years)": [56.1, 59.5, 44.1, 51.1, 61.1, 62.2],
    "Immigration": [5000, 10000, 20000, 30000, 40000, 45000],
    "Emigration": [10000, 20000, 30000, 40000, 50000, 55000]
})

# Check for missing values
print(df.isnull().sum())

# Handle missing values (if necessary)
df.fillna(method='ffill', inplace=True)

# Convert data types (if necessary)
df['Year'] = pd.to_datetime(df['Year'])

# Explore data distribution and summary statistics
print(df.describe())

# Plot population growth
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='Population' / 1e6, data=df)
plt.title('Population Growth Over Time')
plt.show()

# Extract input (X) and output (y) variables
X = df['Year'].dt.year
y = df['Population']

# Scale or normalize data (if necessary)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values.reshape(-1, 1))

# Exponential growth model
def exponential_growth(x, a, b, c):
    """
    Exponential growth model: y = a * exp(b * x) + c
    """
    return a * np.exp(b * x) + c

# Logistic growth model
def logistic_growth(x, a, b, c, d):
    """
    Logistic growth model: y = a / (1 + exp(-b * (x - c))) + d
    """
    return a / (1 + np.exp(-np.clip(b * (x - c), -100, 100))) + d

# Fit exponential growth model
popt_exp, _ = optimize.curve_fit(exponential_growth, X_scaled.flatten(), y, method="trf")

# Fit logistic growth model
popt_log, _ = optimize.curve_fit(logistic_growth, X_scaled.flatten(), y, method="trf")

# Evaluate exponential growth model
y_pred_exp = exponential_growth(X_scaled, *popt_exp)
mae_exp = mean_absolute_error(y, y_pred_exp)
mse_exp = mean_squared_error(y, y_pred_exp)
r2_exp = r2_score(y, y_pred_exp)
print("Exponential Growth Model:")
print(f"MAE: {mae_exp:.2f}, MSE: {mse_exp:.2f}, R2: {r2_exp:.2f}")

# Evaluate logistic growth model
y_pred_log = logistic_growth(X_scaled, *popt_log)
mae_log = mean_absolute_error(y, y_pred_log)
mse_log = mean_squared_error(y, y_pred_log)
r2_log = r2_score(y, y_pred_log)
print("Logistic Growth Model:")
print(f"MAE: {mae_log:.2f}, MSE: {mse_log:.2f}, R2: {r2_log:.2f}")

# Make predictions using the best-performing model
X_new = np.array([2025, 2030, 2035])
X_new_scaled = scaler.transform(X_new.reshape(-1, 1))
X_New = X_new_scaled.flatten()
y_pred = exponential_growth(X_new_scaled, *popt_exp)
y_pred= y_pred.flatten() / 1e6

# Visualize predicted population growth
plt.figure(figsize=(10, 6))
sns.lineplot(x=X, y=y / 1e6, label='Historical Data')
sns.lineplot(x=X_new, y=y_pred, label='Predicted Growth')
plt.title('Population Growth Prediction')
plt.show()
