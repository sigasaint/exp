
import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Define the number of samples
n_samples = 1000

# Generate random data for features
bedrooms = np.random.randint(1, 6, n_samples)
yard_size = np.random.uniform(500, 5000, n_samples)
features = np.random.choice(['pool', 'gym', 'views', 'none'], n_samples)
modern = np.random.choice([0, 1], n_samples)  # 0 = old, 1 = modern

# Generate random data for target variable (price)
price = np.random.uniform(200000, 1000000, n_samples)

# Create a pandas DataFrame
df = pd.DataFrame({
    'bedrooms': bedrooms,
    'yard_size': yard_size,
    'features': features,
    'modern': modern,
    'price': price
})

# Save the dataset to a CSV file
df.to_csv('property_data.csv', index=False)