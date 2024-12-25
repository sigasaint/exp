# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 16:00:40 2024

@author: Crazy_Papi
"""

import pandas as pd

data = {
    "Year": [1980, 1990, 2000, 2010, 2020, 2022],
    "Population": [7570672, 10038000, 11631657, 12973808, 14438802, 15178957],
    "Birth Rate (per 1,000)": [43.8, 38.5, 34.2, 32.2, 30.4, 29.5],
    "Death Rate (per 1,000)": [12.5, 10.3, 21.1, 12.3, 9.5, 9.2],
    "GDP (USD billion)": [2.1, 6.3, 5.5, 10.4, 14.1, 15.5],
    "Life Expectancy (years)": [56.1, 59.5, 44.1, 51.1, 61.1, 62.2],
    "Immigration": [5000, 10000, 20000, 30000, 40000, 45000],
    "Emigration": [10000, 20000, 30000, 40000, 50000, 55000]
}

df = pd.DataFrame(data)

# Save to Excel file
df.to_excel("population_data.xlsx", index=True,)
