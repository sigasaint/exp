# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 16:05:25 2024

@author: Crazy_Papi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
#import matplotlib.animation as animation
import plotly.express as px

df = pd.read_excel('population_data.xlsx')

def exponential_growth(x, a, b):
    return a * np.exp(b * x)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df['Year'].values.reshape(-1, 1))
y = df['Population'].values

popt, _ = curve_fit(exponential_growth, X_scaled.flatten(), y)

X_new = np.array([2025, 2030, 2035])
X_new_scaled = scaler.transform(X_new.reshape(-1, 1))
X_new_scaled = X_new_scaled.flatten()
y_pred = exponential_growth(X_new_scaled, *popt)

plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='Population', data=df)
plt.title('Population Growth Over Time')
plt.xlabel('Year')
plt.ylabel('Population')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(x=df['Year'], y=df['Population'], label='Historical Data')
sns.lineplot(x=X_new, y=y_pred, label='Predicted Growth')
plt.title('Population Growth Prediction')
plt.xlabel('Year')
plt.ylabel('Population')
plt.grid(True)
plt.legend()
plt.show()

# Create animation for historical data
historical_fig = px.line(df, x='Year', y='Population', title='Historical Population Growth')

historical_frames = []
for i in range(len(df)):
    historical_frame = dict(data=[dict(x=df['Year'][:i+1], y=df['Population'][:i+1], mode='lines', name='Historical Data')],
                            traces=[0])
    historical_frames.append(historical_frame)

historical_fig.frames = historical_frames

historical_fig.update_layout(updatemenus=[dict(type='buttons', buttons=[dict(label='Play', method='animate', args=[None])])])

historical_fig.write_html("historical_population_growth_animation.html")

# Create animation for predicted data
predicted_fig = px.line(x=X_new, y=y_pred, title='Predicted Population Growth')

predicted_frames = []
for i in range(len(X_new)):
    predicted_frame = dict(data=[dict(x=X_new[:i+1], y=y_pred[:i+1], mode='lines', line=dict(dash='dash'), name='Predicted Growth')],
                           traces=[0])
    predicted_frames.append(predicted_frame)

predicted_fig.frames = predicted_frames

predicted_fig.update_layout(updatemenus=[dict(type='buttons', buttons=[dict(label='Play', method='animate', args=[None])])])

predicted_fig.write_html("predicted_population_growth_animation.html")

                 
# =============================================================================
# fig , ax = plt.subplots(figsize=(10,6))
# def animate(i):
#     ax.clear()
#     sns.lineplot(x='Year',y='Population',data=df,ax=ax)
#     ax.set_title('Population Growth Over Time')
#     ax.set_xlabel('Year')
#     ax.set_ylabel('Population')
#     ax.grid(True)
#     ax.set_xlim([df['Year'].min(),X_new.max()])
#     ax.set_ylim([df['Population'].min(),max(y_pred)*1.1])
#     ax.plot(X_new[:i],y_pred[:i],'r--',label='Predicted Growth')
#     ax.legend
# ani = animation.FuncAnimation(fig, animate, frames=len(X_new), interval = 1000)
# plt.show()
# ani.save('ppd.gif',writer='pillow')
# =============================================================================

