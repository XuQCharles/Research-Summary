import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

# Load the dataset
file_path = r'C:\Users\CharlesQX\Desktop\python\ResearchProjectData\danmu_analysis.csv'
df = pd.read_csv(file_path)

# Filter data points with "Number of Danmu" less than 200
df_filtered = df[df['Number of Danmu'] >= 50]

# Assuming your CSV has columns named 'Number of Danmu' and 'Percentage of Negative Comments'
x = df_filtered['Number of Danmu'].values
y = df_filtered['Percentage of Negative Comments'].values

# Define functions for various types of regression
def inverse_func(x, a, b):
    return a / x + b

def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

def square_root_func(x, a, b):
    return a * np.sqrt(x) + b

def logarithmic_func(x, a, b):
    return a * np.log(x) + b

# Fit various regression models
models = {
    'linear': LinearRegression(),
    'quadratic': np.polyfit(x, y, 2),
    'inverse': curve_fit(inverse_func, x, y)[0],
    'exponential': curve_fit(exponential_func, x, y, p0=(1, 1e-6, 1))[0],
    'square_root': curve_fit(square_root_func, x, y)[0],
    'logarithmic': curve_fit(logarithmic_func, x, y)[0]
}

# Evaluate R-squared for each model
r2_values = {}

for model_name, model in models.items():
    if model_name == 'linear':
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        y_pred = model.predict(x.reshape(-1, 1))
    elif model_name == 'quadratic':
        coeffs = model
        y_pred = np.polyval(coeffs, x)
    else:
        params = model
        y_pred = eval(f"{model_name}_func(x, *params)")
    
    r2 = r2_score(y, y_pred)
    r2_values[model_name] = r2

# Plot the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', alpha=0.5)
plt.title('Scatter Plot of Number of Danmu vs. Percentage of Negative Comments')
plt.xlabel('Number of Danmu')
plt.ylabel('Percentage of Negative Comments')
plt.grid(True)

# Display R-squared values for each model
for model_name, r2 in r2_values.items():
    plt.text(0.1, 0.9 - list(r2_values.keys()).index(model_name) * 0.05, f'{model_name.capitalize()}: R-squared = {r2:.2f}', transform=plt.gca().transAxes)

plt.show()
