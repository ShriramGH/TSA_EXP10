## DEVELOPED BY: SHRIRAM S
## REGISTER NO: 212222240098
## DATE:
# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('/content/prices.csv')


data['Price Dates'] = pd.to_datetime(data['Price Dates'], format='%d-%m-%Y') 
data.set_index('Price Dates', inplace=True)

# Plot the time series data
plt.plot(data.index, data['Methi'])
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Price Time Series')
plt.show()

# Function to check stationarity using ADF test
def check_stationarity(timeseries):
    # Drop missing values before applying ADF test
    timeseries = timeseries.dropna()
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

# Drop missing values from the 'pH' column before passing it to check_stationarity
check_stationarity(data['Methi'].dropna())

# Plot ACF and PACF to determine SARIMA parameters
plot_acf(data['Methi'])
plt.show()
plot_pacf(data['Methi'])
plt.show()

# Train-test split (80% train, 20% test)
train_size = int(len(data) * 0.8)
train, test = data['Methi'][:train_size], data['Methi'][train_size:]

# Define and fit the SARIMA model on training data
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Make predictions on the test set
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot the actual vs predicted values
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('SARIMA Model Predictions')
plt.legend()
plt.show()

```
### OUTPUT:
![image](https://github.com/user-attachments/assets/caf04793-2bd1-41f7-a6a8-da2ace4ba767)

![image](https://github.com/user-attachments/assets/f61e9919-e7fe-44df-a3dc-0b7263f91045)

![image](https://github.com/user-attachments/assets/f5101667-7c98-4989-8376-83abd0fbda38)
![image](https://github.com/user-attachments/assets/e4d1a471-f43e-461d-ba5d-7ec01a3ca770)
![image](https://github.com/user-attachments/assets/91758e75-fbed-4b1f-8f33-89918db512bb)



### RESULT:
Thus, the pyhton program based on the SARIMA model is executed successfully.
