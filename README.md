# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
1)Import necessary libraries (NumPy, Matplotlib)

2)Load the dataset

3)Calculate the linear trend values using least square method

4)Calculate the polynomial trend values using least square method

5)End the program
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

data = pd.read_csv('/content/astrobiological_activity_monitoring.csv')

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

human_activity = data['Human_Activity_Metrics']
biological_metrics = data['Biological_Metrics']

def linear_trend(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def polynomial_trend(x, y, degree=2):
    coeffs = np.polyfit(x, y, degree)
    return coeffs

x = np.arange(len(data))

m1, c1 = linear_trend(x, human_activity)
m2, c2 = linear_trend(x, biological_metrics)

coeffs1 = polynomial_trend(x, human_activity, degree=2)
coeffs2 = polynomial_trend(x, biological_metrics, degree=2)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(data.index, human_activity, label='Human Activity Metrics', color='blue')
plt.plot(data.index, m1*x + c1, label='Linear Trend', linestyle='--', color='red')
plt.plot(data.index, np.polyval(coeffs1, x), label='Polynomial Trend (Degree 2)', linestyle='-.', color='green')
plt.xlabel('Date')
plt.ylabel('Human Activity Metrics')
plt.title('Human Activity Metrics Trend Estimation')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(data.index, biological_metrics, label='Biological Metrics', color='blue')
plt.plot(data.index, m2*x + c2, label='Linear Trend', linestyle='--', color='red')
plt.plot(data.index, np.polyval(coeffs2, x), label='Polynomial Trend (Degree 2)', linestyle='-.', color='green')
plt.xlabel('Date')
plt.ylabel('Biological Metrics')
plt.title('Biological Metrics Trend Estimation')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

```

### OUTPUT
![image](https://github.com/user-attachments/assets/42fde478-0c57-4335-99d9-c01651a11e11)

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
