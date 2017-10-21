import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data = pd.read_csv('C:/Users/user/Desktop/population2.csv')
data.head()
print(data)
X = data['Year'].values[:,np.newaxis]

y = data['Population for Turkey'].values

model = LinearRegression()

model.fit(X, y)
plt.scatter(X, y,color='r')

plt.plot(X, model.predict(X),color='k')

plt.show()
