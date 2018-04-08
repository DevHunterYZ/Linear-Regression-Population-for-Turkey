#Kütüphaneleri çağıralım.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#Veriyi tanıtalım.
data = pd.read_csv('Population for Turkey.csv')
data.head()
print(data)
X = data['Year'].values[:,np.newaxis]
y = data['Population for Turkey'].values
#Modeli oluşturalım.
model = LinearRegression()
#Modeli eğitelim.
model.fit(X, y)
#Çizdirelim.
plt.scatter(X, y,color='r')
plt.plot(X, model.predict(X),color='k')
plt.show()
