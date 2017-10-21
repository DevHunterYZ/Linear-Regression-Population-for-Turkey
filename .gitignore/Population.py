 import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import numpy as np

# allow plots to appear directly in the notebook
%matplotlib inline
headers = ['Year','Population']
data = pd.read_csv('C:/Users/user/Desktop/population.csv', delim_whitespace=True, names=headers )
print (data)
