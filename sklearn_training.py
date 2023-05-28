#!/usr/bin/env python
# coding: utf-8

# In[99]:


import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]


# In[105]:


data


# In[106]:


x_data = data
y_data = target.reshape(target.size, 1)
y_data.shape


# In[107]:


from sklearn import preprocessing

minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 5)).fit(x_data) # x_data를 0~5 사이의 값으로 정규화함
x_scaled_data = minmax_scale.transform(x_data)
x_scaled_data[:3]


# In[108]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled_data, y_data, test_size=0.33)

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[109]:


from sklearn import linear_model
regr = linear_model.LinearRegression(fit_intercept=True, copy_X=True, n_jobs=8)
lasso_regr = linear_model.Lasso(alpha=0.01, fit_intercept=True, copy_X=True)
ridge_regr = linear_model.Ridge(alpha=0.01, fit_intercept=True, copy_X=True)
SGD__regr = linear_model.SGDRegressor(penalty="l2", alpha=0.01, max_iter=1000, tol=0.001, eta0=0.01)


# In[129]:


regr.fit(x_train, y_train)
lasso_regr.fit(x_train, y_train)
ridge_regr.fit(x_train, y_train)
SGD__regr.fit(x_train, y_train)


# In[119]:


print("Coefficients : ", regr.coef_)
print("intercept : ", regr.intercept_)


# In[147]:


print(regr.predict(x_data[:5]))
print(lasso_regr.predict(x_data[:5]))
print(ridge_regr.predict(x_data[:5]))
print(SGD__regr.predict(x_data[:5]))


# In[123]:


x_data[:5].dot(regr.coef_.T) + regr.intercept_


# In[126]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

y_true = y_test.copy()
y_hat = regr.predict(x_test)

r2_score(y_true, y_hat), mean_absolute_error(y_true, y_hat), mean_squared_error(y_true, y_hat)


# In[128]:


import matplotlib.pyplot as plt

plt.scatter(y_true, y_hat, s=10)
plt.xlabel("Prices; $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ cs $\hat{Y}_i$")

