


import numpy as np
import seaborn as sns
import re
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt

import matplotlib; matplotlib.style.use('ggplot')
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import math
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FactorAnalysis
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

df = pd.read_csv('C:/Users/Zubair Ahmed/PycharmProjects/HousePriceRegression/train.csv')
tdf = pd.read_csv('C:/Users/Zubair Ahmed/PycharmProjects/HousePriceRegression/test.csv', sep='\s*,\s*', encoding="utf-8-sig", engine='python')

salePrice = df['SalePrice']

train = df
test = tdf

train_num = len(train)

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())

X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
print 'generating file'
y_test_pred = model_lasso.predict(X_test)
submission = pd.DataFrame({"Id": test["Id"],"SalePrice": y_test_pred})
submission.loc[submission['SalePrice'] <= 0, 'SalePrice'] = 0
fileName = "submission_.csv"
submission.to_csv(fileName, index=False)
