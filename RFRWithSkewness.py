import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import math
from scipy.stats import skew

import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/Zubair Ahmed/PycharmProjects/HousePriceRegression/train.csv', sep='\s*,\s*', encoding="utf-8-sig", engine='python')
tdf = pd.read_csv('C:/Users/Zubair Ahmed/PycharmProjects/HousePriceRegression/test.csv', sep='\s*,\s*', encoding="utf-8-sig", engine='python')

salePrice = df['SalePrice']

# df = df.set_index('Id')
# tdf = tdf.set_index('Id')

cols = ['OverallQual','GrLivArea','1stFlrSF','TotalBsmtSF','YearBuilt','YearRemodAdd','TotRmsAbvGrd','SalePrice']
xcols = ['OverallQual','GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF','YearBuilt',
         'TotRmsAbvGrd','YearRemodAdd','BsmtFinSF1','LotArea','FullBath','HalfBath']
df.fillna(df.mean(), inplace=True)
TotalBsmtSFMean = df['TotalBsmtSF'].mean()
df.loc[df['TotalBsmtSF'] == 0, 'TotalBsmtSF'] = np.round(TotalBsmtSFMean).astype(int)

tdf.fillna(tdf.mean(), inplace=True)
TTotalBsmtSFMean = tdf['TotalBsmtSF'].mean()
tdf.loc[tdf['TotalBsmtSF'] == 0, 'TotalBsmtSF'] = np.round(TotalBsmtSFMean).astype(int)

# X = df[xcols]
y = df['SalePrice']
df = df.drop('SalePrice', axis=1)
X = df
# tX = tdf[xcols]
tX = tdf

# print  X.columns.tolist()
# print  "*****************************"
# print tX.columns.tolist()

train_num = len(X)
dataset = pd.concat(objs=[X, tX], axis=0)

#log transform the target:
# y = np.log1p(y)

#log transform skewed numeric features:
numeric_feats = dataset.dtypes[dataset.dtypes != "object"].index

skewed_feats = X[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

dataset[skewed_feats] = np.log1p(dataset[skewed_feats])

dataset_preprocessed = pd.get_dummies(dataset)
train_preprocessed = dataset_preprocessed[:train_num]
test_preprocessed = dataset_preprocessed[train_num:]
#
# print train_preprocessed.columns[pd.isnull(train_preprocessed).sum() > 0].tolist()
# print test_preprocessed.columns[pd.isnull(test_preprocessed).sum() > 0].tolist()

std = StandardScaler()
X_std = std.fit_transform(train_preprocessed)
X_std_test = std.fit_transform(test_preprocessed)

X_train, X_test, y_train, y_test = train_test_split(train_preprocessed, y, test_size=0.3, random_state=0)

rmse_est = {}
for est in range(360,550,20):
    model = RandomForestRegressor(n_estimators=est, n_jobs=-1)
    model.fit(X_train, y_train)
    predictions = np.array(model.predict(X_test))
    rmse = math.sqrt(np.mean((np.array(y_test) - predictions)**2))
    imp = sorted(zip(X.columns, model.feature_importances_), key=lambda tup: tup[1], reverse=True)
    print "RMSE: {0} - est: {1}".format(str(rmse), est)
    print "10 Most Important Variables:" + str(imp[:10])
    rmse_est[rmse]= est
    # if rmse < 29657:
    #     print 'generating file'
    #     y_test_pred = model.predict(test_preprocessed)
    #     submission = pd.DataFrame({"Id": test_preprocessed["Id"],"SalePrice": y_test_pred})
    #     submission.loc[submission['SalePrice'] <= 0, 'SalePrice'] = 0
    #     fileName = "submission_{0}_.csv".format(rmse)
    #     submission.to_csv(fileName, index=False)
import collections
d = collections.OrderedDict(sorted(rmse_est.items()))
# print d.items()[0][1]
# exit()
print 'generating file'
model = RandomForestRegressor(n_estimators=d.items()[0][1], n_jobs=-1)
model.fit(train_preprocessed, y)
y_test_pred = model.predict(test_preprocessed)
submission = pd.DataFrame({"Id": test_preprocessed["Id"],"SalePrice": y_test_pred})
submission.loc[submission['SalePrice'] <= 0, 'SalePrice'] = 0
fileName = "submission.csv".format(rmse)
submission.to_csv(fileName, index=False)

