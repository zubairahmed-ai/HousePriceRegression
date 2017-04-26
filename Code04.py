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
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/Zubair Ahmed/PycharmProjects/HousePriceRegression/train.csv', sep='\s*,\s*', encoding="utf-8-sig", engine='python')
tdf = pd.read_csv('C:/Users/Zubair Ahmed/PycharmProjects/HousePriceRegression/test.csv', sep='\s*,\s*', encoding="utf-8-sig", engine='python')

salePrice = df['SalePrice']

df = df.set_index('Id')
tdf = tdf.set_index('Id')

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
dataset_preprocessed = pd.get_dummies(dataset)
train_preprocessed = dataset_preprocessed[:train_num]
test_preprocessed = dataset_preprocessed[train_num:]
#
# print train_preprocessed.columns[pd.isnull(train_preprocessed).sum() > 0].tolist()
# print test_preprocessed.columns[pd.isnull(test_preprocessed).sum() > 0].tolist()

std = StandardScaler()
slr = ElasticNet(alpha=0.6, l1_ratio=0.5)
X_std = std.fit_transform(train_preprocessed)
X_std_test = std.fit_transform(test_preprocessed)

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=0)

for est in range(360,550,20):
    model = RandomForestRegressor(n_estimators=est, n_jobs=-1)
    model.fit(X_train, y_train)
    predictions = np.array(model.predict(X_test))
    rmse = math.sqrt(np.mean((np.array(y_test) - predictions)**2))
    imp = sorted(zip(X.columns, model.feature_importances_), key=lambda tup: tup[1], reverse=True)
    print "RMSE: {0} - est: {1}".format(str(rmse), est)
    print "10 Most Important Variables:" + str(imp[:10])
    if rmse < 29657:
        print 'generating file'
        y_test_pred = model.predict(X_std_test)
        submission = pd.DataFrame({"Id": tdf["Id"],"SalePrice": y_test_pred})
        submission.loc[submission['SalePrice'] <= 0, 'SalePrice'] = 0
        fileName = "submission_{0}_.csv".format(rmse)
        submission.to_csv(fileName, index=False)

# RMSE: 31561.2160876 - est: 360
# 10 Most Important Variables:[(u'LotArea', 0.54402599126018425), (u'HouseStyle', 0.12569572208668497), (u'Utilities', 0.035040099031639439), (u'Neighborhood', 0.028979297738794062), (u'MasVnrArea', 0.02817545744438114), (u'Condition1', 0.026073970501780813), (u'ExterQual', 0.022750830379986951), (u'LotFrontage', 0.015373014928081057), (u'Alley', 0.013977682356939912), (u'LandContour', 0.013015156623021556)]