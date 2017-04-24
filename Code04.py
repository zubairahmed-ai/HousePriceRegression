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

df = pd.read_csv('C:/Users/Zubair Ahmed/PycharmProjects/HousePriceRegression/train.csv')
tdf = pd.read_csv('C:/Users/Zubair Ahmed/PycharmProjects/HousePriceRegression/test.csv')

salePrice = df['SalePrice']

df = df.set_index('Id')

cols = ['OverallQual','GrLivArea','1stFlrSF','TotalBsmtSF','YearBuilt','YearRemodAdd','TotRmsAbvGrd','SalePrice']
xcols = ['OverallQual','GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF','YearBuilt','TotRmsAbvGrd','YearRemodAdd','BsmtFinSF1','LotArea']
df.fillna(df.mean(), inplace=True)
TotalBsmtSFMean = df['TotalBsmtSF'].mean()
df.loc[df['TotalBsmtSF'] == 0, 'TotalBsmtSF'] = np.round(TotalBsmtSFMean).astype(int)

tdf.fillna(tdf.mean(), inplace=True)
TTotalBsmtSFMean = tdf['TotalBsmtSF'].mean()
tdf.loc[tdf['TotalBsmtSF'] == 0, 'TotalBsmtSF'] = np.round(TotalBsmtSFMean).astype(int)

X = df[xcols]
y = df['SalePrice']
tX = tdf[xcols]

std = StandardScaler()
slr = ElasticNet(alpha=0.6, l1_ratio=0.5)
model = RandomForestRegressor(n_estimators=500, n_jobs=-1)

X_std = std.fit_transform(X)
X_std_test = std.fit_transform(tX)

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=0)

slr.fit(X_train, y_train)
model.fit(X_train, y_train)

y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

predictions = np.array(model.predict(X_test))
rmse = math.sqrt(np.mean((np.array(y_test) - predictions)**2))
imp = sorted(zip(X.columns, model.feature_importances_), key=lambda tup: tup[1], reverse=True)

plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10000, xmax=1000000, lw=2, color='red')
plt.xlim([-10000, 1000000])
plt.tight_layout()

# plt.savefig('./figures/slr_residuals.png', dpi=300)
# plt.show()

print "RMSE: " + str(rmse)
print "10 Most Important Variables:" + str(imp[:10])

# y_test_pred = slr.predict(X_std_test)
y_test_pred = model.predict(X_std_test)
submission = pd.DataFrame({"Id": tdf["Id"],"SalePrice": y_test_pred})
submission.loc[submission['SalePrice'] <= 0, 'SalePrice'] = 0
submission.to_csv('submission.csv', index=False)