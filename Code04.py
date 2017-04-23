import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

df = pd.read_csv('C:/Users/Zubair Ahmed/PycharmProjects/HousePriceRegression/train.csv')
tdf = pd.read_csv('C:/Users/Zubair Ahmed/PycharmProjects/HousePriceRegression/test.csv')

salePrice = df['SalePrice']

df = df.set_index('Id')

cols = ['OverallQual','GrLivArea','1stFlrSF','TotalBsmtSF','YearBuilt','YearRemodAdd','TotRmsAbvGrd','SalePrice']
xcols = ['OverallQual','GrLivArea','1stFlrSF','TotalBsmtSF','TotRmsAbvGrd','YearBuilt','YearRemodAdd','GarageArea']
df.fillna(df.mean(), inplace=True)
TotalBsmtSFMean = df['TotalBsmtSF'].mean()
df.loc[df['TotalBsmtSF'] == 0, 'TotalBsmtSF'] = TotalBsmtSFMean

tdf.fillna(tdf.mean(), inplace=True)
TTotalBsmtSFMean = tdf['TotalBsmtSF'].mean()
tdf.loc[tdf['TotalBsmtSF'] == 0, 'TotalBsmtSF'] = TotalBsmtSFMean

X = df[xcols]
y = df['SalePrice']

tX = tdf[xcols]

std = StandardScaler()
slr = ElasticNet(alpha=0.6, l1_ratio=0.5)
X_std = std.fit_transform(X)
X_std_test = std.fit_transform(tX)

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=0)

slr.fit(X_train, y_train)

y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

y_test_pred = slr.predict(X_std_test)
submission = pd.DataFrame({"Id": tdf["Id"],"SalePrice": y_test_pred})
submission.loc[submission['SalePrice'] <= 0, 'SalePrice'] = 0
submission.to_csv('submission.csv', index=False)