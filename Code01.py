


import numpy as np
import seaborn as sns
import re
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
import matplotlib; matplotlib.style.use('ggplot')
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('C:/Users/Zubair Ahmed/PycharmProjects/HousePriceRegression/train.csv')
tdf = pd.read_csv('C:/Users/Zubair Ahmed/PycharmProjects/HousePriceRegression/test.csv', sep='\s*,\s*', encoding="utf-8-sig", engine='python')

salePrice = df['SalePrice']

df = df.set_index('Id')
tdf = tdf.set_index('Id')

minP = df['SalePrice'].min()
maxP = df['SalePrice'].max()

# id min = 496, max = 692

sns.set_style('whitegrid')
sns.set_context('notebook')
# sns.boxplot(x=df['SalePrice'], orient='v')
# sns.swarmplot(x=df['SalePrice'], orient='v', color=".25")
# sns.violinplot(salePrice)
# sns.plt.show()
# cols = ['MSSubClass','MSZoning','LotFrontage','LotArea','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','OverallQual','SalePrice']
# cols = ['OverallCond',	'YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','SalePrice']
# cols = ['BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','Heating','SalePrice']
# cols = ['HeatingQC','CentralAir','Electrical','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','SalePrice']
# cols = ['FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces','FireplaceQu','GarageType','SalePrice']
# cols = ['FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces','FireplaceQu','GarageType','SalePrice']
# cols = ['GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond','PavedDrive','WoodDeckSF','SalePrice']
# cols = ['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','PoolQC','Fence','SalePrice']
# cols = ['MiscFeature','MiscVal','MoSold','YrSold','SaleType','SaleCondition','SalePrice']
cols = ['OverallQual','GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF','YearBuilt',
         'TotRmsAbvGrd','YearRemodAdd','BsmtFinSF1','LotArea','FullBath','HalfBath','SalePrice']

ohenc = ['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2',
         'BldgType','HouseStyle','RoofStyle','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond',
         'BsmtExposure','BsmtFinSF2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish',
         'GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition']

# cols = ['OverallQual','GrLivArea','1stFlrSF','TotalBsmtSF','TotRmsAbvGrd','YearBuilt','YearRemodAdd','2ndFlrSF','GarageArea','ScreenPorch','SalePrice']
# cols = ['2ndFlrSF','GarageArea','ScreenPorch','3SsnPorch','SalePrice']
df.fillna(df.mean(), inplace=True)
# sns.pairplot(df[cols],size=2.0)

# sns.plt.show()

X = df
tX = tdf

train_num = len(X)
dataset = pd.concat(objs=[X, tX], axis=0)
dataset_preprocessed = pd.get_dummies(dataset)
train_preprocessed = dataset_preprocessed[:train_num]
test_preprocessed = dataset_preprocessed[train_num:]

cm = {}
# sns.set(font_scale=.7)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
# sns.plt.show()
count = 0
for col in train_preprocessed.columns:
    columns = [col, 'SalePrice']
    vals = train_preprocessed[columns]
    p = re.compile('1|2|3|4|5|6|7|8|9|0')
    # col = p.sub('a',col)

    if not col in cm:
        cm[col] = [np.corrcoef(vals.values.T)[1][0]]
    else:
        cm[col].append(np.corrcoef(vals.values.T)[1][0])

    # sns.set(font_scale=.7)
    # hm = sns.heatmap(np.corrcoef(vals.values.T), cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
    # sns.plt.show()
    # global count
    # count += 1
# print [value for (key,value) in  sorted(cm, reverse=True, key=cm.__getitem__)]

print ["{0}-{1}".format(x, cm[x]) for x in sorted(cm, reverse=True, key=cm.__getitem__)]

# for x in cm:
#     print "{0}-{1}".format(x, cm[x])

# print cm
