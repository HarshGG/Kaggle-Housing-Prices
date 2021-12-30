import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import re
# from xgboost import XGBRegressor
# import seaborn as sns

print('imported')
# uploaded = files.upload()

df_train = pd.read_csv("https://raw.githubusercontent.com/HarshGG/Kaggle-Housing-Prices/main/train.csv")
df_train.head()

#making HouseStyle values numerical. 
#Still not ok because some columns are all non numerical, such as 75, SVL. 
#Some are also unfinished i assume like column 9, 1.5Unf
# non_decimal = re.compile(r'[^\d.]+')
# for i in range(len(df_train['HouseStyle'])):
#   df_train['HouseStyle'][i] = non_decimal.sub('',df_train['HouseStyle'][i])
# df_train.loc[367,'HouseStyle']
# df_train.head()

numerical=['LotFrontage', 'BsmtFinSF1', 'WoodDeckSF', 'MSSubClass', '2ndFlrSF', 'TotRmsAbvGrd', 'MasVnrArea', 'YearRemodAdd', 'ScreenPorch', 'OverallCond', 'EnclosedPorch', 'GrLivArea', 'GarageArea', 'LowQualFinSF', 'OverallQual', 'TotalBsmtSF', 'KitchenAbvGr', 'GarageYrBlt', 'BedroomAbvGr', '1stFlrSF', 'YearBuilt', 'OpenPorchSF', 'LotArea', 'BsmtUnfSF']
categorical=['MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'YrSold', 'Fireplaces','PoolQC','Alley','Fence']

df_train.dropna(subset=numerical,inplace=True)
df_train = OneHotEncoder(df_train, categories=categorical)

print(df_train.head())
# X_train = df_train[numerical][:1168]
# y_train = df_train['SalePrice'][:1168]
# X_val = df_train[numerical][1169:1200]
# y_val = df_train['SalePrice'][1168:-1]


# X_train, X_val, y_train, y_val = train_test_split(df_train[numerical], df_train['SalePrice'], test_size=0.2, random_state=42)

# print(X_train.head())
# print('ooooo')
# print(X_val.head())
# print('01910')

# regr = linear_model.LinearRegression()
# regr.fit(X_train, y_train)
# print('regr model made')

# df_test = pd.read_csv('https://raw.githubusercontent.com/HarshGG/Kaggle-Housing-Prices/main/test.csv')
# df_test.dropna(subset=numerical,inplace=True)
# yPred = regr.predict(X_val)

# print("Coefficients: \n", regr.coef_)
# # The mean squared error
# print("Mean squared error: %.2f" % mean_squared_error(y_val, yPred))
# # The coefficient of determination: 1 is perfect prediction
# print("Coefficient of determination: %.2f" % r2_score(y_val, yPred))

