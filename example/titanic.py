import numpy as np
import xgboost as xgb
import _init_path
from utils.recover_data import recover_missing, feature_transform
import pandas as pd
import sklearn.preprocessing as preprocessing

DATA_PATH = '../data/titanic_disaster/'


def set_Cabin(df, i):
    if pd.isnull(df.loc[i, 'Cabin']):
        df.loc[i, 'Cabin'] = 1
    else:
        df.loc[i, 'Cabin'] = 0


train_data = pd.read_csv(DATA_PATH + 'train.csv')
test_data = pd.read_csv(DATA_PATH + 'test.csv')

# 使用xgboost进行值填充，将age条目，通过fare，parch，sibsp和pclass进行预测填充
df, recover_model = recover_missing(train_data, ['Fare', 'Parch', 'SibSp', 'Pclass'], 'Age', method='Xgboost')
df = feature_transform(df, set_Cabin)
# print(1)

# 将字符属性转换为0-1值表示
dummies_Cabin = pd.get_dummies(train_data['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(train_data['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(train_data['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(train_data['Pclass'], prefix= 'Pclass')

df = pd.concat([df, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
# 将属性的数值归一化
scaler = preprocessing.StandardScaler()
