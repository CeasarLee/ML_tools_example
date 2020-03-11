import numpy as np
import xgboost as xgb
import _init_path
from utils.recover_data import recover_missing, feature_transform
import pandas as pd
DATA_PATH = '../data/titanic_disaster/'


def set_Cabin(df, i):
    if pd.isnull(df.loc[i, 'Cabin']):
        df.loc[i, 'Cabin'] = 1
    else:
        df.loc[i, 'Cabin'] = 0


train_data = pd.read_csv(DATA_PATH + 'train.csv')
test_data = pd.read_csv(DATA_PATH + 'test.csv')

df = recover_missing(train_data, ['Fare', 'Parch', 'SibSp', 'Pclass'], 'Age', method='Xgboost')
df = feature_transform(df, set_Cabin)
print(1)