from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import pandas as pd

def recover_missing(df, input_attr, missing_attr, method='RandomForest'):
    """
    :param df: input dataframe
    :param input_attr: type -> list:
    :param missing_attr: str: attrb need to be recover
    :param method: which method choose to recover data -> 'RandomForest', 'Xgboost', 'lightgbm'
    :return: df
    """
    attrb_df = df[[missing_attr] + input_attr]
    train = attrb_df[getattr(attrb_df, missing_attr).notnull()].values
    test = attrb_df[getattr(attrb_df, missing_attr).isnull()].values

    train_label = train[:, 0]
    train_data = train[:, 1:]

    if method == 'RandomForest':
        model = RandomForestRegressor(random_state=1, n_estimators=2000, n_jobs=-1)
        model.fit(train_data, train_label)
        predict_result = model.predict(test[:, 1:])
        df.loc[(hasattr(df, missing_attr).isnull()), missing_attr] = predict_result
    elif method == 'Xgboost':
        best_params = {
            'max_depth': 5,
            'learning_rate': 0.01,
            'n_estimators': 3000,
            'gamma': 0.8,
            'min_child_weight': 2,
            'reg_alpha': 0.001,
            'max_delta_step': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.9,
            'base_score': 10,
            'seed': 1,
            'nthread': 30
        }
        model = xgb.XGBRegressor(**best_params)
        model.fit(train_data, train_label)
        predict_result = model.predict(test[:, 1:])
        df.loc[(getattr(df, missing_attr).isnull()), missing_attr] = predict_result
    return df, model

def feature_transform(df, func):
    nrow, ncol = df.shape
    for i in range(nrow):
        func(df, i)
    return df


if __name__ == '__main__':
    data = pd.read_csv('../../data/titanic_disaster/train.csv')
    df = recover_missing(data, ['Fare', 'Parch', 'SibSp', 'Pclass'], 'Age', method='Xgboost')
