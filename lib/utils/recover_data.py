from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def recover_data(df, input_attr, missing_attr, method='RandomForest'):
    """
    :param df: input dataframe
    :param input_attr: type -> list:
    :param missing_attr: str: attrb need to be recover
    :param method: which method choose to recover data -> 'RandomForest', 'Xgboost', 'lightgbm'
    :return: df
    """
    attrb_df = df[list(missing_attr) + input_attr]
    train = attrb_df[getattr(attrb_df, missing_attr).notnull()].as_matrix()
    test = attrb_df[getattr(attrb_df, missing_attr).isnull()].as_matrix()

    train_label = train[:, 0]
    train_data = train[:, 1:]

    if method == 'RandomForest':
        model = RandomForestRegressor(random_state=1, n_estimators=2000, n_jobs=-1)
        model.fit(train_data, train_label)
        predict_result = model.predict(test[:, 1:])
        df.loc[(hasattr(df, missing_attr).isnull()), missing_attr] = predict_result
    elif method == 'Xgboost':
        model = 

    return df