import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from data_utils import DataUtils as du

def get_rmsle(y_pred, y_actual):
    diff = np.log(y_pred + 1) - np.log(y_actual + 1)
    mean_error = np.square(diff).mean()
    return np.sqrt(mean_error)

if __name__ == '__main__':
    df_x, _, df_y_log, train_x, train_y, train_y_log, val_x, val_y, test_x, test_date_df = du.get_processed_df(
        'data/train.csv', 'data/test.csv', output_cols=['registered', 'casual', 'count'])

    train_y_log_reg = train_y_log['registered'].as_matrix()
    train_y_log_cas = train_y_log['casual'].as_matrix()
    train_y = train_y['count'].as_matrix()

    params = {'n_estimators': 1000, 'max_depth': 15, 'random_state': 0, 'min_samples_split' : 5, 'n_jobs': -1}
    rf_model = RandomForestRegressor(**params)

    # Training registered model and making predictions
    model_r = rf_model.fit(train_x, train_y_log_reg)
    y_pred_train_reg = np.exp(model_r.predict(train_x)) - 1
    y_pred_val_reg = np.exp(model_r.predict(val_x)) - 1
    y_pred_test_reg = np.exp(model_r.predict(test_x)) - 1

    #Training casual model and making predictions
    model_c = rf_model.fit(train_x, train_y_log_cas)
    y_pred_train_cas = np.exp(model_c.predict(train_x)) - 1
    y_pred_val_cas = np.exp(model_c.predict(val_x)) - 1
    y_pred_test_cas = np.exp(model_c.predict(test_x)) - 1

    #Evaluating train and val score
    y_pred_train = np.round(y_pred_train_reg + y_pred_train_cas)
    y_pred_train[y_pred_train < 0] = 0
    train_score = get_rmsle(y_pred_train, train_y)

    y_pred_val = np.round(y_pred_val_reg + y_pred_val_cas)
    y_pred_val[y_pred_val < 0] = 0
    val_score = get_rmsle(y_pred_val, val_y)

    print ("Train score:",train_score,"Val score:",val_score)

    #Saving predictions to submission file
    y_pred_test = y_pred_test_reg + y_pred_test_cas
    test_date_df['count'] = y_pred_test
    test_date_df.to_csv('predictions_rrf.csv', index=False)