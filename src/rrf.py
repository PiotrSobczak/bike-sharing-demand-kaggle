import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from data_utils import DataUtils as du


def get_rmsle(y_pred, y_actual):
    diff = np.log(y_pred + 1) - np.log(y_actual + 1)
    mean_error = np.square(diff).mean()
    return np.sqrt(mean_error)

if __name__ == '__main__':
    df_x, _, df_y_log, train_x, train_y, train_y_log, val_x, val_y, test_x, test_date_df = du.get_processed_df(
        '../data/train.csv', '../data/test.csv', output_cols=['registered', 'casual', 'count'],model = "rrf",normalize=False)

    train_y_log_reg = train_y_log['registered'].as_matrix()
    train_y_log_cas = train_y_log['casual'].as_matrix()
    train_y = train_y['count'].as_matrix()

    max_depth_params = np.arange(1,35,1)
    max_depth = 20
    n_estimators_params = np.arange(1,1000,10)
    n_estimators = 300
    min_split_params = np.arange(12,13,1)
    min_split = 12

    tested_params = min_split_params

    val_scores = np.zeros(len(tested_params))
    train_scores = np.zeros(len(tested_params))

    for i,min_split in enumerate(tested_params):

        params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'random_state': 0, 'min_samples_split' : int(min_split), 'n_jobs': -1}
        rf_model = RandomForestRegressor(**params)

        # Training registered model and making predictions
        model_r = rf_model.fit(df_x, df_y_log['registered'])
        y_pred_train_reg = np.exp(model_r.predict(train_x)) - 1
        y_pred_val_reg = np.exp(model_r.predict(val_x)) - 1
        y_pred_test_reg = np.exp(model_r.predict(test_x)) - 1

        # save the model
        pickle.dump(model_r, open('rrf_reg_model.sav', 'wb'))

        #Training casual model and making predictions
        model_c = rf_model.fit(df_x, df_y_log['casual'])
        y_pred_train_cas = np.exp(model_c.predict(train_x)) - 1
        y_pred_val_cas = np.exp(model_c.predict(val_x)) - 1
        y_pred_test_cas = np.exp(model_c.predict(test_x)) - 1

        pickle.dump(model_c, open('rrf_cas_model.sav', 'wb'))

        #Evaluating train and val score
        y_pred_train = np.round(y_pred_train_reg + y_pred_train_cas)
        y_pred_train[y_pred_train < 0] = 0
        train_score = get_rmsle(y_pred_train, train_y)
        train_scores[i] = train_score

        y_pred_val = np.round(y_pred_val_reg + y_pred_val_cas)
        y_pred_val[y_pred_val < 0] = 0
        val_score = get_rmsle(y_pred_val, val_y)
        val_scores[i] = val_score

        print ("Max depth:",max_depth,"Number of trees:",n_estimators,"Train score:",train_score,"Val score:",val_score)

        #Saving predictions to submission file
        y_pred_test = y_pred_test_reg + y_pred_test_cas
        test_date_df['count'] = y_pred_test
        test_date_df.to_csv('predictions_rrf.csv', index=False)

    plt.plot(tested_params,val_scores)
    plt.plot(tested_params, train_scores)
    plt.ylabel('RMSLE', fontsize=18)
    plt.xlabel('max depth', fontsize=18)
    plt.legend(['validation error', 'train error'], fontsize=14)
    plt.show()
