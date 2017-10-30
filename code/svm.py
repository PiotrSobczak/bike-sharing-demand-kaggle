from sklearn.svm import SVR
import numpy as np
from matplotlib import pyplot as plt
from data_utils import DataUtils as du
TOTAL_DATASET_SIZE = 10887

HOURS_IN_DAY = 24
START_YEAR = 2011
DAYS_IN_YEAR = 365
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTHS_IN_YEAR = 12

TRAIN_SIZE = 9000

def norm_arr(array):
    return (array - array.min() - (array.max() - array.min()) / 2) / ((array.max() - array.min()) / 2)

def get_train_error(predictions_train,labels_train):
    total_error = 0
    for y_pred, label in zip(predictions_train, labels_train):
        total_error = total_error + (np.log(y_pred + 1) - np.log(label + 1)) ** 2
    return np.sqrt(total_error / len(predictions_train))

def rmsle(y_pred,y_true):
    #print(np.log(y_pred+1).shape,np.log(y_true+1).shape)
    log_err = (np.log(y_pred + 1) - np.log(y_true + 1))
    squared_le = np.power(log_err,2)
    mean_sle = np.mean(squared_le)
    root_msle = np.sqrt(mean_sle)
    return (root_msle)

if __name__ == '__main__':
    df_x, _, df_y_log, train_x, train_y, train_y_log, val_x, val_y, test_x, test_date_df = du.get_processed_df(
        '../data/train.csv', '../data/test.csv',output_cols=['registered','casual','count'])

    train_y_log_reg = np.array(train_y_log['registered'])
    train_y_log_cas = np.array(train_y_log['casual'])
    train_y = np.array(train_y['count'])
    df_y_log_reg = np.array(df_y_log['registered'])
    df_y_log_cas = np.array(df_y_log['casual'])

    gamma_cas = 0.015
    C_cas = 150
    gamma_reg = 0.0769
    C_reg = 143

    reverse_opts = [False]

    tested_params = [C_reg]
    val_error_hist = np.zeros(len(tested_params))
    train_error_hist = np.zeros(len(tested_params))

    for i,C_reg in enumerate(tested_params):
        for name in ["Gaussian"]:
            #Training our regression model
            regressor_reg = SVR(kernel='rbf', C=C_reg, gamma=gamma_reg)
            regressor_cas = SVR(kernel='rbf', C=C_cas, gamma=gamma_cas)
            regressor_reg.fit(train_x, train_y_log_reg)
            regressor_cas.fit(train_x, train_y_log_cas)

            #Making predictions on train set
            predictions_train_log_reg = regressor_reg.predict(train_x)
            predictions_train_reg = np.exp(predictions_train_log_reg) - 1
            predictions_train_log_cas = regressor_cas.predict(train_x)
            predictions_train_cas = np.exp(predictions_train_log_cas) - 1
            predictions_train = np.maximum(0, predictions_train_reg) + np.maximum(0, predictions_train_cas)
            train_error = rmsle(predictions_train,train_y)
            train_error_hist[i] += train_error

            # Making predictions on val set
            predictions_val_log_reg = regressor_reg.predict(val_x)
            predictions_val_reg = np.exp(predictions_val_log_reg) - 1
            predictions_val_log_cas = regressor_cas.predict(val_x)
            predictions_val_cas = np.exp(predictions_val_log_cas) - 1
            predictions_val = np.maximum(0, predictions_val_reg) + np.maximum(0, predictions_val_cas)
            val_error = rmsle(predictions_val,val_y)
            val_error_hist[i] += val_error

            #Making predictions on test set and saving them
            predictions_test_log_reg = regressor_reg.predict(test_x)
            predictions_test_reg = np.exp(predictions_test_log_reg) - 1
            predictions_test_log_cas = regressor_cas.predict(test_x)
            predictions_test_cas = np.exp(predictions_test_log_cas) - 1
            predictions_test = np.maximum(0, predictions_test_reg) + np.maximum(0, predictions_test_cas)
            test_date_df['count'] = predictions_test
            test_date_df.to_csv('predictions_svm.csv', index=False)

        #Computing avg error from reversed and non-reversed data
        val_error_hist[i] = val_error_hist[i] / len(reverse_opts)
        train_error_hist[i] = train_error_hist[i] / len(reverse_opts)

        print (name,"kernel, gamma_cas = ",gamma_cas,", gamma_reg = ",gamma_reg,
               ", C_cas = ",C_cas,", C_reg = ",C_reg,", Train error:",train_error_hist[i],", Val error:",val_error_hist[i])

    plt.plot(tested_params,val_error_hist)
    plt.plot(tested_params,train_error_hist)
    plt.show()
