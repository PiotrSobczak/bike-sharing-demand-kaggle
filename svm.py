from sklearn.svm import SVR
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

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

    # datasetX,datasetY,datasetX_pred,df_predictions = du.get_processed_df_nn('data/train.csv','data/test.csv')

    #Conversion from DF to numpyarray for Keras duncs
    # datasetX = np.array(datasetX)
    # datasetY = np.array(datasetY)
    # datasetX_pred = np.array(datasetX_pred)

    gammas = np.linspace(0.07,0.2,30)
    Cs = np.linspace(100,1000,10)
    val_error_hist = np.zeros(len(gammas))
    train_error_hist = np.zeros(len(gammas))

    reverse_opts = [False]

    for i,gamma in enumerate(gammas):
        # Definitions of other kernels one may want to use
        # svr_lin = SVR(kernel='linear', C=1000)
        # svr_poly = SVR(kernel='poly', C=1000, degree=2, gamma=gamma)
        for name in ["Gaussian"]:
            for reverse in reverse_opts:
                #Getting seperate train and val datasets to control data distribution
                #train_x,train_y,Y_train_log,val_x,val_y = du.get_sep_datasets(datasetX,datasetY,TRAIN_SIZE,reverse_data_order=reverse)
                train_x, train_y, val_x, val_y,test_x,test_date_df = du.get_processed_df_nn('data/train.csv', 'data/test.csv')
                #train_x, train_y, val_x, val_y,test_x,test_date_df = du.get_processed_df('data/train.csv','data/test.csv')

                train_x = np.array(train_x)
                train_y = np.array(train_y)
                val_x = np.array(val_x)
                val_y = np.array(val_y)
                #test_x = np.array(test_x)

                #Training our regression model
                regressor = SVR(kernel='rbf', C=1000, gamma=gamma)
                #regressor.fit(X_train, Y_train_log)
                #print(np.max(train_x),np.min(train_x),np.max(train_y),np.min(train_y))
                regressor.fit(train_x, train_y)

                #Making predictions on train set
                predictions_train = regressor.predict(train_x)
                #predictions_train = np.exp(predictions_train_log) - 1
                predictions_train = np.maximum(0, predictions_train)
                train_error = rmsle(predictions_train,train_y)
                train_error_hist[i] += train_error

                # Making predictions on val set
                predictions_val = regressor.predict(val_x)
                # predictions_val = np.exp(predictions_val_log) - 1
                predictions_val = np.maximum(0, predictions_val)
                val_error = rmsle(predictions_val,val_y)
                val_error_hist[i] += val_error

                #Making predictions on test set and setting negative results to zero
                #predictions_test = regressor.predict(test_x)
                #predictions_test = np.exp(predictions_test) - 1
                #predictions_test = np.maximum(0,predictions_test)

                #Saving predictions
                #test_date_df['count'] = predictions_test
                #test_date_df.to_csv('predictions.csv', index=False)

                # print(name, "kernel, gamma = ", gamma, ", data reversed = ", reverse,
                #      ", Train error:", train_error, ", Val error:", val_error)

            #Computing avg error from reversed and non-reversed data
            val_error_hist[i] = val_error_hist[i] / len(reverse_opts)
            train_error_hist[i] = train_error_hist[i] / len(reverse_opts)

            print (name,"kernel, gamma = ",gamma,", Train error:",train_error_hist[i],", Val error:",val_error_hist[i])

    plt.plot(gammas,val_error_hist)
    plt.plot(gammas,train_error_hist)
    plt.show()
