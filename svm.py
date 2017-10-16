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

    datasetX,datasetY,datasetX_pred = du.get_processed_df('data/train.csv','data/test.csv')

    #Conversion from DF to numpyarray for Keras duncs
    datasetX = np.array(datasetX)
    datasetY = np.array(datasetY)
    datasetX_pred = np.array(datasetX_pred)

    gammas = np.linspace(0.1,0.3,21)

    val_error_hist = np.zeros(len(gammas))
    train_error_hist = np.zeros(len(gammas))

    for i,gamma in enumerate(gammas):
        svr_rbf = SVR(kernel='rbf', C=325, gamma=gamma)

        # Definitions of other kernels one may want to use
        # svr_lin = SVR(kernel='linear', C=1000)
        # svr_poly = SVR(kernel='poly', C=1000, degree=2, gamma=gamma)
        for name,regressor in zip(["Gaussian"],[svr_rbf]):
            for reverse in [False,True]:
                #Getting seperate train and val datasets to control data distribution
                X_train,Y_train,Y_train_log,X_val,Y_val = du.get_sep_datasets(datasetX,datasetY,TRAIN_SIZE,reverse_data_order=reverse)
                #print("Loaded dataset with reverse =",reverse,
                #      ",Dataset sizes: {X_train,Y_train,Y_train_log,X_val,Y_val}:{"
                #      ,X_train.shape,Y_train.shape,Y_train_log.shape,X_val.shape,Y_val.shape,"}")
                
                #Training our regression model
                regressor.fit(X_train, Y_train_log)

                #Making predictions on train set
                predictions_train_log = regressor.predict(X_train)
                predictions_train = np.exp(predictions_train_log) - 1
                train_error = rmsle(predictions_train,Y_train)
                train_error_hist[i] += train_error

                # Making predictions on val set
                predictions_val_log = regressor.predict(X_val)
                predictions_val = np.exp(predictions_val_log) - 1
                val_error = rmsle(predictions_val,Y_val)
                val_error_hist[i] += val_error

                print(name, "kernel, gamma = ", gamma, ", data reversed = ", reverse,
                      ", Train error:", train_error, ", Val error:", val_error)

            #Computing avg error from reversed and non-reversed data
            val_error_hist[i] = val_error_hist[i] / 2
            train_error_hist[i] = train_error_hist[i] / 2

            print (name,"kernel, gamma = ",gamma,", Train error:",train_error_hist[i],", Val error:",val_error_hist[i])

    #Making predictions on test set and setting negative results to zero
    #predictions_test = regressor.predict(datasetX_pred)
    #predictions_test = np.exp(predictions_test) - 1

    #Saving predictions
    #np.savetxt("svm_" + name + "_predictions.csv", predictions_test, delimiter=",")

    plt.plot(val_error_hist)
    plt.plot(train_error_hist)
    plt.show()
