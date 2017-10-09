from sklearn.svm import SVR
import numpy as np
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

    #Dividing the original train dataset into train/test set, whole set because keras provides spliting to cross-validation and train set
    X_train = datasetX[:TRAIN_SIZE]
    Y_train = datasetY[:TRAIN_SIZE]
    X_val = datasetX[TRAIN_SIZE:]
    Y_val = datasetY[TRAIN_SIZE:]

    #Training our model
    svr_lin = SVR(kernel='linear', C=1000)
    svr_poly = SVR(kernel='poly', C=1000, degree=2, gamma=0.5)
    svr_rbf = SVR(kernel='rbf', C=1000, gamma=2)

    for name,classifier in zip(["Gaussian"],[svr_rbf]):

        classifier.fit(datasetX, datasetY)

        #Making predictions on train set and setting negative results to zero
        predictions_train = classifier.predict(X_train)
        predictions_train = np.maximum(predictions_train, 0)
        train_error = rmsle(predictions_train,Y_train)
        predictions_val = classifier.predict(X_val)
        predictions_val = np.maximum(predictions_val, 0)
        val_error = rmsle(predictions_val,Y_val)
        print (name,"kernel: Train error:",train_error,", Val error:",val_error)

        #Making predictions on test set and setting negative results to zero
        predictions_test = classifier.predict(datasetX_pred)
        predictions_test = np.maximum(predictions_test, 0)

        #Saving predictions
        np.savetxt("svm_"+name+"_predictions.csv", predictions_test, delimiter=",")
