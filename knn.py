from sklearn.neighbors import KNeighborsRegressor
from data_utils import DataUtils as du
import numpy as np
import matplotlib.pyplot as plt

num_k_neighbours = 5
start_k_neighbours = 2

val_errors = np.zeros(num_k_neighbours)
train_errors = np.zeros(num_k_neighbours)
best_val_error = 1000
opt_train_error = 1000
test_error = 1000

def rmsle(y_pred,y_true):
    log_err = (np.log(y_pred + 1) - np.log(y_true + 1))
    squared_le = np.power(log_err,2)
    mean_sle = np.mean(squared_le)
    root_msle = np.sqrt(mean_sle)
    return (root_msle)

if __name__ == '__main__':

    df_x, df_y, df_y_log, train_x, train_y, train_y_log, val_x, val_y, test_x, test_date_df = du.get_processed_df(
        'data/train.csv', 'data/test.csv')

    k_parameters = range(start_k_neighbours, start_k_neighbours + num_k_neighbours)

    for i, k in enumerate(k_parameters):
        # Training Knn regressor
        knn_regressor = KNeighborsRegressor(n_neighbors=k, weights='uniform')
        knn_regressor.fit(train_x, train_y_log)

        # Making predictions on train set
        predictions_train_log = knn_regressor.predict(train_x)
        predictions_train = np.exp(predictions_train_log) - 1
        predictions_train = np.maximum(0, predictions_train)
        train_error = rmsle(predictions_train, train_y)
        train_errors[i] = train_error

        # Making predictions on val set
        predictions_val_log = knn_regressor.predict(val_x)
        predictions_val = np.exp(predictions_val_log) - 1
        predictions_val = np.maximum(0, predictions_val)
        val_error = rmsle(predictions_val, val_y)
        val_errors[i] = val_error

        #Making predictions on test set and saving them
        predictions_test_log = knn_regressor.predict(test_x)
        predictions_test = np.exp(predictions_test_log) - 1
        predictions_test = np.maximum(0, predictions_test)
        test_date_df['count'] = predictions_test
        test_date_df.to_csv('predictions_k_'+str(k)+'.csv', index=False)

        print("K =",k,", train error:", train_error, ", val error:", val_error)

    plt.plot(k_parameters,train_errors, label='avg_train')
    plt.plot(k_parameters,val_errors, label='avg_test')
    plt.xlabel('k parameter', fontsize=16)
    plt.ylabel('Error', fontsize=16)
    plt.legend()
    plt.show()