import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint
import keras.backend as KB
from data_utils import DataUtils as du

TOTAL_DATASET_SIZE = 10887

def rmsle(y_true, y_pred):
    y_count_pred = KB.sum(y_pred,axis=1)
    y_count_true = KB.sum(y_true,axis=1)
    return KB.sqrt(KB.mean(KB.square(KB.log(y_count_pred+1) - KB.log(y_count_true+1))))

if __name__ == '__main__':
    output_columns = ['registered', 'casual']

    df_x, _, df_y_log, train_x, train_y, train_y_log, val_x, val_y, test_x, test_date_df = \
        du.get_processed_df('data/train.csv', 'data/test.csv',output_cols=output_columns)

    print("Dataset loaded, train_setX:",train_x.shape,", train_setY:",train_y.shape,", val_setX:",val_x.shape,", val_setY:",val_y.shape)

    df_x = np.array(df_x)
    df_y_log = np.array(df_y_log)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_y_log = np.array(train_y_log)
    val_x = np.array(val_x)
    val_y = np.array(val_y)
    val_y = np.reshape(val_y, newshape=(val_y.shape[0], 1))
    test_x = np.array(test_x)

    deep_layers_size = 10

    #Defining our NN model
    model = Sequential()
    model.add(Dense(units=deep_layers_size, input_dim=13,kernel_initializer='he_normal',
                    bias_initializer='zeros'))
    model.add(Activation("tanh"))
    model.add(Dense(units=deep_layers_size,kernel_initializer='he_normal',
                    bias_initializer='zeros'))
    model.add(Activation("tanh"))
    model.add(Dense(units=deep_layers_size,kernel_initializer='he_normal',
                    bias_initializer='zeros'))
    model.add(Activation("tanh"))
    model.add(Dense(units=deep_layers_size,kernel_initializer='he_normal',
                    bias_initializer='zeros'))
    model.add(Activation("tanh"))
    model.add(Dense(units=len(output_columns),kernel_initializer='he_normal',
                    bias_initializer='zeros'))
    model.add(Activation("relu"))
    model.compile(loss=rmsle, optimizer='adam')

    #Defining checkpoint and callbacks to save the best set of weights and limit printing
    checkpoint = ModelCheckpoint('best_weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    #Start training
    history_callback = model.fit(train_x, train_y, epochs=10000, batch_size=64,validation_data=(val_x,val_y),verbose=2,callbacks=callbacks_list)

    #Recovering val and train loss history from callbacks
    loss_history = history_callback.history["loss"]
    val_loss_history = history_callback.history["val_loss"]

    np.savetxt("error_plot.csv", [loss_history,val_loss_history], delimiter=",")

    #Loading best model
    model.load_weights('best_weights.hdf5')

    #Making predictionsand saving them to csv
    predictions = model.predict(test_x)
    if output_columns == ['registered','casual']:
        test_date_df['count'] = np.sum(predictions,axis=1)
    elif len(output_columns) == 1:
        test_date_df['count'] = predictions
    test_date_df.to_csv('predictions_keras_nn.csv', index=False)

    #Plotting training loss and validation loss to control overfitting
    plt.plot(loss_history)
    plt.plot(val_loss_history)
    plt.show()
