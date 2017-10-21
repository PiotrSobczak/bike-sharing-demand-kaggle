import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint
import keras.backend as KB
from keras.layers import Dropout
from data_utils import DataUtils as du

TOTAL_DATASET_SIZE = 10887

def rmsle(y_true, y_pred):
    return KB.sqrt(KB.mean(KB.square(KB.log(y_pred+1) - KB.log(y_true+1)), axis=-1))

def norm_arr(array):
    return (array - array.min() - (array.max() - array.min()) / 2) / ((array.max() - array.min()) / 2)

df_x, _, df_y_log, train_x, train_y, train_y_log, val_x, val_y, test_x, test_date_df = du.get_processed_df(
        'data/train.csv', 'data/test.csv')

df_x = np.array(df_x)
df_y_log = np.array(df_y_log)
train_x = np.array(train_x)
train_y = np.array(train_y)
train_y_log = np.array(train_y_log)
val_x = np.array(val_x)
val_y = np.array(val_y)
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
model.add(Dense(units=3,kernel_initializer='he_normal',
                bias_initializer='zeros'))
model.add(Activation("relu"))
model.compile(loss=rmsle, optimizer='adam')

#Defining checkpoint and callbacks to save the best set of weights and limit printing
checkpoint = ModelCheckpoint('best_weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#Start training
history_callback = model.fit(train_x, train_y, epochs=10000, batch_size=64,validation_data=(val_x,val_y),verbose=2,callbacks=callbacks_list)

#Recovering val_loss history and training loss history from callbacks to arrays
loss_history = history_callback.history["loss"]
val_loss_history = history_callback.history["val_loss"]

np.savetxt("error_plot.csv", [loss_history,val_loss_history], delimiter=",")

#Loading best model
model.load_weights('best_weights.hdf5')

#Making predictionsand saving them to csv
predictions = model.predict(np.array(test_x))
predictions_count = predictions[:,-1]
test_date_df['count'] = predictions_count
test_date_df.to_csv('predictions_keras_nn.csv', index=False)

#Plotting training loss and validation loss to control overfitting
plt.plot(loss_history)
plt.plot(val_loss_history)
plt.show()
