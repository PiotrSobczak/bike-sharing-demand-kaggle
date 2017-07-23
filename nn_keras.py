import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

TOTAL_DATASET_SIZE = 10887
TRAIN_SIZE = 10000 #Total dataset size is 10887
TEST_SIZE = TOTAL_DATASET_SIZE - TRAIN_SIZE

HOURS_IN_DAY = 24
START_YEAR = 2011
DAYS_IN_YEAR = 365
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTHS_IN_YEAR = 12

def get_total_day_count(datetime):
    date,time = datetime.split(' ')
    year,month,day_of_month = date.split('-') #['yyyy', 'mm', 'dd']
    time_split = time.split(':') #['hh', 'mm', 'ss']
    day_count = (int(year) - START_YEAR)*DAYS_IN_YEAR + get_day_of_year(year,month,day_of_month)
    return day_count

def datetime_to_total_hours(datetime):
    date,time = datetime.split(' ')
    year,month,day_of_month = date.split('-') #['yyyy', 'mm', 'dd']
    time_split = time.split(':') #['hh', 'mm', 'ss']
    cont_time = (int(year) - START_YEAR)*DAYS_IN_YEAR*HOURS_IN_DAY + get_day_of_year(year,month,day_of_month)*HOURS_IN_DAY + int(time_split[0])
    return cont_time

def get_hour(datetime):
    _, time = datetime.split(' ')
    time_split = time.split(':')  # ['hh', 'mm', 'ss']
    return int(time_split[0])

def get_day_of_year(year,month,day_of_month):
    day_count = 0
    if year == '2012':
        MONTH_DAYS[1] = 29
    for i in range(int(month)-1):
        day_count += MONTH_DAYS[i]
    return day_count + int(day_of_month) -1

df = pd.read_csv('data/train.csv')
df_to_predict = pd.read_csv('data/test.csv')

#Adding continous time as variable because of the increasing amount of bikes over time
#Adding day count to group data of one day(avg or sum) because plotting each hour is too messy
#Consider adding day of the week
df['cont_time'] = df.datetime.apply(datetime_to_total_hours)
df['hour'] = df.datetime.apply(get_hour)
df = df.drop('datetime',1)

df_to_predict['cont_time'] = df_to_predict.datetime.apply(datetime_to_total_hours)
df_to_predict['hour'] = df_to_predict.datetime.apply(get_hour)
df_to_predict = df_to_predict.drop('datetime',1)

#df['day_count'] = df.datetime.apply(get_total_day_count)

#df = df.drop('season',1)
#df = df.drop('humidity',1)
#df.groupby('day_count')[['temp','count']].mean().plot()
#plt.show()

#Data randomization(shuffling)
df = df.sample(frac=1).reset_index(drop=True)

#Spitting data into input and output
datasetY = df.ix[:,'casual':'count']
datasetX = df.drop(['casual','registered','count'],1)

#Normalizing inputs
datasetX = (datasetX - datasetX.min() - (datasetX.max() - datasetX.min())/2) / ((datasetX.max() - datasetX.min())/2)
df_to_predict = (df_to_predict - df_to_predict.min() - (df_to_predict.max() - df_to_predict.min())/2) / ((df_to_predict.max() - df_to_predict.min())/2)
#print(X.head(10))

#Dividing the original train dataset into train/test set
train_setX = datasetX.ix[:TRAIN_SIZE-1,:]
train_setY = datasetY.ix[:TRAIN_SIZE-1,:]
test_setX = datasetX.ix[TRAIN_SIZE-1:,:]
test_setY = datasetY.ix[TRAIN_SIZE-1:,:]

#Conversion from DF to numpyarray for Keras funcs
train_setX = np.array(train_setX)
train_setY = np.array(train_setY)
test_setX = np.array(test_setX)
test_setY = np.array(test_setY)

print(train_setX.shape)
print(train_setY.shape)
print(test_setX.shape)
print(test_setY.shape)

#Defining our NN model
model = Sequential()
model.add(Dense(units=55, input_dim=10))
model.add(Activation("tanh"))
model.add(Dense(units=33))
model.add(Activation("tanh"))
model.add(Dense(units=11))
model.add(Activation("tanh"))
model.add(Dense(units=3))
model.add(Activation("linear"))
model.compile(loss='mean_squared_error', optimizer='adam')
history_callback = model.fit(train_setX, train_setY, epochs=1000, batch_size=50)
loss_history = history_callback.history["loss"]
print(loss_history)
plt.plot(loss_history)
plt.show()
e = model.evaluate(test_setX, test_setY)
print (' Evaluation: ', e)
predictions = model.predict(np.array(df_to_predict))
predictions_count = predictions[:,-1]
np.savetxt("predictions.csv", predictions_count, delimiter=",")