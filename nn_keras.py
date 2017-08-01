import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint
import keras.backend as KB

TOTAL_DATASET_SIZE = 10887
TRAIN_SIZE = 10000 #Total dataset size is 10887
TEST_SIZE = TOTAL_DATASET_SIZE - TRAIN_SIZE

HOURS_IN_DAY = 24
START_YEAR = 2011
DAYS_IN_YEAR = 365
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTHS_IN_YEAR = 12

hour_slope=[]
hour_peak=[]
hours_impact=[]
#hours_reg=[]
#hours_cas=[]

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

def get_day_of_week(datetime):
    total_day_count = get_total_day_count(datetime)
    day_of_week = np.mod(total_day_count+5,7) #0-Monday,6-Sunday
    day_of_week = day_of_week*31*np.pi/(6*18) #Refactoring day_of_week from 0-6 to 0-(31/18*PI) to make it possible to place days of week on circle
    return day_of_week

def get_humidity_impact(humidity):
    lin_part = [400 - 3.5 * i for i in range(101)]
    quad_part = [30 + 0.8 * x ** 2 for x in range(20)]
    if humidity < 20:
        return quad_part[humidity]
    else:
        return lin_part[humidity]

def get_month(datetime):
    date, _ = datetime.split(' ')
    _,month,_ = date.split('-') #['yyyy', 'mm', 'dd']
    return int(month)

def get_month_impact(datetime):
    date, _ = datetime.split(' ')
    _,month,_ = date.split('-') #['yyyy', 'mm', 'dd']
    return months_impact[int(month)-1]

def get_hour_impact(datetime):
    _, time = datetime.split(' ')
    hour,_,_ = time.split(':')  # ['hh', 'mm', 'ss']
    #hours_impact = [55,33,20,10,7,20,70,210,340,190,130,150,185,180,170,180,240,390,370,270,195,145,110,75]
    return hours_impact[int(hour)]

def get_hour_slope(datetime):
    #hour_slope = [-0.76,7,5,3,1,1,4,11,22,31,46,60,69,74,77,76,75,75,61,49,37,29,23,15]
    _, time = datetime.split(' ')
    hour,_,_ = time.split(':')  # ['hh', 'mm', 'ss']
    return hour_slope[int(hour)]

def get_hour_peak(datetime):
    #hour_peak = [35,21,13,6,4,17,68,191,320,160,83,91,119,110,90,102,166,318,308,217,155,116,88,59]
    hour_peak=[-0.7,-0.8,-0.9,-0.95,-1,-0.8,-0.4, 0.4, 1, -0.2, -0.8, -0.6, -0.5, -0.5, -0.6, -0.8, 0.2, 1, 1, 0.5, 0.1,-0.2, -0.4, -0.5,-0.6]
    _, time = datetime.split(' ')
    hour,_,_ = time.split(':')  # ['hh', 'mm', 'ss']
    return hour_peak[int(hour)]

def get_hour_registered(datetime):
    _, time = datetime.split(' ')
    hour,_,_ = time.split(':')  # ['hh', 'mm', 'ss']
    return hours_reg[int(hour)]

def get_hour_casual(datetime):
    _, time = datetime.split(' ')
    hour,_,_ = time.split(':')  # ['hh', 'mm', 'ss']
    return hours_cas[int(hour)]

def get_day_of_week_reg(day_of_week):
    return days_of_week_reg[int(day_of_week)]

def get_day_of_week_cas(day_of_week):
    return days_of_week_cas[int(day_of_week)]

def rmsle(y_true, y_pred):
    return KB.sqrt(KB.mean(KB.square(KB.log(y_pred+1) - KB.log(y_true+1)), axis=0))

df = pd.read_csv('data/train.csv')
df_to_predict = pd.read_csv('data/test.csv')

#Adding continous time as variable because of the increasing amount of bikes over time
df['cont_time'] = df.datetime.apply(datetime_to_total_hours)
df_to_predict['cont_time'] = df_to_predict.datetime.apply(datetime_to_total_hours)

#Adding hour temporarily
df['hour'] = df.datetime.apply(get_hour)
#df_to_predict['hour'] = df_to_predict.datetime.apply(get_hour)

#Little refactor of humidity to make it more linear
humidity_impact = np.array(df.groupby('humidity')['count'].mean())

#a = df.groupby('humidity')['count'].mean().plot()
#a.legend()

df['humidity'] = df.humidity.apply(get_humidity_impact)
df_to_predict['humidity'] = df_to_predict.humidity.apply(get_humidity_impact)
df['month'] = df.datetime.apply(get_month)
months_impact = np.array(df.groupby('month')['count'].mean())

#Getting month impact, which tells us how good is the month for bikes, far better than 'season'
df['month_impact'] = df.datetime.apply(get_month_impact)
df_to_predict['month_impact'] = df_to_predict.datetime.apply(get_month_impact)
#df_without_outliners = df[np.abs(df["count"]-df["count"].mean())<=(3*df["count"].std())]

df['day_of_week'] = df.datetime.apply(get_day_of_week)
df_to_predict['day_of_week'] = df_to_predict.datetime.apply(get_day_of_week)

days_of_week_reg = np.array(df.groupby('day_of_week')['registered'].mean())
days_of_week_cas = np.array(df.groupby('day_of_week')['casual'].mean())

df['day_of_week_reg'] = df.day_of_week.apply(get_day_of_week_reg)
df['day_of_week_cas'] = df.day_of_week.apply(get_day_of_week_cas)
df_to_predict['day_of_week_reg'] = df_to_predict.day_of_week.apply(get_day_of_week_reg)
df_to_predict['day_of_week_cas'] = df_to_predict.day_of_week.apply(get_day_of_week_cas)

hours_impact = np.array(df.groupby('hour')['count'].mean())
hours_cas = np.array(df.groupby('hour')['casual'].mean())
hours_reg = np.array(df.groupby('hour')['registered'].mean())

# hour_peak_ = np.array(df.groupby('hour')['diff'].mean())
# hour_slope_ = np.array(df.groupby('hour')['casual'].mean())

#Hour impact for registered and casual
df['hours_cas'] = df.datetime.apply(get_hour_casual)
df['hours_reg'] = df.datetime.apply(get_hour_registered)
df_to_predict['hours_cas'] = df_to_predict.datetime.apply(get_hour_casual)
df_to_predict['hours_reg'] = df_to_predict.datetime.apply(get_hour_registered)

#Hour impact for count
#df['hour_impact'] = df.datetime.apply(get_hour_impact)
#df_to_predict['hour_impact'] = df_to_predict.datetime.apply(get_hour_impact)
#Hour peak & slope _/\_/|_ and _--_
#df['hour_peak'] = df.datetime.apply(get_hour_peak)
#df['hour_slope'] = df.datetime.apply(get_hour_slope)
#df_to_predict['hour_peak'] = df_to_predict.datetime.apply(get_hour_peak)
#df_to_predict['hour_slope'] = df_to_predict.datetime.apply(get_hour_slope)

#hours_impact = np.array(df_without_outliners.groupby('hour')['count'].mean())

df['registered_norm'] = (df['registered'] - df['registered'].min() - (df['registered'].max() - df['registered'].min())/2) / ((df['registered'].max() - df['registered'].min())/2)
df['casual_norm'] = (df['casual'] - df['casual'].min() - (df['casual'].max() - df['casual'].min())/2) / ((df['casual'].max() - df['casual'].min())/2)
df['diff'] = df['registered_norm']-df['casual_norm']


# hour_peak = (hour_peak_ - hour_peak_.min() - (hour_peak_.max() - hour_peak_.min())/2) / ((hour_peak_.max() - hour_peak_.min())/2)
# hour_slope = (hour_slope_ - hour_slope_.min() - (hour_slope_.max() - hour_slope_.min())/2) / ((hour_slope_.max() - hour_slope_.min())/2)

#Data randomization(shuffling)
df = df.sample(frac=1).reset_index(drop=True)

#Spitting data into input features and labels
#datasetY = df.ix[:,'casual':'count']
datasetY = df.ix[:,'count']
#datasetX = df.drop(['casual','registered','count','datetime','windspeed','atemp','season','diff','registered_norm','casual_norm'],1)
datasetX = df.drop(['casual','registered','count','datetime','windspeed','atemp','season','diff','registered_norm','casual_norm','hour','day_of_week','month'],1)
datasetX_pred = df_to_predict.drop(['datetime','windspeed','atemp','season','day_of_week'],1)
print(datasetY.head(10))

#Normalizing inputs
datasetX = (datasetX - datasetX.min() - (datasetX.max() - datasetX.min())/2) / ((datasetX.max() - datasetX.min())/2)
datasetX_pred = (datasetX_pred - datasetX_pred.min() - (datasetX_pred.max() - datasetX_pred.min())/2) / ((datasetX_pred.max() - datasetX_pred.min())/2)


# datasetX['day_of_week_1'] = df.day_of_week.apply(lambda x:np.sin(x))
# datasetX['day_of_week_2'] = df.day_of_week.apply(lambda x:np.cos(x))
# datasetX_pred['day_of_week_1'] = df_to_predict.day_of_week.apply(lambda x:np.sin(x))
# datasetX_pred['day_of_week_2'] = df_to_predict.day_of_week.apply(lambda x:np.cos(x))

print(datasetX.head(10))
print(datasetX_pred.head(10))
#Dividing the original train dataset into train/test set
train_setX = datasetX
train_setY = datasetY
test_setX = datasetX.ix[TRAIN_SIZE-1:,:]
test_setY = datasetY.ix[TRAIN_SIZE-1:]
# train_setX = datasetX.ix[:TRAIN_SIZE-1,:]
# train_setY = datasetY.ix[:TRAIN_SIZE-1]
# test_setX = datasetX.ix[TRAIN_SIZE-1:,:]
# test_setY = datasetY.ix[TRAIN_SIZE-1:]

# datasetX.groupby('hour')[['hour_peak','hour_slope']].mean().plot()
# plt.show()

#Conversion from DF to numpyarray for Keras funcs
train_setX = np.array(train_setX)
train_setY = np.array(train_setY)
test_setX = np.array(test_setX)
test_setY = np.array(test_setY)

#Defining our NN model
model = Sequential()
model.add(Dense(units=9, input_dim=11))
model.add(Activation("tanh"))
model.add(Dense(units=9))
model.add(Activation("tanh"))
model.add(Dense(units=9))
model.add(Activation("tanh"))
model.add(Dense(units=9))
model.add(Activation("tanh"))
model.add(Dense(units=1))
model.add(Activation("relu"))
model.compile(loss=rmsle, optimizer='adam')
#model.compile(loss='mse', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
checkpoint = ModelCheckpoint('best_weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
history_callback = model.fit(train_setX, train_setY, epochs=20000, batch_size=50,validation_split=0.2,verbose=2,callbacks=callbacks_list)
loss_history = history_callback.history["loss"]
val_loss_history = history_callback.history["val_loss"]
#model.load_weights('best_weights.hdf5')
model.load_weights('best_weights.hdf5')

predictions = model.predict(np.array(datasetX_pred))

predictions_count = predictions[:,-1]
np.savetxt("predictions.csv", predictions_count, delimiter=",")
np.savetxt("all_predictions.csv", predictions, delimiter=",")
plt.plot(loss_history)
plt.plot(val_loss_history)
plt.show()

print(predictions)