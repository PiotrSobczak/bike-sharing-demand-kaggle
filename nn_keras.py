import pandas as pd
from matplotlib import pyplot as plt

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

def get_day_of_year(year,month,day_of_month):
    day_count = 0
    if year == '2012':
        MONTH_DAYS[1] = 29
    for i in range(int(month)-1):
        day_count += MONTH_DAYS[i]
    return day_count + int(day_of_month) -1

df = pd.read_csv('data/train.csv')

#Adding continous time as variable because of the increasing amount of bikes over time
#Adding day count to group data of one day(avg or sum) because plotting each hour is too messy
#Consider adding day of the week
df['cont_time'] = df.datetime.apply(datetime_to_total_hours)
df['day_count'] = df.datetime.apply(get_total_day_count)
df = df.drop('datetime',1)

df.groupby('day_count')[['temp','count']].mean().plot()
plt.show()

#Data randomization(shuffling)
df = df.sample(frac=1)

#Spitting data into input and output
datasetY = df.ix[:,'casual':'count']
datasetX = df.drop(['casual','registered','count'],1)

#Normalizing input
datasetX = (datasetX - datasetX.min() - (datasetX.max() - datasetX.min())/2) / ((datasetX.max() - datasetX.min())/2)
#print(X.head(10))

#Dividing the original train dataset into train/test set
train_setX = datasetX.ix[:TRAIN_SIZE,:]
train_setY = datasetY.ix[:TRAIN_SIZE,:]
test_setX = datasetX.ix[TRAIN_SIZE,:]
test_setY = datasetY.ix[TRAIN_SIZE,:]
