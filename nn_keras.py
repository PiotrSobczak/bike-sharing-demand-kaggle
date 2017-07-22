import pandas as pd
from matplotlib import pyplot as plt

HOURS_IN_DAY = 24
START_YEAR = 2011
DAYS_IN_YEAR = 365
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTHS_IN_YEAR = 12

def get_day_count(datetime):
    date,time = datetime.split(' ')
    year,month,day_of_month = date.split('-') #['yyyy', 'mm', 'dd']
    time_split = time.split(':') #['hh', 'mm', 'ss']
    day_count = (int(year) - START_YEAR)*DAYS_IN_YEAR + get_day_of_year(year,month,day_of_month)
    return day_count

def to_hours(datetime):
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
df['cont_time'] = df.datetime.apply(to_hours)
df['day_count'] = df.datetime.apply(get_day_count)
df.drop('datetime',1)

df.plot(x="cont_time", y='temp')
plt.show()
df.groupby('day_count')[['temp','count']].mean().plot()
plt.show()
print(df.head(100))