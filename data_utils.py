import numpy as np
import pandas as pd

HOURS_IN_DAY = 24
START_YEAR = 2011
DAYS_IN_YEAR = 365
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTHS_IN_YEAR = 12

class DataUtils:
    @staticmethod
    def get_total_day_count(datetime):
        date,time = datetime.split(' ')
        year,month,day_of_month = date.split('-') #['yyyy', 'mm', 'dd']
        time_split = time.split(':') #['hh', 'mm', 'ss']
        day_count = (int(year) - START_YEAR)*DAYS_IN_YEAR + DataUtils.get_day_of_year(year,month,day_of_month)
        return day_count

    @staticmethod
    def datetime_to_total_hours(datetime):
        date,time = datetime.split(' ')
        year,month,day_of_month = date.split('-') #['yyyy', 'mm', 'dd']
        time_split = time.split(':') #['hh', 'mm', 'ss']
        cont_time = (int(year) - START_YEAR)*DAYS_IN_YEAR*HOURS_IN_DAY + DataUtils.get_day_of_year(year,month,day_of_month)*HOURS_IN_DAY + int(time_split[0])
        return cont_time

    @staticmethod
    def get_hour(datetime):
        _, time = datetime.split(' ')
        time_split = time.split(':')  # ['hh', 'mm', 'ss']
        return int(time_split[0])

    @staticmethod
    def get_day_of_year(year,month,day_of_month):
        day_count = 0
        if year == '2012':
            MONTH_DAYS[1] = 29
        for i in range(int(month)-1):
            day_count += MONTH_DAYS[i]
        return day_count + int(day_of_month) -1

    @staticmethod
    def get_day_of_week(datetime):
        total_day_count = DataUtils.get_total_day_count(datetime)
        day_of_week = np.mod(total_day_count+5,7) #0-Monday,6-Sunday
        #day_of_week = day_of_week*31*np.pi/(6*18) #Refactoring day_of_week from 0-6 to 0-(31/18*PI) to make it possible to place days of week on circle
        return day_of_week

    @staticmethod
    def get_humidity_impact(humidity):
        lin_part = [400 - 3.5 * i for i in range(101)]
        quad_part = [30 + 0.8 * x ** 2 for x in range(20)]
        if humidity < 20:
            return quad_part[humidity]
        else:
            return lin_part[humidity]

    @staticmethod
    def get_month(datetime):
        date, _ = datetime.split(' ')
        _,month,_ = date.split('-') #['yyyy', 'mm', 'dd']
        return int(month)

    @staticmethod
    def get_month_impact(datetime):
        date, _ = datetime.split(' ')
        _,month,_ = date.split('-') #['yyyy', 'mm', 'dd']
        return DataUtils.months_impact[int(month)-1]

    @staticmethod
    def get_hour_impact(datetime):
        _, time = datetime.split(' ')
        hour,_,_ = time.split(':')  # ['hh', 'mm', 'ss']
        #hours_impact = [55,33,20,10,7,20,70,210,340,190,130,150,185,180,170,180,240,390,370,270,195,145,110,75]
        return DataUtils.hours_impact[int(hour)]

    @staticmethod
    def get_hour_slope(datetime):
        #hour_slope = [-0.76,7,5,3,1,1,4,11,22,31,46,60,69,74,77,76,75,75,61,49,37,29,23,15]
        _, time = datetime.split(' ')
        hour,_,_ = time.split(':')  # ['hh', 'mm', 'ss']
        return DataUtils.hour_slope[int(hour)]

    @staticmethod
    def get_hour_peak(datetime):
        #hour_peak = [35,21,13,6,4,17,68,191,320,160,83,91,119,110,90,102,166,318,308,217,155,116,88,59]
        hour_peak=[-0.7,-0.8,-0.9,-0.95,-1,-0.8,-0.4, 0.4, 1, -0.2, -0.8, -0.6, -0.5, -0.5, -0.6, -0.8, 0.2, 1, 1, 0.5, 0.1,-0.2, -0.4, -0.5,-0.6]
        _, time = datetime.split(' ')
        hour,_,_ = time.split(':')  # ['hh', 'mm', 'ss']
        return hour_peak[int(hour)]

    @staticmethod
    def get_hour_registered(datetime):
        _, time = datetime.split(' ')
        hour,_,_ = time.split(':')  # ['hh', 'mm', 'ss']
        return DataUtils.hours_reg[int(hour)]

    @staticmethod
    def get_hour_casual(datetime):
        _, time = datetime.split(' ')
        hour,_,_ = time.split(':')  # ['hh', 'mm', 'ss']
        return DataUtils.hours_cas[int(hour)]

    @staticmethod
    def get_hour_work(datetime):
        _, time = datetime.split(' ')
        hour,_,_ = time.split(':')  # ['hh', 'mm', 'ss']
        return DataUtils.hours_workday[int(hour)]

    @staticmethod
    def get_hour_free(datetime):
        _, time = datetime.split(' ')
        hour,_,_ = time.split(':')  # ['hh', 'mm', 'ss']
        return DataUtils.hours_freeday[int(hour)]

    @staticmethod
    def get_hour_sun(datetime):
        _, time = datetime.split(' ')
        hour,_,_ = time.split(':')  # ['hh', 'mm', 'ss']
        return DataUtils.hours_sun[int(hour)]

    @staticmethod
    def get_hour_sat(datetime):
        _, time = datetime.split(' ')
        hour,_,_ = time.split(':')  # ['hh', 'mm', 'ss']
        return DataUtils.hours_sat[int(hour)]

    @staticmethod
    def get_day_of_week_reg(day_of_week):
        return DataUtils.days_of_week_reg[int(day_of_week)]

    @staticmethod
    def get_day_of_week_cas(day_of_week):
        return DataUtils.days_of_week_cas[int(day_of_week)]

    @staticmethod
    def get_year(datetime):
        date, _ = datetime.split(' ')
        year,_,_ = date.split('-') #['yyyy', 'mm', 'dd']
        return int(year)

    @staticmethod
    def norm_arr(array):
        return (array - array.min() - (array.max() - array.min()) / 2) / ((array.max() - array.min()) / 2)

    @staticmethod
    def get_sep_datasets(X,Y,size_train,reverse_data_order = False):
        if reverse_data_order is True:
            size_val = X.shape[0] - size_train
            X_train = X[size_val:]
            Y_train_log = Y[size_val:, 1]
            Y_train = Y[size_val:, 0]
            X_val = X[:size_val]
            Y_val = Y[:size_val, 0]
        else:
            X_train = X[:size_train]
            Y_train_log = Y[:size_train, 1]
            Y_train = Y[:size_train, 0]
            X_val = X[size_train:]
            Y_val = Y[size_train:, 0]
        return X_train,Y_train,Y_train_log,X_val,Y_val

    @staticmethod
    def get_processed_df(train_path,test_path):

        # Reading datasets
        df = pd.read_csv(train_path)
        df_to_predict = pd.read_csv(test_path)

        # Adding continous time as variable because of the increasing amount of bikes over time
        df['cont_time'] = df.datetime.apply(DataUtils.datetime_to_total_hours)
        df_to_predict['cont_time'] = df_to_predict.datetime.apply(DataUtils.datetime_to_total_hours)

        # Adding hour temporarily
        df['hour'] = df.datetime.apply(DataUtils.get_hour)
        df_to_predict['hour'] = df_to_predict.datetime.apply(DataUtils.get_hour)

        # Little refactor of humidity to make it easier to learn
        DataUtils.humidity_impact = np.array(df.groupby('humidity')['count'].mean())

        # Month
        df['month'] = df.datetime.apply(DataUtils.get_month)

        # Getting month impact, which tells us how good is the month for bikes, far better than 'season' and is easier to learn than pure month value
        DataUtils.months_impact = np.array(df.groupby('month')['count'].mean())
        df['month_impact'] = df.datetime.apply(DataUtils.get_month_impact)
        df_to_predict['month_impact'] = df_to_predict.datetime.apply(DataUtils.get_month_impact)

        # Year
        df['year'] = df.datetime.apply(DataUtils.get_year)
        df_to_predict['year'] = df_to_predict.datetime.apply(DataUtils.get_year)

        # Day of week
        df['day_of_week'] = df.datetime.apply(DataUtils.get_day_of_week)
        df_to_predict['day_of_week'] = df_to_predict.datetime.apply(DataUtils.get_day_of_week)

        # DAY OF WEEK REG/CAS
        DataUtils.days_of_week_reg = np.array(df.groupby('day_of_week')['registered'].mean())
        DataUtils.days_of_week_cas = np.array(df.groupby('day_of_week')['casual'].mean())
        df['day_of_week_reg'] = df.day_of_week.apply(DataUtils.get_day_of_week_reg)
        df['day_of_week_cas'] = df.day_of_week.apply(DataUtils.get_day_of_week_cas)
        df_to_predict['day_of_week_reg'] = df_to_predict.day_of_week.apply(DataUtils.get_day_of_week_reg)
        df_to_predict['day_of_week_cas'] = df_to_predict.day_of_week.apply(DataUtils.get_day_of_week_cas)

        # Hour impact array
        DataUtils.hours_impact = np.array(df.groupby('hour')['count'].mean())

        # Hour impact arrays for registered and casual
        DataUtils.hours_cas = np.array(df.groupby('hour')['casual'].mean())
        DataUtils.hours_reg = np.array(df.groupby('hour')['registered'].mean())

        # Hour impact arrays for workingday, freeday, sat, sun
        DataUtils.hours_workday = DataUtils.norm_arr(df.loc[df['workingday'] == 1].groupby('hour')['count'].mean())
        DataUtils.hours_freeday = DataUtils.norm_arr(
            df.loc[(df['workingday'] == 0) & (df['day_of_week'] < 5)].groupby('hour')['count'].mean())
        DataUtils.hours_sat = DataUtils.norm_arr(df.loc[df['day_of_week'] == 5].groupby('hour')['count'].mean())
        DataUtils.hours_sun = DataUtils.norm_arr(df.loc[df['day_of_week'] == 6].groupby('hour')['count'].mean())

        # Hour impact for registered and casual
        df['hour_reg'] = df.datetime.apply(DataUtils.get_hour_registered)
        df['hour_cas'] = df.datetime.apply(DataUtils.get_hour_casual)
        df_to_predict['hour_reg'] = df_to_predict.datetime.apply(DataUtils.get_hour_registered)
        df_to_predict['hour_cas'] = df_to_predict.datetime.apply(DataUtils.get_hour_casual)
        # print(df.head(10))

        # Data randomization(shuffling)
        df = df.sample(frac=1).reset_index(drop=True)
        # print(df.head(30))

        #Logarithmic transformation
        df['count_log'] = np.log(df[['count']] + 1)

        # Spitting data into input features and labels
        features = ['year', 'month_impact', 'day_of_week_reg', 'day_of_week_cas', 'cont_time',
                    'hour_reg', 'hour_cas', 'workingday', 'holiday', 'temp', 'humidity', 'weather']

        datasetX = df[features]
        datasetX_pred = df_to_predict[features]
        datasetY = df[['count','count_log']].astype(float)

        #datasetX = df.drop(['casual', 'registered', 'count', 'datetime', 'windspeed', 'atemp', 'season', 'month'], 1)
        #datasetX_pred = df_to_predict.drop(['datetime', 'windspeed', 'atemp', 'season'], 1)
        # print(datasetY.head(10))

        # Normalizing inputs
        datasetX = (datasetX - datasetX.min() - (datasetX.max() - datasetX.min()) / 2) / (
        (datasetX.max() - datasetX.min()) / 2)
        datasetX_pred = (datasetX_pred - datasetX_pred.min() - (datasetX_pred.max() - datasetX_pred.min()) / 2) / (
        (datasetX_pred.max() - datasetX_pred.min()) / 2)
        #print(df_to_predict[['datetime']])
        dataset_pred_date = df_to_predict[['datetime']]

        #datasetX = datasetX.drop(['day_of_week'], 1)
        #datasetX_pred = datasetX_pred.drop(['day_of_week'], 1)

        print("DF loaded, columns:", datasetX.columns.values,", shape:", datasetX.shape)

        return datasetX,datasetY,datasetX_pred,dataset_pred_date