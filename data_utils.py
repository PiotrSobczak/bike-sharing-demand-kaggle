import numpy as np
import pandas as pd

HOURS_IN_DAY = 24
START_YEAR = 2011
DAYS_IN_YEAR = 365
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTHS_IN_YEAR = 12


class DataUtils:
    @staticmethod
    def get_total_day_count(date):
        #date, time = datetime.split(' ')
        year, month, day_of_month = date.year,date.month,date.day  # ['yyyy', 'mm', 'dd']
        day_count = (int(year) - START_YEAR) * DAYS_IN_YEAR + DataUtils.get_day_of_year(year, month, day_of_month)
        return day_count

    @staticmethod
    def datetime_to_total_days(date):
        day_of_month, month, year = date.day,date.month,date.year # ['yyyy', 'mm', 'dd']
        cont_time = (int(year) - START_YEAR) * DAYS_IN_YEAR * HOURS_IN_DAY + DataUtils.get_day_of_year(year, month,day_of_month)
        return cont_time

    @staticmethod
    def get_hour(datetime):
        _, time = datetime.split(' ')
        time_split = time.split(':')  # ['hh', 'mm', 'ss']
        return int(time_split[0])

    @staticmethod
    def get_day_of_year(year, month, day_of_month):
        day_count = 0
        if year == '2012':
            MONTH_DAYS[1] = 29
        for i in range(int(month) - 1):
            day_count += MONTH_DAYS[i]
        #print("inside get_day_of_year", day_count + int(day_of_month) - 1,year,month,day_of_month)
        return (day_count + int(day_of_month) - 1)

    @staticmethod
    def get_day_of_week(date):
        total_day_count = DataUtils.get_total_day_count(date)
        day_of_week = np.mod(total_day_count + 5, 7)  # 0-Monday,6-Sunday
        # day_of_week = day_of_week*31*np.pi/(6*18) #Refactoring day_of_week from 0-6 to 0-(31/18*PI) to make it possible to place days of week on circle
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
        _, month, _ = date.split('-')  # ['yyyy', 'mm', 'dd']
        return int(month)

    @staticmethod
    def get_month_impact(datetime):
        date, _ = datetime.split(' ')
        _, month, _ = date.split('-')  # ['yyyy', 'mm', 'dd']
        return DataUtils.months_impact[int(month) - 1]

    @staticmethod
    def get_hour_impact(datetime):
        _, time = datetime.split(' ')
        hour, _, _ = time.split(':')  # ['hh', 'mm', 'ss']
        # hours_impact = [55,33,20,10,7,20,70,210,340,190,130,150,185,180,170,180,240,390,370,270,195,145,110,75]
        return DataUtils.hours_impact[int(hour)]

    @staticmethod
    def get_hour_slope(datetime):
        # hour_slope = [-0.76,7,5,3,1,1,4,11,22,31,46,60,69,74,77,76,75,75,61,49,37,29,23,15]
        _, time = datetime.split(' ')
        hour, _, _ = time.split(':')  # ['hh', 'mm', 'ss']
        return DataUtils.hour_slope[int(hour)]

    @staticmethod
    def get_hour_peak(datetime):
        # hour_peak = [35,21,13,6,4,17,68,191,320,160,83,91,119,110,90,102,166,318,308,217,155,116,88,59]
        hour_peak = [-0.7, -0.8, -0.9, -0.95, -1, -0.8, -0.4, 0.4, 1, -0.2, -0.8, -0.6, -0.5, -0.5, -0.6, -0.8, 0.2, 1,
                     1, 0.5, 0.1, -0.2, -0.4, -0.5, -0.6]
        _, time = datetime.split(' ')
        hour, _, _ = time.split(':')  # ['hh', 'mm', 'ss']
        return hour_peak[int(hour)]

    @staticmethod
    def get_hour_registered(datetime):
        _, time = datetime.split(' ')
        hour, _, _ = time.split(':')  # ['hh', 'mm', 'ss']
        return DataUtils.hours_reg[int(hour)]

    @staticmethod
    def get_hour_casual(datetime):
        _, time = datetime.split(' ')
        hour, _, _ = time.split(':')  # ['hh', 'mm', 'ss']
        return DataUtils.hours_cas[int(hour)]

    @staticmethod
    def get_hour_work(datetime):
        _, time = datetime.split(' ')
        hour, _, _ = time.split(':')  # ['hh', 'mm', 'ss']
        return DataUtils.hours_workday[int(hour)]

    @staticmethod
    def get_hour_free(datetime):
        _, time = datetime.split(' ')
        hour, _, _ = time.split(':')  # ['hh', 'mm', 'ss']
        return DataUtils.hours_freeday[int(hour)]

    @staticmethod
    def get_hour_sun(datetime):
        _, time = datetime.split(' ')
        hour, _, _ = time.split(':')  # ['hh', 'mm', 'ss']
        return DataUtils.hours_sun[int(hour)]

    @staticmethod
    def get_hour_sat(datetime):
        _, time = datetime.split(' ')
        hour, _, _ = time.split(':')  # ['hh', 'mm', 'ss']
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
        year, _, _ = date.split('-')  # ['yyyy', 'mm', 'dd']
        return int(year)

    def get_day_of_month(datetime):
        print(datetime,type(datetime))
        date, _ = datetime.split(' ')
        _, _, day = date.split('-')  # ['yyyy', 'mm', 'dd']
        return int(day)

    @staticmethod
    def norm_arr(array):
        return (array - array.min() - (array.max() - array.min()) / 2) / ((array.max() - array.min()) / 2)

    @staticmethod
    def save_to_csv(file_name, elements):
        with open(file_name, 'ab') as f:
            for elem in elements:
                np.savetxt(f, elem, delimiter=',')

    @staticmethod
    def sort_df(df, col_name, asc=1):
        return df.sort_values([col_name], ascending=[asc])

    @staticmethod
    def get_sep_datasets(X, Y, size_train, reverse_data_order=False):
        if reverse_data_order is True:
            size_val = X.shape[0] - size_train
            # print("val size:",size_val)
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
        return X_train, Y_train, Y_train_log, X_val, Y_val

    @staticmethod
    def get_processed_df(train_path, test_path):
        # Reading datasets
        df_train = pd.read_csv(train_path)
        df_train['dataset'] = -1
        df_test = pd.read_csv(test_path)
        df_test['dataset'] = 1

        df = df_train.append(df_test)

        dt = pd.DatetimeIndex(df['datetime'])

        df['date'] = dt.date
        df['day_of_month'] = dt.day
        df['month'] = dt.month
        df['year'] = dt.year
        df['hour'] = dt.hour
        #df['day_of_week'] = dt.dayofweek #Does not include  Feb 29th
        # df['week_of_year'] = dt.weekofyear

        df['day_of_week'] = df.date.apply(DataUtils.get_day_of_week)
        df['cont_time'] = df.date.apply(DataUtils.datetime_to_total_days)

        # Getting month impact, which tells us how good is the month for bikes, far better than 'season' and is easier to learn than pure month value
        DataUtils.months_impact = np.array(df.groupby('month')['count'].mean())
        df['month_impact'] = df.datetime.apply(DataUtils.get_month_impact)

        # DAY OF WEEK REG/CAS
        DataUtils.days_of_week_reg = np.array(df.groupby('day_of_week')['registered'].mean())
        DataUtils.days_of_week_cas = np.array(df.groupby('day_of_week')['casual'].mean())

        df['day_of_week_reg'] = df.day_of_week.apply(DataUtils.get_day_of_week_reg)
        df['day_of_week_cas'] = df.day_of_week.apply(DataUtils.get_day_of_week_cas)

        # Hour impact array
        DataUtils.hours_impact = np.array(df.groupby('hour')['count'].mean())

        # Hour impact arrays for registered and casual
        DataUtils.hours_cas = np.array(df.groupby('hour')['casual'].mean())
        DataUtils.hours_reg = np.array(df.groupby('hour')['registered'].mean())

        # Hour impact for registered and casual
        df['hour_reg'] = df.datetime.apply(DataUtils.get_hour_registered)
        df['hour_cas'] = df.datetime.apply(DataUtils.get_hour_casual)

        # Data randomization(shuffling)
        # df = df.sample(frac=1).reset_index(drop=True)

        # Logarithmic transformation
        df['count_log'] = np.log(df[['count']] + 1)

        # Spitting data into input features and labels
        features = ['year', 'month_impact', 'day_of_week_reg', 'day_of_week_cas', 'cont_time', 'hour',
                    'hour_reg', 'hour_cas', 'workingday', 'holiday', 'temp', 'humidity', 'weather', 'dataset', 'day_of_month']

        df_train_X = df.loc[df['dataset'] == -1, features]
        df_train_Y = df.loc[(df['dataset'] == -1) & (df['day_of_month'] < 16), 'count']
        df_val_Y = df.loc[(df['dataset'] == -1) & (df['day_of_month'] >= 16), 'count']
        df_test_X = df.loc[df['dataset'] == 1, features]
        df_Y = df.loc[df['dataset'] == -1, 'count']
        # Setting validation set
        df_train_X.loc[(df_train_X['dataset'] == -1) & (df_train_X['day_of_month'] >= 16), 'dataset'] = 0

        # Normalizing inputs
        df_merged = df_train_X.append(df_test_X)
        df_merged = (df_merged - df_merged.min() - (df_merged.max() - df_merged.min()) / 2) / ((df_merged.max() - df_merged.min()) / 2)

        # Recovering train & test sets
        df_train_X = df_merged[df_merged['dataset'] == -1].drop(['dataset','day_of_month'], 1)
        df_val_X = df_merged[df_merged['dataset'] == 0].drop(['dataset','day_of_month'], 1)
        df_test_X = df_merged[df_merged['dataset'] == 1].drop(['dataset','day_of_month'], 1)
        df_X = df_merged[df_merged['dataset'] != 1].drop(['dataset','day_of_month'], 1)

        dataset_pred_date = df_test[['datetime']]

        return df_X,df_Y,df_train_X, df_train_Y, df_val_X, df_val_Y, df_test_X, dataset_pred_date