import numpy as np

HOURS_IN_DAY = 24
START_YEAR = 2011
DAYS_IN_YEAR = 365
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTHS_IN_YEAR = 12

class FeatureUtils:
    hour_peak=[]
    months_impact=[]
    hours_impact=[]
    hour_slope=[]
    hours_sat=[]
    hours_sun=[]
    hours_workday=[]
    hours_freeday=[]
    hours_cas=[]
    hours_reg = []
    days_of_week_cas=[]
    days_of_week_reg=[]

    @staticmethod
    def get_total_day_count(datetime):
        date,time = datetime.split(' ')
        year,month,day_of_month = date.split('-') #['yyyy', 'mm', 'dd']
        time_split = time.split(':') #['hh', 'mm', 'ss']
        day_count = (int(year) - START_YEAR)*DAYS_IN_YEAR + FeatureUtils.get_day_of_year(year,month,day_of_month)
        return day_count

    @staticmethod
    def datetime_to_total_hours(datetime):
        date,time = datetime.split(' ')
        year,month,day_of_month = date.split('-') #['yyyy', 'mm', 'dd']
        time_split = time.split(':') #['hh', 'mm', 'ss']
        cont_time = (int(year) - START_YEAR)*DAYS_IN_YEAR*HOURS_IN_DAY + FeatureUtils.get_day_of_year(year,month,day_of_month)*HOURS_IN_DAY + int(time_split[0])
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
        total_day_count = FeatureUtils.get_total_day_count(datetime)
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
        return FeatureUtils.months_impact[int(month)-1]

    @staticmethod
    def get_hour_impact(datetime):
        _, time = datetime.split(' ')
        hour,_,_ = time.split(':')  # ['hh', 'mm', 'ss']
        #hours_impact = [55,33,20,10,7,20,70,210,340,190,130,150,185,180,170,180,240,390,370,270,195,145,110,75]
        return FeatureUtils.hours_impact[int(hour)]

    @staticmethod
    def get_hour_slope(datetime):
        #hour_slope = [-0.76,7,5,3,1,1,4,11,22,31,46,60,69,74,77,76,75,75,61,49,37,29,23,15]
        _, time = datetime.split(' ')
        hour,_,_ = time.split(':')  # ['hh', 'mm', 'ss']
        return FeatureUtils.hour_slope[int(hour)]

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
        return FeatureUtils.hours_reg[int(hour)]

    @staticmethod
    def get_hour_casual(datetime):
        _, time = datetime.split(' ')
        hour,_,_ = time.split(':')  # ['hh', 'mm', 'ss']
        return FeatureUtils.hours_cas[int(hour)]

    @staticmethod
    def get_hour_work(datetime):
        _, time = datetime.split(' ')
        hour,_,_ = time.split(':')  # ['hh', 'mm', 'ss']
        return FeatureUtils.hours_workday[int(hour)]

    @staticmethod
    def get_hour_free(datetime):
        _, time = datetime.split(' ')
        hour,_,_ = time.split(':')  # ['hh', 'mm', 'ss']
        return FeatureUtils.hours_freeday[int(hour)]

    @staticmethod
    def get_hour_sun(datetime):
        _, time = datetime.split(' ')
        hour,_,_ = time.split(':')  # ['hh', 'mm', 'ss']
        return FeatureUtils.hours_sun[int(hour)]

    @staticmethod
    def get_hour_sat(datetime):
        _, time = datetime.split(' ')
        hour,_,_ = time.split(':')  # ['hh', 'mm', 'ss']
        return FeatureUtils.hours_sat[int(hour)]

    @staticmethod
    def get_day_of_week_reg(day_of_week):
        return FeatureUtils.days_of_week_reg[int(day_of_week)]

    @staticmethod
    def get_day_of_week_cas(day_of_week):
        return FeatureUtils.days_of_week_cas[int(day_of_week)]

    @staticmethod
    def get_year(datetime):
        date, _ = datetime.split(' ')
        year,_,_ = date.split('-') #['yyyy', 'mm', 'dd']
        return int(year)