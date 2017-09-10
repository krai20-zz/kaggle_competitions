from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import time

sample_size = 100000
target = 'trip_duration'
features = [
    'vendor_id',
    'is_weekday',
    'is_morning',
    'is_afternoon',
    'is_evening',
    'is_early_morning',
    'dist_kms',
    'passenger_count_0',
    'passenger_count_between_1_6',
    'passenger_count_between_7_9',
    'store_and_fwd_flag_is_N'
    ]

def read_data(filename):
    df = pd.read_csv(filename, compression='zip')
    return df

def get_datetime_features(df):
    """
    Get datetime features
    """
    for dt in ['pickup_datetime', 'dropoff_datetime']:
        df[dt] = pd.to_datetime(df[dt])
        df['{}_day'.format(dt.split('_')[0])] = df[dt].apply(lambda x: x.weekday())
        df['{}_hour'.format(dt.split('_')[0])] = df[dt].apply(lambda x: x.hour)
    return df

def get_weekday_features(df):
    """
    Calculates weekday features
    """
    df['is_weekday'] = df.pickup_day.apply(lambda x: True if x in range(0, 5) else False)
    return df

def get_tod_features(df):
    """
    Calculates time on day features
    Chosen times are based on trip duration by hour chart in notebook
    """
    df['is_morning'] = df.pickup_day.apply(lambda x: True if x in range(7, 14) else False)
    df['is_afternoon'] = df.pickup_day.apply(lambda x: True if x in range(14, 19) else False)
    df['is_evening'] = df.pickup_day.apply(lambda x: True if x in range(19, 23).append(0) else False)
    df['is_early_morning'] = df.pickup_day.apply(lambda x: True if x in range(1, 7).append(0) else False)
    return df

def haversine_distance(row):
    """
    Calculates the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [
        row['pickup_longitude'],
        row['pickup_latitude'],
        row['dropoff_longitude'],
        row['dropoff_latitude']
        ])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    return km

def get_distance(df):
    """
    Calculates distance between pickup and dropoff destinations
    """
    df['dist_kms'] = df.apply(haversine, axis=1)
    return df

def get_passenger_count_features(df):
    """
    Calculates passenger count features
    Chosen features are based on trip duration by passenger count chart in notebook
    """
    df['passenger_count_0'] = df.passenger_count.apply(lambda x: True if x == 0 else False)
    df['passenger_count_between_1_6'] = df.passenger_count.apply(lambda x: True if x in range(1, 7) else False)
    df['passenger_count_between_7_9',] = df.passenger_count.apply(lambda x: True if x in range(7, 10) else False)
    return df

def get_store_flag_feature(df):
    df['store_and_fwd_flag_is_N'] = df.store_and_fwd_flag.apply(lambda x: True if x == N else False)
    return df

def get_regressor():
    reg = linear_model.Lasso(alpha=0.1) # do hyperparameter search for alpha
    return reg

def get_cv_scores(clf, train_data, train_target):
    """
    Get cross validated R squared scores
    """
    r2_scores = cross_val_score(clf, train_data, train_target, cv=4,
                                      scoring='r2_score', multioutput='variance_weighted')
    return r2_scores

def fit(clf, train_data, train_target):
    reg.fit(train_data,train_target)
    return clf

def split_train_test(df):
    data, target = df[features], df[target]
    train_data, test_data, train_target, test_target = train_test_split(data, target,
                                                        test_size=0.25)
    return train_data, test_data, train_target, test_target

def predict_n_measure(reg, test_data, test_target):
    pred_target = reg.predict(test_data)
    r2_score = r2_score(test_target, pred_target)
    return r2_score

def remove_outliers(df):
    """
    Removes trip duration values above 3 std from mean
    """
    avg_d = df['trip_duration'].mean()
    dev_d = df['trip_duration'].std()
    df = df[df.trip_duration < (avg_d + 3*dev_d)]
    return df

def main():
    # read data
    print 'reading data'
    filename = '../data/train.zip'
    df = read_data(filename)
    print 'done reading data'

    # remove outliers
    df = remove_outliers(df)

    # build features
    df = get_datetime_features(df)
    df = get_weekday_features(df)
    df = get_tod_features(df)
    df = get_distance(df)
    df = get_passenger_count_features(df)
    df = get_store_flag_feature(df)

    # split train and test data
    train_data, test_data, train_target, test_target = split_train_test(df)

    # get regressor
    reg = get_regressor()

    # get cross validated r2_scores
    print 'running cross validation'
    start_time = time.time()
    r2_scores = get_cv_scores(clf, train_data, train_target)
    print 'cross validated accuracy scores', accuracy_scores, np.mean(accuracy_scores)
    print 'cross validated f1 scores', f1_scores, np.mean(f1_scores)
    print 'time taken', time.time() - start_time

    # fit data
    print 'Fitting data'
    start_time = time.time()
    fit(reg, train_data, train_target)
    print 'time taken', time.time() - start_time

    # predict and report metrics for test data
    print 'predicting tagets for test data'
    r2_scores = predict_n_measure(clf, test_data, test_target)

if __name__ == "__main__":
    main()
