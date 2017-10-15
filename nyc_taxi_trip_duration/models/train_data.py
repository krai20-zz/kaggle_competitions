from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import time

sample_size = 1000000
# combination of learning_rate and subsample are used for regularization/
# controlling overfitting
learning_rate = 0.1
subsample = 0.5
estimators = 100
target_var = 'trip_duration'
features = [
    'vendor_id',
    # 'avg_speed',
    'pickup_hour',
    # 'is_weekday',
    'pickup_day',
    # 'is_morning',
    # 'is_afternoon',
    # 'is_evening',
    # 'is_early_morning',
    'dist_kms',
    'passenger_count_0',
    'passenger_count_between_1_6',
    'passenger_count_between_7_9',
    'store_and_fwd_flag_is_N',
    'pickup_longitude',
    'pickup_latitude',
    'dropoff_longitude',
    'dropoff_latitude'
    ]

def read_data(filename):
    df = pd.read_csv(filename, compression='zip')
    return df

def get_log_trip_duration(df):
    """
    Kaggle uses RMSLE, so we transform trip duration
    to log trip duration
    """
    df['trip_duration'] = np.log(df['trip_duration'].values + 1)
    return df

def get_datetime_features(df):
    """
    Get datetime features
    """
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['{}_day'.format('pickup_datetime'.split('_')[0])] = df['pickup_datetime'].apply(lambda x: x.weekday())
    df['{}_hour'.format('pickup_datetime'.split('_')[0])] = df['pickup_datetime'].apply(lambda x: x.hour)
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
    df['is_morning'] = df.pickup_hour.apply(lambda x: True if x in range(7, 14) else False)
    df['is_afternoon'] = df.pickup_hour.apply(lambda x: True if x in range(14, 19) else False)
    df['is_evening'] = df.pickup_hour.apply(lambda x: True if x in range(19, 23) + [0] else False)
    df['is_early_morning'] = df.pickup_hour.apply(lambda x: True if x in range(1, 7) else False)
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

def bearing_array(row):
    lon1, lat1, lon2, lat2 = map(radians, [
        row['pickup_longitude'],
        row['pickup_latitude'],
        row['dropoff_longitude'],
        row['dropoff_latitude']
        ])
    AVG_EARTH_RADIUS = 6371  # in km
    lon_delta_rad = np.radians(lon2 - lon1)
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    y = np.sin(lon_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon_delta_rad)
    return np.degrees(np.arctan2(y, x))

def get_distance(df):
    """
    Calculates distance between pickup and dropoff destinations
    """
    df['dist_kms'] = df.apply(haversine_distance, axis=1)
    return df

def get_diretion(df):
    """ Calculates bearing (direction) in degrees
    """
    df['direction'] = df.apply(bearing_array, axis=1)
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
    df['store_and_fwd_flag_is_N'] = df.store_and_fwd_flag.apply(lambda x: True if x == 'N' else False)
    return df

def get_avg_speed(df):
    df['avg_speed'] = df['dist_kms']/df['trip_duration']
    return df

def get_regressor():
    reg = GradientBoostingRegressor(n_estimators=estimators, subsample=0.5,
                                    learning_rate=learning_rate)
    return reg

def get_cv_scores(reg, train_data, train_target):
    """
    Get cross validated R squared scores
    """
    scorer = make_scorer(r2_score, multioutput='variance_weighted')
    r2_scores = cross_val_score(reg, train_data, train_target, cv=4,
                            scoring=scorer)
    mean_squared_scores = cross_val_score(reg, train_data, train_target, cv=4,
                            scoring='mean_squared_error')
    mean_squared_scores = [sqrt(x*-1) for x in mean_squared_scores]
    return r2_scores, mean_squared_scores

def fit(reg, train_data, train_target):
    reg.fit(train_data,train_target)
    return reg

def split_train_test(df):
    data, target = df[features], df[target_var]
    train_data, test_data, train_target, test_target = train_test_split(data, target,
                                                        test_size=0.25)
    return train_data, test_data, train_target, test_target

def predict_n_measure(reg, test_data, test_target):
    pred_target = reg.predict(test_data)
    rmse = sqrt(mean_squared_error(test_target, pred_target))
    return rmse

def predict_kaggle_test_data(reg, test_data):
    ids = test_data['id'].tolist()
    test_data = test_data[features]
    pred_target = reg.predict(test_data)
    submission_df = pd.DataFrame(pred_target, ids).reset_index().rename(
                        columns={'index':'id', 0:'trip_duration'})
    submission_df['trip_duration'] = np.exp(submission_df['trip_duration']) - 1
    return submission_df

def remove_outliers(df):
    """
    Removes trip duration values above 3 std from mean
    """
    avg_d = df['trip_duration'].mean()
    dev_d = df['trip_duration'].std()
    df = df[df.trip_duration < (avg_d + 3*dev_d)]
    return df

def preprocesss(df):
    df = get_datetime_features(df)
    df = get_weekday_features(df)
    df = get_tod_features(df)
    df = get_distance(df)
    df = get_diretion(df)
    # df = get_avg_speed(df)
    df = get_passenger_count_features(df)
    df = get_store_flag_feature(df)
    # only needed for training set
    try:
        df = get_log_trip_duration(df)
    except KeyError:
        pass

    return df

def get_feature_importance(reg):
    importances = reg.feature_importances_
    importances = 100.0 * (importances / importances.max())
    imp_df = pd.DataFrame({
                    'features':features,
                    'relative_importances': importances,
                    })
    return imp_df

def plot_feature_imp(imp_df):
    imp_df = imp_df.sort_values('relative_importances', ascending=False)
    imp_df = imp_df.reset_index().drop('index', axis=1)
    sns_plot = sns.barplot(x=imp_df['relative_importances'], y=imp_df['features'])
    plt.tight_layout()
    plt.subplots_adjust(left=0.5)
    plt.savefig('../plots/feature_importances')

def train_xgboost(train_data, test_data, train_target, test_target):
    d_train = xgb.DMatrix(train_data, label=train_target)
    d_test = xgb.DMatrix(test_data, label=test_target)
    watchlist = [(d_train, 'train'), (d_test, 'test')]

    xgb_pars = {'min_child_weight': 50, 'eta': 0.3, 'colsample_bytree': 0.3,
                'max_depth': 10, 'subsample': 0.8, 'lambda': 1., 'nthread': -1,
                'booster' : 'gbtree', 'silent': 1, 'eval_metric': 'rmse',
                'objective': 'reg:linear'}

    model = xgb.train(xgb_pars, d_train, 200, watchlist, early_stopping_rounds=50,
                      maximize=False, verbose_eval=10)

    print "xgb model results", model
    return model

def get_xgb_submissions(kaggle_test_data, model):
    ids = kaggle_test_data['id'].tolist()
    kaggle_test_data = kaggle_test_data[features]
    kaggle_test_matrix = xgb.DMatrix(kaggle_test_data)
    pred_target = model.predict(kaggle_test_matrix)
    submission_df = pd.DataFrame(pred_target, ids).reset_index().rename(
                        columns={'index':'id', 0:'trip_duration'})
    submission_df['trip_duration'] = np.exp(submission_df['trip_duration']) - 1
    print "writing submissions file"
    submission_df.to_csv('../data/xgb_submissions.csv', index=False)
    print "Done writing submissions file"

def main():
    # read data
    print 'reading data'
    filename = '../data/train.zip'
    df = read_data(filename)
    print 'done reading data'

    df = df.sample(sample_size)

    # remove outliers
    df = remove_outliers(df)

    # build features
    df = preprocesss(df)

    # split train and test data
    train_data, test_data, train_target, test_target = split_train_test(df)

    # # get regressor
    # reg = get_regressor()
    #
    # # get cross validated r2_scores
    # print 'running cross validation'
    # start_time = time.time()
    # r2_scores, mean_squared_scores = get_cv_scores(reg, train_data, train_target)
    # print 'cross validated r2 scores', r2_scores, np.mean(r2_scores)
    # print 'cross validated rmse', mean_squared_scores, np.mean(mean_squared_scores)
    # print 'time taken', time.time() - start_time
    #
    # # fit data
    # print 'Fitting data'
    # start_time = time.time()
    # fit(reg, train_data, train_target)
    # print 'time taken', time.time() - start_time
    #
    # # plot feature important
    # imp_df = get_feature_importance(reg)
    # plot_feature_imp(imp_df)
    #
    # # predict and report metrics for test data
    # print 'predicting targets for test data'
    # rmse = predict_n_measure(reg, test_data, test_target)
    # print "test set rmse", rmse

    # train xgboost
    model = train_xgboost(train_data, test_data, train_target, test_target)

    # predict and report metrics for validation data
    # get validation set
    print "Reading test data"
    kaggle_test_file = '../data/test.zip'
    kaggle_test_data = read_data(kaggle_test_file)
    kaggle_test_df = preprocesss(kaggle_test_data)
    # submission_df = predict_kaggle_test_data(reg, kaggle_test_df)
    # submission_df.to_csv('../data/submission_data2.csv', index=False)
    print "fitting test data"
    get_xgb_submissions(kaggle_test_data, model)


if __name__ == "__main__":
    main()
