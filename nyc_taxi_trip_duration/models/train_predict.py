from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from math import sqrt
import time
from build_features import preprocess


SAMPLE_SIZE = 1000000
TARGET_VAR = 'trip_duration'
FEATURES = [
    'vendor_id',
    'pickup_hour',
    'pickup_day',
    'dist_kms',
    'passenger_count',
    'store_and_fwd_flag_is_N',
    'pickup_longitude',
    'pickup_latitude',
    'dropoff_longitude',
    'dropoff_latitude',
    'direction'
    ]

XGB_PARAMS = {
    'min_child_weight':50,
    'colsample_bytree':0.3,
    'max_depth':10,
    'subsample':0.8,
    'booster':'gbtree',
    'silent':1,
    'eval_metric':'rmse',
    'objective': 'reg:linear'
    }

# combination of learning_rate and subsample are used for regularization/
# controlling overfitting
GBR_PARAMS = {
    'learning_rate':0.1,
    'subsample':0.5,
    'n_estimators':100
    }

def read_data(filename):
    df = pd.read_csv(filename, compression='zip')
    return df

def get_regressor():
    """
        Returns gradient boosting regressor object
    """
    reg = GradientBoostingRegressor(
        learning_rate=GBR_PARAMS['learning_rate'],
        subsample=GBR_PARAMS['subsample'],
        n_estimators=GBR_PARAMS['n_estimators'],
        )
    return reg

def get_cv_scores(reg, train_data, train_target):
    """
        Returns cross validated R squared scores
    """
    scorer = make_scorer(r2_score, multioutput='variance_weighted')
    r2_scores = cross_val_score(reg, train_data, train_target, cv=4,
                            scoring=scorer)
    mean_squared_scores = cross_val_score(reg, train_data, train_target, cv=4,
                            scoring='mean_squared_error')
    mean_squared_scores = [sqrt(x*-1) for x in mean_squared_scores]
    return r2_scores, mean_squared_scores

def fit(reg, train_data, train_target):
    """
        Fits the training data
    """
    reg.fit(train_data,train_target)
    return reg

def split_train_test(df):
    """
        Returns train and test set splits for training and validation
    """
    data, target = df[FEATURES], df[TARGET_VAR]
    train_data, test_data, train_target, test_target = train_test_split(data, target,
                                                        test_size=0.25)
    return train_data, test_data, train_target, test_target

def predict_n_measure(reg, validation_data, validation_target):
    """
        Returns root mean squared error for validation data
    """
    pred_target = reg.predict(validation_data)
    rmse = sqrt(mean_squared_error(validation_target, pred_target))
    return rmse

def predict_kaggle_test_data(reg, test_data):
    """
        Returns predictions for kaggle's test dataset
    """
    ids = test_data['id'].tolist()
    test_data = test_data[FEATURES]
    pred_target = reg.predict(test_data)
    submission_df = pd.DataFrame(pred_target, ids).reset_index().rename(
                        columns={'index':'id', 0:'trip_duration'})
    submission_df['trip_duration'] = np.exp(submission_df['trip_duration']) - 1
    return submission_df

def get_feature_importance(reg):
    """
        Returns a dataframe with feature importance scores
    """
    importances = reg.feature_importances_
    importances = 100.0 * (importances / importances.max())
    imp_df = pd.DataFrame({
                    'features':FEATURES,
                    'relative_importances':importances,
                    })
    return imp_df

def plot_feature_imp(imp_df):
    """
        Builds and saves a plot with feature importance scores
    """
    imp_df = imp_df.sort_values('relative_importances', ascending=False)
    imp_df = imp_df.reset_index().drop('index', axis=1)
    sns_plot = sns.barplot(x=imp_df['relative_importances'], y=imp_df['features'])
    plt.tight_layout()
    plt.subplots_adjust(left=0.5)
    plt.savefig('../plots/feature_importances')

def train_xgboost(train_data, test_data, train_target, test_target):
    """
        Trains the data with xgboost model
    """
    d_train = xgb.DMatrix(train_data, label=train_target)
    d_test = xgb.DMatrix(test_data, label=test_target)
    watchlist = [(d_train, 'train'), (d_test, 'test')]

    model = xgb.train(XGB_PARAMS, d_train, 200, watchlist, early_stopping_rounds=50,
                      maximize=False, verbose_eval=10)

    print "xgb model results", model
    return model

def get_xgb_submissions(kaggle_test_data, model, filename):
    """
        Predicts trip duration for kaggle's test data and writes to a file
    """
    ids = kaggle_test_data['id'].tolist()
    kaggle_test_data = kaggle_test_data[FEATURES]
    kaggle_test_matrix = xgb.DMatrix(kaggle_test_data)
    pred_target = model.predict(kaggle_test_matrix)
    submission_df = pd.DataFrame(pred_target, ids).reset_index().rename(
                        columns={'index':'id', 0:'trip_duration'})
    submission_df['trip_duration'] = np.exp(submission_df['trip_duration']) - 1
    print "writing submissions file"
    submission_df.to_csv(filename, index=False)
    print "Done writing submissions file"

if __name__ == "__main__":
    # read data
    print 'reading data'
    filename = '../data/train.zip'
    df = read_data(filename)
    print 'done reading data'

    df = df.sample(SAMPLE_SIZE)

    # build features
    df = preprocess(df)

    # split train and test data
    train_data, validation_data, train_target, validation_target = split_train_test(df)

    # get regressor
    reg = get_regressor()

    # get cross validated r2_scores
    print 'running cross validation'
    start_time = time.time()
    r2_scores, mean_squared_scores = get_cv_scores(reg, train_data, train_target)
    print 'cross validated r2 scores', r2_scores, np.mean(r2_scores)
    print 'cross validated rmse', mean_squared_scores, np.mean(mean_squared_scores)
    print 'time taken', time.time() - start_time

    # fit data
    print 'Fitting data'
    start_time = time.time()
    fit(reg, train_data, train_target)
    print 'time taken', time.time() - start_time

    # plot feature important
    imp_df = get_feature_importance(reg)
    plot_feature_imp(imp_df)

    # predict and report metrics for validation  set
    print 'predicting targets for validation data'
    rmse = predict_n_measure(reg, validation_data, validation_target)
    print "validation set rmse", rmse

    # train xgboost
    # model = train_xgboost(train_data, test_data, train_target, test_target)
    #
    # # predict and report metrics for validation data
    # # get validation set
    # print "Reading test data"
    # kaggle_test_file = '../data/test.zip'
    # kaggle_test_data = read_data(kaggle_test_file)
    # kaggle_test_df = preprocess(kaggle_test_data)
    # submission_df = predict_kaggle_test_data(reg, kaggle_test_df)
    # submission_df.to_csv('../data/submission_data_gb.csv', index=False)
    # print "fitting test data"
    # xgb_submission_filename = '../data/xgb_submissions.csv'
    # get_xgb_submissions(kaggle_test_data, model, xgb_submission_filename)
