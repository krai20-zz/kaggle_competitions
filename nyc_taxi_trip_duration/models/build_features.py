from operator import methodcaller
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import time

def get_log_trip_duration(df):
    """
        Returns log trip duration.
        Kaggle uses RMSLE for evaluation,
        so we transform trip duration
        to log trip duration
    """
    df['trip_duration'] = np.log(df['trip_duration'].values + 1)
    return df

def remove_outliers(df):
    """
        Removes trip duration values above 3 std from mean
    """
    avg_d = df['trip_duration'].mean()
    dev_d = df['trip_duration'].std()
    df = df[df.trip_duration < (avg_d + 3*dev_d)]
    return df

def get_datetime_features(df):
    """
        Creates datetime, hour and day columns
    """
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['{}_day'.format('pickup_datetime'.split('_')[0])] = df['pickup_datetime'].apply(
                lambda x: x.weekday()
            )
    df['{}_hour'.format('pickup_datetime'.split('_')[0])] = df['pickup_datetime'].apply(
                lambda x: x.hour
            )
    return df

def get_weekday_features(df):
    """
        Calculates weekday features
    """
    df['is_weekday'] = df.pickup_day.apply(
            lambda x: True if x in range(0, 5) else False
        )
    return df

def get_tod_features(df):
    """
        Calculates time on day features
        Chosen times are based on trip duration by hour chart in notebook
    """
    df['is_morning'] = df.pickup_hour.apply(
            lambda x: True if x in range(7, 14) else False
        )
    df['is_afternoon'] = df.pickup_hour.apply(
            lambda x: True if x in range(14, 19) else False
        )
    df['is_evening'] = df.pickup_hour.apply(
            lambda x: True if x in range(19, 23) + [0] else False
        )
    df['is_early_morning'] = df.pickup_hour.apply(
            lambda x: True if x in range(1, 7) else False
        )
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
    """
        Returns bearing direction b/w origin and destination
    """
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

def get_direction(df):
    """
        Calculates bearing (direction) in degrees
    """
    df['direction'] = df.apply(bearing_array, axis=1)
    return df

def get_store_flag_feature(df):
    df['store_and_fwd_flag_is_N'] = df.store_and_fwd_flag.apply(
            lambda x: True if x == 'N' else False
        )
    return df

def preprocess(df):
    """
        Preprocesses data by removing outliers and extracting features
    """
    map(methodcaller('__call__', df), [
            get_datetime_features,
            get_tod_features,
            get_weekday_features,
            get_distance,
            get_direction,
            get_store_flag_feature,
            ]
        )

    try:
        # only needed for training and validation data
        df = remove_outliers(df)
        df = get_log_trip_duration(df)
    except KeyError:
        pass
    return df
