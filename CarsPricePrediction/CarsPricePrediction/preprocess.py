import datetime
import joblib
import os

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode


def fix_car(df):
    for index in range(len(df['type'])):
        if df['type'][index] == 'small car' and df['engine_capacity'][index] > 5:
            df['engine_capacity'][index] = np.nan
        elif df['engine_capacity'][index] > 9:
            df['engine_capacity'][index] = np.nan
        elif df['engine_capacity'][index] < 0.5:
            df['engine_capacity'][index] = np.nan
    
    return df


def group_rare_fuel(df):
    for i in range(len(df['fuel'])):
        if df['fuel'][i] != 'diesel':
            df['fuel'][i] = 'other'
    
    return df


def group_rare_gearbox(df):
    for i in range(len(df['gearbox'])):
        if df['gearbox'][i] != 'manual':
            df['gearbox'][i] = 'other'
    
    return df


def recency(year):
    today_year = datetime.datetime.now().year
    if year <= 20:
        return today_year - (year + 2000)
    elif year > 20 and year < 100:
        return today_year - (year + 1900)
    else:
        return today_year - year


def get_year(year):
    if year <= 20:
        return (year + 2000) % 10
    elif year > 20 and year < 100:
        return (year + 1900) % 10
    else:
        return year % 10


def get_decade(year):
    if year <= 20:
        return (year + 2000) % 100
    elif year > 20 and year < 100:
        return (year + 1900) % 100
    else:
        return year % 100


def mileage_group(mileage):
    if mileage <= 30000:
        return 0 #low
    elif mileage > 30000 and mileage <= 60000:
        return 1 #less than average
    elif mileage > 60000 and mileage <= 80000:
        return 2 #average
    elif mileage > 80000 and mileage <= 120000:
        return 3 #more than average
    else:
        return 4 #high


def zipcode_group(zipcode, zip_geo_groups, coords):
    if zip_geo_groups.get(zipcode):
        return zip_geo_groups.get(zipcode)
    else:
        for i, z in enumerate(coords['zipcode']):
            if abs(z - zipcode) < 50:
                return coords['group'][i]


def fill_nans(my_df):
    my_df['type'] = my_df['type'].fillna('other')
    my_df['damage'] = my_df['damage'].fillna(-1)
    my_df['model'] = my_df['model'].fillna(my_df.groupby(['brand'])['model']\
                          .transform(lambda x: 'other' if mode(x)[0][0] == 0 else mode(x)[0][0]))
    return my_df


def preprocess(tdf, train=False):
    tdf = fix_car(tdf)
    tdf = group_rare_fuel(tdf)
    tdf = group_rare_gearbox(tdf)
    
    tdf['recency'] = np.array([recency(year) for year in tdf['registration_year']])
    tdf['year'] = np.array([get_year(year) for year in tdf['registration_year']])
    tdf['decade'] = np.array([get_decade(year) for year in tdf['registration_year']])
    tdf = tdf.drop(columns=['registration_year'])
    
    tdf['mileage'] = np.array([mileage_group(ml) for ml in tdf['mileage']])
    
    coords = pd.read_csv(
        os.path.dirname(__file__) + '/utils/zipcodes.csv'
    )
    zip_geo_groups = coords[['zipcode', 'group']].set_index('zipcode').to_dict()['group']

    tdf['geo_group'] = [zipcode_group(z, zip_geo_groups, coords) for z in tdf['zipcode']]
    tdf = tdf.drop(columns=['zipcode'])
    
    tdf = fill_nans(tdf)
    
    corrs = ['insurance_price', 'engine_capacity']
    for cr in corrs:
        tdf[cr].fillna(-1, inplace = True)

    cat_features = ["type", "model", "brand", "gearbox", "fuel"]
    for cat_feature in cat_features:
        if train:
            le = LabelEncoder()
            le.fit(tdf[cat_feature])
        else:
            le = joblib.load(os.path.dirname(__file__) + f'/utils/label_encoder_{cat_feature}.pkl')
        tdf[cat_feature] = le.transform(tdf[cat_feature])

        if train:
            joblib.dump(le, f'./CarsPricePrediction/utils/label_encoder_{cat_feature}.pkl')

    if train:
        return tdf.drop(columns=['price']).to_numpy(), tdf['price'].apply(np.log1p)
    else:
        return tdf.to_numpy()
