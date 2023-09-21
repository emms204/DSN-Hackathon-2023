import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from typing import Union, List, Literal, Tuple

oe = OrdinalEncoder()


def Load_Data(file_path: str) -> Tuple[pd.DataFrame]:
  train = pd.read_csv(file_path+'/Housing_dataset_train.csv')
  test = pd.read_csv(file_path+'/Housing_dataset_test.csv')
  sub = pd.read_csv(file_path+'/Sample_submission.csv')

  return train, test, sub

def baseline_preprocessing(train:pd.DataFrame, 
                           test:pd.DataFrame, 
                           cat_features=['loc','title']) -> Tuple[pd.DataFrame]:

    TT = train.drop(['ID'],axis=1)
    tests = test.drop(['ID'], axis=1)

    for col in cat_features:
        tests[col] = oe.fit_transform(tests[[col]])
        TT[col] = oe.fit_transform(TT[[col]])

    tests.fillna(0, inplace=True)
    TT.fillna(0, inplace=True)

    return TT, tests

def preprocessing(Train:pd.DataFrame, Test:pd.DataFrame) -> Tuple[pd.DataFrame]:
    Train['target'] = 'Train'
    Test['target'] = 'Test'
    data = pd.concat([Train, Test],axis=0).reset_index(drop=True)

    ohe = OneHotEncoder(handle_unknown="ignore",sparse_output=False)
    ohe_mod = ohe.fit(data[['loc','title','bathroom','bedroom','parking_space']])
    ohe_cols = ohe_mod.get_feature_names_out()
    ohe_data = ohe.transform(data[['loc','title','bathroom','bedroom','parking_space']])

    ohe_data = pd.DataFrame(ohe_data, columns=ohe_cols)
    ohe_concat = pd.concat([data, ohe_data],axis=1)

    title_columns = ohe_concat.filter(like="title_").columns
    bathroom_columns = ohe_concat.filter(like="bathroom_").columns
    bedroom_columns = ohe_concat.filter(like="bedroom_").columns
    parking_space_columns = ohe_concat.filter(like="parking_space_").columns
    loc_columns = ohe_concat.filter(like="loc_").columns

    loc_title_sum = ohe_concat.groupby(['loc'])[title_columns].transform('sum')
    loc_bedroom_sum = ohe_concat.groupby(['loc'])[bedroom_columns].transform('sum')
    loc_bathroom_sum = ohe_concat.groupby(['loc'])[bathroom_columns].transform('sum')
    loc_parking_space_sum = ohe_concat.groupby(['loc'])[parking_space_columns].transform('sum')
    loc_bedroom = ohe_concat.groupby(['loc'])[['bedroom']].sum().sort_values(by=['bedroom'],ascending=False)

    loc_bedroom_map = {key: value for key, value in zip(loc_bedroom.index, range(len(loc_bedroom.index)))}

    loc_bathroom = ohe_concat.groupby(['loc'])[['bathroom']].sum().sort_values(by=['bathroom'],ascending=False)
    loc_bathroom_map = {key: value for key, value in zip(loc_bathroom.index, range(len(loc_bathroom.index)))}

    loc_parking_space = ohe_concat.groupby(['loc'])[['parking_space']].sum().sort_values(by=['parking_space'],ascending=False)
    loc_parking_space_map = {key: value for key, value in zip(loc_parking_space.index, range(len(loc_parking_space.index)))}

    ohe_concat['map_loc_bedroom'] = ohe_concat['loc'].map(loc_bedroom_map)
    ohe_concat['map_loc_bathroom'] = ohe_concat['loc'].map(loc_bathroom_map)
    ohe_concat['map_loc_parking_space'] = ohe_concat['loc'].map(loc_parking_space_map)

    ohe_concat['map_loc_sum_weighted'] = ohe_concat[['map_loc_bedroom', 'map_loc_bathroom', 'map_loc_parking_space']].sum(axis=1)

    loc_values = {key: value for key, value in zip(ohe_concat['loc'].values, ohe_concat['map_loc_sum_weighted'].values)}
    for col in loc_columns:
        location = col.split("_")[1]
        if location in loc_values.keys():
            ohe_concat[col] = np.where(ohe_concat[col]==1,loc_values[location],0)

    ohe_concat['N_loc_weighted'] = np.log1p(ohe_concat['map_loc_bedroom'] * ohe_concat['map_loc_bathroom'] * ohe_concat['map_loc_parking_space'])

    title_loc_sum = ohe_concat.groupby(['title'])[loc_columns].transform('sum')
    title_bedroom_sum = ohe_concat.groupby(['title'])[bedroom_columns].transform('sum')
    title_bathroom_sum = ohe_concat.groupby(['title'])[bathroom_columns].transform('sum')
    title_parking_space_sum = ohe_concat.groupby(['title'])[parking_space_columns].transform('sum')

    title_bedroom = ohe_concat.groupby(['title'])[['bedroom']].sum().sort_values(by=['bedroom'],ascending=False)
    title_bedroom_map = {key: value for key, value in zip(title_bedroom.index, range(len(title_bedroom.index)))}

    title_bathroom = ohe_concat.groupby(['title'])[['bathroom']].sum().sort_values(by=['bathroom'],ascending=False)
    title_bathroom_map = {key: value for key, value in zip(title_bathroom.index, range(len(title_bathroom.index)))}

    title_parking_space = ohe_concat.groupby(['title'])[['parking_space']].sum().sort_values(by=['parking_space'],ascending=False)
    title_parking_space_map = {key: value for key, value in zip(title_parking_space.index, range(len(title_parking_space.index)))}
        
    ohe_concat['map_title_bedroom'] = ohe_concat['title'].map(title_bedroom_map)
    ohe_concat['map_title_bathroom'] = ohe_concat['title'].map(title_bathroom_map)
    ohe_concat['map_title_parking_space'] = ohe_concat['title'].map(title_parking_space_map)

    ohe_concat['map_title_sum_weighted'] = ohe_concat[['map_title_bedroom', 'map_title_bathroom', 'map_title_parking_space']].sum(axis=1)

    title_values = {key: value for key, value in zip(ohe_concat['title'].values, ohe_concat['map_title_sum_weighted'].values)}
    
    for col in title_columns:
        titles = col.split("_")[1]
        if titles in loc_values.keys():
            ohe_concat[col] = np.where(ohe_concat[col]==1,loc_values[titles],0)

    ohe_concat['N_title_weighted'] = np.log1p(ohe_concat['map_title_bedroom'] * ohe_concat['map_title_bathroom'] * ohe_concat['map_title_parking_space'])

    ohe_concat['N_Unweighted'] = ohe_concat['map_loc_sum_weighted']//3 * ohe_concat['map_title_sum_weighted'] //3 * ohe_concat['bedroom'] * ohe_concat['bathroom'] * ohe_concat['parking_space']

    ohe_concat = ohe_concat.join(loc_title_sum, rsuffix="_loc_sum")
    ohe_concat = ohe_concat.join(loc_bedroom_sum, rsuffix="_loc_sum")
    ohe_concat = ohe_concat.join(loc_bathroom_sum, rsuffix="_loc_sum")
    ohe_concat = ohe_concat.join(loc_parking_space_sum, rsuffix="_loc_sum")
    ohe_concat = ohe_concat.join(title_bedroom_sum, rsuffix="_title_sum")
    ohe_concat = ohe_concat.join(title_bathroom_sum, rsuffix="_title_sum")
    ohe_concat = ohe_concat.join(title_parking_space_sum, rsuffix="_title_sum")

    ohe_concat['loc/title'] = data['loc'] + "_" + ohe_concat['title']
    ohe_concat['loc/title_bedroom_mean'] = ohe_concat.groupby(['loc/title'])[['bedroom']].transform('mean')
    ohe_concat['loc/title_bathroom_mean'] = ohe_concat.groupby(['loc/title'])[['bathroom']].transform('mean')
    ohe_concat['loc/title_parking_space_mean'] = ohe_concat.groupby(['loc/title'])[['parking_space']].transform('mean')

    ohe_concat['bedroom_check'] = np.where(ohe_concat['bedroom']<3,1,2)
    ohe_concat['bathroom_check'] = np.where(ohe_concat['bathroom']<3,1,2)
    ohe_concat['parking_space_check'] = np.where(ohe_concat['parking_space']<3,1,2)

    ohe_concat['sum_weighted'] = ohe_concat[['map_loc_sum_weighted','map_title_sum_weighted','bedroom_check','bathroom_check','parking_space_check']].sum(axis=1)
    ohe_concat['mean_weighted'] = ohe_concat[['map_loc_sum_weighted','map_title_sum_weighted','bedroom_check','bathroom_check','parking_space_check']].mean(axis=1)

    ohe_concat['N_log_weighted'] = ohe_concat['N_loc_weighted'] * ohe_concat['N_title_weighted'] * ohe_concat['bedroom_check'] * ohe_concat['bathroom_check'] * ohe_concat['parking_space_check']

    ohe_concat['N_weighted'] = np.log(ohe_concat['sum_weighted'] * ohe_concat['mean_weighted'])

    train = ohe_concat[ohe_concat['target'] == "Train"]
    test = ohe_concat[ohe_concat['target'] == "Test"]

    bedroom_target = train.groupby(['bedroom'])[['price']].mean().to_dict()['price']
    title_target = train.groupby(['title'])[['price']].mean().to_dict()['price']
    loc_target = train.groupby(['loc'])[['price']].mean().to_dict()['price']

    train['bedroom'] = train['bedroom'].map(bedroom_target)
    test['bedroom'] = test['bedroom'].map(bedroom_target)

    train['loc'] = train['loc'].map(loc_target)
    test['loc'] = test['loc'].map(loc_target)

    train['title'] = train['title'].map(title_target)
    test['title'] = test['title'].map(title_target)

    train['bedroom'] = np.log(train['bedroom'])
    test['bedroom'] = np.log(test['bedroom'])

    train['loc'] = np.log(train['loc'])
    test['loc'] = np.log(test['loc'])

    train['title'] = np.log(train['title'])
    test['title'] = np.log(test['title'])

    train = train.drop(['target','ID'],axis=1)
    test = test.drop(['target','ID'],axis=1)

    TT = train[[col for col in train.columns if col != 'price'] + ['price']]

    cat_features = ['loc/title']
    for col in cat_features:
        test[col] = oe.fit_transform(test[[col]])
        TT[col] = oe.fit_transform(TT[[col]])

    TT =  TT.drop(['bedroom_1.0', 'bedroom_2.0', 'bedroom_3.0', 'bedroom_4.0',
        'bedroom_5.0', 'bedroom_6.0', 'bedroom_7.0', 'bedroom_8.0',
        'bedroom_9.0', 'bedroom_nan','parking_space_1.0', 'parking_space_2.0', 'parking_space_3.0',
        'parking_space_4.0', 'parking_space_5.0', 'parking_space_6.0',
        'parking_space_nan','title_Apartment','title_Detached duplex','title_Flat', 'title_Mansion',
        'title_Penthouse', 'title_Semi-detached duplex', 'title_Terrace duplex',
        'title_Townhouse', 'title_nan','loc/title','loc/title_bedroom_mean','loc/title_parking_space_mean','parking_space_1.0_loc_sum',
        'parking_space_2.0_loc_sum','parking_space_3.0_loc_sum', 'parking_space_4.0_loc_sum',
        'parking_space_5.0_loc_sum', 'parking_space_6.0_loc_sum',
        'parking_space_nan_loc_sum','map_title_bedroom', 'map_title_bathroom',
        'map_title_parking_space','parking_space','bathroom'],axis=1)
    
    return TT, test



    

    