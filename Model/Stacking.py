import numpy as np
import pandas as pd
import gc

import os
import sys
sys.path.append("../")

import Tool.config as config

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import lightgbm as lgb


trainFileNames = ['xgboost_oof.txt', 'lightgbm_oof.txt', 'lightgbm2_oof.txt',
                  #'xgboost_LR_oof.txt', 'xgboost_FM_FTRL_oof.txt'
                  ]

testFileNames = ['sub0420_xgboost.txt', 'sub0419_lightgbm.txt', 'sub0420_lightgbm2.txt',
                 #'sub0421_xgboost_LR.txt','sub0421_xgboost_FM_FTRL.txt'
                 ]

features = ['instance_id', 'is_trade', 'day' ,
            'diffTime_first', 'diffTime_last', 'user_hour_cntx', 'category_IOU',
            'shop_score_delivery','shop_score_description',
            'user_mean_hour'
            ]

def loadData(trainFileNames, testFileNames, features):

    print("loading data...")
    i = 0

    for file in trainFileNames:
        i += 1
        if file == trainFileNames[0]:
            train = pd.read_csv(config.data_prefix_path + file, sep = ' ')[['instance_id', 'is_trade_oof']].rename(
                columns = {'is_trade_oof':'is_trade_oof'+str(i)})
        else:
            train = pd.merge(
                train, pd.read_csv(config.data_prefix_path + file, sep = ' ')[['instance_id', 'is_trade_oof']].rename(
                columns = {'is_trade_oof':'is_trade_oof'+str(i)}), 
                on = ['instance_id'], how = 'left')

    for file in testFileNames:
        if file == testFileNames[0]:
            test = pd.read_csv(config.data_prefix_path + file, sep = ' ')[['instance_id', 'predicted_score']].rename(
                columns = {'predicted_score':'predicted_score'+str(i)})
        else:
            test = pd.merge(
                test, pd.read_csv(config.data_prefix_path + file, sep = ' ')[['instance_id', 'predicted_score']].rename(
                columns = {'predicted_score':'predicted_score'+str(i)}), 
                on = ['instance_id'], how = 'left')
    
    data = pd.read_pickle(config.data_prefix_path + 'data.pkl')[features]

    traindata = data[data.is_trade.notnull()]
    testdata = data[data.is_trade.isnull()]

    del data
    gc.collect()

    train = pd.merge(train, traindata, on = ['instance_id'], how = 'left')
    test = pd.merge(test, testdata, on = ['instance_id'], how = 'left')

    return train,test

def OOFStacking(tag = 'val'):
    train,test = loadData(trainFileNames,testFileNames,features)

    params = {
            'task': 'train',
            'objective': 'binary',
            'boosting_type': 'gbdt',

            'learning_rate': 0.03,# 学习率
            'num_leaves': 10,
            'max_depth': 3,
            'feature_fraction': 0.7,
            'colsample_bytree': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,

            'eval_metric': 'logloss',
            'num_threads':7,
            'seed':2018
        }

    if tag == 'val':
        print("线下验证...")

        X_train = train[train.day < 24]
        X_test = train[train.day == 24]


        for feature in ['instance_id', 'day']:
            X_train.drop([feature], axis=1, inplace=True)
            X_test.drop([feature], axis=1, inplace=True)
        
        del train,test
        gc.collect()

        labels = [ head for head in X_train.head(0)]
        labels.remove('is_trade')

        target = ['is_trade']

        train_X = X_train[labels]
        train_Y = X_train[target].values.ravel()
        val_X = X_test[labels]
        val_Y = X_test[target].values.ravel()

        print(train_X.head(0))
        print("")
        print(labels)

        del X_train,X_test

        lgb_train = lgb.Dataset(data=train_X, label=train_Y, feature_name=labels)
        lgb_eval = lgb.Dataset(data=val_X, label=val_Y, feature_name=labels, reference=lgb_train)

    
        gbm = lgb.train(params,lgb_train,\
                valid_sets=[lgb_train,lgb_eval],
                num_boost_round = 2000,
                early_stopping_rounds=100,
                verbose_eval=20)

    if tag == 'train':
        print("线上提交...")

        train_labels = [ head for head in train.head(0) if head not in ['is_trade','instance_id','day']]
        test_labels = [ head for head in test.head(0) if head not in ['is_trade','instance_id','day']]

        target = ['is_trade']

        train_X = train[train_labels].values
        train_Y = train[target].values.ravel()
        test_X = test[test_labels].values

        del train
        gc.collect()

        lgb_train = lgb.Dataset(data=train_X, label=train_Y, feature_name=train_labels)
       
        gbm = lgb.train(params,lgb_train,\
                valid_sets=[lgb_train],
                num_boost_round = 200,
                verbose_eval = 20)

        test['predicted_score'] = gbm.predict(test_X)
        test[['instance_id', 'predicted_score']].to_csv(config.data_prefix_path + 'sub0422_stacking_5.txt',sep=" ",index=False)

if __name__ == "__main__":
    OOFStacking('val')