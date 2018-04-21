import pandas as pd
import numpy as np
import gc

import os
import sys
sys.path.append("../")

import Tool.utils as utils
import Tool.config as config
from sklearn.preprocessing.data import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from scipy.sparse import hstack
from scipy.sparse import vstack

def get_cat_one_hot_feature(tag = 'val'):

    if tag == 'train':

        print("获取类别特征one_hot线上提交数据")

        print("训练集长度: " + '478032')
        print("测试集长度：" + '42888')

        path = config.cache_prefix_path + 'cat_one_hot_train.npz'

        if os.path.exists(path):
            cat_one_hot_train = utils.load_sparse_csr(path)
            return cat_one_hot_train

        data = pd.read_pickle(config.data_prefix_path + 'data.pkl')[config.CAT_COLS]

        labelEncoding = LabelEncoder()
        for col in data.head(0):
            data[col] = labelEncoding.fit_transform(data[col].astype(str))

        onehotEncoding = OneHotEncoder()
        data = onehotEncoding.fit_transform(data)
        print(data.shape)

        utils.save_sparse_csr(path,data)

        return data

    elif tag == 'val':

        print("获取类别特征one_hot线下验证数据")

        print("训练集长度: " + '420627')
        print("验证集长度：" + '57405')

        path = config.cache_prefix_path + 'cat_one_hot_val.npz'

        if os.path.exists(path):
            cat_one_hot_val = utils.load_sparse_csr(path)
            return cat_one_hot_val

        data = pd.read_pickle(config.data_prefix_path + 'data.pkl')[config.CAT_COLS + ['day']]

        train = data[data.day < 24]
        test = data[data.day == 24]

        del data
        gc.collect()

        train.drop(['day'],axis=1,inplace=True)
        test.drop(['day'],axis=1,inplace=True)

        data = pd.concat([train,test],axis=0)
        del train,test
 
        gc.collect()

        labelEncoding = LabelEncoder()
        for col in data.head(0):
            data[col] = labelEncoding.fit_transform(data[col].astype(str))

        onehotEncoding = OneHotEncoder()
        data = onehotEncoding.fit_transform(data)
        print(data.shape)

        utils.save_sparse_csr(path,data)

        return data


def get_xgboost_one_hot_feature(tag = 'val'):

    params = {  'booster':'gbtree', 
            'num_leaves':35, 
            'max_depth':7, 
            'eta':0.05, 
            'max_bin':425, 
            'subsample_for_bin':50000, 
            'objective':'binary:logistic', 
            'min_split_gain':0,
            'min_child_weight':6, 
            'min_child_samples':10, 
            #'colsample_bytree':0.8,#在建立树时对特征采样的比例。缺省值为1
            #'subsample':0.9,#用于训练模型的子样本占整个样本集合的比例。 
            'subsample_freq':1,
            'colsample_bytree':1, 
            'reg_lambda':4,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
            'alpha':4,   #L1正则化 
            'seed':2018,
            'nthread':7, 
            'silent':True,
            'gamma':0.2,
            'eval_metric':'logloss'
         }


    object_features = ["predict_category_1","predict_category_2","predict_category_0",
                    "predict_property_0","predict_property_1","predict_property_2",
                    "property_1","property_0","property_2",
                    "category_1","category_0","category_2",
                    'category_cross_0', 'category_cross_1', 'category_cross_2',
                    'hour_and_category_1',
                    'user_gender_id','user_occupation_id',]
    
    if tag == 'train':

        print("获取xgboost特征one_hot线上提交数据")

        print("训练集长度: " + '478032')
        print("测试集长度：" + '42888')

        path = config.cache_prefix_path + 'xgboost_one_hot_train.npz'

        if os.path.exists(path):
            xgboost_one_hot_train = utils.load_sparse_csr(path)
            return xgboost_one_hot_train

        data = pd.read_pickle(config.data_prefix_path + 'data.pkl')

        features = [c for c in data.columns if c not in ['is_trade', 'instance_id','index',
                                            'context_id', 'time', 'day','context_timestamp',
                                            'property_list','category_list','property_predict_list','category_predict_list',
                                            'item_category_list', 'item_property_list', 'predict_category_property',
                                            'user_id','item_id','item_brand_id','item_city_id','shop_id',
                                            ]
                and c not in object_features]
        target = ['is_trade']

        train = data[data.is_trade.notnull()]
        test = data[data.is_trade.isnull()]
        del data
        gc.collect()

        xgb_train = xgb.DMatrix(train[features], label=train[target])
        xgb_test = xgb.DMatrix(test[features])
        del train,test
        gc.collect()

        model = xgb.train(params, xgb_train, 200, [(xgb_train, 'train')])
        
        train_leaves = model.predict(xgb_train, pred_leaf=True)
        test_leaves = model.predict(xgb_test, pred_leaf=True)
        del xgb_train,xgb_test
        gc.collect()

        onehotEncoding = OneHotEncoder()
        trans = onehotEncoding.fit_transform(np.concatenate((train_leaves, test_leaves), axis=0))

        utils.save_sparse_csr(path, trans)
        return trans

    elif tag == 'val':

        print("获取xgboost特征one_hot线下验证数据")

        print("训练集长度: " + '420627')
        print("测试集长度：" + '57405')

        path = config.cache_prefix_path + 'xgboost_one_hot_val.npz'

        if os.path.exists(path):
            xgboost_one_hot_val = utils.load_sparse_csr(path)
            return xgboost_one_hot_val

        data = pd.read_pickle(config.data_prefix_path + 'data.pkl')

        features = [c for c in data.columns if c not in ['is_trade', 'instance_id','index',
                                            'context_id', 'time', 'day','context_timestamp',
                                            'property_list','category_list','property_predict_list','category_predict_list',
                                            'item_category_list', 'item_property_list', 'predict_category_property',
                                            'user_id','item_id','item_brand_id','item_city_id','shop_id',
                                            ]
                    and c not in object_features]
        target = ['is_trade']

        data = data[data.is_trade.notnull()]
        train = data[data.day < 24]
        val = data[data.day == 24]

        xgb_train = xgb.DMatrix(train[features], label=train[target])
        xgb_val = xgb.DMatrix(val[features], label=val[target])
        
        del train,val,data
        gc.collect()

        model = xgb.train(params, xgb_train, 200, [(xgb_train, 'train'),(xgb_val,'valid')])
        
        train_leaves = model.predict(xgb_train, pred_leaf=True)
        val_leaves = model.predict(xgb_val, pred_leaf=True)
        
        del xgb_train,xgb_val
        gc.collect()

        onehotEncoding = OneHotEncoder()
        trans = onehotEncoding.fit_transform(np.concatenate((train_leaves, val_leaves), axis=0))

        utils.save_sparse_csr(path, trans)
        return trans


def get_one_hot_data(tag = 'val'):
    cat_one_hot = get_cat_one_hot_feature(tag)
    xgboost_one_hot = get_xgboost_one_hot_feature(tag)

    if tag == 'val':
        return hstack([xgboost_one_hot[:420627,:],cat_one_hot[:420627,:]]),hstack([xgboost_one_hot[420627:,:],cat_one_hot[420627:,:]])
    elif tag == 'train':
        return hstack([xgboost_one_hot[:478032,:],cat_one_hot[:478032,:]]),hstack([xgboost_one_hot[478032:,:],cat_one_hot[478032:,:]])
        data[:478032,:],data[478032:,:]


if __name__ == '__main__':
    train,val = get_one_hot_data('val')
    print(train.shape)
    print(val.shape)





