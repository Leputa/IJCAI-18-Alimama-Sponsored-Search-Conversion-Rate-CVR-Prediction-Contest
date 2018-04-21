import pandas as pd
import numpy as np
import gc

import os
import sys
sys.path.append("../")

from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing.data import OneHotEncoder
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import matplotlib.pyplot as plt

import Tool.utils as utils
import Tool.config as config

from scipy.sparse import hstack
from scipy.sparse import vstack

cat_features = ['user_gender_id',
                'user_occupation_id',
                'context_id',
                'context_page_id',
                'item_category_list',
                'hour'
                 "category_1",
                 "category_2",]

object_features = ["predict_category_1","predict_category_2","predict_category_0",
                    "predict_property_0","predict_property_1","predict_property_2",
                    "property_1","property_0","property_2",
                    "category_0",
                    'hour_and_category_1',
                    'category_cross_0', 'category_cross_1', 'category_cross_2',
                    ]




if __name__ == "__main__":

    path = config.cache_prefix_path + 'tree_leaves_train.npz'
    if os.path.exists(path):
