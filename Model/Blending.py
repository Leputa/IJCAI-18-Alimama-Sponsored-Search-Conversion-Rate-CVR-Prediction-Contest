import numpy as np
import pandas as pd
import gc

import os
import sys
sys.path.append("../")

from scipy.stats import rankdata
import Tool.config as config

def get_subs(label, fileNames):
    predict_list = []
    for file in fileNames:
        predict_list.append(pd.read_csv(config.data_prefix_path + file, sep=' ')[label].values)
    return predict_list



if __name__ == '__main__':
    label = ['predicted_score']
    # fileNames = ['sub0420_xgboost.txt','sub0420_lightgbm2.txt','sub0419_lightgbm.txt']
    fileNames = ['sub0420_blending.txt','sub0420_stacking.txt']

    predict_list = get_subs(label,fileNames)
    gc.collect()

    print("Rank averaging on ", len(predict_list), " files")
    predictions = np.zeros_like(predict_list[0])
    # for predict in predict_list:
    #     predictions = np.add(predictions, rankdata(predict)/predictions.shape[0])
    #     gc.collect()
    # predictions /= len(predict_list)
    # 内存原因，上述代码跑不出来
    # predictions = predict_list[0]*0.45 + predict_list[1]*0.35 + predict_list[2]*0.2
    predictions = predict_list[0]*0.55 + predict_list[1]*0.45
    
    sub = pd.read_csv(config.data_prefix_path + 'test_b.csv', sep=" ")
    sub['predicted_score'] = predictions
    sub[['instance_id', 'predicted_score']].to_csv(config.data_prefix_path + 'sub0422.txt', sep=" ",
                                                    index=False)