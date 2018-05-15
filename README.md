# IJCAI-18-Alimama-Sponsored-Search-Conversion-Rate-CVR-Prediction-Contest

#### 赛题详情
    https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100150.711.5.18af2009XtMy26&raceId=231647 

#### 题目描述： 
    本次比赛以阿里电商广告为研究对象，提供了淘宝平台的海量真实交易数据，参赛选手通过人工智能技术构建预测模型预估用户的购买意向。

#### 方案说明
 * ./Feature:
    1. feature.py
    人工特征提取：具体说明见"特征检查.ipynb"
    2. one_hot_feature.py
    利用xgboost提取组合特征，并加入类别one-hot组合特征
* ./Model:
    1. Xgboost.ipynb
    xgboost模型训练，输出结果，并保存OOF结果
    2. Lightgbm.ipynb
    LightGBM模型训练数据，输出结果，并保存OOF结果
    3. xgboost + LR.ipynb
    利用Xgboost提取的组合特征及one-hot类别特征，再加上原始特征,用Logistic Regression模型训练，输出结果，并保存OOF结果
    4. xgboost + FM_FTRL
    利用Xgboost提取的组合特征及one-hot类别特征，用FM_FTRL模型训练，输出结果，并保存OOF结果
    FM_FTRM:https://github.com/anttttti/Wordbatch/blob/master/wordbatch/models/fm_ftrl.pyx
    5. Stacking.py
    用stacking方法做模型融和
    6. Blending.py
    用blending方法做模型融和
    7. smooth.py
    购买率平滑处理代码
    https://blog.csdn.net/wwqwkg6e/article/details/55000216
* ./Tool:
     配置信息
* ./Paper:
    参考论文
#### 比赛成绩 
* 初赛成绩： 196/5204, Top 4%
* 复赛成绩： 未参加
#### 参考代码
* https://github.com/SnowColdplay/almm_baseline
