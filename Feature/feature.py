import os

#便于导入上级目录
import sys
sys.path.append("../")

import numpy as np
import pandas as pd
import time
import gc

from Model.smooth import BayesianSmoothing
from Model.smooth import HyperParam
import Tool.config as config
from tqdm import tqdm

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN



class Feature():

    def __init__(self):
        self.all_data = None
        self.train_length = 478085

    """
        降采样，负样本太多
        如果不采用，会发现准确率超高98%，auc只有0.7
        SMOTE 算法
        原始统计：[(0, 469117), (1, 9021)] 52:1
        采样完：[(0, 469117), (1, 93823)]  5: 1,ratio=0.2
        采样完：[(0, 469117), (1, 46911)]
        库：http://contrib.scikit-learn.org/imbalanced-learn/stable/over_sampling.html
        """
    def sub_sample(self,X,y):
        print(sorted(Counter(y).items()))
        # X_resampled, y_resampled = X,y
        # X_resampled, y_resampled = RandomUnderSampler(ratio=0.2).fit_sample(X,y)
        X_resampled, y_resampled = SMOTE().fit_sample(X, y)
        if Config.DEBUG:
            print('数据smote采样完成')
            # print(sorted(Counter(y).items()))
        print(sorted(Counter(y_resampled).items()))

        return X_resampled,y_resampled

    def add_user_click_cat(self, data):
        print('处理用户点击类别标记')

        path = config.cache_prefix_path + 'click_cat_feature.pkl'
        if os.path.exists(path):
            time_data = pd.read_pickle(path)
            return pd.merge(data, time_data, on=['instance_id'], how='left')
        subset = ['category_1', 'user_id']

        data['user_click_cat'] = 0
        # subset：用于识别重复的列标签或列标签序列，默认所有列标签;只要有1个不重复就视为不重复
        # keep = 'frist'：除了第一次出现外，其余相同的被标记为重复
        # keep = 'last'：除了最后一次出现外，其余相同的被标记为重复
        # keep = False：所有相同的都被标记为重复

        # 'maybe' == 1: user_id点击某类别广告不止一次
        pos = data.duplicated(subset=subset, keep=False)
        data.loc[pos, 'user_click_cat'] = 1
        # 'maybe' == 2: user_id重复点击某广告，且此次点击是第一次点击
        pos = (~data.duplicated(subset=subset, keep='first')) & data.duplicated(subset=subset, keep=False)
        data.loc[pos, 'user_click_cat'] = 2
        # 'maybe' == 3: user_id重复点击某广告，且此次点击是最后一次点击
        pos = (~data.duplicated(subset=subset, keep='last')) & data.duplicated(subset=subset, keep=False)
        data.loc[pos, 'user_click_cat'] = 3

        # 统计user_id和cat1的组合出现次数
        # 若该次数大于2(说明用户对该类别有偏好)，则添加一个标记1,否则标记为0
        temp = data.groupby(subset)['is_trade'].count().reset_index()
        temp.columns = ['category_1', 'user_id', 'large2']
        temp['large2'] = 1 * (temp['large2'] > 2)
        data = pd.merge(data, temp, how='left', on=subset)

        data[['instance_id', 'user_click_cat', 'large2']].to_pickle(path)

        return data

    def PurchaseRate(self,data):

        print('一般特征购买率')

        path = config.cache_prefix_path + 'purchaseRate_feature.pkl'
        if os.path.exists(path):
            time_data = pd.read_pickle(path)
            return pd.merge(data, time_data, on=['instance_id'], how='left')

        newcolumns = []

        cols = ['item_category_list','item_price_level', \
                'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_gender_id', \
                'user_age_level', 'user_occupation_id', 'user_star_level', 'context_page_id', \
                'shop_review_num_level', 'shop_star_level']

        for col in cols:
            item_cnt = data.groupby([col]).apply(lambda x: x['instance_id'][(x['day'] < 25).values].count()).reset_index(name = col+'_cnt')
            item_cv = data.groupby([col]).apply(lambda x: x['is_trade'][(x['day'] < 25).values].sum()).reset_index(name = col+'_purRate')
            item_cv[col+'_purRate'] = item_cv[col+'_purRate']/item_cnt[col+'_cnt']

            data = pd.merge(data, item_cv, how= 'left', on = col)
            newcolumns.append(col+'_purRate')

        data[newcolumns + ['instance_id']].to_pickle(path)
        return data



    def add_diffTime_feature(self, data):
        #这个特征非常重要
        print('处理时间差数据')

        path = config.cache_prefix_path + 'diffTime_feature.pkl'
        if os.path.exists(path):
            time_data = pd.read_pickle(path)
            return pd.merge(data, time_data, on=['instance_id'], how='left')

        #to_datatime()————将Str和Unicode转化为时间格式
        data['time'] = pd.to_datetime(data['time'])
        subset = ['user_id', 'category_1']

        # 时间差Trick，对clickTime处理成分钟
        temp = data.loc[:, ['time', 'user_id', 'category_1']].drop_duplicates(subset=subset, keep='first')
        temp.rename(columns={'time': 'diffTime_first'}, inplace=True)
        data = pd.merge(data, temp, how='left', on=subset)

        # 当前时间与某用户第一点击某类别广告的时间的时间差
        data['diffTime_first'] = data['time'] - data['diffTime_first']
        data['diffTime_first'] = data['diffTime_first'].apply(lambda x: x.seconds / 60)

        temp = data.loc[:, ['time', 'user_id', 'category_1']].drop_duplicates(subset=subset, keep='last')
        temp.rename(columns={'time': 'diffTime_last'}, inplace=True)
        data = pd.merge(data, temp, how='left', on=subset)

        # 某用户最后一次点击某类别广告的时间与当前时间的时间差
        data['diffTime_last'] = data['diffTime_last'] - data['time']
        data['diffTime_last'] = data['diffTime_last'].apply(lambda x: x.seconds / 60)
        # data.loc[~data.duplicated(subset=subset, keep=False), ['diffTime_first', 'diffTime_last']] = -1
        data[['instance_id', 'diffTime_last', 'diffTime_first']].to_pickle(path)

        return data


    def add_slide_is_trade_count_smooth(self,data):
        # 1、用一天做滑动窗口
        # 2、用之前所有日期作滑动窗口
        print('滑窗统计')
        path = config.cache_prefix_path + 'slide_is_trade_count.pkl'
        if os.path.exists(path):
            slide_count_data = pd.read_pickle(path)
            return pd.merge(data, slide_count_data, on=['instance_id'], how='left')

        new_column_list = []

        for feat in ['item_brand_id', 'item_id', 'shop_id']:

            print(feat + "滑窗平滑购买率")
            res_1 = pd.DataFrame()
            res_all = pd.DataFrame()

            for day in range(19, 26):
                # pre1day
                item_pre1day = data.groupby([feat]).apply(
                    lambda x: x['user_id'][(x['day'] == day - 1).values].count()).reset_index(name = feat + '_show_cnt_pre1day')
                item_trade_cnt_pre1day = data.groupby([feat]).apply(
                    lambda x: x['is_trade'][(x['day'] == day - 1).values].sum()).reset_index(name=feat + '_trade_rate_smooth_pre1day')

                item_pre1day[feat + '_trade_rate_smooth_pre1day'] = item_trade_cnt_pre1day[feat + '_trade_rate_smooth_pre1day']
                item_pre1day['day'] = day
                item_pre1day.fillna(0, inplace=True)
                res_1 = res_1.append(item_pre1day, ignore_index=True)

                # preallday
                item_preallday = data.groupby([feat]).apply(
                    lambda x: x['user_id'][(x['day'] < day).values].count()).reset_index(name = feat+ '_show_cnt_preallday')
                item_trade_cnt_preallday = data.groupby([feat]).apply(
                    lambda x: x['is_trade'][(x['day'] < day).values].sum()).reset_index(name=feat + '_trade_rate_smooth_preallday')

                item_preallday[feat + '_trade_rate_smooth_preallday'] = item_trade_cnt_preallday[feat + '_trade_rate_smooth_preallday']
                item_preallday['day'] = day
                item_preallday.fillna(0, inplace=True)
                res_all = res_all.append(item_preallday, ignore_index=True)

            # pre1day
            bs = BayesianSmoothing(1, 1)
            bs.update(res_1[feat + '_show_cnt_pre1day'].values, res_1[feat + '_trade_rate_smooth_pre1day'].values, 1000, 0.001)
            res_1[feat + '_trade_rate_smooth_pre1day'] = (res_1[feat + '_trade_rate_smooth_pre1day'] + bs.alpha) / (res_1[feat + '_show_cnt_pre1day'] + bs.alpha + bs.beta)
            data = pd.merge(data, res_1, on=[feat, 'day'], how='left')
            data[feat + '_show_cnt_pre1day'].fillna(0, inplace=True)
            data[feat + '_trade_rate_smooth_pre1day'].fillna(value=bs.alpha / (bs.alpha + bs.beta), inplace=True)
            new_column_list.extend([feat + '_show_cnt_pre1day',feat +'_trade_rate_smooth_pre1day'])

            # preallday
            bs = BayesianSmoothing(1, 1)
            bs.update(res_all[feat + '_show_cnt_preallday'].values, res_all[feat + '_trade_rate_smooth_preallday'].values, 1000, 0.001)
            res_all[feat + '_trade_rate_smooth_preallday'] = (res_all[feat + '_trade_rate_smooth_preallday'] + bs.alpha) / (res_all[feat + '_show_cnt_preallday'] + bs.alpha + bs.beta)
            data = pd.merge(data, res_all, on=[feat, 'day'], how='left')
            data[feat + '_show_cnt_preallday'].fillna(0, inplace=True)
            data[feat + '_trade_rate_smooth_preallday'].fillna(value=bs.alpha / (bs.alpha + bs.beta), inplace=True)
            new_column_list.extend([feat + '_show_cnt_preallday',feat +'_trade_rate_smooth_preallday'])

        slide_count_data = data[['instance_id']+new_column_list]
        slide_count_data.to_pickle(path)
        return data

    def add_item_feature(self, df_merge):
        print('item相关统计')

        item_feature_path = config.cache_prefix_path + 'item_feature.pkl'
        if os.path.exists(item_feature_path):
            item_feature_data = pd.read_pickle(item_feature_path)
            return pd.merge(df_merge, item_feature_data, on=['instance_id'], how='left')

        new_column_list = []

        cols = ['item_id', 'item_category_list', 'item_brand_id', 'item_city_id', \
                'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']

        for main_col in ['item_id', 'item_category_list', 'item_brand_id', 'item_city_id', \
                         'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']:
            print(main_col + ' 出现了几次')

            itemcnt = df_merge.groupby([main_col], as_index=False)['instance_id'].agg({main_col + '_cnt': 'count'})
            df_merge = pd.merge(df_merge, itemcnt, on=[main_col], how='left')
            new_column_list.append(main_col + '_cnt')

            cols.remove(main_col)

            if main_col == 'item_id':
                continue

            print(main_col + ' 和 ' + str(cols) + ' 同时出现了几次\n')

            for col in cols:
                itemcnt = df_merge.groupby([col, main_col], as_index=False)['instance_id'].agg(
                    {str(col) + '_' + main_col + '_cnt': 'count'})
                df_merge = pd.merge(df_merge, itemcnt, on=[col, main_col], how='left')
                df_merge[str(col) + '_' + main_col + '_prob'] = df_merge[str(col) + '_' + main_col + '_cnt'] / df_merge[
                    main_col + '_cnt']  # 结合后占比

                new_column_list.extend([str(col) + '_' + main_col + '_cnt', str(col) + '_' + main_col + '_prob'])

        item_feature_data = df_merge[['instance_id']+new_column_list]
        item_feature_data.to_pickle(item_feature_path)

        return df_merge

    def add_user_feature(self, df_merge):

        print('user相关统计')
        user_feature_path = config.cache_prefix_path + 'user_feature.pkl'
        if os.path.exists(user_feature_path):
            user_feature_data = pd.read_pickle(user_feature_path)
            return pd.merge(df_merge, user_feature_data, on=['instance_id'], how='left')

        new_column_list = []

        cols = ['user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']

        for main_col in ['user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']:
            print(main_col + ' 出现了几次')

            itemcnt = df_merge.groupby([main_col], as_index=False)['instance_id'].agg({main_col + '_cnt': 'count'})
            df_merge = pd.merge(df_merge, itemcnt, on=[main_col], how='left')
            new_column_list.append(main_col + '_cnt')

            cols.remove(main_col)

            if main_col == 'user_id':
                continue

            print(main_col + ' 和 ' + str(cols) + ' 同时出现了几次\n')

            for col in cols:
                itemcnt = df_merge.groupby([col, main_col], as_index=False)['instance_id'].agg(
                    {str(col) + '_' + main_col + '_cnt': 'count'})
                df_merge = pd.merge(df_merge, itemcnt, on=[col, main_col], how='left')
                df_merge[str(col) + '_' + main_col + '_prob'] = df_merge[str(col) + '_' + main_col + '_cnt'] / df_merge[
                    main_col + '_cnt']  # 结合后占比

                new_column_list.extend([str(col) + '_' + main_col + '_cnt', str(col) + '_' + main_col + '_prob'])

        user_feature_data = df_merge[['instance_id']+new_column_list]
        user_feature_data.to_pickle(user_feature_path)

        return df_merge



    def add_shop_feature(self, df_merge):

        print('shop相关统计')
        shop_feature_path = config.cache_prefix_path + 'shop_feature.pkl'
        if os.path.exists(shop_feature_path):
            user_feature_data = pd.read_pickle(shop_feature_path)
            return pd.merge(df_merge, user_feature_data, on=['instance_id'], how='left')

        new_column_list = []

        cols = ['shop_id','shop_review_num_level','shop_star_level']

        for main_col in ['shop_id','shop_review_num_level','shop_star_level']:
            print(main_col + ' 出现了几次')

            itemcnt = df_merge.groupby([main_col], as_index=False)['instance_id'].agg({main_col + '_cnt': 'count'})
            df_merge = pd.merge(df_merge, itemcnt, on=[main_col], how='left')
            new_column_list.append(main_col + '_cnt')

            cols.remove(main_col)

            if main_col == 'shop_id':
                continue

            print(main_col + ' 和 ' + str(cols) + ' 同时出现了几次\n')

            for col in cols:
                itemcnt = df_merge.groupby([col, main_col], as_index=False)['instance_id'].agg(
                    {str(col) + '_' + main_col + '_cnt': 'count'})
                df_merge = pd.merge(df_merge, itemcnt, on=[col, main_col], how='left')
                df_merge[str(col) + '_' + main_col + '_prob'] = df_merge[str(col) + '_' + main_col + '_cnt'] / df_merge[
                    main_col + '_cnt']  # 结合后占比

                new_column_list.extend([str(col) + '_' + main_col + '_cnt', str(col) + '_' + main_col + '_prob'])

        shop_feature_data = df_merge[['instance_id']+new_column_list]
        shop_feature_data.to_pickle(shop_feature_path)

        return df_merge

    def add_user_shop_feature(self, df_merge):

        print('用户、商店属性之间的统计')

        user_shop_feature_path = config.cache_prefix_path + 'user_shop_feature.pkl'
        if os.path.exists(user_shop_feature_path):
            user_shop_feature_data = pd.read_pickle(user_shop_feature_path)
            return pd.merge(df_merge, user_shop_feature_data, on=['instance_id'], how='left')

        user_shop_column = []  # 记录增加多少特征

        cols = ['shop_id', 'shop_review_num_level', 'shop_star_level']

        for main_col in ['user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']:
            print(main_col + ' 和 ' + str(cols) + ' 同时出现了几次\n')

            for col in cols:
                itemcnt = df_merge.groupby([col, main_col], as_index=False)['instance_id'].agg(
                    {str(col) + '_' + main_col + '_cnt': 'count'})
                df_merge = pd.merge(df_merge, itemcnt, on=[col, main_col], how='left')
                df_merge[str(col) + '_' + main_col + '_prob'] = df_merge[str(col) + '_' + main_col + '_cnt'] / df_merge[
                    main_col + '_cnt']  # 结合后占比

                user_shop_column.extend([str(col) + '_' + main_col + '_cnt',str(col) + '_' + main_col + '_prob'])

        df_merge[user_shop_column + ['instance_id']].to_pickle(user_shop_feature_path)
        return df_merge

    def add_user_item_feature(self, df_merge):

        print('用户、物品属性之间的统计')

        user_item_feature_path = config.cache_prefix_path + 'user_item_feature.pkl'
        if os.path.exists(user_item_feature_path):
            user_item_feature_data = pd.read_pickle(user_item_feature_path)
            return pd.merge(df_merge, user_item_feature_data, on=['instance_id'], how='left')

        user_item_column = []  # 记录增加多少特征

        cols = ['item_id', 'item_brand_id', 'item_city_id', 'item_price_level', \
                'item_sales_level', 'item_collected_level', 'item_pv_level']

        for main_col in ['user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']:
            print(main_col + ' 和 ' + str(cols) + ' 同时出现了几次\n')

            for col in cols:

                if col == 'item_id' and main_col == 'user_id':
                    continue

                itemcnt = df_merge.groupby([col, main_col], as_index=False)['instance_id'].agg(
                    {str(col) + '_' + main_col + '_cnt': 'count'})
                df_merge = pd.merge(df_merge, itemcnt, on=[col, main_col], how='left')
                df_merge[str(col) + '_' + main_col + '_prob'] = df_merge[str(col) + '_' + main_col + '_cnt'] / df_merge[
                    main_col + '_cnt']  # 结合后占比

                user_item_column.extend([str(col) + '_' + main_col + '_cnt', str(col) + '_' + main_col + '_prob'])

        df_merge[user_item_column + ['instance_id']].to_pickle(user_item_feature_path)

        return df_merge

    def add_shop_item_feature(self, df_merge):

        print('物品、商店属性之间的统计')

        item_shop_feature_path = config.cache_prefix_path + 'item_shop_feature.pkl'
        if os.path.exists(item_shop_feature_path):
            item_shop_feature_data = pd.read_pickle(item_shop_feature_path)
            return pd.merge(df_merge, item_shop_feature_data, on=['instance_id'], how='left')

        shop_item_column = []  # 记录增加多少特征

        cols = ['item_category_list', 'item_brand_id', 'item_city_id', \
                'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']

        for main_col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:

            print(main_col + ' 和 ' + str(cols) + ' 同时出现了几次\n')

            for col in cols:

                itemcnt = df_merge.groupby([col, main_col], as_index=False)['instance_id'].agg(
                    {str(col) + '_' + main_col + '_cnt': 'count'})
                df_merge = pd.merge(df_merge, itemcnt, on=[col, main_col], how='left')
                df_merge[str(col) + '_' + main_col + '_prob'] = df_merge[str(col) + '_' + main_col + '_cnt'] / df_merge[
                    main_col + '_cnt']  # 结合后占比

                shop_item_column.extend([str(col) + '_' + main_col + '_cnt', str(col) + '_' + main_col + '_prob'])

        df_merge[shop_item_column + ['instance_id']].to_pickle(item_shop_feature_path)
        return df_merge


    def load_data_with_target(self):
        print('开始')

        data = self.gen_global_index()

        data = self.process_miss_value(data)
        data = self.convert_data_in_timestamp(data)
        gc.collect()

        data = self.process_numeric(data)

        # 列表特征
        data = self.add_property_list_item(data)
        data = self.add_user_click_cat(data)
        gc.collect()

        # 时间差特征
        data = self.add_diffTime_feature(data)
        gc.collect()

        data = self.add_slide_count(data)
        data = self.add_slide_is_trade_count_smooth(data) #item_id,shop_id,brand_id
        gc.collect()

        data = self.add_item_feature(data)
        data = self.add_user_feature(data)
        data = self.add_shop_feature(data)
        gc.collect()

        data = self.add_user_shop_feature(data)
        data = self.add_user_item_feature(data)
        data = self.add_shop_item_feature(data)
        gc.collect()

        data = self.get_trick_1(data)
        data = self.get_trick_2(data)
        data = self.get_trick_3(data)
        gc.collect()

        data = self.add_two_feature_smooth_rate(data)
        data = self.add_three_feature_smooth_rate(data)
        gc.collect()

        data = self.PurchaseRate(data)

        data.to_pickle(config.data_prefix_path + 'data')

        return data

    def rate_three(self,data,col_1,col_2,col_3,type,col_pre,new_column_list):
        show_count_name = col_1 + '_' + col_2 + '_' + col_3 + col_pre + '_show_count'
        trade_count_name = col_1 + '_' + col_2 + '_' + col_3 + col_pre + '_trade_count'
        DATA = pd.DataFrame()
        for d in range(19, 26):  # 18到24号


            if type == 1:
                df1 = data[data['day'] == d - 1]
            else:
                df1 = data[data['day'] <= d - 1]
            # 缩小范围
            df1 = df1[[col_1,col_2,col_3,'is_trade']]
            #点击次数
            show_count = df1.groupby([col_1,col_2,col_3]).size().reset_index().rename(
                columns={0: show_count_name})
            #交易次数
            trade_count = df1.groupby([col_1,col_2,col_3]).sum().reset_index().rename(
                columns={'is_trade': trade_count_name})

            show_trade_count = pd.merge(trade_count, show_count, on=[col_1,col_2,col_3], how='right')

            show_trade_count['day'] = d

            DATA = DATA.append(show_trade_count)
        data = pd.merge(data, DATA, on=[col_1, col_2, col_3, 'day'], how='left')
        del DATA,show_count,trade_count,show_trade_count

        # temp = data[[show_count_name, trade_count_name]]
        # print(temp.shape)
        # temp = temp.dropna()
        # print(temp.shape)
        # C = temp[show_count_name].values
        #
        # I = temp[trade_count_name].values
        # bs = HyperParam(1,1)
        # bs.update_from_data_by_moment(I, C)
        #
        # print(bs.alpha, bs.beta)

        data[col_1+'_' +col_2+'_'+col_3+col_pre + '_show_trade_smooth_rate'] = (data[trade_count_name]) / (data[show_count_name])

        del data[trade_count_name]

        new_column_list.append(col_1+'_' +col_2+'_'+col_3+col_pre+'_show_trade_smooth_rate')

        return data,new_column_list

    def add_three_feature_smooth_rate(self,data):
        path = config.cache_prefix_path + 'three_feature_smooth_rate.pkl'
        if os.path.exists(path):
            slide_count_data = pd.read_pickle(path)
            return pd.merge(data, slide_count_data, on=['instance_id'], how='left')
        list=[]

        # for col_1 in ['user_id','user_star_level','age_star','user_gender_id','user_occupation_id']:
        #     for col_2 in ['item_id','item_brand_id','item_city_id','item_price_level','item_sales_level',
        #                   'item_collected_level','item_pv_level','shop_star_level','category_1','category_2']:
        #         for col_3 in ['hour','context_page_id']:
        #             data,list = self.rate_three(data,col_1=col_1,col_2=col_2,col_3=col_3,col_pre='_pre_all',type=2,new_column_list=list)
        data, list = self.rate_three(data, col_1='user_star_level', col_2='user_gender_id', col_3='category_1', col_pre='_pre_all', type=2,
                                   new_column_list=list)
        data, list = self.rate_three(data, col_1='user_star_level', col_2='user_age_level', col_3='category_1',
                                     col_pre='_pre_all', type=2,
                                     new_column_list=list)
        data, list = self.rate_three(data, col_1='user_gender_id', col_2='user_age_level', col_3='category_1',
                                     col_pre='_pre_all', type=2,
                                     new_column_list=list)
        data, list = self.rate_three(data, col_1='user_occupation_id', col_2='user_age_level', col_3='category_1',
                                     col_pre='_pre_all', type=2,
                                     new_column_list=list)

        data[['instance_id']+list].to_pickle(path)
        return data

    def rate_two(self, data, col_1, col_2, type, col_pre, new_column_list):
        show_count_name = col_1 + '_' + col_2 + col_pre + '_show_count'
        trade_count_name = col_1 + '_' + col_2 + col_pre + '_trade_count'
        DATA = pd.DataFrame()
        for d in range(19, 26):  # 18到24号
            if type == 1:
                df1 = data[data['day'] == d - 1]
            else:
                df1 = data[data['day'] <= d - 1]
            # 缩小范围
            df1 = df1[[col_1, col_2, 'is_trade']]
            # 点击次数
            show_count = df1.groupby([col_1, col_2]).size().reset_index().rename(
                columns={0: show_count_name})
            # 交易次数
            trade_count = df1.groupby([col_1, col_2]).sum().reset_index().rename(
                columns={'is_trade': trade_count_name})

            show_trade_count = pd.merge(trade_count, show_count, on=[col_1, col_2], how='right')

            show_trade_count['day'] = d

            DATA = DATA.append(show_trade_count)

            print(DATA.shape)

        data = pd.merge(data, DATA, on=[col_1, col_2, 'day'], how='left')
        del DATA, show_count, trade_count, show_trade_count
        temp = data[[show_count_name, trade_count_name]]

        # print(temp.shape[0])
        # temp = temp.dropna()
        # print(temp.shape)
        # C = temp[trade_count_name].values
        # print(C.shape)
        # I = temp[show_count_name].values
        bs = HyperParam(1, 1)
        bs.alpha = 0
        bs.beta = 0
        # bs.update_from_data_by_moment(I, C)
        # bs.update(I, C, 1000,0.01)
        print(bs.alpha, bs.beta)

        data[col_1 + '_' + col_2 + col_pre + '_show_trade_smooth_rate'] = (data[trade_count_name] + bs.alpha) / (
        data[show_count_name] + bs.beta)

        del temp, data[show_count_name], data[trade_count_name]
        new_column_list.append(col_1 + '_' + col_2 + col_pre + '_show_trade_smooth_rate')

        return data, new_column_list

    def add_two_feature_smooth_rate(self, data):
        path = config.cache_prefix_path + 'two_feature_smooth_rate.pkl'
        if os.path.exists(path):
            slide_count_data = pd.read_pickle(path)
            return pd.merge(data, slide_count_data, on=['instance_id'], how='left')
        list = []
        # for col_1,col_2 in [('user_id','item_id'),('user_id','item_brand_id'),('user_id','item_city_id'),
        #                     ('user_id','context_page_id'),
        #                     ('user_id','shop_id'),('user_id','shop_id')]:
        for col_1 in ['user_id']:
            for col_2 in ['item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
                          'item_collected_level', 'item_pv_level', 'context_page_id', 'shop_star_level', 'category_1',
                          'category_2']:
                data, list = self.rate_two(data, col_1=col_1, col_2=col_2, col_pre='_pre_all', type=2,
                                           new_column_list=list)
                gc.collect()

        for col_1 in ['user_star_level']:
            for col_2 in ['item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
                          'item_collected_level', 'item_pv_level', 'context_page_id', 'shop_star_level', 'category_1',
                          'category_2']:
                data, list = self.rate_two(data, col_1=col_1, col_2=col_2, col_pre='_pre_all', type=2,
                                           new_column_list=list)
                gc.collect()

        for col_1 in ['item_id', 'category_1']:
            for col_2 in ['context_page_id', 'hour']:
                data, list = self.rate_two(data, col_1=col_1, col_2=col_2, col_pre='_pre_all', type=2,
                                           new_column_list=list)

        data, list = self.rate_two(data, col_1='category_1', col_2='user_age_level', col_pre='_pre_all', type=2,
                                   new_column_list=list)
        gc.collect()

        data[['instance_id'] + list].to_pickle(path)
        return data

    def get_trick_1(self, data):

        print('trick_1')
        trick1_feature_path = config.cache_prefix_path + 'trick1_feature.pkl'
        if os.path.exists(trick1_feature_path):
            trick1_feature_data = pd.read_pickle(trick1_feature_path)
            return pd.merge(data, trick1_feature_data, on=['instance_id'], how='left')

        trick1_columns = []

        #这个有点迷
        for col in ['hour']:
            data[col] = data[col].astype(str)
        data['hour_and_category_1'] = data['hour']+data['category_1']
        for col in ['hour']:
            data[col] = data[col].astype(int)

        trick1_columns.append('hour_and_category_1')

        data['map_hour'] = data['hour'].apply(lambda x: self.map_hour(x))
        trick1_columns.append('map_hour')

        grouped = data.groupby('user_id')['hour'].mean().reset_index()
        grouped.columns = ['user_id', 'user_mean_hour']
        data = data.merge(grouped, how='left', on='user_id')
        trick1_columns.append('user_mean_hour')

        data[['instance_id'] + trick1_columns].to_pickle(trick1_feature_path)

        return data

    def get_trick_2(self, data):
        print("trick_2")
        path = config.cache_prefix_path + 'trick2_feature.pkl'
        if os.path.exists(path):
            slide_count_data = pd.read_pickle(path)
            return pd.merge(data, slide_count_data, on=['instance_id'], how='left')

        print("前2个小时的统计量")

        result = pd.DataFrame()

        for day in range(19,26):
            df = data[data.day < day]
            item_id_user_number = 'item_id_user_number_pre_all'
            item_id_trade_number = 'item_id_trade_number_pre_all'

            temp = df.groupby(['user_id', 'item_id']).size().reset_index()
            temp = temp.groupby(['item_id']).size().reset_index()
            # item_id_user_number:item_id被不同的user_id的点击次数
            temp.columns = ['item_id', item_id_user_number]

            cont = df.groupby(['item_id'])['is_trade'].sum().reset_index().rename(
                columns={'is_trade': item_id_trade_number})
            # item_id_trade_number:item_id的购买次数
            cont = pd.merge(temp, cont, on=['item_id'], how='right')
            # 点击了该item_id的用户，平均购买了几次
            cont[item_id_trade_number+'_rate'] = cont[item_id_trade_number] / cont[item_id_user_number]

            cont['day'] = day
            result = result.append(cont)

        data = pd.merge(data,result,on=['item_id','day'],how='left')

        data[['item_id_user_number_pre_all','item_id_trade_number_pre_all',
              'item_id_trade_number_pre_all'+'_rate','instance_id']].to_pickle(path)

        return data

    def get_trick_3(self, data):

        print('trick_3')
        trick3_feature_path = config.cache_prefix_path + 'trick3_feature.pkl'
        if os.path.exists(trick3_feature_path):
            trick3_feature_data = pd.read_pickle(trick3_feature_path)
            return pd.merge(data, trick3_feature_data, on=['instance_id'], how='left')

        trick3_columns = []

        result = pd.DataFrame()

        for day in range(19,26):
            df = data[data.day < day]
            page_id_user_number = 'page_id_user_number_pre_all'
            page_id_trade_number = 'page_id_trade_number_pre_all'

            temp = df.groupby(['user_id', 'context_page_id']).size().reset_index()
            temp = temp.groupby(['context_page_id']).size().reset_index()
            temp.columns = ['context_page_id', page_id_user_number]
            # page_id_user_number:每一页被多少不同的用户点击
            cont = df.groupby(['context_page_id'])['is_trade'].sum().reset_index().rename(
                columns={'is_trade': page_id_trade_number})
            # 点击了该页的用户，平均购买了几次
            cont = pd.merge(temp, cont, on=['context_page_id'], how='right')
            cont[page_id_trade_number+'_rate'] = cont[page_id_trade_number] / cont[page_id_user_number]

            cont['day'] = day
            result = result.append(cont)
            print(result.shape)

        trick3_columns.extend(
            ['page_id_user_number_pre_all','page_id_trade_number_pre_all','page_id_trade_number_pre_all'+'_rate'])
        data = pd.merge(data,result,on=['context_page_id','day'],how='left')

        result = pd.DataFrame()
        for day in range(19,26):

            df = data[data['day'] < day ]
            df = df[df['day'] >= day - 2]

            item_id_user_number = 'page_id_user_number_pre_2'
            item_id_trade_number = 'page_id_trade_number_pre_2'

            temp = df.groupby(['user_id', 'context_page_id']).size().reset_index()
            temp = temp.groupby(['context_page_id']).size().reset_index()
            temp.columns = ['context_page_id', item_id_user_number]

            cont = df.groupby(['context_page_id'])['is_trade'].sum().reset_index().rename(
                columns={'is_trade': item_id_trade_number})
            cont = pd.merge(temp, cont, on=['context_page_id'], how='right')
            cont[item_id_trade_number+'_rate'] = cont[item_id_trade_number] / cont[item_id_user_number]
            # del cont[item_id_trade_number], cont[item_id_user_number]
            cont['day'] = day
            result = result.append(cont)
            print(result.shape)

        trick3_columns.extend(
            ['page_id_user_number_pre_2', 'page_id_trade_number_pre_2', 'page_id_trade_number_pre_2' + '_rate'])
        data = pd.merge(data,result,on=['context_page_id','day'],how='left')

        data[['instance_id'] + trick3_columns].to_pickle(trick3_feature_path)

        return data

    def get_brand_info(self,data):
        print('品牌相关信息统计')

        brand_info_feature_path = config.cache_prefix_path + 'brand_info_feature.pkl'
        if os.path.exists(brand_info_feature_path):
            brand_info_feature_data = pd.read_pickle(brand_info_feature_path)
            return pd.merge(data, brand_info_feature_data, on=['instance_id'], how='left')

        # get the features of item brand
        # 记录每个品牌有多少个不同的'item_id','shop_id'
        nums_item = data.groupby('item_brand_id', as_index=False)['item_id'].agg(
            {'item_nums_brand': lambda x: len(x.unique())})
        nums_shop = data.groupby('item_brand_id', as_index=False)['shop_id'].agg(
            {'shop_nums_brand': lambda x: len(x.unique())})

        # 记录每个品牌有多少不同的'user_id'
        nums_user = data.groupby('item_brand_id', as_index=False)['user_id'].agg(
            {'fans_nums_brand': lambda x: len(x.unique())})

        # 记录每个品牌的平均收藏等级
        avg_collected = data.groupby('item_brand_id', as_index=False)['item_collected_level'].agg(
            {'avg_collected_brand': lambda x: x.sum() / x.shape[0]})

        brand_info = None
        for tmp in [nums_item, nums_shop, nums_user, avg_collected]:
            if brand_info is None:
                brand_info = tmp
            else:
                brand_info = pd.merge(left=brand_info, right=tmp, how='left', on='item_brand_id')

        brand_info['item_per_shop'] = brand_info['item_nums_brand'] / brand_info['shop_nums_brand']
        # normalization
        for col in brand_info.columns:
            if col not in ['item_brand_id', 'item_per_shop']:
                mean = brand_info[col].mean()
                std = brand_info[col].std()
                brand_info[col] = (brand_info[col] - mean) / std

        data = pd.merge(data, brand_info, how='left', on='item_brand_id')

        newColumns = brand_info.columns.tolist()
        newColumns.remove('item_brand_id')

        data[newColumns+ ['instance_id']].to_pickle(brand_info_feature_path)

        return data

    def get_shop_info(self,data):
        print('店铺相关信息统计')

        shop_info_feature_path = config.cache_prefix_path + 'shop_info_feature.pkl'
        if os.path.exists(shop_info_feature_path):
            shop_info_feature_data = pd.read_pickle(shop_info_feature_path)
            return pd.merge(data, shop_info_feature_data, on=['instance_id'], how='left')

        nums_item = data.groupby('shop_id', as_index=False)['item_id'].agg(
            {'item_nums_shop': lambda x: len(x.unique())})
        nums_brand = data.groupby('shop_id', as_index=False)['item_brand_id'].agg(
            {'brand_nums_shop': lambda x: len(x.unique())})

        nums_user = data.groupby('shop_id', as_index=False)['user_id'].agg(
            {'fans_nums_shop': lambda x: len(x.unique())})

        shop_info = None
        for tmp in [nums_item, nums_brand, nums_user]:
            if shop_info is None:
                shop_info = tmp
            else:
                shop_info = pd.merge(left=shop_info, right=tmp, how='left', on='shop_id')

        shop_info['item_brand'] = shop_info['item_nums_shop'] / shop_info['brand_nums_shop']

        for col in shop_info.columns:
            if col not in ['shop_id', 'item_brand', 'shop_review_positive_rate', 'shop_score_service',
                           'shop_score_delivery', 'shop_score_description']:
                mean = shop_info[col].mean()
                std = shop_info[col].std()
                shop_info[col] = (shop_info[col] - mean) / std

        data = pd.merge(left = data, right = shop_info, how='left', on='shop_id')

        newColumns = shop_info.columns.tolist()
        newColumns.remove('shop_id')

        data[newColumns + ['instance_id']].to_pickle(shop_info_feature_path)

        return data


    def add_slide_count(self,df_merge):
        print('hour时间滑窗统计')
        if os.path.exists(config.cache_prefix_path + 'slide_count.pkl'):
            slide_count_data = pd.read_pickle(config.cache_prefix_path + 'slide_count.pkl')
            return pd.merge(df_merge, slide_count_data, on=['instance_id'], how='left')

        print("当天当前小时之前24小时的统计量\n")
        list_1 = ['user_hour_cntx', 'item_hour_cntx', 'shop_hour_cntx']

        for d in range(18, 26):
            # 19到25，25是test
            for h in range(0, 24):
                # df_day_hour1：（今天且在该hour之前)或者(昨天且在该小时之后)
                df_day_hour1 = df_merge[((df_merge['day'] == d) & (df_merge['hour'] < h)) | ((df_merge['day'] == d - 1) & (h <= df_merge['hour']))]
                # df_day_hour2 : 当前日期，当前时间
                df_day_hour2 = df_merge[(df_merge['day'] == d) & (df_merge['hour'] == h)]

                user_hour_cnt = df_day_hour1.groupby('user_id').count()['instance_id'].to_dict()
                item_hour_cnt = df_day_hour1.groupby('item_id').count()['instance_id'].to_dict()
                shop_hour_cnt = df_day_hour1.groupby('shop_id').count()['instance_id'].to_dict()

                df_day_hour2['user_hour_cntx'] = df_day_hour2['user_id'].apply(lambda x: user_hour_cnt.get(x, 0))
                df_day_hour2['item_hour_cntx'] = df_day_hour2['item_id'].apply(lambda x: item_hour_cnt.get(x, 0))
                df_day_hour2['shop_hour_cntx'] = df_day_hour2['shop_id'].apply(lambda x: shop_hour_cnt.get(x, 0))
                df_day_hour2 = df_day_hour2[['user_hour_cntx', 'item_hour_cntx', 'shop_hour_cntx', 'instance_id']]

                if h == 0:
                    Df = df_day_hour2
                else:
                    Df = pd.concat([df_day_hour2, Df])

            if d == 18:
                DF = Df
            else:
                DF = pd.concat([Df, DF])
        df_merge = pd.merge(df_merge, DF, on=['instance_id'], how='left')


        print("前2个小时的统计量")
        list_2 = ['user_hour_cnt2', 'item_hour_cnt2', 'shop_hour_cnt2']
        for d in range(18, 26):
            # 19到25，25是test
            for h in range(0, 24):

                if h == 0:
                    df_day_hour1 = df_merge[(df_merge['day'] == d - 1) & (df_merge['hour'] == 23)]
                    df_day_hour2 = df_merge[(df_merge['day'] == d - 1) & (df_merge['hour'] == 22)]
                elif h == 1:
                    df_day_hour1 = df_merge[(df_merge['day'] == d) & (df_merge['hour'] == h - 1)]
                    df_day_hour2 = df_merge[(df_merge['day'] == d - 1) & (df_merge['hour'] == 23)]
                else:
                    df_day_hour1 = df_merge[(df_merge['day'] == d) & (df_merge['hour'] == h - 1)]
                    df_day_hour2 = df_merge[(df_merge['day'] == d) & (df_merge['hour'] == h - 2)]

                df_day_hour = df_merge[(df_merge['day'] == d) & (df_merge['hour'] == h)]

                # 前一个小时统计量
                user_hour_cnt1 = df_day_hour1.groupby('user_id').count()['instance_id'].to_dict()
                item_hour_cnt1 = df_day_hour1.groupby('item_id').count()['instance_id'].to_dict()
                shop_hour_cnt1 = df_day_hour1.groupby('shop_id').count()['instance_id'].to_dict()
                # 前两个小时统计量
                user_hour_cnt2 = df_day_hour2.groupby('user_id').count()['instance_id'].to_dict()
                item_hour_cnt2 = df_day_hour2.groupby('item_id').count()['instance_id'].to_dict()
                shop_hour_cnt2 = df_day_hour2.groupby('shop_id').count()['instance_id'].to_dict()

                df_day_hour['user_hour_cnt2'] = df_day_hour['user_id'].apply(lambda x: user_hour_cnt1.get(x, 0) + user_hour_cnt2.get(x, 0))
                df_day_hour['item_hour_cnt2'] = df_day_hour['item_id'].apply(lambda x: item_hour_cnt1.get(x, 0) + item_hour_cnt2.get(x, 0))
                df_day_hour['shop_hour_cnt2'] = df_day_hour['shop_id'].apply(lambda x: shop_hour_cnt1.get(x, 0) + shop_hour_cnt2.get(x, 0))

                df_day_hour = df_day_hour[['user_hour_cnt2', 'item_hour_cnt2', 'shop_hour_cnt2', 'instance_id']]

                if h == 0:
                    Df = df_day_hour
                else:
                    Df = pd.concat([df_day_hour, Df])

            if d == 18:
                Df2 = Df
            else:
                Df2 = pd.concat([Df, Df2])

        df_merge= pd.merge(df_merge, Df2, on=['instance_id'], how='left')

        print("当天前一个小时的统计量\n")  # 待续
        list_3 = ['user_hour_cnt1', 'item_hour_cnt1', 'shop_hour_cnt1']
        for d in range(18, 26):
            # 19到25，25是test

            for h in range(0, 24):

                if h == 0:
                    df_day_hour1 = df_merge[(df_merge['day'] == d - 1) & (df_merge['hour'] == 23)]
                    df_day_hour2 = df_merge[(df_merge['day'] == d) & (df_merge['hour'] == h)]
                else:
                    df_day_hour1 = df_merge[(df_merge['day'] == d) & (df_merge['hour'] == h - 1)]
                    df_day_hour2 = df_merge[(df_merge['day'] == d) & (df_merge['hour'] == h)]

                user_hour_cnt = df_day_hour1.groupby('user_id').count()['instance_id'].to_dict()
                item_hour_cnt = df_day_hour1.groupby('item_id').count()['instance_id'].to_dict()
                shop_hour_cnt = df_day_hour1.groupby('shop_id').count()['instance_id'].to_dict()

                df_day_hour2['user_hour_cnt1'] = df_day_hour2['user_id'].apply(lambda x: user_hour_cnt.get(x, 0))
                df_day_hour2['item_hour_cnt1'] = df_day_hour2['item_id'].apply(lambda x: item_hour_cnt.get(x, 0))
                df_day_hour2['shop_hour_cnt1'] = df_day_hour2['shop_id'].apply(lambda x: shop_hour_cnt.get(x, 0))
                df_day_hour2 = df_day_hour2[['user_hour_cnt1', 'item_hour_cnt1', 'shop_hour_cnt1', 'instance_id']]

                if h == 0:
                    Df = df_day_hour2
                else:
                    Df = pd.concat([df_day_hour2, Df])

            if d == 18:
                DF3 = Df
            else:
                DF3 = pd.concat([Df, DF3])

        df_merge = pd.merge(df_merge, DF3, on=['instance_id'], how='left')

        slide_count_data = df_merge[['instance_id'] + list_1 + list_2 + list_3]
        slide_count_data.to_pickle(config.cache_prefix_path + 'slide_count.pkl')

        return df_merge


    def gen_test_instance_id_list(self):

        if os.path.exists(config.ORIGINAL_TEST_PKL):
            test = pd.read_pickle(config.ORIGINAL_TEST_PKL)
        else:
            test = pd.read_table(config.TEST_FILE, sep=' ')
            test.to_pickle(config.ORIGINAL_TEST_PKL)
        return test

    def gen_global_index(self):

        #读取训练数据
        if os.path.exists(config.ORIGINAL_TRAIN_PKL):
            train = pd.read_pickle(config.ORIGINAL_TRAIN_PKL)
        else:
            train = pd.read_table(config.TRAIN_FILE, sep=' ')
            train.to_pickle(config.ORIGINAL_TRAIN_PKL)

        #读取测试数据
        if os.path.exists(config.ORIGINAL_TEST_PKL):
            test = pd.read_pickle(config.ORIGINAL_TEST_PKL)
        else:
            test = pd.read_table(config.TEST_FILE, sep=' ')
            test.to_pickle(config.ORIGINAL_TEST_PKL)

        '''
        去除重复数据
        训练数据只要有重复，多个重复数值多去除
        训练数据和测试数据重复时，删除训练数据中的重复值
        测试数据没有重复值
        '''
        train.drop_duplicates(subset='instance_id', keep=False, inplace=True)
        all_data = train.append(test).reset_index()
        all_data.drop_duplicates(subset='instance_id', keep='last', inplace=True)
        all_data.reset_index()

        print('加载数据')

        return all_data

    def find_miss_value_is_what(self, data):
        """
        ('int', 'item_brand_id')
        ('int', 'item_city_id')
        ('int', 'item_sales_level')
        ('str', 'predict_category_property')
        ('int', 'user_age_level')
        ('int', 'user_gender_id')
        ('int', 'user_occupation_id')
        ('int', 'user_star_level')
        加上shop的四个连续值特征
        """
        print("缺失值统计")

        for name in data.head(0):
            if -1 in data[name].value_counts():
                print(name, " " * (30 - len(name)), data[name].value_counts()[-1])


    def process_miss_value(self,data):
        # 四个user特征有缺省值
        # 三个item特征有缺省值
        print('填充缺省值')

        # 用众数填充
        # 'user_gendier_id'缺失值较多，感觉没必要填充，直接把缺失值当作一个类别处理（转化为非负数，方便后续处理）
        data['user_gender_id'] = data['user_gender_id'].apply(lambda x: 3 if x == -1 else x)

        # item_city_id,item_price_level,item_collected_level，item_pv_level，context_page_id数据完整
        for col in ['user_age_level', 'user_occupation_id', 'user_star_level']:
            most = 0
            if col == 'user_occupation_id':
                most = 2005
            elif col == 'user_star_level':
                most = 3006
            else:
                most = 1003
            data[col] = data[col].apply(lambda x: most if x == -1 else x)

        data['item_sales_level'] = data['item_sales_level'].apply(lambda x: 12 if x == -1 else x)
        data['item_city_id'] = data['item_city_id'].apply(lambda x: 7534238860363577544 if x == -1 else x)

        # item_brand_id的众数为7838285046767229711
        # item_brand_id为长尾分布，可以用任意值填充
        data['item_brand_id'] = data['item_brand_id'].apply(lambda x: 0 if x == -1 else x)

        return data

    def timestamp_datetime(self, value):
        format = '%Y-%m-%d %H:%M:%S'
        value = time.localtime(value)
        dt = time.strftime(format, value)
        return dt

    # 增加时间段属性,处理时间戳
    def convert_data_in_timestamp(self, all_data):
        print('处理时间')

        if os.path.exists(config.cache_prefix_path+'time.pkl'):
            time_data = pd.read_pickle(config.cache_prefix_path+'time.pkl')
            return pd.merge(all_data, time_data, on=['instance_id'], how='left')

        all_data['time'] = all_data.context_timestamp.apply(self.timestamp_datetime)
        all_data['day'] = all_data.time.apply(lambda x: int(x[8:10]))
        all_data['hour'] = all_data.time.apply(lambda x: int(x[11:13]))

        # 关于一下代码有个疑问，可以尝试优化
        # 25号的测试数据的数据量明显低于其他日期，明显做过抽样（但抽样方法未知，可能是按照小时抽样）
        # 是否应为将25号的这些数量统计相关的组合数据按照比例扩大
        user_query_day = all_data.groupby(['user_id', 'day']).size(
        ).reset_index().rename(columns={0: 'user_query_day'})
        all_data = pd.merge(all_data, user_query_day, 'left', on=['user_id', 'day'])

        user_query_day_hour = all_data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
            columns={0: 'user_query_day_hour'})
        all_data = pd.merge(all_data, user_query_day_hour, 'left',
                        on=['user_id', 'day', 'hour'])

        all_data_time = all_data[['instance_id','time','day','hour','user_query_day','user_query_day_hour']]
        all_data_time.to_pickle(config.cache_prefix_path+'time.pkl')

        all_data.drop('context_timestamp', axis=1, inplace=True)

        return all_data

    def map_hour(self, x):
        if (x >= 7) & (x <= 12):
            return 1
        elif (x >= 13) & (x <= 20):
            return 2
        else:
            return 3


    def process_numeric(self,data):
        # 记录shop,item 相关的分数总和
        data['shop_score_sum'] = data["shop_score_service"]+data["shop_score_delivery"]+data["shop_score_description"]
        data['shop_good'] = data['shop_score_sum'].apply(lambda x:1 if x>2.931 else 0)

        data['shop_score_sum_4'] = data["shop_score_delivery"]+data['shop_score_sum']

        data['item_level_sum_2'] = data['item_sales_level']+data['item_pv_level']

        # 用0来填充缺失值
        for col in config.NUMERIC_COLS:
            data[col] = data[col].apply(lambda x: 0 if x == -1 else x)

        for col in config.NUMERIC_COLS:
            data[col] = data[col].apply(lambda x: np.nan if x == 0 else x)

        return data

    def split_category_list(self,x):
        list = []
        for str in x.split(";"):
            if str == '-1':
                continue
            else:
                list.append(str)
        return list

    def split_to_category(self, x):
        # 切割property_predict_list，保存其类别
        array = x.split(';')
        truth = []
        for cp in array:
            all = cp.split(':')
            if len(all) > 0:
                key = all[0]

                truth.append(key)
        return truth

    def split_to_dict2(self, x):
        # 切割property_predict_list，保存其属性
        array = x.split(';')
        truth = []
        for cp in array:
            all = cp.split(':')
            if len(all) > 1:
                key = all[1]
                for z in key.split(','):
                    if z == '-1':
                        continue
                    else:
                        truth.append(z)
        return truth

    def get_property_matching_number(self,category,category_predict):
        count = 0
        for str in category:
            if str in category_predict:
                count += 1
        return count

    def get_category_cross(self,category,category_predict,index):
        n = len(category)
        if n < index+1:
            return np.nan
        if category[index] in category_predict:
            return category[index]
        else:
            return np.nan


    def add_property_list_item(self, data):
        print('处理列表特征')

        if os.path.exists(config.prefix_path + config.property_category_pkl):
            list_item = pd.read_pickle(config.prefix_path + config.property_category_pkl)
            head = list_item.head()
            return pd.merge(data, list_item, on=['instance_id'], how='left')

        # 处理缺省值
        data['category_list'] = data['item_category_list'].apply(lambda x: self.split_category_list(x))
        # 计算cat_list长度作为一个特征
        data['category_list_num'] = data['category_list'].apply(lambda x: len(x))

        """
        category
        2 494401
        3 2108
        """

        print('property')
        # 处理缺省值
        data['property_list'] = data['item_property_list'].apply(lambda x: self.split_category_list(x))
        # 计算pro_list长度作为一个特征
        data['property_list_num'] = data['property_list'].apply(lambda x: len(x))

        print('property_category')
        # 将预测的属性保存在property_predict_list中
        data['property_predict_list'] = data['predict_category_property'].apply(lambda x: self.split_to_dict2(x))
        # 将预测的类别保存在category_predict_list中
        data['category_predict_list'] = data['predict_category_property'].apply(lambda x: self.split_to_category(x))

        # 计算各个预测类别是否预测正确
        # eg: 如果cat0出现在predict_cat中，则保存为cat_cross_0为category_list号；否则保存为NaN
        # 个人感觉没什么用处，待更新
        for index in range(0, 3):
            data['category_cross_%d' % index] = data[['category_list', 'category_predict_list']].apply(
                lambda x: self.get_category_cross(x.category_list, x.category_predict_list, index), axis=1
            )

        # 匹配正确的类别数量，匹配正确的属性数量
        # 个人认为不如计算IOU
        data['category_matching_number'] = data[['category_list', 'category_predict_list']].apply(
            lambda x: self.get_property_matching_number(x.category_list, x.category_predict_list), axis=1
        )
        data['property_matching_number'] = data[['property_list', 'property_predict_list']].apply(
            lambda x: self.get_property_matching_number(x.property_list, x.property_predict_list), axis=1
        )
        data['cateqory_and_property_matching_number'] = data['category_matching_number'] + data[
            'property_matching_number']

        # 如果cat_list的长度大于i,则将其类别编号
        # 否则，填充NaN
        # 个人感觉没有什么意义
        for i in range(3):
            data['category_%d' % (i)] = data['category_list'].apply(
                lambda x: x[i] if len(x) > i else np.nan
            )
        # 同上，不过是对属性做
        # 个人感觉没什么用处
        for i in range(3):
            data['property_%d' % (i)] = data['property_list'].apply(
                lambda x: x[i] if len(x) > i else np.nan
            )

        # 同上，对预测类别和预测属性做
        # 同个人感觉没什么卵用
        print('取前几个')
        for i in range(3):
            data['predict_category_%d' % (i)] = data['category_predict_list'].apply(
                lambda x: x[i] if len(x) > i else np.nan
            )
        for i in range(3):
            data['predict_property_%d' % (i)] = data['property_predict_list'].apply(
                lambda x: x[i] if len(x) > i else np.nan
            )

        # 添加属性————catIOU,proIOU
        # 每行的单属性出现的平均次数

        print ('category_IOU')
        data['category_IOU'] = data[['category_list', 'category_predict_list']].apply(
            lambda x: self.get_cat_IOU(x.category_list, x.category_predict_list), axis=1
        )

        print ('property_IOU')
        data['property_IOU'] = data[['category_list', 'property_list', 'predict_category_property']].apply(
            lambda x: self.get_pre_IOU(x.category_list, x.property_list, x.predict_category_property.split(";")), axis=1
        )

        # property_dic：计算每个属性出现的次数
        property_dic = self.get_property_dic(data['property_list'])
        # 每行的单属性的平均出现次数,可以衡量该商品属性的热门程度
        data['property_mean_len'] = data[['property_list_num', 'property_list']].apply(
            lambda x: self.get_pro_mean_len(x.property_list_num, x.property_list, property_dic), axis=1
        )

        del property_dic
        gc.collect()

        data.drop(config.LIST_COLS, axis=1, inplace=True)

        train = data[['instance_id', 'category_list_num', 'property_list_num',
                      'category_matching_number', 'property_matching_number', 'cateqory_and_property_matching_number',
                      'category_0', 'category_1', 'category_2',
                      'property_0', 'property_1', 'property_2',
                      'predict_category_0', 'predict_category_1', 'predict_category_2',
                      'predict_property_0', 'predict_property_1', 'predict_property_2',
                      'category_cross_0', 'category_cross_1', 'category_cross_2',
                      'category_IOU','property_IOU','property_mean_len']]
        train.to_pickle(config.prefix_path + config.property_category_pkl)

        return data

    def get_pro_mean_len(self,property_list_num, property_list, property_dic):
        cnt = 0
        for property in property_list:
            cnt += property_dic[property]
        return cnt/property_list_num

    def get_property_dic(self, data):
        sr = data.values
        property_dic = {}

        for property_list in sr:
            for property in property_list:
                property_dic[property] = property_dic.get(property,0) + 1
        return property_dic


    def get_cat_IOU(self, category_list, category_predict_list):
        '''
        cat0: 7908382889764677758
        存在cat2的cat1: 2642175453151805566
        cat2:8868887661186419229,6233669177166538628
        '''
        cat0 = '7908382889764677758'

        cat_set = set(category_list)
        pre_cat_set = set(category_predict_list)

        cat_set.remove(cat0)

        if cat0 in pre_cat_set:
            pre_cat_set.remove(cat0)

        # 处理cat2
        cat_set = self.deal_cat2(cat_set)
        pre_set = self.deal_cat2(pre_cat_set)

        # 计算IOU
        inter = len(cat_set & pre_cat_set)
        union = len(cat_set | pre_cat_set)

        return inter/union

    def deal_cat2(self,cat_set, cat1='2642175453151805566', cat2_0='8868887661186419229', cat2_1='6233669177166538628'):
        if cat1 in cat_set:
            if cat2_0 in cat_set:
                cat_set.add(cat1 + ':' + cat2_0)
            if cat2_1 in cat_set:
                cat_set.add(cat1 + ':' + cat2_1)
        return cat_set

    def get_pre_IOU(self, category_list, property_list, predict_category_property):
        # cat0预测正确不算预测正确
        cat_set = set(category_list[1:])
        pro_set = set(property_list)

        hit_pro_list = []

        for cat_pro in predict_category_property:
            if cat_pro == '-1':
                continue
            cat = cat_pro.split(":")[0]
            pros = cat_pro.split(":")[1]

            if cat in cat_set:  # 预测正确
                if pros != '-1':
                    hit_pro_list.extend(pros.split(","))

        hit_pro_set = set(hit_pro_list)
        inter = len(pro_set & hit_pro_set)
        union = len(pro_set | hit_pro_set)

        return inter / union


if  __name__ == "__main__":
    f = Feature()
    f.load_data_with_target()




