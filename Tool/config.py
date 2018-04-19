import os

data_prefix_path = '../../DataSet/IJCAI-18 Alimama Sponsored Search Conversion Rate(CVR) Prediction Contest/'

test_text = data_prefix_path + 'test_b.csv'
train_text = data_prefix_path + 'train.csv'




prefix_path = os.getcwd()  #当前工作目录
prefix_path = prefix_path.replace(prefix_path.split("\\")[-1],'')


cache_prefix_path = prefix_path + 'Cache/'
property_category_pkl = "Cache/property_category_process.pkl"
original_pkl = "original_train.pkl"
original_test_pkl = "original_test.pkl"
TEST_FILE = test_text
TRAIN_FILE = train_text

ORIGINAL_TRAIN_PKL = cache_prefix_path + original_pkl
ORIGINAL_TEST_PKL = cache_prefix_path + original_test_pkl
IMAGE_PATH = prefix_path + 'Image/'


# instance_id 样本编号
# item_id 广告商品编号
# item_category_list 广告商品的的类目列表 
# item_property_list 广告商品的属性列表 
# item_brand_id 广告商品的品牌编号
# item_city_id 广告商品的城市编号
# item_price_level 广告商品的价格等级
# item_sales_level 广告商品的销量等级
# item_collected_level 广告商品被收藏次数的等级
# item_pv_level 广告商品被展示次数的等级
# user_id 用户的编号
# user_gender_id, 用户的预测性别编号
# user_age_level, 用户的预测年龄等级
# user_occupation_id, 用户的预测职业编号
# user_star_level 用户的星级编号
# context_id 上下文信息的编号
# context_timestamp 广告商品的展示时间
# context_page_id 广告商品的展示页面编号
# predict_category_property
NUM_SPLITS = 2
RANDOM_SEED = 2017
DEBUG = False

SPECIAL_CATEGORICAL_COLS = [
    "item_city_id",
    "item_brand_id"
]

LIST_COLS = [
    'item_category_list',
    'item_property_list',
    'predict_category_property'
]

# types of columns of the dataset dataframe分类特征
CATEGORICAL_COLS = [

    "item_price_level",
    "item_sales_level",
    "item_collected_level",
    "item_pv_level",
    "user_gender_id",
    "user_age_level",
    "user_occupation_id",
    "user_star_level",
    "context_page_id",

    "shop_review_num_level",
    "shop_star_level",

    #特征工程
    "day",
    "hour",
    "predict_category_1",
    "predict_category_2",
    "predict_category_0",
    "predict_property_0",
    "predict_property_1",
    "predict_property_2",

    "property_1",
    "property_0",

    "category_1",
    "category_0",
    
    #特征工程
    "user_query_day_hour",#用户这个小时，查询了多少次
    "user_query_day",#用户一天，查询了多少次

    "property_matching_number",
    "category_matching_number",
]

#连续特征
NUMERIC_COLS = [
    "shop_score_service",
    "shop_score_delivery",
    "shop_score_description",
    "shop_review_positive_rate",
]

IGNORE_COLS = [
    # "is_trade",
    "instance_id",
    "item_id",
    "user_id",
    "shop_id",
    "context_id"
]