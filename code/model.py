# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 09:40:49 2018

@author: yuwei
"""

import pandas as pd
import numpy as np
import math
import random
import time
import scipy as sp
import xgboost as xgb



def loadData():
    "下载数据"
    trainSet = pd.read_table('round1_ijcai_18_train_20180301.txt',sep=' ')
    testSet = pd.read_table('round1_ijcai_18_test_a_20180301.txt',sep=' ')
    return trainSet,testSet

def splitData(trainSet,testSet):
    "按时间划分验证集"
    #转化测试集时间戳为标准时间
    time_local = testSet.context_timestamp.map(lambda x :time.localtime(x))
    time_local = time_local.map(lambda x :time.strftime("%Y-%m-%d %H:%M:%S",x))
    testSet['context_timestamp'] = time_local
    #转化训练集时间戳为标准时间
    time_local = trainSet.context_timestamp.map(lambda x :time.localtime(x))
    time_local = time_local.map(lambda x :time.strftime("%Y-%m-%d %H:%M:%S",x))
    trainSet['context_timestamp'] = time_local
    del time_local
    
    #处理训练集item_category_list属性
    trainSet['item_category_list'] = trainSet.item_category_list.map(lambda x :x.split(';'))
    trainSet['item_category_list_2'] = trainSet.item_category_list.map(lambda x :x[1])
    trainSet['item_category_list_3'] = trainSet.item_category_list.map(lambda x :x[2] if len(x) >2 else -1)
    trainSet['item_category_list_2'] = list(map(lambda x,y : x if (y == -1) else y,trainSet['item_category_list_2'],trainSet['item_category_list_3']))
    #处理测试集item_category_list属性
    testSet['item_category_list'] = testSet.item_category_list.map(lambda x :x.split(';'))
    testSet['item_category_list_2'] = testSet.item_category_list.map(lambda x :x[1])
    testSet['item_category_list_3'] = testSet.item_category_list.map(lambda x :x[2] if len(x) >2 else -1)
    testSet['item_category_list_2'] = list(map(lambda x,y : x if (y == -1) else y,testSet['item_category_list_2'],testSet['item_category_list_3']))
    del trainSet['item_category_list_3'];del testSet['item_category_list_3'];

    #处理predict_category_property的排名
    trainSet['predict_category'] = trainSet['predict_category_property'].map(lambda x :[y.split(':')[0] for y in x.split(';')])
    trainSet['predict_category_property_rank'] = list(map(lambda x,y:y.index(x) if x in y else -1,trainSet['item_category_list_2'],trainSet['predict_category']))
    testSet['predict_category'] = testSet['predict_category_property'].map(lambda x :[y.split(':')[0] for y in x.split(';')])
    testSet['predict_category_property_rank'] = list(map(lambda x,y:y.index(x) if x in y else -1,testSet['item_category_list_2'],testSet['predict_category']))
    #统计item_category_list中和predict_category共同的个数
    trainSet['item_category_count'] = list(map(lambda x,y:len(set(x)&set(y)),trainSet.item_category_list,trainSet.predict_category))
    testSet['item_category_count'] = list(map(lambda x,y:len(set(x)&set(y)),testSet.item_category_list,testSet.predict_category))
    #不同个数
    trainSet['item_category_count'] = list(map(lambda x,y:len(set(x)) - len(set(x)&set(y)),trainSet.item_category_list,trainSet.predict_category))
    testSet['item_category_count'] = list(map(lambda x,y:len(set(x)) - len(set(x)&set(y)),testSet.item_category_list,testSet.predict_category))
    del trainSet['predict_category']; del testSet['predict_category']
    

    "划分数据集"
    #测试集 23-24号特征提取,25号打标
    test = testSet
    testFeat = trainSet[trainSet['context_timestamp']>'2018-09-23']
    #验证集 22-23号特征提取,24号打标
    validate = trainSet[trainSet['context_timestamp']>'2018-09-24']
    validateFeat = trainSet[(trainSet['context_timestamp']>'2018-09-22') & (trainSet['context_timestamp']<'2018-09-24')]
    #训练集 21-22号特征提取,23号打标;20-21号特征提取,22号打标;19-20号特征提取,21号打标;18-19号特征提取,20号打标
    #标签区间
    train1 = trainSet[(trainSet['context_timestamp']>'2018-09-23') & (trainSet['context_timestamp']<'2018-09-24')]
    train2 = trainSet[(trainSet['context_timestamp']>'2018-09-22') & (trainSet['context_timestamp']<'2018-09-23')]
    train3 = trainSet[(trainSet['context_timestamp']>'2018-09-21') & (trainSet['context_timestamp']<'2018-09-22')]
    train4 = trainSet[(trainSet['context_timestamp']>'2018-09-20') & (trainSet['context_timestamp']<'2018-09-21')]
    #特征区间
    trainFeat1 = trainSet[(trainSet['context_timestamp']>'2018-09-21') & (trainSet['context_timestamp']<'2018-09-23')]
    trainFeat2 = trainSet[(trainSet['context_timestamp']>'2018-09-20') & (trainSet['context_timestamp']<'2018-09-22')]
    trainFeat3 = trainSet[(trainSet['context_timestamp']>'2018-09-19') & (trainSet['context_timestamp']<'2018-09-21')]
    trainFeat4 = trainSet[(trainSet['context_timestamp']>'2018-09-18') & (trainSet['context_timestamp']<'2018-09-20')]


    return test,testFeat,validate,validateFeat,train1,trainFeat1,train2,trainFeat2,train3,trainFeat3,train4,trainFeat4
    
def modelXgb(train,test):
    "xgb模型"
    train_y = train['is_trade'].values
#    train_x = train.drop(['item_brand_id','item_city_id','user_id','shop_id','context_id','instance_id', 'item_id','item_category_list','item_property_list', 'context_timestamp', 
#                          'predict_category_property','is_trade'
#                          ],axis=1).values
#    test_x = test.drop(['item_brand_id','item_city_id','user_id','shop_id','context_id','instance_id', 'item_id','item_category_list','item_property_list', 'context_timestamp', 
#                          'predict_category_property','is_trade'
#                          ],axis=1).values
#    test_x = test.drop(['item_brand_id','item_city_id','user_id','shop_id','context_id','instance_id', 'item_id','item_category_list','item_property_list', 'context_timestamp', 
#                          'predict_category_property'
#                          ],axis=1).values

    #根据皮卡尔相关系数，drop相关系数低于-0.2的属性         
    train_x = train.drop(['item_brand_id',
    'item_city_id','user_id','shop_id','context_id',
    'instance_id', 'item_id','item_category_list',
    'item_property_list', 'context_timestamp', 
    'predict_category_property','is_trade',
    'item_price_level','user_rank_down',
    'item_category_list_2_not_buy_count',
    'item_category_list_2_count',
    'user_first'
#    'user_count_label',
#    'item_city_not_buy_count',
#    'item_city_count',
#    'user_shop_rank_down',
#    'item_city_buy_count',
#    'user_item_rank_down',
#    'shop_score_description',
#    'shop_review_positive_rate',
#    'shop_score_delivery',
#    'shop_score_service',
                        ],axis=1).values
                          
#    test_x = test.drop(['item_brand_id',
#    'item_city_id','user_id','shop_id','context_id',
#    'instance_id', 'item_id','item_category_list',
#    'item_property_list', 'context_timestamp', 
#    'predict_category_property','is_trade',
#    'item_price_level','user_rank_down',
#    'item_category_list_2_not_buy_count',
#    'item_category_list_2_count',
#    'user_first',
#    'user_count_label',
#    'item_city_not_buy_count',
#    'item_city_count',
#    'user_shop_rank_down',
#    'item_city_buy_count',
#    'user_item_rank_down',
#    'shop_score_description',
#    'shop_review_positive_rate',
#    'shop_score_delivery',
#    'shop_score_service'
#                        ],axis=1).values
    test_x = test.drop(['item_brand_id',
    'item_city_id','user_id','shop_id','context_id',
    'instance_id', 'item_id','item_category_list',
    'item_property_list', 'context_timestamp', 
    'predict_category_property',
    'item_price_level','user_rank_down',
    'item_category_list_2_not_buy_count',
    'item_category_list_2_count',
    'user_first',
#    'user_count_label',
#    'item_city_not_buy_count',
#    'item_city_count',
#    'user_shop_rank_down',
#    'item_city_buy_count',
#    'user_item_rank_down',
#    'shop_score_description',
#    'shop_review_positive_rate',
#    'shop_score_delivery',
#    'shop_score_service'
                        ],axis=1).values                         
                         
                    
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x)
    
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'binary:logistic',
              'eval_metric':'logloss',
              'eta': 0.03,
              'max_depth': 5,  # 6
              'colsample_bytree': 0.8,#0.8
              'subsample': 0.8,
              'scale_pos_weight': 1,
              'min_child_weight': 18  # 2
              }
    # 训练
    watchlist = [(dtrain,'train')]
    bst = xgb.train(params, dtrain, num_boost_round=700,evals=watchlist)
    # 预测
    predict = bst.predict(dtest)
#    test_xy = test[['instance_id','is_trade']]
    test_xy = test[['instance_id']]
    test_xy['predicted_score'] = predict
    
    return test_xy
    
def get_item_feat(data,dataFeat):
    "item的特征提取"
    
    result = pd.DataFrame(dataFeat['item_id'])
    result = result.drop_duplicates(['item_id'],keep='first')
   
    "1.统计item出现次数"
    dataFeat['item_count'] = dataFeat['item_id']
    feat = pd.pivot_table(dataFeat,index=['item_id'],values='item_count',aggfunc='count').reset_index()
    del dataFeat['item_count']
    result = pd.merge(result,feat,on=['item_id'],how='left')
    
    "2.统计item历史被购买的次数"
    dataFeat['item_buy_count'] = dataFeat['is_trade']
    feat = pd.pivot_table(dataFeat,index=['item_id'],values='item_buy_count',aggfunc='sum').reset_index()
    del dataFeat['item_buy_count']
    result = pd.merge(result,feat,on=['item_id'],how='left')
    
    "3.统计item转化率特征"
    buy_ratio = list(map(lambda x,y : -1 if y == 0 else x/y,result.item_buy_count,result.item_count))
    result['item_buy_ratio'] = buy_ratio

    "4.统计item历史未被够买的次数"
    result['item_not_buy_count'] = result['item_count'] - result['item_buy_count']
    
    return result
    
def get_user_feat(data,dataFeat):
    "user的特征提取"
    
    result = pd.DataFrame(dataFeat['user_id'])
    result = result.drop_duplicates(['user_id'],keep='first')
   
    "1.统计user出现次数"
    dataFeat['user_count'] = dataFeat['user_id']
    feat = pd.pivot_table(dataFeat,index=['user_id'],values='user_count',aggfunc='count').reset_index()
    del dataFeat['user_count']
    result = pd.merge(result,feat,on=['user_id'],how='left')
    
    "2.统计user历史被购买的次数"
    dataFeat['user_buy_count'] = dataFeat['is_trade']
    feat = pd.pivot_table(dataFeat,index=['user_id'],values='user_buy_count',aggfunc='sum').reset_index()
    del dataFeat['user_buy_count']
    result = pd.merge(result,feat,on=['user_id'],how='left')
    
    "3.统计user转化率特征"
    buy_ratio = list(map(lambda x,y : -1 if y == 0 else x/y,result.user_buy_count,result.user_count))
    result['user_buy_ratio'] = buy_ratio

    "4.统计user历史未被够买的次数"
    result['user_not_buy_count'] = result['user_count'] - result['user_buy_count']
    
    return result
    
def get_context_feat(data,dataFeat):
    "context的特征提取"
    
    result = pd.DataFrame(dataFeat['context_id'])
    result = result.drop_duplicates(['context_id'],keep='first')
   
    "1.统计context出现次数"
    dataFeat['context_count'] = dataFeat['context_id']
    feat = pd.pivot_table(dataFeat,index=['context_id'],values='context_count',aggfunc='count').reset_index()
    del dataFeat['context_count']
    result = pd.merge(result,feat,on=['context_id'],how='left')
    
    "2.统计context历史被购买的次数"
    dataFeat['context_buy_count'] = dataFeat['is_trade']
    feat = pd.pivot_table(dataFeat,index=['context_id'],values='context_buy_count',aggfunc='sum').reset_index()
    del dataFeat['context_buy_count']
    result = pd.merge(result,feat,on=['context_id'],how='left')
    
    "3.统计context转化率特征"
    buy_ratio = list(map(lambda x,y : -1 if y == 0 else x/y,result.context_buy_count,result.context_count))
    result['context_buy_ratio'] = buy_ratio

    "4.统计context历史未被够买的次数"
    result['context_not_buy_count'] = result['context_count'] - result['context_buy_count']
    
    return result
    
def get_shop_feat(data,dataFeat):
    "shop的特征提取"
    
    result = pd.DataFrame(dataFeat['shop_id'])
    result = result.drop_duplicates(['shop_id'],keep='first')
   
    "1.统计shop出现次数"
    dataFeat['shop_count'] = dataFeat['shop_id']
    feat = pd.pivot_table(dataFeat,index=['shop_id'],values='shop_count',aggfunc='count').reset_index()
    del dataFeat['shop_count']
    result = pd.merge(result,feat,on=['shop_id'],how='left')
    
    "2.统计shop历史被购买的次数"
    dataFeat['shop_buy_count'] = dataFeat['is_trade']
    feat = pd.pivot_table(dataFeat,index=['shop_id'],values='shop_buy_count',aggfunc='sum').reset_index()
    del dataFeat['shop_buy_count']
    result = pd.merge(result,feat,on=['shop_id'],how='left')
    
    "3.统计shop转化率特征"
    buy_ratio = list(map(lambda x,y : -1 if y == 0 else x/y,result.shop_buy_count,result.shop_count))
    result['shop_buy_ratio'] = buy_ratio

    "4.统计shop历史未被够买的次数"
    result['shop_not_buy_count'] = result['shop_count'] - result['shop_buy_count']
    
    return result
    
def get_timestamp_feat(data,dataFeat):
    "context_timestamp的特征提取"
    
    result = pd.DataFrame(dataFeat['context_timestamp'])
    result = result.drop_duplicates(['context_timestamp'],keep='first')
   
    "1.统计context_timestamp出现次数"
    dataFeat['context_timestamp_count'] = dataFeat['context_timestamp']
    feat = pd.pivot_table(dataFeat,index=['context_timestamp'],values='context_timestamp_count',aggfunc='count').reset_index()
    del dataFeat['context_timestamp_count']
    result = pd.merge(result,feat,on=['context_timestamp'],how='left')
    
    "2.统计context_timestamp历史被购买的次数"
    dataFeat['context_timestamp_buy_count'] = dataFeat['is_trade']
    feat = pd.pivot_table(dataFeat,index=['context_timestamp'],values='context_timestamp_buy_count',aggfunc='sum').reset_index()
    del dataFeat['context_timestamp_buy_count']
    result = pd.merge(result,feat,on=['context_timestamp'],how='left')
    
    "3.统计context_timestamp转化率特征"
    buy_ratio = list(map(lambda x,y : -1 if y == 0 else x/y,result.context_timestamp_buy_count,result.context_timestamp_count))
    result['context_timestamp_buy_ratio'] = buy_ratio

    "4.统计context_timestamp历史未被够买的次数"
    result['context_timestamp_not_buy_count'] = result['context_timestamp_count'] - result['context_timestamp_buy_count']
    
    return result
    
def get_item_brand_feat(data,dataFeat):
    "item_brand的特征提取"
    
    result = pd.DataFrame(dataFeat['item_brand_id'])
    result = result.drop_duplicates(['item_brand_id'],keep='first')
   
    "1.统计item_brand出现次数"
    dataFeat['item_brand_count'] = dataFeat['item_brand_id']
    feat = pd.pivot_table(dataFeat,index=['item_brand_id'],values='item_brand_count',aggfunc='count').reset_index()
    del dataFeat['item_brand_count']
    result = pd.merge(result,feat,on=['item_brand_id'],how='left')
    
    "2.统计item_brand历史被购买的次数"
    dataFeat['item_brand_buy_count'] = dataFeat['is_trade']
    feat = pd.pivot_table(dataFeat,index=['item_brand_id'],values='item_brand_buy_count',aggfunc='sum').reset_index()
    del dataFeat['item_brand_buy_count']
    result = pd.merge(result,feat,on=['item_brand_id'],how='left')
    
    "3.统计item_brand转化率特征"
    buy_ratio = list(map(lambda x,y : -1 if y == 0 else x/y,result.item_brand_buy_count,result.item_brand_count))
    result['item_brand_buy_ratio'] = buy_ratio

    "4.统计item_brand历史未被够买的次数"
    result['item_brand_not_buy_count'] = result['item_brand_count'] - result['item_brand_buy_count']
    
    return result

def get_item_city_feat(data,dataFeat):
    "item_city的特征提取"
    
    result = pd.DataFrame(dataFeat['item_city_id'])
    result = result.drop_duplicates(['item_city_id'],keep='first')
   
    "1.统计item_city出现次数"
    dataFeat['item_city_count'] = dataFeat['item_city_id']
    feat = pd.pivot_table(dataFeat,index=['item_city_id'],values='item_city_count',aggfunc='count').reset_index()
    del dataFeat['item_city_count']
    result = pd.merge(result,feat,on=['item_city_id'],how='left')
    
    "2.统计item_city历史被购买的次数"
    dataFeat['item_city_buy_count'] = dataFeat['is_trade']
    feat = pd.pivot_table(dataFeat,index=['item_city_id'],values='item_city_buy_count',aggfunc='sum').reset_index()
    del dataFeat['item_city_buy_count']
    result = pd.merge(result,feat,on=['item_city_id'],how='left')
    
    "3.统计item_city转化率特征"
    buy_ratio = list(map(lambda x,y : -1 if y == 0 else x/y,result.item_city_buy_count,result.item_city_count))
    result['item_city_buy_ratio'] = buy_ratio

    "4.统计item_city历史未被够买的次数"
    result['item_city_not_buy_count'] = result['item_city_count'] - result['item_city_buy_count']
    
    return result

def get_user_gender_feat(data,dataFeat):
    "user_gender的特征提取"
    
    result = pd.DataFrame(dataFeat['user_gender_id'])
    result = result.drop_duplicates(['user_gender_id'],keep='first')
   
    "1.统计user_gender出现次数"
    dataFeat['user_gender_count'] = dataFeat['user_gender_id']
    feat = pd.pivot_table(dataFeat,index=['user_gender_id'],values='user_gender_count',aggfunc='count').reset_index()
    del dataFeat['user_gender_count']
    result = pd.merge(result,feat,on=['user_gender_id'],how='left')
    
    "2.统计user_gender历史被购买的次数"
    dataFeat['user_gender_buy_count'] = dataFeat['is_trade']
    feat = pd.pivot_table(dataFeat,index=['user_gender_id'],values='user_gender_buy_count',aggfunc='sum').reset_index()
    del dataFeat['user_gender_buy_count']
    result = pd.merge(result,feat,on=['user_gender_id'],how='left')
    
    "3.统计user_gender转化率特征"
    buy_ratio = list(map(lambda x,y : -1 if y == 0 else x/y,result.user_gender_buy_count,result.user_gender_count))
    result['user_gender_buy_ratio'] = buy_ratio

    "4.统计user_gender历史未被够买的次数"
    result['user_gender_not_buy_count'] = result['user_gender_count'] - result['user_gender_buy_count']
    
    return result    

    
def get_user_occupation_feat(data,dataFeat):
    "user_occupation的特征提取"
    
    result = pd.DataFrame(dataFeat['user_occupation_id'])
    result = result.drop_duplicates(['user_occupation_id'],keep='first')
   
    "1.统计user_occupation出现次数"
    dataFeat['user_occupation_count'] = dataFeat['user_occupation_id']
    feat = pd.pivot_table(dataFeat,index=['user_occupation_id'],values='user_occupation_count',aggfunc='count').reset_index()
    del dataFeat['user_occupation_count']
    result = pd.merge(result,feat,on=['user_occupation_id'],how='left')
    
    "2.统计user_occupation历史被购买的次数"
    dataFeat['user_occupation_buy_count'] = dataFeat['is_trade']
    feat = pd.pivot_table(dataFeat,index=['user_occupation_id'],values='user_occupation_buy_count',aggfunc='sum').reset_index()
    del dataFeat['user_occupation_buy_count']
    result = pd.merge(result,feat,on=['user_occupation_id'],how='left')
    
    "3.统计user_occupation转化率特征"
    buy_ratio = list(map(lambda x,y : -1 if y == 0 else x/y,result.user_occupation_buy_count,result.user_occupation_count))
    result['user_occupation_buy_ratio'] = buy_ratio

    "4.统计user_occupation历史未被够买的次数"
    result['user_occupation_not_buy_count'] = result['user_occupation_count'] - result['user_occupation_buy_count']
    
    return result    
 
def get_context_page_feat(data,dataFeat):
    "context_page的特征提取"
    
    result = pd.DataFrame(dataFeat['context_page_id'])
    result = result.drop_duplicates(['context_page_id'],keep='first')
   
    "1.统计context_page出现次数"
    dataFeat['context_page_count'] = dataFeat['context_page_id']
    feat = pd.pivot_table(dataFeat,index=['context_page_id'],values='context_page_count',aggfunc='count').reset_index()
    del dataFeat['context_page_count']
    result = pd.merge(result,feat,on=['context_page_id'],how='left')
    
    "2.统计context_page历史被购买的次数"
    dataFeat['context_page_buy_count'] = dataFeat['is_trade']
    feat = pd.pivot_table(dataFeat,index=['context_page_id'],values='context_page_buy_count',aggfunc='sum').reset_index()
    del dataFeat['context_page_buy_count']
    result = pd.merge(result,feat,on=['context_page_id'],how='left')
    
    "3.统计context_page转化率特征"
    buy_ratio = list(map(lambda x,y : -1 if y == 0 else x/y,result.context_page_buy_count,result.context_page_count))
    result['context_page_buy_ratio'] = buy_ratio

    "4.统计context_page历史未被够买的次数"
    result['context_page_not_buy_count'] = result['context_page_count'] - result['context_page_buy_count']
    
    return result

def get_shop_review_num_level_feat(data,dataFeat):
    "context_page的特征提取"
    
    result = pd.DataFrame(dataFeat['shop_review_num_level'])
    result = result.drop_duplicates(['shop_review_num_level'],keep='first')
   
    "1.统计shop_review_num_level出现次数"
    dataFeat['shop_review_num_level_count'] = dataFeat['shop_review_num_level']
    feat = pd.pivot_table(dataFeat,index=['shop_review_num_level'],values='shop_review_num_level_count',aggfunc='count').reset_index()
    del dataFeat['shop_review_num_level_count']
    result = pd.merge(result,feat,on=['shop_review_num_level'],how='left')
    
    "2.统计shop_review_num_level历史被购买的次数"
    dataFeat['shop_review_num_level_buy_count'] = dataFeat['is_trade']
    feat = pd.pivot_table(dataFeat,index=['shop_review_num_level'],values='shop_review_num_level_buy_count',aggfunc='sum').reset_index()
    del dataFeat['shop_review_num_level_buy_count']
    result = pd.merge(result,feat,on=['shop_review_num_level'],how='left')
    
    "3.统计shop_review_num_level转化率特征"
    buy_ratio = list(map(lambda x,y : -1 if y == 0 else x/y,result.shop_review_num_level_buy_count,result.shop_review_num_level_count))
    result['shop_review_num_level_buy_ratio'] = buy_ratio

    "4.统计shop_review_num_level历史未被够买的次数"
    result['shop_review_num_level_not_buy_count'] = result['shop_review_num_level_count'] - result['shop_review_num_level_buy_count']
    
    return result 

def get_item_category_list_2_feat(data,dataFeat):
    "item_category_list_2的特征提取"
    
    result = pd.DataFrame(dataFeat['item_category_list_2'])
    result = result.drop_duplicates(['item_category_list_2'],keep='first')
   
    "1.统计item_category_list_2出现次数"
    dataFeat['item_category_list_2_count'] = dataFeat['item_category_list_2']
    feat = pd.pivot_table(dataFeat,index=['item_category_list_2'],values='item_category_list_2_count',aggfunc='count').reset_index()
    del dataFeat['item_category_list_2_count']
    result = pd.merge(result,feat,on=['item_category_list_2'],how='left')
    
    "2.统计item_category_list_2历史被购买的次数"
    dataFeat['item_category_list_2_buy_count'] = dataFeat['is_trade']
    feat = pd.pivot_table(dataFeat,index=['item_category_list_2'],values='item_category_list_2_buy_count',aggfunc='sum').reset_index()
    del dataFeat['item_category_list_2_buy_count']
    result = pd.merge(result,feat,on=['item_category_list_2'],how='left')
    
    "3.统计item_category_list_2转化率特征"
    buy_ratio = list(map(lambda x,y : -1 if y == 0 else x/y,result.item_category_list_2_buy_count,result.item_category_list_2_count))
    result['item_category_list_2_buy_ratio'] = buy_ratio

    "4.统计item_category_list_2历史未被够买的次数"
    result['item_category_list_2_not_buy_count'] = result['item_category_list_2_count'] - result['item_category_list_2_buy_count']
    
    return result        
     
    
def get_user_item_feat(data,dataFeat):
    "user-item的特征提取"
    
    result = pd.DataFrame(dataFeat[['user_id','item_id']])
    result = result.drop_duplicates(['user_id','item_id'],keep='first')
   
    "1.统计user-item出现次数"
    dataFeat['user_item_count'] = dataFeat['user_id']
    feat = pd.pivot_table(dataFeat,index=['user_id','item_id'],values='user_item_count',aggfunc='count').reset_index()
    del dataFeat['user_item_count']
    result = pd.merge(result,feat,on=['user_id','item_id'],how='left')
    
    "2.统计user-item历史被购买的次数"
    dataFeat['user_item_buy_count'] = dataFeat['is_trade']
    feat = pd.pivot_table(dataFeat,index=['user_id','item_id'],values='user_item_buy_count',aggfunc='sum').reset_index()
    del dataFeat['user_item_buy_count']
    result = pd.merge(result,feat,on=['user_id','item_id'],how='left')
    
    "3.统计user-item转化率特征"
    buy_ratio = list(map(lambda x,y : -1 if y == 0 else x/y,result.user_item_buy_count,result.user_item_count))
    result['user_item_buy_ratio'] = buy_ratio

    "4.统计user-item历史未被够买的次数"
    result['user_item_not_buy_count'] = result['user_item_count'] - result['user_item_buy_count']
    

    
    return result

def get_user_shop_feat(data,dataFeat):
    "user-shop的特征提取"
    
    result = pd.DataFrame(dataFeat[['user_id','shop_id']])
    result = result.drop_duplicates(['user_id','shop_id'],keep='first')
   
    "1.统计user-shop出现次数"
    dataFeat['user_shop_count'] = dataFeat['user_id']
    feat = pd.pivot_table(dataFeat,index=['user_id','shop_id'],values='user_shop_count',aggfunc='count').reset_index()
    del dataFeat['user_shop_count']
    result = pd.merge(result,feat,on=['user_id','shop_id'],how='left')
    
    "2.统计user-shop历史被购买的次数"
    dataFeat['user_shop_buy_count'] = dataFeat['is_trade']
    feat = pd.pivot_table(dataFeat,index=['user_id','shop_id'],values='user_shop_buy_count',aggfunc='sum').reset_index()
    del dataFeat['user_shop_buy_count']
    result = pd.merge(result,feat,on=['user_id','shop_id'],how='left')
    
    "3.统计user-shop转化率特征"
    buy_ratio = list(map(lambda x,y : -1 if y == 0 else x/y,result.user_shop_buy_count,result.user_shop_count))
    result['user_shop_buy_ratio'] = buy_ratio

    "4.统计user-shop历史未被够买的次数"
    result['user_shop_not_buy_count'] = result['user_shop_count'] - result['user_shop_buy_count']
    
    return result    

def get_user_context_feat(data,dataFeat):
    "user-context的特征提取"
    
    result = pd.DataFrame(dataFeat[['user_id','context_id']])
    result = result.drop_duplicates(['user_id','context_id'],keep='first')
   
    "1.统计user-context出现次数"
    dataFeat['user_context_count'] = dataFeat['user_id']
    feat = pd.pivot_table(dataFeat,index=['user_id','context_id'],values='user_context_count',aggfunc='count').reset_index()
    del dataFeat['user_context_count']
    result = pd.merge(result,feat,on=['user_id','context_id'],how='left')
    
    "2.统计user-context历史被购买的次数"
    dataFeat['user_context_buy_count'] = dataFeat['is_trade']
    feat = pd.pivot_table(dataFeat,index=['user_id','context_id'],values='user_context_buy_count',aggfunc='sum').reset_index()
    del dataFeat['user_context_buy_count']
    result = pd.merge(result,feat,on=['user_id','context_id'],how='left')
    
    "3.统计user-context转化率特征"
    buy_ratio = list(map(lambda x,y : -1 if y == 0 else x/y,result.user_context_buy_count,result.user_context_count))
    result['user_context_buy_ratio'] = buy_ratio

    "4.统计user-context历史未被够买的次数"
    result['user_context_not_buy_count'] = result['user_context_count'] - result['user_context_buy_count']
    
    return result  
    
def get_user_timestamp_feat(data,dataFeat):
    "user-context_timestamp的特征提取"
    
    result = pd.DataFrame(dataFeat[['user_id','context_timestamp']])
    result = result.drop_duplicates(['user_id','context_timestamp'],keep='first')
   
    "1.统计user-context_timestamp出现次数"
    dataFeat['user_context_timestamp_count'] = dataFeat['user_id']
    feat = pd.pivot_table(dataFeat,index=['user_id','context_timestamp'],values='user_context_timestamp_count',aggfunc='count').reset_index()
    del dataFeat['user_context_timestamp_count']
    result = pd.merge(result,feat,on=['user_id','context_timestamp'],how='left')
    
    "2.统计user-context_timestamp历史被购买的次数"
    dataFeat['user_context_timestamp_buy_count'] = dataFeat['is_trade']
    feat = pd.pivot_table(dataFeat,index=['user_id','context_timestamp'],values='user_context_timestamp_buy_count',aggfunc='sum').reset_index()
    del dataFeat['user_context_timestamp_buy_count']
    result = pd.merge(result,feat,on=['user_id','context_timestamp'],how='left')
    
    "3.统计user-context_timestamp转化率特征"
    buy_ratio = list(map(lambda x,y : -1 if y == 0 else x/y,result.user_context_timestamp_buy_count,result.user_context_timestamp_count))
    result['user_context_timestamp_buy_ratio'] = buy_ratio

    "4.统计user-context_timestamp历史未被够买的次数"
    result['user_context_timestamp_not_buy_count'] = result['user_context_timestamp_count'] - result['user_context_timestamp_buy_count']
    
    return result

def get_user_item_brand_feat(data,dataFeat):
    "user-item_brand的特征提取"
    
    result = pd.DataFrame(dataFeat[['user_id','item_brand_id']])
    result = result.drop_duplicates(['user_id','item_brand_id'],keep='first')
   
    "1.统计user-item_brand_id出现次数"
    dataFeat['user_item_brand_id_count'] = dataFeat['user_id']
    feat = pd.pivot_table(dataFeat,index=['user_id','item_brand_id'],values='user_item_brand_id_count',aggfunc='count').reset_index()
    del dataFeat['user_item_brand_id_count']
    result = pd.merge(result,feat,on=['user_id','item_brand_id'],how='left')
    
    "2.统计user-item_brand_id历史被购买的次数"
    dataFeat['user_item_brand_id_buy_count'] = dataFeat['is_trade']
    feat = pd.pivot_table(dataFeat,index=['user_id','item_brand_id'],values='user_item_brand_id_buy_count',aggfunc='sum').reset_index()
    del dataFeat['user_item_brand_id_buy_count']
    result = pd.merge(result,feat,on=['user_id','item_brand_id'],how='left')
    
    "3.统计user-item_brand_id转化率特征"
    buy_ratio = list(map(lambda x,y : -1 if y == 0 else x/y,result.user_item_brand_id_buy_count,result.user_item_brand_id_count))
    result['user_item_brand_id_buy_ratio'] = buy_ratio

    "4.统计user-item_brand_id历史未被够买的次数"
    result['user_item_brand_id_not_buy_count'] = result['user_item_brand_id_count'] - result['user_item_brand_id_buy_count']
    
    return result  
    
def get_user_user_gender_feat(data,dataFeat):
    "user-user_gender的特征提取"
    
    result = pd.DataFrame(dataFeat[['user_id','user_gender_id']])
    result = result.drop_duplicates(['user_id','user_gender_id'],keep='first')
   
    "1.统计user-user_gender_id出现次数"
    dataFeat['user_user_gender_id_count'] = dataFeat['user_id']
    feat = pd.pivot_table(dataFeat,index=['user_id','user_gender_id'],values='user_user_gender_id_count',aggfunc='count').reset_index()
    del dataFeat['user_user_gender_id_count']
    result = pd.merge(result,feat,on=['user_id','user_gender_id'],how='left')
    
    "2.统计user-user_gender_id历史被购买的次数"
    dataFeat['user_user_gender_id_buy_count'] = dataFeat['is_trade']
    feat = pd.pivot_table(dataFeat,index=['user_id','user_gender_id'],values='user_user_gender_id_buy_count',aggfunc='sum').reset_index()
    del dataFeat['user_user_gender_id_buy_count']
    result = pd.merge(result,feat,on=['user_id','user_gender_id'],how='left')
    
    "3.统计user-user_gender_id转化率特征"
    buy_ratio = list(map(lambda x,y : -1 if y == 0 else x/y,result.user_user_gender_id_buy_count,result.user_user_gender_id_count))
    result['user_user_gender_id_buy_ratio'] = buy_ratio

    "4.统计user-user_gender_id历史未被够买的次数"
    result['user_user_gender_id_not_buy_count'] = result['user_user_gender_id_count'] - result['user_user_gender_id_buy_count']
    
    return result  

def get_user_item_city_feat(data,dataFeat):
    "user-item_city的特征提取"
    
    result = pd.DataFrame(dataFeat[['user_id','item_city_id']])
    result = result.drop_duplicates(['user_id','item_city_id'],keep='first')
   
    "1.统计user-item_city_id出现次数"
    dataFeat['user_item_city_id_count'] = dataFeat['user_id']
    feat = pd.pivot_table(dataFeat,index=['user_id','item_city_id'],values='user_item_city_id_count',aggfunc='count').reset_index()
    del dataFeat['user_item_city_id_count']
    result = pd.merge(result,feat,on=['user_id','item_city_id'],how='left')
    
    "2.统计user-item_city_id历史被购买的次数"
    dataFeat['user_item_city_id_buy_count'] = dataFeat['is_trade']
    feat = pd.pivot_table(dataFeat,index=['user_id','item_city_id'],values='user_item_city_id_buy_count',aggfunc='sum').reset_index()
    del dataFeat['user_item_city_id_buy_count']
    result = pd.merge(result,feat,on=['user_id','item_city_id'],how='left')
    
    "3.统计user-item_city_id转化率特征"
    buy_ratio = list(map(lambda x,y : -1 if y == 0 else x/y,result.user_item_city_id_buy_count,result.user_item_city_id_count))
    result['user_item_city_id_buy_ratio'] = buy_ratio

    "4.统计user-item_city_id历史未被够买的次数"
    result['user_item_city_id_not_buy_count'] = result['user_item_city_id_count'] - result['user_item_city_id_buy_count']
    
    return result 
    
def get_user_context_page_feat(data,dataFeat):
    "user-context_page的特征提取"
    
    result = pd.DataFrame(dataFeat[['user_id','context_page_id']])
    result = result.drop_duplicates(['user_id','context_page_id'],keep='first')
   
    "1.统计user-context_page_id出现次数"
    dataFeat['user_context_page_id_count'] = dataFeat['user_id']
    feat = pd.pivot_table(dataFeat,index=['user_id','context_page_id'],values='user_context_page_id_count',aggfunc='count').reset_index()
    del dataFeat['user_context_page_id_count']
    result = pd.merge(result,feat,on=['user_id','context_page_id'],how='left')
    
    "2.统计user-context_page_id历史被购买的次数"
    dataFeat['user_context_page_id_buy_count'] = dataFeat['is_trade']
    feat = pd.pivot_table(dataFeat,index=['user_id','context_page_id'],values='user_context_page_id_buy_count',aggfunc='sum').reset_index()
    del dataFeat['user_context_page_id_buy_count']
    result = pd.merge(result,feat,on=['user_id','context_page_id'],how='left')
    
    "3.统计user-context_page_id转化率特征"
    buy_ratio = list(map(lambda x,y : -1 if y == 0 else x/y,result.user_context_page_id_buy_count,result.user_context_page_id_count))
    result['user_context_page_id_buy_ratio'] = buy_ratio

    "4.统计user-context_page_id历史未被够买的次数"
    result['user_context_page_id_not_buy_count'] = result['user_context_page_id_count'] - result['user_context_page_id_buy_count']
    
    return result  
    
def get_user_user_occupation_feat(data,dataFeat):
    "user-user_occupation的特征提取"
    
    result = pd.DataFrame(dataFeat[['user_id','user_occupation_id']])
    result = result.drop_duplicates(['user_id','user_occupation_id'],keep='first')
   
    "1.统计user-user_occupation_id出现次数"
    dataFeat['user_user_occupation_id_count'] = dataFeat['user_id']
    feat = pd.pivot_table(dataFeat,index=['user_id','user_occupation_id'],values='user_user_occupation_id_count',aggfunc='count').reset_index()
    del dataFeat['user_user_occupation_id_count']
    result = pd.merge(result,feat,on=['user_id','user_occupation_id'],how='left')
    
    "2.统计user-user_occupation_id历史被购买的次数"
    dataFeat['user_user_occupation_id_buy_count'] = dataFeat['is_trade']
    feat = pd.pivot_table(dataFeat,index=['user_id','user_occupation_id'],values='user_user_occupation_id_buy_count',aggfunc='sum').reset_index()
    del dataFeat['user_user_occupation_id_buy_count']
    result = pd.merge(result,feat,on=['user_id','user_occupation_id'],how='left')
    
    "3.统计user-user_occupation_id转化率特征"
    buy_ratio = list(map(lambda x,y : -1 if y == 0 else x/y,result.user_user_occupation_id_buy_count,result.user_user_occupation_id_count))
    result['user_user_occupation_id_buy_ratio'] = buy_ratio

    "4.统计user-user_occupation_id历史未被够买的次数"
    result['user_user_occupation_id_not_buy_count'] = result['user_user_occupation_id_count'] - result['user_user_occupation_id_buy_count']
    
    return result 
    
def get_user_shop_review_num_level_feat(data,dataFeat):
    "user-shop_review_num_level的特征提取"
    
    result = pd.DataFrame(dataFeat[['user_id','shop_review_num_level']])
    result = result.drop_duplicates(['user_id','shop_review_num_level'],keep='first')
   
    "1.统计user-shop_review_num_level出现次数"
    dataFeat['user_shop_review_num_level_count'] = dataFeat['user_id']
    feat = pd.pivot_table(dataFeat,index=['user_id','shop_review_num_level'],values='user_shop_review_num_level_count',aggfunc='count').reset_index()
    del dataFeat['user_shop_review_num_level_count']
    result = pd.merge(result,feat,on=['user_id','shop_review_num_level'],how='left')
    
    "2.统计user-shop_review_num_level历史被购买的次数"
    dataFeat['user_shop_review_num_level_buy_count'] = dataFeat['is_trade']
    feat = pd.pivot_table(dataFeat,index=['user_id','shop_review_num_level'],values='user_shop_review_num_level_buy_count',aggfunc='sum').reset_index()
    del dataFeat['user_shop_review_num_level_buy_count']
    result = pd.merge(result,feat,on=['user_id','shop_review_num_level'],how='left')
    
    "3.统计user-shop_review_num_level转化率特征"
    buy_ratio = list(map(lambda x,y : -1 if y == 0 else x/y,result.user_shop_review_num_level_buy_count,result.user_shop_review_num_level_count))
    result['user_shop_review_num_level_buy_ratio'] = buy_ratio

    "4.统计user-shop_review_num_level历史未被够买的次数"
    result['user_shop_review_num_level_not_buy_count'] = result['user_shop_review_num_level_count'] - result['user_shop_review_num_level_buy_count']
    
    return result 
    
def get_user_item_category_list_2_feat(data,dataFeat):
    "user-item_category_list_2的特征提取"
    
    result = pd.DataFrame(dataFeat[['user_id','item_category_list_2']])
    result = result.drop_duplicates(['user_id','item_category_list_2'],keep='first')
   
    "1.统计user-item_category_list_2出现次数"
    dataFeat['user_item_category_list_2_count'] = dataFeat['user_id']
    feat = pd.pivot_table(dataFeat,index=['user_id','item_category_list_2'],values='user_item_category_list_2_count',aggfunc='count').reset_index()
    del dataFeat['user_item_category_list_2_count']
    result = pd.merge(result,feat,on=['user_id','item_category_list_2'],how='left')
    
    "2.统计user-item_category_list_2历史被购买的次数"
    dataFeat['user_item_category_list_2_buy_count'] = dataFeat['is_trade']
    feat = pd.pivot_table(dataFeat,index=['user_id','item_category_list_2'],values='user_item_category_list_2_buy_count',aggfunc='sum').reset_index()
    del dataFeat['user_item_category_list_2_buy_count']
    result = pd.merge(result,feat,on=['user_id','item_category_list_2'],how='left')
    
    "3.统计user-item_category_list_2转化率特征"
    buy_ratio = list(map(lambda x,y : -1 if y == 0 else x/y,result.user_item_category_list_2_buy_count,result.user_item_category_list_2_count))
    result['user_item_category_list_2_buy_ratio'] = buy_ratio

    "4.统计user-item_category_list_2历史未被够买的次数"
    result['user_item_category_list_2_not_buy_count'] = result['user_item_category_list_2_count'] - result['user_item_category_list_2_buy_count']
    
    return result  
    
def merge_feat(data,dataFeat):
    "特征的merge"
    
    #生成特征
    item = get_item_feat(data,dataFeat)
    user = get_user_feat(data,dataFeat)
    context = get_context_feat(data,dataFeat)
    shop = get_shop_feat(data,dataFeat)
    timestamp = get_timestamp_feat(data,dataFeat)
    item_brand = get_item_brand_feat(data,dataFeat)
    user_gender = get_user_gender_feat(data,dataFeat)
    item_city = get_item_city_feat(data,dataFeat)
    context_page = get_context_page_feat(data,dataFeat)
    user_occupation = get_user_occupation_feat(data,dataFeat)
    shop_review_num_level = get_shop_review_num_level_feat(data,dataFeat)
    item_category_list_2 = get_item_category_list_2_feat(data,dataFeat)
    #交互特征
    user_item = get_user_item_feat(data,dataFeat)
    user_shop = get_user_shop_feat(data,dataFeat)
    user_context = get_user_context_feat(data,dataFeat)
    user_timestamp = get_user_timestamp_feat(data,dataFeat)
    user_item_brand = get_user_item_brand_feat(data,dataFeat)
    user_user_gender = get_user_user_gender_feat(data,dataFeat)
    user_item_city = get_user_item_city_feat(data,dataFeat)
    user_context_page = get_user_context_page_feat(data,dataFeat)
    user_user_occupation = get_user_user_occupation_feat(data,dataFeat)
    user_shop_review_num_level = get_user_shop_review_num_level_feat(data,dataFeat)
    user_item_category_list_2 = get_user_item_category_list_2_feat(data,dataFeat)

    #merge特征
    data = pd.merge(data,item,on='item_id',how='left')
    data = pd.merge(data,user,on='user_id',how='left')
    data = pd.merge(data,context,on='context_id',how='left')
    data = pd.merge(data,timestamp,on='context_timestamp',how='left')
    data = pd.merge(data,shop,on='shop_id',how='left')
    data = pd.merge(data,item_brand,on='item_brand_id',how='left')
    data = pd.merge(data,user_gender,on='user_gender_id',how='left')
    data = pd.merge(data,item_city,on='item_city_id',how='left')
    data = pd.merge(data,context_page,on='context_page_id',how='left')
    data = pd.merge(data,user_occupation,on='user_occupation_id',how='left')
    data = pd.merge(data,shop_review_num_level,on='shop_review_num_level',how='left')
    data = pd.merge(data,item_category_list_2,on='item_category_list_2',how='left')
    #交互特征
    data = pd.merge(data,user_item,on=['user_id','item_id'],how='left')
    data = pd.merge(data,user_shop,on=['user_id','shop_id'],how='left')
    data = pd.merge(data,user_context,on=['user_id','context_id'],how='left')
    data = pd.merge(data,user_timestamp,on=['user_id','context_timestamp'],how='left')
    data = pd.merge(data,user_item_brand,on=['user_id','item_brand_id'],how='left')
    data = pd.merge(data,user_user_gender,on=['user_id','user_gender_id'],how='left')
    data = pd.merge(data,user_item_city,on=['user_id','item_city_id'],how='left')
    data = pd.merge(data,user_context_page,on=['user_id','context_page_id'],how='left')
    data = pd.merge(data,user_user_occupation,on=['user_id','user_occupation_id'],how='left')
    data = pd.merge(data,user_shop_review_num_level,on=['user_id','shop_review_num_level'],how='left')
    data = pd.merge(data,user_item_category_list_2,on=['user_id','item_category_list_2'],how='left')

    return data

def get_label_feat(data):
    "标签数据集特征提取"
    
    data['hour'] = data.context_timestamp.map(lambda x:int(x[11:13]))
    
    "1.统计user当天点击广告次数"
    data['user_count_label'] = data['user_id']
    feat = pd.pivot_table(data,index=['user_id'],values='user_count_label',aggfunc='count').reset_index()
    del data['user_count_label']
    result = pd.merge(data,feat,on=['user_id'],how='left')
    
    "2.统计user当天重复点击item广告次数"
    data['user_item_count_label'] = data['item_id']
    feat = pd.pivot_table(data,index=['user_id','item_id'],values='user_item_count_label',aggfunc='count').reset_index()
    del data['user_item_count_label']
    result = pd.merge(result,feat,on=['user_id','item_id'],how='left')
    
    "3.统计user当天点击特定shop的广告次数"
    data['user_shop_count_label'] = data['item_id']
    feat = pd.pivot_table(data,index=['user_id','shop_id'],values='user_shop_count_label',aggfunc='count').reset_index()
    del data['user_shop_count_label']
    result = pd.merge(result,feat,on=['user_id','shop_id'],how='left')
    
    "4.统计user当天点击特定context的广告次数"
    data['user_context_count_label'] = data['item_id']
    feat = pd.pivot_table(data,index=['user_id','context_id'],values='user_context_count_label',aggfunc='count').reset_index()
    del data['user_context_count_label']
    result = pd.merge(result,feat,on=['user_id','context_id'],how='left')
    
    "5.统计item当天被点击次数"
    data['item_count_label'] = data['item_id']
    feat = pd.pivot_table(data,index=['item_id'],values='item_count_label',aggfunc='count').reset_index()
    del data['item_count_label']
    result = pd.merge(result,feat,on=['item_id'],how='left')   
    
    "6.统计shop当天被点击次数"
    data['shop_count_label'] = data['shop_id']
    feat = pd.pivot_table(data,index=['shop_id'],values='shop_count_label',aggfunc='count').reset_index()
    del data['shop_count_label']
    result = pd.merge(result,feat,on=['shop_id'],how='left')   
    
    "7.统计context当天被点击次数"
    data['context_count_label'] = data['context_id']
    feat = pd.pivot_table(data,index=['context_id'],values='context_count_label',aggfunc='count').reset_index()
    del data['context_count_label']
    result = pd.merge(result,feat,on=['context_id'],how='left')  
    
    
    "8.统计user-item/user特征"
    user_item_user_ratio_label = list(map(lambda x,y : -1 if y == 0 else x/y,result.user_item_count_label,result.user_count_label))
    result['user_item_user_ratio_label'] = user_item_user_ratio_label

    "9.统计user-shop/user特征"
    user_shop_user_ratio_label = list(map(lambda x,y : -1 if y == 0 else x/y,result.user_shop_count_label,result.user_count_label))
    result['user_shop_user_ratio_label'] = user_shop_user_ratio_label
    
    "10.统计user-context/user特征"
    user_context_user_ratio_label = list(map(lambda x,y : -1 if y == 0 else x/y,result.user_context_count_label,result.user_count_label))
    result['user_context_user_ratio_label'] = user_context_user_ratio_label

    "11.统计用户点击的排序"
    up = result.groupby(['user_id'])['context_timestamp'].rank(ascending=True)
    result['user_rank_up'] = up
    
    "12.统计item点击的排序"
    up = result.groupby(['item_id'])['context_timestamp'].rank(ascending=True)
    result['item_rank_up'] = up

    "13.统计shop点击的排序"
    up = result.groupby(['shop_id'])['context_timestamp'].rank(ascending=True)
    result['shop_rank_up'] = up

    "14.统计context_id点击的排序"
    up = result.groupby(['context_id'])['context_timestamp'].rank(ascending=True)
    result['context_rank_up'] = up

    "15.统计user和item点击的排序"
    up = result.groupby(['user_id','item_id'])['context_timestamp'].rank(ascending=True)
    result['user_item_rank_up'] = up

    "16.统计user和shop点击的排序"
    up = result.groupby(['user_id','shop_id'])['context_timestamp'].rank(ascending=True)
    result['user_shop_rank_up'] = up

    "17.统计user和context点击的排序"
    up = result.groupby(['user_id','context_id'])['context_timestamp'].rank(ascending=True)
    result['user_context_rank_up'] = up

#    "17.统计user和item_category_list_2点击的排序"
#    up = result.groupby(['user_id','item_category_list_2'])['context_timestamp'].rank(ascending=True)
#    result['user_item_category_list_2_rank_up'] = up


    "18.统计用户点击的排序"
    down = result.groupby(['user_id'])['context_timestamp'].rank(ascending=False)
    result['user_rank_down'] = down
    
    "19.统计item点击的排序"
    down = result.groupby(['item_id'])['context_timestamp'].rank(ascending=False)
    result['item_rank_down'] = down

    "20.统计shop点击的排序"
    down = result.groupby(['shop_id'])['context_timestamp'].rank(ascending=False)
    result['shop_rank_down'] = down

    "21.统计context_id点击的排序"
    down = result.groupby(['context_id'])['context_timestamp'].rank(ascending=False)
    result['context_rank_down'] = down

    "22.统计user和item点击的排序"
    down = result.groupby(['user_id','item_id'])['context_timestamp'].rank(ascending=False)
    result['user_item_rank_down'] = down

    "23.统计user和shop点击的排序"
    down = result.groupby(['user_id','shop_id'])['context_timestamp'].rank(ascending=False)
    result['user_shop_rank_down'] = down

    "24.统计user和context点击的排序"
    down = result.groupby(['user_id','context_id'])['context_timestamp'].rank(ascending=False)
    result['user_context_rank_down'] = down

#    "25.统计user和item_category_list_2点击的排序"
#    up = result.groupby(['user_id','item_category_list_2'])['context_timestamp'].rank(ascending=False)
#    result['user_item_category_list_2_rank_down'] = down



    "25.统计当天每小时的item点击次数"
    pivot = pd.pivot_table(result,index = 'hour',values = 'item_id',aggfunc = 'count')
    pivot = pivot.reset_index()
    pivot.rename(columns = {'item_id':'item_count_hour'},inplace = True)
    result = pd.merge(result,pivot,on='hour',how='left')
    
    "26.统计当天每小时的user点击次数"
    pivot = pd.pivot_table(result,index = 'hour',values = 'user_id',aggfunc = 'count')
    pivot = pivot.reset_index()
    pivot.rename(columns = {'user_id':'user_count_hour'},inplace = True)
    result = pd.merge(result,pivot,on='hour',how='left')
    
    "27.统计当天每小时的shop点击次数"
    pivot = pd.pivot_table(result,index = 'hour',values = 'shop_id',aggfunc = 'count')
    pivot = pivot.reset_index()
    pivot.rename(columns = {'shop_id':'shop_count_hour'},inplace = True)
    result = pd.merge(result,pivot,on='hour',how='left')
    
    "28.统计当天每小时的context点击次数"
    pivot = pd.pivot_table(result,index = 'hour',values = 'context_id',aggfunc = 'count')
    pivot = pivot.reset_index()
    pivot.rename(columns = {'context_id':'context_count_hour'},inplace = True)
    result = pd.merge(result,pivot,on='hour',how='left')
    
    "28.统计当天每小时的context_page点击次数"
    pivot = pd.pivot_table(result,index = 'hour',values = 'context_page_id',aggfunc = 'count')
    pivot = pivot.reset_index()
    pivot.rename(columns = {'context_page_id':'context_page_count_hour'},inplace = True)
    result = pd.merge(result,pivot,on='hour',how='left')
    
    "28.统计当天每小时的item_brand点击次数"
    pivot = pd.pivot_table(result,index = 'hour',values = 'item_brand_id',aggfunc = 'count')
    pivot = pivot.reset_index()
    pivot.rename(columns = {'item_brand_id':'item_brand_count_hour'},inplace = True)
    result = pd.merge(result,pivot,on='hour',how='left')
    
#    "28.统计当天每小时的item_category_list_2点击次数"
#    pivot = pd.pivot_table(result,index = 'hour',values = 'item_category_list_2',aggfunc = 'count')
#    pivot = pivot.reset_index()
#    pivot.rename(columns = {'item_category_list_2':'item_category_list_2_count_hour'},inplace = True)
#    result = pd.merge(result,pivot,on='hour',how='left')

    
    "29.统计user是否是第一次/最后一次领券"
    time = data[['user_id','context_timestamp']]
    time = time.groupby(['user_id'])['context_timestamp'].agg(lambda x:','.join(x)).reset_index()
    time['user_number'] = time.context_timestamp.apply(lambda s:len(s.split(',')))
    t = time[time.user_number>1]
    tb = time[time.user_number==1];tb['user_first'] = -1
    t['user_number_max'] = t.context_timestamp.map(lambda s:max(s.split(',')))
    t['user_number_min'] = t.context_timestamp.map(lambda s:min(s.split(',')))
    del t['context_timestamp'];del t['user_number'];del tb['user_number']
    t1 = t[['user_id','user_number_min']]
    t1 = t1.rename(columns={'user_number_min':'context_timestamp'})
    t1['user_first'] = 1
    t1 = t1.append(tb)
    t2 = t[['user_id','user_number_max']]
    t2 = t2.rename(columns={'user_number_max':'context_timestamp'})
    t2['user_last'] = 1
    tb = tb.rename(columns={'user_first':'user_last'})
    t2 = t2.append(tb)
    result = pd.merge(result,t1,how='left')
    result = pd.merge(result,t2,how='left')
    result.user_last = result.user_last.fillna(0)
    result.user_first = result.user_first.fillna(0)
    
    "30.统计item是否是第一次/最后一次领券"
    time = data[['item_id','context_timestamp']]
    time = time.groupby(['item_id'])['context_timestamp'].agg(lambda x:','.join(x)).reset_index()
    time['item_number'] = time.context_timestamp.apply(lambda s:len(s.split(',')))
    t = time[time.item_number>1]
    tb = time[time.item_number==1];tb['item_first'] = -1
    t['item_number_max'] = t.context_timestamp.map(lambda s:max(s.split(',')))
    t['item_number_min'] = t.context_timestamp.map(lambda s:min(s.split(',')))
    del t['context_timestamp'];del t['item_number'];del tb['item_number']
    t1 = t[['item_id','item_number_min']]
    t1 = t1.rename(columns={'item_number_min':'context_timestamp'})
    t1['item_first'] = 1
    t1 = t1.append(tb)
    t2 = t[['item_id','item_number_max']]
    t2 = t2.rename(columns={'item_number_max':'context_timestamp'})
    t2['item_last'] = 1
    tb = tb.rename(columns={'item_first':'item_last'})
    t2 = t2.append(tb)
    result = pd.merge(result,t1,how='left')
    result = pd.merge(result,t2,how='left')
    result.user_last = result.user_last.fillna(0)
    result.user_first = result.user_first.fillna(0) 
    
    "31.统计item是否是第一次/最后一次点击"
    time = data[['item_id','context_timestamp']]
    time = time.groupby(['item_id'])['context_timestamp'].agg(lambda x:','.join(x)).reset_index()
    time['item_number'] = time.context_timestamp.apply(lambda s:len(s.split(',')))
    t = time[time.item_number>1]
    tb = time[time.item_number==1];tb['item_first'] = -1
    t['item_number_max'] = t.context_timestamp.map(lambda s:max(s.split(',')))
    t['item_number_min'] = t.context_timestamp.map(lambda s:min(s.split(',')))
    del t['context_timestamp'];del t['item_number'];del tb['item_number']
    t1 = t[['item_id','item_number_min']]
    t1 = t1.rename(columns={'item_number_min':'context_timestamp'})
    t1['item_first'] = 1
    t1 = t1.append(tb)
    t2 = t[['item_id','item_number_max']]
    t2 = t2.rename(columns={'item_number_max':'context_timestamp'})
    t2['item_last'] = 1
    tb = tb.rename(columns={'item_first':'item_last'})
    t2 = t2.append(tb)
    result = pd.merge(result,t1,how='left')
    result = pd.merge(result,t2,how='left')
    result.item_last = result.item_last.fillna(0)
    result.item_first = result.item_first.fillna(0) 
    
    "32.统计shop是否是第一次/最后一次点击"
    time = data[['shop_id','context_timestamp']]
    time = time.groupby(['shop_id'])['context_timestamp'].agg(lambda x:','.join(x)).reset_index()
    time['shop_number'] = time.context_timestamp.apply(lambda s:len(s.split(',')))
    t = time[time.shop_number>1]
    tb = time[time.shop_number==1];tb['shop_first'] = -1
    t['shop_number_max'] = t.context_timestamp.map(lambda s:max(s.split(',')))
    t['shop_number_min'] = t.context_timestamp.map(lambda s:min(s.split(',')))
    del t['context_timestamp'];del t['shop_number'];del tb['shop_number']
    t1 = t[['shop_id','shop_number_min']]
    t1 = t1.rename(columns={'shop_number_min':'context_timestamp'})
    t1['shop_first'] = 1
    t1 = t1.append(tb)
    t2 = t[['shop_id','shop_number_max']]
    t2 = t2.rename(columns={'shop_number_max':'context_timestamp'})
    t2['shop_last'] = 1
    tb = tb.rename(columns={'shop_first':'shop_last'})
    t2 = t2.append(tb)
    result = pd.merge(result,t1,how='left')
    result = pd.merge(result,t2,how='left')
    result.shop_last = result.shop_last.fillna(0)
    result.shop_first = result.shop_first.fillna(0) 
    
    "33.统计context是否是第一次/最后一次点击"
    time = data[['context_id','context_timestamp']]
    time = time.groupby(['context_id'])['context_timestamp'].agg(lambda x:','.join(x)).reset_index()
    time['context_number'] = time.context_timestamp.apply(lambda s:len(s.split(',')))
    t = time[time.context_number>1]
    tb = time[time.context_number==1];tb['context_first'] = -1
    t['context_number_max'] = t.context_timestamp.map(lambda s:max(s.split(',')))
    t['context_number_min'] = t.context_timestamp.map(lambda s:min(s.split(',')))
    del t['context_timestamp'];del t['context_number'];del tb['context_number']
    t1 = t[['context_id','context_number_min']]
    t1 = t1.rename(columns={'context_number_min':'context_timestamp'})
    t1['context_first'] = 1
    t1 = t1.append(tb)
    t2 = t[['context_id','context_number_max']]
    t2 = t2.rename(columns={'context_number_max':'context_timestamp'})
    t2['context_last'] = 1
    tb = tb.rename(columns={'context_first':'shop_last'})
    t2 = t2.append(tb)
    result = pd.merge(result,t1,how='left')
    result = pd.merge(result,t2,how='left')
    result.context_last = result.context_last.fillna(0)
    result.context_first = result.context_first.fillna(0) 

#    "34.统计user对特定的predict_category_property_rank的次数统计"
#    data['user_predict_category'] = data['predict_category_property_rank']
#    feat = pd.pivot_table(data,index=['user_id','predict_category_property_rank'],values='predict_category_property_rank',aggfunc='count').reset_index()
#    del data['user_predict_category']
#    result = pd.merge(result,feat,on=['user_id','predict_category_property_rank'],how='left')
#    
#    "35.统计item对特定的predict_category_property_rank的次数统计"
#    data['item_predict_category'] = data['predict_category_property_rank']
#    feat = pd.pivot_table(data,index=['item_id','predict_category_property_rank'],values='predict_category_property_rank',aggfunc='count').reset_index()
#    del data['item_predict_category']
#    result = pd.merge(result,feat,on=['item_id','predict_category_property_rank'],how='left')
    
    
    
    del result['hour']
    
   
    return result
    
    
def logloss(act, pred):
    "评价函数"
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll
    
if __name__ == '__main__':
    "主函数入口"
    #下载数据
    trainSet,testSet = loadData()
    #划分数据集
    test,testFeat,validate,validateFeat,train1,trainFeat1,train2,trainFeat2,train3,trainFeat3,train4,trainFeat4 = splitData(trainSet,testSet)
    #训练集特征提取
    train1 = get_label_feat(train1)
    train1 = merge_feat(train1,trainFeat1)
    train2 = get_label_feat(train2)
    train2 = merge_feat(train2,trainFeat2)
    train3 = get_label_feat(train3)
    train3 = merge_feat(train3,trainFeat3)
    train4 = get_label_feat(train4)
    train4 = merge_feat(train4,trainFeat4)
    #加验证集加入到训练集中
    train5 = get_label_feat(validate)
    train5 = merge_feat(train5,validateFeat)
    train = train1.append(train2)
    train = train.append(train3)
    train = train.append(train4)
    train = train.append(train5)
    
    
#    "线下验证"
#    #验证集特征提取
#    validate = get_label_feat(validate)
#    validate = merge_feat(validate,validateFeat)
#    ans = modelXgb(train,validate)
#    ll = logloss(ans['is_trade'], ans['predicted_score'])
    

    
    "线上测试"
    test = get_label_feat(test)
    test = merge_feat(test,testFeat)
    ans = modelXgb(train,test)
    ans.to_csv('ans.txt',sep=' ',line_terminator='\r',index=False)

    

#    #抽样
#    train['rand'] = train.instance_id.map(lambda x : random.randint(1,6))
#    t1 = train[(train['rand']==1)|(train['rand']==2)|(train['rand']==3)|(train['is_trade']==0)]
#    del t1['rand']
#    ans = modelXgb(t1,validate);ll = logloss(ans['is_trade'], ans['pred'])
#    
    
    
    
    
    
    
    




