#!/usr/bin/python
# coding=utf-8
import os
import pandas as pd
import sys
import numpy as np
import time
import math

# 属性特征处理类
class AttributeFeatureUtil:

    # 载入数据文件列表
    @staticmethod
    def loadTranSetFileList(filelist, types):
        features = []  # 特征向量
        results = []  # 每条记录对应的结果
        fieldnames = []
        for i in range(len(filelist)):
            filename = filelist[i]
            (fieldnames, f, r) = AttributeFeatureUtil.loadTranSetFile(filename, types[i]);
            features.extend(f)
            results.extend(r)
            print("finish loadTranSetFile: " + filename + " linenum: " + str(len(f)))
        print("")
        # 从特征集中删除了三个特征
        return (np.array(fieldnames), np.array(features), np.array(results))

    # 载入单个数据文件
    @staticmethod
    def loadTranSetFile(filename, type):
        # 第一列为 uid, 第二列为结果标签， 之后的列为feature
        features = []
        results = []
        fieldnames = []
        if not os.path.exists(filename):
            return (fieldnames, features, results)
        df = pd.read_csv(filename, encoding="utf-8")
        fieldnames = df.columns.values.tolist()
        features = df.values.tolist()
        for feature in features:
            AttributeFeatureUtil.preprocessData(fieldnames, feature)
            results.append(type)
        fieldnames.append("coin_buy_ratio")
        fieldnames.remove("phone_valid")
        return (fieldnames, features, results)

    # 预处理数据：使得数据在给定的数据空间内,同时对数据进行一定的变换
    @staticmethod
    def preprocessData(fieldnames, parts):
        # 最大值的设定会影响数据的变换，可以考虑调参
        maxlimit = {'flower_num': 4, 'friend_num': 4, 'fan_num': 4, 'work_num': 3, 'coin_sum': 6, 'coin_buy': 6}
        numeric_attributes = ['flower_num', 'friend_num', 'fan_num', 'work_num', 'coin_sum', 'coin_buy']
        size = len(parts)
        buy_gold_coin = round(math.log(parts[size - 3] * 1.0 / (parts[size - 4] + 1) * 100 + 1, 2))
        parts.append(buy_gold_coin)
        for i in range(len(fieldnames)):
            name = fieldnames[i]
            if name == "nickname":
                parts[i] = AttributeFeatureUtil.getNicknameType(parts[i])
            if name == "version":
                parts[i] = AttributeFeatureUtil.getVersion(parts[i])
            if name == "src":
                parts[i] = AttributeFeatureUtil.getSrc(parts[i])
            if name == "phone":
                parts[i] = AttributeFeatureUtil.getPhone(parts[i])
            if name == "registertime":
                parts[i] = AttributeFeatureUtil.getHour(parts[i])
            if name == "province" and type(parts[i]) == float:
                parts[i] = "none"
            if name in numeric_attributes:
                parts[i] = round(math.log10(parts[i] + 1))
                parts[i] = min(parts[i], maxlimit[name])
        # 新增一列 付费金币占比 +  手机号
        phone_index = fieldnames.index("phone")
        parts[phone_index] = parts[phone_index] + parts[phone_index + 1]
        del parts[phone_index + 1]
        return (fieldnames, parts)

    # 获取注册时区
    @staticmethod
    def getHour(register_time):
        register_time = time.localtime(time.mktime(time.strptime(register_time, "%Y-%m-%d %H:%M:%S")))
        return register_time.tm_hour

    # 得到三位的版本号
    @staticmethod
    def getVersion(version):
        if type(version) == float:
            return 0;
        else:
            print(version)
            str = version.split(".")
            if len(str) <=1:
                return 0
            if len(str) == 2:
                value = str[0] + str[1] + "0"
            else:
                value = str[0] + str[1] + str[2]
            return value

    # 四种情况：1.纯数字 2.字母+数字 3.包含表情 4.其他类型
    @staticmethod
    def getNicknameType(nickname):
        nickname = nickname.lower()
        # 判断是否是纯数字
        flag = True
        for ch in nickname:
            if '0' >= ch or ch >= '9':
                flag = False;
        if flag:
            return 0
        flag = True
        # 判断是否是字母加数字的组合
        for ch in nickname:
            if '0' <= ch <= '9' or 'a' <= ch <= 'z':
                continue
            else:
                flag = False
        if flag:
            return 1;
        # 判断是否包含表情
        for ch in nickname:
            if AttributeFeatureUtil.isEmoji(ch):
                return 2;
        return 3

    # 得到来源
    @staticmethod
    def getSrc(src):
        if type(src) == float:
            return "none"
        else:
            return src

    # 是否包含表情
    @staticmethod
    def isEmoji(content):
        if not content:
            return False
        if u"\U0001F600" <= content and content <= u"\U0001F64F":
            return True
        elif u"\U0001F300" <= content and content <= u"\U0001F5FF":
            return True
        elif u"\U0001F680" <= content and content <= u"\U0001F6FF":
            return True
        elif u"\U0001F1E0" <= content and content <= u"\U0001F1FF":
            return True
        else:
            return False

    # 是否具有手机号
    @staticmethod
    def getPhone(phone):
        if phone == -1:
            return 0
        else:
            return 1

    # 返回一列的取值集合
    @staticmethod
    def getValues(fieldName):
        # 设定属性对应的取值范围
        maxlimit = {'flower_num': 4, 'friend_num': 4, 'fan_num': 4, 'work_num': 3, 'coin_sum': 6, 'coin_buy': 6,
                    'phone': 2, 'registertime': 23, 'coin_buy_ratio': 7, 'richlevel': 27, 'starlevel': 33,'nickname':3}
        if maxlimit.__contains__(fieldName):
            return list(range(0, maxlimit.get(fieldName) + 1))
        elif fieldName == "type":
            return ['weixin', 'weibo', 'tencent', 'phone', 'renren']

    @staticmethod
    def getDistinctValues(values):
        values = list(set(values))
        values.append("default")
        return values

