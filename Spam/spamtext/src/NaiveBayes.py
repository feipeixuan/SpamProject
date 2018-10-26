#!/usr/bin/python
# coding=utf-8

import numpy as np
import math
import sklearn.model_selection  as  selection
import sys
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from src.common.AttributeFeatureUtil import AttributeFeatureUtil
from src.common.TextFeatureUtil import TextFeatureUtil
import json


# 特征类
class Feature:

    # 属性特征+文本特征
    def __init__(self, attributeFeature, textFeature):
        self.attributeFeature = attributeFeature
        self.textFeature = textFeature


# 朴素贝叶斯分类器
class NaiveBayes:

    def __init__(self):
        self.fieldNames = ['id', 'nickname', 'registertime', 'type', 'friend_num', 'fan_num']
        self.fieldValues = self.getFieldnameValues()
        self.vocabList = None
        self.mnb = None

    # 获取特征集合
    def getFeatures(self, attributeFile, textFile):
        textFeatures = TextFeatureUtil.getTextsByUser([textFile])
        (attributeFeatures, results) = AttributeFeatureUtil.loadTranSetFileList([attributeFile], [1])
        testFeatures = []
        for i in range(len(attributeFeatures)):
            attributeFeature = attributeFeatures[i]
            userid = attributeFeature[0]
            if userid in textFeatures:
                texts = textFeatures[userid]
                for text in texts:
                    testFeature = Feature(attributeFeature, text)
                    testFeatures.append(testFeature)
        return testFeatures

    # 获取属性的取值列表
    def getFieldnameValues(self):
        # 属性对应的取值集合
        fieldname_values = {}
        for i in range(len(self.fieldNames)):
            fieldname = self.fieldNames[i]
            # id 列直接跳过
            if fieldname == "id":
                continue
            else:
                # 获取属性对应的值列表
                values = AttributeFeatureUtil.getValues(fieldname)
                fieldname_values[fieldname] = values
        return fieldname_values

    # 产生特征向量集合
    def generateVectors(self, features):
        vectors = []
        for feature in features:
            vector = self.generateVector(feature)
            vectors.append(vector)
        return vectors

    # 产生特征向量
    def generateVector(self, feature):
        vector = []
        textFeature = feature.textFeature
        attributeFeature = feature.attributeFeature
        textVector = self.generateTextVector(textFeature)
        attributeVector = self.generateAttributeVector(attributeFeature)
        vector.extend(textVector)
        vector.extend(attributeVector)
        return vector

    # 产生文本特征向量
    def generateTextVector(self, textFeature):
        vocabList = self.vocabList
        textVector = TextFeatureUtil.getTextVector(vocabList, textFeature)
        textLength = len(textFeature)
        textLength = round(math.log2(textLength))
        textLength = min(textLength, 8)
        textLengthVector = [0] * 9
        textLengthVector[textLength] = 1
        textVector.extend(textLengthVector)
        return textVector

    # 产生属性特征向量
    def generateAttributeVector(self, attributeFeature):
        data_vector = []
        for i in range(len(self.fieldNames)):
            fieldname = self.fieldNames[i]
            if fieldname == "id":
                continue
            value = attributeFeature[i]
            values = self.fieldValues[fieldname]
            if not isinstance(values[0], str):
                value = int(value)
            if value in values:
                index = values.index(value)
            # TODO 不在取值集合中
            field_vector = [0] * len(values)
            field_vector[index] = 1
            data_vector.extend(field_vector)
        return data_vector

    # 训练模型
    # 1.正样本文件列表 2.负样本文件列表
    def trainModel(self, positiveFileList, negativeFileList):
        # 载入数据
        textFeatures = TextFeatureUtil.getTextsByUser([positiveFileList[1], negativeFileList[1]])
        (attributeFeatures, results) = AttributeFeatureUtil.loadTranSetFileList(
            [positiveFileList[0], negativeFileList[0]], [1, 0])

        # 构建全部的训练数据
        trainFeatures = []
        trainTexts = []
        trainResults = []

        for i in range(len(attributeFeatures)):
            attributeFeature = attributeFeatures[i]
            userid = attributeFeature[0]
            if textFeatures.__contains__(userid):
                texts = textFeatures[userid]
                for text in texts:
                    trainFeature = Feature(attributeFeature, text)
                    trainFeatures.append(trainFeature)
                    trainTexts.append(text)
                    trainResults.append(results[i])

        # 构建词典
        self.vocabList = TextFeatureUtil.constrcutVocabList(trainTexts)
        print(len(self.vocabList))
        # 构建训练向量
        trainVectors = self.generateVectors(trainFeatures)
        # 使用分类器进行训练
        mnb = MultinomialNB()
        mnb.fit(trainVectors, trainResults)
        self.mnb = mnb

    # 持久化模型
    def persistModel(self, url):
        joblib.dump(self.mnb, url + '/multinomialNB')
        with open(url + '/vocabList.json', 'w') as json_file:
            json_file.write(json.dumps(self.vocabList))

    # 载入模型
    def loadModel(self, url):
        mnb = joblib.load(url + '/multinomialNB')
        self.mnb = mnb
        with open(url + '/vocabList.json') as json_file:
            self.vocabList = json.load(json_file)

    # 针对单条特征做预测
    # 1.True代表有问题 2.False是没问题
    def predict(self, test_feature):
        vector = self.generateVector(test_feature)
        predict_result = self.mnb.predict([vector])
        if predict_result[0] == 0:
            return True
        else:
            return False


# 主函数
def main():
    positeiveFileList=['正常用户完整信息PART.csv','正常评论1026[去重以后].txt']
    negativeFileList=['骂人用户完整信息PART.csv','骂人评论[去重以后].txt']
    # # 1.初始化贝叶斯分类器
    naiveBayes = NaiveBayes()
    # naiveBayes.trainModel(positeiveFileList,negativeFileList)
    # naiveBayes.persistModel("rs")
    naiveBayes.loadModel("rs")
    # 2.做预测
    out = open('rs/判断为有问题的日常文本[文本+属性特征].txt', 'w')
    # 3.获取测试集特征集合
    testFeatures = naiveBayes.getFeatures("日常用户完整信息PART.csv", "日常文本集合[去重以后].txt")
    for testFeature in testFeatures:
        result = naiveBayes.predict(testFeature)
        if result:
            print(testFeature.textFeature)
            userid = str(testFeature.attributeFeature[0])
            out.write(userid + ":" + testFeature.textFeature + "\n")
    out.close()


main()
