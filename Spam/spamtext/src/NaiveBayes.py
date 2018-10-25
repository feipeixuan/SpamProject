#!/usr/bin/python
# coding=utf-8

import numpy as np
import math
import sklearn.model_selection  as  selection
from src.common import AttributeFeatureUtil
from src.common import TextFeatureUtil
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


# 1.词集模型

# 特征+类别
class Feature_Result:
    # 属性特征+文本特征+类别
    def __init__(self, attribute_feature, text_feature, result):
        self.attribute_feature = attribute_feature
        self.text_feature = text_feature
        self.result = result


def getFeatureResults(attributeFile, textFile):
    user_texts = TextFeatureUtil.TextFeatureUtil.getTextsByUser([textFile])
    (fieldnames, features, results) = AttributeFeatureUtil.AttributeFeatureUtil.loadTranSetFileList([attributeFile],
                                                                                                    [1])
    test_feature_results = []
    for i in range(len(features)):
        test_data = features[i]
        userid = test_data[0]
        if user_texts.__contains__(userid):
            texts = user_texts[userid]
            for text in texts:
                test_feature_result = Feature_Result(test_data, text, 1)
                test_feature_results.append(test_feature_result)
    return test_feature_results


# 划分训练集和测试集,基于用户级别
# 参数：属性文件列表、文本文件列表，文本类别、测试数据集的比例
def train_test_split(attributeFileList, textFileList, textFileTypes, factor):
    user_texts = TextFeatureUtil.TextFeatureUtil.getTextsByUser(textFileList)
    (fieldnames, features, results) = AttributeFeatureUtil.AttributeFeatureUtil.loadTranSetFileList(attributeFileList,
                                                                                                    textFileTypes)
    train_datas, test_datas, train_results, test_results = selection.train_test_split(features, results,
                                                                                      test_size=factor, random_state=42)
    train_feature_results = []
    test_feature_results = []

    for i in range(len(train_datas)):
        train_data = train_datas[i]
        userid = train_data[0]
        if user_texts.__contains__(userid):
            texts = user_texts[userid]
            for text in texts:
                train_feature_result = Feature_Result(train_data, text, train_results[i])
                train_feature_results.append(train_feature_result)

    for i in range(len(test_datas)):
        test_data = test_datas[i]
        userid = test_data[0]
        if user_texts.__contains__(userid):
            texts = user_texts[userid]
            for text in texts:
                test_feature_result = Feature_Result(test_data, text, test_results[i])
                test_feature_results.append(test_feature_result)

    return (fieldnames, train_feature_results, test_feature_results)


# 构建ont hot 编码【针对属性】
def constuct_one_hot(train_data_features, fieldnames):
    # 属性对应的取值集合
    fieldname_values = {}
    # 属性对应的onehot 编码器
    fieldname_encoder = {}
    for i in range(len(fieldnames)):
        fieldname = fieldnames[i]
        enc = preprocessing.OneHotEncoder()
        # id 列直接跳过
        if fieldname == "id":
            continue
        else:
            # 获取属性对应的值列表
            if (fieldname == "province" or fieldname == "version" or fieldname == "src"):
                values = AttributeFeatureUtil.AttributeFeatureUtil.getDistinctValues(train_data_features[:, i])
            elif (fieldname == "type"):
                values = list(set(train_data_features[:, i]))
            else:
                values = AttributeFeatureUtil.AttributeFeatureUtil.getValues(fieldname)
            fieldname_values[fieldname] = values
            #  将值列表转换为数字，以便做one hot
            valueList = []
            for i in range(len(values)):
                value = [i]
                valueList.append(value)
            enc.fit(valueList)
            # if (fieldname == "province" or fieldname == "version" or fieldname == "src" or fieldname == "type"):
            #     enc.fit(valueList)
            # else:
            #     enc.fit(valueList)
            fieldname_encoder[fieldname] = enc

    text_encoder = preprocessing.OneHotEncoder()
    text_lengthList = []
    for i in range(0, 9):
        text_lengthList.append([i])
    text_encoder.fit(text_lengthList)
    fieldname_encoder['text'] = text_encoder
    return (fieldname_values, fieldname_encoder)


# 产生特征向量
def generateVectors(feature_results, fieldname_values, fieldname_encoders, fieldnames, vocabList):
    vectors = []
    size = len(feature_results)
    for i in range(size):
        vector=generateVector(feature_results[i],fieldname_values,fieldname_encoders,fieldnames,vocabList)
        vectors.append(vector)
    return vectors


# 产生特征向量
def generateVector(feature_result, fieldname_values, fieldname_encoders, fieldnames, vocabList):
    vector = []
    text_feature = feature_result.text_feature
    data_feature = feature_result.attribute_feature
    text_vector = generateTextVector(text_feature, fieldname_encoders.get('text'), vocabList)
    data_vector = generateAttributeVector(data_feature, fieldname_values, fieldname_encoders, fieldnames)
    vector.extend(text_vector)
    vector.extend(data_vector)
    return vector


# 产生文本特征向量
def generateTextVector(text_feature, encoder, vocabList):
    text_vector = TextFeatureUtil.TextFeatureUtil.getTextVector(vocabList, text_feature)
    text_length = len(text_feature)
    text_length = round(math.log2(text_length))
    text_length = min(text_length, 8)
    text_vector.extend(encoder.transform([[text_length]]).toarray()[0].tolist())
    return text_vector


# 产生属性特征向量
def generateAttributeVector(data_feature, fieldname_values, fieldname_encoder, fieldnames):
    data_vector = []
    for i in range(len(fieldnames)):
        fieldname = fieldnames[i]
        if fieldname == "id":
            continue
        value = data_feature[i]
        values = fieldname_values[fieldname]
        if not isinstance(values[0], str):
            value = int(value)
        if value in values:
            index = values.index(value)
        else:
            index = len(values) - 1
        encoder = fieldname_encoder[fieldname]
        field_vector = encoder.transform([[index]]).toarray()[0].tolist()
        data_vector.extend(field_vector)
    return data_vector


# 主函数
def main():
    attributeFileList = ["resource/正常用户完整信息.csv", "resource/骂人用户完整信息.csv"]
    textFileList = ["resource/text/正常评论[去重以后].txt", "resource/text/骂人评论[去重以后].txt"]
    textFileTypes = [1, 0]

    # 1.划分训练集和测试集
    (fieldnames, train_feature_results, test_feature_results) = train_test_split(attributeFileList, textFileList,
                                                                                 textFileTypes, 0)
    train_data_features = []
    train_text_features = []
    train_results = []

    for train_feature_result in train_feature_results:
        train_data_features.append(train_feature_result.attribute_feature)
        train_text_features.append(train_feature_result.text_feature)
        train_results.append(train_feature_result.result)

    test_data_features = []
    test_text_features = []
    test_results = []
    for test_feature_result in test_feature_results:
        test_data_features.append(test_feature_result.attribute_feature)
        test_text_features.append(test_feature_result.text_feature)
        test_results.append(test_feature_result.result)

    # 2.构建one hot 编码器
    (fieldname_values, fieldname_encoders) = constuct_one_hot(np.array(train_data_features), fieldnames)
    # 产生词袋
    vocabList = TextFeatureUtil.TextFeatureUtil.constrcutVocabList(train_text_features)
    print(len(vocabList))
    # 产生向量
    train_vectors = generateVectors(train_feature_results, fieldname_values, fieldname_encoders, fieldnames, vocabList)
    # test_vectors = generateVector(test_feature_results, fieldname_values, fieldname_encoders, fieldnames, vocabList)
    # print(len(test_vectors))

    # 3.使用朴素贝叶斯进行训练
    mnb = MultinomialNB()  # 使用默认配置初始化朴素贝叶斯
    mnb.fit(train_vectors, train_results)  # 利用训练数据对模型参数进行估计
    joblib.dump(mnb, "multinomialNB")
    # out = open("报告数据.txt", "w")
    # feature_probs = mnb.feature_log_prob_
    # for i in range(len(feature_probs)):
    #     for j in range(len(feature_probs[i])):
    #         if j < len(vocabList):
    #             word = vocabList[j]
    #         else:
    #             word = str(j - len(vocabList))
    #         prob = feature_probs[i][j]
    #         out.write(word + ":" + str(prob) + "\n")
    # out.close()
    # predict_results = mnb.predict(test_vectors)     # 对参数进行预测
    # # 4.获取结果报告
    # print ('The Accuracy of Naive Bayes Classifier is:', mnb.score(test_vectors,test_results))
    # print(confusion_matrix(test_results,predict_results ))
    # print (classification_report(test_results, predict_results))
    # 5.获取分类错误的文本
    # out = open('属性+文本分类结果.csv','w', newline='')
    # csvwriter = csv.writer(out, dialect=("excel"))
    # names=['用户id','文本内容','实际类型']
    # csvwriter.writerow(names)
    #
    # for i in range(len(predict_results)):
    #     predict_result=predict_results[i]
    #     test_result=test_results[i]
    #     if predict_result!=test_result:
    #         userid=test_data_features[i][0]
    #         text=test_text_features[i]
    #         text=text.replace("\n","")
    #         if(test_result==1):
    #             reuslt="正常用户"
    #         else:
    #             reuslt="异常用户"
    #         row=[userid,text,reuslt]
    #         csvwriter.writerow(row)
    # 5.1提取被标记为错误的文本
    # user_texts=TextFeatureUtil.TextFeatureUtil.getTextsByUser(["resource/text/普通评论.txt"])
    # texts=[]
    # out = open('判断为有问题的日常文本.txt','w')
    # count_text=300000
    # for userid in user_texts:
    #     texts=user_texts[userid]
    #     for text in texts:
    #         text_vector=generateTextVector(text,fieldname_encoders['text'],vocabList)
    #         predict_result=mnb.predict([text_vector])
    #         if(predict_result[0]==0):
    #             out.write(text+"\n")
    #             print(text)
    #         count_text-=1
    #         if count_text==0:
    #             return
    # test_vector=generateTextVector("互相关注s",fieldname_encoders['text'],vocabList)

    # 5.1提取被标记为错误的文本
    # out = open('判断为有问题的日常文本[文本+属性特征].txt', 'w')
    # test_feature_results = getFeatureResults("other/日常用户/日常用户完整信息.csv", "other/日常用户/日常文本集合[去重以后].txt")
    # for test_feature_result in test_feature_results:
    #     vector = generateVector(test_feature_result, fieldname_values, fieldname_encoders, fieldnames, vocabList)
    #     predict_result = mnb.predict([vector])
    #     if predict_result[0] == 0:
    #         userid=test_feature_result.attribute_feature[0]
    #         out.write(userid+":"+test_feature_result.text_feature + "\n")
    #         print(test_feature_result.text_feature)


main()

# enc = preprocessing.OneHotEncoder()
# list=[[1],[2],[3]]
# enc.fit(list)
# print(enc.transform([[2]]).toarray())
