#!/usr/bin/python
# coding=utf-8
import re
from enum import Enum
import io


# 字符类型
class CharType(Enum):
    chinese = 1
    number = 2
    english = 3


# 文本特征处理工具类
class TextFeatureUtil:

    # # 计算IDF值
    # @staticmethod
    # def calculateIdf(vocabList,texts):
    #     vocabDict={}
    #     for i in range(len(vocabList)):
    #         word=vocabList[i]
    #         vocabDict[word]=i
    #     idfVec=[0] * len(vocabList)
    #     for text in  texts:
    #         wordSet = set(TextFeatureUtil.splitText(text))
    #         for word in  wordSet:
    #             if word in vocabDict:
    #                 idfVec[vocabDict.get(word)] += 1
    #     texts_len=len(texts)
    #     for i in range(len(idfVec)):
    #         idfVec[i]=math.log((1.0+texts_len)/(1.0+idfVec[i]))
    #     TextFeatureUtil.idfVec=idfVec
    #
    # @staticmethod
    # def calculateTfIdf(vocabList,text):
    #     tfidfVec = [0] * len(vocabList)
    #     vocabDict = {}
    #     for i in range(len(vocabList)):
    #         word = vocabList[i]
    #         vocabDict[word] = i
    #     text = TextFeatureUtil.cleanText(text)
    #     wordSet = set(TextFeatureUtil.splitText(text))
    #     for word in wordSet:
    #         if word in vocabDict:
    #             tfidfVec[vocabDict.get(word)] += 1
    #     for i in range(len(vocabList)):
    #         tfidfVec[i]*=TextFeatureUtil.idfVec[i]
    #     return tfidfVec

    # 构建词向量
    @staticmethod
    def getTextVector(vocabList, text):
        returnVec = [0] * len(vocabList)
        text = TextFeatureUtil.cleanText(text)
        wordSet = TextFeatureUtil.splitText(text)
        for word in wordSet:
            if word in vocabList:
                returnVec[vocabList.get(word)] = 1
        return returnVec

    # 得到停用词列表
    @staticmethod
    def getStopWords(filename):
        fo = open(filename, "r", encoding="utf-8")
        stopwords = set()
        stopwords.add(" ")

        while True:
            line = fo.readline()
            if not line:
                break
            stopword = line.replace("\n", "")
            stopwords.add(stopword)

        fo.close()
        return stopwords

    # 构建词袋
    @staticmethod
    def constrcutVocabList(texts):
        stopwords = TextFeatureUtil.getStopWords("../data/text/stopwords.txt")
        stopwords.add(" ")
        stopwords.add(":")
        words = set()
        for text in texts:
            words = words.union(TextFeatureUtil.splitText(text) - stopwords)
        words=list(words)
        words.sort
        vocabList = {}
        index = 0
        for word in words:
            vocabList[word] = index
            index += 1
        return vocabList

    # 输出字符的类型
    @staticmethod
    def getCharType(ch):
        ch = ch.lower()
        if '\u4e00' <= ch <= '\u9fff':
            return CharType.chinese
        elif '0' <= ch <= '9':
            return CharType.number
        elif 'a' <= ch <= 'z':
            return CharType.english

    # 拆分文本从而得到词
    # 1.汉字单字 2.数字连续组合 3.字符连续组合
    @staticmethod
    def splitText(text):

        words = set()
        lastType = CharType.chinese
        lastIndex = 0

        for i in range(len(text)):
            ch = text[i]
            currentType = TextFeatureUtil.getCharType(ch)
            if (currentType == CharType.chinese):
                words.add(str(ch))
            if (currentType != lastType):
                # 上一个字符不是中文需要添加
                if (lastType != CharType.chinese):
                    words.add(text[lastIndex:i])
                lastType = currentType
                lastIndex = i
            # 最后一个字符的话
            if ((i + 1) >= len(text) and currentType != CharType.chinese):
                words.add(text[lastIndex:])

        return words

    # 利用正则表达式过滤无效文本，只保留数字+汉字+英文+常用特殊字符，其他全部去掉
    @staticmethod
    def cleanText(text):
        cop = re.compile("[^\u4e00-\u9fa5^\s^a-z^A-Z^0-9]")#^\:^\，^\：^\。^\.
        return cop.sub("", text)

    # 读出多行文本,只是进行了最基本的过滤操作
    @staticmethod
    def getTexts(filename):
        fo = open(filename, "r", encoding="utf-8")
        lines = fo.readlines();
        texts = []
        for line in lines:
            strs = line.split(":")
            if len(strs) < 2:
                continue
            else:
                text = line[len(strs[0]) + 1:]
                text = TextFeatureUtil.cleanText(text)
                texts.append(text)
        return texts

    # 判断是否是无效文本，系统产生视为无效
    @staticmethod
    def isInvalidText(text):
        filterStrs = ['来听听我唱', 'gif', '击败', '转发', '打败']
        for filterStr in filterStrs:
            if text.find(filterStr) == -1:
                continue
            return True
        if len(text) <= 5:
            return True
        else:
            return False

    # 读出多行文本,只是进行了最基本的过滤操作
    @staticmethod
    def getTextsByUser(filelist):
        users = {}
        for filename in filelist:
            fo = io.open(filename, "r")
            lines = fo.readlines();
            for line in lines:
                strs = line.split(":")
                if len(strs) < 2:
                    continue
                else:
                    userid = int(strs[0])
                    text = line[len(strs[0]) + 1:]
                    if TextFeatureUtil.isInvalidText(text):
                        continue
                    text = TextFeatureUtil.cleanText(text)
                    if len(text) == 0:
                        continue
                    else:
                        if not users.__contains__(userid):
                            users[userid] = []
                        users[userid].append(text)
        return users

# texts="傻狗诗人么123jjj操:aaa,bb"
# print(TextFeatureUtil.constrcutVocabList([texts]))